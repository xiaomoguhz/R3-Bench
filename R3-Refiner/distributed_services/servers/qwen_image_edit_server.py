#!/usr/bin/env python3
"""Qwen-Image-Edit service with optional cache-dit acceleration."""

import argparse
import sys
import logging
import os

# Keep service logs on the original process streams.
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

os.environ["TQDM_DISABLE"] = "1"

import base64
import io
import json
from pathlib import Path

import torch
import torch.distributed as dist
from flask import Flask, request, jsonify
from PIL import Image
import threading

# Suppress progress bars in service logs.
try:
    from tqdm import tqdm

    class SilentTqdm:
        def __init__(self, *args, **kwargs):
            self.iterable = args[0] if args else kwargs.get('iterable', [])
            self.total = kwargs.get('total', len(self.iterable) if hasattr(self.iterable, '__len__') else None)
        
        def __iter__(self):
            return iter(self.iterable)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            return False
        
        def update(self, n=1):
            pass
        
        def close(self):
            pass
    
    import tqdm as tqdm_module
    tqdm_module.tqdm = SilentTqdm
except ImportError:
    pass

from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
import cache_dit
from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig
# Used to adjust VAE_IMAGE_SIZE per request.
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus

app = Flask(__name__)

MODEL = None
DEVICE = None
# Protect model calls and cache reconfiguration in threaded mode.
MODEL_LOCK = threading.Lock()
USE_THREADING = False


def setup_distributed_if_needed():
    """Initialize the distributed process group when running in a multi-GPU environment."""
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank

    return 0


def load_qwen_model(
    model_path: str,
    device: torch.device,
    enable_cache: bool = True,
    cache_config: DBCacheConfig = None,
    calibrator_config: TaylorSeerCalibratorConfig = None,
):
    """Load the Qwen Image Edit model and enable cache-dit acceleration."""
    global MODEL, DEVICE

    print(f"Loading Qwen Image Edit model...", flush=True)
    print(f"Model path: {model_path}", flush=True)
    print(f"Device: {device}", flush=True)

    print("Loading QwenImageEditPlusPipeline...", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    
    if enable_cache:
        print("Enabling cache-dit acceleration...", flush=True)
        
        if cache_config is None:
            cache_config = DBCacheConfig(
                Fn_compute_blocks=8,      
                Bn_compute_blocks=0,        
                residual_diff_threshold=0.08,  
                steps_computation_mask=[1 if (i-1) % 3 == 0 or i<5 else 0 for i in range(50)],
                steps_computation_policy="dynamic",
            )
        
        if calibrator_config is None:
            calibrator_config = TaylorSeerCalibratorConfig(
                enable_calibrator=True,
                enable_encoder_calibrator=True,
                calibrator_type="taylorseer",
                calibrator_cache_type="residual",
                taylorseer_order=6,
            )
        
        cache_dit.enable_cache(
            pipe,
            cache_config=cache_config,
            calibrator_config=calibrator_config,
        )
        print("cache-dit acceleration enabled (DBCache + TaylorSeer)", flush=True)
    
    MODEL = pipe
    DEVICE = device
    
    print(f"[OK] Qwen Image Edit model loaded successfully on {device}", flush=True)
    print(f"Model loaded. GPU memory usage:", flush=True)
    import subprocess
    try:
        physical_gpu_id = None
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
            if cuda_visible.strip().isdigit():
                physical_gpu_id = int(cuda_visible.strip())
            elif ',' in cuda_visible:
                physical_gpu_id = int(cuda_visible.split(',')[0].strip())

        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpu_idx = int(parts[0])
                        mem_used = parts[1]
                        mem_total = parts[2]

                        if physical_gpu_id is not None:
                            if gpu_idx == physical_gpu_id:
                                print(f"GPU {gpu_idx} memory: {mem_used}MB / {mem_total}MB", flush=True)
                                break
                        else:
                            print(f"GPU {gpu_idx} memory: {mem_used}MB / {mem_total}MB", flush=True)
    except Exception as e:
        # Query failed; suppress silently (does not affect service startup)
        pass


@torch.no_grad()
def edit_image_qwen(
    image: Image.Image,
    edit_prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    true_cfg_scale: float = 4.0,
    negative_prompt: str = "blurry, ugly, low quality, distorted",
    steps_computation_mask: list = None,
    resolution_scale: float = 1.0,  # Scale factor (0, 1.0]; default 1.0 (1024x1024)
) -> Image.Image:
    """Edit an image using the Qwen model.

    Args:
        resolution_scale: factor controlling target VAE area in (0, 1.0].
    """
    device = DEVICE
    
    # VAE_IMAGE_SIZE is a target area; the pipeline derives dimensions from
    # the input aspect ratio.
    base_vae_image_size = 1024 * 1024  # Baseline area
    target_vae_image_size = int(base_vae_image_size * resolution_scale * resolution_scale)
    
    if USE_THREADING:
        MODEL_LOCK.acquire()
    try:
        original_vae_image_size = pipeline_qwenimage_edit_plus.VAE_IMAGE_SIZE
        
        if target_vae_image_size != original_vae_image_size:
            pipeline_qwenimage_edit_plus.VAE_IMAGE_SIZE = target_vae_image_size
            if not hasattr(edit_image_qwen, '_resolution_scale_logged') or edit_image_qwen._resolution_scale_logged != resolution_scale:
                print(f"[INFO] resolution_scale = {resolution_scale}", flush=True)
                print(f"  - VAE_IMAGE_SIZE (target area) = {target_vae_image_size} pixels", flush=True)
                if resolution_scale < 1.0:
                    speedup = 1.0 / (resolution_scale * resolution_scale)
                    print(f"  - Expected speedup: ~{speedup:.1f}x (vs. resolution_scale=1.0)", flush=True)
                edit_image_qwen._resolution_scale_logged = resolution_scale
    finally:
        if USE_THREADING:
            MODEL_LOCK.release()
    
    # Reconfigure cache only when step count or mask changes.
    if MODEL is not None:
        if steps_computation_mask is not None:
            actual_mask = steps_computation_mask
            if len(actual_mask) != num_inference_steps:
                print(f"[WARNING] Supplied mask length ({len(actual_mask)}) does not match num_inference_steps ({num_inference_steps}); regenerating mask dynamically", flush=True)
                actual_mask = [1 if (i-1) % 3 == 0 or i<5 else 0 for i in range(num_inference_steps)]
        else:
            actual_mask = [1 if (i-1) % 3 == 0 or i<5 else 0 for i in range(num_inference_steps)]

        need_reconfig = False

        if not hasattr(edit_image_qwen, '_last_steps'):
            edit_image_qwen._last_steps = 50
        if not hasattr(edit_image_qwen, '_last_mask'):
            edit_image_qwen._last_mask = None

        mask_changed = (edit_image_qwen._last_mask != actual_mask)
        steps_changed = (edit_image_qwen._last_steps != num_inference_steps)

        if mask_changed or steps_changed:
            need_reconfig = True
            edit_image_qwen._last_steps = num_inference_steps
            edit_image_qwen._last_mask = actual_mask

        if need_reconfig:
            if USE_THREADING:
                MODEL_LOCK.acquire()
            try:
                cache_dit.disable_cache(MODEL)

                cache_config = DBCacheConfig(
                    Fn_compute_blocks=8,
                    Bn_compute_blocks=0,
                    residual_diff_threshold=0.08,
                    max_warmup_steps=5,
                    steps_computation_mask=actual_mask,
                    steps_computation_policy="static",
                )
                calibrator_config = TaylorSeerCalibratorConfig(
                    enable_calibrator=True,
                    enable_encoder_calibrator=True,
                    calibrator_type="taylorseer",
                    calibrator_cache_type="residual",
                    taylorseer_order=6,
                )

                cache_dit.enable_cache(MODEL, cache_config=cache_config, calibrator_config=calibrator_config)
            finally:
                if USE_THREADING:
                    MODEL_LOCK.release()
    
    # Match output height/width to the VAE target area and input aspect ratio.
    image_aspect_ratio = image.width / image.height
    import math
    base_area = 1024 * 1024  # Baseline area
    target_area = int(base_area * resolution_scale * resolution_scale)
    
    calculated_width = math.sqrt(target_area * image_aspect_ratio)
    calculated_height = calculated_width / image_aspect_ratio

    multiple_of = 32
    calculated_width = int(round(calculated_width / multiple_of) * multiple_of)
    calculated_height = int(round(calculated_height / multiple_of) * multiple_of)
    
    model_kwargs = {
        "prompt": edit_prompt,
        "negative_prompt": negative_prompt,
        "image": image,
        "num_inference_steps": num_inference_steps,
        "true_cfg_scale": true_cfg_scale,
        "generator": torch.manual_seed(42),
        "height": calculated_height,
        "width": calculated_width,
        "guidance_scale": guidance_scale
    }
    
    if USE_THREADING:
        MODEL_LOCK.acquire()
    try:
        output = MODEL(**model_kwargs).images[0]
    finally:
        if USE_THREADING:
            MODEL_LOCK.release()
    
    return output


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_type": "qwen_image_edit"})


@app.route("/edit", methods=["POST"])
def edit_image_endpoint():
    """
    Image editing API endpoint.

    Request body (JSON):
    {
        "image": "<base64-encoded input image>",
        "edit_prompt": "<editing instruction>",
        "num_inference_steps": 50,           # optional
        "guidance_scale": 1.0,               # optional
        "true_cfg_scale": 4.0,               # optional
        "negative_prompt": "blurry, ugly, low quality, distorted",  # optional
        "resolution_scale": 1.0,             # optional; scale factor (0, 1.0] — trades speed vs. quality (default 1.0)
        "steps_computation_mask": [1,0,0,1,0,0,...],  # optional; custom mask (length must match num_inference_steps)
    }

    About steps_computation_mask:
    - If provided, the supplied mask is used as-is.
    - If omitted, the service computes every third step after warmup.

    Response body (JSON):
    {
        "success": true,
        "image": "<base64-encoded edited image>",
        "error": null
    }
    """
    try:
        data = request.get_json()
        
        image_b64 = data.get("image")
        edit_prompt = data.get("edit_prompt")
        
        if not image_b64 or not edit_prompt:
            return jsonify({
                "success": False,
                "image": None,
                "error": "Missing required fields: image or edit_prompt"
            }), 400
        
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        num_inference_steps = data.get("num_inference_steps", 50)
        guidance_scale = data.get("guidance_scale", 1.0)
        true_cfg_scale = data.get("true_cfg_scale", 4.0)
        negative_prompt = data.get("negative_prompt", "blurry, ugly, low quality, distorted")
        resolution_scale = data.get("resolution_scale", 1.0)
        steps_computation_mask = data.get("steps_computation_mask", None)

        if resolution_scale <= 0 or resolution_scale > 1.0:
            print(f"[WARNING] resolution_scale={resolution_scale} is outside valid range (0, 1.0]; clamping to 1.0", flush=True)
            resolution_scale = 1.0

        edited_image = edit_image_qwen(
            image=image,
            edit_prompt=edit_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            steps_computation_mask=steps_computation_mask,
            resolution_scale=resolution_scale,
        )

        buffer = io.BytesIO()
        edited_image.save(buffer, format="PNG")
        edited_image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return jsonify({
            "success": True,
            "image": edited_image_b64,
            "error": None
        })
        
    except Exception as e:
        # Keep traceback in service logs; return only the error message to clients.
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"[ERROR] /edit endpoint failed: {error_msg}", flush=True)
        print(f"[ERROR] Traceback:\n{error_traceback}", flush=True)
        
        return jsonify({
            "success": False,
            "image": None,
            "error": error_msg
        }), 500


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen Image Edit server (with cache-dit acceleration)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--disable_cache", action="store_true", help="Disable cache-dit acceleration")
    parser.add_argument("--taylorseer_order", type=int, default=6, help="TaylorSeer order")
    parser.add_argument("--residual_diff_threshold", type=float, default=0.08, help="Residual difference threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60, flush=True)
    print("Starting Qwen Image Edit server...", flush=True)
    print(f"Model path: {args.model_path}", flush=True)
    print("=" * 60, flush=True)
    
    try:
        local_rank = setup_distributed_if_needed()
        device = torch.device(f"cuda:{local_rank}")

        physical_gpu_id = None
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
            if cuda_visible.strip().isdigit():
                physical_gpu_id = int(cuda_visible.strip())
            elif ',' in cuda_visible:
                physical_gpu_id = int(cuda_visible.split(',')[0].strip())

        global DEVICE
        DEVICE = device

        if physical_gpu_id is not None:
            print(f"Using device: {device} (physical GPU: {physical_gpu_id})", flush=True)
        else:
            print(f"Using device: {device}", flush=True)

        cache_config = None
        calibrator_config = None
        if not args.disable_cache:
            cache_config = DBCacheConfig(
                Fn_compute_blocks=8,
                Bn_compute_blocks=0,
                residual_diff_threshold=args.residual_diff_threshold,
                steps_computation_mask=[1 if (i-1) % 3 == 0 or i<5 else 0 for i in range(50)],
                steps_computation_policy="dynamic",
            )
            calibrator_config = TaylorSeerCalibratorConfig(
                enable_calibrator=True,
                enable_encoder_calibrator=True,
                calibrator_type="taylorseer",
                calibrator_cache_type="residual",
                taylorseer_order=args.taylorseer_order,
            )
        
        print("Loading model...", flush=True)
        load_qwen_model(
            args.model_path,
            device,
            enable_cache=not args.disable_cache,
            cache_config=cache_config,
            calibrator_config=calibrator_config,
        )
        
        print(f"[START] Starting Qwen Image Edit server on {args.host}:{args.port}", flush=True)
        print(f"   Model path: {args.model_path}", flush=True)
        if physical_gpu_id is not None:
            print(f"   Device: {device} (physical GPU: {physical_gpu_id})", flush=True)
        else:
            print(f"   Device: {device}", flush=True)
        print(f"   Cache-dit: {'enabled' if not args.disable_cache else 'disabled'}", flush=True)
        if not args.disable_cache:
            print(f"   TaylorSeer order: {args.taylorseer_order}", flush=True)
            print(f"   Residual diff threshold: {args.residual_diff_threshold}", flush=True)

        log = logging.getLogger('werkzeug')
        log.setLevel(logging.INFO)
        logging.getLogger('werkzeug').disabled = False

        qwen_threaded = os.environ.get("EDIT_SERVER_THREADED", "false").lower() == "true"
        global USE_THREADING
        USE_THREADING = qwen_threaded
        print(f"   Threaded mode: {qwen_threaded}", flush=True)
        if qwen_threaded:
            print(f"   Warning: threaded mode enabled; model calls are protected by a lock", flush=True)
        else:
            print(f"   Single-threaded mode", flush=True)
        
        app.run(host=args.host, port=args.port, threaded=qwen_threaded)
    except Exception as e:
        print(f"[ERROR] Server startup failed: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
