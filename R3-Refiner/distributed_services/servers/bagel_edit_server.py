#!/usr/bin/env python3
"""BAGEL image-edit service."""

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

current_file = os.path.abspath(__file__)
bagel_deps_dir = os.path.join(os.path.dirname(os.path.dirname(current_file)), "bagel_deps")
bagel_deps_dir = os.path.abspath(bagel_deps_dir)

if os.path.exists(bagel_deps_dir):
    if bagel_deps_dir not in sys.path:
        sys.path.insert(0, bagel_deps_dir)
else:
    raise RuntimeError(
        f"bagel_deps directory not found. Please ensure the path exists.\n"
        f"Attempted path: {bagel_deps_dir}\n"
        f"Current file location: {os.path.abspath(__file__)}"
    )

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.qwen2 import Qwen2Tokenizer

app = Flask(__name__)

MODEL = None
VAE_MODEL = None
TOKENIZER = None
NEW_TOKEN_IDS = None
DEVICE = None
MODEL_TYPE = "bagel"


def setup_distributed_if_needed():
    """Initialize the distributed process group when running in a multi-GPU environment."""
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank

    return 0


def load_bagel_model(model_path: str, device: torch.device, max_latent_size: int = 64):
    """Load the BAGEL model."""
    global MODEL, VAE_MODEL, TOKENIZER, NEW_TOKEN_IDS, MODEL_TYPE

    print(f"Loading BAGEL model...", flush=True)
    print(f"Model path: {model_path}", flush=True)
    print(f"Device: {device}", flush=True)
    MODEL_TYPE = "bagel"

    print("Loading LLM config...", flush=True)
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    print("Loading Vision config...", flush=True)
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    
    print("Loading VAE model...", flush=True)
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    
    print("Creating BAGEL config...", flush=True)
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=max_latent_size,
    )
    
    print("Initializing model architecture...", flush=True)
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, bagel_config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    
    print("Loading tokenizer...", flush=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    checkpoint_path = os.path.join(model_path, "ema.safetensors")
    print(f"Loading model checkpoint: {checkpoint_path}", flush=True)
    print("Checkpoint loading can take several minutes.", flush=True)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint_path,
        device_map={"": "cpu"},
        dtype=torch.float32,
    )
    print("Moving model to GPU...", flush=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    vae_model = vae_model.to(device=device, dtype=torch.bfloat16).eval()
    
    MODEL = model
    VAE_MODEL = vae_model
    TOKENIZER = tokenizer
    NEW_TOKEN_IDS = new_token_ids
    
    print(f"[OK] BAGEL model loaded successfully on {device}", flush=True)
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
def edit_image_bagel(
    image: Image.Image,
    edit_prompt: str,
    num_timesteps: int = 50,
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 2.0,
    cfg_interval: list = None,
    timestep_shift: float = 3.0,
    cfg_renorm_min: float = 0.0,
    cfg_renorm_type: str = "text_channel",
    resolution_scale: float = 1.0,  # Reduce latent size to improve speed
    enable_taylorseer: bool = True,
    fresh_threshold: int = 3,
    max_order: int = 6,
    first_enhance: int = 5,
) -> Image.Image:
    """Edit an image using the BAGEL inference path.

    Args:
        resolution_scale: factor for reducing latent size in (0, 1.0].
            The final side length is clamped to at least 512 pixels.
    """
    from modeling.bagel.qwen2_navit import NaiveCache
    from data.transforms import ImageTransform
    from data.data_utils import pil_img2rgb
    import copy
    
    if cfg_interval is None:
        cfg_interval = [0.0, 1.0]
    
    device = DEVICE
    
    image = pil_img2rgb(image)

    vae_transform = ImageTransform(1024, 512, 16)  # long edge <= 1024, short edge >= 512, stride=16
    vit_transform = ImageTransform(980, 224, 14)   # long edge <= 980, short edge >= 224, stride=14
    
    
    gen_context = {
        'kv_lens': [0],
        'ropes': [0],
        'past_key_values': NaiveCache(MODEL.config.llm_config.num_hidden_layers),
    }
    
    cfg_text_context = copy.deepcopy(gen_context)
    cfg_img_context = copy.deepcopy(gen_context)
    
    # Add image tokens.
    image = vae_transform.resize_transform(image)
    image_shapes = image.size[::-1]  # (H, W)
    
    # Lower latent resolution when requested, keeping the side length >= 512.
    if resolution_scale < 1.0:
        H, W = image_shapes
        new_H = int(H * resolution_scale) // MODEL.latent_downsample * MODEL.latent_downsample
        new_W = int(W * resolution_scale) // MODEL.latent_downsample * MODEL.latent_downsample
        
        min_size = 512
        new_H = max(new_H, min_size)
        new_W = max(new_W, min_size)
        
        if new_H != H or new_W != W:
            image_shapes = (new_H, new_W)
            image = image.resize((new_W, new_H), Image.BICUBIC)

            if not hasattr(edit_image_bagel, '_last_resolution_scale') or edit_image_bagel._last_resolution_scale != resolution_scale:
                h, w = new_H // MODEL.latent_downsample, new_W // MODEL.latent_downsample
                num_tokens = h * w
                orig_h, orig_w = H // MODEL.latent_downsample, W // MODEL.latent_downsample
                orig_tokens = orig_h * orig_w
                speedup = orig_tokens / num_tokens if num_tokens > 0 else 1.0
                print(f"[Latent optimization] resolution_scale={resolution_scale}")
                print(f"  - Original image: {W}x{H} -> Latent tokens: {orig_tokens}")
                print(f"  - Optimized size: {new_W}x{new_H} -> Latent tokens: {num_tokens}")
                print(f"  - Expected speedup: ~{speedup:.2f}x (based on latent token count reduction)")
                edit_image_bagel._last_resolution_scale = resolution_scale
    
    generation_input_vae, gen_context['kv_lens'], gen_context['ropes'] = MODEL.prepare_vae_images(
        curr_kvlens=gen_context['kv_lens'],
        curr_rope=gen_context['ropes'],
        images=[image],
        transforms=vae_transform,
        new_token_ids=NEW_TOKEN_IDS,
    )
    for key, value in generation_input_vae.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input_vae[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input_vae[key] = value.to(device=device)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_context['past_key_values'] = MODEL.forward_cache_update_vae(
            VAE_MODEL, gen_context['past_key_values'], **generation_input_vae
        )
    
    generation_input_vit, gen_context['kv_lens'], gen_context['ropes'] = MODEL.prepare_vit_images(
        curr_kvlens=gen_context['kv_lens'],
        curr_rope=gen_context['ropes'],
        images=[image],
        transforms=vit_transform,
        new_token_ids=NEW_TOKEN_IDS,
    )
    for key, value in generation_input_vit.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input_vit[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input_vit[key] = value.to(device=device)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_context['past_key_values'] = MODEL.forward_cache_update_vit(
            gen_context['past_key_values'], **generation_input_vit
        )
    
    cfg_text_context = copy.deepcopy(gen_context)
    cfg_img_context = copy.deepcopy(gen_context)
    
    # Add editing instruction.
    generation_input, gen_context['kv_lens'], gen_context['ropes'] = MODEL.prepare_prompts(
        curr_kvlens=gen_context['kv_lens'],
        curr_rope=gen_context['ropes'],
        prompts=[edit_prompt],
        tokenizer=TOKENIZER,
        new_token_ids=NEW_TOKEN_IDS,
    )
    for key, value in generation_input.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input[key] = value.to(device=device)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_context['past_key_values'] = MODEL.forward_cache_update_text(
            gen_context['past_key_values'], **generation_input
        )
        cfg_img_context['past_key_values'] = MODEL.forward_cache_update_text(
            cfg_img_context['past_key_values'], **generation_input
        )
        cfg_img_context['kv_lens'] = gen_context['kv_lens']
        cfg_img_context['ropes'] = gen_context['ropes']
    
    # Prepare generation and CFG inputs.
    generation_input = MODEL.prepare_vae_latent(
        curr_kvlens=gen_context['kv_lens'],
        curr_rope=gen_context['ropes'],
        image_sizes=[image_shapes],
        new_token_ids=NEW_TOKEN_IDS,
    )
    for key, value in generation_input.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input[key] = value.to(device=device)
    
    # CFG text context
    generation_input_cfg_text = MODEL.prepare_vae_latent_cfg(
        curr_kvlens=cfg_text_context['kv_lens'],
        curr_rope=cfg_text_context['ropes'],
        image_sizes=[image_shapes],
    )
    for key, value in generation_input_cfg_text.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input_cfg_text[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input_cfg_text[key] = value.to(device=device)
    
    # CFG img context
    generation_input_cfg_img = MODEL.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_context['kv_lens'],
        curr_rope=cfg_img_context['ropes'],
        image_sizes=[image_shapes],
    )
    for key, value in generation_input_cfg_img.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                generation_input_cfg_img[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                generation_input_cfg_img[key] = value.to(device=device)
    
    os.environ["TQDM_DISABLE"] = "1"
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        unpacked_latent = MODEL.generate_image(
            past_key_values=gen_context['past_key_values'],
            cfg_text_past_key_values=cfg_text_context['past_key_values'],
            cfg_img_past_key_values=cfg_img_context['past_key_values'],
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            enable_taylorseer=enable_taylorseer,
            fresh_threshold=fresh_threshold,
            max_order=max_order,
            first_enhance=first_enhance,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
        )
    
    # Decode latent.
    H, W = image_shapes
    h, w = H // MODEL.latent_downsample, W // MODEL.latent_downsample
    
    latent = unpacked_latent[0].reshape(1, h, w, MODEL.latent_patch_size, MODEL.latent_patch_size, MODEL.latent_channel)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, MODEL.latent_channel, h * MODEL.latent_patch_size, w * MODEL.latent_patch_size)
    latent = latent.to(device=device, dtype=torch.bfloat16)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        decoded_image = VAE_MODEL.decode(latent)
    
    edited_image = (
        (decoded_image * 0.5 + 0.5).clamp(0, 1)[0]
        .permute(1, 2, 0)
        .mul(255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    
    return Image.fromarray(edited_image)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_type": MODEL_TYPE})


@app.route("/edit", methods=["POST"])
def edit_image_endpoint():
    """
    Image editing API endpoint.

    Request body (JSON):
    {
        "image": "<base64-encoded input image>",
        "edit_prompt": "<editing instruction>",
        "num_timesteps": 50,           # optional
        "cfg_text_scale": 4.0,         # optional (cfg_scale alias is also accepted)
        "cfg_img_scale": 2.0,          # optional
        "cfg_interval": [0.0, 1.0],    # optional
        "timestep_shift": 3.0,         # optional
        "cfg_renorm_min": 0.0,         # optional
        "cfg_renorm_type": "text_channel",  # optional
        "resolution_scale": 0.75,      # optional; reduces latent size for speed (0, 1.0] (default 0.75)
        "enable_taylorseer": true,     # optional; enable TaylorSeer acceleration (default true)
        "fresh_threshold": 3,          # optional; TaylorSeer parameter (default 3)
        "max_order": 6,                # optional; TaylorSeer parameter (default 6)
        "first_enhance": 5,            # optional; TaylorSeer parameter (default 5)
    }

    Response body (JSON):
    {
        "success": true,
        "image": "<base64-encoded edited image>",
        "error": null
    }

    resolution_scale controls latent size and speed/quality tradeoff.
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

        num_timesteps = data.get("num_timesteps", 50)
        cfg_text_scale = data.get("cfg_text_scale", data.get("cfg_scale", 4.0))
        cfg_img_scale = data.get("cfg_img_scale", 2.0)
        cfg_interval = data.get("cfg_interval")
        if not isinstance(cfg_interval, list) or len(cfg_interval) != 2:
            cfg_interval = [0.0, 1.0]
        timestep_shift = data.get("timestep_shift", 3.0)
        cfg_renorm_min = data.get("cfg_renorm_min", 0.0)
        cfg_renorm_type = data.get("cfg_renorm_type", "text_channel")
        resolution_scale = data.get("resolution_scale", 0.75)
        enable_taylorseer = data.get("enable_taylorseer", True)
        fresh_threshold = data.get("fresh_threshold", 3)
        max_order = data.get("max_order", 6)
        first_enhance = data.get("first_enhance", 5)

        edited_image = edit_image_bagel(
            image=image,
            edit_prompt=edit_prompt,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            timestep_shift=timestep_shift,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            resolution_scale=resolution_scale,
            enable_taylorseer=enable_taylorseer,
            fresh_threshold=fresh_threshold,
            max_order=max_order,
            first_enhance=first_enhance,
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
    parser = argparse.ArgumentParser(description="BAGEL server (supports text-to-image and image editing)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--max_latent_size", type=int, default=64, help="BAGEL latent size")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60, flush=True)
    print("Starting BAGEL server...", flush=True)
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

        print("Loading model...", flush=True)
        load_bagel_model(args.model_path, device, args.max_latent_size)
        
        print(f"[START] Starting BAGEL server on {args.host}:{args.port}", flush=True)
        print(f"   Model path: {args.model_path}", flush=True)
        if physical_gpu_id is not None:
            print(f"   Device: {device} (physical GPU: {physical_gpu_id})", flush=True)
        else:
            print(f"   Device: {device}", flush=True)

        log = logging.getLogger('werkzeug')
        log.setLevel(logging.INFO)
        logging.getLogger('werkzeug').disabled = False

        # Threaded mode is configurable because model calls are GPU-bound.
        bagel_threaded = os.environ.get("EDIT_SERVER_THREADED", "true").lower() == "true"
        print(f"   Threaded mode: {bagel_threaded}", flush=True)

        print(f"   TaylorSeer acceleration: enabled by default; request can override enable_taylorseer", flush=True)
        print(f"   [TaylorSeer] Defaults: fresh_threshold=3, max_order=6, first_enhance=5", flush=True)
        
        app.run(host=args.host, port=args.port, threaded=bagel_threaded)
    except Exception as e:
        print(f"[ERROR] Server startup failed: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
