#!/usr/bin/env python3
"""SAM3 reward service."""

import argparse
import base64
import io
import os
import sys
import json
import logging
from typing import Optional

from flask import Flask, request, jsonify
from PIL import Image
import torch

# Add distributed_services root so sam3_deps imports resolve.
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICES_ROOT = os.path.normpath(os.path.join(SERVER_DIR, ".."))
if SERVICES_ROOT not in sys.path:
    sys.path.insert(0, SERVICES_ROOT)

from sam3_deps.sam3_reward import initialize_sam3_reward, compute_sam3_reward  # type: ignore  # noqa: E402

app = Flask(__name__)

SAM3_DEVICE: Optional[torch.device] = None


def load_sam3(bpe_path: Optional[str], ckpt_path: Optional[str], metadata_jsonl: Optional[str], device: torch.device):
    """Pre-load the SAM3 model and metadata."""
    global SAM3_DEVICE
    SAM3_DEVICE = device
    initialize_sam3_reward(
        metadata_jsonl_path=metadata_jsonl,
        device=str(device),
        bpe_path=bpe_path,
        checkpoint_path=ckpt_path,
    )
    print(f"[INFO] SAM3 preloaded on {device} (bpe={bpe_path}, ckpt={ckpt_path})")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "device": str(SAM3_DEVICE)})


def _decode_image(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


@app.route("/compute_sam3_reward", methods=["POST"])
def compute_sam3_reward_endpoint():
    try:
        print(f"[REQUEST] POST /compute_sam3_reward | from: {request.remote_addr}", flush=True)
        
        data = request.get_json(force=True)
        image_b64 = data.get("image")
        prompt = data.get("prompt")
        category = data.get("category", "object")
        ground_truth = data.get("ground_truth")

        prompt_preview = prompt[:50] + "..." if prompt and len(prompt) > 50 else prompt
        print(f"[REQUEST] category={category}, prompt_preview={prompt_preview}, has_ground_truth={bool(ground_truth)}", flush=True)

        if not image_b64 or not prompt:
            print(f"[ERROR] Missing image or prompt", flush=True)
            return jsonify({"success": False, "score": 0.0, "error": "Missing image or prompt"}), 400

        image = _decode_image(image_b64)

        result = compute_sam3_reward(
            image_path=None,
            prompt=prompt,
            category=category,
            ground_truth=ground_truth,
            device=str(SAM3_DEVICE or "cuda"),
            image=image,
        )
        error_msg = result.get("error")
        if error_msg is None or error_msg == "":
            if not result.get("success", False):
                error_msg = "SAM3 reward computation failed (no error message provided)"
            else:
                error_msg = ""
        
        if result["success"]:
            print(f"[RESPONSE] Success | score={result['score']:.4f}", flush=True)
        else:
            print(f"[RESPONSE] Failed | error={str(error_msg)[:100]}", flush=True)
        
        return jsonify({"success": result["success"], "score": result["score"], "error": str(error_msg)})
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] SAM3 reward computation exception: {error_detail}", flush=True)
        return jsonify({"success": False, "score": 0.0, "error": str(e)}), 500


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 reward server")
    parser.add_argument("--bpe_path", type=str, default=os.environ.get("SAM3_BPE_PATH"))
    parser.add_argument("--ckpt_path", type=str, default=os.environ.get("SAM3_CKPT_PATH"))
    parser.add_argument("--metadata_jsonl", type=str, default=os.environ.get("SAM3_METADATA_JSONL"))
    parser.add_argument("--device", type=int, default=int(os.environ.get("SAM3_DEVICE", "0")))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7001)
    return parser.parse_args()


def main():
    args = parse_args()
    # With CUDA_VISIBLE_DEVICES set, each process sees its assigned GPU as cuda:0.
    if torch.cuda.is_available():
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    
    load_sam3(args.bpe_path, args.ckpt_path, args.metadata_jsonl, device)
    print(f"SAM3 reward server running on {args.host}:{args.port}, device={device}")
    
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.disabled = False
    
    app.run(host=args.host, port=args.port, threaded=True, processes=1)


if __name__ == "__main__":
    main()
