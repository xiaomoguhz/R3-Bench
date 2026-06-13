#!/usr/bin/env python3
"""CLIP reward service."""

import argparse
import io
import base64
import os
import sys
import logging

import torch
from flask import Flask, request, jsonify
from PIL import Image

CLIP_MODULE_PATH = None
CLIP_AVAILABLE = False
clip = None

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    pass

app = Flask(__name__)

CLIP_MODEL = None
CLIP_PREPROCESS = None
DEVICE = None


def load_clip_module(clip_module_path: str = None):
    """Load a custom CLIP module from the specified path."""
    global clip, CLIP_AVAILABLE, CLIP_MODULE_PATH
    
    if clip_module_path:
        clip_module_path = os.path.abspath(clip_module_path)
        if not os.path.exists(clip_module_path):
            raise RuntimeError(f"CLIP module path does not exist: {clip_module_path}")
        
        if clip_module_path.endswith('.py'):
            clip_dir = os.path.dirname(clip_module_path)
        else:
            clip_dir = clip_module_path
        
        clip_parent_dir = os.path.dirname(clip_dir)
        if clip_parent_dir not in sys.path:
            sys.path.insert(0, clip_parent_dir)
        
        try:
            if 'clip' in sys.modules:
                del sys.modules['clip']
            
            import clip as custom_clip
            clip = custom_clip
            CLIP_AVAILABLE = True
            CLIP_MODULE_PATH = clip_module_path
            print(f"Loaded custom CLIP module from: {clip_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to load custom CLIP module from {clip_module_path}: {e}")
    elif not CLIP_AVAILABLE:
        raise RuntimeError("CLIP is not available. Please install: pip install git+https://github.com/openai/CLIP.git or specify --clip_module_path")


def load_clip_model(model_name: str, device: torch.device, model_path: str = None):
    """Load the CLIP model onto the specified device."""
    global CLIP_MODEL, CLIP_PREPROCESS
    
    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP is not available. Please install: pip install git+https://github.com/openai/CLIP.git")
    
    if model_path and os.path.isfile(model_path):
        print(f"Loading CLIP model from checkpoint: {model_path}...")
        CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_path, device=device)
    else:
        print(f"Loading CLIP model: {model_name}...")
        
        # Multiple server instances can share the same CLIP cache on one node.
        cache_dir = os.path.expanduser("~/.cache/clip")
        lock_file = os.path.join(cache_dir, ".cache_lock")
        
        if os.path.exists(cache_dir):
            model_name_clean = model_name.replace('/', '-').replace('_', '-')
            possible_files = [
                os.path.join(cache_dir, f"{model_name_clean}.pt"),
                os.path.join(cache_dir, f"{model_name.replace('/', '_')}.pt"),
                os.path.join(cache_dir, f"{model_name}.pt"),
            ]
            if "ViT-B/32" in model_name:
                possible_files.append(os.path.join(cache_dir, "ViT-B-32.pt"))
            if "RN50" in model_name:
                possible_files.append(os.path.join(cache_dir, "RN50.pt"))

            # Remove clearly incomplete cache files under a file lock.
            import fcntl
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(lock_file, 'w') as lock:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        
                        for cache_file in possible_files:
                            if os.path.exists(cache_file):
                                file_size = os.path.getsize(cache_file)
                                if file_size < 300 * 1024 * 1024:  # less than 300 MB
                                    print(f"Warning: possibly incomplete cache file: {cache_file} (size: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                                    print(f"  Removing incomplete cache file; will re-download...", flush=True)
                                    try:
                                        os.remove(cache_file)
                                    except Exception as e:
                                        print(f"  Warning: failed to delete cache file: {e}", flush=True)

                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                    except BlockingIOError:
                        print(f"  Another instance is using the cache; skipping pre-check...", flush=True)
            except ImportError:
                print(f"  Warning: file locking not supported (non-Linux), skipping pre-check", flush=True)
            except Exception as e:
                print(f"  Warning: file lock operation failed: {e}, skipping pre-check", flush=True)
        
        cache_dir = os.path.expanduser("~/.cache/clip")
        lock_file = os.path.join(cache_dir, ".download_lock")
        model_cache_file = None
        
        if "ViT-B/32" in model_name:
            model_cache_file = os.path.join(cache_dir, "ViT-B-32.pt")
        else:
            model_name_clean = model_name.replace('/', '-').replace('_', '-')
            model_cache_file = os.path.join(cache_dir, f"{model_name_clean}.pt")
        
        import fcntl
        import time
        cache_exists = False
        is_downloader = False
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(lock_file, 'w') as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    is_downloader = True
                    
                    if model_cache_file and os.path.exists(model_cache_file):
                        file_size = os.path.getsize(model_cache_file)
                        if file_size >= 300 * 1024 * 1024:  # at least 300 MB
                            cache_exists = True
                            print(f"Found existing model cache: {model_cache_file} (size: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                        else:
                            print(f"Cache file is incomplete; this instance will download...", flush=True)
                            # Keep the lock and proceed to download
                    else:
                        print(f"Cache file not found; this instance will download...", flush=True)
                        # Keep the lock and proceed to download

                except BlockingIOError:
                    print(f"Another instance is downloading the model; waiting...", flush=True)
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)  # block until lock is released
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

                    max_wait = 120  # wait at most 120 seconds
                    wait_interval = 2  # check every 2 seconds
                    waited = 0
                    while waited < max_wait:
                        if model_cache_file and os.path.exists(model_cache_file):
                            file_size = os.path.getsize(model_cache_file)
                            if file_size >= 300 * 1024 * 1024:
                                print(f"Another instance finished downloading; using existing cache (size: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                                cache_exists = True
                                break
                        time.sleep(wait_interval)
                        waited += wait_interval
                        if waited % 10 == 0:
                            print(f"   Still waiting... ({waited}s / {max_wait}s elapsed)", flush=True)

                    if not cache_exists:
                        print(f"Warning: wait timed out; will attempt to re-download...", flush=True)
                        is_downloader = True

        except ImportError:
            print(f"Warning: file locking not supported (non-Linux); checking cache directly", flush=True)
            if model_cache_file and os.path.exists(model_cache_file):
                file_size = os.path.getsize(model_cache_file)
                if file_size >= 300 * 1024 * 1024:
                    cache_exists = True
                    print(f"Found existing model cache: {model_cache_file} (size: {file_size / 1024 / 1024:.1f}MB)", flush=True)
        except Exception as e:
            print(f"Warning: file lock operation failed: {e}; checking cache directly", flush=True)
            if model_cache_file and os.path.exists(model_cache_file):
                file_size = os.path.getsize(model_cache_file)
                if file_size >= 300 * 1024 * 1024:
                    cache_exists = True
                    print(f"Found existing model cache: {model_cache_file} (size: {file_size / 1024 / 1024:.1f}MB)", flush=True)

        if cache_exists:
            time.sleep(0.5)  # wait 0.5 s to ensure file write is complete

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not cache_exists and is_downloader:
                    try:
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(lock_file, 'w') as lock:
                            print(f"Acquiring download lock (this instance will download; others will wait)...", flush=True)
                            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                            
                            if model_cache_file and os.path.exists(model_cache_file):
                                file_size = os.path.getsize(model_cache_file)
                                if file_size >= 300 * 1024 * 1024:
                                    print(f"Another instance already downloaded; using existing cache", flush=True)
                                    cache_exists = True
                                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                                else:
                                    print(f"Downloading CLIP model (this instance; others will wait)...", flush=True)
                                    CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                    print(f"Model download complete; releasing lock", flush=True)
                                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                                    break
                            else:
                                print(f"Downloading CLIP model (this instance; others will wait)...", flush=True)
                                CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                print(f"Model download complete; releasing lock", flush=True)
                                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                                break
                    except ImportError:
                        print(f"Warning: file locking not supported (non-Linux); checking cache...", flush=True)
                        if model_cache_file and os.path.exists(model_cache_file):
                            file_size = os.path.getsize(model_cache_file)
                            if file_size >= 300 * 1024 * 1024:
                                print(f"Found existing model cache; using it", flush=True)
                                cache_exists = True
                            else:
                                print(f"Warning: cache file is incomplete; will re-download", flush=True)
                                cache_exists = False
                        if not cache_exists:
                            print(f"Downloading CLIP model...", flush=True)
                            CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                        else:
                            CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                        break
                    except Exception as e:
                        print(f"Warning: file lock operation failed: {e}; checking cache...", flush=True)
                        import time
                        time.sleep(2)
                        if model_cache_file and os.path.exists(model_cache_file):
                            file_size = os.path.getsize(model_cache_file)
                            if file_size >= 300 * 1024 * 1024:
                                print(f"Found existing model cache; using it (size: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                                cache_exists = True
                                CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                break
                        print(f"Cache not found; waiting for another instance to download (up to 30s)...", flush=True)
                        max_wait_retry = 30
                        wait_interval_retry = 2
                        waited_retry = 0
                        while waited_retry < max_wait_retry:
                            if model_cache_file and os.path.exists(model_cache_file):
                                file_size = os.path.getsize(model_cache_file)
                                if file_size >= 300 * 1024 * 1024:
                                    print(f"Another instance finished; using existing cache (size: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                                    cache_exists = True
                                    CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                    break
                            time.sleep(wait_interval_retry)
                            waited_retry += wait_interval_retry
                        if cache_exists:
                            break
                        print(f"Warning: wait timed out; retrying lock acquisition and download...", flush=True)
                        try:
                            os.makedirs(cache_dir, exist_ok=True)
                            with open(lock_file, 'w') as lock_retry:
                                fcntl.flock(lock_retry.fileno(), fcntl.LOCK_EX)
                                if model_cache_file and os.path.exists(model_cache_file):
                                    file_size = os.path.getsize(model_cache_file)
                                    if file_size >= 300 * 1024 * 1024:
                                        print(f"Another instance already downloaded; using existing cache", flush=True)
                                        cache_exists = True
                                        fcntl.flock(lock_retry.fileno(), fcntl.LOCK_UN)
                                        CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                        break
                                print(f"Downloading CLIP model (this instance)...", flush=True)
                                CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                                fcntl.flock(lock_retry.fileno(), fcntl.LOCK_UN)
                                break
                        except Exception as retry_error:
                            print(f"Warning: lock retry failed: {retry_error}; downloading directly as last resort (concurrent download possible)", flush=True)
                            CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                            break
                elif not cache_exists and not is_downloader:
                    print(f"Waiting for another instance to download the model...", flush=True)
                    max_wait = 180  # wait at most 180 seconds
                    wait_interval = 2
                    waited = 0
                    while waited < max_wait:
                        try:
                            with open(lock_file, 'w') as lock:
                                try:
                                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                                    if model_cache_file and os.path.exists(model_cache_file):
                                        file_size = os.path.getsize(model_cache_file)
                                        if file_size >= 300 * 1024 * 1024:
                                            print(f"Another instance finished; using existing cache (size: {file_size / 1024 / 1024:.1f}MB)", flush=True)
                                            cache_exists = True
                                            break
                                except BlockingIOError:
                                    pass
                        except Exception:
                            pass
                        
                        time.sleep(wait_interval)
                        waited += wait_interval
                        if waited % 10 == 0:
                            print(f"   Still waiting... ({waited}s / {max_wait}s elapsed)", flush=True)

                    if cache_exists:
                        CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                        break
                    else:
                        print(f"Warning: wait timed out; this instance will attempt to download...", flush=True)
                        is_downloader = True
                        cache_exists = False
                        continue
                else:
                    CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
                    break
            except (RuntimeError, Exception) as e:
                error_msg = str(e).lower()
                is_checksum_error = (
                    "sha256 checksum" in error_msg or 
                    "checksum" in error_msg or
                    "does not not match" in error_msg or
                    "does not match" in error_msg
                )
                
                if is_checksum_error:
                    if attempt < max_retries - 1:
                        print(f"Warning: CLIP model checksum failed; clearing cache and retrying (attempt {attempt + 1}/{max_retries})...", flush=True)
                        print(f"  Error: {e}", flush=True)

                        cache_dir = os.path.expanduser("~/.cache/clip")
                        lock_file = os.path.join(cache_dir, ".download_lock")  # use the shared lock file
                        cache_exists = False
                        is_downloader = True
                        
                        if os.path.exists(cache_dir):
                            import shutil
                            import fcntl
                            try:
                                os.makedirs(cache_dir, exist_ok=True)
                                with open(lock_file, 'w') as lock:
                                    try:
                                        max_lock_wait = 10  # wait at most 10 seconds
                                        lock_acquired = False
                                        for lock_attempt in range(max_lock_wait):
                                            try:
                                                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                                                lock_acquired = True
                                                break
                                            except BlockingIOError:
                                                if lock_attempt < max_lock_wait - 1:
                                                    import time
                                                    time.sleep(1)
                                                    continue
                                                else:
                                                    print(f"  Warning: could not acquire file lock; another instance may be operating on cache", flush=True)
                                                    print(f"  Skipping delete; retrying download directly...", flush=True)
                                        
                                        if lock_acquired:
                                            model_name_clean = model_name.replace('/', '-').replace('_', '-')
                                            possible_patterns = [
                                                f"{model_name_clean}.pt",
                                                f"{model_name.replace('/', '_')}.pt",
                                                f"{model_name}.pt",
                                            ]
                                            if "ViT-B/32" in model_name:
                                                possible_patterns.append("ViT-B-32.pt")
                                            if "RN50" in model_name:
                                                possible_patterns.append("RN50.pt")
                                            
                                            deleted_any = False
                                            for pattern in possible_patterns:
                                                cache_file = os.path.join(cache_dir, pattern)
                                                if os.path.exists(cache_file):
                                                    print(f"  Removing corrupted cache file: {cache_file}", flush=True)
                                                    try:
                                                        os.remove(cache_file)
                                                        deleted_any = True
                                                    except Exception as remove_error:
                                                        print(f"  Warning: failed to delete {cache_file}: {remove_error}", flush=True)

                                            temp_patterns = [
                                                f"{model_name_clean}.pt.tmp",
                                                f"{model_name.replace('/', '_')}.pt.tmp",
                                                "ViT-B-32.pt.tmp" if "ViT-B/32" in model_name else None,
                                            ]
                                            for pattern in temp_patterns:
                                                if pattern is None:
                                                    continue
                                                temp_file = os.path.join(cache_dir, pattern)
                                                if os.path.exists(temp_file):
                                                    try:
                                                        os.remove(temp_file)
                                                        print(f"  Removed temp file: {temp_file}", flush=True)
                                                    except Exception:
                                                        pass
                                            
                                            if not deleted_any:
                                                print(f"  No specific model files found; leaving cache directory intact", flush=True)
                                            else:
                                                print(f"  Corrupted model cache files removed", flush=True)

                                            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                                    except ImportError:
                                        print(f"  Warning: file locking not supported (non-Linux); deleting cache directly", flush=True)
                                        model_name_clean = model_name.replace('/', '-').replace('_', '-')
                                        if "ViT-B/32" in model_name:
                                            cache_file = os.path.join(cache_dir, "ViT-B-32.pt")
                                            if os.path.exists(cache_file):
                                                try:
                                                    os.remove(cache_file)
                                                    print(f"  Removing corrupted cache file: {cache_file}", flush=True)
                                                except Exception:
                                                    pass
                            except Exception as cleanup_error:
                                print(f"  Warning: error while clearing cache: {cleanup_error}", flush=True)
                                import traceback
                                traceback.print_exc()
                        
                        import time
                        wait_time = 1 + (attempt * 2)
                        print(f"  Waiting {wait_time}s before retrying...", flush=True)
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Error: CLIP model load failed after {max_retries} attempts", flush=True)
                        print(f"  Last error: {e}", flush=True)
                        print(f"  Remove ~/.cache/clip and restart the server if the error persists", flush=True)
                        raise
                else:
                    print(f"Error: CLIP model load failed: {e}", flush=True)
                    raise
    
    print(f"CLIP model loaded successfully on {device}")


@torch.no_grad()
def compute_clip_score(image: Image.Image, prompt: str) -> float:
    """Compute CLIP cosine similarity (raw value, approximately in [-1, 1])."""
    try:
        image_tensor = CLIP_PREPROCESS(image).unsqueeze(0).to(DEVICE)
        text_tokens = clip.tokenize([prompt]).to(DEVICE)
        
        image_features = CLIP_MODEL.encode_image(image_tensor)
        text_features = CLIP_MODEL.encode_text(text_tokens)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).item()
        return similarity
    except Exception as e:
        print(f"CLIP score computation error: {e}")
        return 0.0


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "clip_reward"})


@app.route("/compute_reward", methods=["POST"])
def compute_reward_endpoint():
    """
    Reward computation API endpoint.

    Request format (JSON):
    {
        "image": "base64-encoded image",
        "prompt": "text prompt",
        "reward_type": "clip"  # only "clip" is supported
    }

    Response format (JSON):
    {
        "success": true,
        "score": 0.85,
        "raw_score": 0.85,
        "reward_type": "clip",
        "error": null
    }
    """
    print(f"[REQUEST] POST /compute_reward | from: {request.remote_addr}", flush=True)
    try:
        data = request.get_json()
        
        image_b64 = data.get("image")
        prompt = data.get("prompt")
        reward_type = data.get("reward_type", "clip")
        
        if not image_b64 or not prompt:
            return jsonify({
                "success": False,
                "score": 0.0,
                "error": "Missing required fields: image or prompt"
            }), 400
        
        if reward_type != "clip":
            return jsonify({
                "success": False,
                "score": 0.0,
                "error": f"Unsupported reward type: {reward_type}. Only 'clip' is supported."
            }), 400
        
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt_preview = prompt[:50] + "..." if prompt and len(prompt) > 50 else prompt
        print(f"[REQUEST] reward_type={reward_type}, prompt_preview={prompt_preview}", flush=True)
        
        score = compute_clip_score(image, prompt)
        
        print(f"[RESPONSE] Success | score={score:.4f}", flush=True)
        
        return jsonify({
            "success": True,
            "score": score,
            "raw_score": score,
            "reward_type": "clip",
            "error": None
        })
        
    except Exception as e:
        print(f"[RESPONSE] Failed | error={str(e)[:100]}", flush=True)
        return jsonify({
            "success": False,
            "score": 0.0,
            "error": str(e)
        }), 500



def parse_args():
    parser = argparse.ArgumentParser(description="CLIP reward server")
    parser.add_argument("--model_name", type=str, default="ViT-B/32", help="CLIP model name (for standard CLIP)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a CLIP model checkpoint (.pt file)")
    parser.add_argument("--clip_module_path", type=str, default=None, help="Path to a custom CLIP module (clip.py)")
    parser.add_argument("--port", type=int, default=6001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
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
    
    global DEVICE
    DEVICE = device
    
    if args.clip_module_path:
        load_clip_module(args.clip_module_path)

    load_clip_model(args.model_name, device, args.model_path)

    print(f"Starting CLIP reward server on {args.host}:{args.port}")
    if args.model_path:
        print(f"   Model checkpoint: {args.model_path}")
    else:
        print(f"   Model: {args.model_name}")
    if args.clip_module_path:
        print(f"   CLIP module: {args.clip_module_path}")
    print(f"   Device: {device}")
    
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
