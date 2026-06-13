#!/usr/bin/env python3
"""Check the local R3-Refiner training runtime.

The script verifies service endpoints, data paths, torch/model metadata, and
Ray startup. It is intended for local troubleshooting; review paths and endpoint
addresses before sharing the output.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path


EXPORT_RE = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80, flush=True)


def item(label: str, value: object) -> None:
    print(f"{label}: {value}", flush=True)


def warn(message: str) -> None:
    print(f"[WARNING] {message}", flush=True)


def error(message: str) -> None:
    print(f"[ERROR] {message}", flush=True)


def ok(message: str) -> None:
    print(f"[OK] {message}", flush=True)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def split_specs(value: str) -> list[str]:
    specs = []
    for item_value in (value or "").split(","):
        item_value = item_value.strip().strip("'").strip('"')
        if not item_value:
            continue
        specs.append(item_value.split("@", 1)[0])
    return specs


def split_endpoints(value: str, limit: int) -> list[str]:
    endpoints = []
    for endpoint in (value or "").split(","):
        endpoint = endpoint.strip()
        if endpoint:
            endpoints.append(endpoint)
    return endpoints[:limit]


def load_service_env(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        warn(f"service env does not exist: {env_path}")
        return values

    for line in env_path.read_text(encoding="utf-8").splitlines():
        match = EXPORT_RE.match(line.strip())
        if not match:
            continue
        key, raw_value = match.groups()
        try:
            parsed = shlex.split(raw_value, posix=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = raw_value.strip().strip("'").strip('"')
        values[key] = value
        os.environ[key] = value
    return values


def run_get_config(repo_root: Path, timeout: int) -> None:
    deploy_script = repo_root / "distributed_services" / "scripts" / "deploy_services.sh"
    if not deploy_script.exists():
        error(f"deploy script not found: {deploy_script}")
        return

    print(f"[cmd] bash {deploy_script.relative_to(repo_root)} get_config", flush=True)
    try:
        result = subprocess.run(
            ["bash", str(deploy_script), "get_config"],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        error(f"get_config timed out after {timeout}s")
        return

    if result.stdout.strip():
        print(result.stdout.rstrip(), flush=True)
    if result.stderr.strip():
        print("[get_config stderr]", flush=True)
        print(result.stderr.rstrip(), flush=True)
    if result.returncode == 0:
        ok("get_config finished")
    else:
        error(f"get_config exited with code {result.returncode}")


def check_path(path_text: str, label: str, expect_dir: bool = False) -> bool:
    if not path_text:
        warn(f"{label} is empty")
        return False
    path = Path(path_text)
    exists = path.is_dir() if expect_dir else path.exists()
    if exists:
        kind = "dir" if path.is_dir() else "file"
        ok(f"{label} exists ({kind}): {path}")
        return True
    error(f"{label} does not exist: {path}")
    return False


def load_records(path: Path, limit: int = 64) -> list[dict]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if len(records) >= limit:
                    break
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data[:limit] if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "train", "validation", "val"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value[:limit] if isinstance(x, dict)]
    return []


def resolve_image(image_path: str, image_dir: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    if image_dir:
        return Path(image_dir) / image_path
    return path


def inspect_data_file(data_file: str, image_dir: str) -> None:
    path = Path(data_file)
    if not check_path(str(path), "data file"):
        return
    try:
        records = load_records(path)
    except Exception:
        error(f"failed to load data file: {path}")
        traceback.print_exc()
        return

    item("sampled records", len(records))
    if not records:
        warn(f"no records sampled from {path}")
        return

    first = records[0]
    item("first record keys", sorted(first.keys()))
    images = first.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        resolved = resolve_image(images[0], image_dir)
        if resolved.exists():
            ok(f"first image exists: {resolved}")
        else:
            error(f"first image is missing: {resolved}")
    else:
        warn("first record has no images[0] string; this may be OK for text-only samples")


def endpoint_health(endpoint: str, timeout: float) -> None:
    url = endpoint.rstrip("/") + "/health"
    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(300).decode("utf-8", errors="replace").strip()
            elapsed_ms = int((time.time() - start) * 1000)
            ok(f"{url} -> HTTP {response.status} in {elapsed_ms} ms; body={body!r}")
    except urllib.error.HTTPError as exc:
        elapsed_ms = int((time.time() - start) * 1000)
        error(f"{url} -> HTTP {exc.code} in {elapsed_ms} ms")
    except Exception as exc:
        elapsed_ms = int((time.time() - start) * 1000)
        error(f"{url} -> {type(exc).__name__}: {exc} after {elapsed_ms} ms")


def check_transformers(model_path: str) -> None:
    if not model_path:
        warn("MODEL_PATH is empty; skip tokenizer/processor check")
        return
    if not Path(model_path).exists():
        error(f"MODEL_PATH does not exist; skip tokenizer/processor check: {model_path}")
        return
    try:
        from transformers import AutoProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        ok(f"tokenizer loaded: {tokenizer.__class__.__name__}")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        ok(f"processor loaded: {processor.__class__.__name__}")
    except Exception:
        error("tokenizer/processor load failed")
        traceback.print_exc()


def check_torch() -> None:
    try:
        import torch

        item("torch version", torch.__version__)
        item("torch.cuda.is_available", torch.cuda.is_available())
        item("torch.cuda.device_count", torch.cuda.device_count())
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                item(f"cuda:{index}", torch.cuda.get_device_name(index))
    except Exception:
        error("torch check failed")
        traceback.print_exc()


def run_ray_probe(timeout: int, gpu_actor: bool) -> None:
    code = r'''
import json
import os
import socket
import traceback

try:
    import ray

    print("[ray-probe] ray version:", ray.__version__, flush=True)
    print("[ray-probe] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    include_dashboard = os.environ.get("R3REFINER_RAY_INCLUDE_DASHBOARD", "0").lower() in {"1", "true", "yes"}
    init_kwargs = {
        "ignore_reinit_error": True,
        "log_to_driver": True,
    }
    ray_address = os.environ.get("RAY_ADDRESS")
    if ray_address:
        init_kwargs["address"] = ray_address
    else:
        init_kwargs["include_dashboard"] = include_dashboard
        if os.environ.get("RAY_NUM_CPUS"):
            init_kwargs["num_cpus"] = int(os.environ["RAY_NUM_CPUS"])
        if os.environ.get("RAY_NUM_GPUS"):
            init_kwargs["num_gpus"] = int(os.environ["RAY_NUM_GPUS"])
        if os.environ.get("RAY_OBJECT_STORE_MEMORY"):
            init_kwargs["object_store_memory"] = int(os.environ["RAY_OBJECT_STORE_MEMORY"])
        if os.environ.get("RAY_TMPDIR"):
            init_kwargs["_temp_dir"] = os.environ["RAY_TMPDIR"]
        if os.environ.get("RAY_WORKER_REGISTER_TIMEOUT_SECONDS"):
            init_kwargs["_system_config"] = {
                "worker_register_timeout_seconds": int(os.environ["RAY_WORKER_REGISTER_TIMEOUT_SECONDS"])
            }
    print("[ray-probe] ray.init kwargs:", json.dumps(init_kwargs, sort_keys=True), flush=True)
    ray.init(**init_kwargs)
    print("[ray-probe] ray.init returned", flush=True)
    print("[ray-probe] cluster resources:", json.dumps(ray.cluster_resources(), sort_keys=True), flush=True)
    print("[ray-probe] available resources:", json.dumps(ray.available_resources(), sort_keys=True), flush=True)
    print("[ray-probe] nodes:", json.dumps(ray.nodes(), default=str)[:4000], flush=True)

    @ray.remote
    class CpuProbe:
        def run(self):
            return {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }

    cpu_actor = CpuProbe.remote()
    print("[ray-probe] cpu actor:", json.dumps(ray.get(cpu_actor.run.remote(), timeout=30), sort_keys=True), flush=True)

    if os.environ.get("R3REFINER_RAY_GPU_ACTOR", "1") == "1":
        @ray.remote(num_gpus=1)
        class GpuProbe:
            def run(self):
                result = {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                }
                try:
                    import torch
                    result["torch_version"] = torch.__version__
                    result["torch_cuda_is_available"] = torch.cuda.is_available()
                    result["torch_cuda_device_count"] = torch.cuda.device_count()
                    if torch.cuda.is_available():
                        result["cuda0_name"] = torch.cuda.get_device_name(0)
                except Exception as exc:
                    result["torch_error"] = repr(exc)
                return result

        gpu_actor = GpuProbe.remote()
        print("[ray-probe] gpu actor:", json.dumps(ray.get(gpu_actor.run.remote(), timeout=60), sort_keys=True), flush=True)

    ray.shutdown()
except Exception:
    print("[ray-probe] FAILED", flush=True)
    traceback.print_exc()
    raise
'''
    env = os.environ.copy()
    env["R3REFINER_RAY_GPU_ACTOR"] = "1" if gpu_actor else "0"
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", code],
            env=env,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        error(f"ray probe timed out after {timeout}s")
        return
    if result.returncode == 0:
        ok("ray probe finished")
    else:
        error(f"ray probe exited with code {result.returncode}")


def tail_file(path: Path, max_lines: int) -> None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        warn(f"cannot read {path}: {exc}")
        return
    if not lines:
        return
    print(f"--- tail {path} ---", flush=True)
    for line in lines[-max_lines:]:
        print(line, flush=True)


def inspect_ray_logs(max_lines: int) -> None:
    log_dir = Path(os.environ.get("RAY_TMPDIR", "/tmp/ray")) / "session_latest" / "logs"
    item("ray log dir", log_dir)
    if not log_dir.exists():
        warn("ray log dir does not exist")
        return

    files = sorted(log_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    item("latest ray log files", [p.name for p in files[:20]])
    for name in ("gcs_server.err", "raylet.err", "dashboard_agent.log"):
        path = log_dir / name
        if path.exists():
            tail_file(path, max_lines)

    worker_errs = [p for p in files if p.name.startswith("worker") and p.suffix == ".err"]
    for path in worker_errs[:3]:
        tail_file(path, max_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local R3-Refiner training runtime.")
    parser.add_argument("--repo-root", default=str(repo_root_from_script()))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--train-data-path", default=os.environ.get("TRAIN_DATA_PATH", ""))
    parser.add_argument("--val-data-path", default=os.environ.get("VAL_DATA_PATH", ""))
    parser.add_argument("--image-dir", default=os.environ.get("IMAGE_DIR", ""))
    parser.add_argument("--skip-get-config", action="store_true")
    parser.add_argument("--get-config-timeout", type=int, default=60)
    parser.add_argument("--endpoint-timeout", type=float, default=3.0)
    parser.add_argument("--max-endpoints", type=int, default=4)
    parser.add_argument("--ray-timeout", type=int, default=180)
    parser.add_argument("--skip-ray", action="store_true")
    parser.add_argument("--skip-gpu-actor", action="store_true")
    parser.add_argument("--tail-ray-log-lines", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    env_path = repo_root / "distributed_services" / "config" / "service_endpoints.env"

    section("1. Basic environment")
    item("hostname", socket.gethostname())
    item("cwd", os.getcwd())
    item("repo_root", repo_root)
    item("python", sys.executable)
    item("python version", sys.version.replace("\n", " "))
    for key in (
        "CONDA_DEFAULT_ENV",
        "CUDA_VISIBLE_DEVICES",
        "N_GPUS_PER_NODE",
        "GPUS_PER_NODE",
        "MODEL_PATH",
        "TRAIN_DATA_PATH",
        "VAL_DATA_PATH",
        "IMAGE_DIR",
        "HYDRA_FULL_ERROR",
        "PYTHONFAULTHANDLER",
        "RAY_ADDRESS",
        "RAY_TMPDIR",
        "RAY_INCLUDE_DASHBOARD",
        "RAY_NUM_CPUS",
        "RAY_NUM_GPUS",
        "RAY_OBJECT_STORE_MEMORY",
        "RAY_WORKER_REGISTER_TIMEOUT_SECONDS",
        "RAY_USE_MULTIPROCESSING_CPU_COUNT",
    ):
        item(key, os.environ.get(key, ""))

    if not args.skip_get_config:
        section("2. Generate and load service endpoints")
        run_get_config(repo_root, args.get_config_timeout)
    else:
        section("2. Load service endpoints")
        warn("skip get_config by request")

    service_values = load_service_env(env_path)
    if service_values:
        for key in (
            "EDIT_SERVER_ENDPOINTS",
            "REWARD_SERVER_ENDPOINTS",
            "CLIP_REWARD_SERVER_ENDPOINTS",
            "SELF_REWARD_SERVER_ENDPOINTS",
            "SAM3_REWARD_SERVER_ENDPOINTS",
            "REWARD_TYPE",
            "REWARD_TYPE_PER_GPU",
        ):
            value = os.environ.get(key, "")
            endpoints = split_endpoints(value, 999)
            if key.endswith("ENDPOINTS"):
                item(key, f"{len(endpoints)} endpoint(s): {endpoints[:args.max_endpoints]}")
            else:
                item(key, value)

    section("3. Path and data checks")
    check_path(args.model_path, "MODEL_PATH")
    if args.image_dir:
        check_path(args.image_dir, "IMAGE_DIR", expect_dir=True)
    else:
        warn("IMAGE_DIR is empty; relative images cannot be resolved here")

    train_files = split_specs(args.train_data_path)
    val_files = split_specs(args.val_data_path)
    item("train files", train_files)
    item("val files", val_files)
    for data_file in train_files + val_files:
        inspect_data_file(data_file, args.image_dir)

    section("4. Endpoint health checks")
    edit_endpoints = split_endpoints(os.environ.get("EDIT_SERVER_ENDPOINTS", ""), args.max_endpoints)
    reward_endpoints = split_endpoints(os.environ.get("REWARD_SERVER_ENDPOINTS", ""), args.max_endpoints)
    self_reward_endpoints = split_endpoints(os.environ.get("SELF_REWARD_SERVER_ENDPOINTS", ""), args.max_endpoints)
    if not edit_endpoints:
        warn("EDIT_SERVER_ENDPOINTS is empty")
    for endpoint in edit_endpoints:
        endpoint_health(endpoint, args.endpoint_timeout)
    if reward_endpoints:
        for endpoint in reward_endpoints:
            endpoint_health(endpoint, args.endpoint_timeout)
    elif self_reward_endpoints:
        for endpoint in self_reward_endpoints:
            endpoint_health(endpoint, args.endpoint_timeout)
    else:
        warn("REWARD_SERVER_ENDPOINTS and SELF_REWARD_SERVER_ENDPOINTS are empty")

    section("5. Torch and model metadata checks")
    check_torch()
    check_transformers(args.model_path)

    section("6. Ray init and actor checks")
    if args.skip_ray:
        warn("skip ray probe by request")
    else:
        run_ray_probe(args.ray_timeout, gpu_actor=not args.skip_gpu_actor)

    section("7. Ray log summary")
    inspect_ray_logs(args.tail_ray_log_lines)

    section("Done")
    print("For troubleshooting, sections 4-7 are usually the most relevant. Review local paths and addresses before sharing logs.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
