"""Clients for image-editing and reward services."""

import base64
import io
import os
import random
import threading
import time
from typing import List, Optional, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image


def _get_env_int(key: str, default: int) -> int:
    """Read an integer environment variable."""
    value = os.environ.get(key, str(default))
    value = value.strip().strip('"').strip("'")
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _get_env_bool(key: str, default: bool = True) -> bool:
    """Read a boolean environment variable."""
    value = os.environ.get(key, str(default).lower())
    value = value.strip().strip('"').strip("'").lower()
    return value == "true"


class ServiceClient:
    """Base service client with health checks and retry routing."""

    def __init__(
        self,
        endpoints: List[str],
        timeout: int = 180,
        health_check_timeout: int = 5,
        enable_health_check: bool = True
        ):
        """
        Args:
            endpoints: List of service endpoints for load balancing
            timeout: Request timeout in seconds
            health_check_timeout: Health-check request timeout in seconds
            enable_health_check: Whether to enable health checking (affects failure marking and logging)
        """
        self.endpoints = endpoints
        self.timeout = timeout
        self.health_check_timeout = health_check_timeout
        self.enable_health_check = enable_health_check
        # Round-robin pointer:
        # - Must be thread-safe (training workers issue concurrent requests via ThreadPoolExecutor)
        # - Randomize starting index so multiple worker processes don't create a hotspot at index 0
        self._rr_lock = threading.Lock()
        self._current_index = random.randint(0, max(0, len(endpoints) - 1))

        # endpoint -> (is_healthy, last_check_time, consecutive_failures)
        self._health_status: Dict[str, tuple] = {}
        self._last_full_health_check: float = 0.0
        self._last_request_time: float = 0.0

        self._last_health_summary: Optional[tuple[int, int]] = None  # (healthy_count, unhealthy_count)

        self._last_full_check_log_time: float = 0.0
        self._full_check_log_interval: float = 60.0

        self._unhealthy_endpoint_log_count: Dict[str, int] = {}
        self._unhealthy_log_interval: int = 50

        self._endpoints_lock = threading.Lock()

        # Endpoints are verified on first use.
        for endpoint in endpoints:
            self._health_status[endpoint] = (True, 0.0, 0)

    def update_endpoints(self, new_endpoints: List[str]) -> bool:
        """Update the endpoint list (thread-safe)."""
        if set(new_endpoints) == set(self.endpoints):
            return False

        if not self._endpoints_lock.acquire(blocking=False):
            return False
        try:
            old_count = len(self.endpoints)
            self.endpoints = new_endpoints
            self._current_index = 0

            # Keep status for existing endpoints and initialize new endpoints.
            new_health_status = {}
            for endpoint in new_endpoints:
                if endpoint in self._health_status:
                    new_health_status[endpoint] = self._health_status[endpoint]
                else:
                    new_health_status[endpoint] = (True, 0.0, 0)
            self._health_status = new_health_status

            # Keep cached model types only for active endpoints.
            if hasattr(self, '_model_types'):
                new_model_types = {}
                for endpoint in new_endpoints:
                    if endpoint in self._model_types:
                        new_model_types[endpoint] = self._model_types[endpoint]
                self._model_types = new_model_types

            print(f"[INFO] Endpoints updated: {old_count} -> {len(new_endpoints)}")
            return True
        finally:
            self._endpoints_lock.release()

    def _get_healthy_endpoints(self, force_check: bool = False) -> List[str]:
        """
        Return the list of currently healthy endpoints.

        Args:
            force_check: bypass cached health state

        Returns:
            List of healthy endpoint URLs
        """
        if not self.enable_health_check:
            return self.endpoints

        current_time = time.time()

        # Prefer passive checks; run active checks on first use, after long idle
        # periods, or when all cached endpoints appear unhealthy.
        long_idle_time = 600
        should_full_check = (
            force_check or
            (self._last_request_time > 0 and (current_time - self._last_request_time) >= long_idle_time) or
            (self._last_full_health_check == 0.0)  # First use
        )

        if should_full_check:
            if not force_check:
                random_delay = random.uniform(0, 1)
                time.sleep(random_delay)

            healthy_endpoints = []
            unhealthy_endpoints = []

            def check_single_endpoint(endpoint: str) -> tuple[str, bool]:
                """Check a single endpoint and return (endpoint, is_healthy)."""
                is_healthy = self._check_endpoint_health(endpoint)
                return endpoint, is_healthy

            max_concurrent_checks = min(16, len(self.endpoints))
            with ThreadPoolExecutor(max_workers=max_concurrent_checks) as executor:
                future_to_endpoint = {
                    executor.submit(check_single_endpoint, endpoint): endpoint
                    for endpoint in self.endpoints
                }

                for future in as_completed(future_to_endpoint):
                    try:
                        endpoint, is_healthy = future.result()
                        if is_healthy:
                            healthy_endpoints.append(endpoint)
                        else:
                            unhealthy_endpoints.append(endpoint)
                    except Exception as e:
                        endpoint = future_to_endpoint[future]
                        unhealthy_endpoints.append(endpoint)
                        print(f"[WARNING] Health check exception: {endpoint}, error: {e}")

            self._last_full_health_check = current_time

            current_summary = (len(healthy_endpoints), len(unhealthy_endpoints))

            if healthy_endpoints:
                should_log = False
                if self._last_health_summary is None:
                    if len(unhealthy_endpoints) > 0:
                        should_log = True
                elif current_summary != self._last_health_summary:
                    last_unhealthy = self._last_health_summary[1]

                    if len(unhealthy_endpoints) > 0:
                        if abs(len(unhealthy_endpoints) - last_unhealthy) >= 2:
                            should_log = True
                    elif last_unhealthy > 0:
                        should_log = True

                if should_log:
                    if len(unhealthy_endpoints) > 0:
                        print(
                            f"[INFO] Health check: {len(healthy_endpoints)}/{len(self.endpoints)} endpoints healthy, unhealthy: {len(unhealthy_endpoints)}"
                        )
                    else:
                        print(f"[INFO] Health check: {len(healthy_endpoints)}/{len(self.endpoints)} endpoints healthy")

                self._last_health_summary = current_summary
                return healthy_endpoints
            else:
                if self._last_health_summary != current_summary:
                    print(f"[WARNING] All endpoints are unhealthy; will still attempt requests (service may be starting up)")
                self._last_health_summary = current_summary
                return self.endpoints

        healthy_endpoints = [
            endpoint for endpoint in self.endpoints
            if self._health_status.get(endpoint, (True, 0.0, 0))[0]
        ]
        if not healthy_endpoints:
            return self._get_healthy_endpoints(force_check=True)

        return healthy_endpoints

    def _check_endpoint_health(self, endpoint: str) -> bool:
        """
        Check the health of a single endpoint.

        Args:
            endpoint: Service endpoint URL

        Returns:
            True if healthy, False otherwise
        """
        try:
            health_url = f"{endpoint}/health"
            response = requests.get(
                health_url,
                timeout=self.health_check_timeout,
                headers={"Content-Type": "application/json"}
            )
            is_healthy = response.status_code == 200
        except Exception as e:
            is_healthy = False

        current_time = time.time()
        was_healthy, last_check, failures = self._health_status.get(endpoint, (True, 0.0, 0))

        if is_healthy:
            consecutive_failures = 0
        else:
            consecutive_failures = failures + 1
            max_failure_count = 1000
            if consecutive_failures > max_failure_count:
                consecutive_failures = max_failure_count
            if consecutive_failures == max_failure_count:
                print(f"[INFO] Endpoint failure count capped at {max_failure_count}; will re-check periodically: {endpoint}")

            health_check_failure_threshold = _get_env_int("API_HEALTH_CHECK_FAILURE_THRESHOLD", 5)
            if consecutive_failures < health_check_failure_threshold:
                is_healthy = True
                consecutive_failures = failures + 1
            if was_healthy and consecutive_failures >= health_check_failure_threshold:
                if consecutive_failures == health_check_failure_threshold:
                    print(f"[WARNING] Endpoint became unhealthy: {endpoint} (consecutive failures: {consecutive_failures}, threshold: {health_check_failure_threshold})")

        self._health_status[endpoint] = (is_healthy, current_time, consecutive_failures)

        return is_healthy

    def _get_next_endpoint(self) -> str:
        """
        Return the next healthy endpoint using round-robin selection.

        Returns:
            A healthy service endpoint URL
        """
        healthy_endpoints = self._get_healthy_endpoints()

        if not healthy_endpoints:
            healthy_endpoints = self.endpoints

        if healthy_endpoints:
            with self._rr_lock:
                healthy_index = self._current_index % len(healthy_endpoints)
                endpoint = healthy_endpoints[healthy_index]
                self._current_index = (self._current_index + 1) % len(healthy_endpoints)
        else:
            with self._rr_lock:
                endpoint = self.endpoints[self._current_index % len(self.endpoints)]
                self._current_index = (self._current_index + 1) % len(self.endpoints)

        return endpoint

    def _post_with_retry(self, path: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST request with retry and health-check integration.

        Args:
            path: API path
            json_data: Request payload

        Returns:
            Response JSON

        Raises:
            RuntimeError: All retry attempts failed
        """
        last_error = None
        tried_endpoints: Set[str] = set()
        max_endpoint_attempts = len(self.endpoints) * 2  # Try all endpoints at most 2 rounds

        for attempt in range(max_endpoint_attempts):
            endpoint = self._get_next_endpoint()

            if len(tried_endpoints) >= len(self.endpoints):
                tried_endpoints.clear()
                current_time = time.time()
                if current_time - self._last_full_check_log_time >= self._full_check_log_interval:
                    print(f"[INFO] All endpoints tried; running full health check...", flush=True)
                self._last_full_check_log_time = current_time

                healthy_endpoints = self._get_healthy_endpoints(force_check=True)

                if not healthy_endpoints or len(healthy_endpoints) == 0:
                    if attempt < max_endpoint_attempts - 1:
                        wait_time = min(5, 2 ** (attempt // len(self.endpoints)))  # Exponential backoff, max 5 s
                        print(f"[WARNING] All endpoints unhealthy; retrying in {wait_time} s...", flush=True)
                        time.sleep(wait_time)

            if self.enable_health_check:
                is_healthy, _, consecutive_failures = self._health_status.get(
                    endpoint, (True, 0.0, 0)
                )

                health_check_failure_threshold = _get_env_int("API_HEALTH_CHECK_FAILURE_THRESHOLD", 5)

                if not is_healthy and consecutive_failures >= health_check_failure_threshold:
                    if consecutive_failures < health_check_failure_threshold + 3:
                        if self._check_endpoint_health(endpoint):
                            pass
                        else:
                            tried_endpoints.add(endpoint)
                            last_error = f"Endpoint {endpoint} is unhealthy (consecutive failures: {consecutive_failures})"
                            print(f"[WARNING] Skipping unhealthy endpoint: {endpoint} (consecutive failures: {consecutive_failures})")
                            continue
                    else:
                        current_time = time.time()
                        last_check_time = self._health_status.get(endpoint, (False, 0.0, 0))[1]
                        time_since_last_check = current_time - last_check_time

                        if consecutive_failures < 100:
                            check_interval = 5
                        elif consecutive_failures < 500:
                            check_interval = 20
                        else:
                            check_interval = 50

                        should_check = (
                            (consecutive_failures % check_interval == 0) or
                            (time_since_last_check > 60.0)
                        )

                        should_log = False

                        if consecutive_failures > 50:
                            log_interval = self._unhealthy_log_interval
                            log_count = self._unhealthy_endpoint_log_count.get(endpoint, 0)
                            expected_log_count = consecutive_failures // log_interval
                            if expected_log_count > log_count:
                                should_log = True
                                self._unhealthy_endpoint_log_count[endpoint] = expected_log_count
                        else:
                            should_log = True

                        if should_check:
                            if self._check_endpoint_health(endpoint):
                                self._unhealthy_endpoint_log_count.pop(endpoint, None)
                                print(f"[INFO] Endpoint recovered: {endpoint} (previous consecutive failures: {consecutive_failures})")
                            else:
                                tried_endpoints.add(endpoint)
                                last_error = f"Endpoint {endpoint} is unhealthy (consecutive failures: {consecutive_failures})"
                                if should_log:
                                    print(f"[WARNING] Skipping unhealthy endpoint: {endpoint} (consecutive failures: {consecutive_failures})")
                                continue
                        else:
                            tried_endpoints.add(endpoint)
                            last_error = f"Endpoint {endpoint} is unhealthy (consecutive failures: {consecutive_failures})"
                            if should_log:
                                print(f"[WARNING] Skipping unhealthy endpoint: {endpoint} (consecutive failures: {consecutive_failures})")
                            continue

            url = f"{endpoint}{path}"
            tried_endpoints.add(endpoint)

            try:
                response = requests.post(
                    url,
                    json=json_data,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    if self.enable_health_check:
                        current_time = time.time()
                        self._health_status[endpoint] = (True, current_time, 0)
                        self._last_request_time = current_time
                    return response.json()
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    print(f"[WARNING] Request failed to {endpoint} (attempt {attempt + 1}/{max_endpoint_attempts}): {last_error}")

                    if self.enable_health_check and 500 <= response.status_code < 600:
                        current_time = time.time()
                        self._last_request_time = current_time
                        _, _, failures = self._health_status.get(endpoint, (True, 0.0, 0))
                        self._health_status[endpoint] = (False, current_time, failures + 1)

            except requests.exceptions.Timeout as e:
                last_error = f"Timeout after {self.timeout}s: {str(e)}"
                print(f"[WARNING] Request timeout to {endpoint} (attempt {attempt + 1}/{max_endpoint_attempts}): {last_error}")

                if self.enable_health_check:
                    current_time = time.time()
                    self._last_request_time = current_time
                    quick_check_timeout = min(5, self.health_check_timeout)
                    try:
                        health_url = f"{endpoint}/health"
                        quick_response = requests.get(health_url, timeout=quick_check_timeout)
                        is_healthy = quick_response.status_code == 200
                    except Exception:
                        is_healthy = False

                    _, _, failures = self._health_status.get(endpoint, (True, 0.0, 0))
                    if is_healthy:
                        if failures + 1 >= 3:
                            self._health_status[endpoint] = (False, current_time, failures + 1)
                        else:
                            self._health_status[endpoint] = (True, current_time, failures + 1)
                    else:
                        self._health_status[endpoint] = (False, current_time, failures + 1)

            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                print(f"[WARNING] Connection error to {endpoint} (attempt {attempt + 1}/{max_endpoint_attempts}): {last_error}")

                if self.enable_health_check:
                    current_time = time.time()
                    self._last_request_time = current_time
                    self._health_status[endpoint] = (False, current_time, 10)

            except Exception as e:
                last_error = str(e)
                print(f"[WARNING] Request failed to {endpoint} (attempt {attempt + 1}/{max_endpoint_attempts}): {last_error}")

            if attempt < max_endpoint_attempts - 1:
                sleep_time = min(2 ** (attempt % 3), 8)
                time.sleep(sleep_time)

        error_msg = f"Request failed after {max_endpoint_attempts} attempts to different endpoints. Last error: {last_error}"

        if self.enable_health_check:
            healthy_count = sum(1 for h, _, _ in self._health_status.values() if h)
            total_count = len(self.endpoints)
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] Endpoint health: {healthy_count}/{total_count} healthy")

            unhealthy_endpoints_info = []
            for endpoint, (is_healthy, last_check, failures) in self._health_status.items():
                if not is_healthy:
                    unhealthy_endpoints_info.append((endpoint, failures, last_check))

            if len(unhealthy_endpoints_info) > 5:
                for endpoint, failures, last_check in unhealthy_endpoints_info[:5]:
                    time_ago = time.time() - last_check
                    print(f"[ERROR]   FAILED {endpoint} (failures: {failures}, last_check: {time_ago:.1f}s ago)")
                print(f"[ERROR]   ... and {len(unhealthy_endpoints_info) - 5} more unhealthy endpoints")
            else:
                for endpoint, failures, last_check in unhealthy_endpoints_info:
                    time_ago = time.time() - last_check
                    print(f"[ERROR]   FAILED {endpoint} (failures: {failures}, last_check: {time_ago:.1f}s ago)")

        raise RuntimeError(error_msg)


class ImageEditClient(ServiceClient):
    """Image-editing service client (supports BAGEL and Qwen-Image-Edit)."""

    def __init__(self, *args, **kwargs):
        """Initialize the image-editing client."""
        super().__init__(*args, **kwargs)
        self._model_types: Dict[str, str] = {}

    def _detect_model_type(self, endpoint: str) -> str:
        """
        Detect the service type (BAGEL or Qwen-Image-Edit) for an endpoint.

        Args:
            endpoint: Service endpoint URL

        Returns:
            "bagel" or "qwen_image_edit"
        """
        if endpoint in self._model_types:
            return self._model_types[endpoint]

        try:
            health_url = f"{endpoint}/health"
            response = requests.get(
                health_url,
                timeout=self.health_check_timeout,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                data = response.json()
                model_type = data.get("model_type", "").lower()
                if "qwen" in model_type:
                    detected_type = "qwen_image_edit"
                else:
                    detected_type = "bagel"
            else:
                detected_type = "bagel"
        except Exception:
            detected_type = "bagel"

        self._model_types[endpoint] = detected_type
        return detected_type

    def edit_image(
        self,
        image: Image.Image,
        edit_prompt: str,
        num_timesteps: int = 50,
        cfg_text_scale: float = None,
        cfg_img_scale: float = None,
        cfg_scale: float = None,  # Alias for cfg_text_scale
        cfg_interval: list = None,
        timestep_shift: float = 3.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        # TaylorSeer acceleration parameters (optional, BAGEL only)
        enable_taylorseer: bool = None,
        fresh_threshold: int = None,
        max_order: int = None,
        first_enhance: int = None,
        # Resolution scaling (optional, supported by both services)
        resolution_scale: float = None,
        # Qwen-specific parameters
        guidance_scale: float = 1.0,  # Qwen guidance_scale (corresponds to BAGEL cfg_text_scale)
        true_cfg_scale: float = 4.0,  # Qwen true_cfg_scale
        negative_prompt: str = None,  # Qwen negative_prompt
        steps_computation_mask: list = None,  # Qwen steps_computation_mask
        # Explicit service type override (auto-detected when None)
        model_type: Optional[str] = None,
    ) -> Image.Image:
        """
        Edit an image (supports both BAGEL and Qwen services).

        Args:
            image: Input image
            edit_prompt: Edit instruction
            num_timesteps: Denoising steps (BAGEL); maps to num_inference_steps for Qwen
            cfg_text_scale: CFG text scale (BAGEL only)
            cfg_img_scale: CFG image scale (BAGEL only)
            cfg_scale: CFG scale alias for cfg_text_scale (BAGEL only)
            cfg_interval: CFG interval (BAGEL only)
            timestep_shift: Timestep shift (BAGEL only)
            cfg_renorm_min: CFG renormalization minimum (BAGEL only)
            cfg_renorm_type: CFG renormalization type (BAGEL only)
            enable_taylorseer: Enable TaylorSeer acceleration (BAGEL only; None uses server default)
            fresh_threshold: TaylorSeer parameter (BAGEL only; None uses server default)
            max_order: TaylorSeer parameter (BAGEL only; None uses server default)
            first_enhance: TaylorSeer parameter (BAGEL only; None uses server default)
            resolution_scale: Resolution scaling factor in (0, 1.0], both services; None uses server default
            guidance_scale: Qwen guidance_scale (takes priority over cfg_text_scale/cfg_scale when provided)
            true_cfg_scale: Qwen true_cfg_scale (default 4.0)
            negative_prompt: Qwen negative_prompt (None uses server default)
            steps_computation_mask: Qwen per-step computation mask; None builds a default mask
                that fully computes the first steps and then every third step
            model_type: Explicit service type ("bagel" or "qwen_image_edit"); auto-detected when None

        Returns:
            Edited image
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        if model_type is None:
            healthy_endpoints = self._get_healthy_endpoints()
            if healthy_endpoints:
                first_endpoint = healthy_endpoints[0]
                detected_type = self._detect_model_type(first_endpoint)
            else:
                if self.endpoints:
                    detected_type = self._detect_model_type(self.endpoints[0])
                else:
                    detected_type = "bagel"
        else:
            detected_type = model_type.lower()
            if detected_type == "qwen":
                detected_type = "qwen_image_edit"

        if detected_type == "qwen_image_edit":
            json_data = {
                "image": image_b64,
                "edit_prompt": edit_prompt,
            }
            
            json_data["num_inference_steps"] = num_timesteps
            json_data["steps_computation_mask"] = steps_computation_mask if steps_computation_mask is not None else [1 if (i-1) % 3 == 0 or i<5 else 0 for i in range(num_timesteps)]
            json_data["true_cfg_scale"] = true_cfg_scale
            json_data["guidance_scale"] = guidance_scale
            if negative_prompt is not None:
                json_data["negative_prompt"] = negative_prompt

            if resolution_scale is not None:
                json_data["resolution_scale"] = resolution_scale
        else:
            json_data = {
                "image": image_b64,
                "edit_prompt": edit_prompt,
                "num_timesteps": num_timesteps,
                "timestep_shift": timestep_shift,
                "cfg_renorm_min": cfg_renorm_min,
                "cfg_renorm_type": cfg_renorm_type,
            }

            # Prefer cfg_text_scale; cfg_scale is accepted as an alias.
            if cfg_text_scale is not None:
                json_data["cfg_text_scale"] = cfg_text_scale
            elif cfg_scale is not None:
                json_data["cfg_scale"] = cfg_scale

            if cfg_img_scale is not None:
                json_data["cfg_img_scale"] = cfg_img_scale
            if cfg_interval is not None:
                json_data["cfg_interval"] = cfg_interval
            if enable_taylorseer is not None:
                json_data["enable_taylorseer"] = enable_taylorseer
            if fresh_threshold is not None:
                json_data["fresh_threshold"] = fresh_threshold
            if max_order is not None:
                json_data["max_order"] = max_order
            if first_enhance is not None:
                json_data["first_enhance"] = first_enhance
            if resolution_scale is not None:
                json_data["resolution_scale"] = resolution_scale

        result = self._post_with_retry("/edit", json_data)

        if not result.get("success"):
            raise RuntimeError(f"Image edit failed: {result.get('error')}")

        edited_image_b64 = result["image"]
        edited_image_bytes = base64.b64decode(edited_image_b64)
        edited_image = Image.open(io.BytesIO(edited_image_bytes)).convert("RGB")
    
        return edited_image


class RewardClient(ServiceClient):
    """Reward computation service client."""

    def compute_reward(
        self,
        image: Image.Image,
        prompt: str,
        reward_type: str = "clip",
        generated_qa: Optional[Dict[str, Any]] = None,
        # SAM3-specific parameters
        category: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Unified reward computation interface.

        Args:
            image: Input image
            prompt: Text prompt
            reward_type: Reward type ("self_reward", "clip", "sam3")
            generated_qa: Optional decomposed questions, format: {"yn_question_list": [...]}
            category: SAM3 only - image category
            ground_truth: SAM3 only - ground truth answer

        Returns:
            Reward result dict
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        json_data = {
            "image": image_b64,
            "prompt": prompt,
            "reward_type": reward_type,
        }
        if generated_qa is not None:
            json_data["generated_qa"] = generated_qa

        if reward_type.lower() == "sam3":
            if not category or not ground_truth:
                raise ValueError("SAM3 reward requires category and ground_truth arguments")

            json_data.update({
                "category": category,
                "ground_truth": ground_truth,
            })

            result = self._post_with_retry("/compute_sam3_reward", json_data)
        else:
            result = self._post_with_retry("/compute_reward", json_data)

        if not result.get("success"):
            error_msg = result.get('error') or result.get('error_message') or "Unknown error"
            raise RuntimeError(f"Reward computation failed: {error_msg}")

        return result


_config_file_mtime: Dict[str, float] = {}
_global_edit_client: Optional[ImageEditClient] = None
_global_reward_client: Optional[RewardClient] = None
_global_sam3_client: Optional[RewardClient] = None

def _get_config_file_path() -> Optional[str]:
    """Return the path to service_endpoints.env, or None if not found."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "config", "service_endpoints.env"),
        os.path.join(os.environ.get("CONFIG_DIR", ""), "service_endpoints.env") if os.environ.get("CONFIG_DIR") else None
    ]

    for path in possible_paths:
        if path and os.path.exists(path):
            return os.path.abspath(path)

    return None


def _reload_config_file(config_file: str) -> bool:
    """
    Reload a config file into environment variables.

    Supports standard bash env file syntax:
    - export KEY="VALUE"
    - export KEY='VALUE'
    - export KEY=VALUE
    - Comment lines starting with #

    Args:
        config_file: Path to the config file

    Returns:
        True if reloaded, False if the file has not changed
    """
    try:
        current_mtime = os.path.getmtime(config_file)
        last_mtime = _config_file_mtime.get(config_file, 0.0)

        if current_mtime <= last_mtime:
            return False

        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                if not line.startswith('export '):
                    continue

                export_part = line[7:].strip()  # Strip 'export '
                if '=' not in export_part:
                    continue

                eq_pos = export_part.find('=')
                key = export_part[:eq_pos].strip()
                value_part = export_part[eq_pos+1:].strip()

                if value_part.startswith('"') and value_part.endswith('"'):
                    value = value_part[1:-1]
                elif value_part.startswith("'") and value_part.endswith("'"):
                    value = value_part[1:-1]
                else:
                    value = value_part
                    if '#' in value:
                        comment_pos = value.find('#')
                        if comment_pos >= 0:
                            value = value[:comment_pos].strip()

                if key:
                    os.environ[key] = value

        _config_file_mtime[config_file] = current_mtime
        print(f"[INFO] Config file reloaded: {config_file}")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to reload config file: {e}")
        import traceback
        traceback.print_exc()
        return False


def _check_and_reload_config(
    edit_client: Optional[ImageEditClient],
    reward_client: Optional[RewardClient],
    sam3_client: Optional[RewardClient]
) -> bool:
    """
    Reload service endpoints when the config file changes.
    """
    config_file = _get_config_file_path()
    if not config_file:
        return False

    if not _reload_config_file(config_file):
        return False

    edit_endpoints_str = os.environ.get("EDIT_SERVER_ENDPOINTS", "")
    reward_endpoints_str = os.environ.get("REWARD_SERVER_ENDPOINTS", "")
    sam3_endpoints_str = os.environ.get("SAM3_REWARD_SERVER_ENDPOINTS", "")

    updated = False

    if edit_client and edit_endpoints_str:
        new_endpoints = [e.strip() for e in edit_endpoints_str.split(",") if e.strip()]
        if new_endpoints:
            if edit_client.update_endpoints(new_endpoints):
                updated = True

    if reward_client and reward_endpoints_str:
        new_endpoints = [e.strip() for e in reward_endpoints_str.split(",") if e.strip()]
        if new_endpoints:
            if reward_client.update_endpoints(new_endpoints):
                updated = True
    if sam3_client and sam3_endpoints_str:
        new_endpoints = [e.strip() for e in sam3_endpoints_str.split(",") if e.strip()]
        if new_endpoints:
            if sam3_client.update_endpoints(new_endpoints):
                updated = True
    return updated


def create_clients_from_env(check_config_file: bool = True) -> tuple[
    Optional[ImageEditClient],
    Optional[RewardClient],
    Optional[RewardClient]  # Third return value is the SAM3 client
]:
    """
    Create client instances from environment variables, with hot-reload support.
    Returns: (edit_client, reward_client, sam3_client)
    """
    global _global_edit_client, _global_reward_client, _global_sam3_client

    if check_config_file and (_global_edit_client or _global_reward_client or _global_sam3_client):
        _check_and_reload_config(_global_edit_client, _global_reward_client, _global_sam3_client)
        return _global_edit_client, _global_reward_client, _global_sam3_client

    edit_client: Optional[ImageEditClient] = None
    reward_client: Optional[RewardClient] = None
    sam3_client: Optional[RewardClient] = None

    if check_config_file:
        config_file = _get_config_file_path()
        if config_file:
            _reload_config_file(config_file)

    timeout = _get_env_int("API_REQUEST_TIMEOUT", 180)
    health_check_timeout = _get_env_int("API_HEALTH_CHECK_TIMEOUT", 5)
    enable_health_check = _get_env_bool("API_ENABLE_HEALTH_CHECK", True)

    edit_endpoints = os.environ.get("EDIT_SERVER_ENDPOINTS", "")
    if edit_endpoints:
        endpoints = [e.strip() for e in edit_endpoints.split(",") if e.strip()]
        if endpoints:
            edit_client = ImageEditClient(
                endpoints,
                timeout=timeout,
                health_check_timeout=health_check_timeout,
                enable_health_check=enable_health_check
            )
            if not _global_edit_client:
                health_status = "enabled" if enable_health_check else "disabled"
                print(f"[OK] Image Edit Client created, endpoints: {len(endpoints)}, health check: {health_status}")

    reward_endpoints = os.environ.get("REWARD_SERVER_ENDPOINTS", "")
    if reward_endpoints:
        endpoints = [e.strip() for e in reward_endpoints.split(",") if e.strip()]
        if endpoints:
            reward_client = RewardClient(
                endpoints,
                timeout=timeout,
                health_check_timeout=health_check_timeout,
                enable_health_check=enable_health_check
            )
            if not _global_reward_client:
                health_status = "enabled" if enable_health_check else "disabled"
                print(f"[OK] Reward Client created, endpoints: {len(endpoints)}, health check: {health_status}")

    sam3_endpoints = os.environ.get("SAM3_REWARD_SERVER_ENDPOINTS", "")
    if sam3_endpoints:
        endpoints = [e.strip() for e in sam3_endpoints.split(",") if e.strip()]
        if endpoints:
            sam3_client = RewardClient(
                endpoints,
                timeout=timeout,
                health_check_timeout=health_check_timeout,
                enable_health_check=enable_health_check
            )
            if not _global_sam3_client:
                health_status = "enabled" if enable_health_check else "disabled"
                print(f"[OK] SAM3 Reward Client created, endpoints: {len(endpoints)}, health check: {health_status}")

    if check_config_file:
        _global_edit_client = edit_client
        _global_reward_client = reward_client
        _global_sam3_client = sam3_client

    return edit_client, reward_client, sam3_client


# Manual smoke test.
if __name__ == "__main__":
    import os
    from PIL import Image

    edit_client, reward_client, sam3_client = create_clients_from_env()

    # The URLs below are PLACEHOLDERS. Replace them with real endpoints from your
    # env config (EDIT_SERVER_ENDPOINTS / REWARD_SERVER_ENDPOINTS) or a config file.
    if not edit_client:
        edit_client = ImageEditClient([
            "http://EDIT_SERVER_HOST:5001",
            "http://EDIT_SERVER_HOST:5002",
            "http://EDIT_SERVER_HOST:5003",
            "http://EDIT_SERVER_HOST:5004",
        ])

    if not reward_client:
        reward_client = RewardClient([
            "http://REWARD_SERVER_HOST:6001",
            "http://REWARD_SERVER_HOST:6002",
        ])

    if edit_client:
        print("\nTesting image edit service...")
        test_image = Image.new("RGB", (512, 512), color=(255, 0, 0))
        try:
            edited = edit_client.edit_image(
                test_image,
                "Make it blue",
                num_timesteps=20,
            )
            print(f"[OK] Edit succeeded, image size: {edited.size}")
        except Exception as e:
            print(f"[ERROR] Edit failed: {e}")

    if reward_client:
        print("\nTesting reward computation service...")
        test_image = Image.new("RGB", (512, 512), color=(0, 255, 0))
        try:
            result = reward_client.compute_reward(
                test_image,
                "A green square",
                reward_type="clip"
            )
            print(f"[OK] Reward computed, score: {result['score']:.4f}")
        except Exception as e:
            print(f"[ERROR] Reward failed: {e}")
