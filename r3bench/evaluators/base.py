# Copyright 2026 R3-Bench Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract evaluator base classes shared by the GPT and Qwen backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

from r3bench.utils import (
    get_logger,
    encode_image,
    parse_json_response,
    extract_yes_no,
    OutputFormatError,
)
from r3bench.prompts import (
    REFLECTION_EVAL_SYSTEM_PROMPT,
    format_reflection_eval_prompt,
    format_qa_prompt,
)

logger = get_logger("evaluators.base")


class BaseEvaluator(ABC):
    """Common retry-and-API plumbing for every evaluator backend.

    Subclasses implement :meth:`call_api`; everything else (retry/backoff,
    response parsing, metric computation) is shared.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        max_retries: int = 5,
        request_timeout: float = 300.0,
    ):
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    @abstractmethod
    def call_api(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Send one request to the underlying judge and return the raw response text."""

    def retry_with_backoff(
        self,
        func,
        *args,
        max_retries: Optional[int] = None,
        **kwargs,
    ):
        retries = max_retries or self.max_retries
        last_error = None

        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e)

                is_rate_limit = any(
                    s in error_str.lower()
                    for s in ["429", "rate limit", "qpm limit"]
                )

                if attempt < retries - 1:
                    if is_rate_limit:
                        wait_time = min(60, 10 * (attempt + 1))
                        logger.warning(
                            f"Rate limit hit, waiting {wait_time}s "
                            f"(attempt {attempt + 1}/{retries})"
                        )
                    else:
                        wait_time = 2 * (attempt + 1)
                        logger.warning(
                            f"API call failed: {e}, retrying in {wait_time}s "
                            f"(attempt {attempt + 1}/{retries})"
                        )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached: {e}")

        raise last_error


class ReflectionEvaluator(BaseEvaluator):
    """Score the reflection stage (step 3).

    The paper's headline metric for this stage is the **Reflective Verdict
    Score** :math:`S_{ref}` — the fraction of samples on which the model
    produces a correct verdict *and*, for misaligned samples, a semantically
    equivalent explanation. Per sample ``i`` with GT verdict ``v_i`` and
    explanation ``e_i``::

        s_i = 1{v̂_i = True}                         if v_i = True (aligned)
        s_i = 1{v̂_i = False} · J(e_i, ê_i)          if v_i = False (misaligned)

        S_ref = (1 / N) · Σ s_i

    Here ``J(·, ·)`` is the LLM-judge equivalence test computed by
    :meth:`evaluate`; the indicator factors are aggregated by
    :mod:`r3bench.step3_eval_reflection`. The step also reports an auxiliary
    **Verdict Accuracy** (raw ``v̂_i = v_i`` rate) for diagnostic comparison.

    Records where ``v̂_i = True`` (model claims the image is fine) need no
    explanation, so :meth:`process_record` skips the LLM-judge call for them.
    """

    def evaluate(
        self,
        model_explanation: str,
        gt_explanation: str,
        original_prompt: str = "",
    ) -> Dict[str, Any]:
        """Judge whether the model's explanation is semantically equivalent to GT.

        Returns:
            ``{"is_correct": bool, "reason": str, ...}`` parsed from the
            judge's JSON response.
        """
        prompt = format_reflection_eval_prompt(
            model_explanation,
            gt_explanation,
            original_prompt=original_prompt,
        )
        full_prompt = REFLECTION_EVAL_SYSTEM_PROMPT + "\n\n" + prompt

        response = self.retry_with_backoff(
            self.call_api,
            prompt=full_prompt,
            image_path=None,
        )

        return parse_json_response(response, key_field="is_correct")

    def process_record(
        self,
        record: Dict[str, Any],
        benchmark_dir: Path,
        overwrite: bool = False,
        completed_indices: Optional[set] = None,
        result_field: str = "reflection_eval",
    ) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        record_idx = record.get("index")
        if record_idx is None:
            return -1, None, "Record missing 'index' field"

        if not overwrite and completed_indices and record_idx in completed_indices:
            return record_idx, record.get(result_field), None

        model_reflection = record.get("model_reflection")
        gt_answer = record.get("answer")
        gt_explanation = record.get("explanation", "")
        original_prompt = record.get("original_prompt", record.get("prompt", ""))

        if not model_reflection or gt_answer is None:
            return record_idx, None, f"[index {record_idx}] Missing reflection data"

        model_answer = model_reflection.get("answer")

        # Only evaluate when both model and GT agree the image is wrong.
        if not (model_answer is False and gt_answer is False):
            return record_idx, None, None

        try:
            eval_result = self.evaluate(
                model_explanation=model_reflection.get("explanation", ""),
                gt_explanation=gt_explanation,
                original_prompt=original_prompt,
            )
            return record_idx, eval_result, None
        except Exception as e:
            return record_idx, None, f"[index {record_idx}] Evaluation failed: {e}"


class ImageEditEvaluator(BaseEvaluator):
    """Score the editing stage (step 4) via the **Rectification Score** S_rect.

    For each *misaligned* record the judge VLM answers the same list of
    yes/no questions against the original ``bad_image`` and the rectified
    image, yielding two VQA-based alignment scores ``V(·, Q) ∈ [0, 1]``::

        V_bad  = V(I^(t),   Q) = correct / total on the original flawed image
        V_edit = V(I^(t+1), Q) = correct / total on the edited image

    The per-sample Rectification Score is then::

        s_rect_i = (V_edit - V_bad) / (1 - V_bad)        if V_bad < 1
        s_rect_i = 0                                      if V_bad == 1

    Aggregated over the ``N_neg`` misaligned records (paper Eq. 3)::

        S_rect = (1 / N_neg) · Σ_{i: v_i = False} s_rect_i

    Properties:
        * Range ``[-1, 1]``. ``> 0`` means the edit improved VQA-alignment;
          ``< 0`` means it regressed.
        * The ``1 - V_bad`` normaliser scores the edit against the room left
          to improve — easy starts no longer dominate.
        * ``V_bad == 1`` (no room to improve) is pinned to ``0`` to avoid
          division by zero.

    JSONL records emitted by :mod:`r3bench.step4_eval_edit` store the
    per-sample value under the ``s_rect`` key.
    """

    def evaluate_image(
        self,
        image_path: str,
        questions: List[str],
        gt_answers: List[str],
    ) -> Dict[str, Any]:
        """Run the QA judge on a single image and return per-question accuracy."""
        if not questions:
            return {
                "correct": 0,
                "total": 0,
                "accuracy": 1.0,
                "model_pred": [],
                "gt_answers": [],
                "model_output": "",
            }

        prompt = "You are a professional image critic.\n" + format_qa_prompt(questions)

        response = self.retry_with_backoff(
            self.call_api,
            prompt=prompt,
            image_path=image_path,
        )

        try:
            model_pred = extract_yes_no(response, len(questions))
        except OutputFormatError as e:
            logger.warning(f"Format error: {e}")
            raise

        correct_count = sum(
            1 for pred, gt in zip(model_pred, gt_answers)
            if pred.lower() == gt.lower()
        )

        return {
            "correct": correct_count,
            "total": len(questions),
            "accuracy": correct_count / len(questions),
            "model_pred": model_pred,
            "gt_answers": gt_answers,
            "model_output": response,
        }

    def process_record(
        self,
        record: Dict[str, Any],
        benchmark_dir: Path,
        edited_images_dir: Path,
        overwrite: bool = False,
        completed_indices: Optional[set] = None,
    ) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        record_idx = record.get("idx")
        if record_idx is None:
            return -1, None, f"Record missing 'idx' field: {record}"

        if not overwrite and completed_indices and record_idx in completed_indices:
            return record_idx, record.get("image_eval"), None

        qa_data = record.get("generated_qa")
        if not qa_data or "yn_question_list" not in qa_data or "yn_answer_list" not in qa_data:
            return record_idx, None, f"[idx {record_idx}] Missing generated_qa field"

        questions = qa_data["yn_question_list"]
        gt_answers = qa_data["yn_answer_list"]
        bad_image_rel = record.get("bad_image")

        if not bad_image_rel:
            return record_idx, None, f"[idx {record_idx}] Missing 'bad_image'"

        bad_image_path = benchmark_dir / bad_image_rel
        bad_image_path_obj = Path(bad_image_rel)
        edited_filename = f"{bad_image_path_obj.stem}_edited{bad_image_path_obj.suffix}"
        edited_image_path = edited_images_dir / edited_filename

        if not bad_image_path.exists():
            return record_idx, None, f"[idx {record_idx}] Original image not found: {bad_image_path}"
        if not edited_image_path.exists():
            return record_idx, None, f"[idx {record_idx}] Edited image not found: {edited_image_path}"

        try:
            bad_eval = self.evaluate_image(str(bad_image_path), questions, gt_answers)
            edited_eval = self.evaluate_image(str(edited_image_path), questions, gt_answers)
        except Exception as e:
            return record_idx, None, f"[idx {record_idx}] Evaluation failed: {e}"

        acc_bad = bad_eval["accuracy"]
        acc_edited = edited_eval["accuracy"]

        if acc_bad == 1.0:
            s_rect = 0.0
        else:
            s_rect = (acc_edited - acc_bad) / (1 - acc_bad)

        return record_idx, {
            "bad_image_eval": bad_eval,
            "edited_image_eval": edited_eval,
            "s_rect": s_rect,
        }, None
