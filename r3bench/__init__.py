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

"""R3-Bench — Reason-Reflect-Rectify benchmark for Reflective Visual Generation.

Paper: "Benchmarking and Evolving Reason-Reflect-Rectify for Reflective Visual
Generation".

The evaluation is a four-step pipeline; each step is a separate CLI entry
point:

1. :mod:`r3bench.step1_reflection` — model produces a structured response
   ``⟨v, e, a⟩``: binary verdict, discrepancy explanation, and rectification
   action.
2. :mod:`r3bench.step2_edit` — an external image editor Φ applies the
   rectification action ``a`` to obtain the rectified image.
3. :mod:`r3bench.step3_eval_reflection` — aggregates the **Reflective
   Verdict Score** ``S_ref`` (verdict-and-explanation correctness) using an
   LLM judge for semantic-equivalence checks.
4. :mod:`r3bench.step4_eval_edit` — computes the **Rectification Score**
   ``S_rect`` (normalised VQA-alignment gain on rectified images).

The reflection / editor / evaluator backends are pluggable; see
``r3bench.models``, ``r3bench.editors`` and ``r3bench.evaluators`` for the
extension protocols.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
