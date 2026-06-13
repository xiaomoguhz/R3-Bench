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

"""Prompt templates for reflection and evaluation.

These templates are the canonical benchmark inputs. Model behaviour, and
therefore the Reflective Verdict Score (S_ref) and Rectification Score
(S_rect), depend on them verbatim. **Do not modify** without bumping a new
dataset version.

Reflection prompts come in two styles, controlled by the ``--prompt`` flag of
``step1_reflection.py``:

- ``default``: the benchmark's standard evaluation prompt for stock models.
  Uses ``<think>...</think>`` reasoning tags and a minimal edit-instruction
  format.
- ``refiner``: the format the released R3-Refiner reflection model was trained
  with, analogous to a model's own chat template. Evaluate that checkpoint
  with ``--prompt refiner``.
"""


REFLECTION_SYSTEM_PROMPT = (
    " You should first think about the reasoning process in the mind and then "
    "provide the user with the answer. The reasoning process is enclosed within "
    "<think> </think> tags, i.e. <think> reasoning process here </think> answer here"
)

# default: the benchmark's standard prompt, minimal edit instruction format.
REFLECTION_USER_PROMPT = """This image was generated from the prompt: {origin_prompt}.
Please carefully analyze the image and determine whether all the objects, attributes, and spatial relationships mentioned in the prompt are correctly represented in the image.

If the image accurately reflects the prompt, please answer 'true'; otherwise, answer 'false'.

When the answer is false, you must:
1. Identify the main error and describe it briefly in "explanation".
2. In "edit_prompt", provide a **concrete image editing instruction** to fix the error.
- The instruction must specify the exact action (e.g., add / remove / replace / move).
- The instruction must specify the location or reference point (e.g., "delete the bottle in the bottom-right corner", "add a dog next to the left pillar").
- Do not give vague instructions such as "add more bottles" or "ensure the count is correct". Be precise and actionable.

Respond strictly in the following JSON format:

{{
    "answer": true/false,
    "explanation": "If the answer is false, briefly summarize the main error.",
    "edit_prompt": "If the answer is false, provide a concrete and location-specific editing instruction."
}}
"""

# refiner: the released R3-Refiner model's training format, with explicit
# action/location requirements and worked examples in the edit instruction.
REFLECTION_USER_PROMPT_REFINER = """This image was generated from the prompt: {origin_prompt}. Please carefully analyze the image and determine whether all the objects, attributes, count, and spatial relationships mentioned in the prompt are correctly represented in the image.

If the image accurately reflects the prompt, please answer 'true'; otherwise, answer 'false'.

When the answer is false, you must:
1. Identify the main error and describe it briefly in "explanation".
   - Clearly state what the prompt requires vs. what is actually shown in the image.
   - If there are multiple discrepancies (e.g., object missing, wrong color, wrong position, wrong count), you should mention all of the important ones in a concise way.
2. In "edit_prompt", provide a **direct and specific image editing instruction** to fix the error.
   - Choose the most appropriate action based on the actual error: add / remove / replace / move / change color / change shape / change texture / modify attribute / adjust count / resize / swap positions
   - The instruction must specify the exact action and the location or reference point when relevant (e.g., "on the left side of the table", "above the cat", "next to the toaster", "in the background").
   - Do not give vague instructions such as "fix the image", "make it match the prompt", or "ensure the count is correct". Be precise and actionable.
   - **Important**: The text prompt is the Ground Truth. Your goal is to modify the IMAGE to match the text. NEVER suggest changing the text description or caption.

Examples of good edit_prompt instructions:
- "Replace the white candle on the right side of the image with a white candle holder."
- "Change the fork's color from silver to gold in the image."
- "Remove the pink toy vehicle attached to the airplane's nose and add a pink toaster next to the airplane."
- "Change the orange's color from orange to green."
- "Move the cat to the left side of the pizza so that the cat is clearly positioned to the left of the pizza."
- "Add one more donut to the plate on the left side so that there are exactly three donuts."
- "Swap the positions of the giraffe and the traffic light so that the giraffe is clearly to the right of the traffic light, keeping both fully visible."

Respond strictly in the following JSON format:

{{
    "answer": true/false,
    "explanation": "A brief, specific description of the main error (if answer is false).",
    "edit_prompt": "A concrete, location-specific editing instruction to fix the error (if answer is false)."
}}
"""

REFLECTION_EVAL_SYSTEM_PROMPT = """You are an expert evaluator for image reflection tasks. Your task is to compare two explanations for why an image fails to match a prompt and determine if they are semantically equivalent.

You are given:
- The original image prompt.
- A Model Explanation and a GT (Ground Truth) Explanation.

Typical error dimensions include:

- Color: wrong or missing colors.
- Object: wrong, missing, or extra objects.
- Numeracy: wrong object counts or quantities.
- Spatial: wrong positions, relative locations, or spatial relations.
- Shape: wrong shapes or geometric properties.
- Texture: wrong material or surface appearance.
- Complex: complex combinations of multiple basic errors (for example, several objects and relations are all wrong at the same time, or multiple dimensions are intertwined).
- Non: more subjective or high-level mismatches that are not purely low-level visual attributes, such as incorrect actions or activities, scene type, atmosphere, style, or other semantic aspects that do not clearly fall into the categories above.

These are general categories that describe how an image can fail to match a prompt
(for example: wrong color, wrong object, wrong count, wrong spatial relation, wrong shape or texture, wrong action or atmosphere, etc.).

Definitions:
- A "core error" is the main reason why the image does NOT satisfy the prompt
  (for example: wrong object count, wrong object type, wrong attribute, wrong spatial relation, missing or extra object, wrong action, wrong style, etc.).
- There can be multiple low-level details, but usually only a small number of core errors.

The model's explanation is considered correct if it identifies the SAME CORE ERROR as the GT (Ground Truth) explanation, even if:
- The wording is different
- Additional context or details are mentioned (for example, mentioning other objects that are present)
- The phrasing or style differs

IMPORTANT:
- Use the original prompt to understand what the image is supposed to contain or look like.
- Focus on whether both explanations point to the SAME fundamental problem in how the image fails to match the prompt,
  considering the typical dimensions listed above (color, object, numeracy, spatial relation, shape, texture, complex combinations, and non-visual-high-level aspects such as action or atmosphere).
- Do NOT reject explanations just because one includes extra information or uses different words to describe the same error.
- However, if the Model Explanation introduces a NEW, SEPARATE core error that is not implied by the GT Explanation,
  or criticizes something that is actually correct according to the original prompt,
  then they are NOT semantically equivalent.

You should respond in JSON format:
{
    "is_correct": true/false,
    "reasoning": "A brief explanation of why the explanations are or are not semantically equivalent."
}
"""

REFLECTION_EVAL_USER_PROMPT = """Original Prompt:
"{original_prompt}"

Compare the following two explanations:

Model Explanation:
"{model_explanation}"

GT Explanation:
"{gt_explanation}"

Are these two explanations semantically equivalent? Respond in JSON format as specified.
"""

QA_EVAL_PROMPT_TEMPLATE = """
You are tasked with conducting a careful examination of the image. Based on the content of the image, please answer the following yes or no questions:

Questions:
{questions}

Note that:
1. Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
2. The order of answers must correspond exactly to the order of the questions.
3. Each question must have only one answer.
4. Directly return the answers to each question, without any additional content.
5. Each answer must be on its own line!
6. Make sure the number of output answers equal to the number of questions!
"""


def format_reflection_prompt(origin_prompt: str, with_system: bool = True, style: str = "default") -> str:
    """Format the reflection task prompt for a given prompt style (default/refiner)."""
    if style == "refiner":
        user_prompt = REFLECTION_USER_PROMPT_REFINER.format(origin_prompt=origin_prompt)
    else:
        user_prompt = REFLECTION_USER_PROMPT.format(origin_prompt=origin_prompt)
    system_prompt = REFLECTION_SYSTEM_PROMPT

    if with_system:
        return user_prompt + system_prompt
    return user_prompt


def format_reflection_eval_prompt(
    model_explanation: str,
    gt_explanation: str,
    original_prompt: str = "",
) -> str:
    """Format the reflection evaluation user prompt."""
    model_explanation = model_explanation.replace('"', '\\"')
    gt_explanation = gt_explanation.replace('"', '\\"')
    original_prompt = original_prompt.replace('"', '\\"')

    return REFLECTION_EVAL_USER_PROMPT.format(
        model_explanation=model_explanation,
        gt_explanation=gt_explanation,
        original_prompt=original_prompt,
    )


def format_qa_prompt(questions: list) -> str:
    """Format the QA evaluation prompt from a list of yes/no questions."""
    formatted_questions = "\n".join(q.strip() for q in questions)
    return QA_EVAL_PROMPT_TEMPLATE.format(questions=formatted_questions)
