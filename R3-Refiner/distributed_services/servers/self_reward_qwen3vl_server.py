#!/usr/bin/env python3
"""
R3-Refiner self-reward service with a Qwen3-VL backend.
"""

import argparse
import io
import base64
import os
import sys
import json
import re
import logging

import torch
from flask import Flask, request, jsonify
from PIL import Image

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    QWEN3VL_AVAILABLE = True
except ImportError:
    QWEN3VL_AVAILABLE = False
    print("Warning: Qwen3-VL dependencies not available. Please install transformers")

app = Flask(__name__)

QWEN3VL_MODEL = None
QWEN3VL_PROCESSOR = None
DEVICE = None

def normalize_reward_type(reward_type: str | None) -> str:
    return (reward_type or "self_reward").strip().lower().replace("-", "_")


def load_qwen3vl_model(model_path: str, device: torch.device):
    """Load the Qwen3-VL model."""
    global QWEN3VL_MODEL, QWEN3VL_PROCESSOR
    
    if not QWEN3VL_AVAILABLE:
        raise RuntimeError("Qwen3-VL dependencies not available. Please install transformers")
    
    print(f"Loading Qwen3-VL model from {model_path}...")
    
    try:
        # Let transformers handle Qwen3-VL device placement.
        QWEN3VL_MODEL = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        QWEN3VL_PROCESSOR = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print(f"Qwen3-VL model loaded successfully")
        print(f"   Model path: {model_path}")
        print(f"   Device map: auto")
        print(f"   Model dtype: {QWEN3VL_MODEL.dtype if hasattr(QWEN3VL_MODEL, 'dtype') else 'N/A'}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load Qwen3-VL model: {e}")


def build_verification_prompt(original_prompt: str) -> str:
    """Build the verification prompt for a given original prompt."""
    question = f"""This image was generated from the prompt: {original_prompt}. 
    Please carefully analyze the image and determine whether all the objects, attributes, count, and spatial relationships mentioned in the prompt are correctly represented in the image. 

    If the image accurately reflects the prompt, please answer 'true'; otherwise, answer 'false'.  

    Respond strictly in the following JSON format: 

    {{
        "answer": true/false,
        "explanation": "If the answer is false, briefly summarize the main error.",
    }}

 You should first think about the reasoning process in your mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"""

    return question

def build_question_prompt(question: str) -> str:
    """Build the verification prompt for a single yes/no question."""
    prompt = f"""Please carefully analyze the image and answer the following question: {question}

    Respond strictly in the following JSON format:

    {{
        "answer": true/false,
        "explanation": "Brief explanation of your answer.",
    }}
    """
    return prompt


def extract_answer_from_response(response_text: str) -> tuple[bool, str]:
    """
    Extract the answer field from a model response.
    
    Returns:
        (answer: bool, explanation: str)
    """
    json_part = response_text.strip()
    for tag in ("</think>", "</thinking>"):
        if tag in json_part:
            json_part = json_part.split(tag, 1)[-1].strip()
            break
    
    try:
        response_json = json.loads(json_part)
        answer = response_json.get("answer", False)
        explanation = response_json.get("explanation", "")
        return bool(answer), explanation
    except json.JSONDecodeError as e:
        if not json_part.rstrip().endswith("}"):
            json_part_fixed = json_part.rstrip()
            if json_part_fixed.count('"') % 2 == 1:  # odd number of quotes means one is unmatched
                json_part_fixed += '"}'
            else:
                json_part_fixed += '}'
            
            try:
                response_json = json.loads(json_part_fixed)
                answer = response_json.get("answer", False)
                explanation = response_json.get("explanation", "")
                print(f"[WARNING] JSON was truncated, fix succeeded | original_error={str(e)[:50]}", flush=True)
                return bool(answer), explanation
            except json.JSONDecodeError:
                pass  # repair failed; fall through to regex
        
        last_brace = json_part.rfind("}")
        if last_brace > 0:
            json_part_truncated = json_part[:last_brace + 1]
            try:
                response_json = json.loads(json_part_truncated)
                answer = response_json.get("answer", False)
                explanation = response_json.get("explanation", "")
                print(f"[WARNING] Used truncated but valid JSON portion", flush=True)
                return bool(answer), explanation
            except json.JSONDecodeError:
                pass
        
        answer_match = re.search(r'"answer"\s*:\s*(true|false)', response_text, re.IGNORECASE)
        if answer_match:
            answer_str = answer_match.group(1).lower()
            answer = answer_str == "true"
            
            explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', response_text, re.IGNORECASE)
            explanation = explanation_match.group(1) if explanation_match else ""
            
            print(f"[WARNING] JSON parse failed, used regex fallback | error={str(e)[:80]}", flush=True)
            return answer, explanation
        
        print(f"[ERROR] Failed to parse response | error={str(e)}", flush=True)
        return False, "Failed to parse response"


def compute_qwen3vl_score(image: Image.Image, prompt: str) -> float:
    """
    Compute the reward score using the Qwen3-VL model.

    Args:
        image: PIL Image object
        prompt: original text prompt

    Returns:
        Reward score: 1.0 if answer is true, 0.0 otherwise.
    """
    global QWEN3VL_MODEL, QWEN3VL_PROCESSOR
    
    if QWEN3VL_MODEL is None or QWEN3VL_PROCESSOR is None:
        raise RuntimeError("Qwen3-VL model not loaded")
    
    try:
        question = build_verification_prompt(prompt)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        inputs = QWEN3VL_PROCESSOR.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(QWEN3VL_MODEL.device)
        
        with torch.no_grad():
            generated_ids = QWEN3VL_MODEL.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            gen_length = len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0
            if gen_length >= 2045:
                print(f"[WARNING] Generation may be truncated | length={gen_length}/2048", flush=True)
            
            output_text = QWEN3VL_PROCESSOR.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        response_text = output_text[0] if output_text else ""
        answer, explanation = extract_answer_from_response(response_text)

        score = 1.0 if answer else 0.0
        
        return score
        
    except Exception as e:
        print(f"Error computing Qwen3-VL score: {e}", flush=True)
        raise


def compute_qwen3vl_score_with_qa(image: Image.Image, yn_question_list: list[str]) -> float:
    """
    Compute the reward score using multiple yes/no questions via the Qwen3-VL model.

    Args:
        image: PIL Image object
        yn_question_list: list of yes/no questions, e.g. ["Is there a cup in the image?", "Is the cup red in color?"]

    Returns:
        Reward score: correct_count / total_questions, in range [0.0, 1.0].
    """
    global QWEN3VL_MODEL, QWEN3VL_PROCESSOR
    
    if QWEN3VL_MODEL is None or QWEN3VL_PROCESSOR is None:
        raise RuntimeError("Qwen3-VL model not loaded")
    
    if not yn_question_list:
        return 0.0
    
    correct_count = 0
    total_questions = len(yn_question_list)
    
    try:
        for question in yn_question_list:
            question_prompt = build_question_prompt(question)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": question_prompt},
                    ],
                }
            ]

            inputs = QWEN3VL_PROCESSOR.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            inputs = inputs.to(QWEN3VL_MODEL.device)
            
            with torch.no_grad():
                generated_ids = QWEN3VL_MODEL.generate(**inputs, max_new_tokens=2048)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                gen_length = len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0
                if gen_length >= 2045:
                    print(f"[WARNING] QA generation may be truncated | length={gen_length}/2048", flush=True)
                
                output_text = QWEN3VL_PROCESSOR.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
            response_text = output_text[0] if output_text else ""
            answer, explanation = extract_answer_from_response(response_text)

            if answer:
                correct_count += 1

        score = correct_count / total_questions if total_questions > 0 else 0.0
        
        return score
        
    except Exception as e:
        print(f"Error computing Qwen3-VL score with QA: {e}", flush=True)
        raise


@app.route("/health", methods=["GET"])
def health_check():
    """Health-check endpoint."""
    return jsonify({
        "status": "healthy",
        "reward_type": "self_reward",
        "self_reward_model_type": "qwen3vl",
        "model_loaded": QWEN3VL_MODEL is not None,
        "processor_loaded": QWEN3VL_PROCESSOR is not None,
    })


@app.route("/compute_reward", methods=["POST"])
def compute_reward_endpoint():
    """
    Compute reward API endpoint.

    Request body (JSON):
    {
        "image": "<base64-encoded image>",
        "prompt": "<text prompt>",
        "reward_type": "self_reward",
        "generated_qa": {
            "yn_question_list": ["Is there a cup in the image?", "Is the cup red in color?"]
        }
    }

    Response (JSON):
    {
        "success": true,
        "score": 1.0,
        "raw_score": 1.0,
        "reward_type": "self_reward",
        "error": null
    }
    """
    print(f"[REQUEST] POST /compute_reward | from: {request.remote_addr}", flush=True)
    try:
        data = request.get_json()
        
        image_b64 = data.get("image")
        prompt = data.get("prompt")
        reward_type = normalize_reward_type(data.get("reward_type", "self_reward"))
        generated_qa = data.get("generated_qa")
        
        if generated_qa is not None and not isinstance(generated_qa, dict):
            if isinstance(generated_qa, str):
                try:
                    generated_qa = json.loads(generated_qa)
                except json.JSONDecodeError:
                    generated_qa = None
            else:
                generated_qa = None
        
        has_qa = generated_qa is not None and isinstance(generated_qa, dict) and "yn_question_list" in generated_qa
        qa_count = len(generated_qa.get("yn_question_list", [])) if has_qa else 0
        prompt_preview = prompt[:50] + "..." if prompt and len(prompt) > 50 else prompt
        print(f"[REQUEST] reward_type={reward_type}, prompt_preview={prompt_preview}, has_qa={has_qa}, qa_count={qa_count}", flush=True)
        
        if not image_b64:
            return jsonify({
                "success": False,
                "score": 0.0,
                "error": "Missing required field: image"
            }), 400
        
        if not has_qa and not prompt:
            return jsonify({
                "success": False,
                "score": 0.0,
                "error": "Missing required field: prompt (or generated_qa)"
            }), 400
        
        if reward_type != "self_reward":
            return jsonify({
                "success": False,
                "score": 0.0,
                "error": f"Unsupported reward type: {reward_type}. Only 'self_reward' is supported."
            }), 400
        
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if has_qa:
            yn_question_list = generated_qa.get("yn_question_list", [])
            if not isinstance(yn_question_list, list):
                return jsonify({
                    "success": False,
                    "score": 0.0,
                    "error": f"generated_qa.yn_question_list must be a list, got {type(yn_question_list)}"
                }), 400
            if not yn_question_list:
                return jsonify({
                    "success": False,
                    "score": 0.0,
                    "error": "generated_qa.yn_question_list is empty"
                }), 400
            yn_question_list = [str(q) for q in yn_question_list if q]
            if not yn_question_list:
                return jsonify({
                    "success": False,
                    "score": 0.0,
                    "error": "generated_qa.yn_question_list contains no valid questions"
                }), 400
            score = compute_qwen3vl_score_with_qa(image, yn_question_list)
        else:
            score = compute_qwen3vl_score(image, prompt)
        
        print(f"[RESPONSE] Success | score={score:.4f}, mode={'qa' if has_qa else 'prompt'}", flush=True)
        
        return jsonify({
            "success": True,
            "score": score,
            "raw_score": score,
            "reward_type": "self_reward",
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
    parser = argparse.ArgumentParser(description="Qwen3-VL reward server")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen3-VL model")
    parser.add_argument("--port", type=int, default=6001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    
    load_qwen3vl_model(args.model_path, device)

    print(f"Starting Qwen3-VL reward server on {args.host}:{args.port}")
    print(f"   Model path: {args.model_path}")
    print(f"   Device: {device} (device_map='auto')")
    
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
