"""Utility helpers for text-to-image."""

from typing import Dict


def validate_prompt(prompt: str) -> bool:
    return bool(prompt and prompt.strip())


def validate_generation_params(params: Dict) -> bool:
    if not isinstance(params, dict):
        return False
    steps = params.get("steps", params.get("num_inference_steps", 1))
    guidance = params.get("guidance_scale", 1.0)
    width = params.get("width", 1)
    height = params.get("height", 1)
    return steps > 0 and guidance > 0 and width > 0 and height > 0
