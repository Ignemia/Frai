"""Utility functions for img2img tests."""

from typing import Dict


def validate_image_input(path: str) -> bool:
    if not path or not isinstance(path, str):
        return False
    return path.lower().endswith((".jpg", ".jpeg", ".png"))


def validate_transformation_params(params: Dict) -> bool:
    if not isinstance(params, dict):
        return False
    strength = params.get("strength", 0.0)
    guidance = params.get("guidance_scale", 1.0)
    steps = params.get("num_inference_steps", 1)
    return 0.0 <= strength <= 1.0 and guidance > 0 and steps > 0
