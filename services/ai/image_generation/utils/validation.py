"""
Prompt validation utilities.

Provides validation functions for user prompts with safety checks
and improvement suggestions.
"""
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_prompt(prompt: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate the input prompt and provide suggestions if needed.
    
    Args:
        prompt: The text prompt to validate
        
    Returns:
        Tuple containing:
            - Boolean indicating if prompt is valid
            - Validation message
            - Optional suggested improved prompt
    """
    if not prompt or not prompt.strip():
        return False, "Prompt cannot be empty", None
    
    prompt = prompt.strip()
    
    # Check length
    if len(prompt) > 1000:
        return False, "Prompt is too long (max 1000 characters)", prompt[:1000]
    
    # Check for potentially problematic content
    problematic_words = [
        "nude", "naked", "nsfw", "explicit", "sexual", "porn", "xxx",
        "violence", "blood", "gore", "death", "kill", "murder",
        "hate", "racist", "nazi", "terrorist"
    ]
    
    prompt_lower = prompt.lower()
    found_problematic = [word for word in problematic_words if word in prompt_lower]
    
    if found_problematic:
        return False, f"Prompt contains potentially inappropriate content: {', '.join(found_problematic)}", None
    
    # Basic suggestions for improvement
    suggestions = []
    if len(prompt) < 10:
        suggestions.append("Consider adding more descriptive details")
    
    if "," not in prompt and len(prompt.split()) > 3:
        suggestions.append("Consider using commas to separate concepts")
    
    # Check if it's a good artistic prompt
    artistic_keywords = ["style", "art", "painting", "digital", "illustration", "concept", "detailed", "beautiful"]
    has_artistic_terms = any(keyword in prompt_lower for keyword in artistic_keywords)
    
    if not has_artistic_terms:
        suggestions.append("Consider adding artistic style descriptors (e.g., 'digital art', 'detailed illustration')")
    
    suggested_prompt = None
    if suggestions:
        suggested_prompt = prompt
        if not has_artistic_terms:
            suggested_prompt = f"{prompt}, digital art, detailed"
    
    message = "Prompt is valid"
    if suggestions:
        message += f". Suggestions: {'; '.join(suggestions)}"
    
    return True, message, suggested_prompt


def validate_generation_parameters(
    height: int, 
    width: int, 
    steps: int, 
    guidance_scale: float
) -> Tuple[bool, str]:
    """
    Validate generation parameters.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        guidance_scale: Guidance scale value
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    errors = []
    
    # Validate dimensions
    if height <= 0 or width <= 0:
        errors.append("Height and width must be positive")
    
    if height > 2048 or width > 2048:
        errors.append("Height and width should not exceed 2048 pixels")
    
    if height % 8 != 0 or width % 8 != 0:
        errors.append("Height and width should be multiples of 8")
    
    # Validate steps
    if steps <= 0:
        errors.append("Number of steps must be positive")
    
    if steps > 100:
        errors.append("Number of steps should not exceed 100 for practical reasons")
    
    # Validate guidance scale
    if guidance_scale <= 0:
        errors.append("Guidance scale must be positive")
    
    if guidance_scale > 20:
        errors.append("Guidance scale should typically be between 1-20")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "Parameters are valid"


def sanitize_prompt(prompt: str) -> str:
    """
    Sanitize a prompt by removing potentially problematic content.
    
    Args:
        prompt: The prompt to sanitize
        
    Returns:
        Sanitized prompt
    """
    if not prompt:
        return ""
    
    # Remove excessive whitespace
    prompt = " ".join(prompt.split())
    
    # Remove or replace problematic characters
    problematic_chars = {
        "\n": " ",
        "\r": " ",
        "\t": " ",
    }
    
    for char, replacement in problematic_chars.items():
        prompt = prompt.replace(char, replacement)
    
    # Limit length
    if len(prompt) > 1000:
        prompt = prompt[:1000].rsplit(' ', 1)[0]  # Cut at word boundary
    
    return prompt.strip()


def validate_generation_params(
    prompt: str,
    height: int, 
    width: int, 
    steps: int, 
    guidance_scale: float = 7.5
) -> Tuple[bool, str]:
    """
    Validate all generation parameters including prompt.
    
    Args:
        prompt: The text prompt to validate
        height: Image height in pixels
        width: Image width in pixels
        steps: Number of inference steps
        guidance_scale: Guidance scale value
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    # Validate prompt first
    prompt_valid, prompt_message, _ = validate_prompt(prompt)
    if not prompt_valid:
        return False, f"Prompt validation failed: {prompt_message}"
    
    # Validate generation parameters
    params_valid, params_message = validate_generation_parameters(height, width, steps, guidance_scale)
    if not params_valid:
        return False, f"Parameter validation failed: {params_message}"
    
    return True, "All parameters are valid"
