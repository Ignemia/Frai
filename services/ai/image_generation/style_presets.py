"""
from test_mock_helper import List
Style presets and prompt enhancement system for image generation.

This module provides three main style presets (Riot Games, Realistic, Anime)
with verbose prompting systems to enhance user inputs.
"""
import logging
from typing import Dict, Tuple, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)

class StylePreset(Enum):
    """Available style presets for image generation."""
    RIOT_GAMES = "riot_games"
    REALISTIC = "realistic"
    ANIME = "anime"

# Style-specific prompt templates and enhancements
STYLE_TEMPLATES = {
    StylePreset.RIOT_GAMES: {
        "style_prefix": """Professional game art in Riot Games style, League of Legends aesthetic, 
high-quality digital illustration with vibrant colors and dynamic composition. 
Clean vector-style shading with bold outlines and stylized proportions. 
Cinematic lighting with dramatic contrast and rich color palette.""",
        
        "positive_keywords": [
            "professional game art", "riot games style", "league of legends aesthetic",
            "high-quality digital illustration", "vibrant colors", "dynamic composition",
            "clean vector-style shading", "bold outlines", "stylized proportions",
            "cinematic lighting", "dramatic contrast", "rich color palette",
            "polished", "masterful", "detailed", "sharp focus"
        ],
        
        "negative_keywords": [
            "photorealistic", "real photo", "photograph", "3D render",
            "blurry", "pixelated", "low quality", "amateur", "sketchy",
            "watercolor", "oil painting", "traditional art", "messy lines",
            "inconsistent style", "flat lighting", "dull colors"
        ],
        
        "technical_params": {
            "guidance_scale": 8.0,
            "steps": 40
        }
    },
    
    StylePreset.REALISTIC: {
        "style_prefix": """Photorealistic, high-resolution photography with professional lighting and composition.
Studio quality with perfect focus, natural skin tones, and realistic textures.
Shot with professional camera equipment, detailed and lifelike.""",
        
        "positive_keywords": [
            "photorealistic", "high-resolution photography", "professional lighting",
            "studio quality", "perfect focus", "natural skin tones", "realistic textures",
            "professional camera", "detailed", "lifelike", "sharp focus",
            "natural lighting", "high detail", "crisp", "clear", "masterful photography"
        ],
        
        "negative_keywords": [
            "cartoon", "anime", "illustration", "painting", "drawing", "sketch",
            "stylized", "fantasy", "unrealistic", "artificial", "CGI",
            "oversaturated", "painterly", "abstract", "low resolution",
            "blurry", "pixelated", "amateur", "poor lighting"
        ],
        
        "technical_params": {
            "guidance_scale": 7.5,
            "steps": 50
        }
    },
    
    StylePreset.ANIME: {
        "style_prefix": """High-quality anime artwork with clean cel-shading and vibrant colors.
Studio-quality animation style with detailed character design and expressive features.
Professional manga/anime aesthetic with smooth gradients and perfect lineart.""",
        
        "positive_keywords": [
            "high-quality anime artwork", "clean cel-shading", "vibrant colors",
            "studio-quality animation", "detailed character design", "expressive features",
            "professional manga aesthetic", "smooth gradients", "perfect lineart",
            "anime style", "masterful", "detailed", "sharp focus", "clean lines",
            "consistent art style", "beautiful", "polished"
        ],
        
        "negative_keywords": [
            "photorealistic", "real photo", "3D render", "western cartoon",
            "rough sketch", "amateur", "inconsistent style", "blurry",
            "pixelated", "low quality", "messy lines", "poor anatomy",
            "distorted proportions", "ugly", "bad art", "watermark"
        ],
        
        "technical_params": {
            "guidance_scale": 7.0,
            "steps": 35
        }
    }
}

def enhance_prompt_with_style(
    user_prompt: str, 
    style: StylePreset,
    enhancement_level: float = 1.0
) -> Tuple[str, str, Dict]:
    """
    Enhance a user prompt with style-specific elements.
    
    Args:
        user_prompt: The original user prompt
        style: The style preset to apply
        enhancement_level: How much to enhance (0.0 to 1.0)
        
    Returns:
        Tuple of (enhanced_positive_prompt, negative_prompt, technical_params)
    """
    if style not in STYLE_TEMPLATES:
        logger.warning(f"Unknown style preset: {style}")
        return user_prompt, "", {}
    
    template = STYLE_TEMPLATES[style]
    
    # Build enhanced positive prompt
    enhanced_prompt_parts = []
    
    # Add style prefix with enhancement level
    if enhancement_level > 0.3:
        enhanced_prompt_parts.append(template["style_prefix"])
    
    # Add original user prompt
    enhanced_prompt_parts.append(user_prompt)
    
    # Add positive keywords based on enhancement level
    if enhancement_level > 0.5:
        keyword_count = min(int(len(template["positive_keywords"]) * enhancement_level), 8)
        selected_keywords = template["positive_keywords"][:keyword_count]
        enhanced_prompt_parts.append(", ".join(selected_keywords))
    
    enhanced_positive = ". ".join(enhanced_prompt_parts)
    
    # Build negative prompt
    negative_keywords = template["negative_keywords"]
    if enhancement_level > 0.7:
        # Use more negative keywords for stronger style enforcement
        negative_prompt = ", ".join(negative_keywords)
    else:
        # Use basic negative keywords
        negative_prompt = ", ".join(negative_keywords[:len(negative_keywords)//2])
    
    return enhanced_positive, negative_prompt, template["technical_params"]

def get_style_recommendations(user_prompt: str) -> List[Tuple[StylePreset, float]]:
    """
    Analyze a user prompt and recommend suitable styles with confidence scores.
    
    Args:
        user_prompt: The user's prompt to analyze
        
    Returns:
        List of (StylePreset, confidence_score) tuples, sorted by confidence
    """
    prompt_lower = user_prompt.lower()
    recommendations = []
    
    # Keywords that suggest specific styles
    riot_keywords = ["game", "character", "league", "fantasy", "champion", "warrior", "mage"]
    realistic_keywords = ["photo", "portrait", "person", "realistic", "natural", "human"]
    anime_keywords = ["anime", "manga", "girl", "boy", "cute", "kawaii", "character"]
    
    # Calculate confidence scores
    riot_score = sum(1 for keyword in riot_keywords if keyword in prompt_lower) / len(riot_keywords)
    realistic_score = sum(1 for keyword in realistic_keywords if keyword in prompt_lower) / len(realistic_keywords)
    anime_score = sum(1 for keyword in anime_keywords if keyword in prompt_lower) / len(anime_keywords)
    
    # Add base confidence for each style
    recommendations.append((StylePreset.RIOT_GAMES, max(0.2, riot_score)))
    recommendations.append((StylePreset.REALISTIC, max(0.2, realistic_score)))
    recommendations.append((StylePreset.ANIME, max(0.2, anime_score)))
    
    # Sort by confidence score
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations

def get_available_styles() -> Dict[str, Dict]:
    """
    Get information about all available style presets.
    
    Returns:
        Dictionary mapping style names to their descriptions and parameters
    """
    return {
        style.value: {
            "name": style.value.replace("_", " ").title(),
            "description": template["style_prefix"][:100] + "...",
            "positive_keywords": template["positive_keywords"][:5],
            "technical_params": template["technical_params"]
        }
        for style, template in STYLE_TEMPLATES.items()
    }

def create_style_prompt_variations(
    user_prompt: str, 
    style: StylePreset, 
    num_variations: int = 3
) -> List[Tuple[str, str, Dict]]:
    """
    Create multiple prompt variations for the same style with different enhancement levels.
    
    Args:
        user_prompt: The original user prompt
        style: The style preset to apply
        num_variations: Number of variations to create
        
    Returns:
        List of (positive_prompt, negative_prompt, technical_params) tuples
    """
    variations = []
    enhancement_levels = [0.3, 0.7, 1.0][:num_variations]
    
    for level in enhancement_levels:
        positive, negative, params = enhance_prompt_with_style(user_prompt, style, level)
        variations.append((positive, negative, params))
    
    return variations
