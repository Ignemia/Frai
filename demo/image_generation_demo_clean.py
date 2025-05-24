#!/usr/bin/env python3
"""
Image Generation Demo for Personal Chatter - NO MOCK RESPONSES

This demo showcases the REAL image generation capabilities:
- Configuring image generation settings
- Generating images from text prompts
- Showing various parameter combinations
- Displaying output and metadata

NO FALLBACKS OR MOCK RESPONSES - REAL IMAGE GENERATION ONLY
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required services and utilities
from services.image_generation.image_generator import ImageGenerator
from services.config import ImageGenerationConfig
from demo.utils import print_header, print_section, display_image


def run_image_generation_demo():
    """Run the REAL image generation demonstration."""
    print_header("Personal Chatter - Image Generation Demo (REAL GENERATION ONLY)")
    
    # Create output directory if it doesn't exist
    output_dir = project_root / "outputs" / "demo_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🖼️ Image generation demo starting...")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Initialize with standard configuration
    print("⚙️ Initializing image generator with standard configuration...")
    config = ImageGenerationConfig(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        output_dir=str(output_dir),
        width=512,
        height=512, 
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    generator = ImageGenerator(config)
    print("✅ Image generator initialized successfully")
    
    # Sample prompts for demonstration
    print("\n📝 Sample prompts for demonstration:")
    prompts = [
        "A serene landscape with mountains and a lake at sunset",
        "A futuristic cityscape with flying cars and neon lights",
        "An adorable cat wearing a space helmet",
        "A detailed portrait of a fantasy character with glowing eyes"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")
    
    # Generate first image with standard settings
    print("\n🚀 Generating first image with standard settings...")
    result = generator.generate_image(
        prompt=prompts[0],
        seed=1234,  # Fixed seed for reproducibility
    )
    
    if not result.get("success", False):
        raise Exception(f"Image generation failed: {result.get('error')}")
    
    print(f"✅ Image generated successfully: {result.get('image_path')}")
    display_image(result.get("image_path"))
    
    # Show metadata
    print("\n📊 Image metadata:")
    for key, value in result.get("metadata", {}).items():
        if key not in ["prompt"]:  # Skip prompt as we already showed it
            print(f"  {key}: {value}")
    
    # Generate second image with different parameters
    print("\n🚀 Generating second image with different parameters...")
    result = generator.generate_image(
        prompt=prompts[1],
        width=768,
        height=512,
        num_inference_steps=40,
        guidance_scale=9.0,
        seed=5678
    )
    
    if not result.get("success", False):
        raise Exception(f"Second image generation failed: {result.get('error')}")
    
    print(f"✅ Image generated successfully: {result.get('image_path')}")
    display_image(result.get("image_path"))
    
    # Show comparison of parameters
    print("\n📊 Parameter comparison:")
    print("  Standard vs. Enhanced:")
    print("  - Resolution: 512x512 → 768x512")
    print("  - Steps: 30 → 40")
    print("  - Guidance Scale: 7.5 → 9.0")
    
    # Generate variations of the same prompt
    print("\n🔄 Generating variations with the same prompt but different seeds...")
    base_prompt = prompts[2]
    seeds = [1000, 2000, 3000]
    
    for i, seed in enumerate(seeds):
        print(f"\n🌱 Variation {i+1} with seed {seed}...")
        result = generator.generate_image(
            prompt=base_prompt,
            seed=seed,
        )
        
        if not result.get("success", False):
            raise Exception(f"Variation {i+1} failed: {result.get('error')}")
        
        print(f"✅ Variation {i+1} generated: {result.get('image_path')}")
        display_image(result.get("image_path"))
    
    # Show summary
    print("\n📋 Generation summary:")
    print(f"  Total images attempted: 5")
    print(f"  Successfully generated: {generator.stats.get('successful_generations', 0)}")
    print(f"  Failed generations: {generator.stats.get('failed_generations', 0)}")
    print(f"  Average generation time: {generator.stats.get('avg_generation_time', 0):.2f}s")
    
    return True


def main():
    """Main entry point for the image generation demo."""
    try:
        success = run_image_generation_demo()
        if success:
            print("\n✅ Image generation demo completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Image generation demo failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
