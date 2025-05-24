#!/usr/bin/env python3
"""
Image Generation Demo for Personal Chatter

This demo showcases the image generation capabilities including:
- Configuring image generation settings
- Generating images from text prompts
- Showing various parameter combinations
- Displaying output and metadata
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

# Import demo utils
from demo import utils
if not hasattr(utils, 'display_image'):
    # Create a simple mock module if the real one doesn't exist
    class MockUtils:
        @staticmethod
        def display_image(path, width=None, height=None):
            print(f"[Image would display here: {path}]")
            
        @staticmethod
        def print_header(title):
            print("\n" + "=" * 80)
            print(f" {title} ".center(80, "="))
            print("=" * 80 + "\n")
            
    utils = MockUtils()


def run_image_generation_demo():
    """Run the image generation demonstration."""
    try:
        from services.image_generation.image_generator import ImageGenerator
        from services.config import ImageGenerationConfig
        
        utils.print_header("Personal Chatter - Image Generation Demo")
        
        # Create output directory if it doesn't exist
        output_dir = project_root / "outputs" / "demo_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🖼️ Image generation demo starting...")
        print(f"📁 Output directory: {output_dir}")
        print()
        
        # Initialize with standard configuration
        print("⚙️ Initializing image generator with standard configuration...")
        config = ImageGenerationConfig(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",  # Example model
            output_dir=str(output_dir),
            width=512,
            height=512, 
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        generator = ImageGenerator(config)
        print("✅ Image generator initialized successfully")
        
        # Show some sample prompts
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
        
        if result.get("success", False):
            print(f"✅ Image generated successfully: {result.get('image_path')}")
            utils.display_image(result.get("image_path"))
            
            # Show metadata
            print("\n📊 Image metadata:")
            for key, value in result.get("metadata", {}).items():
                if key not in ["prompt"]:  # Skip prompt as we already showed it
                    print(f"  {key}: {value}")
        else:
            print(f"❌ Image generation failed: {result.get('error')}")
            
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
        
        if result.get("success", False):
            print(f"✅ Image generated successfully: {result.get('image_path')}")
            utils.display_image(result.get("image_path"))
            
            # Show comparison of parameters
            print("\n📊 Parameter comparison:")
            print("  Standard vs. Enhanced:")
            print("  - Resolution: 512x512 → 768x512")
            print("  - Steps: 30 → 40")
            print("  - Guidance Scale: 7.5 → 9.0")
        else:
            print(f"❌ Image generation failed: {result.get('error')}")
            
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
            
            if result.get("success", False):
                print(f"✅ Variation {i+1} generated: {result.get('image_path')}")
                utils.display_image(result.get("image_path"))
            else:
                print(f"❌ Variation {i+1} failed: {result.get('error')}")
        
        # Show summary
        print("\n📋 Generation summary:")
        print(f"  Total images attempted: 5")
        print(f"  Successfully generated: {generator.stats.get('successful_generations', 0)}")
        print(f"  Failed generations: {generator.stats.get('failed_generations', 0)}")
        print(f"  Average generation time: {generator.stats.get('avg_generation_time', 0):.2f}s")
        
        return True
        
    except ImportError:
        print("⚠️ Image generation service not available. Running mock demonstration...")
        return run_mock_image_generation_demo()
    except Exception as e:
        print(f"❌ Error in image generation demo: {e}")
        return False


def run_mock_image_generation_demo():
    """Run a mock version of the image generation demo when services aren't available."""
    utils.print_header("Personal Chatter - Image Generation Demo (Mock)")
    
    # Mock output directory
    output_dir = project_root / "outputs" / "demo_images"
    print(f"🖼️ Image generation demo starting (mock mode)...")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Mock configuration
    print("⚙️ Initializing image generator with standard configuration...")
    time.sleep(1)
    print("✅ Image generator initialized successfully")
    
    # Sample prompts
    print("\n📝 Sample prompts for demonstration:")
    prompts = [
        "A serene landscape with mountains and a lake at sunset",
        "A futuristic cityscape with flying cars and neon lights",
        "An adorable cat wearing a space helmet",
        "A detailed portrait of a fantasy character with glowing eyes"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")
    
    # Mock first generation
    print("\n🚀 Generating first image with standard settings...")
    time.sleep(2)  # Simulate generation time
    mock_path_1 = str(output_dir / "mock_landscape.png")
    print(f"✅ Image generated successfully: {mock_path_1}")
    utils.display_image(mock_path_1)
    
    # Mock metadata
    print("\n📊 Image metadata:")
    print(f"  width: 512")
    print(f"  height: 512")
    print(f"  steps: 30")
    print(f"  guidance_scale: 7.5")
    print(f"  seed: 1234")
    print(f"  generation_time: 3.42s")
    
    # Mock second generation
    print("\n🚀 Generating second image with different parameters...")
    time.sleep(2.5)  # Simulate generation time
    mock_path_2 = str(output_dir / "mock_cityscape.png")
    print(f"✅ Image generated successfully: {mock_path_2}")
    utils.display_image(mock_path_2)
    
    # Mock parameter comparison
    print("\n📊 Parameter comparison:")
    print("  Standard vs. Enhanced:")
    print("  - Resolution: 512x512 → 768x512")
    print("  - Steps: 30 → 40")
    print("  - Guidance Scale: 7.5 → 9.0")
    
    # Mock variations
    print("\n🔄 Generating variations with the same prompt but different seeds...")
    seeds = [1000, 2000, 3000]
    
    for i, seed in enumerate(seeds):
        print(f"\n🌱 Variation {i+1} with seed {seed}...")
        time.sleep(1.5)  # Simulate generation time
        mock_path = str(output_dir / f"mock_cat_variation_{i+1}.png")
        print(f"✅ Variation {i+1} generated: {mock_path}")
        utils.display_image(mock_path)
    
    # Mock summary
    print("\n📋 Generation summary:")
    print(f"  Total images attempted: 5")
    print(f"  Successfully generated: 5")
    print(f"  Failed generations: 0")
    print(f"  Average generation time: 2.37s")
    
    return True


def main():
    """Main entry point for the image generation demo."""
    success = run_image_generation_demo()
    
    if success:
        print("\n✅ Image generation demo completed successfully!")
    else:
        print("\n❌ Image generation demo encountered errors.")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
