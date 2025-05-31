"""
Backend Text-to-Image AI Module Tests

This module defines unit tests for the backend text-to-image functionality, focusing on prompt interpretation, image quality, and style consistency across diverse text descriptions.

Test categories:
1. Unit tests for individual txt2img components (e.g., prompt processors, model wrappers, post-processors).
2. Quality tests comparing generated images against expected visual characteristics for various prompt types.
3. Style consistency tests ensuring similar prompts produce coherent visual styles and themes.
4. Prompt interpretation tests validating the model's understanding of descriptive language, objects, scenes, and artistic styles.
5. Edge case tests for empty prompts, very long prompts, conflicting descriptions, non-English text, and special characters.
6. Performance tests validating generation speed and resource usage for single and batch operations.

Because image quality is subjective and automated evaluation is limited, tests focus on successful generation, basic image properties (resolution, format), and consistency metrics rather than aesthetic quality.

Test data specification:
- testset.csv must contain:
    id: Unique test identifier prefixed with t2i-<id>
    name: Human-readable name for the test
    description: Brief explanation of test objective
    prompt_category: Type of prompt (e.g., simple_object, complex_scene, artistic_style)
    text_prompt: The input text description
    expected_properties: Describable properties the image should have
    evaluation_metric: How success should be measured (e.g., generation_success, resolution_check)

- prompt_categories.csv must contain:
    category: Category name (e.g., portraits, landscapes, abstract)
    description: What this category tests
    example_prompts: Sample prompts for this category
    complexity_level: Simple, medium, or complex
    expected_features: Visual features typically expected
"""