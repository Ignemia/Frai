"""
Backend Text-to-Image AI Module Tests

This module defines unit tests for the backend text-to-image generation functionality.

Test categories:
1. Unit tests for individual functions and classes in text2img components.
2. Prompt formatting tests to verify text and negative prompts are constructed correctly.
3. Consistency tests ensuring generated images align with the specified prompts.
4. Diversity tests confirming varied outputs for identical prompts when randomness is expected.
5. Performance tests validating that image generation meets latency requirements.

Because image quality is subjective, many tests rely on manual review or comparison against reference outputs.

Test data specification:
- testset.csv must contain:
    id: Unique test identifier prefixed with t2i-<id>
    name: Human-readable name for the test
    description: Concise explanation of the test objective
    prompt_group: Logical grouping of related prompts
    generation_params: Parameters used for generation
    expected_outcome: Criteria defining a successful result
"""
