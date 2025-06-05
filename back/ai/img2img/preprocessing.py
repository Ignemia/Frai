"""Placeholder preprocessing utilities for img2img tests."""

def preprocess_image(image_path: str):
    """Return a mock processed representation of an image path."""
    if not image_path:
        return None
    return f"processed:{image_path}"
