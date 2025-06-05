"""Mock style transfer utilities."""


def extract_style_features(image_path: str):
    """Return a dummy representation of style features."""
    if not image_path:
        return None
    return {"style_features": f"features_of:{image_path}"}
