"""
Common utility functions for demo scripts
"""

import time
from pathlib import Path
from typing import Optional, Union

# Check if PIL is available for image display
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def print_header(title: str):
    """Print a well-formatted header for demos."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def print_section(title: str):
    """Print a section divider with title."""
    print("\n" + "-" * 60)
    print(f" {title} ".center(60, "-"))
    print("-" * 60 + "\n")


def simulate_typing(text: str, delay: float = 0.02):
    """Simulate typing effect for demo purposes."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def display_image(path: Union[str, Path], width: Optional[int] = None, height: Optional[int] = None):
    """
    Display an image in the console or print its path if display is not possible.
    
    Args:
        path: Path to the image file
        width: Optional width for display resizing
        height: Optional height for display resizing
    """
    path = Path(path)
    
    # Check if image exists
    if not path.exists() and not str(path).startswith("mock"):
        print(f"[Image not found: {path}]")
        return
    
    # Handle mock paths
    if str(path).startswith("mock"):
        print(f"[Mock image would display here: {Path(path).name}]")
        return
    
    # Try to display the image if PIL is available
    if PIL_AVAILABLE:
        try:
            img = Image.open(path)
            if width and height:
                img = img.resize((width, height))
            
            # Check if running in Jupyter or IPython
            try:
                from IPython.display import display
                display(img)
                return
            except ImportError:
                pass
            
            # If not in Jupyter, just print dimensions
            print(f"[Image: {path.name}, {img.size[0]}x{img.size[1]}]")
            
        except Exception as e:
            print(f"[Error displaying image {path}: {e}]")
    else:
        print(f"[Image: {path.name}]")
        print(f"[Install PIL/Pillow to display images in supported environments]")


def create_progress_bar(total: int, prefix: str = "Progress", suffix: str = "Complete", 
                        length: int = 50, fill: str = "█", print_end: str = "\r"):
    """
    Create and return a progress bar function that can be called repeatedly.
    
    Returns:
        A function that takes the current iteration as an argument and prints a progress bar
    """
    def progress_bar(iteration: int):
        """Print a progress bar showing the current progress."""
        percent = 100 * (iteration / float(total))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + "-" * (length - filled_length)
        print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end=print_end)
        if iteration == total:
            print()
    
    return progress_bar


def mock_api_response(response_type: str, **kwargs):
    """
    Generate a mock API response for demo purposes.
    
    Args:
        response_type: Type of response to mock ('success', 'error', etc.)
        **kwargs: Additional fields to include in the response
    
    Returns:
        A dictionary mimicking an API response
    """
    base_response = {
        "timestamp": time.time(),
        "request_id": f"demo-{int(time.time() * 1000)}"
    }
    
    if response_type == "success":
        base_response.update({
            "status": "success",
            "code": 200
        })
    elif response_type == "error":
        base_response.update({
            "status": "error",
            "code": kwargs.get("code", 400),
            "error": kwargs.get("error", "Unknown error")
        })
    
    # Add any additional fields
    base_response.update({k: v for k, v in kwargs.items() if k not in ["code", "error"]})
    
    return base_response
