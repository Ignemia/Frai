"""
File utility functions for image generation.

Provides utilities for file operations, path generation,
and directory management.
"""
import os
import time
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_filename(prompt: str, extension: str = "png", max_length: int = 50) -> str:
    """
    Generate a safe filename from a prompt.
    
    Args:
        prompt: The prompt to use for filename generation
        extension: File extension (without dot)
        max_length: Maximum length for the prompt part
        
    Returns:
        Generated filename
    """
    # Sanitize the prompt for use in filename
    safe_prompt = "".join(c for c in prompt[:max_length] if c.isalnum() or c in (' ', '_')).strip()
    safe_prompt = safe_prompt.replace(' ', '_')
    
    # If prompt becomes empty after sanitization, use a default
    if not safe_prompt:
        safe_prompt = "generated_image"
    
    # Add timestamp
    timestamp = int(time.time())
    
    # Construct filename
    filename = f"image_{safe_prompt}_{timestamp}.{extension}"
    
    return filename


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {directory_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        return False


def get_safe_file_path(directory: str, filename: str) -> str:
    """
    Get a safe file path, handling naming conflicts.
    
    Args:
        directory: Target directory
        filename: Desired filename
        
    Returns:
        Safe file path that doesn't conflict with existing files
    """
    base_path = os.path.join(directory, filename)
    
    # If file doesn't exist, return as-is
    if not os.path.exists(base_path):
        return base_path
    
    # Extract name and extension
    name, ext = os.path.splitext(filename)
    
    # Find a non-conflicting name
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_path = os.path.join(directory, new_filename)
        
        if not os.path.exists(new_path):
            return new_path
        
        counter += 1
        
        # Safety check to prevent infinite loop
        if counter > 1000:
            # Use timestamp as fallback
            timestamp = int(time.time())
            new_filename = f"{name}_{timestamp}{ext}"
            return os.path.join(directory, new_filename)


def cleanup_old_files(directory: str, max_age_days: int = 7, max_files: int = 100) -> int:
    """
    Clean up old files in a directory.
    
    Args:
        directory: Directory to clean up
        max_age_days: Maximum age of files to keep (in days)
        max_files: Maximum number of files to keep (keeps newest)
        
    Returns:
        Number of files deleted
    """
    if not os.path.exists(directory):
        return 0
    
    try:
        files = []
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        # Get all files with their modification times
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                mtime = os.path.getmtime(filepath)
                files.append((filepath, mtime))
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x[1], reverse=True)
        
        deleted_count = 0
        
        # Delete files that are too old or exceed the maximum count
        for i, (filepath, mtime) in enumerate(files):
            should_delete = False
            
            # Check age
            if current_time - mtime > max_age_seconds:
                should_delete = True
                logger.debug(f"File too old: {filepath}")
            
            # Check count limit (keep newest files)
            elif i >= max_files:
                should_delete = True
                logger.debug(f"File exceeds count limit: {filepath}")
            
            if should_delete:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                    logger.debug(f"Deleted old file: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {filepath}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old files from {directory}")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")
        return 0


def get_file_size_mb(filepath: str) -> Optional[float]:
    """
    Get file size in megabytes.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File size in MB or None if file doesn't exist
    """
    try:
        if os.path.exists(filepath):
            size_bytes = os.path.getsize(filepath)
            return size_bytes / (1024 * 1024)
    except Exception as e:
        logger.warning(f"Could not get file size for {filepath}: {e}")
    
    return None
