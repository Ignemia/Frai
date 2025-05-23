"""
User memory service for Personal Chatter.

This module handles storing and retrieving personal information and general memory
items provided by users.
"""
import logging
import os
import json
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration
USER_INFO_DIR = "./outputs/user_info"
MEMORY_DIR = "./outputs/memory"

def _ensure_dirs():
    """Ensure the required directories exist."""
    for directory in [USER_INFO_DIR, MEMORY_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def store_user_information(user_id: str, information: Dict[str, Any]) -> bool:
    """
    Store user personal information.
    
    Args:
        user_id: The user identifier
        information: Dictionary of user information to store
        
    Returns:
        True if successful, False otherwise
    """
    _ensure_dirs()
    
    try:
        # Ensure we have a file for this user
        user_file = os.path.join(USER_INFO_DIR, f"{user_id}.json")
        
        # If the file exists, load and update it
        existing_data = {}
        if os.path.exists(user_file):
            with open(user_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        
        # Update the data with new information
        existing_data.update(information)
        
        # Add metadata
        existing_data["_last_updated"] = time.time()
        
        # Write back to the file
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"Successfully stored user information for user {user_id}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to store user information: {e}")
        return False

def get_user_information(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve stored user information.
    
    Args:
        user_id: The user identifier
        
    Returns:
        Dictionary of user information if found, None otherwise
    """
    _ensure_dirs()
    
    try:
        user_file = os.path.join(USER_INFO_DIR, f"{user_id}.json")
        if not os.path.exists(user_file):
            logger.warning(f"No user information found for user {user_id}")
            return None
        
        with open(user_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"Successfully retrieved user information for user {user_id}")
        return data
    
    except Exception as e:
        logger.error(f"Failed to retrieve user information: {e}")
        return None

def store_memory_item(user_id: str, content: str, tags: Optional[List[str]] = None) -> bool:
    """
    Store a memory item for later retrieval.
    
    Args:
        user_id: The user identifier
        content: The memory content to store
        tags: Optional list of tags/categories for the memory
        
    Returns:
        True if successful, False otherwise
    """
    _ensure_dirs()
    
    try:
        # Create a memory item with metadata
        memory_item = {
            "content": content,
            "tags": tags or [],
            "created": time.time(),
            "user_id": user_id
        }
        
        # Generate a filename based on timestamp
        timestamp = int(time.time())
        memory_file = os.path.join(MEMORY_DIR, f"{user_id}_{timestamp}.json")
        
        # Write to the file
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memory_item, f, indent=2)
        
        logger.info(f"Successfully stored memory item for user {user_id}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to store memory item: {e}")
        return False

def retrieve_memory_items(user_id: str, query: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Retrieve memory items for a user, optionally filtered by query or tags.
    
    Args:
        user_id: The user identifier
        query: Optional search query to filter memories
        tags: Optional list of tags to filter by
        
    Returns:
        List of memory items matching the criteria
    """
    _ensure_dirs()
    
    results = []
    
    try:
        # List all memory files for this user
        for filename in os.listdir(MEMORY_DIR):
            if filename.startswith(f"{user_id}_") and filename.endswith(".json"):
                filepath = os.path.join(MEMORY_DIR, filename)
                
                with open(filepath, "r", encoding="utf-8") as f:
                    memory_item = json.load(f)
                
                # Apply tag filtering if specified
                if tags and not any(tag in memory_item.get("tags", []) for tag in tags):
                    continue
                
                # Apply query filtering if specified
                if query and query.lower() not in memory_item.get("content", "").lower():
                    continue
                
                # Add the matching item to results
                results.append(memory_item)
        
        # Sort by created timestamp (newest first)
        results.sort(key=lambda x: x.get("created", 0), reverse=True)
        
        logger.info(f"Retrieved {len(results)} memory items for user {user_id}")
        return results
    
    except Exception as e:
        logger.error(f"Failed to retrieve memory items: {e}")
        return []

def extract_user_information(message: str) -> Dict[str, Any]:
    """
    Extract structured user information from a message.
    
    Args:
        message: The message to extract information from
        
    Returns:
        Dictionary of extracted information
    """
    # This is a simple implementation - in a real system, you'd use more
    # sophisticated NLP techniques or a dedicated model for entity extraction
    
    info = {}
    
    # Simple pattern matching for common information
    import re
    
    # Name extraction
    name_match = re.search(r"my name is\s+([A-Za-z\s]+)", message, re.IGNORECASE)
    if name_match:
        info["name"] = name_match.group(1).strip()
    
    # Email extraction
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", message)
    if email_match:
        info["email"] = email_match.group(0)
    
    # Phone extraction
    phone_match = re.search(r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", message)
    if phone_match:
        info["phone"] = phone_match.group(0)
    
    # Location/address extraction
    location_match = re.search(r"I live in\s+([A-Za-z\s,]+)", message, re.IGNORECASE)
    if location_match:
        info["location"] = location_match.group(1).strip()
    
    return info
