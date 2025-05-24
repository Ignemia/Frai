"""
from typing import List
Command processing module for Frai.

This module handles preprocessing of user messages to detect specific command
intents like image generation, online search, storing user information, etc.
It uses a lightweight model to classify commands before passing to the main LLM.
"""
import logging
import re
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

from services.chat.llm_interface import send_query

logger = logging.getLogger(__name__)

class CommandIntent(Enum):
    """Enum representing possible command intents detected in user messages."""
    NONE = auto()  # No special command detected
    GENERATE_IMAGE = auto()  # User wants to generate an image
    ONLINE_SEARCH = auto()  # User wants to search online
    STORE_USER_INFO = auto()  # User wants to store personal information
    STORE_MEMORY = auto()  # User wants to store something for later reference
    SEARCH_LOCAL = auto()  # User wants to search local documents

# Regular expressions for quick pattern matching
PATTERNS = {
    CommandIntent.GENERATE_IMAGE: [
        r"(?i)(create|generate|make|draw|show me|produce|render|design).*?(image|picture|photo|drawing|illustration|art|artwork|visual)",
        r"(?i)(can you|please|would you).*?(create|generate|make|draw|show me|visualize).*?(image|picture|photo|drawing|illustration|art|artwork|visual)",
        r"(?i)(imagine|visualize|draw|create).*?(scene|character|landscape|portrait)",
        r"(?i)(paint|sketch|create).*?(image|picture|photo)",
    ],
    CommandIntent.ONLINE_SEARCH: [
        r"(?i)(search|find|look up|google|research|browse).*?(online|internet|web)",
        r"(?i)(what is|who is|tell me about|information about|details on|search for).*?",
        r"(?i)(find out|research|get information).*?(about|regarding|concerning)",
        r"(?i)(latest|recent|current|news).*?(about|on|regarding)",
    ],
    CommandIntent.STORE_USER_INFO: [
        r"(?i)(remember|store|save|keep|record).*?(my|about me|information|profile|details)",
        r"(?i)(my name is|I am|I'm|I live in|my address is|my email is|my phone is|my birthday is)",
        r"(?i)(update|change|modify).*?(my profile|my information|my details)",
    ],
    CommandIntent.STORE_MEMORY: [
        r"(?i)(remember|store|save|memorize|note).*?(this|that|the following|these details)",
        r"(?i)(keep|retain|preserve).*?(this information|this for later|this data)",
        r"(?i)(save to|add to).*?(memory|your memory|your knowledge|your database)",
    ],
    CommandIntent.SEARCH_LOCAL: [
        r"(?i)(search|find|locate|look through).*?(local|my files|my documents|documents|file)",
        r"(?i)(find me|show me|retrieve).*?(file|document|record|note).*?(that contains|about|related to)",
        r"(?i)(open|access|get).*?(file|document|record|note).*?(with|containing)",
    ],
}

def preprocess_with_main_model(message: str) -> Tuple[CommandIntent, Dict[str, Any]]:
    """
    Process a message using the main LLM to detect command intents.
    
    Args:
        message: The user's message
        
    Returns:
        Tuple containing:
            - The detected command intent
            - Extra parameters for the command (if any)
    """
    logger.debug(f"Preprocessing message with main model: {message[:50]}...")
    
    # Construct a prompt for command detection
    prompt = f"""Analyze the following user message and classify it into one of these intents:
1. GENERATE_IMAGE: If the user wants an image to be created/generated/visualized
2. ONLINE_SEARCH: If the user wants to search for information on the internet
3. STORE_USER_INFO: If the user is providing personal information about themselves to be remembered
4. STORE_MEMORY: If the user wants to store some general information/facts for later reference
5. SEARCH_LOCAL: If the user wants to search through local documents or files
6. NONE: If none of the above commands are being requested

User message: "{message}"

Output the classification as a single word (one of the above intent names) followed by any relevant extraction parameters as JSON.
For example, for image generation, include the description of what should be in the image.
For searches, include the search query.
For storing information, include what specific information should be stored.
"""
    
    try:
        # Create a system message for command classification
        system_message = {"role": "system", "content": "You are a command classifier that identifies user intentions."}
        
        # Call the LLM to detect the intent with the system message as the first history entry
        response, _ = send_query(prompt, [system_message])
        logger.debug(f"LLM command classification response: {response}")
        
        # Parse the response to extract intent and parameters
        intent, params = _parse_llm_command_response(response, message)
        return intent, params
    except Exception as e:
        logger.error(f"Error in command preprocessing with main model: {e}")
        return CommandIntent.NONE, {"error": str(e)}

def _parse_llm_command_response(response: str, original_message: str) -> Tuple[CommandIntent, Dict[str, Any]]:
    """
    Parse the LLM's response to extract command intent and parameters.
    
    Args:
        response: The LLM's classification response
        original_message: The original user message for fallback extraction
        
    Returns:
        Tuple containing:
            - The detected command intent
            - Extra parameters for the command (if any)
    """
    # Default response
    params = {"original_message": original_message}
    
    # Look for intent classification at the beginning of the response
    for intent in CommandIntent:
        if intent.name in response.upper().split():
            # Found a matching intent
            if intent == CommandIntent.GENERATE_IMAGE:
                # Extract image description
                params["image_prompt"] = _extract_text_after_marker(response, ["image prompt:", "description:", "image:"])
                if not params.get("image_prompt"):
                    params["image_prompt"] = original_message
            
            elif intent == CommandIntent.ONLINE_SEARCH:
                # Extract search query
                params["search_query"] = _extract_text_after_marker(response, ["query:", "search for:", "search query:"])
                if not params.get("search_query"):
                    params["search_query"] = original_message
                    
            elif intent == CommandIntent.STORE_USER_INFO:
                # Extract user information
                params["user_info"] = _extract_text_after_marker(response, ["user info:", "information:", "profile:"])
                if not params.get("user_info"):
                    params["user_info"] = original_message
                    
            elif intent == CommandIntent.STORE_MEMORY:
                # Extract memory content
                params["memory_content"] = _extract_text_after_marker(response, ["memory:", "content:", "remember:"])
                if not params.get("memory_content"):
                    params["memory_content"] = original_message
                    
            elif intent == CommandIntent.SEARCH_LOCAL:
                # Extract local search query
                params["local_query"] = _extract_text_after_marker(response, ["local query:", "document query:", "file search:"])
                if not params.get("local_query"):
                    params["local_query"] = original_message
            
            return intent, params
    
    # Default to NONE if no intent was detected
    return CommandIntent.NONE, params

def _extract_text_after_marker(text: str, markers: List[str]) -> Optional[str]:
    """
    Extract text after a specific marker or label.
    
    Args:
        text: The text to extract from
        markers: List of possible marker strings to look for
        
    Returns:
        The extracted text, or None if no marker was found
    """
    lower_text = text.lower()
    for marker in markers:
        marker = marker.lower()
        if marker in lower_text:
            # Find the marker's position and extract everything after it
            pos = lower_text.find(marker) + len(marker)
            extracted = text[pos:].strip()
            
            # Look for end markers like the next line or period
            lines = extracted.split('\n', 1)
            if len(lines) > 1:
                extracted = lines[0].strip()
            
            # If it's still too long, limit it
            if len(extracted) > 500:
                extracted = extracted[:497] + "..."
                
            return extracted
    
    return None

def preprocess_with_pattern_matching(message: str) -> Tuple[CommandIntent, Dict[str, Any]]:
    """
    Process a message using regex pattern matching for quick intent detection.
    
    Args:
        message: The user's message
        
    Returns:
        Tuple containing:
            - The detected command intent
            - Extra parameters for the command (if any)
    """
    message_lower = message.lower()
    params = {"original_message": message}
    
    for intent, patterns in PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, message):
                if intent == CommandIntent.GENERATE_IMAGE:
                    params["image_prompt"] = message
                elif intent == CommandIntent.ONLINE_SEARCH:
                    params["search_query"] = message
                elif intent == CommandIntent.STORE_USER_INFO:
                    params["user_info"] = message
                elif intent == CommandIntent.STORE_MEMORY:
                    params["memory_content"] = message
                elif intent == CommandIntent.SEARCH_LOCAL:
                    params["local_query"] = message
                    
                return intent, params
    
    return CommandIntent.NONE, params

def preprocess_message(message: str, use_main_model: bool = True) -> Tuple[CommandIntent, Dict[str, Any]]:
    """
    Preprocess a user message to detect command intents.
    
    This function first tries pattern matching for efficiency, and if that fails
    and use_main_model is True, it uses the main LLM for more accurate detection.
    
    Args:
        message: The user's message to process
        use_main_model: Whether to use the main LLM if pattern matching fails
        
    Returns:
        Tuple containing:
            - The detected command intent
            - Extra parameters for the command (if any)
    """
    logger.debug(f"Preprocessing message: {message[:50]}...")
    
    # First try with simple pattern matching (fast)
    intent, params = preprocess_with_pattern_matching(message)
    
    # If no intent was detected and we're allowed to use the main model, try that
    if intent == CommandIntent.NONE and use_main_model:
        intent, params = preprocess_with_main_model(message)
    
    logger.info(f"Message preprocessed, intent detected: {intent.name}")
    return intent, params
