# -*- coding: utf-8 -*-
# This module initializes the chat moderator
# Chat Moderator deals with chat moderation that is safety, message filtering, user message validation, etc.
# Chat Moderator does not handle chat actions (sending/receiving messages, storing/loading message history)

import logging
import re
from typing import Dict, List, Optional, Tuple
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

class ChatModerator:
    """
    Chat moderator for content safety, filtering, and validation.
    """
    
    def __init__(self):
        self.toxic_keywords = [
            # Add common toxic keywords here - keeping it minimal for example
            "hate", "violence", "harmful", "illegal", "abuse"
        ]
        self.spam_patterns = [
            r"(.)\1{10,}",  # Repeated characters
            r"[A-Z]{5,}",   # All caps sequences
            r"(\w+\s+){20,}",  # Excessive repetition
        ]
        self.max_message_length = 8192
        self.min_message_length = 1
          # Initialize sentiment analysis model if available
        try:
            # Use local sentiment analysis model
            import os
            model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "multilingual-sentiment-analysis")
            if os.path.exists(model_path):
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_path,
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info(f"Local sentiment analyzer loaded successfully from {model_path}")
            else:
                # Fallback to online model
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Online sentiment analyzer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    def validate_message_format(self, message: str) -> Tuple[bool, Optional[str]]:
        """
        Validate basic message format and structure.
        
        Args:
            message: The message to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(message, str):
            return False, "Message must be a string"
        
        if len(message) < self.min_message_length:
            return False, f"Message too short (minimum {self.min_message_length} characters)"
        
        if len(message) > self.max_message_length:
            return False, f"Message too long (maximum {self.max_message_length} characters)"
        
        # Check for null bytes and other problematic characters
        if '\x00' in message:
            return False, "Message contains null bytes"
        
        return True, None
    
    def check_spam_patterns(self, message: str) -> Tuple[bool, Optional[str]]:
        """
        Check if message matches common spam patterns.
        
        Args:
            message: The message to check
            
        Returns:
            Tuple of (is_spam, reason)
        """
        for pattern in self.spam_patterns:
            if re.search(pattern, message):
                return True, f"Message matches spam pattern: {pattern}"
        
        return False, None
    
    def check_toxic_content(self, message: str) -> Tuple[bool, Optional[str]]:
        """
        Check for basic toxic content using keyword matching.
        
        Args:
            message: The message to check
            
        Returns:
            Tuple of (is_toxic, reason)
        """
        message_lower = message.lower()
        
        for keyword in self.toxic_keywords:
            if keyword in message_lower:
                return True, f"Message contains potentially harmful content"
        
        return False, None
    
    def analyze_sentiment(self, message: str) -> Optional[Dict]:
        """
        Analyze message sentiment if sentiment analyzer is available.
        
        Args:
            message: The message to analyze
            
        Returns:
            Sentiment analysis results or None
        """
        if not self.sentiment_analyzer:
            return None
        
        try:
            result = self.sentiment_analyzer(message)
            return result[0] if result else None
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return None
    
    def moderate_message(self, message: str, user_id: Optional[str] = None) -> Dict:
        """
        Perform comprehensive message moderation.
        
        Args:
            message: The message to moderate
            user_id: Optional user ID for context
            
        Returns:
            Moderation result dictionary with flags, not approval.
        """
        result = {
            # "approved": False, # Removed
            "message": message,
            "user_id": user_id,
            "filters_triggered": [],
            "warnings": [],
            "sentiment": None,
            "error": None
        }
        
        try:
            # Basic format validation
            is_valid, error_msg = self.validate_message_format(message)
            if not is_valid:
                result["error"] = error_msg
                return result
            
            # Spam detection
            is_spam, spam_reason = self.check_spam_patterns(message)
            if is_spam:
                result["filters_triggered"].append("spam")
                result["warnings"].append(spam_reason)
            
            # Toxic content detection
            is_toxic, toxic_reason = self.check_toxic_content(message)
            if is_toxic:
                result["filters_triggered"].append("toxic_content")
                result["warnings"].append(toxic_reason)
            
            # Sentiment analysis
            sentiment = self.analyze_sentiment(message)
            if sentiment:
                result["sentiment"] = sentiment
                
                # Flag very negative sentiment
                if sentiment.get("label") == "NEGATIVE" and sentiment.get("score", 0) > 0.9:
                    result["filters_triggered"].append("very_negative")
                    result["warnings"].append("Message has very negative sentiment")
            
            # Removed approval logic:
            # critical_filters = ["toxic_content", "spam"]
            # if not any(f in critical_filters for f in result["filters_triggered"]):
            #     result["approved"] = True
            
            logger.info(f"Message moderation completed: filters={result['filters_triggered']}")
            
        except Exception as e:
            logger.error(f"Error during message moderation: {e}")
            result["error"] = f"Moderation error: {str(e)}"
        
        return result
    
    def filter_response(self, response: str) -> Dict:
        """
        Filter and validate AI-generated responses by flagging, not altering.
        
        Args:
            response: The AI response to filter
            
        Returns:
            Filtering result dictionary with flags.
        """
        result = {
            # "approved": False, # Removed
            "original_response": response, # Renamed from "response" for clarity
            # "filtered_response": response, # Removed
            "filters_applied": [],
            "warnings": []
        }
        
        try:
            # Basic validation
            is_valid, error_msg = self.validate_message_format(response)
            if not is_valid:
                result["warnings"].append(error_msg) # Add to warnings, don't necessarily stop
                # If basic validation fails, we might not want to proceed with other checks,
                # or just flag it and let frontend decide. For now, let's flag and continue.
                result["filters_applied"].append("invalid_format")


            # Check for URLs that might be suspicious - now only flags
            url_pattern = r'https?://[^\\s]+'
            if re.search(url_pattern, response): # Check original response
                # filtered_response = re.sub(url_pattern, '[URL removed]', filtered_response) # No longer modifying
                result["filters_applied"].append("url_detected")
                result["warnings"].append("URL detected in response")

            # Check for toxic content in response
            is_toxic, toxic_reason = self.check_toxic_content(response) # Check original response
            if is_toxic:
                result["filters_applied"].append("toxic_content")
                result["warnings"].append(toxic_reason if toxic_reason else "AI response contains potentially harmful content")
            # Removed approval logic:
            # else:
            #     result["approved"] = True
            
            # result["filtered_response"] = filtered_response # Removed
            
        except Exception as e:
            logger.error(f"Error filtering response: {e}")
            result["warnings"].append(f"Filter error: {str(e)}")
        
        return result


# Global moderator instance
_chat_moderator = None

def get_chat_moderator() -> ChatModerator:
    """Get the global chat moderator instance."""
    global _chat_moderator
    if _chat_moderator is None:
        _chat_moderator = ChatModerator()
    return _chat_moderator

def initiate_chat_moderator():
    """Initialize the chat moderator."""
    try:
        logger.info("Initializing chat moderator...")
        moderator = get_chat_moderator()
        logger.info("Chat moderator initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize chat moderator: {e}")
        return False

# Convenience functions for external use
def moderate_user_message(message: str, user_id: Optional[str] = None) -> Dict:
    """Moderate a user message."""
    return get_chat_moderator().moderate_message(message, user_id)

def filter_ai_response(response: str) -> Dict:
    """Filter an AI response."""
    return get_chat_moderator().filter_response(response)