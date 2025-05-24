"""
Backend AI Chat Module

This module handles the AI chat functionality, including model loading,
text generation, and integration with the chat orchestrator and moderator.
"""

import logging
import os
from typing import Dict, Optional, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatAI:
    """
    AI chat handler that integrates with chat orchestrator and moderator.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generation_pipeline = None
        self.is_loaded = False
        self.model_name = "google/gemma-3-4b-it"  # Default model
        self.max_length = 2048
        self.temperature = 0.8
        self.top_p = 0.95
        self.top_k = 50
        
        # System prompts from environment
        self.positive_system_prompt = os.getenv(
            "POSITIVE_SYSTEM_PROMPT_CHAT", 
            "You are a helpful and friendly AI assistant."
        )
        self.negative_system_prompt = os.getenv(
            "NEGATIVE_SYSTEM_PROMPT_CHAT",
            "Do not provide harmful, illegal, or inappropriate content."
        )
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the chat model and tokenizer.
        
        Args:
            model_path: Optional path to local model, otherwise uses default
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if model_path:
                self.model_name = model_path
            
            logger.info(f"Loading chat model: {self.model_name}")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            model_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "device_map": "auto" if device == "cuda" else None,
                "trust_remote_code": True
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                **model_kwargs
            )
            
            # Create generation pipeline
            self.generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            self.is_loaded = True
            logger.info("Chat model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load chat model: {e}")
            self.is_loaded = False
            return False
    
    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        Format conversation messages for the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted conversation string
        """
        formatted_messages = []
        
        # Add system prompt
        system_prompt = f"{self.positive_system_prompt}\n\n{self.negative_system_prompt}"
        formatted_messages.append(f"System: {system_prompt}")
        
        # Add conversation messages
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                formatted_messages.append(f"User: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
        
        # Add prompt for assistant response
        formatted_messages.append("Assistant:")
        
        return "\n".join(formatted_messages)
    
    def generate_response(self, conversation_context: str, 
                         max_new_tokens: int = 256) -> Dict[str, Any]:
        """
        Generate an AI response to the conversation.
        
        Args:
            conversation_context: Formatted conversation context
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Dictionary with generation results
        """
        if not self.is_loaded:
            return {
                "success": False,
                "error": "Model not loaded",
                "response": "",
                "metadata": {}
            }
        
        try:
            logger.info("Generating AI response...")
            start_time = datetime.now()
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            # Generate response
            outputs = self.generation_pipeline(
                conversation_context,
                **generation_kwargs
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"] if outputs else ""
            
            # Clean up the response
            generated_text = generated_text.strip()
            
            # Remove any remaining prompt artifacts
            if generated_text.startswith("Assistant:"):
                generated_text = generated_text[10:].strip()
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "response": generated_text,
                "error": None,
                "metadata": {
                    "generation_time": generation_time,
                    "max_new_tokens": max_new_tokens,
                    "temperature": self.temperature,
                    "model_name": self.model_name
                }
            }
            
            logger.info(f"Response generated successfully in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "",
                "metadata": {}
            }
    
    def chat_with_session(self, session_id: str, user_message: str, 
                         user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a chat message within a session context.
        
        Args:
            session_id: Chat session ID
            user_message: User's message
            user_id: Optional user ID
            
        Returns:
            Complete chat result with orchestrator and moderator integration
        """
        # Import here to avoid circular imports
        from orchestrator.chat import get_chat_orchestrator, MessageType
        from orchestrator.chatmod import moderate_user_message, filter_ai_response
        
        result = {
            "success": False,
            "user_message": user_message,
            "ai_response": "",
            "session_id": session_id,
            "moderation": {},
            "filtering": {},
            "error": None
        }
        
        try:
            orchestrator = get_chat_orchestrator()
            
            # Check if session exists
            session = orchestrator.get_chat_session(session_id)
            if not session:
                result["error"] = f"Session {session_id} not found"
                return result
            
            # Moderate user message
            moderation_result = moderate_user_message(user_message, user_id)
            result["moderation"] = moderation_result
            
            if not moderation_result.get("approved", False):
                result["error"] = "Message rejected by moderation"
                return result
            
            # Add user message to session
            user_msg = orchestrator.add_message(
                session_id, MessageType.USER, user_message, user_id
            )
            
            # Get conversation context
            messages = orchestrator.get_session_messages(session_id, limit=20)
            
            # Format messages for AI
            formatted_messages = []
            for msg in messages:
                if msg.message_type == MessageType.USER:
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif msg.message_type == MessageType.ASSISTANT:
                    formatted_messages.append({"role": "assistant", "content": msg.content})
            
            conversation_context = self.format_conversation(formatted_messages)
            
            # Generate AI response
            generation_result = self.generate_response(conversation_context)
            
            if not generation_result["success"]:
                result["error"] = generation_result["error"]
                return result
            
            ai_response = generation_result["response"]
            
            # Filter AI response
            filtering_result = filter_ai_response(ai_response)
            result["filtering"] = filtering_result
            
            if filtering_result.get("approved", False):
                final_response = filtering_result.get("filtered_response", ai_response)
                
                # Add AI response to session
                orchestrator.add_message(
                    session_id, MessageType.ASSISTANT, final_response
                )
                
                result["ai_response"] = final_response
                result["success"] = True
                
                logger.info(f"Chat completed successfully for session {session_id}")
            else:
                result["error"] = "AI response rejected by filtering"
                result["ai_response"] = "I apologize, but I cannot provide a response to that."
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            result["error"] = str(e)
        
        return result


# Global chat AI instance
_chat_ai = None

def get_chat_ai() -> ChatAI:
    """Get the global chat AI instance."""
    global _chat_ai
    if _chat_ai is None:
        _chat_ai = ChatAI()
    return _chat_ai

def load_chat_model(model_path: Optional[str] = None) -> bool:
    """Load the chat model."""
    try:
        logger.info("Loading chat model...")
        chat_ai = get_chat_ai()
        success = chat_ai.load_model(model_path)
        if success:
            logger.info("Chat model loaded successfully.")
        else:
            logger.error("Failed to load chat model.")
        return success
    except Exception as e:
        logger.error(f"Error loading chat model: {e}")
        return False

def chat_completion(session_id: str, message: str, 
                   user_id: Optional[str] = None) -> Dict[str, Any]:
    """Complete chat interaction with moderation and orchestration."""
    return get_chat_ai().chat_with_session(session_id, message, user_id)

def generate_text_response(prompt: str, max_tokens: int = 256) -> str:
    """Generate a simple text response without session context."""
    chat_ai = get_chat_ai()
    if not chat_ai.is_loaded:
        return "Chat model not loaded"
    
    result = chat_ai.generate_response(prompt, max_tokens)
    return result.get("response", "Error generating response")