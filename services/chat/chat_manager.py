"""
Manages chat sessions, including creation, message processing, and termination.

This module serves as the high-level interface for chat functionality,
coordinating between the database layer and the language model interface.
"""
import logging
import datetime
import time
from services.database.chats import (
    create_chat, 
    open_chat, 
    add_message_to_chat,
    close_chat,
    get_chat_history
)
from services.chat.llm_interface import send_query
from services.communication.websocket import get_websocket_progress_manager

logger = logging.getLogger(__name__)

def start_new_chat(session_token, chat_name=None):
    """
    Start a new chat session with initial encryption setup.
    
    Creates a new chat entry in the database with proper encryption.
    If no chat_name is provided, a timestamp-based name is generated.
    
    Args:
        session_token (str): The user's active session token for authentication
        chat_name (str, optional): Name for the new chat. Defaults to timestamp-based name.
    
    Returns:
        str: The chat_id if successful
        None: If chat creation fails
    """
    logger.info("Starting new chat session")
    if not chat_name:
        chat_name = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        logger.debug(f"Using generated chat name: {chat_name}")
    
    # Create the chat in the database
    chat_id = create_chat(chat_name, session_token)
    if not chat_id:
        logger.error("Failed to create new chat")
        return None
    
    logger.info(f"Successfully created new chat with ID: {chat_id}")    
    return chat_id

def process_user_message(chat_id, session_token, message, thoughts=None):
    """
    Process a user message, add it to the chat, and get AI response.
    
    This function handles the complete flow of:
    1. Checking for command intents (image generation, search, etc.)
    2. Adding the user's message to the database
    3. Retrieving the current chat history
    4. Sending the message + history to the LLM
    5. Storing the AI's response in the database
    
    Args:
        chat_id (str): The ID of the current chat
        session_token (str): The user's active session token
        message (str): The user's message content
        thoughts (str, optional): Any additional metadata/thoughts for the message
    
    Returns:
        str: The AI's response message if successful
        None: If any part of the process fails
    """
    from services.config import get_config
    from services.command_processor import preprocess_message, CommandIntent
    from services.database.sessions import verify_session_token
    
    logger.info(f"Processing user message for chat {chat_id}")
    logger.debug(f"Message content (first 50 chars): {message[:50]}...")
    
    # Get configuration
    config = get_config()
    preprocessing_enabled = config.get("command_preprocessing", {}).get("enabled", True)
    
    # Check if we should look for special commands
    command_response = None
    if preprocessing_enabled:
        logger.debug("Command preprocessing is enabled, checking for commands")
        
        # Check for command intents
        intent, params = preprocess_message(
            message, 
            use_main_model=config.get("command_preprocessing", {}).get("use_main_model", True)
        )
        
        logger.info(f"Command intent detected: {intent.name}")
        
        # Get user ID for commands that need it
        user_id = verify_session_token(session_token)
        
        # Process commands based on intent
        if intent != CommandIntent.NONE:
            command_response = _process_command_intent(intent, params, user_id, chat_id, session_token)
    
    # Add the user message to the chat
    success = add_message_to_chat(
        chat_id, 
        session_token, 
        "user", 
        message, 
        thoughts=thoughts
    )
    
    if not success:
        logger.error(f"Failed to add user message to chat {chat_id}")
        return None
    
    # If we have a command response, use that instead of calling the LLM
    if command_response:
        logger.info(f"Using command response (length: {len(command_response)})")
        
        # Add the command response to the chat
        success = add_message_to_chat(
            chat_id,
            session_token,
            "agent",
            command_response
        )
        
        if not success:
            logger.error(f"Failed to add command response to chat {chat_id}")
            return None
        
        logger.info("Successfully processed command and stored response")
        return command_response
        
    # Generate AI response
    try:
        # Get the current chat history
        logger.debug(f"Retrieving chat history for chat {chat_id}")
        chat_history = get_chat_history(chat_id, session_token)
        logger.info(f"Retrieved chat history with {len(chat_history)} messages")
        
        # Send query to the LLM with chat history
        logger.debug("Sending message to LLM with chat history")
        ai_response, updated_history = send_query(message, chat_history)
        logger.info(f"Generated AI response (length: {len(ai_response)})")
        
        # Add the AI response to the chat
        logger.debug("Storing AI response in database")
        success = add_message_to_chat(
            chat_id,
            session_token,
            "agent",
            ai_response
        )
        
        if not success:
            logger.error(f"Failed to add AI response to chat {chat_id}")
            return None
        
        logger.info("Successfully processed user message and stored response")
        return ai_response
    except Exception as e:
        logger.error(f"Error processing message with AI: {e}", exc_info=True)
        return None
        
def _process_command_intent(intent, params, user_id, chat_id, session_token):
    """
    Process a command intent and generate a response.
    
    Args:
        intent: The detected command intent
        params: Parameters for the command
        user_id: The user ID
        chat_id: The chat ID
        session_token: The user's session token
        
    Returns:
        str: The response to the command, or None if no response was generated    """
    from services.command_processor import CommandIntent
    from services.config import get_config
    
    config = get_config()
    
    try:
        # Image Generation
        if intent == CommandIntent.GENERATE_IMAGE:
            if not config.get("command_preprocessing", {}).get("image_generation_enabled", True):
                return "Image generation is currently disabled."
            
            from services.ai.image_generation import generate_image
              # Extract image prompt
            image_prompt = params.get("image_prompt", "")
            if not image_prompt:
                return "I couldn't understand what kind of image you want me to generate. Please provide more details."
            
            # Create session ID for progress tracking
            import uuid
            session_id = f"img_{chat_id}_{int(time.time())}"
            
            # Define progress callback for sending updates via websocket (if available)
            def progress_callback(step, total_steps, progress, elapsed_time):
                # Send progress update via websocket
                progress_data = {
                    "type": "image_generation",
                    "session_id": session_id,
                    "step": step,
                    "total_steps": total_steps,
                    "progress": progress,
                    "elapsed_time": elapsed_time,
                    "message": f"Generating image... {progress:.1f}% complete"
                }
                
                # Send via websocket manager
                get_websocket_progress_manager().send_progress_update_sync(chat_id, progress_data)
                
                # Also log for server monitoring
                logger.info(f"Image generation progress: {progress:.1f}% (step {step}/{total_steps}, elapsed: {elapsed_time:.1f}s)")
            
            # Generate the image with progress tracking
            image_path, image_url = generate_image(
                prompt=image_prompt,
                session_id=session_id,
                progress_callback=progress_callback
            )
            
            if not image_path:
                return "I was unable to generate the requested image. Please try again with a different description."
            
            return f"I've generated the image you requested:\n\nImage: {image_url}\n\nThe image shows {image_prompt}"
        
        # Online Search
        elif intent == CommandIntent.ONLINE_SEARCH:
            if not config.get("command_preprocessing", {}).get("online_search_enabled", True):
                return "Online search is currently disabled."
            
            from services.online_search import search_online, format_search_results_for_llm
            
            # Extract search query
            search_query = params.get("search_query", "")
            if not search_query:
                return "I couldn't understand what you want me to search for. Please specify your query."
            
            # Perform the search
            search_results = search_online(search_query)
            
            if not search_results:
                return f"I searched online for '{search_query}' but couldn't find any relevant information."
            
            # Format results for the LLM
            formatted_results = format_search_results_for_llm(search_results)
              # Use the LLM to generate a summary response
            from services.chat.llm_interface import send_query
            system_message = {"role": "system", "content": "You are a helpful assistant summarizing online search results. Be factual and concise."}
            response, _ = send_query(
                f"Summarize these search results about '{search_query}':\n\n{formatted_results}",
                [system_message]
            )
            
            return response
        
        # Store User Information
        elif intent == CommandIntent.STORE_USER_INFO:
            if not config.get("command_preprocessing", {}).get("store_user_info_enabled", True):
                return "Storing user information is currently disabled."
            
            from services.user_memory import store_user_information, extract_user_information
            
            # Extract user information from the message
            user_info = params.get("user_info", "")
            if not user_info:
                return "I couldn't understand what personal information you want me to remember."
            
            # Extract structured information
            extracted_info = extract_user_information(user_info)
            
            if not extracted_info:
                return "I couldn't extract any specific information from your message. Please provide details like your name, contact information, etc."
            
            # Store the information
            success = store_user_information(user_id, extracted_info)
            
            if success:
                # Generate a response confirming what was stored
                items = []
                for key, value in extracted_info.items():
                    if key != "_last_updated":
                        items.append(f"{key}: {value}")
                        
                items_str = "\n- ".join(items)
                return f"I've stored the following information about you:\n- {items_str}\n\nI'll remember this for our future conversations."
            else:
                return "I was unable to store your information. Please try again later."
        
        # Store Memory Item
        elif intent == CommandIntent.STORE_MEMORY:
            if not config.get("command_preprocessing", {}).get("store_memory_enabled", True):
                return "Storing memory items is currently disabled."
            
            from services.user_memory import store_memory_item
            
            # Extract memory content
            memory_content = params.get("memory_content", "")
            if not memory_content:
                return "I couldn't understand what you want me to remember. Please provide more details."
            
            # Store the memory item
            success = store_memory_item(user_id, memory_content)
            
            if success:
                return f"I've stored this information in my memory. You can ask me about it later, and I'll do my best to recall it."
            else:
                return "I was unable to store this information. Please try again later."
        
        # Search Local Documents
        elif intent == CommandIntent.SEARCH_LOCAL:
            if not config.get("command_preprocessing", {}).get("local_search_enabled", True):
                return "Local document search is currently disabled."
            
            from services.document_search import search_documents, format_search_results_for_llm
            
            # Extract search query
            local_query = params.get("local_query", "")
            if not local_query:
                return "I couldn't understand what you want me to search for in your documents. Please specify your query."
            
            # Perform the search
            search_results = search_documents(user_id, local_query)
            
            if not search_results:
                return f"I searched your documents for '{local_query}' but couldn't find any relevant information."
            
            # Format results for the LLM
            formatted_results = format_search_results_for_llm(search_results)
              # Use the LLM to generate a summary response
            from services.chat.llm_interface import send_query
            system_message = {"role": "system", "content": "You are a helpful assistant summarizing document search results. Be factual and concise."}
            response, _ = send_query(
                f"Summarize these document search results about '{local_query}':\n\n{formatted_results}",
                [system_message]
            )
            
            return response
    
    except Exception as e:
        logger.error(f"Error processing command intent {intent.name}: {e}", exc_info=True)
        return f"I encountered an error while processing your request: {str(e)}"
    
    return None

def end_chat_session(chat_id, session_token, username, password_hash):
    """
    End a chat session, re-encrypt it, and secure the key.
    
    This function securely closes a chat session, potentially performing
    additional security measures like key rotation or enhanced encryption.
    
    Args:
        chat_id (str): The ID of the chat to close
        session_token (str): The user's active session token
        username (str): The user's username for additional verification
        password_hash (str): Hash of the user's password for key derivation
        
    Returns:
        bool: True if the chat was successfully closed, False otherwise
    """
    logger.info(f"Ending chat session {chat_id}")
    try:
        result = close_chat(chat_id, session_token, username, password_hash)
        if result:
            logger.info(f"Successfully closed chat session {chat_id}")
        else:
            logger.error(f"Failed to close chat session {chat_id}")
        return result
    except Exception as e:
        logger.error(f"Error closing chat session {chat_id}: {e}", exc_info=True)
        return False
