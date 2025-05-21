import logging
from services.chat.pipeline import send_query

logger = logging.getLogger(__name__)

def generate_chat_title(messages, max_length=50):
    """
    Generate a title for a chat based on user messages.
    Uses the AI to create a concise, descriptive title.
    
    Args:
        messages: List of messages to generate title from
        max_length: Maximum length of the generated title
    
    Returns:
        A string with the generated title
    """
    if not messages:
        return "New Chat"
    
    # Take up to 3 user messages to generate the title
    content = "\n".join(messages[:3])
    
    # Create a prompt for title generation
    prompt = (
        "Based on the following conversation, generate a short, descriptive title "
        f"(maximum {max_length} characters):\n\n{content}"
    )
    
    try:
        # Use the AI to generate a title
        title = send_query(prompt, max_tokens=50, temperature=0.7)
        
        # Clean up the title
        title = title.strip().strip('"\'')
        
        # Truncate if too long
        if len(title) > max_length:
            title = title[:max_length-3] + "..."
            
        return title or "Chat"
    except Exception as e:
        logger.error(f"Error generating chat title: {e}")
        return "New Chat"

def should_update_title(message_count):
    """
    Determine if the chat title should be updated based on message count.
    Returns True if the message count is a multiple of 5.
    """
    # Update title every 5 messages, starting from message 5
    return message_count > 0 and message_count % 5 == 0
