import logging
import re
import xml.etree.ElementTree as ET
from services.chat.pipeline import send_query

logger = logging.getLogger(__name__)

def extract_messages_from_xml(chat_xml):
    """
    Extract messages from chat XML format for title generation.
    Returns a list of {role, content} dictionaries.
    """
    try:
        # Parse the XML content
        root = ET.fromstring(chat_xml)
        
        messages = []
        for child in root:
            if child.tag in ["user", "agent", "system"]:
                messages.append({
                    "role": child.tag,
                    "content": child.text
                })
        
        return messages
    except ET.ParseError as e:
        logger.error(f"Error parsing chat XML: {e}")
        return []

def generate_chat_title(chat_xml):
    """
    Generate a title for a chat based on its content.
    Uses the LLM to summarize and create a title.
    """
    # Extract messages from XML
    messages = extract_messages_from_xml(chat_xml)
    
    if not messages:
        return "New Chat"
    
    # Take the first few user messages to generate the title
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"][:3]
    
    if not user_messages:
        return "New Chat"
    
    # Create a prompt for title generation
    prompt = (
        "Based on the following chat messages, generate a short, descriptive title "
        "for this conversation (maximum 50 characters):\n\n"
        f"{' '.join(user_messages)}"
    )
    
    try:
        # Use the LLM to generate a title
        title = send_query(prompt, max_tokens=50, temperature=0.7)
        
        # Clean up the title
        title = title.strip().strip('"\'')
        
        # Truncate if too long
        if len(title) > 50:
            title = title[:47] + "..."
            
        return title or "Chat"
    except Exception as e:
        logger.error(f"Error generating chat title: {e}")
        return "Chat"

def should_update_title(chat_xml):
    """
    Determine if the chat title should be updated based on message count.
    Returns True if the chat has reached a multiple of 5 messages.
    """
    try:
        # Count the number of messages
        root = ET.fromstring(chat_xml)
        user_messages = [child for child in root if child.tag == "user"]
        
        # Update title every 5 messages
        return len(user_messages) % 5 == 0 and len(user_messages) > 0
    except ET.ParseError as e:
        logger.error(f"Error parsing chat XML for title update check: {e}")
        return False
