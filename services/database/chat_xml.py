import logging
import datetime
import os
import xml.etree.ElementTree as ET
from typing import List

logger = logging.getLogger(__name__)

def format_chat_message(role, message, timestamp=None, thoughts=None, sources=None):
    """Format a chat message in XML format"""
    if timestamp is None:
        timestamp = datetime.datetime.now().isoformat()
    
    thoughts_attr = f' thoughts="{thoughts}"' if thoughts else ''
    sources_attr = f' sources="{sources}"' if sources else ''
    
    return f'<{role} timestamp="{timestamp}"{thoughts_attr}{sources_attr}>{message}</{role}>'

def get_system_prompt():
    """Get the system prompt for new chats"""
    return f"{os.environ.get('POSITIVE_SYSTEM_PROMPT_CHAT')}\n{os.environ.get('NEGATIVE_SYSTEM_PROMPT_CHAT')}"

def create_empty_chat_xml():
    """Create an empty chat XML structure with system prompt"""
    timestamp = datetime.datetime.now().isoformat()
    system_prompt = get_system_prompt()
    system_message = format_chat_message("system", system_prompt, timestamp=timestamp)
    return f'<chat created="{timestamp}">{system_message}</chat>'

def append_message_to_xml(chat_xml, new_message_xml):
    """
    Append a new message to the chat XML structure.
    """
    try:
        root = ET.fromstring(chat_xml)
        # Insert the new message XML before the closing </chat> tag
        new_message_element = ET.fromstring(new_message_xml)
        root.append(new_message_element)
        return ET.tostring(root, encoding='unicode')
    except Exception as e:
        logger.error(f"Error appending message to XML: {e}")
        return chat_xml

def format_and_append_message(chat_xml, role, message, thoughts=None, sources=None):
    """Format a new message and append it to the chat XML."""
    new_message_xml = format_chat_message(role, message, thoughts=thoughts, sources=sources)
    return append_message_to_xml(chat_xml, new_message_xml)

def count_user_messages_in_chat(chat_xml):
    """
    Count the number of user messages in a chat XML.
    """
    try:
        root = ET.fromstring(chat_xml)
        user_messages = root.findall("user")
        return len(user_messages)
    except Exception as e:
        logger.error(f"Error counting messages in chat XML: {e}")
        return 0

def parse_chat_xml_to_history(chat_xml: str) -> list[dict[str, str]]:
    """
    Parse chat XML into a list of chat history messages.
    
    Args:
        chat_xml (str): The XML content of the chat
        
    Returns:
        list[dict[str, str]]: List of messages with 'role' and 'content' keys
    """
    try:
        root = ET.fromstring(chat_xml)
        chat_history = []
        
        # Extract all message elements (system, user, agent/assistant)
        for element in root:
            role = element.tag
            content = element.text or ""
            
            # Convert 'agent' role to 'assistant' for consistency with LLM interface
            if role == "agent":
                role = "assistant"
            
            # Skip elements that don't represent messages
            if role not in ["system", "user", "assistant"]:
                continue
                
            chat_history.append({"role": role, "content": content})
        
        logger.debug(f"Parsed {len(chat_history)} messages from chat XML")
        return chat_history
        
    except Exception as e:
        logger.error(f"Error parsing chat XML to history: {e}", exc_info=True)
        return []
