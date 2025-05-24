"""
Database Chat Storage Module

This module handles database operations for chat sessions and messages.
It provides persistence layer for the chat orchestrator.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict

logger = logging.getLogger(__name__)

class ChatDatabase:
    """
    Database handler for chat-related operations.
    Currently uses in-memory storage but can be extended to use PostgreSQL.
    """
    
    def __init__(self):
        # In-memory storage - to be replaced with actual database
        self.sessions_table = {}
        self.messages_table = {}
        self.user_sessions_table = {}
        self.is_connected = False
        
        logger.info("Chat database initialized (in-memory mode)")
    
    def connect(self) -> bool:
        """
        Connect to the database.
        
        Returns:
            True if connected successfully
        """
        try:
            # For now, just mark as connected
            # In the future, this would establish PostgreSQL connection
            self.is_connected = True
            logger.info("Chat database connected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to chat database: {e}")
            return False
    
    def create_tables(self) -> bool:
        """
        Create necessary database tables if they don't exist.
        
        Returns:
            True if tables created successfully
        """
        try:
            # For now, initialize in-memory structures
            # In the future, this would execute CREATE TABLE statements
            
            logger.info("Chat database tables initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to create chat database tables: {e}")
            return False
    
    def save_session(self, session_data: Dict) -> bool:
        """
        Save a chat session to the database.
        
        Args:
            session_data: Dictionary containing session information
            
        Returns:
            True if saved successfully
        """
        try:
            session_id = session_data["id"]
            self.sessions_table[session_id] = session_data.copy()
            
            # Update user sessions mapping
            user_id = session_data["user_id"]
            if user_id not in self.user_sessions_table:
                self.user_sessions_table[user_id] = []
            
            if session_id not in self.user_sessions_table[user_id]:
                self.user_sessions_table[user_id].append(session_id)
            
            logger.debug(f"Saved session {session_id} to database")
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """
        Load a chat session from the database.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            Session data dictionary or None if not found
        """
        try:
            session_data = self.sessions_table.get(session_id)
            if session_data:
                logger.debug(f"Loaded session {session_id} from database")
            return session_data
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def save_message(self, message_data: Dict) -> bool:
        """
        Save a chat message to the database.
        
        Args:
            message_data: Dictionary containing message information
            
        Returns:
            True if saved successfully
        """
        try:
            message_id = message_data["id"]
            session_id = message_data["chat_session_id"]
            
            # Initialize session messages list if needed
            if session_id not in self.messages_table:
                self.messages_table[session_id] = []
            
            # Check if message already exists
            existing_msg = next(
                (msg for msg in self.messages_table[session_id] if msg["id"] == message_id),
                None
            )
            
            if existing_msg:
                # Update existing message
                idx = self.messages_table[session_id].index(existing_msg)
                self.messages_table[session_id][idx] = message_data.copy()
            else:
                # Add new message
                self.messages_table[session_id].append(message_data.copy())
            
            logger.debug(f"Saved message {message_id} to database")
            return True
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False
    
    def load_session_messages(self, session_id: str, 
                            limit: Optional[int] = None,
                            offset: int = 0) -> List[Dict]:
        """
        Load messages for a chat session from the database.
        
        Args:
            session_id: ID of the session
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            List of message dictionaries
        """
        try:
            messages = self.messages_table.get(session_id, [])
            
            # Sort by timestamp
            messages.sort(key=lambda m: m.get("timestamp", ""))
            
            # Apply offset and limit
            if offset > 0:
                messages = messages[offset:]
            
            if limit is not None:
                messages = messages[:limit]
            
            logger.debug(f"Loaded {len(messages)} messages for session {session_id}")
            return messages
        except Exception as e:
            logger.error(f"Failed to load messages for session {session_id}: {e}")
            return []
    
    def load_user_sessions(self, user_id: str) -> List[str]:
        """
        Load session IDs for a user from the database.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of session IDs
        """
        try:
            session_ids = self.user_sessions_table.get(user_id, [])
            logger.debug(f"Loaded {len(session_ids)} sessions for user {user_id}")
            return session_ids
        except Exception as e:
            logger.error(f"Failed to load sessions for user {user_id}: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its messages from the database.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Get session to find user_id
            session_data = self.sessions_table.get(session_id)
            if session_data:
                user_id = session_data["user_id"]
                
                # Remove from user sessions
                if user_id in self.user_sessions_table:
                    self.user_sessions_table[user_id] = [
                        sid for sid in self.user_sessions_table[user_id] 
                        if sid != session_id
                    ]
            
            # Remove session
            if session_id in self.sessions_table:
                del self.sessions_table[session_id]
            
            # Remove messages
            if session_id in self.messages_table:
                del self.messages_table[session_id]
            
            logger.info(f"Deleted session {session_id} from database")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            total_sessions = len(self.sessions_table)
            total_messages = sum(len(msgs) for msgs in self.messages_table.values())
            total_users = len(self.user_sessions_table)
            
            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "total_users": total_users,
                "is_connected": self.is_connected
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}


# Global database instance
_chat_database = None

def get_chat_database() -> ChatDatabase:
    """Get the global chat database instance."""
    global _chat_database
    if _chat_database is None:
        _chat_database = ChatDatabase()
    return _chat_database

def initiate_chat_database() -> bool:
    """Initialize the chat database."""
    try:
        logger.info("Initializing chat database...")
        db = get_chat_database()
        
        if not db.connect():
            logger.error("Failed to connect to chat database")
            return False
        
        if not db.create_tables():
            logger.error("Failed to create chat database tables")
            return False
        
        logger.info("Chat database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize chat database: {e}")
        return False

def test_chat_database() -> bool:
    """Test the chat database connection and functionality."""
    try:
        logger.info("Testing chat database...")
        db = get_chat_database()
        
        if not db.is_connected:
            logger.error("Chat database not connected")
            return False
        
        # Test basic operations
        test_session = {
            "id": "test-session",
            "user_id": "test-user",
            "title": "Test Session",
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        
        # Test save and load
        if not db.save_session(test_session):
            logger.error("Failed to save test session")
            return False
        
        loaded_session = db.load_session("test-session")
        if not loaded_session:
            logger.error("Failed to load test session")
            return False
        
        # Clean up test data
        db.delete_session("test-session")
        
        logger.info("Chat database test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Chat database test failed: {e}")
        return False

# Convenience functions for integration with orchestrator
def save_session_to_db(session_dict: Dict) -> bool:
    """Save a session to the database."""
    return get_chat_database().save_session(session_dict)

def save_message_to_db(message_dict: Dict) -> bool:
    """Save a message to the database."""
    return get_chat_database().save_message(message_dict)

def load_session_from_db(session_id: str) -> Optional[Dict]:
    """Load a session from the database."""
    return get_chat_database().load_session(session_id)

def load_messages_from_db(session_id: str, limit: Optional[int] = None) -> List[Dict]:
    """Load messages from the database."""
    return get_chat_database().load_session_messages(session_id, limit)