"""
Chat API endpoints for Personal Chatter application.

This module provides REST API endpoints for chat operations including:
- Creating new chats
- Loading existing chats
- Sending messages
- Receiving streaming responses
- Closing chats
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Path, Query
from pydantic import BaseModel
from starlette.websockets import WebSocketState
import asyncio
import json

from services.chat.chat_manager import start_new_chat, process_user_message, end_chat_session
from services.database.chats import open_chat, get_chat_history, list_user_chats
from api.auth import get_current_user, oauth2_scheme, verify_token
from services.communication.websocket import get_websocket_progress_manager

# Token validation adapter to convert between JWT auth and session token
async def get_session_token(current_user = Depends(get_current_user)) -> str:
    """
    Adapter function to convert the current user from JWT token to a session token.
    
    In a real implementation, this would look up or create a session token for the user
    in your database using the username from the JWT token.
    
    Args:
        current_user: The user data from the JWT token
    
    Returns:
        str: The session token for the user
    """
    # For now, we'll use a simple mock implementation
    # In a real system, you would look up or create a proper session token
    username = current_user.get("username", "unknown")
    # This is a placeholder. In reality, you'd retrieve a real session token from your auth system
    return f"mock_session_{username}_token"

logger = logging.getLogger(__name__)

# Define data models for API requests/responses
class NewChatRequest(BaseModel):
    """Request model for creating a new chat."""
    chat_name: Optional[str] = None

class NewChatResponse(BaseModel):
    """Response model for created chat."""
    chat_id: str
    chat_name: str

class ChatMessage(BaseModel):
    """A chat message from user to AI."""
    message: str
    thoughts: Optional[str] = None

class ChatResponse(BaseModel):
    """Response model for chat messages."""
    message: str
    
class ChatHistoryItem(BaseModel):
    """A single item in chat history."""
    role: str
    content: str
    timestamp: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""
    chat_id: str
    chat_name: str
    messages: List[ChatHistoryItem]

class CloseChatRequest(BaseModel):
    """Request model for closing a chat."""
    username: str
    password_hash: str

class ChatListItem(BaseModel):
    """A chat item in the list of user chats."""
    chat_id: str
    chat_name: str
    last_modified: str

class ChatListResponse(BaseModel):
    """Response model for listing user chats."""
    chats: List[ChatListItem]

# Create router
chat_router = APIRouter(prefix="/chat", tags=["chat"])

# Connection manager for websockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, chat_id: str):
        await websocket.accept()
        if chat_id not in self.active_connections:
            self.active_connections[chat_id] = []
        self.active_connections[chat_id].append(websocket)
        logger.info(f"WebSocket client connected to chat {chat_id}. Total connections for this chat: {len(self.active_connections[chat_id])}")

    def disconnect(self, websocket: WebSocket, chat_id: str):
        if chat_id in self.active_connections:
            if websocket in self.active_connections[chat_id]:
                self.active_connections[chat_id].remove(websocket)
                logger.info(f"WebSocket client disconnected from chat {chat_id}. Remaining connections: {len(self.active_connections[chat_id])}")
            if not self.active_connections[chat_id]:
                del self.active_connections[chat_id]
                logger.info(f"No more connections for chat {chat_id}")

    async def send_token_update(self, chat_id: str, token: str):
        """Send a token update to all connected clients for a specific chat."""
        if chat_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[chat_id]:
                if connection.client_state != WebSocketState.DISCONNECTED:
                    try:
                        await connection.send_text(json.dumps({"token": token}))
                    except RuntimeError:
                        dead_connections.append(connection)
                else:
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for dead in dead_connections:
                try:
                    self.disconnect(dead, chat_id)
                except:
                    pass

    async def send_complete_response(self, chat_id: str, response: str):
        """Send a complete response to all connected clients for a specific chat."""
        if chat_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[chat_id]:
                if connection.client_state != WebSocketState.DISCONNECTED:
                    try:
                        await connection.send_text(json.dumps({"complete": response}))
                    except RuntimeError:
                        dead_connections.append(connection)
                else:
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for dead in dead_connections:
                try:
                    self.disconnect(dead, chat_id)
                except:
                    pass

    async def send_progress_update(self, chat_id: str, progress_data: dict):
        """Send a progress update to all connected clients for a specific chat."""
        if chat_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[chat_id]:
                if connection.client_state != WebSocketState.DISCONNECTED:
                    try:
                        await connection.send_text(json.dumps({"progress": progress_data}))
                    except RuntimeError:
                        dead_connections.append(connection)
                else:
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for dead in dead_connections:
                try:
                    self.disconnect(dead, chat_id)
                except:
                    pass

# Create connection manager instance
manager = ConnectionManager()

# Register with the global progress manager
get_websocket_progress_manager().register_connection_manager(manager)

@chat_router.post(
    "/new",
    response_model=NewChatResponse,
    summary="Create a new chat",
    status_code=201,
)
async def create_new_chat(
    request: NewChatRequest, 
    session_token: str = Depends(get_session_token)
):
    """
    Create a new chat session.
    
    Args:
        request: Optional chat name
        session_token: User's session token
        
    Returns:
        Chat ID and name of the new chat
        
    Raises:
        HTTPException: If chat creation fails
    """
    chat_id = start_new_chat(session_token, request.chat_name)
    if not chat_id:
        raise HTTPException(
            status_code=500, 
            detail="Failed to create new chat"
        )
    
    return NewChatResponse(
        chat_id=chat_id,
        chat_name=request.chat_name or f"Chat {chat_id}"
    )

@chat_router.get(
    "/list",
    response_model=ChatListResponse,
    summary="List all user chats",
    status_code=200,
)
async def get_user_chats(session_token: str = Depends(get_session_token)):
    """
    List all chats for the authenticated user.
    
    Args:
        session_token: User's session token
        
    Returns:
        List of chat IDs, names, and last modified timestamps
        
    Raises:
        HTTPException: If chat listing fails
    """
    chats_data = list_user_chats(session_token)
    if chats_data is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve chat list"
        )
    
    chat_list = []
    for chat_data in chats_data:
        chat_list.append(ChatListItem(
            chat_id=str(chat_data[0]),
            chat_name=chat_data[1],
            last_modified=chat_data[2].isoformat() if chat_data[2] else None
        ))
    
    return ChatListResponse(chats=chat_list)

@chat_router.get(
    "/{chat_id}",
    response_model=ChatHistoryResponse,
    summary="Get chat history",
    status_code=200,
)
async def get_chat(
    chat_id: str = Path(..., description="ID of the chat to retrieve"),
    session_token: str = Depends(get_session_token)
):
    """
    Get the history for a specific chat.
    
    Args:
        chat_id: ID of the chat to retrieve
        session_token: User's session token
        
    Returns:
        Chat ID, name, and message history
        
    Raises:
        HTTPException: If chat retrieval fails or chat is not found
    """
    # Get decrypted chat XML
    chat_xml = open_chat(chat_id, session_token)
    if not chat_xml:
        raise HTTPException(
            status_code=404,
            detail=f"Chat {chat_id} not found or access denied"
        )
    
    # Get the chat history
    chat_history = get_chat_history(chat_id, session_token)
    
    # Convert to response format
    messages = []
    for msg in chat_history:
        messages.append(ChatHistoryItem(
            role=msg["role"],
            content=msg["content"],
            # Timestamp would be extracted from XML if available
            timestamp=None 
        ))
    
    # Get chat name - in a real implementation, you would extract this from the chat metadata
    # For now, we'll use a placeholder
    chat_name = f"Chat {chat_id}"
    
    return ChatHistoryResponse(
        chat_id=chat_id,
        chat_name=chat_name,
        messages=messages
    )

@chat_router.post(
    "/{chat_id}/close",
    response_model=dict,
    summary="Close a chat session",
    status_code=200,
)
async def close_chat(
    request: CloseChatRequest,
    chat_id: str = Path(..., description="ID of the chat to close"),
    session_token: str = Depends(get_session_token)
):
    """
    Close a chat session with proper encryption.
    
    Args:
        request: Username and password hash for verification
        chat_id: ID of the chat to close
        session_token: User's session token
        
    Returns:
        Success status
        
    Raises:
        HTTPException: If chat closure fails
    """
    success = end_chat_session(
        chat_id,
        session_token,
        request.username,
        request.password_hash
    )
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to close chat {chat_id}"
        )
    
    return {"success": True, "message": f"Chat {chat_id} closed successfully"}

@chat_router.websocket("/{chat_id}/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    chat_id: str,
    token: str = Query(..., description="Session token")
):
    """
    WebSocket endpoint for real-time chat interaction.
    
    Args:
        websocket: WebSocket connection
        chat_id: ID of the chat
        token: Session token for authentication
        
    The websocket connection will:
    1. Accept connections with valid session token
    2. Receive messages from the client
    3. Stream token updates during response generation
    4. Send a final complete message when done    """
    # Validate token (simplified here, should be more robust)
    try:
        # Validate the token against our JWT auth system
        payload = verify_token(token)
        if not payload:
            await websocket.close(code=4000, reason="Invalid authentication token")
            return
            
        # Convert JWT token to a session token
        # In a real implementation, this would look up a session in your database
        session_token = f"mock_session_{payload.username}_token"
    except Exception as e:
        logger.error(f"Error validating token: {e}")
        await websocket.close(code=4000, reason="Authentication error")
        return
    
    # Accept the connection and register it
    try:
        await manager.connect(websocket, chat_id)
        logger.info(f"WebSocket connection established for chat {chat_id}")
    except Exception as e:
        logger.error(f"Error accepting WebSocket connection: {e}")
        return
    
    # Main message handling loop
    try:
        while True:
            # Wait for a message from the client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if "message" not in message_data:
                await websocket.send_text(json.dumps({"error": "Invalid message format"}))
                continue
            
            # Extract message content
            message = message_data["message"]
            thoughts = message_data.get("thoughts")
            
            logger.info(f"Received message for chat {chat_id}: {message[:50]}...")
            
            # In a real implementation, you would:
            # 1. Stream tokens during generation
            # 2. Send the complete response when done
            
            # Placeholder for now - we'll call process_user_message
            try:                # This is synchronous in the current implementation
                # In a real-world setup, this should be async or run in a thread pool
                response = process_user_message(chat_id, session_token, message, thoughts)
                
                # Send complete response
                await manager.send_complete_response(chat_id, response or "Error: Failed to generate response")
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_text(json.dumps({"error": f"Error processing message: {str(e)}"}))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from chat {chat_id}")
        manager.disconnect(websocket, chat_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, chat_id)