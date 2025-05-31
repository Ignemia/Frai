"""
Backend Models Package

This package provides comprehensive Pydantic models for all backend operations.
These models ensure type safety and consistent data structures across the
orchestrator, tests, and API layers.

Usage:
    from back.models import (
        # AI Models
        ChatRequest, ChatResponse, Text2ImgRequest, Text2ImgResponse,
        VoiceInRequest, VoiceInResponse, VoiceOutRequest, VoiceOutResponse,
        
        # Database Models  
        ChatSessionDB, ChatMessageDB, UserDB, SecurityEventDB,
        
        # Preprocessor Models
        TextProcessingRequest, TextProcessingResponse,
        ImageProcessingRequest, ImageProcessingResponse,
        VoiceProcessingRequest, VoiceProcessingResponse
    )
"""

# AI Models
from .ai.models import (
    # Enums and Base Models
    AIStatus,
    DeviceType,
    ModelLoadConfig,
    BaseAIResponse,
    
    # Chat Models
    MessageRole,
    ChatMessage,
    ConversationHistory,
    ChatGenerationParams,
    ChatRequest,
    ChatResponse,
    ChatModelStatus,
    
    # Text2Img Models
    Text2ImgGenerationParams,
    Text2ImgRequest,
    Text2ImgResponse,
    
    # Img2Img Models
    ControlInput,
    Img2ImgGenerationParams,
    Img2ImgRequest,
    Img2ImgResponse,
    
    # Voice Input Models
    AudioFormat,
    VoiceSegment,
    TranscriptionResult,
    VoiceInRequest,
    VoiceInResponse,
    
    # Voice Output Models
    VoiceStyle,
    VoiceOutGenerationParams,
    VoiceOutRequest,
    VoiceOutResponse,
    
    # Sentiment Models
    SentimentLabel,
    SentimentResult,
    SentimentRequest,
    SentimentResponse,
)

# Database Models
from .database.models import (
    # Common Database Models
    DatabaseStatus,
    DatabaseConfig,
    DatabaseStats,
    
    # Chat Database Models
    MessageType,
    ChatMessageDB,
    SessionStatus,
    ChatSessionDB,
    MessageQueryParams,
    SessionQueryParams,
    ChatDatabaseRequest,
    SaveSessionRequest,
    SaveMessageRequest,
    LoadSessionRequest,
    LoadMessagesRequest,
    DeleteSessionRequest,
    ChatDatabaseResponse,
    SessionResponse,
    MessagesResponse,
    SessionListResponse,
    DatabaseStatsResponse,
    
    # User Database Models
    UserRole,
    UserStatus,
    UserPreferences,
    UserDB,
    
    # Security Database Models
    SecurityEventType,
    SecurityLevel,
    SecurityEventDB,
    APIKeyDB,
)

# Preprocessor Models
from .preprocessor.models import (
    # Common Preprocessor Models
    ProcessingStatus,
    DataFormat,
    BaseProcessingRequest,
    BaseProcessingResponse,
    
    # Text Preprocessor Models
    TextLanguage,
    TextProcessingType,
    TextCleaningOptions,
    TokenizationOptions,
    EntityType,
    ExtractedEntity,
    TextProcessingRequest,
    TextProcessingResult,
    TextProcessingResponse,
    
    # Image Preprocessor Models
    ImageFormat,
    ImageProcessingType,
    ResizeOptions,
    CropOptions,
    EnhancementOptions,
    ImageMetadata,
    DetectedObject,
    OCRResult,
    ImageProcessingRequest,
    ImageProcessingResult,
    ImageProcessingResponse,
    
    # Voice Preprocessor Models
    VoiceProcessingType,
    AudioNormalizationOptions,
    DenoiseOptions,
    VoiceActivitySegment,
    SpeakerSegment,
    AudioFeatures,
    VoiceProcessingRequest,
    VoiceProcessingResult,
    VoiceProcessingResponse,
)

# Export all models for easy importing
__all__ = [
    # AI Models
    "AIStatus",
    "DeviceType", 
    "ModelLoadConfig",
    "BaseAIResponse",
    "MessageRole",
    "ChatMessage",
    "ConversationHistory",
    "ChatGenerationParams",
    "ChatRequest",
    "ChatResponse",
    "ChatModelStatus",
    "Text2ImgGenerationParams",
    "Text2ImgRequest",
    "Text2ImgResponse",
    "ControlInput",
    "Img2ImgGenerationParams",
    "Img2ImgRequest",
    "Img2ImgResponse",
    "AudioFormat",
    "VoiceSegment",
    "TranscriptionResult",
    "VoiceInRequest",
    "VoiceInResponse",
    "VoiceStyle",
    "VoiceOutGenerationParams",
    "VoiceOutRequest",
    "VoiceOutResponse",
    "SentimentLabel",
    "SentimentResult",
    "SentimentRequest",
    "SentimentResponse",
    
    # Database Models
    "DatabaseStatus",
    "DatabaseConfig",
    "DatabaseStats",
    "MessageType",
    "ChatMessageDB",
    "SessionStatus",
    "ChatSessionDB",
    "MessageQueryParams",
    "SessionQueryParams",
    "ChatDatabaseRequest",
    "SaveSessionRequest",
    "SaveMessageRequest",
    "LoadSessionRequest",
    "LoadMessagesRequest",
    "DeleteSessionRequest",
    "ChatDatabaseResponse",
    "SessionResponse",
    "MessagesResponse",
    "SessionListResponse",
    "DatabaseStatsResponse",
    "UserRole",
    "UserStatus",
    "UserPreferences",
    "UserDB",
    "SecurityEventType",
    "SecurityLevel",
    "SecurityEventDB",
    "APIKeyDB",
    
    # Preprocessor Models
    "ProcessingStatus",
    "DataFormat",
    "BaseProcessingRequest",
    "BaseProcessingResponse",
    "TextLanguage",
    "TextProcessingType",
    "TextCleaningOptions",
    "TokenizationOptions",
    "EntityType",
    "ExtractedEntity",
    "TextProcessingRequest",
    "TextProcessingResult",
    "TextProcessingResponse",
    "ImageFormat",
    "ImageProcessingType",
    "ResizeOptions",
    "CropOptions",
    "EnhancementOptions",
    "ImageMetadata",
    "DetectedObject",
    "OCRResult",
    "ImageProcessingRequest",
    "ImageProcessingResult",
    "ImageProcessingResponse",
    "VoiceProcessingType",
    "AudioNormalizationOptions",
    "DenoiseOptions",
    "VoiceActivitySegment",
    "SpeakerSegment",
    "AudioFeatures",
    "VoiceProcessingRequest",
    "VoiceProcessingResult",
    "VoiceProcessingResponse",
]


# ============================================================================
# Convenience Factory Functions
# ============================================================================

def create_chat_request(
    messages: list,
    positive_prompt: str = None,
    negative_prompt: str = None,
    **kwargs
) -> ChatRequest:
    """Convenience function to create a ChatRequest"""
    chat_messages = [
        ChatMessage(role=MessageRole(msg["role"]), content=msg["content"])
        for msg in messages
    ]
    conversation = ConversationHistory(messages=chat_messages)
    
    return ChatRequest(
        conversation_history=conversation,
        positive_system_prompt=positive_prompt,
        negative_system_prompt=negative_prompt,
        **kwargs
    )


def create_text2img_request(prompt: str, **kwargs) -> Text2ImgRequest:
    """Convenience function to create a Text2ImgRequest"""
    return Text2ImgRequest(text_prompt=prompt, **kwargs)


def create_img2img_request(prompt: str, images: list, **kwargs) -> Img2ImgRequest:
    """Convenience function to create an Img2ImgRequest"""
    return Img2ImgRequest(
        text_prompt=prompt,
        reference_images=images,
        **kwargs
    )


def create_voice_in_request(audio_data: bytes, **kwargs) -> VoiceInRequest:
    """Convenience function to create a VoiceInRequest"""
    return VoiceInRequest(audio_data=audio_data, **kwargs)


def create_voice_out_request(text: str, **kwargs) -> VoiceOutRequest:
    """Convenience function to create a VoiceOutRequest"""
    return VoiceOutRequest(text=text, **kwargs)


def create_text_processing_request(text: str, operations: list, **kwargs) -> TextProcessingRequest:
    """Convenience function to create a TextProcessingRequest"""
    processing_types = [TextProcessingType(op) for op in operations]
    return TextProcessingRequest(
        text=text,
        processing_types=processing_types,
        **kwargs
    )


def create_chat_session_db(user_id: str, title: str = "New Chat", **kwargs) -> ChatSessionDB:
    """Convenience function to create a ChatSessionDB"""
    return ChatSessionDB(user_id=user_id, title=title, **kwargs)


def create_chat_message_db(session_id: str, role: str, content: str, **kwargs) -> ChatMessageDB:
    """Convenience function to create a ChatMessageDB"""
    return ChatMessageDB(
        session_id=session_id,
        role=MessageType(role),
        content=content,
        **kwargs
    )


def create_user_db(username: str, email: str, password_hash: str, **kwargs) -> UserDB:
    """Convenience function to create a UserDB"""
    return UserDB(
        username=username,
        email=email,
        password_hash=password_hash,
        **kwargs
    )
