"""
AI Backend Data Models

This module contains Pydantic models for all AI backend modules.
These models ensure type safety and consistent data structures across
the orchestrator and test layers.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


# ============================================================================
# Common Enums and Base Models
# ============================================================================

class AIStatus(str, Enum):
    """Status of AI model/service"""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    LOADED = "loaded"
    PROCESSING = "processing"
    ERROR = "error"
    UNLOADED = "unloaded"


class DeviceType(str, Enum):
    """Device types for model execution"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class ModelLoadConfig(BaseModel):
    """Base configuration for model loading"""
    model_name: str
    model_path: str
    device: DeviceType = DeviceType.CPU
    torch_dtype: str = "float32"
    enable_offloading: bool = True


class BaseAIResponse(BaseModel):
    """Base response model for all AI operations"""
    success: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ============================================================================
# Chat AI Models
# ============================================================================

class MessageRole(str, Enum):
    """Chat message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Individual chat message"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationHistory(BaseModel):
    """Collection of chat messages"""
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    
    def to_dict_list(self) -> List[Dict[str, str]]:
        """Convert to simple dict format expected by ChatAI"""
        return [{"role": msg.role.value, "content": msg.content} for msg in self.messages]


class ChatGenerationParams(BaseModel):
    """Parameters for chat text generation"""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    repetition_penalty: float = Field(default=1.1, ge=0.5, le=2.0)
    max_new_tokens: int = Field(default=8192, ge=1, le=32768)


class ChatRequest(BaseModel):
    """Request for chat generation"""
    conversation_history: ConversationHistory
    positive_system_prompt: Optional[str] = None
    negative_system_prompt: Optional[str] = None
    generation_params: Optional[ChatGenerationParams] = None
    max_new_tokens: Optional[int] = None


class ChatResponse(BaseAIResponse):
    """Response from chat generation"""
    response: str = ""
    conversation_id: Optional[str] = None
    model_name: Optional[str] = None
    device_used: Optional[DeviceType] = None


class ChatModelStatus(BaseModel):
    """Status of chat AI model"""
    status: AIStatus
    model_name: str
    model_path: str
    is_loaded: bool = False
    current_device: Optional[DeviceType] = None
    memory_usage_mb: Optional[float] = None
    last_generation_time: Optional[datetime] = None


# ============================================================================
# Text2Img AI Models
# ============================================================================

class Text2ImgGenerationParams(BaseModel):
    """Parameters for text-to-image generation"""
    guidance_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    height: int = Field(default=1024, ge=256, le=2048)
    width: int = Field(default=1024, ge=256, le=2048)
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None


class Text2ImgRequest(BaseModel):
    """Request for text-to-image generation"""
    text_prompt: str = Field(min_length=1, max_length=2000)
    generation_params: Optional[Text2ImgGenerationParams] = None


class Text2ImgResponse(BaseAIResponse):
    """Response from text-to-image generation"""
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None  # Base64 encoded
    prompt_used: Optional[str] = None
    negative_prompt_used: Optional[str] = None
    generation_params: Optional[Text2ImgGenerationParams] = None
    model_name: Optional[str] = None
    device_used: Optional[DeviceType] = None


# ============================================================================
# Img2Img AI Models  
# ============================================================================

class ControlInput(BaseModel):
    """Control inputs for img2img generation"""
    type: str  # "canny", "depth", "pose", etc.
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    data: Any  # Image data or processed control map


class Img2ImgGenerationParams(BaseModel):
    """Parameters for image-to-image generation"""
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    seed: Optional[int] = None


class Img2ImgRequest(BaseModel):
    """Request for image-to-image generation"""
    text_prompt: str = Field(min_length=1, max_length=2000)
    reference_images: List[str]  # Paths to reference images
    control_inputs: Optional[Dict[str, ControlInput]] = None
    generation_params: Optional[Img2ImgGenerationParams] = None


class Img2ImgResponse(BaseAIResponse):
    """Response from image-to-image generation"""
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None
    prompt_used: Optional[str] = None
    reference_images_used: List[str] = Field(default_factory=list)
    generation_params: Optional[Img2ImgGenerationParams] = None
    model_name: Optional[str] = None
    device_used: Optional[DeviceType] = None


# ============================================================================
# Voice Input AI Models
# ============================================================================

class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class VoiceSegment(BaseModel):
    """Individual voice segment from diarization"""
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    speaker_id: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    @field_validator('end_time')
    @classmethod
    def end_after_start(cls, v, info):
        if 'start_time' in info.data and v <= info.data['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v


class TranscriptionResult(BaseModel):
    """Result of voice transcription"""
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    language: Optional[str] = None
    segments: List[VoiceSegment] = Field(default_factory=list)


class VoiceInRequest(BaseModel):
    """Request for voice input processing"""
    audio_data: bytes
    audio_format: AudioFormat = AudioFormat.WAV
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    enable_diarization: bool = True
    enable_vad: bool = True


class VoiceInResponse(BaseAIResponse):
    """Response from voice input processing"""
    transcription: Optional[TranscriptionResult] = None
    audio_length_seconds: Optional[float] = None
    processing_stages: Dict[str, float] = Field(default_factory=dict)  # Stage -> time taken


# ============================================================================
# Voice Output AI Models
# ============================================================================

class VoiceStyle(str, Enum):
    """Available voice styles"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    CALM = "calm"
    ANDROGYNOUS = "androgynous"


class VoiceOutGenerationParams(BaseModel):
    """Parameters for voice output generation"""
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=0.0, ge=-1.0, le=1.0)
    energy: float = Field(default=1.0, ge=0.5, le=2.0)
    style: VoiceStyle = VoiceStyle.NEUTRAL
    voice_style: str = "androgynous"


class VoiceOutRequest(BaseModel):
    """Request for voice output generation"""
    text: str = Field(min_length=1, max_length=5000)
    output_format: AudioFormat = AudioFormat.WAV
    generation_params: Optional[VoiceOutGenerationParams] = None


class VoiceOutResponse(BaseAIResponse):
    """Response from voice output generation"""
    audio_data: Optional[bytes] = None
    audio_path: Optional[str] = None
    audio_length_seconds: Optional[float] = None
    sample_rate: int = Field(default=22050)
    generation_params: Optional[VoiceOutGenerationParams] = None
    model_name: Optional[str] = None


# ============================================================================
# Sentiment Analysis Models
# ============================================================================

class SentimentLabel(str, Enum):
    """Sentiment analysis labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentResult(BaseModel):
    """Result of sentiment analysis"""
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    scores: Dict[str, float] = Field(default_factory=dict)  # All label scores


class SentimentRequest(BaseModel):
    """Request for sentiment analysis"""
    text: str = Field(min_length=1, max_length=10000)
    return_all_scores: bool = False


class SentimentResponse(BaseAIResponse):
    """Response from sentiment analysis"""
    sentiment: Optional[SentimentResult] = None
    model_name: Optional[str] = None
