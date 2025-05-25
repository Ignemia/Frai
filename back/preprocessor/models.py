"""
Preprocessor Backend Data Models

This module contains Pydantic models for all preprocessor backend modules.
These models ensure type safety and consistent data structures for
text, image, and voice preprocessing operations.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field, validator
import uuid


# ============================================================================
# Common Preprocessor Models
# ============================================================================

class ProcessingStatus(str, Enum):
    """Status of preprocessing operations"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataFormat(str, Enum):
    """Supported data formats"""
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"
    PDF = "pdf"
    DOCX = "docx"


class BaseProcessingRequest(BaseModel):
    """Base model for preprocessing requests"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseProcessingResponse(BaseModel):
    """Base model for preprocessing responses"""
    request_id: str
    status: ProcessingStatus
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Text Preprocessor Models
# ============================================================================

class TextLanguage(str, Enum):
    """Supported languages for text processing"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    AUTO_DETECT = "auto"


class TextProcessingType(str, Enum):
    """Types of text processing operations"""
    CLEAN = "clean"
    TOKENIZE = "tokenize"
    NORMALIZE = "normalize"
    EXTRACT_ENTITIES = "extract_entities"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    LANGUAGE_DETECTION = "language_detection"
    SUMMARIZE = "summarize"
    TRANSLATE = "translate"
    SPELL_CHECK = "spell_check"
    PROFANITY_FILTER = "profanity_filter"


class TextCleaningOptions(BaseModel):
    """Options for text cleaning"""
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = False
    remove_special_chars: bool = False
    normalize_whitespace: bool = True
    convert_to_lowercase: bool = False
    remove_stopwords: bool = False
    language: TextLanguage = TextLanguage.AUTO_DETECT


class TokenizationOptions(BaseModel):
    """Options for text tokenization"""
    method: str = "word"  # "word", "sentence", "subword"
    preserve_case: bool = True
    include_punctuation: bool = True
    max_tokens: Optional[int] = None
    tokenizer_model: Optional[str] = None


class EntityType(str, Enum):
    """Types of entities to extract"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    MONEY = "money"
    PERCENTAGE = "percentage"


class ExtractedEntity(BaseModel):
    """Extracted entity from text"""
    text: str
    entity_type: EntityType
    start_pos: int = Field(ge=0)
    end_pos: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('end_pos')
    def end_after_start(cls, v, values):
        if 'start_pos' in values and v <= values['start_pos']:
            raise ValueError('end_pos must be greater than start_pos')
        return v


class TextProcessingRequest(BaseProcessingRequest):
    """Request for text preprocessing"""
    text: str = Field(min_length=1, max_length=1000000)
    processing_types: List[TextProcessingType]
    input_format: DataFormat = DataFormat.TEXT
    output_format: DataFormat = DataFormat.TEXT
    language: TextLanguage = TextLanguage.AUTO_DETECT
    cleaning_options: Optional[TextCleaningOptions] = None
    tokenization_options: Optional[TokenizationOptions] = None
    entity_types: Optional[List[EntityType]] = None
    target_language: Optional[TextLanguage] = None  # For translation


class TextProcessingResult(BaseModel):
    """Result of a text processing operation"""
    operation: TextProcessingType
    success: bool
    input_text: str
    output_text: Optional[str] = None
    entities: Optional[List[ExtractedEntity]] = None
    tokens: Optional[List[str]] = None
    language_detected: Optional[TextLanguage] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextProcessingResponse(BaseProcessingResponse):
    """Response from text preprocessing"""
    original_text: str
    processed_text: Optional[str] = None
    results: List[TextProcessingResult] = Field(default_factory=list)
    language_detected: Optional[TextLanguage] = None
    input_format: DataFormat
    output_format: DataFormat
    total_operations: int = 0
    successful_operations: int = 0


# ============================================================================
# Image Preprocessor Models
# ============================================================================

class ImageFormat(str, Enum):
    """Supported image formats"""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    TIFF = "tiff"
    BMP = "bmp"
    GIF = "gif"


class ImageProcessingType(str, Enum):
    """Types of image processing operations"""
    RESIZE = "resize"
    CROP = "crop"
    ROTATE = "rotate"
    FLIP = "flip"
    ENHANCE = "enhance"
    FILTER = "filter"
    NORMALIZE = "normalize"
    CONVERT_FORMAT = "convert_format"
    EXTRACT_METADATA = "extract_metadata"
    DETECT_OBJECTS = "detect_objects"
    EXTRACT_TEXT = "extract_text"  # OCR
    GENERATE_THUMBNAIL = "generate_thumbnail"


class ResizeOptions(BaseModel):
    """Options for image resizing"""
    width: Optional[int] = Field(ge=1, le=8192)
    height: Optional[int] = Field(ge=1, le=8192)
    maintain_aspect_ratio: bool = True
    interpolation: str = "lanczos"  # "nearest", "bilinear", "bicubic", "lanczos"
    
    @validator('height')
    def at_least_one_dimension(cls, v, values):
        if v is None and values.get('width') is None:
            raise ValueError('At least one of width or height must be specified')
        return v


class CropOptions(BaseModel):
    """Options for image cropping"""
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class EnhancementOptions(BaseModel):
    """Options for image enhancement"""
    brightness: float = Field(default=1.0, ge=0.0, le=3.0)
    contrast: float = Field(default=1.0, ge=0.0, le=3.0)
    saturation: float = Field(default=1.0, ge=0.0, le=3.0)
    sharpness: float = Field(default=1.0, ge=0.0, le=3.0)


class ImageMetadata(BaseModel):
    """Image metadata"""
    width: int
    height: int
    channels: int
    format: ImageFormat
    mode: str  # RGB, RGBA, L, etc.
    size_bytes: int
    dpi: Optional[Tuple[int, int]] = None
    exif_data: Optional[Dict[str, Any]] = None
    color_profile: Optional[str] = None
    created_at: Optional[datetime] = None


class DetectedObject(BaseModel):
    """Object detected in image"""
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    center_point: Tuple[int, int]  # (x, y)
    area: int = Field(ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OCRResult(BaseModel):
    """OCR text extraction result"""
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    language: Optional[str] = None
    line_number: Optional[int] = None
    word_count: int = 0


class ImageProcessingRequest(BaseProcessingRequest):
    """Request for image preprocessing"""
    image_data: Optional[bytes] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    processing_types: List[ImageProcessingType]
    output_format: ImageFormat = ImageFormat.PNG
    resize_options: Optional[ResizeOptions] = None
    crop_options: Optional[CropOptions] = None
    enhancement_options: Optional[EnhancementOptions] = None
    rotation_angle: Optional[float] = None
    flip_horizontal: bool = False
    flip_vertical: bool = False
    quality: int = Field(default=95, ge=1, le=100)

    @validator('image_data', 'image_path', 'image_url')
    def at_least_one_image_source(cls, v, values, field):
        sources = [values.get('image_data'), values.get('image_path'), values.get('image_url')]
        non_none_sources = [s for s in sources if s is not None]
        if len(non_none_sources) != 1:
            raise ValueError('Exactly one of image_data, image_path, or image_url must be provided')
        return v


class ImageProcessingResult(BaseModel):
    """Result of an image processing operation"""
    operation: ImageProcessingType
    success: bool
    output_data: Optional[bytes] = None
    output_path: Optional[str] = None
    metadata: Optional[ImageMetadata] = None
    detected_objects: Optional[List[DetectedObject]] = None
    ocr_results: Optional[List[OCRResult]] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None


class ImageProcessingResponse(BaseProcessingResponse):
    """Response from image preprocessing"""
    original_metadata: Optional[ImageMetadata] = None
    output_data: Optional[bytes] = None
    output_path: Optional[str] = None
    output_metadata: Optional[ImageMetadata] = None
    results: List[ImageProcessingResult] = Field(default_factory=list)
    total_operations: int = 0
    successful_operations: int = 0


# ============================================================================
# Voice Preprocessor Models
# ============================================================================

class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    OPUS = "opus"


class VoiceProcessingType(str, Enum):
    """Types of voice processing operations"""
    NORMALIZE = "normalize"
    DENOISE = "denoise"
    AMPLIFY = "amplify"
    TRIM_SILENCE = "trim_silence"
    CHANGE_SPEED = "change_speed"
    CHANGE_PITCH = "change_pitch"
    CONVERT_FORMAT = "convert_format"
    RESAMPLE = "resample"
    EXTRACT_FEATURES = "extract_features"
    DETECT_VOICE_ACTIVITY = "detect_voice_activity"
    SEGMENT_SPEAKERS = "segment_speakers"
    EXTRACT_TRANSCRIPTION = "extract_transcription"


class AudioNormalizationOptions(BaseModel):
    """Options for audio normalization"""
    target_loudness_lufs: float = Field(default=-23.0, ge=-70.0, le=0.0)
    peak_normalize: bool = False
    target_peak_db: float = Field(default=-1.0, ge=-10.0, le=0.0)


class DenoiseOptions(BaseModel):
    """Options for audio denoising"""
    method: str = "spectral_subtraction"  # "spectral_subtraction", "wiener", "rnnoise"
    noise_reduction_db: float = Field(default=10.0, ge=0.0, le=30.0)
    preserve_speech: bool = True


class VoiceActivitySegment(BaseModel):
    """Voice activity detection segment"""
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    is_speech: bool
    
    @validator('end_time')
    def end_after_start(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v


class SpeakerSegment(BaseModel):
    """Speaker diarization segment"""
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    speaker_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    
    @validator('end_time')
    def end_after_start(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v


class AudioFeatures(BaseModel):
    """Extracted audio features"""
    duration_seconds: float = Field(ge=0.0)
    sample_rate: int = Field(ge=8000, le=192000)
    channels: int = Field(ge=1, le=8)
    bit_depth: Optional[int] = None
    format: AudioFormat
    loudness_lufs: Optional[float] = None
    peak_db: Optional[float] = None
    rms_db: Optional[float] = None
    zero_crossing_rate: Optional[float] = None
    spectral_centroid: Optional[float] = None
    spectral_rolloff: Optional[float] = None
    mfcc: Optional[List[float]] = None  # Mel-frequency cepstral coefficients


class VoiceProcessingRequest(BaseProcessingRequest):
    """Request for voice preprocessing"""
    audio_data: Optional[bytes] = None
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    processing_types: List[VoiceProcessingType]
    output_format: AudioFormat = AudioFormat.WAV
    output_sample_rate: Optional[int] = Field(ge=8000, le=192000)
    normalization_options: Optional[AudioNormalizationOptions] = None
    denoise_options: Optional[DenoiseOptions] = None
    speed_factor: float = Field(default=1.0, ge=0.25, le=4.0)
    pitch_semitones: float = Field(default=0.0, ge=-12.0, le=12.0)
    amplification_db: float = Field(default=0.0, ge=-30.0, le=30.0)
    
    @validator('audio_data', 'audio_path', 'audio_url')
    def at_least_one_audio_source(cls, v, values, field):
        sources = [values.get('audio_data'), values.get('audio_path'), values.get('audio_url')]
        non_none_sources = [s for s in sources if s is not None]
        if len(non_none_sources) != 1:
            raise ValueError('Exactly one of audio_data, audio_path, or audio_url must be provided')
        return v


class VoiceProcessingResult(BaseModel):
    """Result of a voice processing operation"""
    operation: VoiceProcessingType
    success: bool
    output_data: Optional[bytes] = None
    output_path: Optional[str] = None
    features: Optional[AudioFeatures] = None
    voice_activity: Optional[List[VoiceActivitySegment]] = None
    speaker_segments: Optional[List[SpeakerSegment]] = None
    transcription: Optional[str] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None


class VoiceProcessingResponse(BaseProcessingResponse):
    """Response from voice preprocessing"""
    original_features: Optional[AudioFeatures] = None
    output_data: Optional[bytes] = None
    output_path: Optional[str] = None
    output_features: Optional[AudioFeatures] = None
    results: List[VoiceProcessingResult] = Field(default_factory=list)
    total_operations: int = 0
    successful_operations: int = 0
