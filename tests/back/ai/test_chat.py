"""
Backend AI Chat Tests

Tests for the chat AI module using proper data models.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from back.models import (
    ChatRequest, ChatResponse, ChatMessage, ConversationHistory,
    MessageRole, ChatGenerationParams, ChatModelStatus, AIStatus, DeviceType
)
from back.ai.chat import ChatAI, get_chat_ai_instance, initialize_chat_system


class TestChatAIModels:
    """Test chat AI data models"""
    
    def test_message_role_enum(self):
        """Test MessageRole enum values"""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"
    
    def test_chat_message_creation(self):
        """Test ChatMessage model creation"""
        message = ChatMessage(
            role=MessageRole.USER,
            content="Hello, how are you?"
        )
        
        assert message.role == MessageRole.USER
        assert message.content == "Hello, how are you?"
        assert isinstance(message.timestamp, datetime)
        assert message.metadata == {}
    
    def test_conversation_history_to_dict_list(self):
        """Test ConversationHistory conversion to dict list"""
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        conversation = ConversationHistory(messages=messages)
        
        dict_list = conversation.to_dict_list()
        
        assert len(dict_list) == 2
        assert dict_list[0] == {"role": "user", "content": "Hello"}
        assert dict_list[1] == {"role": "assistant", "content": "Hi there!"}
    
    def test_chat_generation_params_validation(self):
        """Test ChatGenerationParams validation"""
        # Valid params
        params = ChatGenerationParams(
            temperature=0.8,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.2,
            max_new_tokens=4096
        )
        
        assert params.temperature == 0.8
        assert params.top_p == 0.9
        assert params.top_k == 40
        assert params.repetition_penalty == 1.2
        assert params.max_new_tokens == 4096
        
        # Test validation errors
        with pytest.raises(ValueError):
            ChatGenerationParams(temperature=-0.1)  # Below 0.0
        
        with pytest.raises(ValueError):
            ChatGenerationParams(temperature=2.1)   # Above 2.0
        
        with pytest.raises(ValueError):
            ChatGenerationParams(top_p=-0.1)        # Below 0.0
        
        with pytest.raises(ValueError):
            ChatGenerationParams(top_p=1.1)         # Above 1.0
    
    def test_chat_request_creation(self):
        """Test ChatRequest model creation"""
        messages = [
            ChatMessage(role=MessageRole.USER, content="What is AI?")
        ]
        conversation = ConversationHistory(messages=messages)
        
        request = ChatRequest(
            conversation_history=conversation,
            positive_system_prompt="Be helpful",
            negative_system_prompt="Don't be harmful",
            max_new_tokens=100
        )
        
        assert request.conversation_history == conversation
        assert request.positive_system_prompt == "Be helpful"
        assert request.negative_system_prompt == "Don't be harmful"
        assert request.max_new_tokens == 100
    
    def test_chat_response_model(self):
        """Test ChatResponse model"""
        response = ChatResponse(
            success=True,
            response="AI is artificial intelligence.",
            conversation_id="conv_123",
            model_name="gemma-3-4b-it",
            device_used=DeviceType.CUDA
        )
        
        assert response.success is True
        assert response.response == "AI is artificial intelligence."
        assert response.conversation_id == "conv_123"
        assert response.model_name == "gemma-3-4b-it"
        assert response.device_used == DeviceType.CUDA
        assert isinstance(response.timestamp, datetime)


class TestChatAI:
    """Test ChatAI class functionality"""
    
    @pytest.fixture
    def mock_model_loader(self):
        """Mock model loader functions"""
        with patch('back.ai.chat.load_model_and_tokenizer') as mock_load, \
             patch('back.ai.chat.move_model_to_device') as mock_move, \
             patch('back.ai.chat.get_generation_pipeline') as mock_pipeline, \
             patch('back.ai.chat.generate_text') as mock_generate:
            
            # Setup mocks
            mock_load.return_value = (MagicMock(), MagicMock())  # model, tokenizer
            mock_move.return_value = True
            mock_pipeline.return_value = MagicMock()
            mock_generate.return_value = {
                "success": True,
                "response": "Generated response",
                "metadata": {}
            }
            
            yield {
                'load': mock_load,
                'move': mock_move,
                'pipeline': mock_pipeline,
                'generate': mock_generate
            }
    
    def test_chat_ai_initialization(self, mock_model_loader):
        """Test ChatAI initialization"""
        chat_ai = ChatAI(
            model_name="test-model",
            model_path="test/path"
        )
        
        assert chat_ai.model_name == "test-model"
        assert chat_ai.model_path == "test/path"
        assert chat_ai.is_loaded is True
        assert chat_ai.vram_device in ["cuda", "cpu"]
        assert chat_ai.ram_device == "cpu"
        
        # Verify model loading was called
        mock_model_loader['load'].assert_called_once()
    
    def test_chat_ai_generation_with_models(self, mock_model_loader):
        """Test chat generation using models"""
        chat_ai = ChatAI()
        
        # Create test conversation
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello")
        ]
        conversation = ConversationHistory(messages=messages)
        
        # Create request
        request = ChatRequest(conversation_history=conversation)
        
        # Generate response
        response_dict = chat_ai.generate_response(
            conversation_history=request.conversation_history.to_dict_list(),
            positive_system_prompt_override=request.positive_system_prompt,
            negative_system_prompt_override=request.negative_system_prompt,
            max_new_tokens=request.max_new_tokens
        )
        
        # Verify response structure
        assert response_dict["success"] is True
        assert "response" in response_dict
        assert "metadata" in response_dict
        
        # Create response model
        response = ChatResponse(**response_dict)
        assert response.success is True
        assert response.response == "Generated response"
    
    def test_format_conversation_for_model(self):
        """Test conversation formatting"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        formatted = ChatAI.format_conversation_for_model(
            messages=messages,
            positive_system_prompt="Be helpful",
            negative_system_prompt="Don't be harmful"
        )
        
        assert "System:" in formatted
        assert "Be helpful" in formatted
        assert "Don't be harmful" in formatted
        assert "User: Hello" in formatted
        assert "Assistant: Hi there!" in formatted
        assert "User: How are you?" in formatted
        assert formatted.endswith("Assistant:")


class TestChatAIIntegration:
    """Test ChatAI integration functions"""
    
    @patch('back.ai.chat.get_chat_ai_instance')
    def test_initialize_chat_system_success(self, mock_get_instance):
        """Test successful chat system initialization"""
        # Mock ChatAI instance
        mock_chat_ai = Mock()
        mock_chat_ai.is_loaded = True
        mock_get_instance.return_value = mock_chat_ai
        
        result = initialize_chat_system("test-model", "test/path")
        
        assert result is True
        mock_get_instance.assert_called_once_with("test-model", "test/path")
    
    @patch('back.ai.chat.get_chat_ai_instance')
    def test_initialize_chat_system_failure(self, mock_get_instance):
        """Test failed chat system initialization"""
        # Mock ChatAI instance that failed to load
        mock_chat_ai = Mock()
        mock_chat_ai.is_loaded = False
        mock_get_instance.return_value = mock_chat_ai
        
        result = initialize_chat_system("test-model", "test/path")
        
        assert result is False
    
    @patch('back.ai.chat.get_chat_ai_instance')
    def test_initialize_chat_system_exception(self, mock_get_instance):
        """Test chat system initialization with exception"""
        mock_get_instance.side_effect = Exception("Model loading failed")
        
        result = initialize_chat_system("test-model", "test/path")
        
        assert result is False
    
    def test_get_chat_ai_instance_singleton(self):
        """Test that get_chat_ai_instance returns singleton"""
        with patch('back.ai.chat.ChatAI') as mock_chat_ai_class:
            mock_instance = Mock()
            mock_chat_ai_class.return_value = mock_instance
            
            # First call should create instance
            instance1 = get_chat_ai_instance()
            mock_chat_ai_class.assert_called_once()
            
            # Second call should return same instance
            instance2 = get_chat_ai_instance()
            mock_chat_ai_class.assert_called_once()  # Should not be called again
            
            assert instance1 is instance2


class TestChatAIErrorHandling:
    """Test ChatAI error handling"""
    
    @patch('back.ai.chat.load_model_and_tokenizer')
    def test_model_loading_failure(self, mock_load):
        """Test handling of model loading failure"""
        mock_load.side_effect = Exception("Failed to load model")
        
        with pytest.raises(RuntimeError, match="Could not load model"):
            ChatAI(model_name="test-model")
    
    def test_generation_with_unloaded_model(self):
        """Test generation attempt with unloaded model"""
        with patch('back.ai.chat.load_model_and_tokenizer') as mock_load:
            mock_load.return_value = (None, None)  # Simulate failed loading
            
            # This should raise during initialization
            with pytest.raises(RuntimeError):
                ChatAI(model_name="test-model")
    
    @patch('back.ai.chat.load_model_and_tokenizer')
    @patch('back.ai.chat.get_generation_pipeline')
    def test_generation_pipeline_failure(self, mock_pipeline, mock_load):
        """Test handling of generation pipeline failure"""
        # Mock successful model loading
        mock_load.return_value = (MagicMock(), MagicMock())
        
        # Mock pipeline failure
        mock_pipeline.return_value = None
        
        chat_ai = ChatAI()
        
        response = chat_ai.generate_response(
            conversation_history=[{"role": "user", "content": "Hello"}]
        )
        
        assert response["success"] is False
        assert "error" in response


class TestChatModelStatus:
    """Test ChatModelStatus model"""
    
    def test_chat_model_status_creation(self):
        """Test ChatModelStatus model creation"""
        status = ChatModelStatus(
            status=AIStatus.LOADED,
            model_name="gemma-3-4b-it",
            model_path="models/gemma-3-4b-it",
            is_loaded=True,
            current_device=DeviceType.CUDA,
            memory_usage_mb=2048.5,
            last_generation_time=datetime.now()
        )
        
        assert status.status == AIStatus.LOADED
        assert status.model_name == "gemma-3-4b-it"
        assert status.model_path == "models/gemma-3-4b-it"
        assert status.is_loaded is True
        assert status.current_device == DeviceType.CUDA
        assert status.memory_usage_mb == 2048.5
        assert isinstance(status.last_generation_time, datetime)
    
    def test_ai_status_enum(self):
        """Test AIStatus enum values"""
        assert AIStatus.UNINITIALIZED == "uninitialized"
        assert AIStatus.LOADING == "loading"
        assert AIStatus.LOADED == "loaded"
        assert AIStatus.PROCESSING == "processing"
        assert AIStatus.ERROR == "error"
        assert AIStatus.UNLOADED == "unloaded"
    
    def test_device_type_enum(self):
        """Test DeviceType enum values"""
        assert DeviceType.CPU == "cpu"
        assert DeviceType.CUDA == "cuda"
        assert DeviceType.MPS == "mps"
