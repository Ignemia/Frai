# Frai - Folder Structure Refactoring Plan

## Current Issues
1. Services are mixed in a flat structure
2. Image generation and memory management could be better organized
3. Configuration management could be more modular
4. WebSocket functionality is scattered

## Proposed New Structure

```
services/
├── ai/                         # AI/ML related services
│   ├── __init__.py
│   ├── image_generation/       # Image generation module
│   │   ├── __init__.py
│   │   ├── flux_generator.py   # Main Flux.1 implementation
│   │   ├── progress_tracker.py # Progress tracking utilities
│   │   ├── langchain_integration.py # LangChain wrapper
│   │   └── memory_manager.py   # GPU/VRAM memory management
│   ├── language_models/        # LLM related services
│   │   ├── __init__.py
│   │   ├── model_loader.py     # Model loading utilities
│   │   └── inference.py        # Inference management
│   └── embeddings/             # Embedding services
│       ├── __init__.py
│       └── embedding_service.py
├── communication/              # Communication services
│   ├── __init__.py
│   ├── websocket/             # WebSocket management
│   │   ├── __init__.py
│   │   ├── connection_manager.py
│   │   └── progress_broadcaster.py
│   └── notifications/         # Notification services
│       ├── __init__.py
│       └── notification_service.py
├── core/                      # Core services
│   ├── __init__.py
│   ├── config/               # Configuration management
│   │   ├── __init__.py
│   │   ├── config_manager.py  # Main config handling
│   │   ├── memory_config.py   # Memory management settings
│   │   └── model_config.py    # Model configuration
│   ├── state/                # State management
│   │   ├── __init__.py
│   │   ├── app_state.py      # Application state
│   │   └── session_state.py  # Session management
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── file_utils.py
│       └── validation.py
├── data/                     # Data services
│   ├── __init__.py
│   ├── search/              # Search services
│   │   ├── __init__.py
│   │   ├── document_search.py
│   │   └── online_search.py
│   └── memory/              # Memory and storage
│       ├── __init__.py
│       └── user_memory.py
├── processing/              # Command and data processing
│   ├── __init__.py
│   ├── command_processor.py
│   └── pipeline/           # Processing pipelines
│       ├── __init__.py
│       └── command_pipeline.py
└── legacy/                 # Legacy modules (to be refactored)
    ├── __init__.py
    └── deprecated_modules/
```

## Migration Steps

### Phase 1: Create new directory structure
1. Create new directories
2. Move image_generator.py to ai/image_generation/
3. Split into specialized modules

### Phase 2: Refactor configuration management
1. Split config.py into specialized modules
2. Update imports across the application

### Phase 3: Refactor WebSocket management
1. Move websocket_manager.py to communication/websocket/
2. Update connection management

### Phase 4: Update imports and dependencies
1. Update all import statements
2. Add __init__.py files with proper exports
3. Test all functionality

### Phase 5: Legacy cleanup
1. Remove old files
2. Update documentation
3. Final testing

## Benefits
1. Better code organization and maintainability
2. Clearer separation of concerns
3. Easier to locate and modify specific functionality
4. More professional project structure
5. Better support for future scaling
