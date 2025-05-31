# Frai Installation Guide

## Overview

Frai is an AI-powered chat and image generation system that requires careful installation due to its complex dependencies and local model requirements. This guide will walk you through the complete installation process.

## System Requirements

### Python Version
- **Required**: Python 3.12 or higher
- **Recommended**: Python 3.12.x
- **Compatibility Note**: Python 3.13 is supported but some dependencies may need to be built from source

### Hardware Requirements
- **Minimum RAM**: 8GB (16GB+ recommended for larger models)
- **GPU**: Optional but recommended (CUDA-compatible for acceleration)
- **Storage**: 50GB+ free space for models

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install cmake pkg-config git-lfs

# macOS (with Homebrew)
brew install cmake pkg-config git-lfs

# Windows
# Install Git LFS from: https://git-lfs.github.io/
# Install Visual Studio Build Tools or Visual Studio Community
```

## Installation Methods

### Method 1: Automatic Installation (Recommended)

1. **Clone the repository with submodules**:
```bash
git clone --recursive https://github.com/Ignemia/Frai.git
cd Frai
```

2. **Install Git LFS and download model files**:
```bash
git lfs install
git lfs pull
```

3. **Create and activate virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

4. **Run the installation script**:
```bash
python install.py
```

For development installation:
```bash
python install.py --dev
```

### Method 2: Manual Installation

1. **Clone and setup repository**:
```bash
git clone https://github.com/Ignemia/Frai.git
cd Frai
git submodule update --init --recursive
git lfs install
git lfs pull
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
```

3. **Install PyTorch** (choose based on your system):
```bash
# CPU only
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html

# CUDA (Linux/Windows)
pip install torch torchvision torchaudio

# ROCm (AMD GPUs)
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/rocm5.6/torch_stable.html
```

4. **Install core dependencies**:
```bash
pip install -e .
```

### Method 3: Using Make

```bash
make install      # Standard installation
make install-dev  # Development installation
```

## Model Management

### Local Models vs HuggingFace

Frai supports both local models (via Git submodules) and automatic downloading from HuggingFace:

- **Local models**: Faster startup, no internet required after setup
- **HuggingFace fallback**: Automatic download if local models are unavailable

### Available Models

The following models are configured as Git submodules:

| Model | Type | Description | Size |
|-------|------|-------------|------|
| google/gemma-3-4b-it | Chat | Primary chat model | ~8GB |
| nari-labs/Dia-1.6B | Chat | Alternative chat model | ~3GB |
| black-forest-labs/FLUX.1-dev | Image | Text-to-image generation | ~12GB |
| google/Gemma-Embeddings-v1.0 | Embeddings | Text embeddings | ~2GB |
| pyannote/speaker-diarization-3.1 | Audio | Speaker identification | ~1GB |
| tabularisai/multilingual-sentiment-analysis | NLP | Sentiment analysis | ~500MB |

### Git LFS Issues

If you encounter "expected value at line 1 column 1" errors, it means Git LFS files weren't downloaded:

```bash
# Fix Git LFS issues
git lfs install
git lfs pull

# Verify models
python validate_models.py
```

## Validation and Testing

### Validate Installation
```bash
python validate_models.py
```

This script will:
- Check Python version compatibility
- Verify Git submodules are initialized
- Detect Git LFS issues
- Validate model availability
- Provide troubleshooting recommendations

### Test Application
```bash
python main.py --help
```

### Test Individual Components
```bash
# Test imports
python -c "import back, front, orchestrator; print('✓ All packages imported')"

# Test model loading (may take time on first run)
python -c "from back.ai import load_chat_model; print('✓ Chat model loaded:', load_chat_model())"
```

## Python 3.13 Compatibility

Python 3.13 is supported but requires some workarounds:

### Known Issues
- `sentencepiece`: Requires cmake/pkg-config for building
- `xformers`: May need to build from source
- Some dependencies may not have pre-built wheels

### Solutions
1. **Install system dependencies**:
```bash
# Ubuntu/Debian
sudo apt install cmake pkg-config build-essential

# macOS
brew install cmake pkg-config
```

2. **Use Python 3.12** (recommended for easier setup):
```bash
pyenv install 3.12.7
pyenv local 3.12.7
```

## Troubleshooting

### Common Issues

#### 1. PyTorch Installation Fails
```bash
# Try CPU-only version first
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

#### 2. sentencepiece Build Failure
```bash
# Install system dependencies
sudo apt install cmake pkg-config  # Ubuntu/Debian
brew install cmake pkg-config      # macOS

# Or skip sentence-transformers for now
pip install transformers  # Core functionality works without sentence-transformers
```

#### 3. xformers Installation Issues
```bash
# Skip xformers (optional for GPU acceleration)
# The system will work without it, just slower on GPU
```

#### 4. Git LFS Files Missing
```bash
# Reinstall and pull LFS files
git lfs install
git lfs pull

# Verify with
python validate_models.py
```

#### 5. Pydantic Version Conflicts
```bash
# Update to latest compatible version
pip install "pydantic>=2.0.0"
```

#### 6. Model Loading Errors
The system automatically falls back to HuggingFace if local models have issues. Check logs for:
- `WARNING - Local model ... has Git LFS issues, falling back to HuggingFace`
- `INFO - Using HuggingFace model: ...`

### Environment Issues

#### Virtual Environment
```bash
# Create fresh environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
python install.py
```

#### Permission Issues
```bash
# Linux/macOS
sudo chown -R $USER:$USER .
chmod +x install.py

# Ensure Git LFS is accessible
which git-lfs
```

### Performance Issues

#### Large Model Downloads
- First run may take 30+ minutes to download models
- Ensure stable internet connection
- Consider using `--timeout 600` for pip installs

#### Memory Issues
- Close other applications when loading large models
- Use CPU-only mode if GPU memory is insufficient
- Consider using smaller models (Dia-1.6B instead of Gemma-3-4B)

## Advanced Configuration

### Environment Variables
```bash
# Optional: Configure model paths
export FRAI_MODELS_DIR="/path/to/models"

# Optional: Configure system prompts
export POSITIVE_SYSTEM_PROMPT_CHAT="Be helpful and concise"
export NEGATIVE_SYSTEM_PROMPT_CHAT="Avoid inappropriate content"
```

### Custom Model Paths
Edit `back/ai/model_config.py` to add custom model configurations.

### Development Setup
```bash
# Install with development dependencies
python install.py --dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
make lint
```

## Getting Help

### Check Logs
- Installation logs are shown in terminal output
- Application logs include model loading details
- Use `python validate_models.py` for diagnostic information

### Common Commands
```bash
# Validate everything
python validate_models.py

# Test minimal functionality
python -c "import back; print('Backend loaded')"

# Check model status
python -c "from back.ai.model_config import validate_model_directory; print(validate_model_directory())"

# Update dependencies
pip install --upgrade pip
python install.py
```

### Support
- Create an issue on GitHub with validation output
- Include Python version and OS information
- Attach relevant error logs

## Quick Start After Installation

1. **Verify installation**:
```bash
python validate_models.py
```

2. **Start the application**:
```bash
python main.py
```

3. **Use CLI interface**:
```bash
frai-chat
```

4. **Start API server**:
```bash
uvicorn api.main:app --reload
```

The installation should now be complete and ready for use!