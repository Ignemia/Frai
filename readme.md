# Frai - Your own AI friend
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/github/actions/workflow/status/your-org/frai/tests.yml?label=tests&logo=github)](https://github.com/your-org/frai/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/your-org/frai/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/frai)

> An open-source AI companion that runs locally and keeps your data private.

## 🎯 Purpose

Frai is a free, privacy-focused AI companion designed to provide an alternative to paid services like ChatGPT and Gemini. It runs entirely on your local machine, ensuring your conversations and data remain private.

## 🌟 Why Frai?

In today's world, AI chatbots typically:
- 💰 **Require subscriptions** for full functionality
- 🔒 **Store your personal data** on company servers
- 🌐 **Need internet connectivity** for basic operation

**Frai is different:**
- ✅ **100% Local** - Runs entirely on your device
- ✅ **Privacy-First** - Your data never leaves your machine
- ✅ **Free** - No subscriptions or hidden costs
- ✅ **Open Source** - Transparent and customizable
- ✅ **Offline-Capable** - Works without internet connection

## 🚀 Features

- **Text Generation**: Powered by `google/gemma-3-4b-it` (swappable with other models)
- **Image Generation**: Create images with `black-forest-labs/FLUX.1-dev`
- **Memory System**: Remembers information about you and your preferences
- **Multiple Chat Sessions**: Organize conversations by topic
- **Customizable**: Swap models or modify code to suit your needs

## 📋 System Requirements

This is a resource-intensive application that requires modern hardware:

- **CPU**: AMD Ryzen 5 7600X+ / Intel 11th gen or newer
- **RAM**: 32GB minimum
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 2060 or better) or AMD 7600XT+
- **Storage**: 20GB free space for models and application data
- **OS**: Windows 10/11 or Linux (Apple products are not officially supported for ethical reasons)
- **Python**: 3.12 (no support for earlier versions)

## 🛠 Installation

**📖 For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md)**

### Quick Start

1. **Clone with submodules**:
```bash
git clone --recursive https://github.com/Ignemia/Frai.git
cd Frai
```

2. **Install Git LFS and download models**:
```bash
git lfs install
git lfs pull
```

3. **Create virtual environment and install**:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate on Windows

python install.py
python setup.py install

# Or use pip in development mode
pip install -e .
```

#### Development Installation

```bash
# Install in development mode
python setup.py develop

# Install development dependencies
pip install -r requirements-dev.txt
```

### 5. Run the Application

```bash
python main.py
```

### Installation

#### Automated Installation (Recommended)

The automated installation script will detect if you have CUDA or ROCm available and install the appropriate PyTorch packages:

```bash
# Clone the repository
git clone https://github.com/Ignemia/Frai.git
cd Frai

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Run the installation script
python install.py  # For regular installation
python install.py --dev  # For development installation with dev dependencies
```

#### Manual Installation

If you prefer to install manually:

1. Install PyTorch packages appropriate for your system from the [official PyTorch website](https://pytorch.org/get-started/locally/)
2. Install Frai:
   ```bash
   pip install -e .  # Regular installation
   pip install -e ".[dev,test]"  # Development installation
   ```

## 🚦 Quick Start

```bash
# Start the application
python main.py

# Follow the setup wizard to create your profile
```

Once running, visit `http://localhost:8000` in your browser to access the web interface.

## 📊 Git Workflow

We follow a structured branch workflow:

- **Master**: Production-ready code and releases
- **Testing**: Beta versions and prerelease code
- **dev**: Active development code (submit PRs to this branch)
- **Release**: Stores the latest major release
- **issue-[issueid]-name-of-issue**: For specific issue resolution

## 🔢 Version Naming Convention

We use standard semantic versioning (SemVer):

```
MAJOR.MINOR.PATCH
```

- **MAJOR**: Incremented when making incompatible API changes
- **MINOR**: Incremented when adding functionality in a backward compatible manner
- **PATCH**: Incremented when making backward compatible bug fixes

Example: `2.3.5` (Major version 2, minor version 3, patch 5)

For pre-releases, we may use suffixes like `-alpha.1`, `-beta.2`, or `-rc.1`.

## 🧪 Testing

```bash
# Run tests
pytest
```

## 🤝 Contributing

We welcome contributions! This is a community project run by volunteers.

1. Clone the repository
2. Create a branch from dev (`git checkout -b issue-123-fix-login-bug`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push the branch (`git push origin issue-123-fix-login-bug`)
6. Open a Pull Request to the `dev` branch

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all the open source AI model providers
- Everyone who has contributed to this project

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-org/frai/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-org/frai/discussions)

---

<div align="center">

**Made with ❤️ by the community**

</div>
