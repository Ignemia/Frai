# Contributing to Frai

Thank you for your interest in contributing to Frai! This is a community-driven project, and we appreciate all help.

## 🤝 Ways to Contribute

- **Report bugs** by opening an issue
- **Suggest features** that would make the app better
- **Improve documentation** to help others understand the project
- **Fix bugs** or implement new features through pull requests
- **Help others** in discussions and issue threads

## 🚀 Getting Started

### Setting Up the Development Environment

1. **Hugging Face Authentication**:
   ```bash
   # Install huggingface_hub
   pip install huggingface_hub
   
   # Login to Hugging Face
   huggingface-cli login
   ```
   This will prompt you to enter your Hugging Face token, which you can find in your [Hugging Face account settings](https://huggingface.co/settings/tokens).

2. **Clone the repository with submodules**:
   ```bash
   # Clone with submodules
   git clone --recursive https://github.com/your-org/personal-chatter.git
   cd personal-chatter
   ```
   
   If you've already cloned the repository without the `--recursive` flag:
   ```bash
   git submodule init
   git submodule update
   ```

3. **Set up your environment**:
    ```bash
    # Create a virtual environment
    python -m venv venv
    
    # Activate virtual environment
    # On Windows:
    venv\Scripts\activate
    # On Linux:
    # source venv/bin/activate

    # Install setuptools first
    pip install --upgrade pip setuptools wheel

    # Install the package in development mode
    python setup.py develop

    # Install development dependencies
    pip install -r requirements-dev.txt
    ```

## 📊 Git Workflow

We follow a structured branch workflow:

- **Master**: Production-ready code and releases
- **Testing**: Beta versions and prerelease code
- **dev**: Active development code (submit PRs to this branch)
- **Release**: Stores the latest major release
- **issue-[issueid]-name-of-issue**: For specific issue resolution

### Creating a Pull Request

1. **Create a branch from dev**:
    ```bash
    git checkout dev
    git checkout -b issue-123-fix-login-bug
    # or for features
    git checkout -b feature/user-profiles
    ```

2. **Make your changes** and commit them with clear messages

3. **Push your branch**:
    ```bash
    git push origin issue-123-fix-login-bug
    ```

4. **Open a pull request** to the `dev` branch through the GitHub interface

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

## 📋 Development Guidelines

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **Ruff** for linting

You can run these tools with:
```bash
black .
isort .
ruff check .
```

### Testing

Please add tests for new functionality:
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Documentation

- Add docstrings to new functions and classes
- Update README.md if needed

## 🏗 Project Structure

```
personal-chatter/
├── api/              # FastAPI endpoints
├── services/         # Core functionality
└── tests/            # Test suite
```

## 🔍 Pull Request Review Process

1. A maintainer will review your PR
2. They may ask for changes or clarification
3. Once approved, your PR will be merged

## 📝 Issue Guidelines

When opening an issue:

1. **Use a clear title** that summarizes the issue
2. **Provide detailed information**:
   - For bugs: Steps to reproduce, expected vs. actual behavior
   - For features: What you want, why it's useful, how it should work
3. **Include your environment**: OS, Python version, etc.

## 🙏 Thank You!

Your contributions help make Frai better for everyone. Every contribution matters, no matter how small.

Happy coding! 🚀
