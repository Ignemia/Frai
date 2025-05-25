
TROUBLESHOOTING GUIDE
=====================

1. Python 3.13 Compatibility Issues:
   - Some packages may not have pre-built wheels for Python 3.13
   - Solution: Install build tools or use Python 3.12

2. sentencepiece Build Failures:
   - Requires cmake and pkg-config
   - Ubuntu/Debian: sudo apt install cmake pkg-config
   - macOS: brew install cmake pkg-config
   - Or use pre-built wheel: pip install sentencepiece --find-links https://download.pytorch.org/whl/torch_stable.html

3. xformers Installation Issues:
   - Only supported with CUDA
   - Build from source may take a long time
   - Optional - application works without it

4. General Installation Issues:
   - Update pip: python -m pip install --upgrade pip
   - Clear cache: python -m pip cache purge
   - Try with --no-cache-dir flag

5. Virtual Environment Issues:
   - Create fresh venv: python -m venv .venv
   - Activate: source .venv/bin/activate (Linux/Mac) or .venv\Scripts\activate (Windows)
   - Install in venv: python install.py

For more help, visit: https://github.com/Ignemia/Frai/issues
