#!/usr/bin/env python3
"""
Model Validation Script for Frai

This script validates that all configured models are available locally
and provides information about missing models.
"""

import sys
import logging
from pathlib import Path
import json

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from back.ai.model_config import (
    validate_model_directory,
    list_available_models,
    get_model_info,
    MODEL_CONFIGS,
    MODELS_BASE_DIR,
    ModelType
)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}")

def is_lfs_pointer_file(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer file."""
    if not file_path.exists() or file_path.stat().st_size > 1000:  # LFS pointers are small
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return first_line.startswith("version https://git-lfs.github.com/spec/v")
    except:
        return False

def check_model_lfs_status(model_path: Path) -> tuple[bool, list[str]]:
    """Check if model has Git LFS files that aren't downloaded."""
    if not model_path.exists():
        return True, []  # Directory doesn't exist, different issue
    
    lfs_files = []
    critical_files = ["tokenizer.json", "model-*.safetensors", "*.safetensors"]
    
    for pattern in critical_files:
        for file_path in model_path.glob(pattern):
            if is_lfs_pointer_file(file_path):
                lfs_files.append(file_path.name)
    
    return len(lfs_files) == 0, lfs_files

def print_model_status(model_name: str, config: dict, is_available: bool):
    """Print the status of a single model."""
    status_color = "\033[92m" if is_available else "\033[91m"  # Green or Red
    reset_color = "\033[0m"
    
    print(f"\nModel: {model_name}")
    print(f"  Type: {config['type'].value}")
    print(f"  Description: {config['description']}")
    print(f"  Local Path: {config['local_path']}")
    
    if is_available:
        # Check for Git LFS issues
        lfs_ok, lfs_files = check_model_lfs_status(config['local_path'])
        if lfs_ok:
            print(f"  Status: {status_color}✓ AVAILABLE{reset_color}")
        else:
            print(f"  Status: \033[93m⚠ AVAILABLE (LFS FILES NOT DOWNLOADED){reset_color}")
            print(f"  LFS Files: {', '.join(lfs_files)}")
    else:
        print(f"  Status: {status_color}✗ MISSING{reset_color}")

def validate_git_submodules():
    """Check if git submodules are properly initialized."""
    print_header("GIT SUBMODULES VALIDATION")
    
    gitmodules_file = project_root / ".gitmodules"
    if not gitmodules_file.exists():
        print("✗ .gitmodules file not found")
        return False, False
    
    print("✓ .gitmodules file found")
    
    # Check if models directory exists
    if not MODELS_BASE_DIR.exists():
        print(f"✗ Models directory not found: {MODELS_BASE_DIR}")
        print("\nTo initialize git submodules, run:")
        print("  git submodule update --init --recursive")
        return False, False
    
    print(f"✓ Models directory exists: {MODELS_BASE_DIR}")
    
    # Check individual model directories
    submodule_dirs = list(MODELS_BASE_DIR.iterdir())
    if not submodule_dirs:
        print("✗ No model subdirectories found")
        print("\nTo initialize git submodules, run:")
        print("  git submodule update --init --recursive")
        return False, False
    
    print(f"✓ Found {len(submodule_dirs)} model directories")
    
    # Check for Git LFS issues
    lfs_issues = 0
    for dir_path in submodule_dirs:
        if dir_path.is_dir():
            lfs_ok, lfs_files = check_model_lfs_status(dir_path)
            if lfs_files:
                print(f"  - {dir_path.name} (⚠ LFS files not downloaded: {len(lfs_files)} files)")
                lfs_issues += 1
            else:
                print(f"  - {dir_path.name}")
    
    submodules_ok = True
    lfs_ok = lfs_issues == 0
    
    if lfs_issues > 0:
        print(f"\n⚠ Git LFS files not downloaded in {lfs_issues} model(s)")
        print("To download LFS files, run:")
        print("  git lfs pull")
        print("Or reinstall Git LFS and pull:")
        print("  git lfs install")
        print("  git lfs pull")
    
    return submodules_ok, lfs_ok

def main():
    """Main validation function."""
    setup_logging()
    
    print_header("FRAI MODEL VALIDATION")
    print(f"Models base directory: {MODELS_BASE_DIR}")
    
    # Validate git submodules first
    submodules_ok, lfs_ok = validate_git_submodules()
    
    # Validate individual models
    print_header("MODEL AVAILABILITY CHECK")
    
    validation_results = validate_model_directory()
    
    # Check for Git LFS issues in available models
    lfs_issue_models = []
    for model_name, config in MODEL_CONFIGS.items():
        if validation_results[model_name]:  # Only check available models
            lfs_ok_model, lfs_files = check_model_lfs_status(config['local_path'])
            if not lfs_ok_model:
                lfs_issue_models.append((model_name, lfs_files))
    
    # Group models by type for better organization
    models_by_type = {}
    for model_name, config in MODEL_CONFIGS.items():
        model_type = config['type']
        if model_type not in models_by_type:
            models_by_type[model_type] = []
        models_by_type[model_type].append((model_name, config))
    
    # Display results by type
    for model_type, models in models_by_type.items():
        print(f"\n--- {model_type.value.upper()} MODELS ---")
        for model_name, config in models:
            is_available = validation_results[model_name]
            print_model_status(model_name, config, is_available)
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    total_models = len(validation_results)
    available_models = sum(validation_results.values())
    missing_models = total_models - available_models
    
    print(f"Total configured models: {total_models}")
    print(f"Available locally: {available_models}")
    print(f"Missing: {missing_models}")
    if lfs_issue_models:
        print(f"LFS issues: {len(lfs_issue_models)}")
    
    if missing_models == 0 and not lfs_issue_models:
        print("\n🎉 All models are available locally and ready to use!")
        print("You can run Frai without downloading additional models.")
    elif missing_models == 0 and lfs_issue_models:
        print("\n⚠️  All models are present but Git LFS files need downloading")
        print("Models have Git LFS files that aren't downloaded yet.")
    else:
        print(f"\n⚠️  {missing_models} model(s) missing")
        if lfs_issue_models:
            print(f"⚠️  {len(lfs_issue_models)} model(s) have LFS issues")
        print("Missing models will be downloaded from HuggingFace on first use.")
        
        if not submodules_ok or not lfs_ok:
            print("\n📋 RECOMMENDED ACTIONS:")
            print("1. Initialize git submodules:")
            print("   git submodule update --init --recursive")
            if not lfs_ok:
                print("\n2. Download Git LFS files:")
                print("   git lfs install")
                print("   git lfs pull")
            print("\n3. If git steps fail, models will be auto-downloaded during usage")
        
        missing_list = [name for name, available in validation_results.items() if not available]
        if missing_list:
            print(f"\nMissing models: {', '.join(missing_list)}")
        
        if lfs_issue_models:
            print(f"Models with LFS issues: {', '.join([name for name, _ in lfs_issue_models])}")
    
    # Installation status
    print_header("INSTALLATION STATUS")
    
    if available_models > 0 and not lfs_issue_models:
        print("✓ Frai can run with available local models")
        if missing_models > 0:
            print("⚠️  Some features may require model downloads on first use")
    elif available_models > 0 and lfs_issue_models:
        print("⚠️  Frai will attempt to run but may fall back to downloading models")
        print("Git LFS files are missing - models may be downloaded from HuggingFace instead")
        if missing_models > 0:
            print("⚠️  Some features may require model downloads on first use")
    else:
        print("⚠️  No local models found - all models will be downloaded on first use")
        print("This may take significant time and bandwidth on first startup")
    
    print(f"\nTo test the installation, run:")
    print(f"  python main.py --help")
    
    return missing_models == 0 and len(lfs_issue_models) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)