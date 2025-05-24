#!/usr/bin/env python3
"""
Script to fix incorrect typing imports across the codebase.
Replaces 'from test_mock_helper import List' with 'from typing import List'
"""

import os
import re
from pathlib import Path

def fix_typing_imports():
    """Fix typing imports in all Python files."""
    
    project_root = Path(__file__).parent
    python_files = list(project_root.rglob("*.py"))
    
    # Exclude test files and __pycache__ directories
    python_files = [
        f for f in python_files 
        if "__pycache__" not in str(f) and not str(f).endswith("fix_typing_imports.py")
    ]
    
    fixed_files = []
    error_files = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file has the problematic import
            if "from test_mock_helper import" in content:
                print(f"Fixing imports in: {file_path}")
                
                # Replace the problematic import
                new_content = re.sub(
                    r'from test_mock_helper import List',
                    'from typing import List',
                    content
                )
                
                # Handle other potential imports from test_mock_helper
                new_content = re.sub(
                    r'from test_mock_helper import (.+)',
                    r'from typing import \1',
                    new_content
                )
                
                # Write back the fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                fixed_files.append(str(file_path))
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            error_files.append((str(file_path), str(e)))
    
    print(f"\nSummary:")
    print(f"Fixed {len(fixed_files)} files")
    if error_files:
        print(f"Errors in {len(error_files)} files")
        for file_path, error in error_files:
            print(f"  {file_path}: {error}")
    
    if fixed_files:
        print(f"\nFixed files:")
        for file_path in fixed_files:
            print(f"  {file_path}")

if __name__ == "__main__":
    fix_typing_imports()