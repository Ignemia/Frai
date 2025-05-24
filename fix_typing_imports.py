#!/usr/bin/env python3
"""
Enhanced script to fix incorrect typing imports across the codebase.
Replaces 'from test_mock_helper import List' with 'from typing import List'
and handles other common import issues.
"""

import os
import re
import sys
from pathlib import Path
from typing import List as ListType, Tuple, Dict, Set
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TypingImportFixer:
    """Fix typing imports across the codebase."""
    
    def __init__(self, project_root: Path, dry_run: bool = False, verbose: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.verbose = verbose
        
        # Common typing imports that might be incorrectly imported
        self.typing_imports = {
            'List', 'Dict', 'Tuple', 'Set', 'Optional', 'Union', 'Any',
            'Callable', 'Iterator', 'Generator', 'Type', 'TypeVar',
            'Generic', 'Protocol', 'Literal', 'Final', 'ClassVar'
        }
        
        # Import patterns to fix
        self.import_patterns = [
            # Fix test_mock_helper imports
            (
                r'from test_mock_helper import (.+)',
                lambda m: f'from typing import {m.group(1)}'
            ),
            # Fix other common incorrect imports
            (
                r'from \.test_mock_helper import (.+)',
                lambda m: f'from typing import {m.group(1)}'
            ),
            # Fix imports from wrong modules
            (
                r'from unittest\.mock import (List|Dict|Tuple|Set|Optional|Union|Any)',
                lambda m: f'from typing import {m.group(1)}'
            ),
        ]
        
    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed."""
        # Skip certain directories and files
        skip_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'node_modules',
            'venv',
            'env',
            '.venv',
            'build',
            'dist',
            '*.egg-info',
        ]
        
        path_str = str(file_path)
        for pattern in skip_patterns:
            if pattern in path_str:
                return False
                
        # Only process Python files
        return file_path.suffix == '.py'
    
    def fix_imports_in_content(self, content: str, file_path: Path) -> Tuple[str, bool]:
        """Fix imports in file content."""
        original_content = content
        modified = False
        changes = []
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            original_line = line
            
            # Apply each import pattern
            for pattern, replacement_func in self.import_patterns:
                match = re.search(pattern, line)
                if match:
                    new_line = replacement_func(match)
                    lines[i] = new_line
                    changes.append(f"Line {i+1}: '{original_line.strip()}' → '{new_line.strip()}'")
                    modified = True
                    if self.verbose:
                        logger.info(f"Fixed import in {file_path}: {original_line.strip()} → {new_line.strip()}")
        
        if modified:
            new_content = '\n'.join(lines)
            return new_content, True
            
        return content, False
    
    def fix_file(self, file_path: Path) -> bool:
        """Fix imports in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file has problematic imports
            has_issues = any(
                re.search(pattern, content) 
                for pattern, _ in self.import_patterns
            )
            
            if not has_issues:
                return False
                
            logger.info(f"Fixing imports in: {file_path}")
            
            new_content, modified = self.fix_imports_in_content(content, file_path)
            
            if modified and not self.dry_run:
                # Create backup
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                # Remove backup if write was successful
                backup_path.unlink()
                
            return modified
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False
    
    def scan_project(self) -> Tuple[ListType[Path], ListType[Tuple[Path, str]]]:
        """Scan project for Python files with import issues."""
        python_files = []
        error_files = []
        
        for file_path in self.project_root.rglob("*.py"):
            if self.should_process_file(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check for problematic imports
                    has_issues = any(
                        re.search(pattern, content) 
                        for pattern, _ in self.import_patterns
                    )
                    
                    if has_issues:
                        python_files.append(file_path)
                        
                except Exception as e:
                    error_files.append((file_path, str(e)))
                    
        return python_files, error_files
    
    def fix_all_files(self) -> Dict[str, any]:
        """Fix imports in all Python files."""
        logger.info("Scanning for files with import issues...")
        
        files_to_fix, error_files = self.scan_project()
        
        if not files_to_fix:
            logger.info("No files with import issues found!")
            return {
                'files_scanned': 0,
                'files_fixed': 0,
                'files_with_errors': len(error_files),
                'errors': error_files
            }
        
        logger.info(f"Found {len(files_to_fix)} files with import issues")
        
        if self.dry_run:
            logger.info("DRY RUN - No files will be modified")
            for file_path in files_to_fix:
                logger.info(f"Would fix: {file_path}")
            return {
                'files_scanned': len(files_to_fix),
                'files_fixed': 0,
                'files_with_errors': len(error_files),
                'errors': error_files
            }
        
        fixed_files = []
        
        for file_path in files_to_fix:
            if self.fix_file(file_path):
                fixed_files.append(file_path)
        
        return {
            'files_scanned': len(files_to_fix),
            'files_fixed': len(fixed_files),
            'files_with_errors': len(error_files),
            'errors': error_files,
            'fixed_files': fixed_files
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fix typing imports across the Personal Chatter codebase"
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be changed without making modifications'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path(__file__).parent,
        help='Project root directory (default: script directory)'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create fixer and run
    fixer = TypingImportFixer(
        project_root=args.project_root,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    results = fixer.fix_all_files()
    
    # Print summary
    print("\n" + "="*50)
    print("TYPING IMPORT FIXER SUMMARY")
    print("="*50)
    print(f"Files scanned: {results['files_scanned']}")
    print(f"Files fixed: {results['files_fixed']}")
    print(f"Files with errors: {results['files_with_errors']}")
    
    if results.get('fixed_files'):
        print(f"\nFixed files:")
        for file_path in results['fixed_files']:
            print(f"  ✅ {file_path}")
    
    if results['errors']:
        print(f"\nFiles with errors:")
        for file_path, error in results['errors']:
            print(f"  ❌ {file_path}: {error}")
    
    if args.dry_run:
        print(f"\n💡 This was a dry run. Use without --dry-run to apply changes.")
    elif results['files_fixed'] > 0:
        print(f"\n🎉 Successfully fixed {results['files_fixed']} files!")
    else:
        print(f"\n✨ No files needed fixing!")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    fix_typing_imports()