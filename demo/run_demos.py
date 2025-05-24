#!/usr/bin/env python3
"""
Demo Runner for Personal Chatter

Executes all or selected demo scripts to showcase functionality
and verify that core features work properly.
"""

import argparse
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo_runner")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DemoRunner:
    """Discovers and runs demo scripts from the demo directory."""
    
    def __init__(self, demo_dir: Path):
        self.demo_dir = demo_dir
        self.demos: Dict[str, Path] = {}
        self.discover_demos()
    
    def discover_demos(self):
        """Find all demo scripts in the demo directory."""
        logger.info(f"Discovering demos in {self.demo_dir}")
        
        for file_path in self.demo_dir.glob("*.py"):
            # Skip this runner script and any hidden files or non-demo files
            if file_path.name == "run_demos.py" or file_path.name.startswith("_"):
                continue
                
            # Extract demo name
            demo_name = file_path.stem
            self.demos[demo_name] = file_path
            
        logger.info(f"Found {len(self.demos)} demos: {', '.join(self.demos.keys())}")
    
    def list_demos(self) -> List[str]:
        """Return a sorted list of available demo names."""
        return sorted(self.demos.keys())
    
    def run_demo(self, demo_name: str, args: Optional[List[str]] = None) -> bool:
        """Run a specific demo by name."""
        if demo_name not in self.demos:
            logger.error(f"Demo '{demo_name}' not found")
            return False
        
        demo_path = self.demos[demo_name]
        logger.info(f"Running demo: {demo_name} from {demo_path}")
        
        # Print a visual separator
        print("\n" + "=" * 80)
        print(f" DEMO: {demo_name} ".center(80, "="))
        print("=" * 80 + "\n")
        
        try:
            # Load the demo module
            spec = importlib.util.spec_from_file_location(demo_name, demo_path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to load spec for {demo_path}")
                return False
                
            demo_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(demo_module)
            
            # Run the demo's main function if it exists
            if hasattr(demo_module, "main"):
                if args:
                    demo_module.main(args)
                else:
                    demo_module.main()
            else:
                logger.warning(f"Demo {demo_name} has no main() function")
            
            print("\n" + "-" * 80)
            print(f" Demo {demo_name} completed successfully ".center(80, "-"))
            print("-" * 80 + "\n")
            return True
            
        except Exception as e:
            logger.error(f"Error running demo {demo_name}: {e}", exc_info=True)
            print("\n" + "!" * 80)
            print(f" Demo {demo_name} failed: {e} ".center(80, "!"))
            print("!" * 80 + "\n")
            return False
    
    def run_all_demos(self, exclude: Optional[List[str]] = None) -> Dict[str, bool]:
        """Run all available demos and return results."""
        exclude = exclude or []
        results = {}
        
        logger.info(f"Running all demos (excluding: {', '.join(exclude) if exclude else 'none'})")
        
        for demo_name in self.list_demos():
            if demo_name in exclude:
                logger.info(f"Skipping excluded demo: {demo_name}")
                continue
                
            result = self.run_demo(demo_name)
            results[demo_name] = result
            
            # Add a small delay between demos
            time.sleep(1)
            
        return results


def main():
    """Main entry point for the demo runner."""
    parser = argparse.ArgumentParser(
        description="Personal Chatter Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available demos"
    )
    
    parser.add_argument(
        "--demo", "-d",
        help="Run a specific demo by name"
    )
    
    parser.add_argument(
        "--exclude", "-e",
        action="append",
        help="Exclude specific demos when running all"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all demos"
    )
    
    parser.add_argument(
        "args",
        nargs="*",
        help="Additional arguments to pass to demos"
    )
    
    args = parser.parse_args()
    
    # Initialize demo runner
    demo_dir = Path(__file__).parent
    runner = DemoRunner(demo_dir)
    
    # List demos if requested
    if args.list:
        print("Available demos:")
        for demo in runner.list_demos():
            print(f"  - {demo}")
        return 0
    
    # Run a specific demo if requested
    if args.demo:
        success = runner.run_demo(args.demo, args.args)
        return 0 if success else 1
    
    # Run all demos if requested or if no specific action
    if args.all or (not args.list and not args.demo):
        results = runner.run_all_demos(args.exclude)
        
        # Print summary
        print("\nDemo Execution Summary:")
        print("=" * 50)
        
        success_count = sum(1 for success in results.values() if success)
        total_demos = len(results)
        
        for demo_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{demo_name:30} {status}")
        
        print("-" * 50)
        print(f"Total: {total_demos}, Passed: {success_count}, Failed: {total_demos - success_count}")
        print("=" * 50)
        
        # Return failure if any demo failed
        return 0 if all(results.values()) else 1
    
    # If we get here, no action was taken
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
