"""
CLI Interface Module

Provides command-line interfaces for the Frai application.
"""

import logging
from front.cli.chat import run_chat_demo

logger = logging.getLogger(__name__)

def initiate_cli():
    """Initialize the CLI interface."""
    try:
        logger.info("CLI interface initialized.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize CLI: {e}")
        return False

def start_chat_cli():
    """Start the chat CLI interface."""
    try:
        logger.info("Starting chat CLI...")
        run_chat_demo()
        return True
    except Exception as e:
        logger.error(f"Error in chat CLI: {e}")
        return False

if __name__ == "__main__":
    start_chat_cli()