"""
Progress tracking utilities for image generation.

This module provides progress callback management and checkpoint
saving functionality for step-by-step image generation.
"""
import logging
import time
import threading
import weakref
from typing import Callable, Optional, Dict, Any
from pathlib import Path
from services.config import get_config

logger = logging.getLogger(__name__)

# Configuration
config = get_config()
memory_config = config.get("memory_management", {})
CHECKPOINT_INTERVAL = memory_config.get("checkpoint_save_interval", 5)
OUTPUT_DIR = "./outputs"

# Global progress tracking
_progress_callbacks = weakref.WeakValueDictionary()  # session_id -> callback function
_progress_lock = threading.Lock()

class ProgressCallback:
    """
    Progress callback class for tracking image generation progress.
    
    This class handles step-by-step progress tracking with optional
    checkpoint saving and external callback integration.
    """
    
    def __init__(self, session_id: str, total_steps: int, 
                 external_callback: Optional[Callable] = None,
                 save_checkpoints: bool = False):
        self.session_id = session_id
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.external_callback = external_callback
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = Path(OUTPUT_DIR) / "checkpoints" / session_id
        
        if self.save_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Created progress callback for session {session_id} with {total_steps} steps")
    
    def __call__(self, step: int, timestep: Any = None, latents: Any = None):
        """
        Progress callback function called during generation.
        
        Args:
            step: Current step number
            timestep: Current timestep (unused but required by diffusers)
            latents: Current latent tensors for checkpoint saving
        """
        self.current_step = step
        elapsed_time = time.time() - self.start_time
        progress = (step / self.total_steps) * 100
        
        # Log progress
        logger.debug(f"Session {self.session_id}: Step {step}/{self.total_steps} ({progress:.1f}%) - {elapsed_time:.1f}s elapsed")
        
        # Save checkpoint if enabled and at interval
        if (self.save_checkpoints and 
            latents is not None and 
            step % CHECKPOINT_INTERVAL == 0 and 
            step > 0):
            self._save_checkpoint(step, latents)
        
        # Call external callback if provided
        if self.external_callback:
            try:
                self.external_callback(step, self.total_steps, progress, elapsed_time)
            except Exception as e:
                logger.error(f"Error in external progress callback: {e}")
    
    def _save_checkpoint(self, step: int, latents: Any):
        """
        Save a checkpoint of the current generation state.
        
        Args:
            step: Current step number
            latents: Current latent tensors to save
        """
        try:
            import torch
            checkpoint_path = self.checkpoint_dir / f"step_{step:03d}.pt"
            torch.save({
                'step': step,
                'session_id': self.session_id,
                'latents': latents,
                'timestamp': time.time()
            }, checkpoint_path)
            logger.debug(f"Saved checkpoint at step {step} to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {step}: {e}")
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get current progress information.
        
        Returns:
            Dictionary containing progress details
        """
        elapsed_time = time.time() - self.start_time
        progress = (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        return {
            "session_id": self.session_id,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": progress,
            "elapsed_time": elapsed_time,
            "estimated_total_time": (elapsed_time / self.current_step * self.total_steps) if self.current_step > 0 else None,
            "save_checkpoints": self.save_checkpoints
        }

def register_progress_callback(session_id: str, callback: ProgressCallback):
    """
    Register a progress callback for a session.
    
    Args:
        session_id: Unique session identifier
        callback: Progress callback instance
    """
    with _progress_lock:
        _progress_callbacks[session_id] = callback
        logger.debug(f"Registered progress callback for session {session_id}")

def unregister_progress_callback(session_id: str):
    """
    Unregister a progress callback for a session.
    
    Args:
        session_id: Session identifier to unregister
    """
    with _progress_lock:
        if session_id in _progress_callbacks:
            del _progress_callbacks[session_id]
            logger.debug(f"Unregistered progress callback for session {session_id}")

def get_progress_callback(session_id: str) -> Optional[ProgressCallback]:
    """
    Get the progress callback for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Progress callback instance or None if not found
    """
    with _progress_lock:
        return _progress_callbacks.get(session_id)

def create_progress_callback(session_id: str, total_steps: int,
                           external_callback: Optional[Callable] = None,
                           save_checkpoints: bool = False) -> ProgressCallback:
    """
    Create and register a new progress callback.
    
    Args:
        session_id: Unique session identifier
        total_steps: Total number of generation steps
        external_callback: Optional external callback function
        save_checkpoints: Whether to save intermediate checkpoints
        
    Returns:
        Created progress callback instance
    """
    callback = ProgressCallback(
        session_id=session_id,
        total_steps=total_steps,
        external_callback=external_callback,
        save_checkpoints=save_checkpoints
    )
    
    register_progress_callback(session_id, callback)
    return callback

def get_all_active_sessions() -> Dict[str, Dict[str, Any]]:
    """
    Get progress information for all active sessions.
    
    Returns:
        Dictionary mapping session IDs to their progress info
    """
    with _progress_lock:
        return {
            session_id: callback.get_progress_info()
            for session_id, callback in _progress_callbacks.items()
        }
