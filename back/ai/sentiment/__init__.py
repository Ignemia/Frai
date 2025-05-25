"""
Backend Sentiment Analysis AI Module

Users can send text to this module and it will return the sentiment of the text as a score between -1 for negative and 1 for positive.

__init__.py should contain class definition for the sentiment handler with its global instance.
pipeline.py should contain the pipeline for the sentiment handler.
prompt_handler.py should contain the prompt handler for the sentiment handler.
other utility files might be needed.

For sentiment analysis we use multilinguage-sentiment-analysis model from models/multilingual-sentiment-analysis.
"""

import logging
import os
from typing import Dict, Optional, Any, List
import torch

from .pipeline import load_sentiment_pipeline, run_sentiment_analysis
# prompt_handler might not be strictly necessary if the model takes raw text,
# but kept for consistency if complex pre-processing is ever needed.
from .prompt_handler import format_sentiment_input

logger = logging.getLogger(__name__)

class SentimentAI:
    """
    AI sentiment analysis handler that coordinates model loading, analysis, and device management.
    The model is loaded into RAM on initialization.
    For analysis, it's moved to VRAM (if available) and then back to RAM.
    """
    
    def __init__(self, model_name: str = "multilingual-sentiment-analysis", model_path: str = "models/multilingual-sentiment-analysis"):
        self.model_name = model_name
        self.model_path = model_path # Actual path to the model files or Hugging Face identifier
        self.pipeline = None # This will hold the sentiment analysis pipeline
        self.is_loaded = False
        self.vram_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ram_device = "cpu"

        # No specific generation_params needed like in text/image generation usually
        # self.analysis_params = {} 

        self._load_model_to_ram()

    def _load_model_to_ram(self):
        logger.info(f"Attempting to load sentiment model {self.model_name} to RAM.")
        try:
            # This will use the function from pipeline.py
            self.pipeline = load_sentiment_pipeline(self.model_path, device=self.ram_device)
            if self.pipeline:
                self.is_loaded = True
                logger.info(f"Sentiment model {self.model_name} (pipeline) loaded to RAM successfully.")
                # Enable sequential CPU offloading if applicable and model is large, 
                # though for many sentiment models this might not be necessary or available directly on pipeline object.
                # if hasattr(self.pipeline, 'model') and hasattr(self.pipeline.model, 'enable_sequential_cpu_offload'):
                #     logger.info("Enabling sequential CPU offloading for the sentiment model.")
                #     self.pipeline.model.enable_sequential_cpu_offload(device="cpu")
            else:
                raise RuntimeError(f"Sentiment pipeline loading returned None.")
        except Exception as e:
            self.is_loaded = False
            logger.error(f"Failed to load sentiment model {self.model_name} to RAM: {e}")
            raise RuntimeError(f"Could not load sentiment model {self.model_name} to RAM: {e}")

    def _ensure_model_on_device(self, target_device: str) -> bool:
        if not self.is_loaded or self.pipeline is None:
            logger.error("Sentiment model/pipeline not loaded, cannot move device.")
            return False
        try:
            # For Hugging Face pipelines, moving the underlying model to the device is key.
            # The pipeline itself is often re-instantiated or its device param is updated.
            if self.pipeline.device.type != target_device:
                logger.info(f"Moving sentiment model from {self.pipeline.device} to {target_device}.")
                # Re-assigning the pipeline with the new device, or moving the model component
                # This depends on how load_sentiment_pipeline is implemented.
                # For simplicity, we assume load_sentiment_pipeline can take a device argument effectively.
                # A more robust way is to move self.pipeline.model.to(target_device) and update pipeline's device.
                self.pipeline.model.to(target_device)
                self.pipeline.device = torch.device(target_device)
                # If it's a full pipeline object, some might need to be recreated or have a .to() method
                # e.g. self.pipeline.to(torch.device(target_device))
                logger.info(f"Sentiment model moved to {target_device}.")
            return True
        except Exception as e:
            logger.error(f"Failed to move sentiment model to {target_device}: {e}")
            return False

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the given text.
        Manages device VRAM/RAM transfers for the model.
        
        Args:
            text: The text to analyze.

        Returns:
            Dictionary with analysis results, including "score", "label", "success", "error", "metadata".
        """
        if not self.is_loaded or self.pipeline is None:
            return {"success": False, "error": "Sentiment model not loaded", "score": None, "label": "", "metadata": {}}

        target_device_for_analysis = self.vram_device
        
        # format_sentiment_input could be used if pre-processing is complex
        # For many sentiment models, raw text is fine.
        # formatted_input = format_sentiment_input(text) 
        input_text = text # Assuming direct text input

        try:
            if not self._ensure_model_on_device(target_device_for_analysis):
                raise RuntimeError(f"Failed to move sentiment model to {target_device_for_analysis}.")

            logger.info(f"Analyzing sentiment on {target_device_for_analysis}...")
            
            # Use the function from pipeline.py
            analysis_result = run_sentiment_analysis(self.pipeline, input_text)
            
            # The result from run_sentiment_analysis should be a dict 
            # e.g. {"label": "POSITIVE", "score": 0.99}
            # We adapt it to our standard response format.
            if analysis_result and isinstance(analysis_result, list):
                # Standard HF pipeline output is a list of dicts
                result_data = analysis_result[0] 
            elif analysis_result and isinstance(analysis_result, dict):
                result_data = analysis_result # If run_sentiment_analysis already formats it
            else:
                raise ValueError("Sentiment analysis did not return expected dict or list of dicts.")

            response = {
                "success": True, 
                "label": result_data.get("label"), 
                "score": result_data.get("score"),
                "error": None,
                "metadata": {"model_name": self.model_name, "device_used": target_device_for_analysis}
            }
            return response
        
        except Exception as e:
            logger.error(f"Error during sentiment analysis with device handling: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "score": None, "label": "", "metadata": {}}
        finally:
            logger.info(f"Attempting to move sentiment model back to {self.ram_device}.")
            if self.pipeline is not None:
                if not self._ensure_model_on_device(self.ram_device):
                    logger.error(f"CRITICAL: Failed to move sentiment model back to {self.ram_device} after analysis.")
                else:
                    logger.info(f"Sentiment model successfully moved back to {self.ram_device}.")
            if target_device_for_analysis == "cuda":
                # No specific pipeline components to delete like in generation, but good to clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# --- Global Instance and Accessor Functions ---
_sentiment_ai_instance: Optional[SentimentAI] = None

def get_sentiment_ai_instance(model_name: Optional[str] = None, model_path: Optional[str] = None) -> SentimentAI:
    global _sentiment_ai_instance
    default_model_name = "multilingual-sentiment-analysis"
    default_model_path = "models/multilingual-sentiment-analysis"
    
    if _sentiment_ai_instance is None:
        logger.info("Initializing global SentimentAI instance.")
        _sentiment_ai_instance = SentimentAI(
            model_name=model_name if model_name else default_model_name,
            model_path=model_path if model_path else default_model_path
        )
    elif model_name and (_sentiment_ai_instance.model_name != model_name or \
                        (model_path and _sentiment_ai_instance.model_path != model_path)):
        logger.warning(
            f"Requesting SentimentAI with new model/path {model_name}/{model_path}, but instance with "
            f"{_sentiment_ai_instance.model_name}/{_sentiment_ai_instance.model_path} exists. Re-initializing."
        )
        _sentiment_ai_instance = SentimentAI(model_name=model_name, model_path=model_path)
    return _sentiment_ai_instance

def initialize_sentiment_system(model_name: Optional[str] = None, model_path: Optional[str] = None) -> bool:
    try:
        logger.info("Initializing sentiment analysis system...")
        sentiment_ai = get_sentiment_ai_instance(model_name, model_path)
        if sentiment_ai.is_loaded:
            logger.info("Sentiment analysis system initialized successfully. Model is in RAM.")
            return True
        else:
            logger.error("Sentiment analysis system initialization failed. Model could not be loaded.")
            return False
    except Exception as e:
        logger.error(f"Error during sentiment system initialization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# High-level interaction function
def get_text_sentiment(text: str) -> Dict[str, Any]:
    """
    High-level function to get the sentiment of a given text.
    """
    sentiment_ai = get_sentiment_ai_instance()
    if not sentiment_ai.is_loaded:
         return {"success": False, "error": "Sentiment model not ready or failed to load.", "score": None, "label": ""}
    return sentiment_ai.analyze_sentiment(text=text)