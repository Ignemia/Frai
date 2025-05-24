"""
Feedback-enabled image generation interface.

Provides image generation with sentiment analysis and feedback integration
for improved results based on user preferences.
"""
import logging
from typing import Dict, Optional, Any, Tuple, Callable

from .sync_generator import SyncImageGenerator
from ..style_presets import StylePreset

logger = logging.getLogger(__name__)


class FeedbackImageGenerator:
    """Image generation interface with feedback and sentiment analysis."""
    
    def __init__(self):
        """Initialize the feedback-enabled image generator."""
        self._sync_generator = SyncImageGenerator()
    
    def generate_with_feedback(
        self,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        style: Optional[StylePreset] = None,
        session_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Generate an image with sentiment analysis and feedback integration.
        
        Args:
            prompt: The text prompt for image generation
            height: Image height in pixels
            width: Image width in pixels
            steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            negative_prompt: Negative prompt to guide what to avoid
            style: Optional style preset to apply
            session_id: Optional session ID for progress tracking
            progress_callback: Optional callback function for progress updates
            user_feedback: Optional feedback from previous generations
            
        Returns:
            Tuple containing:
                - Path to the generated image if successful, None otherwise
                - URL or relative path to the image for displaying
                - Analysis results and suggestions for improvement
        """
        # Analyze feedback and enhance prompt if provided
        enhanced_prompt = prompt
        analysis_results = None
        
        if user_feedback:
            try:
                enhanced_prompt, analysis_results = self._process_feedback(prompt, user_feedback)
                if enhanced_prompt != prompt:
                    logger.info("Applied feedback-based prompt enhancement")
            except Exception as e:
                logger.warning(f"Error processing user feedback: {e}")
        
        # Generate the image with the enhanced prompt
        image_path, image_url = self._sync_generator.generate(
            enhanced_prompt,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            style=style,
            session_id=session_id,
            progress_callback=progress_callback
        )
        
        return image_path, image_url, analysis_results
    
    def _process_feedback(self, prompt: str, user_feedback: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Process user feedback and enhance the prompt.
        
        Args:
            prompt: Original prompt
            user_feedback: User feedback data
            
        Returns:
            Tuple of (enhanced_prompt, analysis_results)
        """
        try:
            from ..sentiment_feedback import analyze_feedback, suggest_prompt_improvements
            
            # Analyze the feedback
            feedback_analysis = analyze_feedback(user_feedback)
            
            # Get prompt suggestions based on feedback
            suggestions = suggest_prompt_improvements(prompt, feedback_analysis)
            enhanced_prompt = suggestions.get("enhanced_prompt", prompt)
            
            return enhanced_prompt, feedback_analysis
            
        except ImportError:
            logger.warning("Sentiment feedback module not available")
            return prompt, {"error": "Sentiment feedback not available"}
        except Exception as e:
            logger.error(f"Error in feedback processing: {e}", exc_info=True)
            return prompt, {"error": str(e)}
    
    def analyze_feedback_only(self, user_feedback: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze user feedback without generating an image.
        
        Args:
            user_feedback: User feedback data
            
        Returns:
            Analysis results or None if analysis failed
        """
        try:
            from ..sentiment_feedback import analyze_feedback
            return analyze_feedback(user_feedback)
        except ImportError:
            logger.warning("Sentiment feedback module not available")
            return {"error": "Sentiment feedback not available"}
        except Exception as e:
            logger.error(f"Error analyzing feedback: {e}", exc_info=True)
            return {"error": str(e)}
    
    def suggest_improvements(self, prompt: str, feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get prompt improvement suggestions based on feedback analysis.
        
        Args:
            prompt: Original prompt
            feedback_analysis: Results from feedback analysis
            
        Returns:
            Dictionary with improvement suggestions
        """
        try:
            from ..sentiment_feedback import suggest_prompt_improvements
            return suggest_prompt_improvements(prompt, feedback_analysis)
        except ImportError:
            logger.warning("Sentiment feedback module not available")
            return {"error": "Sentiment feedback not available"}
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}", exc_info=True)
            return {"error": str(e)}
    
    # Delegate other methods to the sync generator
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the image generation system."""
        return self._sync_generator.get_status()
    
    def unload(self) -> bool:
        """Unload the model from memory."""
        return self._sync_generator.unload()
    
    def reload(self) -> bool:
        """Reload the model."""
        return self._sync_generator.reload()
