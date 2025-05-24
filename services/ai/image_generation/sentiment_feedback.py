"""
from typing import List
Sentiment analysis feedback system for image refinement.

This module analyzes user feedback on generated images and provides
intelligent suggestions for prompt improvements and regeneration strategies.
"""
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch

from services.config import get_config

logger = logging.getLogger(__name__)

# Load configuration
config = get_config()
MODEL_PATH = config.get("models", {}).get("sentiment_path", "./models/multilingual-sentiment-analysis")

# Global sentiment model
_sentiment_pipeline = None

class FeedbackAction:
    """Represents an action to take based on sentiment analysis."""
    COMPLETE_REDO = "complete_redo"
    PARTIAL_ADJUSTMENT = "partial_adjustment" 
    MINOR_TWEAKS = "minor_tweaks"
    ASK_REGENERATION = "ask_regeneration"
    ACCEPT = "accept"

class SentimentFeedback:
    """Container for sentiment analysis results and recommendations."""
    
    def __init__(self, text: str, score: float, label: str, confidence: float):
        self.text = text
        self.score = score  # 0.0 to 1.0, where 1.0 is most positive
        self.label = label  # POSITIVE, NEGATIVE, NEUTRAL
        self.confidence = confidence
        self.action = self._determine_action()
        self.recommendations = self._generate_recommendations()
    
    def _determine_action(self) -> str:
        """Determine what action to take based on sentiment score."""
        if self.score < 0.6:
            return FeedbackAction.COMPLETE_REDO
        elif self.score < 0.75:
            return FeedbackAction.PARTIAL_ADJUSTMENT
        elif self.score < 0.95:
            return FeedbackAction.MINOR_TWEAKS
        elif self.score < 0.98:
            return FeedbackAction.ASK_REGENERATION
        else:
            return FeedbackAction.ACCEPT
    
    def _generate_recommendations(self) -> List[str]:
        """Generate specific recommendations based on sentiment and action."""
        recommendations = []
        
        if self.action == FeedbackAction.COMPLETE_REDO:
            recommendations.extend([
                "The current image doesn't meet expectations. Let's try a completely different approach.",
                "Consider changing the main subject, style, or composition.",
                "Would you like to try a different style preset?",
                "Should we adjust the core concept of the image?"
            ])
        
        elif self.action == FeedbackAction.PARTIAL_ADJUSTMENT:
            recommendations.extend([
                "The image is on the right track but needs improvements.",
                "Consider adjusting colors, lighting, or composition.",
                "We could refine the style or add more detail.",
                "Would you like to try with different parameters?"
            ])
        
        elif self.action == FeedbackAction.MINOR_TWEAKS:
            recommendations.extend([
                "The image is quite good! Just minor adjustments needed.",
                "Small changes to lighting, colors, or details could help.",
                "Consider slight prompt modifications for better results.",
                "We're close to the ideal result."
            ])
        
        elif self.action == FeedbackAction.ASK_REGENERATION:
            recommendations.extend([
                "This image is very close to perfect!",
                "Would you like me to generate a few more variations?",
                "Small adjustments might give us the perfect result.",
                "The current approach is working well."
            ])
        
        else:  # ACCEPT
            recommendations.extend([
                "Excellent! This image meets your expectations perfectly.",
                "The generation was successful - no changes needed.",
                "Great result! Would you like to create similar images?",
                "Perfect outcome! This approach worked very well."
            ])
        
        return recommendations

def _get_sentiment_pipeline():
    """
    Get or initialize the sentiment analysis pipeline.
    
    Returns:
        The sentiment analysis pipeline or None if initialization failed
    """
    global _sentiment_pipeline
    
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    
    try:
        logger.info("Initializing sentiment analysis pipeline")
        
        # Check if model path exists
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Sentiment model not found at {MODEL_PATH}")
            logger.info("Using fallback sentiment analysis")
            return None
        
        # Import here to avoid loading dependencies unless needed
        from transformers import pipeline
        
        # Load the multilingual sentiment analysis model
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=MODEL_PATH,
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=True
        )
        
        logger.info("Sentiment analysis pipeline loaded successfully")
        return _sentiment_pipeline
        
    except Exception as e:
        logger.error(f"Error initializing sentiment pipeline: {e}")
        return None

def analyze_feedback(feedback_text: str) -> SentimentFeedback:
    """
    Analyze user feedback and provide actionable insights.
    
    Args:
        feedback_text: The user's feedback about the generated image
        
    Returns:
        SentimentFeedback object with analysis and recommendations
    """
    pipeline = _get_sentiment_pipeline()
    
    if pipeline is None:
        # Fallback sentiment analysis using simple keyword matching
        return _fallback_sentiment_analysis(feedback_text)
    
    try:
        # Get sentiment scores
        results = pipeline(feedback_text)
        
        # Find the most confident prediction
        best_result = max(results[0], key=lambda x: x['score'])
        
        # Convert to 0-1 scale where 1.0 is most positive
        if best_result['label'].upper() == 'POSITIVE':
            score = best_result['score']
        elif best_result['label'].upper() == 'NEGATIVE':
            score = 1.0 - best_result['score']
        else:  # NEUTRAL
            score = 0.5
        
        return SentimentFeedback(
            text=feedback_text,
            score=score,
            label=best_result['label'],
            confidence=best_result['score']
        )
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        return _fallback_sentiment_analysis(feedback_text)

def _fallback_sentiment_analysis(feedback_text: str) -> SentimentFeedback:
    """
    Fallback sentiment analysis using keyword matching.
    
    Args:
        feedback_text: The user's feedback text
        
    Returns:
        SentimentFeedback object with basic analysis
    """
    text_lower = feedback_text.lower()
    
    # Positive keywords
    positive_keywords = [
        'good', 'great', 'excellent', 'perfect', 'amazing', 'wonderful',
        'beautiful', 'fantastic', 'awesome', 'love', 'like', 'nice',
        'impressive', 'stunning', 'brilliant', 'superb', 'magnificent'
    ]
    
    # Negative keywords
    negative_keywords = [
        'bad', 'terrible', 'awful', 'hate', 'dislike', 'wrong', 'ugly',
        'horrible', 'disgusting', 'poor', 'disappointing', 'failed',
        'worse', 'worst', 'sucks', 'rubbish', 'garbage'
    ]
    
    # Neutral keywords
    neutral_keywords = [
        'okay', 'ok', 'fine', 'average', 'normal', 'standard', 'regular'
    ]
    
    positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    neutral_count = sum(1 for keyword in neutral_keywords if keyword in text_lower)
    
    total_keywords = positive_count + negative_count + neutral_count
    
    if total_keywords == 0:
        # No clear sentiment indicators, assume neutral
        score = 0.5
        label = "NEUTRAL"
        confidence = 0.3
    else:
        # Calculate sentiment score
        if positive_count > negative_count:
            score = 0.5 + (positive_count / (total_keywords * 2))
            label = "POSITIVE"
            confidence = min(0.8, positive_count / max(1, total_keywords))
        elif negative_count > positive_count:
            score = 0.5 - (negative_count / (total_keywords * 2))
            label = "NEGATIVE"
            confidence = min(0.8, negative_count / max(1, total_keywords))
        else:
            score = 0.5
            label = "NEUTRAL"
            confidence = 0.5
    
    return SentimentFeedback(
        text=feedback_text,
        score=max(0.0, min(1.0, score)),  # Clamp to 0-1 range
        label=label,
        confidence=confidence
    )

def suggest_prompt_improvements(
    original_prompt: str, 
    feedback: SentimentFeedback,
    feedback_text: str
) -> List[str]:
    """
    Suggest specific prompt improvements based on sentiment analysis.
    
    Args:
        original_prompt: The original prompt used for generation
        feedback: SentimentFeedback object from analysis
        feedback_text: The original feedback text for context
        
    Returns:
        List of suggested prompt improvements
    """
    suggestions = []
    feedback_lower = feedback_text.lower()
    
    # Analyze specific issues mentioned in feedback
    if any(word in feedback_lower for word in ['color', 'colours', 'coloring']):
        suggestions.append("Add more specific color descriptions to the prompt")
        suggestions.append("Try phrases like 'vibrant colors', 'warm tones', or 'cool palette'")
    
    if any(word in feedback_lower for word in ['lighting', 'light', 'bright', 'dark']):
        suggestions.append("Improve lighting descriptions in the prompt")
        suggestions.append("Add terms like 'dramatic lighting', 'soft lighting', or 'natural light'")
    
    if any(word in feedback_lower for word in ['detail', 'details', 'sharp', 'blur']):
        suggestions.append("Add detail-enhancing keywords to the prompt")
        suggestions.append("Include terms like 'highly detailed', 'sharp focus', 'intricate details'")
    
    if any(word in feedback_lower for word in ['style', 'artistic', 'art']):
        suggestions.append("Consider using a different style preset")
        suggestions.append("Add more specific artistic style descriptions")
    
    if any(word in feedback_lower for word in ['composition', 'framing', 'angle']):
        suggestions.append("Improve composition descriptions in the prompt")
        suggestions.append("Add camera angle or framing specifications")
    
    # Action-specific suggestions
    if feedback.action == FeedbackAction.COMPLETE_REDO:
        suggestions.extend([
            "Consider completely rewriting the prompt with a different approach",
            "Try a different style preset (Riot Games, Realistic, or Anime)",
            "Change the main subject or scene described in the prompt"
        ])
    
    elif feedback.action == FeedbackAction.PARTIAL_ADJUSTMENT:
        suggestions.extend([
            "Keep the main concept but refine specific details",
            "Add more descriptive adjectives to key elements",
            "Adjust the balance between positive and negative prompts"
        ])
    
    elif feedback.action == FeedbackAction.MINOR_TWEAKS:
        suggestions.extend([
            "Make small additions to enhance existing elements",
            "Add quality-enhancing keywords like 'masterpiece', 'high quality'",
            "Fine-tune specific aspects mentioned in the feedback"
        ])
    
    # Remove duplicates while preserving order
    unique_suggestions = []
    for suggestion in suggestions:
        if suggestion not in unique_suggestions:
            unique_suggestions.append(suggestion)
    
    return unique_suggestions[:5]  # Return top 5 suggestions

def generate_feedback_response(feedback: SentimentFeedback) -> str:
    """
    Generate a natural language response to user feedback.
    
    Args:
        feedback: SentimentFeedback object from analysis
        
    Returns:
        Natural language response string
    """
    responses = {
        FeedbackAction.COMPLETE_REDO: [
            "I understand this image didn't meet your expectations. Let's try a completely different approach!",
            "No worries! Let's go back to the drawing board and create something better.",
            "I see this isn't what you were looking for. Let's redesign this from scratch."
        ],
        
        FeedbackAction.PARTIAL_ADJUSTMENT: [
            "Thanks for the feedback! The image is heading in the right direction but needs some improvements.",
            "I can see what you mean. Let's make some adjustments to get closer to what you want.",
            "Good point! Let's refine this image with some targeted changes."
        ],
        
        FeedbackAction.MINOR_TWEAKS: [
            "Great! We're very close to the perfect result. Just a few small tweaks needed.",
            "Excellent! The image is almost there. Let's make some fine adjustments.",
            "Perfect! Just minor improvements and we'll have exactly what you want."
        ],
        
        FeedbackAction.ASK_REGENERATION: [
            "Wonderful! This is very close to perfect. Would you like me to create a few variations?",
            "Fantastic! The result is nearly ideal. Should I generate some similar options?",
            "Great result! This approach is working well. Want to see more like this?"
        ],
        
        FeedbackAction.ACCEPT: [
            "Excellent! I'm so glad this image turned out perfectly for you!",
            "Perfect! This generation was a complete success. Great result!",
            "Amazing! This image captured exactly what you wanted. Well done!"
        ]
    }
    
    import random
    return random.choice(responses[feedback.action])

def get_feedback_statistics() -> Dict[str, Any]:
    """
    Get statistics about feedback processing (for debugging/monitoring).
    
    Returns:
        Dictionary with feedback processing statistics
    """
    return {
        "sentiment_model_loaded": _sentiment_pipeline is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH) if MODEL_PATH else False,
        "feedback_actions": {
            "complete_redo": "< 60% positive sentiment",
            "partial_adjustment": "60-75% positive sentiment", 
            "minor_tweaks": "75-95% positive sentiment",
            "ask_regeneration": "95-98% positive sentiment",
            "accept": "> 98% positive sentiment"
        }
    }
