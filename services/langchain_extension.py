"""
LangChain integration for the Personal Chatter image generation system.

This module provides LangChain-compatible wrappers for using the Flux.1 image
generation capabilities within LangChain pipelines, agents, and chains.
"""
import logging
import time
import uuid
from typing import Dict, Optional, Any, Union
from test_mock_helper import List

logger = logging.getLogger(__name__)

# Import the image generation functionality
from services.image_generator import (
    generate_image,
    generate_image_async,
    validate_prompt,
    get_model_status
)

try:
    # Try to import LangChain classes if available
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.schema import Generation, LLMResult
    
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain libraries detected - full integration available")
    
    class FluxImageGenerationLLM(LLM):
        """
        LangChain-compatible LLM wrapper for Flux.1 image generation.
        
        This allows Flux.1 to be used within LangChain pipelines and chains
        for image generation tasks.
        
        Example:
            from services.langchain_extension import FluxImageGenerationLLM
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            
            # Create the LangChain-compatible image generator
            image_llm = FluxImageGenerationLLM(
                height=768,
                width=768
            )
            
            # Create a chain that generates images
            template = "Generate an image of {subject} with {style} style."
            prompt = PromptTemplate(template=template, input_variables=["subject", "style"])
            chain = LLMChain(prompt=prompt, llm=image_llm)
            
            # Generate an image
            result = chain.run(subject="a majestic mountain", style="photorealistic")
        """
        
        height: int = 1024
        width: int = 1024
        steps: int = 35
        guidance_scale: float = 7.0
        negative_prompt: Optional[str] = None
        
        class Config:
            """Configuration for this pydantic object."""
            arbitrary_types_allowed = True
        
        @property
        def _llm_type(self) -> str:
            return "flux_image_generator"
        
        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs
        ) -> str:
            """
            Generate an image and return the path/URL.
            
            Args:
                prompt: The image generation prompt
                stop: Not used for image generation
                run_manager: LangChain callback manager
                **kwargs: Additional generation parameters
                
            Returns:
                String containing the image path and URL
            """
            # Extract parameters from kwargs
            height = kwargs.get("height", self.height)
            width = kwargs.get("width", self.width)
            steps = kwargs.get("steps", self.steps)
            guidance_scale = kwargs.get("guidance_scale", self.guidance_scale)
            negative_prompt = kwargs.get("negative_prompt", self.negative_prompt)
            session_id = kwargs.get("session_id", f"langchain_{int(time.time())}")
            
            # Validate prompt
            valid, message, suggested_prompt = validate_prompt(prompt)
            if not valid and suggested_prompt:
                if run_manager:
                    run_manager.on_text(f"⚠️ {message}. Using suggested prompt instead.", verbose=True)
                prompt = suggested_prompt
            elif not valid:
                if run_manager:
                    run_manager.on_text(f"⚠️ {message}", verbose=True)
            
            # Progress callback integration with LangChain
            progress_callback = None
            if run_manager:
                def langchain_progress_callback(step, total_steps, progress, elapsed_time):
                    run_manager.on_text(
                        f"Image generation progress: {progress:.1f}% (step {step}/{total_steps})",
                        verbose=True
                    )
                progress_callback = langchain_progress_callback
            
            # Generate the image
            image_path, image_url = generate_image(
                prompt=prompt,
                height=height,
                width=width,
                steps=steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                session_id=session_id,
                progress_callback=progress_callback
            )
            
            if image_path:
                return f"Generated image: {image_url} (saved to: {image_path})"
            else:
                return "Failed to generate image"
        
        async def _acall(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs
        ) -> str:
            """
            Async version of _call.
            
            Args:
                prompt: The image generation prompt
                stop: Not used for image generation
                run_manager: LangChain callback manager
                **kwargs: Additional generation parameters
                
            Returns:
                String containing the image path and URL
            """
            # Extract parameters from kwargs
            height = kwargs.get("height", self.height)
            width = kwargs.get("width", self.width)
            steps = kwargs.get("steps", self.steps)
            guidance_scale = kwargs.get("guidance_scale", self.guidance_scale)
            negative_prompt = kwargs.get("negative_prompt", self.negative_prompt)
            session_id = kwargs.get("session_id", f"langchain_async_{int(time.time())}")
            
            # Progress callback integration with LangChain
            progress_callback = None
            if run_manager:
                def langchain_progress_callback(step, total_steps, progress, elapsed_time):
                    run_manager.on_text(
                        f"Image generation progress: {progress:.1f}% (step {step}/{total_steps})",
                        verbose=True
                    )
                progress_callback = langchain_progress_callback
            
            # Generate the image asynchronously
            image_path, image_url = await generate_image_async(
                prompt=prompt,
                height=height,
                width=width,
                steps=steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                session_id=session_id,
                progress_callback=progress_callback
            )
            
            if image_path:
                return f"Generated image: {image_url} (saved to: {image_path})"
            else:
                return "Failed to generate image"

except ImportError:
    # Create a minimal compatible class for the demo
    LANGCHAIN_AVAILABLE = False
    logger.info("LangChain not installed - using minimal mock implementation")
    
    class FluxImageGenerationLLM:
        """
        Mock LangChain-compatible wrapper for Flux.1 image generation.
        
        This class provides the basic interface to ensure compatibility with the demo.
        Install LangChain for full functionality.
        """
        
        def __init__(self, height=1024, width=1024, steps=35, guidance_scale=7.0, **kwargs):
            self.height = height
            self.width = width
            self.steps = steps
            self.guidance_scale = guidance_scale
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        @property
        def _llm_type(self):
            return "flux_image_generator"
        
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            """
            Generate an image and return the path/URL.
            """
            # Try to use the real implementation if available
            try:
                # Extract parameters from kwargs
                height = kwargs.get("height", self.height)
                width = kwargs.get("width", self.width)
                steps = kwargs.get("steps", self.steps)
                guidance_scale = kwargs.get("guidance_scale", self.guidance_scale)
                negative_prompt = kwargs.get("negative_prompt", None)
                session_id = kwargs.get("session_id", None)
                
                # Generate the image
                image_path, image_url = generate_image(
                    prompt=prompt,
                    height=height,
                    width=width,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    session_id=session_id
                )
                
                if image_path:
                    return f"Generated image: {image_url} (saved to: {image_path})"
                
            except Exception as e:
                logger.debug(f"Mock LLM encountered an error (normal in demo): {e}")
            
            # Return a mock result
            return f"Generated image: /outputs/mock_image_{uuid.uuid4().hex[:8]}.png (demo/mock)"
        
        async def _acall(self, prompt, stop=None, run_manager=None, **kwargs):
            """Async version of _call (currently just calls sync version)."""
            return self._call(prompt, stop, run_manager, **kwargs)

# Export the LangChain classes
__all__ = ["FluxImageGenerationLLM", "LANGCHAIN_AVAILABLE"]
