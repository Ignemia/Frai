"""Handles text generation using the AI model."""
import logging
import torch
from typing import Dict, Any
from datetime import datetime


logger = logging.getLogger(__name__)

def generate_text(
    generation_pipeline, 
    tokenizer, # Added tokenizer for pad_token_id and eos_token_id
    conversation_context: str, 
    generation_params: Dict[str, Any],
    max_new_tokens: int = 8192 # Added from original generate_response
) -> Dict[str, Any]:
    """Generates text using the model pipeline with robust fallbacks."""
    try:
        logger.info("Generating AI response...")
        start_time = datetime.now()

        # Ensure generation_pipeline is not None
        if generation_pipeline is None:
            logger.error("Generation pipeline is not initialized.")
            return {
                "success": False,
                "error": "Generation pipeline not initialized",
                "response": "",
                "metadata": {}
            }

        # Extract model for direct generation if needed
        model = generation_pipeline.model if hasattr(generation_pipeline, 'model') else None

        # Try pipeline generation first with minimal parameters
        gen_kwargs = {
            "max_new_tokens": min(max_new_tokens, 20),
            "do_sample": False,
            "return_full_text": False,
            "clean_up_tokenization_spaces": True,
            "temperature": None,  # Disable temperature for greedy decoding
            "top_p": None,       # Disable top_p for greedy decoding
            "top_k": None,       # Disable top_k for greedy decoding
        }

        # Add pad_token_id only if available
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

        try:
            logger.info("Attempting pipeline generation with minimal parameters")
            outputs = generation_pipeline(conversation_context, **gen_kwargs)
            
            # Extract generated text
            generated_text = ""
            if outputs and len(outputs) > 0 and "generated_text" in outputs[0]:
                generated_text = outputs[0]["generated_text"].strip()
            
            if generated_text:
                logger.info(f"Pipeline generation successful: {repr(generated_text[:50])}...")
                generation_time = (datetime.now() - start_time).total_seconds()
                return {
                    "success": True,
                    "response": generated_text,
                    "error": None,
                    "metadata": {
                        "generation_time": generation_time,
                        "max_new_tokens": max_new_tokens,
                        "temperature": generation_params.get("temperature", 0.7),
                        "pipeline_generation": True
                    }
                }
                
        except Exception as pipeline_error:
            logger.warning(f"Pipeline generation failed: {pipeline_error}")

        # If pipeline failed or returned empty, try direct generation
        if model is not None:
            logger.info("Attempting direct model generation")
            return _generate_with_model_direct(model, tokenizer, conversation_context, generation_params, max_new_tokens, start_time)
        else:
            # No model available, return error
            generation_time = (datetime.now() - start_time).total_seconds()
            return {
                "success": False,
                "error": "Both pipeline and direct generation failed",
                "response": "",
                "metadata": {
                    "generation_time": generation_time,
                    "max_new_tokens": max_new_tokens
                }
            }

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Log traceback for detailed debugging
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "response": "",
            "metadata": {}
        }





def _generate_with_model_direct(model, tokenizer, conversation_context: str, generation_params: Dict[str, Any], max_new_tokens: int, start_time):
    """Direct generation using model.generate() with CUDA sequential offloading support."""
    try:
        logger.info("Using direct model generation with CUDA sequential offloading")
        
        # Tokenize with proper settings for sequential offloading
        inputs = tokenizer(
            conversation_context, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        logger.info(f"Input tokens shape: {inputs['input_ids'].shape}")
        
        # For models with sequential offloading, inputs should go to the first device
        if hasattr(model, 'hf_device_map'):
            # Find the first device in the device map for input placement
            first_device = None
            for param_name, device in model.hf_device_map.items():
                if 'embed' in param_name.lower() or 'input' in param_name.lower():
                    first_device = device
                    break
            if first_device is None:
                first_device = list(model.hf_device_map.values())[0]
            logger.info(f"Moving inputs to first model device: {first_device}")
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
        else:
            # Single device model
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Optimized parameters for CUDA with sequential offloading
        gen_kwargs = {
            "max_new_tokens": min(max_new_tokens, 100),
            "do_sample": False,  # Use greedy decoding to avoid sampling issues
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict_in_generate": True
        }
        
        # Ensure attention mask is present
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        with torch.no_grad():
            try:
                # Clear CUDA cache before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs
                )
                
                # Extract generated sequences
                if hasattr(outputs, 'sequences'):
                    generated_sequences = outputs.sequences
                else:
                    generated_sequences = outputs
                
                # Decode the response
                input_length = inputs["input_ids"].shape[1]
                if len(generated_sequences.shape) > 1 and generated_sequences.shape[1] > input_length:
                    new_tokens = generated_sequences[0][input_length:]
                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    
                    # Clean up common artifacts
                    if generated_text.startswith("<start_of_turn>model\n"):
                        generated_text = generated_text[19:].strip()
                    elif generated_text.startswith("model\n"):
                        generated_text = generated_text[6:].strip()
                    if generated_text.startswith("Assistant:"):
                        generated_text = generated_text[10:].strip()
                    if generated_text.endswith("<end_of_turn>"):
                        generated_text = generated_text[:-13].strip()
                        
                    logger.info(f"Direct generation result: {repr(generated_text[:100])}...")
                    
                    if generated_text:  # Only return if we got actual content
                        generation_time = (datetime.now() - start_time).total_seconds()
                        return {
                            "success": True,
                            "response": generated_text,
                            "error": None,
                            "metadata": {
                                "generation_time": generation_time,
                                "max_new_tokens": max_new_tokens,
                                "direct_generation": True,
                                "sequential_offloading": hasattr(model, 'hf_device_map')
                            }
                        }
                
            except Exception as gen_error:
                logger.warning(f"Generation failed: {gen_error}")
                # For CUDA errors, try with more conservative settings
                if "CUDA" in str(gen_error) or "out of memory" in str(gen_error).lower():
                    logger.info("Attempting generation with conservative settings")
                    try:
                        torch.cuda.empty_cache()
                        conservative_kwargs = {
                            "max_new_tokens": min(max_new_tokens, 10),
                            "do_sample": False,  # Greedy decoding
                            "pad_token_id": tokenizer.eos_token_id,
                            "eos_token_id": tokenizer.eos_token_id,
                            "use_cache": False,
                            "num_beams": 1,  # No beam search
                        }
                        
                        outputs = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            **conservative_kwargs
                        )
                        
                        if hasattr(outputs, 'sequences'):
                            generated_sequences = outputs.sequences
                        else:
                            generated_sequences = outputs
                        
                        input_length = inputs["input_ids"].shape[1]
                        if len(generated_sequences.shape) > 1 and generated_sequences.shape[1] > input_length:
                            new_tokens = generated_sequences[0][input_length:]
                            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                            
                            if generated_text:
                                generation_time = (datetime.now() - start_time).total_seconds()
                                return {
                                    "success": True,
                                    "response": generated_text,
                                    "error": None,
                                    "metadata": {
                                        "generation_time": generation_time,
                                        "max_new_tokens": max_new_tokens,
                                        "direct_generation": True,
                                        "conservative_mode": True
                                    }
                                }
                    except Exception as conservative_error:
                        logger.warning(f"Conservative generation also failed: {conservative_error}")

        # If we get here, generation failed
        generation_time = (datetime.now() - start_time).total_seconds()
        return {
            "success": False,
            "error": "Model generation failed after all attempts",
            "response": "",
            "metadata": {
                "generation_time": generation_time,
                "max_new_tokens": max_new_tokens,
                "direct_generation": True
            }
        }

    except Exception as e:
        logger.error(f"Error in direct generation: {e}")
        generation_time = (datetime.now() - start_time).total_seconds()
        return {
            "success": False,
            "error": str(e),
            "response": "",
            "metadata": {
                "generation_time": generation_time,
                "max_new_tokens": max_new_tokens,
                "direct_generation": True
            }
        }