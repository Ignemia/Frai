"""
Generates chat titles using a language model.

This module provides functionality to generate a concise and relevant title
for a given chat history using a pre-trained language model. It leverages
Langchain for interacting with the model.
"""

import logging
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Use relative imports for modules within the same package
from .model_loader import model as llm_model, processor as llm_processor, load_model as load_llm_model_from_pipeline

logger = logging.getLogger(__name__)

# Global variable for the Langchain LLM pipeline
llm_pipeline_for_title = None

def init_title_generation_pipeline():
    """
    Initializes the Langchain pipeline for title generation.

    This function loads the model and processor if they are not already loaded,
    then creates and stores a HuggingFacePipeline instance for title generation.
    This should be called once before `generate_chat_title` is used, or
    `generate_chat_title` will attempt to initialize it.
    """
    global llm_pipeline_for_title, llm_model, llm_processor
    logger.debug("Initializing title generation pipeline.")

    if llm_model is None or llm_processor is None:
        logger.info("Model or processor not loaded for title generation. Attempting to load from model_loader module.")
        try:
            load_llm_model_from_pipeline() # This function should load llm_model and llm_processor globally
            if llm_model is None or llm_processor is None: # Check again
                logger.error("Failed to load model and processor via load_llm_model_from_pipeline.")
                raise ValueError("Model and processor must be loaded before initializing title generation pipeline.")
            logger.info("Model and processor loaded successfully via model_loader.")
        except Exception as e:
            logger.error(f"Error loading model/processor for title generation: {e}", exc_info=True)
            raise

    if llm_pipeline_for_title is None:
        logger.info("Creating new HuggingFacePipeline for title generation.")
        try:
            llm_pipeline_for_title = HuggingFacePipeline.from_model_id(
                model_id=llm_model.config._name_or_path, # Use the path from the loaded model
                task="text-generation", 
                model_kwargs={"torch_dtype": llm_model.dtype, "low_cpu_mem_usage": True},
                pipeline_kwargs={
                    "model": llm_model, 
                    "tokenizer": llm_processor.tokenizer, 
                    "max_new_tokens": 20, 
                    "pad_token_id": llm_processor.tokenizer.eos_token_id # Ensure pad token is set
                }
            )
            logger.info("HuggingFacePipeline for title generation created successfully.")
        except Exception as e:
            logger.error(f"Failed to create HuggingFacePipeline for title generation: {e}", exc_info=True)
            llm_pipeline_for_title = None # Ensure it's None if creation failed
            raise
    else:
        logger.debug("Title generation pipeline already initialized.")


def generate_chat_title(chat_history: list[dict[str, str]]) -> str:
    """
    Generates a concise title for a given chat history.

    Args:
        chat_history (list[dict[str, str]]): A list of chat messages,
            where each message is a dictionary with 'role' and 'content' keys.

    Returns:
        str: A generated title for the chat, or a default title if generation fails.
             The title is stripped of leading/trailing whitespace and quotes.
    
    Raises:
        ValueError: If the model/processor cannot be loaded or the pipeline cannot be initialized.
        Exception: For other errors during title generation.
    """
    global llm_pipeline_for_title
    logger.info("Attempting to generate chat title.")
    logger.debug(f"Received chat history for title generation (length {len(chat_history)}): {chat_history}")

    if not chat_history:
        logger.warning("Chat history is empty. Returning default title.")
        return "New Chat"

    if llm_pipeline_for_title is None:
        logger.warning("Title generation pipeline not initialized. Attempting to initialize now.")
        try:
            init_title_generation_pipeline()
            if llm_pipeline_for_title is None: # Check if initialization succeeded
                 logger.error("Failed to initialize title generation pipeline on demand.")
                 raise ValueError("Title generation pipeline could not be initialized.")
        except Exception as e:
            logger.error(f"Error initializing title generation pipeline on demand: {e}", exc_info=True)
            return "Chat Title Error"

    # Combine chat history into a single string for the prompt
    # Taking last N messages to avoid overly long prompts for title generation
    # and to focus on the most recent context.
    # Let's consider the last 3 exchanges (user + assistant = 1 exchange)
    max_messages_for_title_prompt = 6 # e.g., 3 user messages and 3 assistant responses
    relevant_history = chat_history[-max_messages_for_title_prompt:]
    
    # Format the relevant history for the prompt
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in relevant_history])
    logger.debug(f"Formatted history for title prompt: {formatted_history}")

    # Define a prompt template for title generation
    # Updated prompt to be more direct and ask for a very short title.
    prompt_template_str = (
        "Based on the following conversation, generate a very concise and descriptive title "
        "(3-7 words maximum). The title should capture the main topic or question. "
        "Do not include prefixes like 'Title:'. Just provide the title text.\n\n"
        "Conversation:\n{chat_summary}\n\nConcise Title:"
    )
    
    prompt = PromptTemplate(
        input_variables=["chat_summary"],
        template=prompt_template_str,
    )
    
    full_prompt_text = prompt.format(chat_summary=formatted_history)
        
    try:
        logger.debug(f"Invoking LLM for title generation with prompt: {full_prompt_text}")
        raw_title = llm_pipeline_for_title(full_prompt_text)
        logger.info(f"Raw title generated by LLM: '{raw_title}'")

        # Clean the generated title
        # Titles sometimes come with extra quotes or newlines.
        # Also, the model might repeat the prompt or add conversational fluff.
        # We need to be robust in cleaning.
        
        # 1. Strip whitespace
        title = raw_title.strip()
        
        # 2. Remove common model-added prefixes if any (like "Title: ", "Here's a title: ")
        prefixes_to_remove = ["Title: ", "Here's a title: ", "Concise Title: "]
        for prefix in prefixes_to_remove:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        
        # 3. Strip surrounding quotes (single or double)
        # Handles cases like '"Title"' or "'Title'"
        if len(title) > 1 and ((title.startswith('\'') and title.endswith('\'')) or \
                               (title.startswith('"') and title.endswith('"'))):
            title = title[1:-1]
            
        # 4. Truncate if it's too long despite the prompt (e.g., if model ignores length constraint)
        # This is a fallback, the prompt should ideally handle this.
        # A simple word count based truncation:
        max_words_in_title = 7
        words = title.split()
        if len(words) > max_words_in_title:
            title = " ".join(words[:max_words_in_title]) + "..."
            logger.debug(f"Title truncated to: '{title}'")

        # 5. Handle empty or very short (e.g., just punctuation) titles after cleaning
        if not title or len(title.replace("...", "").strip()) < 3 : # Check if meaningful content exists
            logger.warning(f"Generated title is empty or too short after cleaning ('{raw_title}' -> '{title}'). Using default.")
            return "Chat Summary"

        logger.info(f"Cleaned generated title: '{title}'")
        return title

    except Exception as e:
        logger.error(f"Error during title generation LLM call: {e}", exc_info=True)
        return "Chat Conversation" # Fallback title
