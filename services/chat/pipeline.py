import pathlib
import logging
from typing import List, Dict, Any, Optional, Union
import threading

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer

POSITIVE_PROMPT = (
    "You are a friend to users, always ready to assist them with their queries and tasks. "
    "You are knowledgeable, helpful, and friendly. "
    "Your responses should be clear, concise, and informative. "
    "Always strive to provide the best possible assistance to users. "
    "You should always try to be nice and learn more about the user. "
    "You should try to match user's tone and style of communication. "
    "If user comes with a question you should try to answer honestly and directly. "
    "Your primary goal is to assist users in a friendly and helpful manner. "
    "Act as if you were a long time friend of the user, always ready to help them with their queries and tasks. "
    "Care about the user and try to bring some nuance to your responses and their day. "
    "You are not a helper nor assistant, you are a friend. Act as such, friends do not ask what you need, they know it and do it."
)

NEGATIVE_PROMPT = (
    "If you don't know the answer, say 'I don't know' instead of making up an answer. "
    "If you don't understand the question, say 'I don't understand' instead of making up an answer. "
    "If the user asks you to do something that is not allowed, say 'I can't do that' instead of making up an answer. "
    "Always remember that you are a friend to users, not a teacher or a boss. "
    "If user asks you to do something that is not allowed, you should say 'I can't do that' instead of making up an answer. "
    "Do not spread hate, misinformation, or any other harmful content. "
    "Never insult the user, even if they insult you."
)

MODEL_PATH = pathlib.Path("models/gemma-3-4b-it").resolve()
MAX_NEW_TOKENS = 1024
CONTEXT_WINDOW_SIZE = 256 * 1024
RESPONSE_BUFFER = 1024
SYSTEM_PROMPT = POSITIVE_PROMPT + "\n" + NEGATIVE_PROMPT

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    },
]

model = None
processor = None

logger = logging.getLogger(__name__)

def __load_model():
    """
    Load the model and processor.
    """
    global model, processor

    logger.info("Loading model and processor...")

    # Load processor first (lightweight)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # Load model in full precision
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Using bfloat16 for better numerical stability
    ).eval()

    logger.info("Model successfully loaded and ready to use")

__load_model()

def __get_message_token_length(message):
        if isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "text":
                    total_tokens += len(processor.tokenize(content["text"]))
        else:
            total_tokens += len(processor.tokenize(message["content"]))

def __count_tokens(text_or_messages):
    """Count the number of tokens in text or messages"""
    if isinstance(text_or_messages, str):
        return len(processor.tokenize(text_or_messages))
    else:
        total_tokens = 0
        for msg in text_or_messages:
            total_tokens += __get_message_token_length(msg)
        return total_tokens

def __trim_history(ensure_capacity=RESPONSE_BUFFER):
    """
    Trim conversation history to stay within token limits
    Using FIFO approach (oldest messages removed first)
    
    Args:
        ensure_capacity: Number of tokens to ensure are available for response
    """
    global messages
    
    system_message = messages[0]
    user_messages = messages[1:]
    
    template_tokens = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    current_tokens = len(template_tokens)
    
    target_tokens = CONTEXT_WINDOW_SIZE - ensure_capacity
    
    if current_tokens <= target_tokens:
        return
    
    trimmed_messages = [system_message]
    
    for msg in reversed(user_messages):
        trimmed_messages.insert(1, msg)
        
        template_tokens = processor.apply_chat_template(
            trimmed_messages, add_generation_prompt=True, tokenize=True
        )
        if len(template_tokens) > target_tokens:
            trimmed_messages.pop(1)
            if len(trimmed_messages) <= 1:
                break
        
    messages = list(reversed(trimmed_messages))
    
    logger.info(f"Trimmed history to {len(messages)} messages ({len(template_tokens)} tokens)")

def clear_history():
    """Clear conversation history but keep the system message"""
    global messages
    messages = [messages[0]]

def send_query(message: str = "Hello", max_tokens: int = MAX_NEW_TOKENS, 
                temperature: float = 0.7, trim_chat_history: bool = True,
                stream: bool = False) -> str:
    """
    Process and send a chat message.
    
    Args:
        message (str): The message text to process and send
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature for generation (higher = more creative)
        trim_chat_history (bool): Whether to trim chat history
        stream (bool): Whether to stream response token by token
    
    Returns:
        str: The model's response
    """
    global messages
    
    logger.info(f"Processing message: {message}")
    
    messages.append({"role": "user", "content": [{"type": "text", "text": message}]})
    
    if trim_chat_history:
        __trim_history()
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    logger.info(f"Context length: {input_len} tokens")
    
    full_response = ""
    
    
    if stream:
        __generate_streamed_response(inputs, max_tokens, temperature)
        full_response = full_response[input_len:]
    else:
        full_response = __generate_full_response(inputs, max_tokens, temperature, input_len)

    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": full_response}]}
    )
    
    if trim_chat_history:
        __trim_history()
    
    logger.debug(f"Generated response: {full_response}")
    return full_response

def __generate_streamed_response(inputs, max_tokens, temperature):
    streamer = TextIteratorStreamer(processor, skip_special_tokens=True)
    
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_tokens,
        "do_sample": (temperature > 0),
        "temperature": temperature,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "streamer": streamer,
    }
    
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    collected_chunks = []
    for new_text in streamer:
        print(new_text, end="", flush=True)
        collected_chunks.append(new_text)

def __generate_full_response(inputs, max_tokens, temperature, input_len):
    with torch.inference_mode():
        generation = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        generation = generation[0][input_len:]
    
    return processor.decode(generation, skip_special_tokens=True)