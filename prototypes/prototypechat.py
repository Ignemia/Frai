import re
import chromadb
from chromadb.errors import NotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from sentence_transformers import SentenceTransformer
import uuid
import os
import logging
import time
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chat_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("personal-chatter")

logger.info("Starting application...")
logger.info("Python interpreter initialized")
logger.info(f"Current working directory: {os.getcwd()}")

# Log imports and dependencies
logger.info("Checking for required dependencies...")
logger.info(f"chromadb version: {chromadb.__version__ if hasattr(chromadb, '__version__') else 'unknown'}")
logger.info(f"sentence_transformers version: {SentenceTransformer.__version__ if hasattr(SentenceTransformer, '__version__') else 'unknown'}")

# Initialize model with your local Gemma file
model_path = "./models/gemma-3-4b-it/"
logger.info(f"Checking if model exists at {os.path.abspath(model_path)}")
if not os.path.exists(model_path):
    logger.error(f"Model directory not found: {os.path.abspath(model_path)}")
    raise FileNotFoundError(f"Model path {model_path} not found")

model_files = os.listdir(model_path)
logger.info(f"Found {len(model_files)} files in model directory: {', '.join(model_files[:5])}" + 
            ("..." if len(model_files) > 5 else ""))

logger.info(f"Starting to load Gemma model from {os.path.abspath(model_path)}")
logger.info("This may take several minutes depending on your hardware. Please wait...")
start_time = time.time()

try:
    # Log memory usage before loading model
    if hasattr(os, 'getpid'):
        process = psutil.Process(os.getpid())
        logger.info(f"Memory usage before model loading: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    
    # Load tokenizer first
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Tokenizer loaded successfully")
    
    # Load the model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    logger.info("Model loaded successfully")
    
    # Create generation pipeline without device parameter
    logger.info("Creating text generation pipeline...")
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    
    # Log memory after loading model
    if hasattr(os, 'getpid'):
        logger.info(f"Memory usage after model loading: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
    logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Add a proper embedding function class for ChromaDB
class SentenceTransformerEmbedding:
    def __init__(self, model_name):
        logger.info(f"Initializing embedding function with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def __call__(self, input):
        logger.debug(f"Generating embeddings for {len(input)} texts")
        return self.model.encode(input).tolist()

# Initialize sentence transformer for embeddings
logger.info("Loading sentence transformer model for embeddings...")
start_time = time.time()
embedding_function = SentenceTransformerEmbedding('all-MiniLM-L6-v2')
logger.info(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")

class VectorStore:
    def __init__(self, collection_name="documents", persist_dir="./.vectordb"):
        logger.info(f"Initializing VectorStore with collection '{collection_name}' at {persist_dir}")
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize ChromaDB client
        logger.info("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        try:
            logger.info(f"Attempting to load existing collection '{collection_name}'")
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Collection loaded with {self.collection.count()} documents")
        except NotFoundError as e:
            # Collection doesn't exist, create it
            logger.info(f"Collection not found (Error: {str(e)}), creating new collection '{collection_name}'")
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info("New collection created successfully")
    
    def _get_embeddings(self, texts):
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        # Use the embedding function
        return embedding_function(texts)
    
    def add_documents(self, documents):
        if not documents:
            logger.warning("No documents provided to add_documents")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Generate unique IDs for each document
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Add documents to collection
        start_time = time.time()
        self.collection.add(
            documents=documents,
            ids=ids
        )
        logger.info(f"Documents added in {time.time() - start_time:.2f} seconds")
        logger.debug(f"First few document IDs: {ids[:3]}")
    
    def search(self, query, k=5):
        logger.info(f"Searching for: '{query}' (top {k} results)")
        
        # Search the collection
        start_time = time.time()
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        search_time = time.time() - start_time
        
        # Format results
        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            formatted_results.append({
                'document': {
                    'id': results['ids'][0][i],
                    'content': doc
                },
                'distance': results['distances'][0][i] if 'distances' in results else 0.0
            })
        
        logger.info(f"Search completed in {search_time:.2f} seconds with {len(formatted_results)} results")
        logger.debug(f"Top result: {formatted_results[0] if formatted_results else 'None'}")
        
        return formatted_results

def generate_text(prompt, max_new_tokens=256):
    logger.info(f"Generating text with max_new_tokens={max_new_tokens}")
    logger.debug(f"Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    
    # Print the initial prompt
    print(prompt, end="", flush=True)
    
    # Generate text using the Hugging Face pipeline
    logger.info("Starting text generation...")
    generation_start = time.time()
    
    try:
        # More conservative parameters to avoid CUDA errors
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.eos_token_id,  # Ensure proper padding
            "eos_token_id": tokenizer.eos_token_id,  # Ensure proper ending
        }
        
        logger.info(f"Using generation config: {generation_config}")
        
        # Generate text with proper error handling
        outputs = model.generate(
            tokenizer.encode(prompt, return_tensors="pt").to(model.device),
            **generation_config
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text
        generated_text = generated_text[len(prompt):]
        
        # Print the generated text
        print(generated_text, flush=True)
        
    except RuntimeError as e:
        logger.error(f"Error during generation: {str(e)}")
        logger.info("Falling back to simpler generation method...")
        
        try:
            # Fallback to simpler generation method
            inputs = tokenizer(prompt, return_tensors="pt").to("cpu")  # Move to CPU
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Use greedy decoding for stability
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):]
            
            # Print the generated text
            print(generated_text, flush=True)
            
        except Exception as e2:
            logger.error(f"Fallback generation also failed: {str(e2)}")
            generated_text = "Error: Unable to generate text due to a model error."
            print(generated_text, flush=True)
    
    generation_time = time.time() - generation_start
    token_count = len(tokenizer.encode(generated_text))
    logger.info(f"Text generation completed: {token_count} tokens in {generation_time:.2f} seconds ({token_count/generation_time:.1f} tokens/sec)")
    
    return generated_text

if __name__ == "__main__":
    try:
        logger.info("Starting main application flow")
        
        # Initialize vector store
        logger.info("Starting vector store initialization...")
        vector_store = VectorStore()
        
        # Example: Add some documents to the vector store
        sample_documents = [
            "Gemma is an open-source lightweight AI model developed by Google.",
            "Vector databases are used for semantic search applications.",
            "Embeddings convert text into numerical vectors for similarity comparisons.",
            "Once upon a time in a land far, far away, there lived a brave knight.",
            "The quick brown fox jumps over the lazy dog."
        ]
        logger.info(f"Adding {len(sample_documents)} sample documents")
        vector_store.add_documents(sample_documents)
        
        # Example: Search for similar documents
        query = "Tell me about AI models"
        logger.info(f"Performing search with query: '{query}'")
        results = vector_store.search(query, k=2)
        print(f"\nSearching for: '{query}'")
        for i, result in enumerate(results):
            logger.debug(f"Result {i+1}: {result['document']['content'][:30]}... (Distance: {result['distance']:.4f})")
            print(f"{i+1}. {result['document']['content']} (Distance: {result['distance']:.4f})")
        
        # Example: Generate text with context from vector search
        logger.info("Generating text with context from vector search")
        context = f"Context: {results[0]['document']['content']}\n\n"
        prompt = context + "Tell me more about Gemma:"
        print("\nGenerating text with context:")
        generated_text = generate_text(prompt)
        
        logger.info("Application execution completed")
    except Exception as e:
        logger.exception(f"Error in main application flow: {str(e)}")
        raise