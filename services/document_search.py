"""
from typing import List
Local document search service for Frai.

This module handles searching through local documents stored by the user.
It uses a simple vector database to enable semantic search functionality.
"""
import logging
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import shutil

logger = logging.getLogger(__name__)

# Default configuration
DOCUMENTS_DIR = "./outputs/documents"
INDEX_DIR = "./outputs/document_index"
CHUNK_SIZE = 1000  # Characters per chunk when indexing

def _ensure_dirs():
    """Ensure the required directories exist."""
    for directory in [DOCUMENTS_DIR, INDEX_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def index_document(user_id: str, document_path: str, document_name: Optional[str] = None) -> bool:
    """
    Index a document for future searching.
    
    Args:
        user_id: User identifier
        document_path: Path to the document file
        document_name: Optional name for the document (defaults to filename)
        
    Returns:
        True if indexing was successful, False otherwise
    """
    _ensure_dirs()
    
    if not os.path.exists(document_path):
        logger.error(f"Document not found: {document_path}")
        return False
    
    try:
        # Generate a unique ID for the document
        doc_id = f"{int(time.time())}_{os.path.basename(document_path)}"
        
        # If no name was provided, use the filename
        if not document_name:
            document_name = os.path.basename(document_path)
        
        # Copy the document to our storage
        dest_path = os.path.join(DOCUMENTS_DIR, doc_id)
        shutil.copy2(document_path, dest_path)
        
        # Create the document metadata
        metadata = {
            "document_id": doc_id,
            "document_name": document_name,
            "user_id": user_id,
            "original_path": document_path,
            "indexed_at": time.time(),
            "chunks": []
        }
        
        # Read and chunk the document
        with open(document_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        # Simple chunking by character count
        # In a real implementation, you'd want to chunk more intelligently
        chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
        
        # Generate embedding for each chunk
        # In this simplified version, we'll just store the text chunks
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            metadata["chunks"].append({
                "chunk_id": chunk_id,
                "text": chunk,
                "position": i
            })
        
        # Save the metadata and index
        index_path = os.path.join(INDEX_DIR, f"{doc_id}.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully indexed document: {document_name}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to index document: {e}")
        return False

def search_documents(user_id: str, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search through indexed documents using the query.
    
    Args:
        user_id: User identifier
        query: Search query
        max_results: Maximum number of results to return
    
    Returns:
        List of search results with document info and matching text
    """
    _ensure_dirs()
    
    results = []
    
    try:
        # Get all index files for the user
        # In a real implementation, you'd use a proper vector database
        for filename in os.listdir(INDEX_DIR):
            if filename.endswith(".json"):
                index_path = os.path.join(INDEX_DIR, filename)
                
                with open(index_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                # Skip if not owned by this user
                if metadata.get("user_id") != user_id:
                    continue
                
                # Simple text search through chunks
                # In a real implementation, you'd use vector similarity search
                for chunk in metadata.get("chunks", []):
                    if query.lower() in chunk.get("text", "").lower():
                        # Found a match
                        results.append({
                            "document_name": metadata.get("document_name", "Unknown"),
                            "document_id": metadata.get("document_id", ""),
                            "chunk_id": chunk.get("chunk_id", ""),
                            "text": chunk.get("text", ""),
                            "score": 1.0  # In a real implementation, this would be a similarity score
                        })
        
        # Limit to max_results
        results = results[:max_results]
        
        logger.info(f"Found {len(results)} document matches for query: {query}")
        return results
    
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []

def get_document_by_id(document_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a document by its ID.
    
    Args:
        document_id: Document identifier
        user_id: User identifier for access control
        
    Returns:
        Document metadata and content if found and authorized, None otherwise
    """
    _ensure_dirs()
    
    try:
        # Check if the document index exists
        index_path = os.path.join(INDEX_DIR, f"{document_id}.json")
        if not os.path.exists(index_path):
            logger.warning(f"Document index not found: {document_id}")
            return None
        
        # Load the metadata
        with open(index_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Check if the user has access
        if metadata.get("user_id") != user_id:
            logger.warning(f"User {user_id} not authorized to access document {document_id}")
            return None
        
        # Load the document content
        doc_path = os.path.join(DOCUMENTS_DIR, document_id)
        content = ""
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        
        # Return the document with content
        return {
            "metadata": metadata,
            "content": content
        }
    
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        return None

def format_search_results_for_llm(results: List[Dict[str, Any]]) -> str:
    """
    Format search results for use in LLM context.
    
    Args:
        results: List of search results
        
    Returns:
        Formatted string containing search results
    """
    if not results:
        return "No documents found matching the query."
    
    formatted = "Here are the most relevant document excerpts:\n\n"
    
    for i, result in enumerate(results, 1):
        document_name = result.get("document_name", "Unknown document")
        text = result.get("text", "").strip()
        
        # Truncate very long texts
        if len(text) > 500:
            text = text[:497] + "..."
        
        formatted += f"DOCUMENT {i}: {document_name}\n"
        formatted += f"EXCERPT: {text}\n\n"
    
    return formatted
