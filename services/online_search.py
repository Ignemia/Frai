"""
from test_mock_helper import List
Online search service for Personal Chatter.

This module handles searching for information online using the Brave Search API
or other search providers.
"""
import logging
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import requests
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SEARCH_COUNT = 5  # Number of results to return by default
MAX_SEARCH_COUNT = 10     # Maximum number of results to return
SEARCH_CACHE_DIR = "./outputs/search_cache"  # Directory to cache search results

def _ensure_cache_dir():
    """Ensure the search cache directory exists."""
    if not os.path.exists(SEARCH_CACHE_DIR):
        os.makedirs(SEARCH_CACHE_DIR)
        logger.info(f"Created search cache directory at {SEARCH_CACHE_DIR}")

def _get_brave_api_key() -> Optional[str]:
    """
    Get the Brave Search API key from environment variables.
    
    Returns:
        The API key if available, None otherwise
    """
    from services.config import get_config
    
    # Get API key environment variable name from config
    config = get_config()
    env_var = config.get("api", {}).get("brave_search_api_key_env", "BRAVE_SEARCH_API_KEY")
    
    api_key = os.environ.get(env_var)
    if not api_key:
        logger.warning(f"Brave Search API key not found in environment variable {env_var}")
    return api_key

def _get_cache_filename(query: str) -> str:
    """
    Generate a cache filename for a search query.
    
    Args:
        query: The search query
        
    Returns:
        The cache filename
    """
    # Create a safe filename from the query
    safe_query = "".join(c if c.isalnum() else "_" for c in query)
    safe_query = safe_query[:50]  # Limit length
    timestamp = int(time.time())
    return f"{safe_query}_{timestamp}.json"

def _get_cached_results(query: str) -> Optional[Dict[str, Any]]:
    """
    Try to get cached search results for a query.
    
    Args:
        query: The search query
        
    Returns:
        Cached results if available and recent, None otherwise
    """
    _ensure_cache_dir()
    
    # Look for recent cache files matching the query
    safe_query_prefix = "".join(c if c.isalnum() else "_" for c in query)
    safe_query_prefix = safe_query_prefix[:50]
    
    try:
        # Find the most recent cache file for this query
        matching_files = []
        for filename in os.listdir(SEARCH_CACHE_DIR):
            if filename.startswith(safe_query_prefix) and filename.endswith(".json"):
                filepath = os.path.join(SEARCH_CACHE_DIR, filename)
                matching_files.append((os.path.getmtime(filepath), filepath))
        
        # If we found any files, use the most recent one if it's less than 1 day old
        if matching_files:
            # Sort by modification time (newest first)
            matching_files.sort(reverse=True)
            mtime, filepath = matching_files[0]
            
            # Check if the file is less than 24 hours old
            if time.time() - mtime < 86400:  # 24 hours in seconds
                with open(filepath, "r", encoding="utf-8") as f:
                    logger.info(f"Using cached search results from {filepath}")
                    return json.load(f)
    
    except Exception as e:
        logger.error(f"Error reading cache: {e}")
    
    return None

def _cache_results(query: str, results: Dict[str, Any]):
    """
    Cache search results for a query.
    
    Args:
        query: The search query
        results: The search results to cache
    """
    _ensure_cache_dir()
    
    try:
        # Generate a cache filename
        filename = _get_cache_filename(query)
        filepath = os.path.join(SEARCH_CACHE_DIR, filename)
        
        # Write the results to the cache file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Cached search results to {filepath}")
    
    except Exception as e:
        logger.error(f"Error caching results: {e}")

def search_brave(query: str, count: int = DEFAULT_SEARCH_COUNT) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Search the web using the Brave Search API.
    
    Args:
        query: The search query
        count: The number of results to return (max 10)
        
    Returns:
        Tuple containing:
            - Success status (boolean)
            - Message describing the result
            - List of search result dictionaries
    """
    logger.info(f"Searching with Brave API for: {query}")
    
    # Limit the count
    count = min(count, MAX_SEARCH_COUNT)
    
    # Check for cached results first
    cached_results = _get_cached_results(query)
    if cached_results:
        try:
            results = cached_results.get("web", {}).get("results", [])
            if results:
                # Return the requested number of results
                limited_results = results[:count]
                return True, f"Retrieved {len(limited_results)} results from cache", limited_results
        except Exception as e:
            logger.error(f"Error processing cached results: {e}")
    
    # Get the API key
    api_key = _get_brave_api_key()
    if not api_key:
        return False, "Brave Search API key not available", []
    
    # Prepare the API request
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "count": count,
        "search_lang": "en",
        "safesearch": "moderate"
    }
    
    try:
        # Make the API request
        response = requests.get(url, headers=headers, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Cache the results
            _cache_results(query, data)
            
            # Extract and return the search results
            results = data.get("web", {}).get("results", [])
            return True, f"Retrieved {len(results)} results", results
        else:
            logger.error(f"Brave Search API error: {response.status_code} - {response.text}")
            return False, f"Search API error: {response.status_code}", []
    
    except Exception as e:
        logger.error(f"Error searching with Brave API: {e}")
        return False, f"Search error: {str(e)}", []

def search_online(query: str, count: int = DEFAULT_SEARCH_COUNT) -> List[Dict[str, Any]]:
    """
    Search online using available search providers.
    
    This function will try different search providers in order of preference.
    
    Args:
        query: The search query
        count: The number of results to return
        
    Returns:
        List of search result dictionaries
    """
    # Try Brave Search first
    success, message, results = search_brave(query, count)
    
    # TODO: Add fallback search providers if needed
    
    # Log the results
    logger.info(f"Search for '{query}': {message}")
    
    return results

def format_search_results_for_llm(results: List[Dict[str, Any]]) -> str:
    """
    Format search results for use in LLM prompt.
    
    Args:
        results: List of search result dictionaries
        
    Returns:
        A formatted string with the search results suitable for LLM context
    """
    if not results:
        return "No search results found."
    
    summary = "SEARCH RESULTS:\n\n"
    
    for i, result in enumerate(results):
        title = result.get("title", "No title")
        description = result.get("description", "No description")
        url = result.get("url", "No URL")
        
        summary += f"RESULT {i+1}:\n"
        summary += f"TITLE: {title}\n"
        summary += f"DESCRIPTION: {description}\n"
        summary += f"URL: {url}\n\n"
    
    return summary
