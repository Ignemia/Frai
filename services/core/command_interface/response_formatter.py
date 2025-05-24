"""
Response Formatter - Formats command results for different interface types.

This module handles formatting CommandResult objects into appropriate
formats for different interfaces (CLI, API, Web UI, etc.).
"""

import json
import logging
from typing import Any, Dict, Union
from .command_system import CommandResult, ResultType

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Formats command results for different interface types."""
    
    def __init__(self):
        """Initialize the response formatter."""
        self._formatters = {
            "json": self._format_json,
            "api": self._format_api,
            "cli": self._format_cli,
            "dict": self._format_dict,
            "raw": self._format_raw
        }
    
    def format_response(self, result: CommandResult, format_type: str = "json") -> Any:
        """
        Format a command result for the specified interface type.
        
        Args:
            result: CommandResult to format
            format_type: Target format ('json', 'cli', 'api', etc.)
            
        Returns:
            Formatted response appropriate for the interface
        """
        formatter = self._formatters.get(format_type.lower())
        if not formatter:
            logger.warning(f"Unknown format type: {format_type}, defaulting to json")
            formatter = self._formatters["json"]
        
        try:
            return formatter(result)
        except Exception as e:
            logger.error(f"Error formatting response: {e}", exc_info=True)
            return self._format_error_fallback(result, str(e))
    
    def _format_json(self, result: CommandResult) -> str:
        """Format result as JSON string."""
        try:
            return json.dumps(result.to_dict(), indent=2, default=str)
        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            return json.dumps({
                "error": "JSON serialization failed",
                "command_id": result.command_id,
                "success": False
            })
    
    def _format_api(self, result: CommandResult) -> Dict[str, Any]:
        """Format result for API responses."""
        api_response = {
            "success": result.success,
            "data": result.data,
            "message": result.message,
            "command_id": result.command_id,
            "timestamp": result.timestamp.isoformat()
        }
        
        if not result.success:
            api_response["error"] = {
                "code": result.error_code,
                "details": result.error_details
            }
        
        if result.execution_time_ms is not None:
            api_response["execution_time_ms"] = result.execution_time_ms
        
        if result.progress is not None:
            api_response["progress"] = result.progress
        
        if result.async_token:
            api_response["async_token"] = result.async_token
        
        return api_response
    
    def _format_cli(self, result: CommandResult) -> str:
        """Format result for CLI display."""
        if result.success:
            output = []
            
            # Success message
            if result.message:
                output.append(f"✅ {result.message}")
            
            # Data display
            if result.data is not None:
                if isinstance(result.data, str):
                    output.append(result.data)
                elif isinstance(result.data, dict):
                    output.append(self._format_dict_for_cli(result.data))
                elif isinstance(result.data, list):
                    output.append(self._format_list_for_cli(result.data))
                else:
                    output.append(str(result.data))
            
            # Execution time
            if result.execution_time_ms is not None:
                output.append(f"⏱️  Completed in {result.execution_time_ms}ms")
            
            return "\n".join(output) if output else "✅ Command completed successfully"
        
        else:
            # Error formatting
            error_output = [f"❌ Error: {result.message}"]
            
            if result.error_code:
                error_output.append(f"Error Code: {result.error_code}")
            
            if result.error_details:
                error_output.append(f"Details: {result.error_details}")
            
            return "\n".join(error_output)
    
    def _format_dict(self, result: CommandResult) -> Dict[str, Any]:
        """Format result as dictionary."""
        return result.to_dict()
    
    def _format_raw(self, result: CommandResult) -> CommandResult:
        """Return raw CommandResult object."""
        return result
    
    def _format_dict_for_cli(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Format dictionary for CLI display."""
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_dict_for_cli(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_list_for_cli(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return "\n".join(lines)
    
    def _format_list_for_cli(self, data: list, indent: int = 0) -> str:
        """Format list for CLI display."""
        lines = []
        prefix = "  " * indent
        
        for i, item in enumerate(data):
            if isinstance(item, dict):
                lines.append(f"{prefix}[{i}]:")
                lines.append(self._format_dict_for_cli(item, indent + 1))
            elif isinstance(item, list):
                lines.append(f"{prefix}[{i}]:")
                lines.append(self._format_list_for_cli(item, indent + 1))
            else:
                lines.append(f"{prefix}• {item}")
        
        return "\n".join(lines)
    
    def _format_error_fallback(self, result: CommandResult, error_msg: str) -> Dict[str, Any]:
        """Fallback error format when primary formatting fails."""
        return {
            "error": "Response formatting failed",
            "formatting_error": error_msg,
            "command_id": result.command_id,
            "success": result.success,
            "raw_message": result.message
        }
    
    def register_custom_formatter(self, format_type: str, formatter_func: callable):
        """
        Register a custom formatter for a specific format type.
        
        Args:
            format_type: Name of the format type
            formatter_func: Function that takes CommandResult and returns formatted output
        """
        self._formatters[format_type.lower()] = formatter_func
        logger.info(f"Registered custom formatter for format type: {format_type}")
    
    def get_available_formats(self) -> list:
        """Get list of available format types."""
        return list(self._formatters.keys())
