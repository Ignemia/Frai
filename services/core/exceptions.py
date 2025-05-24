"""
Custom exceptions for the core services module.

This module defines custom exception classes used throughout
the core services for better error handling and reporting.
"""


class ServiceError(Exception):
    """Base exception for all service-related errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "GENERIC_ERROR"
        self.details = details or {}


class ValidationError(ServiceError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None, value=None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field
        self.value = value
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = str(value)


class AuthenticationError(ServiceError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTHENTICATION_ERROR")


class AuthorizationError(ServiceError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Authorization failed", required_permission: str = None):
        super().__init__(message, "AUTHORIZATION_ERROR")
        if required_permission:
            self.details["required_permission"] = required_permission


class ConfigurationError(ServiceError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_section: str = None):
        super().__init__(message, "CONFIGURATION_ERROR")
        if config_section:
            self.details["config_section"] = config_section


class ResourceNotFoundError(ServiceError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None):
        super().__init__(message, "RESOURCE_NOT_FOUND")
        if resource_type:
            self.details["resource_type"] = resource_type
        if resource_id:
            self.details["resource_id"] = resource_id


class ModelError(ServiceError):
    """Raised when there's an error with AI model operations."""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(message, "MODEL_ERROR")
        if model_name:
            self.details["model_name"] = model_name


class MemoryError(ServiceError):
    """Raised when there are memory-related errors."""
    
    def __init__(self, message: str, memory_type: str = None):
        super().__init__(message, "MEMORY_ERROR")
        if memory_type:
            self.details["memory_type"] = memory_type


class TimeoutError(ServiceError):
    """Raised when an operation times out."""
    
    def __init__(self, message: str, timeout_seconds: float = None):
        super().__init__(message, "TIMEOUT_ERROR")
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds


class ExternalServiceError(ServiceError):
    """Raised when an external service fails."""
    
    def __init__(self, message: str, service_name: str = None, status_code: int = None):
        super().__init__(message, "EXTERNAL_SERVICE_ERROR")
        if service_name:
            self.details["service_name"] = service_name
        if status_code:
            self.details["status_code"] = status_code


# Export all exceptions
__all__ = [
    'ServiceError',
    'ValidationError', 
    'AuthenticationError',
    'AuthorizationError',
    'ConfigurationError',
    'ResourceNotFoundError',
    'ModelError',
    'MemoryError',
    'TimeoutError',
    'ExternalServiceError'
]
