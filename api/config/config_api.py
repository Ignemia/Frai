"""
Configuration API for Personal Chatter.

This module provides endpoints for viewing and modifying the application configuration.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any

from services.config import get_config, update_config, save_config
from api.auth import get_current_user

logger = logging.getLogger(__name__)

config_router = APIRouter(prefix="/config", tags=["config"])

class CommandProcessingConfig(BaseModel):
    """Request/response model for command processing configuration."""
    enabled: bool
    use_main_model: bool
    store_memory_enabled: bool
    store_user_info_enabled: bool
    image_generation_enabled: bool
    online_search_enabled: bool
    local_search_enabled: bool

class ConfigStatus(BaseModel):
    """Response model for configuration status."""
    success: bool
    message: str

@config_router.get(
    "/command-processing",
    response_model=CommandProcessingConfig,
    summary="Get command processing configuration",
    status_code=200,
)
async def get_command_processing_config(current_user = Depends(get_current_user)):
    """
    Get the current command processing configuration.
    
    Returns:
        CommandProcessingConfig: The current command processing configuration
    """
    config = get_config()
    cmd_config = config.get("command_preprocessing", {})
    
    return CommandProcessingConfig(
        enabled=cmd_config.get("enabled", True),
        use_main_model=cmd_config.get("use_main_model", True),
        store_memory_enabled=cmd_config.get("store_memory_enabled", True),
        store_user_info_enabled=cmd_config.get("store_user_info_enabled", True),
        image_generation_enabled=cmd_config.get("image_generation_enabled", True),
        online_search_enabled=cmd_config.get("online_search_enabled", True),
        local_search_enabled=cmd_config.get("local_search_enabled", True)
    )

@config_router.post(
    "/command-processing",
    response_model=ConfigStatus,
    summary="Update command processing configuration",
    status_code=200,
)
async def update_command_processing_config(
    config_update: CommandProcessingConfig,
    current_user = Depends(get_current_user)
):
    """
    Update the command processing configuration.
    
    Args:
        config_update: New configuration values
        
    Returns:
        ConfigStatus: Status of the update operation
    """
    try:
        # Update configuration
        update_config({
            "command_preprocessing": {
                "enabled": config_update.enabled,
                "use_main_model": config_update.use_main_model,
                "store_memory_enabled": config_update.store_memory_enabled,
                "store_user_info_enabled": config_update.store_user_info_enabled,
                "image_generation_enabled": config_update.image_generation_enabled,
                "online_search_enabled": config_update.online_search_enabled,
                "local_search_enabled": config_update.local_search_enabled
            }
        })
        
        # Save to disk
        if save_config():
            logger.info("Command processing configuration updated and saved")
            return ConfigStatus(
                success=True,
                message="Configuration updated successfully"
            )
        else:
            logger.error("Failed to save configuration")
            return ConfigStatus(
                success=False,
                message="Configuration updated in memory but could not be saved to disk"
            )
    
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration: {str(e)}"
        )
