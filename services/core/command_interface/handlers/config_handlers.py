"""
Configuration handlers for the Service Command Interface.

Handles all configuration management operations including getting,
setting, updating, and reloading configuration settings.
"""
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..command_system import Command, CommandResult, ExecutionContext
from services.config import get_config, update_config, save_config, load_config

logger = logging.getLogger(__name__)


class ConfigHandlers:
    """Handlers for configuration management operations."""
    
    @staticmethod
    def handle_get_config(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Get current configuration or specific section.
        
        Args:
            command: Command with optional 'section' parameter
            context: Execution context
            
        Returns:
            CommandResult with configuration data
        """
        try:
            section = command.parameters.get('section')
            config = get_config()
            
            if section:
                if section in config:
                    result_data = {section: config[section]}
                    message = f"Retrieved configuration section: {section}"
                else:
                    return CommandResult(
                        success=False,
                        message=f"Configuration section '{section}' not found",
                        error_code="CONFIG_SECTION_NOT_FOUND"
                    )
            else:
                result_data = config
                message = "Retrieved full configuration"
            
            return CommandResult(
                success=True,
                message=message,
                data=result_data
            )
            
        except Exception as e:
            logger.error(f"Error getting configuration: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to get configuration: {str(e)}",
                error_code="CONFIG_GET_FAILED"
            )
    
    @staticmethod
    def handle_update_config(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Update configuration with new values.
        
        Args:
            command: Command with 'updates' parameter containing new values
            context: Execution context
            
        Returns:
            CommandResult indicating success/failure
        """
        try:
            updates = command.parameters.get('updates')
            if not updates or not isinstance(updates, dict):
                return CommandResult(
                    success=False,
                    message="Missing or invalid 'updates' parameter",
                    error_code="INVALID_PARAMETERS"
                )
            
            # Validate updates don't contain sensitive data
            if not context.user_id:
                return CommandResult(
                    success=False,
                    message="Authentication required for configuration updates",
                    error_code="AUTHENTICATION_REQUIRED"
                )
            
            # Apply updates
            old_config = get_config().copy()
            updated_config = update_config(updates)
            
            # Determine what changed
            changes = []
            for section, values in updates.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        old_value = old_config.get(section, {}).get(key)
                        if old_value != value:
                            changes.append(f"{section}.{key}: {old_value} -> {value}")
                else:
                    old_value = old_config.get(section)
                    if old_value != values:
                        changes.append(f"{section}: {old_value} -> {values}")
            
            return CommandResult(
                success=True,
                message=f"Configuration updated successfully. Changes: {', '.join(changes)}",
                data={
                    "changes": changes,
                    "updated_config": updated_config
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to update configuration: {str(e)}",
                error_code="CONFIG_UPDATE_FAILED"
            )
    
    @staticmethod
    def handle_save_config(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Save current configuration to disk.
        
        Args:
            command: Command with optional 'path' parameter
            context: Execution context
            
        Returns:
            CommandResult indicating success/failure
        """
        try:
            config_path = command.parameters.get('path')
            
            # Require authentication for saving configuration
            if not context.user_id:
                return CommandResult(
                    success=False,
                    message="Authentication required for saving configuration",
                    error_code="AUTHENTICATION_REQUIRED"
                )
            
            success = save_config(config_path)
            
            if success:
                path_msg = f" to {config_path}" if config_path else " to default location"
                return CommandResult(
                    success=True,
                    message=f"Configuration saved successfully{path_msg}",
                    data={"config_path": config_path}
                )
            else:
                return CommandResult(
                    success=False,
                    message="Failed to save configuration",
                    error_code="CONFIG_SAVE_FAILED"
                )
                
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to save configuration: {str(e)}",
                error_code="CONFIG_SAVE_FAILED"
            )
    
    @staticmethod
    def handle_reload_config(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Reload configuration from disk.
        
        Args:
            command: Command with optional 'path' parameter
            context: Execution context
            
        Returns:
            CommandResult with reloaded configuration
        """
        try:
            config_path = command.parameters.get('path')
            
            # Require authentication for reloading configuration
            if not context.user_id:
                return CommandResult(
                    success=False,
                    message="Authentication required for reloading configuration",
                    error_code="AUTHENTICATION_REQUIRED"
                )
            
            old_config = get_config().copy()
            new_config = load_config(config_path)
            
            # Determine what changed
            changes = []
            for section in set(old_config.keys()) | set(new_config.keys()):
                if section not in old_config:
                    changes.append(f"Added section: {section}")
                elif section not in new_config:
                    changes.append(f"Removed section: {section}")
                elif old_config[section] != new_config[section]:
                    changes.append(f"Modified section: {section}")
            
            path_msg = f" from {config_path}" if config_path else " from default location"
            return CommandResult(
                success=True,
                message=f"Configuration reloaded successfully{path_msg}",
                data={
                    "config": new_config,
                    "changes": changes,
                    "config_path": config_path
                }
            )
            
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to reload configuration: {str(e)}",
                error_code="CONFIG_RELOAD_FAILED"
            )
    
    @staticmethod
    def handle_reset_config(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Reset configuration to defaults.
        
        Args:
            command: Command with optional 'section' parameter
            context: Execution context
            
        Returns:
            CommandResult indicating success/failure
        """
        try:
            section = command.parameters.get('section')
            
            # Require authentication for resetting configuration
            if not context.user_id:
                return CommandResult(
                    success=False,
                    message="Authentication required for resetting configuration",
                    error_code="AUTHENTICATION_REQUIRED"
                )
            
            # Import default config
            from services.config import DEFAULT_CONFIG
            
            if section:
                if section in DEFAULT_CONFIG:
                    # Reset specific section
                    update_config({section: DEFAULT_CONFIG[section]})
                    message = f"Configuration section '{section}' reset to defaults"
                    data = {section: DEFAULT_CONFIG[section]}
                else:
                    return CommandResult(
                        success=False,
                        message=f"Configuration section '{section}' not found in defaults",
                        error_code="CONFIG_SECTION_NOT_FOUND"
                    )
            else:
                # Reset entire configuration
                update_config(DEFAULT_CONFIG)
                message = "Configuration reset to defaults"
                data = DEFAULT_CONFIG
            
            return CommandResult(
                success=True,
                message=message,
                data={"reset_config": data}
            )
            
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to reset configuration: {str(e)}",
                error_code="CONFIG_RESET_FAILED"
            )
    
    @staticmethod
    def handle_validate_config(command: Command, context: ExecutionContext) -> CommandResult:
        """
        Validate current configuration.
        
        Args:
            command: Command with optional validation parameters
            context: Execution context
            
        Returns:
            CommandResult with validation results
        """
        try:
            config = get_config()
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "sections_checked": []
            }
            
            # Validate required sections
            required_sections = ['command_preprocessing', 'models', 'memory_management']
            for section in required_sections:
                validation_results["sections_checked"].append(section)
                if section not in config:
                    validation_results["errors"].append(f"Missing required section: {section}")
                    validation_results["valid"] = False
            
            # Validate model paths
            if 'models' in config:
                models_config = config['models']
                for model_key in ['flux_image_path', 'main_llm_path', 'embedding_model_path']:
                    if model_key in models_config:
                        model_path = Path(models_config[model_key])
                        if not model_path.exists():
                            validation_results["warnings"].append(
                                f"Model path does not exist: {model_path}"
                            )
            
            # Validate memory management settings
            if 'memory_management' in config:
                memory_config = config['memory_management']
                timeout = memory_config.get('gpu_offload_timeout', 0)
                if timeout < 0:
                    validation_results["errors"].append(
                        "gpu_offload_timeout must be non-negative"
                    )
                    validation_results["valid"] = False
            
            # Validate command preprocessing settings
            if 'command_preprocessing' in config:
                cmd_config = config['command_preprocessing']
                # Check for boolean values
                for bool_key in ['enabled', 'use_main_model', 'store_memory_enabled']:
                    if bool_key in cmd_config and not isinstance(cmd_config[bool_key], bool):
                        validation_results["warnings"].append(
                            f"{bool_key} should be a boolean value"
                        )
            
            status = "valid" if validation_results["valid"] else "invalid"
            message = f"Configuration validation complete: {status}"
            
            if validation_results["errors"]:
                message += f" ({len(validation_results['errors'])} errors)"
            if validation_results["warnings"]:
                message += f" ({len(validation_results['warnings'])} warnings)"
            
            return CommandResult(
                success=True,
                message=message,
                data=validation_results
            )
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to validate configuration: {str(e)}",
                error_code="CONFIG_VALIDATION_FAILED"
            )
    
    @staticmethod
    def handle_list_config_sections(command: Command, context: ExecutionContext) -> CommandResult:
        """
        List all configuration sections.
        
        Args:
            command: Command object
            context: Execution context
            
        Returns:
            CommandResult with list of configuration sections
        """
        try:
            config = get_config()
            sections = list(config.keys())
            
            # Get section details
            section_details = {}
            for section in sections:
                section_data = config[section]
                if isinstance(section_data, dict):
                    section_details[section] = {
                        "type": "object",
                        "keys": list(section_data.keys()),
                        "key_count": len(section_data)
                    }
                else:
                    section_details[section] = {
                        "type": type(section_data).__name__,
                        "value": section_data
                    }
            
            return CommandResult(
                success=True,
                message=f"Found {len(sections)} configuration sections",
                data={
                    "sections": sections,
                    "section_details": section_details
                }
            )
            
        except Exception as e:
            logger.error(f"Error listing configuration sections: {e}")
            return CommandResult(
                success=False,
                message=f"Failed to list configuration sections: {str(e)}",
                error_code="CONFIG_LIST_FAILED"
            )
