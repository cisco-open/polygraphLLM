# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Settings utilities for PolyGraph.

Provides configuration management functionality.
"""

import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)


class Settings:
    """Settings management utility."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize settings manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        try:
            # Import the configured settings manager
            from ...polygraphLLM.settings.settings import Settings as OriginalSettings
            self._settings = OriginalSettings(config_path)
        except Exception as e:
            logger.error(f"Failed to initialize settings manager: {e}")
            self._settings = None
    
    def get_custom_settings_value(
        self, 
        settings: Optional[Dict], 
        search_key: str, 
        default: Any = None
    ) -> Any:
        """
        Get a value from custom settings.
        
        Args:
            settings: Settings dictionary
            search_key: Key to search for
            default: Default value if not found
            
        Returns:
            Found value or default
        """
        try:
            if settings and search_key in settings:
                return settings[search_key]
            
            if self._settings is None:
                logger.warning("Settings manager not available")
                return default
            
            return self._settings.get_custom_settings_value(settings, search_key)
        except Exception as e:
            logger.error(f"Settings lookup failed: {e}")
            return default
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.
        
        Args:
            key: Setting key
            default: Default value if not found
            
        Returns:
            Setting value or default
        """
        try:
            if self._settings is None:
                return default
            
            # Try to get from settings object
            if hasattr(self._settings, 'get_setting'):
                return self._settings.get_setting(key, default)
            else:
                return default
        except Exception as e:
            logger.error(f"Setting retrieval failed: {e}")
            return default


# Convenience functions
def get_setting(key: str, default: Any = None, config_path: str = None) -> Any:
    """Get a setting value."""
    settings = Settings(config_path)
    return settings.get_setting(key, default)


def get_custom_setting(
    settings: Optional[Dict], 
    search_key: str, 
    default: Any = None,
    config_path: str = None
) -> Any:
    """Get a custom setting value."""
    settings_manager = Settings(config_path)
    return settings_manager.get_custom_settings_value(settings, search_key, default)
