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
Generator utilities for PolyGraph.

Provides question generation functionality.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Question generation utility."""
    
    def __init__(self):
        """Initialize question generator."""
        try:
            # Import the configured question generator
            from ...polygraphLLM.config import question_generator
            self._generator = question_generator
        except Exception as e:
            logger.error(f"Failed to initialize question generator: {e}")
            self._generator = None
    
    def generate(self, text: str, **kwargs) -> str:
        """
        Generate a question from text.
        
        Args:
            text: The text to generate a question from
            **kwargs: Additional parameters
            
        Returns:
            Generated question
        """
        try:
            if self._generator is None:
                logger.warning("Question generator not available")
                return "Unable to generate question"
            
            return self._generator.generate(text, **kwargs)
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return "Unable to generate question"


# Convenience function
def generate_question(text: str, **kwargs) -> str:
    """Generate a question from text."""
    generator = QuestionGenerator()
    return generator.generate(text, **kwargs)
