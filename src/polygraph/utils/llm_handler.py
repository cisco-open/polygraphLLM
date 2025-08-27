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
LLM Handler utilities for PolyGraph.

Provides unified interface to various LLM providers (OpenAI, Cohere, Mistral).
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


class LLMHandler:
    """Unified LLM handler interface."""
    
    def __init__(self, provider: str = "openai"):
        """
        Initialize LLM handler.
        
        Args:
            provider: LLM provider ("openai", "cohere", "mistral")
        """
        self.provider = provider.lower()
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "openai":
            from ...polygraphLLM.llm.openai import OpenAIHandler
            self._client = OpenAIHandler()
        elif self.provider == "cohere":
            from ...polygraphLLM.llm.cohere import CohereHandler
            self._client = CohereHandler()
        elif self.provider == "mistral":
            from ...polygraphLLM.llm.mistral import MistralHandler
            self._client = MistralHandler()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def ask_llm(
        self, 
        prompt: str, 
        n: int = 1, 
        temperature: float = 0.7, 
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Query the LLM with a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            n: Number of responses to generate
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of generated responses
        """
        try:
            if max_tokens is None:
                max_tokens = int(os.getenv("LLM_MAX_TOKENS", 400))
            
            return self._client.ask_llm(
                prompt=prompt,
                n=n,
                temperature=temperature,
                max_new_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return [f"Error: {str(e)}"]
    
    def get_provider(self) -> str:
        """Get the current provider name."""
        return self.provider


# Legacy wrapper for backward compatibility
def get_llm_handler(provider: str = None) -> LLMHandler:
    """Get LLM handler instance."""
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai")
    return LLMHandler(provider)
