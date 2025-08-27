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
Retriever utilities for PolyGraph.

Provides document retrieval functionality for reference-based algorithms.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class GoogleRetriever:
    """Google search-based document retriever."""
    
    def __init__(self):
        """Initialize Google retriever."""
        try:
            # Import the configured retriever
            from ...polygraphLLM.config import retriever
            self._retriever = retriever
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            self._retriever = None
    
    def retrieve(self, queries: List[str], **kwargs) -> List[str]:
        """
        Retrieve relevant documents for queries.
        
        Args:
            queries: List of queries to retrieve documents for
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved documents
        """
        try:
            if self._retriever is None:
                logger.warning("Retriever not available")
                return []
            
            return self._retriever.retrieve(queries, **kwargs)
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []


class BaseRetriever:
    """Base retriever interface."""
    
    def __init__(self, retriever_type: str = "google"):
        """
        Initialize retriever.
        
        Args:
            retriever_type: Type of retriever to use
        """
        self.retriever_type = retriever_type.lower()
        self._retriever = None
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the appropriate retriever."""
        if self.retriever_type == "google":
            self._retriever = GoogleRetriever()
        else:
            raise ValueError(f"Unsupported retriever type: {self.retriever_type}")
    
    def retrieve(self, queries: List[str], **kwargs) -> List[str]:
        """Retrieve documents for queries."""
        return self._retriever.retrieve(queries, **kwargs)


# Convenience function
def retrieve_documents(queries: List[str], retriever_type: str = "google", **kwargs) -> List[str]:
    """Retrieve documents for queries."""
    retriever = BaseRetriever(retriever_type)
    return retriever.retrieve(queries, **kwargs)
