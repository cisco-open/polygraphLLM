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
Extractor utilities for PolyGraph.

Provides triplet and sentence extraction functionality.
"""

import logging
from typing import List, Any, Optional

logger = logging.getLogger(__name__)


class TripletExtractor:
    """Triplet extraction utility."""
    
    def __init__(self):
        """Initialize triplet extractor."""
        try:
            # Import the configured triplet extractor
            from ...polygraphLLM.config import triplets_extractor
            self._extractor = triplets_extractor
        except Exception as e:
            logger.error(f"Failed to initialize triplet extractor: {e}")
            self._extractor = None
    
    def extract(self, text: str, question: str = None, **kwargs) -> List[tuple]:
        """
        Extract triplets from text.
        
        Args:
            text: The text to extract triplets from
            question: Optional question context
            **kwargs: Additional parameters
            
        Returns:
            List of extracted triplets
        """
        try:
            if self._extractor is None:
                logger.warning("Triplet extractor not available")
                return []
            
            if question:
                return self._extractor.extract(text, question, **kwargs)
            else:
                return self._extractor.extract(text, **kwargs)
        except Exception as e:
            logger.error(f"Triplet extraction failed: {e}")
            return []


class SentenceExtractor:
    """Sentence extraction utility."""
    
    def __init__(self):
        """Initialize sentence extractor."""
        try:
            # Import the configured sentence extractor
            from ...polygraphLLM.config import sentence_extractor
            self._extractor = sentence_extractor
        except Exception as e:
            logger.error(f"Failed to initialize sentence extractor: {e}")
            self._extractor = None
    
    def extract(self, text: str, **kwargs) -> List[Any]:
        """
        Extract sentences from text.
        
        Args:
            text: The text to extract sentences from
            **kwargs: Additional parameters
            
        Returns:
            List of extracted sentences
        """
        try:
            if self._extractor is None:
                logger.warning("Sentence extractor not available")
                return []
            
            return self._extractor.extract(text, **kwargs)
        except Exception as e:
            logger.error(f"Sentence extraction failed: {e}")
            return []


# Convenience functions
def extract_triplets(text: str, question: str = None, **kwargs) -> List[tuple]:
    """Extract triplets from text."""
    extractor = TripletExtractor()
    return extractor.extract(text, question, **kwargs)


def extract_sentences(text: str, **kwargs) -> List[Any]:
    """Extract sentences from text."""
    extractor = SentenceExtractor()
    return extractor.extract(text, **kwargs)
