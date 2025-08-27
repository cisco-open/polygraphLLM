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
Checker utilities for PolyGraph.

Provides fact-checking functionality for triplets against reference documents.
"""

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class Checker:
    """Fact-checking utility for triplets."""
    
    def __init__(self):
        """Initialize checker."""
        try:
            # Import the configured checker
            from ...polygraphLLM.config import checker
            self._checker = checker
        except Exception as e:
            logger.error(f"Failed to initialize checker: {e}")
            self._checker = None
    
    def check(
        self, 
        triplet: Tuple, 
        reference: List[str], 
        question: str = None, 
        **kwargs
    ) -> str:
        """
        Check a triplet against reference documents.
        
        Args:
            triplet: The triplet to check
            reference: Reference documents
            question: Optional question context
            **kwargs: Additional parameters
            
        Returns:
            Check result ("Entailment", "Neutral", "Contradiction", "Abstain")
        """
        try:
            if self._checker is None:
                logger.warning("Checker not available")
                return "Abstain"
            
            return self._checker.check(triplet, reference, question=question, **kwargs)
        except Exception as e:
            logger.error(f"Triplet checking failed: {e}")
            return "Abstain"


# Convenience function
def check_triplet(
    triplet: Tuple, 
    reference: List[str], 
    question: str = None, 
    **kwargs
) -> str:
    """Check a triplet against reference documents."""
    checker = Checker()
    return checker.check(triplet, reference, question, **kwargs)
