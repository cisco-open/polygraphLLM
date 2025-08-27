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
Scorer utilities for PolyGraph.

Provides similarity scoring functionality using BertScore and N-gram methods.
"""

import logging
from typing import List, Any, Dict, Optional

logger = logging.getLogger(__name__)


class BertScorer:
    """BertScore-based similarity scorer."""
    
    def __init__(self):
        """Initialize BertScore scorer."""
        try:
            # Import the configured BertScore scorer
            from ...polygraphLLM.config import bertscorer
            self._scorer = bertscorer
        except Exception as e:
            logger.error(f"Failed to initialize BertScore scorer: {e}")
            self._scorer = None
    
    def predict(
        self, 
        sentences: List[str], 
        sampled_passages: List[str], 
        **kwargs
    ) -> Any:
        """
        Compute BertScore similarity between sentences and passages.
        
        Args:
            sentences: List of sentences
            sampled_passages: List of sampled passages
            **kwargs: Additional parameters
            
        Returns:
            Similarity scores
        """
        try:
            if self._scorer is None:
                logger.warning("BertScore scorer not available")
                return None
            
            return self._scorer.predict(
                sentences=sentences,
                sampled_passages=sampled_passages,
                **kwargs
            )
        except Exception as e:
            logger.error(f"BertScore prediction failed: {e}")
            return None


class NgramScorer:
    """N-gram based similarity scorer."""
    
    def __init__(self):
        """Initialize N-gram scorer."""
        try:
            # Import the configured N-gram scorer
            from ...polygraphLLM.config import ngramscorer
            self._scorer = ngramscorer
        except Exception as e:
            logger.error(f"Failed to initialize N-gram scorer: {e}")
            self._scorer = None
    
    def predict(
        self,
        passage: str,
        sentences: List[str],
        sampled_passages: List[str],
        **kwargs
    ) -> Dict:
        """
        Compute N-gram similarity scores.
        
        Args:
            passage: The main passage
            sentences: List of sentences
            sampled_passages: List of sampled passages
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing similarity scores
        """
        try:
            if self._scorer is None:
                logger.warning("N-gram scorer not available")
                return {'doc_level': {'avg_neg_logprob': 0.0}}
            
            return self._scorer.predict(
                passage=passage,
                sentences=sentences,
                sampled_passages=sampled_passages,
                **kwargs
            )
        except Exception as e:
            logger.error(f"N-gram scoring failed: {e}")
            return {'doc_level': {'avg_neg_logprob': 0.0}}


# Convenience functions
def compute_bertscore(
    sentences: List[str], 
    sampled_passages: List[str], 
    **kwargs
) -> Any:
    """Compute BertScore similarity."""
    scorer = BertScorer()
    return scorer.predict(sentences, sampled_passages, **kwargs)


def compute_ngram_score(
    passage: str,
    sentences: List[str],
    sampled_passages: List[str],
    **kwargs
) -> Dict:
    """Compute N-gram similarity scores."""
    scorer = NgramScorer()
    return scorer.predict(passage, sentences, sampled_passages, **kwargs)
