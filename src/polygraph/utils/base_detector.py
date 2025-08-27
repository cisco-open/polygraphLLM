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
Base detector utility providing common functionality for all algorithms.

This module provides a clean interface to the underlying polygraphLLM infrastructure
while maintaining backward compatibility.
"""

import os
import logging
from typing import List, Any, Optional

logger = logging.getLogger(__name__)


class BaseDetector:
    """
    Base detector providing common functionality for all algorithms.
    
    This class serves as a clean interface to the polygraphLLM infrastructure,
    providing access to LLM handlers, extractors, and other utilities.
    """
    
    def __init__(self):
        """Initialize the base detector with all required components."""
        try:
            # Import and initialize the polygraphLLM config
            from ...polygraphLLM.config import init_config
            
            config_path = os.path.join(os.path.dirname(__file__), '../../polygraphLLM', 'config.json')
            init_config(config_path)
            
            # Import configured components
            from ...polygraphLLM.config import (
                llm_handler, triplets_extractor, sentence_extractor, question_generator,
                retriever, checker, bertscorer, ngramscorer,
            )
            
            self.llm_handler = llm_handler
            self.triplets_extractor = triplets_extractor
            self.sentence_extractor = sentence_extractor
            self.question_generator = question_generator
            self.retriever = retriever
            self.checker = checker
            self.bertscorer = bertscorer
            self.ngramscorer = ngramscorer
            
            # Initialize settings manager
            from ...polygraphLLM.settings.settings import Settings
            self.settings_manager = Settings(config_path)
            
        except Exception as e:
            logger.error(f"Failed to initialize BaseDetector: {e}")
            raise
    
    def ask_llm(self, prompt: str, n: int = 1, temperature: float = 0.7, **kwargs) -> List[str]:
        """
        Query the LLM with a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            n: Number of responses to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            List of generated responses
        """
        try:
            return self.llm_handler.ask_llm(prompt, n, temperature=temperature, **kwargs)
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return ["Error: Failed to generate response"]
    
    def extract_triplets(self, text: str, question: str = None, **kwargs) -> List[tuple]:
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
            if question:
                return self.triplets_extractor.extract(text, question, **kwargs)
            else:
                return self.triplets_extractor.extract(text, **kwargs)
        except Exception as e:
            logger.error(f"Triplet extraction failed: {e}")
            return []
    
    def extract_sentences(self, text: str, **kwargs) -> List[Any]:
        """
        Extract sentences from text.
        
        Args:
            text: The text to extract sentences from
            **kwargs: Additional parameters
            
        Returns:
            List of extracted sentences
        """
        try:
            return self.sentence_extractor.extract(text, **kwargs)
        except Exception as e:
            logger.error(f"Sentence extraction failed: {e}")
            return []
    
    def generate_question(self, text: str, **kwargs) -> str:
        """
        Generate a question from text.
        
        Args:
            text: The text to generate a question from
            **kwargs: Additional parameters
            
        Returns:
            Generated question
        """
        try:
            return self.question_generator.generate(text, **kwargs)
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return "Unable to generate question"
    
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
            return self.retriever.retrieve(queries, **kwargs)
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def check(self, triplet: tuple, reference: List[str], question: str = None, **kwargs) -> str:
        """
        Check a triplet against reference documents.
        
        Args:
            triplet: The triplet to check
            reference: Reference documents
            question: Optional question context
            **kwargs: Additional parameters
            
        Returns:
            Check result
        """
        try:
            return self.checker.check(triplet, reference, question=question, **kwargs)
        except Exception as e:
            logger.error(f"Triplet checking failed: {e}")
            return "Abstain"
    
    def similarity_bertscore(self, sentences: List[str], sampled_passages: List[str], **kwargs) -> Any:
        """
        Compute BertScore similarity.
        
        Args:
            sentences: List of sentences
            sampled_passages: List of sampled passages
            **kwargs: Additional parameters
            
        Returns:
            Similarity scores
        """
        try:
            return self.bertscorer.predict(
                sentences=sentences,
                sampled_passages=sampled_passages,
                **kwargs
            )
        except Exception as e:
            logger.error(f'BertScore failed: {e}')
            return None
    
    def similarity_ngram(self, sentences: List[str], passage: str, sampled_passages: List[str], **kwargs) -> dict:
        """
        Compute N-gram similarity.
        
        Args:
            sentences: List of sentences
            passage: The main passage
            sampled_passages: List of sampled passages
            **kwargs: Additional parameters
            
        Returns:
            Similarity scores
        """
        try:
            return self.ngramscorer.predict(
                passage=passage,
                sentences=sentences,
                sampled_passages=sampled_passages,
                **kwargs
            )
        except Exception as e:
            logger.error(f'N-gram scoring failed: {e}')
            return {'doc_level': {'avg_neg_logprob': 0.0}}
    
    def find_settings_value(self, settings: Optional[dict], search_key: str, default: Any = None) -> Any:
        """
        Find a value in settings.
        
        Args:
            settings: Settings dictionary
            search_key: Key to search for
            default: Default value if not found
            
        Returns:
            Found value or default
        """
        if settings and search_key in settings:
            return settings[search_key]
        
        try:
            return self.settings_manager.get_custom_settings_value(settings, search_key)
        except Exception:
            return default
