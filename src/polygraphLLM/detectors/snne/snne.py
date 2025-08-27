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

import logging
import torch
import evaluate
from rouge_score import tokenizers
from sentence_transformers import SentenceTransformer

from ..base import Detector

from .uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta, soft_nearest_neighbor_loss

logger = logging.getLogger(__name__)


class SNNE(Detector):
    id = 'snne'
    display_name = 'SNNE (Soft Nearest Neighbor Entropy)'

    def __init__(self):
        super().__init__()
        
        # Initialize models lazily to avoid tokenizer issues
        self.entailment_model = None
        self.embedding_model = None
        self.tokenizer = None
        self.rouge = None
        
        # Default parameters
        self.temperature_choice = [0.1, 1, 10, 100]
        self.variant_choice = ['only_denom']
        self.selfsim_choice = [True]

    def _initialize_models(self):
        """Lazy initialization of models to avoid tokenizer issues"""
        if self.entailment_model is None:
            try:
                logger.info("Attempting to load EntailmentDeberta model")
                self.entailment_model = EntailmentDeberta()
                logger.info("Successfully loaded EntailmentDeberta model")
            except Exception as e:
                logger.error(f"Failed to initialize EntailmentDeberta: {e}")
                logger.info("SNNE will use simplified similarity computation without entailment")
                self.entailment_model = None
        
        if self.embedding_model is None:
            try:
                embedding_models = [
                    "sentence-transformers/all-MiniLM-L6-v2",  # ~22MB, fastest
                    "sentence-transformers/paraphrase-MiniLM-L6-v2",  # ~22MB
                    "sentence-transformers/all-MiniLM-L12-v2",  # ~33MB
                    "sentence-transformers/all-mpnet-base-v2",  # ~420MB
                    "Salesforce/SFR-Embedding-2_R",  # Very large, try last
                ]
                
                for model_name in embedding_models:
                    try:
                        logger.info(f"Attempting to load embedding model: {model_name}")
                        self.embedding_model = SentenceTransformer(
                            model_name,
                            device='mps' if torch.backends.mps.is_available() else 'cpu'
                        )
                        if hasattr(self.embedding_model, 'max_seq_length'):
                            self.embedding_model.max_seq_length = min(512, getattr(self.embedding_model, 'max_seq_length', 512))
                        
                        logger.info(f"Successfully loaded embedding model: {model_name}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {e}")
                        if hasattr(self, 'embedding_model'):
                            del self.embedding_model
                        self.embedding_model = None
                        # Force garbage collection
                        import gc
                        gc.collect()
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                        continue
                        
                if self.embedding_model is None:
                    raise Exception("Failed to load any embedding model")
                    
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                self.embedding_model = None
        
        if self.tokenizer is None:
            try:
                self.tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize
            except Exception as e:
                logger.error(f"Failed to initialize tokenizer: {e}")
                self.tokenizer = None
        
        if self.rouge is None:
            try:
                self.rouge = evaluate.load('rouge', keep_in_memory=True)
            except Exception as e:
                logger.error(f"Failed to initialize ROUGE: {e}")
                self.rouge = None
        """Lazy initialization of models to avoid tokenizer issues"""
        if self.entailment_model is None:
            try:
                self.entailment_model = EntailmentDeberta()
            except Exception as e:
                logger.error(f"Failed to initialize EntailmentDeberta: {e}")
                self.entailment_model = None
        
        if self.embedding_model is None:
            try:
                # Try different embedding models if the first one fails
                embedding_models = [
                    "Salesforce/SFR-Embedding-2_R",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2"
                ]
                
                for model_name in embedding_models:
                    try:
                        self.embedding_model = SentenceTransformer(model_name)
                        logger.info(f"Successfully loaded embedding model: {model_name}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {e}")
                        continue
                        
                if self.embedding_model is None:
                    raise Exception("Failed to load any embedding model")
                    
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                self.embedding_model = None
        
        if self.tokenizer is None:
            try:
                self.tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize
            except Exception as e:
                logger.error(f"Failed to initialize tokenizer: {e}")
                self.tokenizer = None
        
        if self.rouge is None:
            try:
                self.rouge = evaluate.load('rouge', keep_in_memory=True)
            except Exception as e:
                logger.error(f"Failed to initialize ROUGE: {e}")
                self.rouge = None

    def compute_lexical_similarity(self, generations):
        """Compute lexical similarity matrix for generations"""
        if self.rouge is None:
            # Fallback to simple string similarity if ROUGE is not available
            similarity_matrix = []
            for i, gen_i in enumerate(generations):
                row = []
                for j, gen_j in enumerate(generations):
                    if i == j:
                        row.append(1.0)
                    else:
                        # Simple Jaccard similarity as fallback
                        words_i = set(gen_i.lower().split())
                        words_j = set(gen_j.lower().split())
                        intersection = len(words_i.intersection(words_j))
                        union = len(words_i.union(words_j))
                        similarity = intersection / union if union > 0 else 0.0
                        row.append(similarity)
                similarity_matrix.append(row)
            return torch.tensor(similarity_matrix)
        
        similarity_matrix = []
        for i, gen_i in enumerate(generations):
            row = []
            for j, gen_j in enumerate(generations):
                if i == j:
                    row.append(1.0)
                else:
                    try:
                        # Compute ROUGE-L similarity as a proxy for lexical similarity
                        rouge_scores = self.rouge.compute(
                            predictions=[gen_i],
                            references=[gen_j],
                            use_stemmer=False
                        )
                        row.append(rouge_scores['rougeL'])
                    except Exception as e:
                        logger.warning(f"ROUGE computation failed, using fallback: {e}")
                        # Fallback to simple similarity
                        words_i = set(gen_i.lower().split())
                        words_j = set(gen_j.lower().split())
                        intersection = len(words_i.intersection(words_j))
                        union = len(words_i.union(words_j))
                        similarity = intersection / union if union > 0 else 0.0
                        row.append(similarity)
            similarity_matrix.append(row)
        return torch.tensor(similarity_matrix)

    def compute_snne_score(self, generations, semantic_ids=None, variant='only_denom', 
                          temperature=1, selfsim=True):
        """Compute SNNE score for a list of generations"""
        self._initialize_models()
        
        if self.entailment_model is None or self.embedding_model is None:
            logger.error("Required models not available for SNNE computation")
            return 0.5  # Return neutral score
        
        if semantic_ids is None:
            semantic_ids = list(range(len(generations)))
        
        try:
            similarity_matrix = self.compute_lexical_similarity(generations)
            
            snne = soft_nearest_neighbor_loss(
                generations,
                self.entailment_model, 
                self.embedding_model, 
                semantic_ids,
                similarity_matrix=similarity_matrix,
                variant=variant, 
                temperature=temperature, 
                exclude_diagonal=not selfsim
            ).item()
            
            return snne
            
        except Exception as e:
            logger.error(f"SNNE computation failed: {e}")
            return 0.5  # Return neutral score on failure

    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        """
        Main scoring function following the detector interface
        """
        try:
            n = 5
            temperature =  1.0
            variant = 'only_denom'
            selfsim = True
            
            # Generate answer if not provided
            if not answer:
                try:
                    answer = self.ask_llm(question.strip())[0]
                except Exception as e:
                    logger.error(f"Failed to generate answer: {e}")
                    answer = "Unable to generate answer"
            
            # Generate multiple samples for SNNE computation
            if samples is None:
                try:
                    samples = self.ask_llm(question.strip(), n=n, temperature=0.8)
                except Exception as e:
                    logger.error(f"Failed to generate samples: {e}")
                    samples = [answer]
            
            # Ensure we have the original answer in our samples
            if answer not in samples:
                samples = [answer] + samples
            
            # Ensure we have enough samples for meaningful computation
            if len(samples) < 2:
                logger.warning("Not enough samples for SNNE computation")
                return 0.5, answer, samples
            
            # Compute SNNE score
            snne_score = self.compute_snne_score(
                samples, 
                variant=variant,
                temperature=temperature,
                selfsim=selfsim
            )
            
            logger.info(f'SNNE score computed: {snne_score}')
            
            return snne_score, answer, samples
            
        except Exception as e:
            logger.error(f'SNNE scoring failed: {e}')
            return 0.5, answer or "Error", samples or []

    def batch_score(self, questions, answers=None, settings=None):
        """
        Batch scoring function adapted from original code structure
        """
        if answers is None:
            answers = [self.ask_llm(q.strip())[0] for q in questions]
        
        results = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            score, answer, samples = self.score(question, answer, settings=settings)
            results.append({
                'question': question,
                'answer': answer,
                'score': score,
                'samples': samples
            })
        
        return results
