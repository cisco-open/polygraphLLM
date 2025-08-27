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
SelfCheckGPT: Self-consistency based hallucination detection.

This module implements multiple variants of SelfCheckGPT that use 
self-consistency checking to detect hallucinations.
"""

import logging
import numpy as np
import spacy
from typing import Dict, Tuple, Optional, List

from ..utils.base_detector import BaseDetector
from ...polygraphLLM.prompts.default import DEFAULT_SELFCHECK_WITH_PROMPT_PROMPT

logger = logging.getLogger(__name__)


def selfcheckgpt_bertscore(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    n_samples: int = 5,
    temperature: float = 0.8,
    threshold: float = 0.5,
    **kwargs
) -> Tuple[Dict, bool]:
    """
    Detect hallucinations using SelfCheckGPT with BertScore.
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused)
        samples: Pre-generated samples (if None, generates them)
        n_samples: Number of samples to generate
        temperature: Temperature for sample generation
        threshold: Threshold for hallucination decision (0.0-1.0)
        **kwargs: Additional parameters
        
    Returns:
        explanation (dict): Contains similarity analysis results
        is_hallucination (bool): True if similarity below threshold
    """
    detector = BaseDetector()
    
    try:
        # Generate answer if not provided
        if not answer:
            answer = detector.ask_llm(question)[0]
        
        # Extract sentences from answer
        sentences = detector.extract_sentences(answer)
        sentences = [s.text for s in sentences]
        
        # Generate samples if not provided
        if not samples:
            samples = detector.ask_llm(question, n=n_samples, temperature=temperature)
        
        # Compute BertScore similarities
        scores = detector.similarity_bertscore(
            sentences=sentences,
            sampled_passages=samples
        )
        
        # Calculate average score
        if hasattr(scores, '__iter__'):
            avg_score = float("{:.2f}".format(sum(scores)/len(scores)))
        else:
            avg_score = 0.0 if scores == 'FAILED' else float(scores)
        
        # Prepare explanation
        explanation = {
            'algorithm': 'selfcheckgpt_bertscore',
            'score': avg_score,
            'threshold': threshold,
            'n_samples': n_samples,
            'temperature': temperature,
            'sentences': sentences,
            'samples': samples,
            'answer': answer,
            'question': question
        }
        
        # Make binary decision (low similarity = potential hallucination)
        is_hallucination = avg_score < threshold
        
        logger.info(f'SelfCheckGPT-BertScore: score={avg_score:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return explanation, is_hallucination
        
    except Exception as e:
        logger.error(f'SelfCheckGPT-BertScore failed: {e}')
        explanation = {
            'algorithm': 'selfcheckgpt_bertscore',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return explanation, False


def selfcheckgpt_ngram(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    n_samples: int = 5,
    temperature: float = 0.8,
    threshold: float = 0.5,
    **kwargs
) -> Tuple[Dict, bool]:
    """
    Detect hallucinations using SelfCheckGPT with N-gram similarity.
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused)
        samples: Pre-generated samples (if None, generates them)
        n_samples: Number of samples to generate
        temperature: Temperature for sample generation
        threshold: Threshold for hallucination decision (0.0-1.0)
        **kwargs: Additional parameters
        
    Returns:
        explanation (dict): Contains N-gram similarity results
        is_hallucination (bool): True if similarity indicates hallucination
    """
    detector = BaseDetector()
    
    try:
        # Generate answer if not provided
        if not answer:
            answer = detector.ask_llm(question)[0]
        
        # Extract sentences from answer
        sentences = detector.extract_sentences(answer)
        sentences = [s.text for s in sentences]
        
        # Generate samples if not provided
        if not samples:
            samples = detector.ask_llm(question, n=n_samples, temperature=temperature)
        
        # Compute N-gram similarities
        scores = detector.similarity_ngram(
            sentences=sentences,
            passage=answer,
            sampled_passages=samples
        )
        
        # Extract average negative log probability
        avg_neg_logprob = float("{:.2f}".format(scores['doc_level']['avg_neg_logprob']))
        
        # Prepare explanation
        explanation = {
            'algorithm': 'selfcheckgpt_ngram',
            'score': avg_neg_logprob,
            'threshold': threshold,
            'n_samples': n_samples,
            'temperature': temperature,
            'sentences': sentences,
            'samples': samples,
            'answer': answer,
            'question': question
        }
        
        # Make binary decision (high negative log prob = potential hallucination)
        is_hallucination = avg_neg_logprob > threshold
        
        logger.info(f'SelfCheckGPT-NGram: score={avg_neg_logprob:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return explanation, is_hallucination
        
    except Exception as e:
        logger.error(f'SelfCheckGPT-NGram failed: {e}')
        explanation = {
            'algorithm': 'selfcheckgpt_ngram',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return explanation, False


def selfcheckgpt_prompt(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    n_samples: int = 5,
    temperature: float = 0.8,
    threshold: float = 0.5,
    prompt_template: str = None,
    **kwargs
) -> Tuple[Dict, bool]:
    """
    Detect hallucinations using SelfCheckGPT with prompt-based checking.
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused)
        samples: Pre-generated samples (if None, generates them)
        n_samples: Number of samples to generate
        temperature: Temperature for sample generation
        threshold: Threshold for hallucination decision (0.0-1.0)
        prompt_template: Custom prompt template
        **kwargs: Additional parameters
        
    Returns:
        explanation (dict): Contains prompt-based check results
        is_hallucination (bool): True if checks indicate hallucination
    """
    detector = BaseDetector()
    
    try:
        if prompt_template is None:
            prompt_template = DEFAULT_SELFCHECK_WITH_PROMPT_PROMPT
        
        text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        
        # Generate answer if not provided
        if not answer:
            answer = detector.ask_llm(question)[0]
        
        # Generate samples if not provided
        if not samples:
            samples = detector.ask_llm(question, n=n_samples, temperature=temperature)
        
        # Extract sentences from answer
        sentences = detector.extract_sentences(answer)
        sentences = [s.text for s in sentences]
        
        # Check each sentence against each sample
        scores = np.zeros((len(sentences), len(samples)))
        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(samples):
                sample = sample.strip()
                prompt = prompt_template.format(context=sample, sentence=sentence)
                generate_text = detector.ask_llm(prompt)[0]
                generate_text = generate_text.replace(prompt, "")
                score_ = _text_postprocessing(generate_text, text_mapping)
                scores[sent_i, sample_i] = score_
        
        # Calculate average score
        scores_per_sentence = scores.mean(axis=-1)
        avg_score = sum(scores_per_sentence) / len(scores_per_sentence)
        
        # Prepare explanation
        explanation = {
            'algorithm': 'selfcheckgpt_prompt',
            'score': avg_score,
            'threshold': threshold,
            'n_samples': n_samples,
            'temperature': temperature,
            'sentences': sentences,
            'samples': samples,
            'answer': answer,
            'question': question,
            'sentence_scores': scores_per_sentence.tolist()
        }
        
        # Make binary decision
        is_hallucination = avg_score >= threshold
        
        logger.info(f'SelfCheckGPT-Prompt: score={avg_score:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return explanation, is_hallucination
        
    except Exception as e:
        logger.error(f'SelfCheckGPT-Prompt failed: {e}')
        explanation = {
            'algorithm': 'selfcheckgpt_prompt',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return explanation, False


def selfcheckgpt_mqag(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    n_samples: int = 3,
    threshold: float = 0.5,
    **kwargs
) -> Tuple[Dict, bool]:
    """
    Detect hallucinations using SelfCheckGPT with MQAG (Question Answering/Generation).
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused)
        samples: Pre-generated samples (if None, generates them)
        n_samples: Number of samples to generate
        threshold: Threshold for hallucination decision (0.0-1.0)
        **kwargs: Additional parameters
        
    Returns:
        explanation (dict): Contains MQAG analysis results
        is_hallucination (bool): True if MQAG score indicates hallucination
    """
    detector = BaseDetector()
    
    try:
        import torch
        from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG
        
        # Initialize MQAG scorer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scorer = SelfCheckMQAG(device=device)
        
        # Generate answer if not provided
        if not answer:
            answer = detector.ask_llm(question)[0]
        
        # Extract sentences using spacy
        nlp = spacy.load("en_core_web_sm")
        sentences = [sent.text.strip() for sent in nlp(answer).sents]
        
        # Generate samples if not provided
        if not samples:
            samples = []
            for _ in range(n_samples):
                samples.append(detector.ask_llm(question)[0])
        
        # Compute MQAG scores
        scores = scorer.predict(
            sentences=sentences,
            passage=answer,
            sampled_passages=samples,
            num_questions_per_sent=5,
            scoring_method='bayes_with_alpha',
            beta1=0.8, beta2=0.8,
        )
        
        # Calculate average score
        avg_score = float("{:.2f}".format(sum(scores)/len(scores)))
        
        # Prepare explanation
        explanation = {
            'algorithm': 'selfcheckgpt_mqag',
            'score': avg_score,
            'threshold': threshold,
            'n_samples': n_samples,
            'sentences': sentences,
            'samples': samples,
            'sentence_scores': scores,
            'answer': answer,
            'question': question
        }
        
        # Make binary decision
        is_hallucination = avg_score >= threshold
        
        logger.info(f'SelfCheckGPT-MQAG: score={avg_score:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return explanation, is_hallucination
        
    except Exception as e:
        logger.error(f'SelfCheckGPT-MQAG failed: {e}')
        explanation = {
            'algorithm': 'selfcheckgpt_mqag',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return explanation, False


def _text_postprocessing(text: str, text_mapping: dict) -> float:
    """
    Map generated text to score.
    Yes -> 0.0
    No  -> 1.0
    everything else -> 0.5
    """
    text = text.lower().strip()
    if text[:3] == 'yes':
        text = 'yes'
    elif text[:2] == 'no':
        text = 'no'
    else:
        text = 'n/a'
    return text_mapping[text]


# Legacy class wrappers for backward compatibility
class SelfCheckGPTBertScore(BaseDetector):
    """Legacy class wrapper for SelfCheckGPT BertScore."""
    
    id = 'self_check_gpt_bertscore'
    display_name = 'Self-Check GPT BertScore'
    
    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        """Legacy interface."""
        n_samples = 5
        temperature = 0.8
        
        if settings:
            n_samples = int(settings.get("BERT_SCORE_SAMPLING_NUMBER", 5))
            temperature = float(settings.get("OPENAI_TEMPERATURE", 0.8))
        
        explanation, is_hallucination = selfcheckgpt_bertscore(
            question=question,
            answer=answer,
            samples=samples,
            n_samples=n_samples,
            temperature=temperature
        )
        
        score = explanation.get('score', 0.0)
        answer = explanation.get('answer', '')
        samples = explanation.get('samples', [])
        
        return score, answer, samples


class SelfCheckGPTNGram(BaseDetector):
    """Legacy class wrapper for SelfCheckGPT NGram."""
    
    id = 'self_check_gpt_ngram'
    display_name = 'Self-Check GPT NGram'
    
    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        """Legacy interface."""
        n_samples = 5
        temperature = 0.8
        
        if settings:
            n_samples = int(settings.get("NGRAM_SAMPLING_NUMBER", 5))
            temperature = float(settings.get("OPENAI_TEMPERATURE", 0.8))
        
        explanation, is_hallucination = selfcheckgpt_ngram(
            question=question,
            answer=answer,
            samples=samples,
            n_samples=n_samples,
            temperature=temperature
        )
        
        score = explanation.get('score', 0.0)
        answer = explanation.get('answer', '')
        samples = explanation.get('samples', [])
        
        return score, answer, samples


class SelfCheckGPTPrompt(BaseDetector):
    """Legacy class wrapper for SelfCheckGPT Prompt."""
    
    id = 'self_check_gpt_prompt'
    display_name = 'Self-Check GPT Prompt'
    
    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        """Legacy interface."""
        n_samples = 5
        temperature = 0.8
        
        if settings:
            n_samples = int(settings.get("GPT_PROMPT_SAMPLING_NUMBER", 5))
            temperature = float(settings.get("OPENAI_TEMPERATURE", 0.8))
        
        explanation, is_hallucination = selfcheckgpt_prompt(
            question=question,
            answer=answer,
            samples=samples,
            n_samples=n_samples,
            temperature=temperature
        )
        
        score = explanation.get('score', 0.0)
        answer = explanation.get('answer', '')
        samples = explanation.get('samples', [])
        
        return score, answer, samples


class SelfCheckGPTMQAG(BaseDetector):
    """Legacy class wrapper for SelfCheckGPT MQAG."""
    
    id = 'self_check_gpt_mqag'
    display_name = 'Self-Check GPT MQAG'
    
    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        """Legacy interface."""
        explanation, is_hallucination = selfcheckgpt_mqag(
            question=question,
            answer=answer,
            samples=samples
        )
        
        score = explanation.get('score', 0.0)
        answer = explanation.get('answer', '')
        samples = explanation.get('samples', [])
        
        return score, answer, samples
