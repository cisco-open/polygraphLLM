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
ChatProtect: Hallucination detection using consistency checking.

This algorithm detects inconsistencies in generated text by checking for 
contradictions between statements using Chain of Thought reasoning.
"""

import logging
import re
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

from ..utils.base_detector import BaseDetector

logger = logging.getLogger(__name__)

CHECKED = 0
INCONSISTENT = 1  
SPURIOUS = -1


def chatprotect(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    threshold: float = 0.5,
    **kwargs
) -> Tuple[Dict, bool]:
    """
    Detect hallucinations using ChatProtect consistency checking.
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused in ChatProtect)
        samples: Pre-generated samples (unused in ChatProtect)
        threshold: Threshold for inconsistency ratio (0.0-1.0)
        **kwargs: Additional parameters
        
    Returns:
        explanation (dict): Contains consistency analysis results
        is_hallucination (bool): True if inconsistencies exceed threshold
    """
    detector = BaseDetector()
    
    try:
        # Generate answer if not provided
        if not answer:
            answer = detector.ask_llm(question.strip())[0]
        
        # Extract sentences from answer
        sentences = detector.extract_sentences(answer)
        sent_tagged_dict = defaultdict(list)
        prefix = ""
        ok_count, inconsistent_count = 0, 0
        inconsistency_details = []
        
        for sentence in sentences:
            # Skip if sentence already processed
            if sentence in sent_tagged_dict:
                for ext_sent in sent_tagged_dict[sentence]:
                    if ext_sent.tag == "ok":
                        ok_count += 1
                    else:
                        inconsistent_count += 1
                continue
            
            # Extract triplets from sentence
            triplets = detector.extract_triplets(sentence)
            if not triplets:
                triplets = [answer]  # Fallback to full answer
            
            # Check each triplet for consistency
            for triple in triplets:
                try:
                    # Generate alternative statement
                    alt_statement = _generate_statement_missing_object(
                        detector, triple[0], triple[1], question, prefix
                    )[0]
                    
                    # Get explanation for consistency
                    explanation_text = _explain_consistent_cot(
                        detector, sentence, alt_statement, question, prefix
                    )[0]
                    
                    # Check for contradiction
                    label = _check_consistent_cot(
                        detector, sentence, alt_statement, question, prefix, explanation_text
                    )
                    
                    tag = _label_to_tag(label)
                    if tag == "ok":
                        ok_count += 1
                    else:
                        inconsistent_count += 1
                        inconsistency_details.append({
                            'sentence': sentence,
                            'alternative': alt_statement, 
                            'explanation': explanation_text,
                            'label': label
                        })
                        
                except Exception as e:
                    logging.error(f"Error processing sentence: {sentence}. Error: {str(e)}")
        
        # Calculate inconsistency ratio
        total_checks = ok_count + inconsistent_count
        inconsistency_ratio = inconsistent_count / total_checks if total_checks > 0 else 0.0
        
        # Prepare explanation
        explanation = {
            'algorithm': 'chatprotect',
            'inconsistency_count': inconsistent_count,
            'ok_count': ok_count,
            'inconsistency_ratio': inconsistency_ratio,
            'threshold': threshold,
            'inconsistency_details': inconsistency_details,
            'answer': answer,
            'question': question
        }
        
        # Make binary decision
        is_hallucination = inconsistency_ratio >= threshold
        
        logger.info(f'ChatProtect: inconsistencies={inconsistent_count}/{total_checks}, ratio={inconsistency_ratio:.3f}, hallucination={is_hallucination}')
        
        return explanation, is_hallucination
        
    except Exception as e:
        logger.error(f'ChatProtect failed: {e}')
        explanation = {
            'algorithm': 'chatprotect',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return explanation, False


def _label_to_tag(label):
    """Convert label to tag."""
    if label == INCONSISTENT:
        return "strong"
    elif label == CHECKED:
        return "ok"
    else:
        return "ok"


def _explain_consistent_cot(detector, stmt1, stmt2, target, prefix):
    """Ask the model if it finds a contradiction between the sentences."""
    explain_prompt = f"""\
        I give you the beginning of a text answering the prompt "{target}".
        Then follow two statements.

        Text:
        {prefix}

        Statement 1:
        {stmt1}

        Statement 2:
        {stmt2}

        Please explain whether the statements about {target} are contradictory or factually wrong.
        Provide your explanation only.
    """
    
    res = detector.ask_llm(explain_prompt)
    return res


def _check_consistent_cot(detector, stmt1, stmt2, target, prefix, reason):
    """Check consistency using Chain of Thought."""
    if stmt1 == stmt2:
        return CHECKED
        
    explain_prompt = f"""\
        I gave the beginning of a text answering the prompt "{target}".
        Then followed two statements.

        Text:
        {prefix}

        Statement 1:
        {stmt1}

        Statement 2:
        {stmt2}

        I asked whether the statements about {target} are contradictory.
        The explanation is:
        {reason}

        Please conclude whether the statements are contradictory with Yes or No.
    """
    
    conclusions_raw = detector.ask_llm(explain_prompt)
    conclusions = []
    
    for conclusion in conclusions_raw:
        follows_yes = re.findall(r"\byes\b", conclusion.lower())
        follows_no = re.findall(r"\bno\b", conclusion.lower())
        
        if follows_yes and not follows_no:
            conclusions.append(INCONSISTENT)
        elif follows_no and not follows_yes:
            conclusions.append(CHECKED)
        else:
            conclusions.append(CHECKED)  # Default to checked
            
    return sum(conclusions) / len(conclusions)


def _generate_statement_missing_object(detector, subject, predicate, target, prefix):
    """Generate a follow up statement for the description based on a triple."""
    topic = target.replace("Please tell me about ", "")
    if not prefix:
        prefix = "There is no preceding description."
        
    statement_template = """
        You are a description generator. You are given the start of an description and a question that should be answered by the next sentence. You return the next sentence for the description.
        Here is the start of a description about {}:
        {}

        Please generate the next sentence of this description.
        The generated sentence must fill the gap in this Subject;Predicate;Object triple: ({}; {}; _)
        The sentence should contain as little other information as possible.
    """
    
    generate_sentence = detector.ask_llm(statement_template.format(topic, prefix, subject, predicate))
    return generate_sentence


# Legacy class wrapper for backward compatibility
class ChatProtect(BaseDetector):
    """Legacy class wrapper for ChatProtect algorithm."""
    
    id = 'chatProtect'
    display_name = 'ChatProtect'
    
    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        """Legacy interface."""
        explanation, is_hallucination = chatprotect(
            question=question,
            answer=answer
        )
        
        # Convert to legacy format  
        inconsistencies = explanation.get('inconsistency_count', 0)
        answer = explanation.get('answer', '')
        summary = []
        
        if inconsistencies > 0:
            summary.append(f"the count of detected hallucinations is {inconsistencies}")
            for detail in explanation.get('inconsistency_details', []):
                summary.append(detail.get('explanation', ''))
        else:
            summary.append("Hallucination is not detected.")
            
        return inconsistencies, answer, summary
