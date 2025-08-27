#!/usr/bin/env python3
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
Example usage of the standardized PolyGraph API.

This script demonstrates how to use the new clean and standardized
PolyGraph library for hallucination detection.
"""

import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Non-uncertainty algorithms
from polygraph.algorithms import chainpoll, chatprotect, geval, refchecker
from polygraph.algorithms import selfcheckgpt_bertscore, selfcheckgpt_ngram, selfcheckgpt_prompt

# Uncertainty-based algorithms  
from polygraph.algorithms.uncertainty import SNNE, llm_uncertainty, semantic_entropy


def main():
    """Demonstrate usage of standardized PolyGraph algorithms."""
    
    # Example question and answer
    question = "What is the capital of France?"
    answer = "The capital of France is Paris, which is located in the northern part of the country."
    
    print("=" * 80)
    print("PolyGraph Standardized API Usage Examples")
    print("=" * 80)
    
    # Non-Uncertainty Algorithms
    print("\n🔍 NON-UNCERTAINTY ALGORITHMS")
    print("-" * 50)
    
    # ChainPoll
    print("\n1. ChainPoll")
    try:
        explanation, is_hallucination = chainpoll(
            question=question,
            answer=answer,
            n_samples=3,
            threshold=0.5
        )
        print(f"   Result: {'🚨 Hallucination detected' if is_hallucination else '✅ No hallucination'}")
        print(f"   Score: {explanation.get('score', 'N/A')}")
        print(f"   Details: {len(explanation.get('responses', []))} polling responses")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # ChatProtect
    print("\n2. ChatProtect")
    try:
        explanation, is_hallucination = chatprotect(
            question=question,
            answer=answer,
            threshold=0.3
        )
        print(f"   Result: {'🚨 Hallucination detected' if is_hallucination else '✅ No hallucination'}")
        print(f"   Inconsistencies: {explanation.get('inconsistency_count', 'N/A')}")
        print(f"   Ratio: {explanation.get('inconsistency_ratio', 'N/A'):.3f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # G-Eval
    print("\n3. G-Eval")
    try:
        explanation, is_hallucination = geval(
            question=question,
            answer=answer,
            threshold=0.6,
            n_samples=2
        )
        print(f"   Result: {'🚨 Low quality detected' if is_hallucination else '✅ Good quality'}")
        scores = explanation.get('scores', {})
        print(f"   Overall Score: {scores.get('Total', 'N/A')}")
        print(f"   Coherence: {scores.get('Coherence', 'N/A')}, Consistency: {scores.get('Consistency', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # RefChecker
    print("\n4. RefChecker")
    try:
        explanation, is_hallucination = refchecker(
            question=question,
            answer=answer,
            threshold=0.3
        )
        print(f"   Result: {'🚨 Contradiction detected' if is_hallucination else '✅ No contradiction'}")
        results = explanation.get('entailment_results', {})
        print(f"   Contradiction ratio: {explanation.get('contradiction_ratio', 'N/A'):.3f}")
        print(f"   Entailment: {results.get('Entailment', 'N/A'):.2f}, Neutral: {results.get('Neutral', 'N/A'):.2f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # SelfCheckGPT BertScore
    print("\n5. SelfCheckGPT-BertScore")
    try:
        explanation, is_hallucination = selfcheckgpt_bertscore(
            question=question,
            answer=answer,
            n_samples=3,
            threshold=0.5
        )
        print(f"   Result: {'🚨 Low similarity detected' if is_hallucination else '✅ Good similarity'}")
        print(f"   Similarity Score: {explanation.get('score', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Uncertainty-Based Algorithms
    print("\n\n🌡️  UNCERTAINTY-BASED ALGORITHMS")
    print("-" * 50)
    
    # LLM Uncertainty
    print("\n1. LLM Uncertainty")
    try:
        uncertainty_score, is_hallucination, explanation = llm_uncertainty(
            question=question,
            answer=answer,
            threshold=0.7,  # High confidence threshold
            prompt_strategy="cot"
        )
        print(f"   Result: {'🚨 Low confidence detected' if is_hallucination else '✅ High confidence'}")
        print(f"   Uncertainty: {uncertainty_score:.3f}")
        print(f"   Confidence: {explanation.get('confidence_fraction', 'N/A'):.3f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # SNNE
    print("\n2. SNNE (Soft Nearest Neighbor Entropy)")
    try:
        uncertainty_score, is_hallucination, explanation = SNNE(
            question=question,
            answer=answer,
            threshold=0.5,
            n_samples=3
        )
        print(f"   Result: {'🚨 High uncertainty detected' if is_hallucination else '✅ Low uncertainty'}")
        print(f"   Uncertainty Score: {uncertainty_score:.3f}")
        print(f"   Samples used: {explanation.get('n_samples', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Semantic Entropy
    print("\n3. Semantic Entropy")
    try:
        uncertainty_score, is_hallucination, explanation = semantic_entropy(
            question=question,
            answer=answer,
            threshold=0.5,
            n_samples=3,
            clustering_method="embedding"
        )
        print(f"   Result: {'🚨 High entropy detected' if is_hallucination else '✅ Low entropy'}")
        print(f"   Entropy Score: {uncertainty_score:.3f}")
        cluster_info = explanation.get('cluster_info', {})
        print(f"   Clusters: {cluster_info.get('n_clusters', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✨ All algorithms tested with standardized interface!")
    print("📊 Each algorithm returns:")
    print("   - Non-uncertainty: (explanation_dict, is_hallucination_bool)")
    print("   - Uncertainty: (uncertainty_score_float, is_hallucination_bool, explanation_dict)")
    print("🎯 Threshold parameters allow flexible decision boundaries")
    print("=" * 80)


if __name__ == "__main__":
    main()
