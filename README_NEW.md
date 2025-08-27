# PolyGraph: Standardized LLM Hallucination Detection

🚀 **A comprehensive, clean, and easy-to-use library for detecting hallucinations in Large Language Models**

## ✨ What's New in v1.0.0

- **🎯 Standardized API**: All algorithms now follow consistent interfaces
- **📚 Clean Organization**: Clear separation between uncertainty and non-uncertainty methods
- **🔧 Easy Integration**: Simple function calls with flexible parameters
- **📊 Rich Explanations**: Detailed output for transparency and debugging
- **⚙️ Flexible Thresholds**: Customizable decision boundaries for all algorithms

## 🏗️ Library Structure

```python
# Non-uncertainty algorithms (return explanation + binary decision)
from polygraph.algorithms import chainpoll, chatprotect, geval, refchecker, selfcheckgpt

# Uncertainty-based algorithms (return uncertainty score + binary decision + explanation) 
from polygraph.algorithms.uncertainty import SNNE, TESNNE, llm_uncertainty, semantic_entropy, kernel_uncertainty, p_true

# Utilities
from polygraph.utils import LLMHandler, TripletExtractor, Settings
```

## 🚀 Quick Start

### Installation

```bash
pip install polygraph
```

### Basic Usage

```python
from polygraph.algorithms import chainpoll
from polygraph.algorithms.uncertainty import SNNE

# Non-uncertainty algorithm
explanation, is_hallucination = chainpoll(
    question="What is the capital of France?",
    answer="Paris is the capital of France",
    threshold=0.5,
    n_samples=5
)

# Uncertainty-based algorithm  
uncertainty_score, is_hallucination, explanation = SNNE(
    question="Explain quantum computing",
    answer="Quantum computing uses quantum bits...",
    threshold=0.7,
    n_samples=5
)
```

## 📚 Algorithm Reference

### Non-Uncertainty Algorithms

All non-uncertainty algorithms return `(explanation_dict, is_hallucination_bool)`:

#### **ChainPoll**
```python
explanation, is_hallucination = chainpoll(
    question="Your question here",
    answer=None,  # Auto-generated if None
    threshold=0.5,  # Hallucination threshold (0.0-1.0)
    n_samples=5,    # Number of polling samples
    temperature=0.2 # Sampling temperature
)
```

#### **ChatProtect**
```python
explanation, is_hallucination = chatprotect(
    question="Your question here", 
    answer="Answer to analyze",
    threshold=0.3  # Inconsistency ratio threshold
)
```

#### **G-Eval**
```python
explanation, is_hallucination = geval(
    question="Your question here",
    answer="Answer to evaluate", 
    threshold=0.6,  # Quality threshold
    metrics=['coherence', 'consistency', 'fluency', 'relevance']
)
```

#### **RefChecker**
```python
explanation, is_hallucination = refchecker(
    question="Your question here",
    answer="Answer to fact-check",
    threshold=0.3  # Contradiction threshold  
)
```

#### **SelfCheckGPT Variants**
```python
# BertScore variant
explanation, is_hallucination = selfcheckgpt_bertscore(
    question="Your question here",
    answer="Answer to check",
    threshold=0.5,
    n_samples=5
)

# N-gram variant  
explanation, is_hallucination = selfcheckgpt_ngram(
    question="Your question here", 
    answer="Answer to check",
    threshold=0.5
)

# Prompt-based variant
explanation, is_hallucination = selfcheckgpt_prompt(
    question="Your question here",
    answer="Answer to check", 
    threshold=0.5
)
```

### Uncertainty-Based Algorithms

All uncertainty algorithms return `(uncertainty_score_float, is_hallucination_bool, explanation_dict)`:

#### **LLM Uncertainty**
```python
uncertainty, is_hallucination, explanation = llm_uncertainty(
    question="Your question here",
    answer="Answer to analyze",
    threshold=0.7,  # Confidence threshold (high = less hallucination tolerance)
    prompt_strategy="cot"  # "vanilla", "cot", "self-probing", "multi-step"
)
```

#### **SNNE (Soft Nearest Neighbor Entropy)**
```python
uncertainty, is_hallucination, explanation = SNNE(
    question="Your question here",
    answer="Answer to analyze", 
    threshold=0.5,      # Uncertainty threshold
    n_samples=5,        # Number of samples to generate
    temperature=0.8,    # Generation temperature
    variant='only_denom' # SNNE variant
)
```

#### **Semantic Entropy**
```python
uncertainty, is_hallucination, explanation = semantic_entropy(
    question="Your question here",
    answer="Answer to analyze",
    threshold=0.5,               # Entropy threshold  
    clustering_method="entailment", # "entailment" or "embedding"
    n_samples=5
)
```

#### **Kernel Uncertainty** 
```python
uncertainty, is_hallucination, explanation = kernel_uncertainty(
    question="Your question here",
    answer="Answer to analyze",
    threshold=0.5,           # Uncertainty threshold
    is_weighted=True,        # Use weighted edges in graph
    weight_strategy="manual" # "manual" or "deberta"
)
```

#### **P(True)**
```python
uncertainty, is_hallucination, explanation = p_true(
    question="Your question here", 
    answer="Answer to analyze",
    threshold=0.5,        # P(True) threshold
    use_few_shot=True,    # Use few-shot prompting
    n_samples=5
)
```

## 🎯 Key Features

### **Standardized Interfaces**
- Consistent function signatures across all algorithms
- Clear return types for easy integration
- Flexible parameter configuration

### **Rich Explanations**
Every algorithm returns detailed explanations:
```python
explanation = {
    'algorithm': 'chainpoll',
    'score': 0.3,
    'threshold': 0.5, 
    'n_samples': 5,
    'responses': [...],  # Raw LLM responses
    'votes': [...],      # Polling votes
    'answer': "...",     # Analyzed answer
    'question': "..."    # Input question
}
```

### **Flexible Thresholds**
- **Non-uncertainty algorithms**: Threshold determines when to flag as hallucination
- **Uncertainty algorithms**: Threshold determines uncertainty tolerance
- All thresholds are configurable (0.0-1.0 range)

### **Binary Decisions + Raw Scores**
- Get both binary hallucination decisions AND raw scores
- Easy integration into larger systems
- Transparency for analysis and debugging

## 🔧 Advanced Usage

### Custom Configuration
```python
# Use specific LLM provider
from polygraph.utils import LLMHandler

llm = LLMHandler(provider="openai")  # "openai", "cohere", "mistral"

# Custom settings
explanation, is_hallucination = chainpoll(
    question="...",
    answer="...",
    temperature=0.1,      # Low temperature for consistency  
    n_samples=10,         # More samples for better accuracy
    threshold=0.3         # Stricter threshold
)
```

### Batch Processing
```python
questions = ["Question 1", "Question 2", "Question 3"]
results = []

for question in questions:
    uncertainty, is_hallucination, explanation = SNNE(
        question=question,
        threshold=0.5
    )
    results.append({
        'question': question,
        'uncertainty': uncertainty,
        'is_hallucination': is_hallucination,
        'details': explanation
    })
```

## 🏛️ Legacy Support

The library maintains backward compatibility with the original class-based interface:

```python
# Legacy usage (still supported)
from polygraphLLM.detectors.chainpoll import ChainPoll

detector = ChainPoll()
score, answer, responses = detector.score(question, answer, settings=settings)

# New standardized usage (recommended)
from polygraph.algorithms import chainpoll

explanation, is_hallucination = chainpoll(question, answer, threshold=0.5)
```

## 📊 Comparison: Old vs New API

| Aspect | Old API | New API |
|--------|---------|---------|
| **Import** | `from polygraphLLM.detectors.chainpoll import ChainPoll` | `from polygraph.algorithms import chainpoll` |
| **Usage** | `detector = ChainPoll(); score, answer, responses = detector.score(...)` | `explanation, is_hallucination = chainpoll(...)` |
| **Returns** | Algorithm-specific formats | Standardized (explanation, bool) or (uncertainty, bool, explanation) |
| **Thresholds** | Manual interpretation of scores | Built-in threshold handling |
| **Consistency** | Different interfaces per algorithm | Unified interface across all algorithms |

## 🚀 Migration Guide

### From v0.x to v1.0

1. **Update imports**:
   ```python
   # Old
   from polygraphLLM.detectors.chainpoll import ChainPoll
   
   # New  
   from polygraph.algorithms import chainpoll
   ```

2. **Update function calls**:
   ```python
   # Old
   detector = ChainPoll()
   score, answer, responses = detector.score(question, answer, settings)
   is_hallucination = score > 0.5  # Manual threshold
   
   # New
   explanation, is_hallucination = chainpoll(question, answer, threshold=0.5)
   score = explanation['score']  # Access raw score if needed
   ```

3. **Update uncertainty algorithms**:
   ```python
   # Old
   from polygraphLLM.detectors.snne.snne import SNNE
   detector = SNNE()
   score, answer, samples = detector.score(question, answer)
   
   # New
   from polygraph.algorithms.uncertainty import SNNE
   uncertainty, is_hallucination, explanation = SNNE(question, answer, threshold=0.5)
   ```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## 📞 Support

- 📧 Email: apayani@cisco.com  
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Documentation: [Full Documentation](https://your-docs-site.com)

---

**PolyGraph v1.0** - Making LLM hallucination detection clean, standardized, and easy to use! 🎉
