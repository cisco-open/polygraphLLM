<p align="center"><img src="https://raw.githubusercontent.com/cisco-open/polygraphLLM/refs/heads/main/logo.png" alt="polygraphLLM logo" width="400"/></p>

**PolygraphLLM** creates building blocks for generic approaches for hallucination detection in **Large Language Models (LLMs)**.

In the context of LLMs, **hallucination** refers to the generation of text that includes information or details that are not supported by the input or context provided to the model. Hallucinations occur when the model produces text that is incorrect, irrelevant, or not grounded in reality based on the input it receives.

**PolygraphLLM** is intended to help in the detection of hallucinations.


## Installation

    pip install polygraphLLM


The source code is currently hosted on GitHub at: https://github.com/cisco-open/polygraphLLM


## Export envs and install a small SpaCy model

    export OPENAI_API_KEY=
    export SERPER_API_KEY=
    python3 -m spacy download en_core_web_sm

## Usage

### Basic Usage with Base Detector

    from polygraphLLM import Detector
    detector = Detector()

### Using Specific Detection Algorithms

You can import and use specific hallucination detection algorithms:

    # Import individual detectors
    from polygraphLLM.algorithms import ChainPoll, RefChecker, GEval, ChatProtect
    from polygraphLLM.algorithms import SelfCheckGPTBertScore, SelfCheckGPTNGram
    
    # Initialize a specific detector
    detector = ChainPoll()
    
    # Detect hallucinations with threshold
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    is_hallucinated, score, answer, additional_data = detector.detect_hallucination(
        question, answer, threshold=0.5
    )
    print(f"Is hallucinated: {is_hallucinated}, Score: {score}")

### Quick Start Examples

    # Using ChainPoll for consistency checking
    from polygraphLLM.algorithms import ChainPoll
    detector = ChainPoll()
    question = "What is the largest mammal?"
    answer = "The largest mammal is the blue whale."
    is_hallucinated, score = detector.detect_hallucination(question, answer)
    
    # Using SelfCheckGPT for sampling-based detection
    from polygraphLLM.algorithms import SelfCheckGPTBertScore
    self_check = SelfCheckGPTBertScore()
    is_hallucinated, score = self_check.detect_hallucination(question, answer)

#### Requesting results from the LLM


    responses = detector.ask_llm(
    'Which Lactobacililus casei strain does not have the cholera toxin subunit A1 (CTA1) on the surface?',
    n=2, # the number of responses
    temperature=0.5, # temperature give to the LLM
    max_new_tokens=100 # number of tokens for response
    )
    print(responses)

#### Extract triplets from a text. (subject, predicate, object)


    triplets = detector.extract_triplets(
    'Which Lactobacililus casei strain does not have the cholera toxin subunit A1 (CTA1) on the surface?,
    )
    print(triplets)

#### Extract sentences from a text.


    sentences = detector.extract_sentences(
    'There is no specific Lactobacillus casei strain that is known to not have the cholera toxin subunit A1 (CTA1) on its surface.'
    'However, some strains may have a lower expression of CTA1 or may not have the gene for CTA1 at all. '
    'The presence or absence of CTA1 on the surface of Lactobacillus casei strains can vary depending on the specific strain and its genetic makeup.',
    )
    print(sentences)

#### Generate question from a given text.


    question = detector.generate_question(
    'There is no specific Lactobacillus casei strain that is known to not have the cholera toxin subunit A1 (CTA1) on its surface.'
    'However, some strains may have a lower expression of CTA1 or may not have the gene for CTA1 at all. '
    'The presence or absence of CTA1 on the surface of Lactobacillus casei strains can vary depending on the specific strain and its genetic makeup.',
    )
    print(question)

#### Retrieve information from the internet for a list of inputs


    results = detector.retrieve(
    ['What factors can affect the presence or absence of the cholera toxin subunit A1 on the surface of Lactobacillus casei strains?'],
    )

    print(results)


#### Check the hallucination scores using the triplets.


    question = 'What factors can affect the presence or absence of the cholera toxin subunit A1 on the surface of Lactobacillus casei strains?'
    answer = detector.ask_llm(question, n=1)[0]
    triplets = detector.extract_triplets(answer)
    reference = detector.retrieve([question])
    results = [
    detector.check(t, reference, answer, question=question)
    for t in triplets
    ]
    print(results)


#### Check the similarity of texts using bert score.


    question = 'What factors can affect the presence or absence of the cholera toxin subunit A1 on the surface of Lactobacillus casei strains?'
    answers = detector.ask_llm(question, n=5)
    first_answer = answers[0]
    sentences = detector.extract_sentences(first_answer)
    sentences = [s.text for s in sentences]
    sampled_passages = answers[1:]
    results = detector.similarity_bertscore(sentences, sampled_passages)
    scores = float("{:.2f}".format(sum(results)/len(results)))
    print(scores)


#### Check the similarity of texts using nGram model.


    passage = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Nation."
    sentences = detector.extract_sentences(passage)
    sentences = [s.text for s in sentences]

    sample1 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Country."
    sample2 = "Michael Alan Weiner (born January 13, 1960) is a Canadian radio host. He works at The New York Times."
    sample3 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He obtained his PhD from MIT."

    results = detector.similarity_ngram(sentences, passage, [sample1, sample2, sample3])
    scores = float("{:.2f}".format(results['doc_level']['avg_neg_logprob']))

    print(scores)


## Building blocks

This project implements generic approaches for hallucination detection through multiple specialized algorithms.

### Core Methods

Each detector implements these key methods:

``detect_hallucination(question, answer, threshold)`` - main detection method returning boolean result and confidence score

``score(question, answer, samples, summary, settings)`` - returns numerical hallucination score

``ask_llm(prompt, n, temperature, max_tokens)`` - request N responses from an LLM

### Utility Methods

The base ``Detector`` class also provides building blocks for custom implementations:

``extract_triplets`` - extract (subject, predicate, object) triplets from text

``extract_sentences`` - split text into sentences using spacy

``generate_question`` - generate questions from given text

``retrieve`` - retrieve information from Google via the Serper API

``check`` - verify claims against reference information

``similarity_bertscore`` - measure text similarity using BERTScore

``similarity_ngram`` - measure text similarity using n-gram models

You can implement custom detectors by inheriting from the base class and combining these methods.


## References

**G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment**

https://arxiv.org/abs/2303.16634

**Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models**

https://arxiv.org/abs/2303.08896

**RefChecker for Fine-grained Hallucination Detection**

https://github.com/amazon-science/RefChecker

**Chainpoll: A high efficacy method for LLM hallucination detection**

https://arxiv.org/abs/2310.18344

**Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs**

https://openreview.net/pdf?id=gjeQKFxFpZ

**Self-contradictory hallucinations of LLMs: Evaluation, detection and mitigation**

https://arxiv.org/pdf/2305.15852


## Contributing

Any contributions you make are greatly appreciated. For detailed contributing instructions, please check out [Contributing Guidelines](https://github.com/cisco-open/polygraphLLM/blob/main/CONTRIBUTING.md).

## License

[Apache License 2.0](https://github.com/cisco-open/polygraphLLM/blob/main/LICENSE).