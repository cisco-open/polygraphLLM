<p align="center"><img src="logo.png" alt="polygraphLLM logo" width="400"/></p>

**PolygraphLLM** creates building blocks for generic approaches for hallucination detection in **Large Language Models (LLMs)**.

In the context of LLMs, **hallucination** refers to the generation of text that includes information or details that are not supported by the input or context provided to the model. Hallucinations occur when the model produces text that is incorrect, irrelevant, or not grounded in reality based on the input it receives.

**PolygraphLLM** is intended to help in the detection of hallucinations.

## Table of Contents

- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Usage](#usage)
        - [Playground](#playground)
        - [Library](#library)
- [Building blocks](#building-blocks)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Installation

#### Use a conda environment and install the followings


    pip install -e .
    pip install -r requirements.txt
    python3 -m spacy download en_core_web_sm
 

#### Export envs for openai and google wrapper


    export OPENAI_API_KEY=
    export SERPER_API_KEY=


### Usage

#### Playground


    cd playground
    npm i
    npm run dev

Go to http://localhost:3000/ and use the app.



#### Library

##### Instantiate the Base Detectors


    from src.detectors.base import Detector
    detector = Detector()

##### Requesting results from the LLM


    responses = detector.ask_llm(
    'Which Lactobacililus casei strain does not have the cholera toxin subunit'
    ' A1 (CTA1) on the surface?',
    n=2, # the number of responses
    temperature=0.5, # temperature give to the LLM
    max_new_tokens=100 # number of tokens for response
    )
    print(responses)

##### Extract triplets from a text. (subject, predicate, object)


    triplets = detector.extract_triplets(
    'Which Lactobacililus casei strain does not have the cholera toxin subunit'
    ' A1 (CTA1) on the surface?',
    )
    print(triplets)

##### Extract sentences from a text.


    sentences = detector.extract_sentences(
    'There is no specific Lactobacillus casei strain that is known to not have the cholera toxin subunit A1 (CTA1) on its surface.'
    'However, some strains may have a lower expression of CTA1 or may not have the gene for CTA1 at all. '
    'The presence or absence of CTA1 on the surface of Lactobacillus casei strains can vary depending on the specific strain and its genetic makeup.',
    )
    print(sentences)

##### Generate question from a given text.


    question = detector.generate_question(
    'There is no specific Lactobacillus casei strain that is known to not have the cholera toxin subunit A1 (CTA1) on its surface.'
    'However, some strains may have a lower expression of CTA1 or may not have the gene for CTA1 at all. '
    'The presence or absence of CTA1 on the surface of Lactobacillus casei strains can vary depending on the specific strain and its genetic makeup.',
    )
    print(question)

##### Retrieve information from the internet for a list of inputs


    results = detector.retrieve(
    ['What factors can affect the presence or absence of the cholera toxin subunit A1 on the surface of Lactobacillus casei strains?'],
    )

    print(results)


##### Check the hallucination scores using the triplets.


    question = 'What factors can affect the presence or absence of the cholera toxin subunit A1 on the surface of Lactobacillus casei strains?'
    answer = detector.ask_llm(question, n=1)[0]
    triplets = detector.extract_triplets(answer)
    reference = detector.retrieve([question])
    results = [
    detector.check(t, reference, answer, question=question)
    for t in triplets
    ]
    print(results)


##### Check the similarity of texts using bert score.


    question = 'What factors can affect the presence or absence of the cholera toxin subunit A1 on the surface of Lactobacillus casei strains?'
    answers = detector.ask_llm(question, n=5)
    first_answer = answers[0]
    sentences = detector.extract_sentences(first_answer)
    sentences = [s.text for s in sentences]
    sampled_passages = answers[1:]
    results = detector.similarity_bertscore(sentences, sampled_passages)
    scores = float("{:.2f}".format(sum(results)/len(results)))
    print(scores)


##### Check the similarity of texts using nGram model.


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

This project implements generic approaches for hallucination detection.

The ``Detector`` base class implements the building blocks to detect
hallucinations and score them.

``ask_llm`` - method to request N responses from an LLM via a prompt

``extract_triplets`` - method to extract subject, predicate, object from
a text.

``extract_sentences`` - method to split a text into sentences using
spacy

``generate_question`` - method to generate a question from a text

``retrieve`` - method to retrieve information from google via the serper
api

``check`` - method to check if the claims contain hallucinations

``similarity_bertscore`` - method to check the similarity between texts
via bertscore

``similarity_ngram`` - method to check the similarity between texts via
ngram model

You can implement any custom detector and combine all the available
methods from above.


#### Creating a new detector

In the detectors folder create a new file for your detector.
Inherit the Detector Base class and implement the score method.


    from src.detectors.base import Detector
    class CustomDetector(Detector):

        def score(self, question, answer=None, samples=None, summary=None):
            # do your logic.
            return score, answer, responses

#### Creating a new LLM Handler

In the llm folder create a new file with your handler.
See an example below.


    class CustomHandler:
        def __init__(self):
            self.model = AutoModelForCausalLM.from_pretrained("your-model", device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained("your-model")

        def ask_llm(self, prompt, n=1, temperature=0, max_new_tokens=400):
            model_inputs = self.tokenizer([prompt] * n, return_tensors="pt")
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=True)
            results = [r for r in self.tokenizer.batch_decode(generated_ids)]
            logger.info(f'Prompt responses: {results}')
            return results

In **config.py** in **init_building_blocks** update the **llm_handler** to your new handler.

Instead of

``llm_handler = OpenAIHandler()``

use

``llm_handler = CustomHandler()``


#### Implementing a new Benchmark

In the benchmarks folder add a new file with your benchmark.

Inherit the **DatasetParser** class and implement the **display** function as in this example.

To use it with the UI you must add your newly implemented benchmark to the **BENCHMARKS** list in the **__init__.py** file of the same folder.


    class DatabricksDollyParser(DatasetParser):
        display_name = 'Databricks dolly'
        id = 'databricks-dolly'

        def download_data(self):
            self.dataset = load_dataset('databricks/databricks-dolly-15k')
            self.dataset = self.dataset['train']

        def display(self, offset, limit):
            result = []
            for item in self.dataset:
                result.append({
                    "id": uuid.uuid4(),
                    "question": item["instruction"],
                    "answer": item["response"],
                    "context": item["context"],
                })
            return self.apply_offset_limit(result, offset, limit)



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

Any contributions you make are greatly appreciated. For detailed contributing instructions, please check out [Contributing Guidelines](/CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE).