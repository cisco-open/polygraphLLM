Hallucination detector
======================

This application creates building blocks for generic approaches for hallucination detection in Large language models.


Installation
------------

Use a conda environment and install the followings.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   pip install -e .
   pip install -r requirements.txt

   python3 -m spacy download en_core_web_sm

Export envs for openai and google wrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   export OPENAI_API_KEY=
   export SERPER_API_KEY=

Usage
-----

as server
^^^^^^^^^

::

   python3 server.py

Go to http://127.0.0.1:5000 and use the app.

as library
^^^^^^^^^^

Instantiate the Base Detectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    from halludetector.detectors.base import Detector
    detector = Detector()

Requesting results from the LLM
^^^^^^^^^

::

    responses = detector.ask_llm(
    'Which Lactobacililus casei strain does not have the cholera toxin subunit'
    ' A1 (CTA1) on the surface?',
    n=2, # the number of responses
    temperature=0.5, # temperature give to the LLM
    max_new_tokens=100 # number of tokens for response
    )
    print(responses)

Extract triplets from a text. (subject, predicate, object)
^^^^^^^^^

::

    triplets = detector.extract_triplets(
    'Which Lactobacililus casei strain does not have the cholera toxin subunit'
    ' A1 (CTA1) on the surface?',
    )
    print(triplets)

Extract sentences from a text.
^^^^^^^^^

::

    sentences = detector.extract_sentences(
    'There is no specific Lactobacillus casei strain that is known to not have the cholera toxin subunit A1 (CTA1) on its surface.'
    'However, some strains may have a lower expression of CTA1 or may not have the gene for CTA1 at all. '
    'The presence or absence of CTA1 on the surface of Lactobacillus casei strains can vary depending on the specific strain and its genetic makeup.',
    )
    print(sentences)

Generate question from a given text.
^^^^^^^^^^^

::

    question = detector.generate_question(
    'There is no specific Lactobacillus casei strain that is known to not have the cholera toxin subunit A1 (CTA1) on its surface.'
    'However, some strains may have a lower expression of CTA1 or may not have the gene for CTA1 at all. '
    'The presence or absence of CTA1 on the surface of Lactobacillus casei strains can vary depending on the specific strain and its genetic makeup.',
    )
    print(question)

Retrieve information from the internet for a list of inputs
^^^^^^^^^^^^^^^^^^

::

    results = detector.retrieve(
    ['What factors can affect the presence or absence of the cholera toxin subunit A1 on the surface of Lactobacillus casei strains?'],
    )

    print(results)


Check the hallucination scores using the triplets.
^^^^^^^^^^^^^^^^^

::

    question = 'What factors can affect the presence or absence of the cholera toxin subunit A1 on the surface of Lactobacillus casei strains?'
    answer = detector.ask_llm(question, n=1)[0]
    triplets = detector.extract_triplets(answer)
    reference = detector.retrieve([question])
    results = [
    detector.check(t, reference, answer, question=question)
    for t in triplets
    ]
    print(results)


Check the similarity of texts using bert score.
^^^^^^^^^^^^^^^^^^

::

    question = 'What factors can affect the presence or absence of the cholera toxin subunit A1 on the surface of Lactobacillus casei strains?'
    answers = detector.ask_llm(question, n=5)
    first_answer = answers[0]
    sentences = detector.extract_sentences(first_answer)
    sentences = [s.text for s in sentences]
    sampled_passages = answers[1:]
    results = detector.similarity_bertscore(sentences, sampled_passages)
    scores = float("{:.2f}".format(sum(results)/len(results)))
    print(scores)


Check the similarity of texts using nGram model.
^^^^^^^^^^^^^^^^^

::

    passage = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Nation."
    sentences = detector.extract_sentences(passage)
    sentences = [s.text for s in sentences]

    sample1 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Country."
    sample2 = "Michael Alan Weiner (born January 13, 1960) is a Canadian radio host. He works at The New York Times."
    sample3 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He obtained his PhD from MIT."

    results = detector.similarity_ngram(sentences, passage, [sample1, sample2, sample3])
    scores = float("{:.2f}".format(results['doc_level']['avg_neg_logprob']))

    print(scores)


Building blocks
---------------

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


Creating a new detector
^^^^^^^^^^^^
In the detectors folder create a new file for your detector.
Inherit the Detector Base class and implement the score method.

::

    from halludetector.detectors.base import Detector
    class CustomDetector(Detector):

        def score(self, question, answer=None, samples=None, summary=None):
            # do your logic.
            return score, answer, responses

Creating a new LLM Handler
^^^^^^^^^^

In the llm folder create a new file with your handler.
See an example below.

::

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


Implementing a new Benchmark
^^^^^^^^^^
In the datasets folder add a new file with your benchmark.

Inherit the **Parser** class and implement the **display** function as in this example.

You must return the **data** and the **columns** you want to display in a specific order.

To use it with the UI you must add your newly implemented benchmark to the **BENCHMARKS** list in the **__init__.py** file of the same folder.

::

    class DollyParser(Parser):
        display_name = 'Databricks Dolly'
        _id = 'databricks-dolly'

        def __init__(self):
            self.dataset = load_dataset('databricks/databricks-dolly-15k')
            self.dataset = self.dataset['train']

        def display(self):
            results = []

            for element in self.dataset:
                results.append(
                    {
                        'question': element['instruction'],
                        'context': element['context'],
                        'answer': element['response'],
                        'category': element['category']
                    }
                )
            return {
                'data': results,
                'columns': ['question', 'context', 'answer', 'category']
            }


References
^^^^^^^^^^
**G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment**

https://arxiv.org/abs/2303.16634

**Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models**

https://arxiv.org/abs/2303.08896

**RefChecker for Fine-grained Hallucination Detection**

https://github.com/amazon-science/RefChecker

**Chainpoll: A high efficacy method for LLM hallucination detection**

https://arxiv.org/abs/2310.18344





