import logging
import os
import spacy
import numpy as np
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG

from .base import Detector
from ..prompts.default import DEFAULT_SELFCHECK_WITH_PROMPT_PROMPT

logger = logging.getLogger(__name__)


class SelfCheckGPTBertScore(Detector):

    def __init__(self):
        super().__init__()
        self.sample_number = int(os.getenv("BERT_SCORE_SAMPLING_NUMBER", 3))

    def score(self, question, answer=None, samples=None, summary=None):
        if not answer:
            answer = self.ask_llm(question)[0]

        sentences = self.extract_sentences(answer)
        sentences = [s.text for s in sentences]
        if not samples:
            samples = self.ask_llm(question, n=self.sample_number, temperature=0.4)

        scores = self.similarity_bertscore(
            sentences=sentences,  # list of sentences
            sampled_passages=samples,  # list of sampled passages
        )
        if hasattr(scores, '__iter__'):
            scores = float("{:.2f}".format(sum(scores)/len(scores)))
        else:
            scores = 'FAILED'
        return scores, answer, samples


class SelfCheckGPTNGram(Detector):

    def __init__(self):
        super().__init__()
        self.sample_number = int(os.getenv("NGRAM_SAMPLING_NUMBER", 3))

    def score(self, question, answer=None, samples=None, summary=None):
        if not answer:
            answer = self.ask_llm(question)[0]

        sentences = self.extract_sentences(answer)
        sentences = [s.text for s in sentences]
        if not samples:
            samples = self.ask_llm(question, n=self.sample_number, temperature=0.4)

        scores = self.similarity_ngram(
            sentences=sentences,
            passage=answer,
            sampled_passages=samples
        )

        scores = float("{:.2f}".format(scores['doc_level']['avg_neg_logprob']))
        return scores, answer, samples


class SelfCheckGPTPrompt(Detector):

    def __init__(self):
        super().__init__()
        self.sample_number = int(os.getenv("PROMPT_SAMPLING_NUMBER", 3))
        self.prompt_template = DEFAULT_SELFCHECK_WITH_PROMPT_PROMPT
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}

    def score(self, question, answer=None, samples=None, summary=None):

        if not answer:
            answer = self.ask_llm(question)[0]

        if not samples:
            samples = self.ask_llm(question, n=self.sample_number, temperature=0.4)

        sentences = self.extract_sentences(answer)
        sentences = [s.text for s in sentences]
        scores = np.zeros((len(sentences), len(samples)))
        for sent_i, sentence in enumerate(sentences):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(samples):
                sample = sample.strip()
                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                generate_text = self.ask_llm(prompt)[0]
                generate_text = generate_text.replace(prompt, "")
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(axis=-1)
        return sum(scores_per_sentence)/len(scores_per_sentence), answer, samples

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        # tested on Llama-2-chat (7B, 13B) --- this code has 100% coverage on wikibio gpt3 generated data
        # however it may not work with other datasets, or LLMs
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]


class SelfCheckGPTMQAG(Detector):

    def __init__(self):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = SelfCheckMQAG(device=device)
        self.sample_number = 3

    def score(self, question, answer=None, samples=None, summary=None):
        samples = []
        if not answer:
            answer = self.ask_llm(question)[0]

        nlp = spacy.load("en_core_web_sm")
        sentences = [sent.text.strip() for sent in nlp(answer).sents]

        for _ in range(self.sample_number):
            samples.append(self.ask_llm(question)[0])

        scores = self.scorer.predict(
            sentences=sentences,  # list of sentences
            passage=answer,  # passage (before sentence-split)
            sampled_passages=samples,  # list of sampled passages
            num_questions_per_sent=5,  # number of questions to be drawn
            scoring_method='bayes_with_alpha',  # options = 'counting', 'bayes', 'bayes_with_alpha'
            beta1=0.8, beta2=0.8,
        )
        scores = float("{:.2f}".format(sum(scores)/len(scores)))
        return scores, answer, samples
