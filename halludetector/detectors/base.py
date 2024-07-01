import os
import logging
logger = logging.getLogger(__name__)


class Detector:

    def __init__(self):
        from halludetector.config import init_config
        init_config(f'{os.path.dirname(os.path.realpath(__file__))}/../../config.json')

        from halludetector.config import (
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

    def ask_llm(self, *args, **kwargs):
        return self.llm_handler.ask_llm(*args, **kwargs)

    def extract_triplets(self, *args, **kwargs):
        return self.triplets_extractor.extract(*args, **kwargs)

    def extract_sentences(self, *args, **kwargs):
        return self.sentence_extractor.extract(*args, **kwargs)

    def generate_question(self, *args, **kwargs):
        return self.question_generator.generate(*args, **kwargs)

    def retrieve(self, *args, **kwargs):
        return self.retriever.retrieve(*args, **kwargs)

    def check(self, *args, **kwargs):
        return self.checker.check(*args, **kwargs)

    def similarity_bertscore(self, sentences, sampled_passages):
        try:
            return self.bertscorer.predict(
                sentences=sentences,  # list of sentences
                sampled_passages=sampled_passages,  # list of sampled passages
            )
        except Exception as e:
            logger.error(f'Bertscore failed due to {e}')

    def similarity_ngram(self, sentences, passage, sampled_passages):
        return self.ngramscorer.predict(
            passage=passage,
            sentences=sentences,  # list of sentences
            sampled_passages=sampled_passages,  # list of sampled passages
        )

    def score(self, question, answer=None, samples=None, summary=None):
        raise NotImplementedError
