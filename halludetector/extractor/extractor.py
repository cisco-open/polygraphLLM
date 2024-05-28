import spacy

from .gpt4_extractor import GPT4Extractor


class TripletsExtractorHandler:
    def __init__(self, llm_handler):
        self.extractor = GPT4Extractor(llm_handler)

    def extract(self, response, question=None, max_new_tokens=200):
        return self.extractor.extract_claim_triplets(response, question, max_new_tokens)


class SentenceExtractorHandler:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text):
        return [sent for sent in self.nlp(text).sents]
