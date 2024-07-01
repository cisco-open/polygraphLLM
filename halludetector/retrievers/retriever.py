from .google_retriever import GoogleRetriever


class RetrieverHandler:
    def __init__(self, sentence_extractor):
        self.retriever = GoogleRetriever(sentence_extractor)

    def retrieve(self, *args, **kwargs):
        return self.retriever.retrieve(*args, **kwargs)
