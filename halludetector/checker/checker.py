from .gpt4_checker import GPT4Checker


class CheckerHandler:
    def __init__(self, sentence_extractor, llm_handler):
        self.checker = GPT4Checker(sentence_extractor, llm_handler)

    def check(self, *args, **kwargs):
        return self.checker.check(*args, **kwargs)
