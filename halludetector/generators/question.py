import logging

PROMPT = """Please generate a question on the given text so that when searching on Google with the question, it's possible to get some relevant information on the topics addressed in the text. Note, you just need to give the final question without quotes in one line, and additional illustration should not be included.

For example:
Input text: The Lord of the Rings trilogy consists of The Fellowship of the Ring, The Two Towers, and The Return of the King.
Output: What are the three books in The Lord of the Rings trilogy?

Input text: %s
Output: """


logger = logging.getLogger(__name__)


class QuestionGenerator:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler

    def generate(self, paragraph):
        prompt = PROMPT % paragraph
        response = self.llm_handler.ask_llm(prompt)[0]
        logger.info(f'Question generated: {response}')
        return response
