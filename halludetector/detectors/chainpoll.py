import logging
import os

from .base import Detector
from ..prompts.default import DEFAULT_CHAINPOLL_PROMPT

logger = logging.getLogger(__name__)


class ChainPoll(Detector):
    id = 'chainpoll'
    display_name = 'Chain Poll'

    def __init__(self):
        super().__init__()
        self.sample_number = int(os.getenv("CHAINPOLL_SAMPLING_NUMBER", 5))
        try:
            with open(f'{os.path.dirname(os.path.realpath(__file__))}/../prompts/chainpoll.txt', 'r') as pf:
                self.prompt = pf.read()
        except:
            self.prompt = DEFAULT_CHAINPOLL_PROMPT

    def check_hallucinations(self, completion, question):
        text = self.prompt.format(completion=completion, question=question)
        responses = self.ask_llm(text, n=self.sample_number, temperature=0.2)
        logger.info(f'Hallucination check response: {responses}')
        return [response.lower().startswith("yes") for response in responses], responses

    def score(self, question, answer=None, samples=None, summary=None):
        if not answer:
            answer = self.ask_llm(question.strip())[0]
        hallucinations, responses = self.check_hallucinations(answer.strip(), question.strip())

        score = hallucinations.count(True) / len(hallucinations)
        return score, answer, responses