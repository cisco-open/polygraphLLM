import logging

from halludetector.detectors import (
    ChainPoll, SelfCheckGPTBertScore,
    SelfCheckGPTNGram, SelfCheckGPTMQAG,
    RefChecker, GEval,
    SelfCheckGPTPrompt
)

scorer_mapping = {
    'Self-Check GPT Bert Score': SelfCheckGPTBertScore,
    'Self-Check GPT NGram': SelfCheckGPTNGram,
    'Self-Check GPT Prompt': SelfCheckGPTPrompt,
    'RefChecker': RefChecker,
    'G-Eval': GEval,
    'Chain Poll': ChainPoll,
}

logger = logging.getLogger(__name__)


def calculate_score(method, question, answer=None, samples=None, summary=None):
    scorer_class = scorer_mapping.get(method)
    if scorer_class:
        scorer = scorer_class()
    return scorer.score(question, answer, samples, summary)
