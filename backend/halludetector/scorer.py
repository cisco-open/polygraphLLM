import logging

from halludetector.detectors import (
    ChainPoll, SelfCheckGPTBertScore,
    SelfCheckGPTNGram, SelfCheckGPTMQAG,
    RefChecker, GEval,
    SelfCheckGPTPrompt
)

scorer_mapping = {
    'self_check_gpt_bertscore': SelfCheckGPTBertScore,
    'self_check_gpt_ngram': SelfCheckGPTNGram,
    'self_check_gpt_prompt': SelfCheckGPTPrompt,
    'refchecker': RefChecker,
    'g_eval': GEval,
    'chainpoll': ChainPoll,
}

logger = logging.getLogger(__name__)


def calculate_score(method, question, answer=None, samples=None, summary=None):
    scorer_class = scorer_mapping.get(method)
    if scorer_class:
        scorer = scorer_class()
    return scorer.score(question, answer, samples, summary)
