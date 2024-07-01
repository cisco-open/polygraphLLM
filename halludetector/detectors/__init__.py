from .chainpoll import ChainPoll
from .selfcheckgpt import SelfCheckGPTBertScore, SelfCheckGPTNGram, SelfCheckGPTMQAG, SelfCheckGPTPrompt
from .refchecker import RefChecker
from .geval import GEval

DETECTORS = [
    ChainPoll, 
    SelfCheckGPTBertScore, 
    SelfCheckGPTNGram,
    SelfCheckGPTPrompt,
    RefChecker,
    GEval
]

def get_detectors_display_names():
    return [{"id": detector.id, "display_name": detector.display_name} for detector in DETECTORS]

def get_detector(id):
    for detector in DETECTORS:
        if detector.id == id:
            return detector()
