import json
import os

from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore, SelfCheckNgram

from halludetector.llm.openai import OpenAIHandler
from halludetector.llm.mistral import MistralHandler
from halludetector.extractor.extractor import TripletsExtractorHandler, SentenceExtractorHandler
from halludetector.generators.question import QuestionGenerator
from halludetector.retrievers.retriever import RetrieverHandler
from halludetector.checker.checker import CheckerHandler

llm_handler = None
triplets_extractor = None
sentence_extractor = None
question_generator = None
retriever = None
checker = None
bertscorer = None
ngramscorer = None


def init_config(file, force=False):
    try:
        with open(file, 'r') as cfgfile:
            config = json.loads(cfgfile.read())
            for c in config:
                key = c['name']
                value = c['value']
                if force or not os.getenv(key):
                    os.environ[key] = str(value)
    except Exception:
        print('No config file. Loading variables from environment.')
    init_building_blocks()


def init_building_blocks(force=False):
    global llm_handler, triplets_extractor, sentence_extractor, question_generator, \
        retriever, checker, bertscorer, ngramscorer

    if llm_handler is None or force:
        llm_handler = OpenAIHandler()
        triplets_extractor = TripletsExtractorHandler(llm_handler)
        sentence_extractor = SentenceExtractorHandler()
        question_generator = QuestionGenerator(llm_handler)
        retriever = RetrieverHandler(sentence_extractor)
        checker = CheckerHandler(sentence_extractor, llm_handler)
        bertscorer = SelfCheckBERTScore(rescale_with_baseline=True)
        ngramscorer = SelfCheckNgram(n=int(os.getenv("NGRAM_N", 1)))
