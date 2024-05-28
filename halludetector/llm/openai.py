import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def ask_llm(self, prompt, n=1, temperature=0.5, max_new_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 400))):
        response = self.client.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-instruct"),
            prompt=prompt,
            max_tokens=max_new_tokens,
            n=n,
            temperature=temperature,

        )
        results = [r.text.strip() for r in response.choices]
        logger.info(f'Prompt responses: {results}')
        return results
