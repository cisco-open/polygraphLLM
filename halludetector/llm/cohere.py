import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class CohereHandler:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("CohereForAI/c4ai-command-r-plus-4bit", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-plus-4bit")

    def ask_llm(self, prompt, n=1, temperature=0.4, max_new_tokens=100):
        results = []
        for x in range(n):
            logger.info(f'Asking: {prompt}')
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to('cuda')
            gen_tokens = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )
            gen_text = self.tokenizer.decode(gen_tokens[0])
            gen_text = gen_text.rsplit('<|CHATBOT_TOKEN|>', 1)[-1]
            gen_text = gen_text.replace('<|END_OF_TURN_TOKEN|>', '')
            logger.info(f'Cohere response: {gen_text}')
            results.append(gen_text)
        return results

    def process_response(self, text):
        tokens = ['<BOS_TOKEN>', '<EOS_TOKEN>']
        for token in tokens:
            text = text.replace(token, '')
        return text
