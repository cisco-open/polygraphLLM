from .base import Detector


class RefChecker(Detector):
    def __init__(self):
        super().__init__()

    def score(self, question=None, answer=None, samples=None):
        if not question:
            question = self.generate_question(answer)

        if not answer:
            answer = self.ask_llm(question.strip())[0]

        triplets = self.extract_triplets(answer, question, max_new_tokens=200)
        reference = self.retrieve([question])

        results = [
            self.check(t, reference, question=question)
            for t in triplets
        ]
        agg_results = self.soft_agg(results)
        for k, v in agg_results.items():
            agg_results[k] = float("{:.2f}".format(v))
        return agg_results, answer, None

    def soft_agg(self, results):
        """Aggregate results by taking the ratio of each category."""
        if not results:
            return {
                "Entailment": 0.0,
                "Neutral": 0.0,
                "Contradiction": 0.0,
                "Abstain": 1.0,
            }
        total = len(results)
        agg = {
            "Entailment": 0.0,
            "Neutral": 0.0,
            "Contradiction": 0.0,
            "Abstain": 0.0,
        }
        for result in results:
            agg[result] += 1.0
        for key in agg:
            agg[key] /= total
        print(results)
        return agg
