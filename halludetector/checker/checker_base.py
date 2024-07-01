from typing import List, Union



def merge_ret(ret):
    """Merge results from multiple paragraphs"""
    if "Entailment" in ret:
        return "Entailment"
    if "Contradiction" in ret:
        return "Contradiction"
    return "Neutral"


def merge_multi_psg_ret(ret):
    """Merge results from multiple passages
    TODO: consider possible cases where the results are inconsistent.
    """
    if "Entailment" in ret:
        return "Entailment"
    if "Contradiction" in ret:
        return "Contradiction"
    return "Neutral"


class CheckerBase:
    def __init__(self, sentence_extractor) -> None:
        self.label_entailment = 'Entailment'
        self.label_neutral = 'Neutral'
        self.label_contradiction = 'Contradiction'
        self.labels = ["Entailment", "Neutral", "Contradiction"]
        self.sentence_extractor = sentence_extractor

    def check(
        self, 
        claim: str, 
        reference: Union[str, List], 
        response: str = None,
        question: str = None,
        max_reference_segment_length: int = 200, 
    ):
        ret = []
        if isinstance(reference, str):
            reference = [reference]
        for psg in reference:
            if max_reference_segment_length > 0:
                segments = self.split_text(psg, max_reference_segment_length)
            else:
                segments = [psg]
            psg_ret = self._check(
                claims=[claim] * len(segments),
                references=segments,
                response=response,
                question=question,
            )
            ret.append(merge_ret(psg_ret))
        return merge_multi_psg_ret(ret)

    def _check(
        self,
        claims: List,
        references: List,
        response: str,
        question: str = None
    ):
        raise NotImplementedError

    def split_text(self, text, segment_len=200):
        """Split text into segments according to sentence boundaries."""
        segments, seg = [], []
        sents = [[token.text for token in sent] for sent in self.sentence_extractor.extract(text)]
        for sent in sents:
            if len(seg) + len(sent) > segment_len:
                segments.append(" ".join(seg))
                seg = sent
                # single sentence longer than segment_len
                if len(seg) > segment_len:
                    # split into chunks of segment_len
                    seg = [
                        " ".join(seg[i:i + segment_len])
                        for i in range(0, len(seg), segment_len)
                    ]
                    segments.extend(seg)
                    seg = []
            else:
                seg.extend(sent)
        if seg:
            segments.append(" ".join(seg))
        return segments
