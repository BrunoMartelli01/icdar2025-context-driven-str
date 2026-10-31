"""Pinned-source DeepSolo lexicon correction; never receives evaluation targets."""

from functools import lru_cache
from pathlib import Path

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


class DeepSoloLexicon:
    """Order-sensitive nearest word and pair-list mapping from the official source."""

    def __init__(self, vocabulary, pairs, *, distance_exclusive):
        self.words = tuple(word.upper() for word in vocabulary)
        self.pairs = dict(pairs)
        self.distance_exclusive = distance_exclusive
        if not self.words or any(word not in self.pairs for word in self.words):
            raise ValueError("Every ordered lexicon entry requires a canonical pair")
        if distance_exclusive != 1.5:
            raise ValueError("This qualified source recipe requires distance < 1.5")

    @classmethod
    def from_files(cls, vocabulary_path, pairs_path, *, distance_exclusive):
        words = [line.strip() for line in Path(vocabulary_path).read_text().splitlines()]
        pairs = {}
        for line in Path(pairs_path).read_text().splitlines():
            line = line.strip()
            word = line.split(" ")[0].upper()
            pairs[word] = line[len(word)+1:]
        return cls(words, pairs, distance_exclusive=distance_exclusive)

    @lru_cache(maxsize=None)
    def nearest(self, text):
        # Distances are integer. A cutoff of one is exactly equivalent for the
        # source's accept/reject decision; first minimal-distance ties survive.
        match = process.extractOne(text.upper(), self.words, scorer=Levenshtein.distance,
                                   processor=None, score_cutoff=1)
        if match is None:
            return None
        word, distance, index = match
        return dict(text=self.pairs[word], lexicon_word=word, distance=int(distance),
                    lexicon_index=index)

    def correct(self, predictions, *, unmatched_policy):
        if unmatched_policy not in ("source_reject", "keep_unmatched"):
            raise ValueError("Explicit unmatched-word policy required")
        selected, decisions = [], []
        for p in predictions:
            match = self.nearest(p["text"])
            accepted = match is not None or unmatched_policy == "keep_unmatched"
            text = match["text"] if match is not None else p["text"]
            if accepted:
                selected.append(dict(p, text=text))
            decisions.append(dict(instance_id=p["instance_id"], original_text=p["text"],
                                  accepted=accepted, corrected_text=text if accepted else None,
                                  match=match, unmatched_policy=unmatched_policy))
        return selected, decisions
