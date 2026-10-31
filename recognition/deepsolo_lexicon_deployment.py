"""GT-blind DeepSolo vocabulary postprocessor with explicit artifact identities."""

import hashlib
from pathlib import Path

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.deepsolo_word_f1 import recognition_confidence


class DeepSoloVocabularyPostprocessor:
    def __init__(self, profile):
        if (profile["mode"] != "approximate" or profile["claim_label"] != "approximate reproduction"
                or profile["lexicon_kind"] != "Generic"):
            raise ValueError("Explicit qualified approximate Generic profile required")
        self.profile = profile
        for key in ("vocabulary", "pairs"):
            p = Path(profile["lexicon_paths"][key])
            if hashlib.sha256(p.read_bytes()).hexdigest() != profile["lexicon_sha256"][key]:
                raise ValueError("Changed qualified lexicon: " + key)
        self.lexicon = DeepSoloLexicon.from_files(profile["lexicon_paths"]["vocabulary"],
            profile["lexicon_paths"]["pairs"], distance_exclusive=profile["distance_exclusive"])

    def apply(self, response):
        if not response["success"]:
            raise RuntimeError("Cannot postprocess failed DeepSolo inference")
        if response["artifact"]["checkpoint_sha256"] != self.profile["checkpoint_sha256"]:
            raise ValueError("DeepSolo checkpoint differs from qualified vocabulary profile")
        rows, raw = [], {}
        for d in response["detections"]:
            i = d["source_index"]
            if i in raw:
                raise ValueError("Duplicate DeepSolo source index")
            raw[i] = d
            if (d["score"] >= self.profile["neural"]["score_threshold"] and
                    recognition_confidence(d["raw"]["recognition_class_probabilities"]) >=
                    self.profile["neural"]["recognition_threshold"]):
                rows.append(dict(instance_id=i, text=d["text"], score=d["score"]))
        rows.sort(key=lambda p: (-p["score"], p["instance_id"]))
        corrected, decisions = self.lexicon.correct(rows, unmatched_policy=self.profile["unmatched_policy"])
        detections = [dict(raw[p["instance_id"]], text=p["text"]) for p in corrected]
        return dict(detections=detections, recognized_strings=[d["text"] for d in detections],
                    lexicon_decisions=decisions, expert="DeepSolo plus Generic vocabulary",
                    checkpoint_sha256=self.profile["checkpoint_sha256"],
                    lexicon_sha256=self.profile["lexicon_sha256"],
                    mode="approximate", claim_label="approximate reproduction")
