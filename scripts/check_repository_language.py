"""Detect known Italian prose/label leftovers; preserve scientific payload languages.

This lexical check complements review; it is not a universal language classifier.
"""

import re
from pathlib import Path

from scripts.collaboration import visible_files


ROOT = Path(__file__).resolve().parents[1]
MARKERS = re.compile(r"\b(?:risultati|parole|immagini|maschere|confronto|nessuna|nessuno|questa|questo|questi|queste|epoche|riproduzione|quindi|sorgenti|esperimenti|verificare|aggiornamento|perché|soltanto|raggio|soglia|corrette|recuperate|eseguito|senza)\b|[àèìòù]", re.IGNORECASE)

# Exact synthetic Unicode inputs exercise byte encoding and edit reconstruction.
# Keep their source bytes; surrounding comments and other strings remain checked.
SCIENTIFIC_TEST_LITERALS = {
    "tests/recognition/test_encoder_family_router.py": ("'citt\u00e0'", "'\u00e0'"),
    "tests/recognition/test_grounded_word_recovery.py": ("'\u00e8!'",),
}


def find_leftovers(root):
    problems = []
    for name in visible_files(root):
        path = root / name
        if path.suffix not in {".md", ".py", ".sh", ".json", ".yaml", ".yml", ".txt", ".in"}:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if name == "scripts/check_repository_language.py" and line.startswith("MARKERS ="):
                continue  # Detector vocabulary is data, not untranslated prose.
            if name == "configs/approximate/clip_router_prompt_ablation_v2.json" and '"all_it"' in line:
                continue  # Original multilingual model input; changing it changes the experiment.
            if name == "diagnostics/project_storage_cleanup.py" and '"removed_empty_files"' in line:
                continue  # Historical literal filenames, not interface labels.
            for literal in SCIENTIFIC_TEST_LITERALS.get(name, ()):
                line = line.replace(literal, "")
            if path.suffix == ".md":
                line = re.sub(r"`[^`]*`", "", line)  # Scientific identifiers/payload examples.
            if MARKERS.search(line):
                problems.append((name, number))
    return problems


def main():
    problems = find_leftovers(ROOT)
    for name, number in problems:
        print("Review possible untranslated prose: {}:{}".format(name, number))
    print("English source check: {} (scientific payload exceptions retained)".format("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
