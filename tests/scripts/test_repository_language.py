import subprocess

from scripts.check_repository_language import find_leftovers


def test_language_check_reports_prose_but_preserves_scientific_prompt(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/example.md").write_text("# " + "Risul" + "tati\n")
    prompt = tmp_path / "configs/approximate/clip_router_prompt_ablation_v2.json"
    prompt.parent.mkdir(parents=True)
    prompt.write_text('{"all_it": "' + 'tutto il testo presente nell\'immagine ' + chr(232) + ': {text}' + '"}')
    assert find_leftovers(tmp_path) == [("docs/example.md", 1)]


def test_english_and_scientific_non_english_identifier_pass(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "README.md").write_text("# Results\n\nHistorical identifier: `con" + "fronto`.\n")
    assert find_leftovers(tmp_path) == []


def test_exact_unicode_fixtures_preserve_surrounding_language_checks(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    tests = tmp_path / "tests/recognition"
    tests.mkdir(parents=True)
    encoder = tests / "test_encoder_family_router.py"
    encoder.write_text("values = ['citt\u00e0', '\u00e0' * 130]\n", encoding="utf-8")
    edits = tests / "test_grounded_word_recovery.py"
    edits.write_text("pair = ('\u00e9?', '\u00e8!')\n", encoding="utf-8")
    assert find_leftovers(tmp_path) == []
    encoder.write_text(encoder.read_text(encoding="utf-8") + "# Risul" + "tati\n")
    edits.write_text(edits.read_text(encoding="utf-8") + "label = 'ques" + "ta'\n")
    assert find_leftovers(tmp_path) == [
        ("tests/recognition/test_encoder_family_router.py", 2),
        ("tests/recognition/test_grounded_word_recovery.py", 2),
    ]


def test_unicode_exception_does_not_apply_to_other_files(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "example.py").write_text("label = 'citt\u00e0'\n", encoding="utf-8")
    assert find_leftovers(tmp_path) == [("example.py", 1)]
