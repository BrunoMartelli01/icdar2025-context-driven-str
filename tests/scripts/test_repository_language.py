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
