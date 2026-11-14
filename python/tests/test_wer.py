import json
import subprocess
import sys

from grassmann_speech.wer import char_errors, paired_bootstrap, per_utterance, totals, word_edits


def test_word_edits_count_each_error_type():
    assert word_edits(["A", "B", "C", "D"], ["A", "X", "C", "D", "E"]) == {"errors": 2, "sub": 1, "del": 0, "ins": 1}
    assert word_edits(["A", "B", "C"], [])["del"] == 3
    assert word_edits([], ["A", "B"])["ins"] == 2


def test_char_errors_is_levenshtein():
    assert char_errors("KITTEN", "SITTING") == 3
    assert char_errors("", "AB") == 2


def test_totals_normalize_case_and_count_missing_hypotheses_as_empty():
    rows = per_utterance({"1-1-1": "THE CAT", "2-1-1": "A DOG"}, {"1-1-1": "the cat"})
    t = totals(rows)
    assert (t["errors"], t["words"], t["wer"]) == (2, 4, 50.0)


def test_bootstrap_calls_identical_systems_a_tie_and_a_clear_gap_not():
    refs = {f"{s}-1-{u}": "ONE TWO THREE FOUR" for s in range(20) for u in range(5)}
    good = per_utterance(refs, refs)
    bad = per_utterance(refs, {k: "ONE TWO" for k in refs})
    assert paired_bootstrap(good, good)["tie"]
    gap = paired_bootstrap(bad, good)
    assert gap["delta_wer"] == 50.0 and not gap["tie"]


def test_cli_reads_kaldi_text(tmp_path):
    (tmp_path / "ref").write_text("1-1-1 THE CAT SAT\n")
    (tmp_path / "hyp").write_text("1-1-1 THE CAT\n")
    out = subprocess.run([sys.executable, "-m", "grassmann_speech.wer", tmp_path / "ref", tmp_path / "hyp"], capture_output=True, text=True,
                         check=True)
    assert json.loads(out.stdout)["del"] == 1
