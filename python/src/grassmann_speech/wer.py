"""The one scorer for every model and corpus: normalization, WER and CER with error counts, paired bootstrap over speakers."""

import argparse
import json
import random
from pathlib import Path

# Bump on any change to normalize(); a change rescores everything
NORMALIZATION = 1


def normalize(text: str) -> list[str]:
    """Upper case, split on whitespace; LibriSpeech references are upper case without punctuation already."""
    return text.upper().split()


def word_edits(ref: list[str], hyp: list[str]) -> dict[str, int]:
    """Levenshtein alignment with substitution, deletion and insertion counts; ties prefer substitution."""
    rows = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        new = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            sub, dele, ins = rows[j - 1], rows[j], new[j - 1]
            options = (
                (sub[0] + (ref[i - 1] != hyp[j - 1]), sub[1] + (ref[i - 1] != hyp[j - 1]), sub[2], sub[3]),
                (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
            )
            new.append(min(options, key=lambda o: o[0]))
        rows = new
    errors, s, d, i = rows[-1]
    return {"errors": errors, "sub": s, "del": d, "ins": i}


def char_errors(ref: str, hyp: str) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j - 1] + (r != h), prev[j] + 1, cur[j - 1] + 1))
        prev = cur
    return prev[-1]


def per_utterance(refs: dict[str, str], hyps: dict[str, str]) -> dict[str, dict[str, int]]:
    """Every reference is scored; a missing hypothesis counts as empty."""
    out = {}
    for uid, ref_text in refs.items():
        ref, hyp = normalize(ref_text), normalize(hyps.get(uid, ""))
        out[uid] = {**word_edits(ref, hyp), "words": len(ref), "char_errors": char_errors(" ".join(ref), " ".join(hyp)),
                    "chars": len(" ".join(ref))}
    return out


def totals(rows: dict[str, dict[str, int]]) -> dict:
    sums = {k: sum(r[k] for r in rows.values()) for k in ("errors", "sub", "del", "ins", "words", "char_errors", "chars")}
    return {**sums, "wer": 100 * sums["errors"] / sums["words"], "cer": 100 * sums["char_errors"] / sums["chars"],
            "utterances": len(rows), "normalization": NORMALIZATION}


def speaker(uid: str) -> str:
    """LibriSpeech ids are speaker-chapter-utterance."""
    return uid.split("-")[0]


def paired_bootstrap(a: dict[str, dict[str, int]], b: dict[str, dict[str, int]], draws: int = 1000, seed: int = 0) -> dict:
    """WER(a) - WER(b) in points with a 95% interval from resampling speakers with replacement; an interval holding 0 is a tie."""
    groups: dict[str, list[str]] = {}
    for uid in a:
        groups.setdefault(speaker(uid), []).append(uid)
    speakers, rng = sorted(groups), random.Random(seed)

    def delta(picked: list[str]) -> float:
        uids = [u for s in picked for u in groups[s]]
        words = sum(a[u]["words"] for u in uids)
        return 100 * (sum(a[u]["errors"] for u in uids) - sum(b[u]["errors"] for u in uids)) / words

    deltas = sorted(delta(rng.choices(speakers, k=len(speakers))) for _ in range(draws))
    low, high = deltas[int(0.025 * draws)], deltas[int(0.975 * draws) - 1]
    return {"delta_wer": delta(speakers), "ci95": [low, high], "tie": low <= 0 <= high, "speakers": len(speakers), "draws": draws}


def read_kaldi_text(path: Path) -> dict[str, str]:
    """Kaldi-style "utterance-id words..." lines, as ESPnet writes them."""
    pairs = (line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip())
    return {p[0]: p[1] if len(p) > 1 else "" for p in pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Kaldi-style hypothesis files against a reference; JSON to stdout")
    parser.add_argument("ref", type=Path)
    parser.add_argument("hyp", type=Path)
    parser.add_argument("--against", type=Path, help="second hypothesis file: paired bootstrap of hyp minus this")
    args = parser.parse_args()
    refs = read_kaldi_text(args.ref)
    rows = per_utterance(refs, read_kaldi_text(args.hyp))
    report = totals(rows)
    if args.against is not None:
        report["paired"] = paired_bootstrap(rows, per_utterance(refs, read_kaldi_text(args.against)))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
