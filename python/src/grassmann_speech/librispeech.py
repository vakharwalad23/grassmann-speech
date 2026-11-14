import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.utils.data import Dataset

from grassmann_speech.features import speed_perturb

SPLITS = ("train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other")
SAMPLE_RATE = 16000


def speakers(root: Path) -> dict[str, str]:
    """Speaker id to sex from SPEAKERS.TXT ("ID | SEX | SUBSET | MINUTES | NAME", ";" comments)."""
    rows = (line.split("|") for line in (root / "SPEAKERS.TXT").read_text().splitlines() if line.strip() and not line.startswith(";"))
    return {r[0].strip(): r[1].strip() for r in rows}


def scan(root: Path, split: str) -> list[dict]:
    """One row per utterance of an extracted split: id, speaker, path relative to root, samples, reference text."""
    rows = []
    for trans in sorted((root / split).glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            path = trans.parent / f"{uid}.flac"
            rows.append({"id": uid, "speaker": uid.split("-")[0], "path": str(path.relative_to(root)), "samples": sf.info(path).frames,
                         "text": text})
    return rows


def sealed_manifest(root: Path, split: str, rows: list[dict]) -> dict:
    """Test-set manifest: per clip the sha256 of the flac and of its 16 kHz PCM16 samples, duration, speaker, reference, license."""
    sex = speakers(root)
    clips = []
    for row in rows:
        audio, rate = sf.read(root / row["path"], dtype="int16")
        assert rate == SAMPLE_RATE, (row["id"], rate)
        clips.append({"id": row["id"], "speaker": row["speaker"], "sex": sex.get(row["speaker"]), "path": row["path"],
                      "sha256_flac": hashlib.sha256((root / row["path"]).read_bytes()).hexdigest(),
                      "sha256_pcm16": hashlib.sha256(audio.tobytes()).hexdigest(), "duration_s": len(audio) / rate,
                      "reference": row["text"]})
    return {"corpus": f"librispeech-{split}", "source": f"OpenSLR SLR12 {split}.tar.gz (www.openslr.org/resources/12)",
            "license": "CC BY 4.0", "clips": clips}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def train_tokenizer(texts: list[str], vocab: int, path: Path) -> Tokenizer:
    """Unigram word pieces over the training transcripts (the recipe uses SentencePiece unigram); <blank> = 0 for CTC, <unk> = 1."""
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.train_from_iterator(texts, trainers.UnigramTrainer(vocab_size=vocab, special_tokens=["<blank>", "<unk>"], unk_token="<unk>"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def numel_batches(items: list[dict], bins: int, vocab: int) -> list[list[int]]:
    """ESPnet batch_type numel: sort by samples, close a batch once size x (samples + tokens x vocab) of its newest item exceeds bins."""
    batches, current = [], []
    for i in sorted(range(len(items)), key=lambda k: items[k]["samples"]):
        current.append(i)
        if len(current) * (items[i]["samples"] + len(items[i]["tokens"]) * vocab) > bins:
            batches.append(current)
            current = []
    return batches + ([current] if current else [])


class Utterances(Dataset):
    """Items carry id, path, speed factor, perturbed samples and token ids; audio is read and perturbed on load."""

    def __init__(self, items: list[dict], root: Path):
        self.items, self.root = items, root

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        item = self.items[i]
        wave, rate = sf.read(self.root / item["path"], dtype="float32")
        assert rate == SAMPLE_RATE, (item["id"], rate)
        return torch.from_numpy(speed_perturb(wave, item["factor"], rate)), torch.tensor(item["tokens"], dtype=torch.long), item["id"]


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, str]]) -> dict:
    waves, tokens, ids = zip(*batch, strict=True)
    return {"wave": torch.nn.utils.rnn.pad_sequence(waves, batch_first=True), "wave_lens": torch.tensor([len(w) for w in waves]),
            "tokens": torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True), "token_lens": torch.tensor([len(t) for t in tokens]),
            "ids": list(ids)}


def synthetic_corpus(root: Path, per_split: int, seed: int) -> None:
    """LibriSpeech layout with tone-per-word audio, for code-path checks without the 7 GB download."""
    rng, words = random.Random(seed), ["THE", "CAT", "SAT", "ON", "A", "MAT", "WITH", "ONE", "DOG", "AND", "TWO", "RED", "BIRDS"]
    speakers_txt = ["; ID | SEX | SUBSET | MINUTES | NAME"]
    for k, split in enumerate(SPLITS):
        for u in range(per_split):
            spk, chapter = str(100 * (k + 1) + u % 3), "10"
            text = " ".join(rng.choices(words, k=rng.randint(3, 8)))
            tones = [np.sin(2 * np.pi * (200 + 40 * words.index(w)) * np.arange(4000) / SAMPLE_RATE) for w in text.split()]
            folder = root / split / spk / chapter
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / f"{spk}-{chapter}-{u:04d}.flac", 0.3 * np.concatenate(tones), SAMPLE_RATE)
            with (folder / f"{spk}-{chapter}.trans.txt").open("a") as f:
                f.write(f"{spk}-{chapter}-{u:04d} {text}\n")
            speakers_txt.append(f"{spk} | {'F' if u % 2 else 'M'} | {split} | 1.0 | synthetic")
    (root / "SPEAKERS.TXT").write_text("\n".join(dict.fromkeys(speakers_txt)) + "\n")
