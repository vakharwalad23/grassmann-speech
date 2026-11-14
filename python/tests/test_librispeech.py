from grassmann_speech.librispeech import (
    SPLITS,
    Utterances,
    collate,
    numel_batches,
    scan,
    sealed_manifest,
    synthetic_corpus,
    train_tokenizer,
)


def test_scan_and_sealed_manifest_on_a_synthetic_corpus(tmp_path):
    synthetic_corpus(tmp_path, 6, 0)
    rows = scan(tmp_path, SPLITS[3])
    assert len(rows) == 6 and rows[0]["samples"] > 0 and not rows[0]["path"].startswith("/")
    manifest = sealed_manifest(tmp_path, SPLITS[3], rows)
    clip = manifest["clips"][0]
    assert len(clip["sha256_flac"]) == 64 and clip["sex"] in ("F", "M") and clip["reference"] == rows[0]["text"]


def test_tokenizer_reserves_blank_and_round_trips(tmp_path):
    tok = train_tokenizer(["THE CAT SAT ON A MAT", "A DOG AND TWO RED BIRDS"] * 20, 40, tmp_path / "tok.json")
    assert tok.token_to_id("<blank>") == 0 and tok.token_to_id("<unk>") == 1
    assert tok.decode(tok.encode("THE DOG SAT").ids) == "THE DOG SAT"


def test_numel_batches_cover_every_item_once_and_respect_bins():
    items = [{"samples": 1000 * (i % 7 + 1), "tokens": [1] * (i % 5 + 1)} for i in range(50)]
    batches = numel_batches(items, 30_000, 100)
    assert sorted(i for b in batches for i in b) == list(range(50))
    cost = [it["samples"] + 100 * len(it["tokens"]) for it in items]
    for b in batches[:-1]:
        # Closed by its newest item, which the batch keeps; one item earlier it still fit
        assert len(b) * cost[b[-1]] > 30_000 and (len(b) == 1 or (len(b) - 1) * cost[b[-2]] <= 30_000)


def test_dataset_perturbs_and_collates(tmp_path):
    synthetic_corpus(tmp_path, 2, 0)
    rows = scan(tmp_path, SPLITS[0])
    items = [{**rows[0], "factor": 1.1, "tokens": [3, 4]}, {**rows[1], "factor": 1.0, "tokens": [5]}]
    batch = collate([Utterances(items, tmp_path)[i] for i in range(2)])
    assert abs(batch["wave_lens"][0].item() - rows[0]["samples"] / 1.1) <= 2
    assert batch["tokens"].tolist()[1] == [5, 0] and batch["token_lens"].tolist() == [2, 1]
