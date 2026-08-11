"""DTM data loading, option shuffling, normalization, and stratified splitting.
Self-contained: loads the DTM benchmark JSON (data/DTM_benchmark.json)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import dspy

CHOICE_LETTERS = "ABCD"
DEFAULT_DATASET = str(Path(__file__).resolve().parent.parent / "data" / "DTM_benchmark.json")
DEFAULT_RATIOS = (0.5, 0.25, 0.25)

# Gold answers for rows whose `answer` was nulled after the paper's runs; recovered by
# aligning the reconstructed split with the shipped traces (analysis/verify_split.py).
RECOVERED_ANSWERS = {299: "A", 314: "A"}


def load_raw(dataset: str = DEFAULT_DATASET) -> list[dict]:
    raw = json.load(open(dataset))
    for r in raw:
        a = str(r.get("answer", "")).strip().upper()
        if not (len(a) == 1 and a in CHOICE_LETTERS) and r.get("id") in RECOVERED_ANSWERS:
            r["answer"] = RECOVERED_ANSWERS[r["id"]]
    return raw


def normalize_row(row: dict, rng: random.Random, shuffle: bool = True) -> dict:
    """Normalize a raw DTM row. Rows with an invalid answer are kept (usable=False)
    and STILL consume the shuffle rng, so the split matches the original runs."""
    a = str(row.get("answer", "")).strip().upper()
    usable = len(a) == 1 and a in CHOICE_LETTERS
    opts = [row["option_A"], row["option_B"], row["option_C"], row["option_D"]]
    if shuffle:
        idx = list(range(4))
        rng.shuffle(idx)
        options = [opts[i] for i in idx]
        answer = CHOICE_LETTERS[idx.index(CHOICE_LETTERS.index(a))] if usable else None
    else:
        options, answer = opts, (a if usable else None)
    return {"question": row["question"], "options": options, "answer": answer,
            "subject": row.get("subject", "unknown"), "qid": row.get("id"),
            "usable": usable}


def format_options(options: list[str]) -> str:
    return "\n".join(f"{CHOICE_LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def to_example(record: dict) -> dspy.Example:
    return dspy.Example(
        question=record["question"], options=format_options(record["options"]),
        answer_letter=record["answer"], subject=record["subject"], qid=record.get("qid"),
    ).with_inputs("question", "options")


def stratified_split(records, seed=42, ratios=DEFAULT_RATIOS):
    """Split records into (train, dev, test), stratified by 'subject'."""
    rng = random.Random(seed)
    by_subject: dict[str, list[dict]] = {}
    for r in records:
        by_subject.setdefault(r["subject"], []).append(r)
    train, dev, test = [], [], []
    for subject in sorted(by_subject):
        items = list(by_subject[subject]); rng.shuffle(items)
        n = len(items); n_train = int(n*ratios[0]); n_dev = int(n*ratios[1])
        train += items[:n_train]; dev += items[n_train:n_train+n_dev]; test += items[n_train+n_dev:]
    return train, dev, test


def load_splits(dataset: str = DEFAULT_DATASET, seed: int = 42, shuffle: bool = True):
    raw = load_raw(dataset)
    rng = random.Random(seed)
    records = [normalize_row(r, rng, shuffle=shuffle) for r in raw]
    train, dev, test = stratified_split(records, seed=seed)
    bad_test = [r["qid"] for r in test if not r["usable"]]
    if bad_test:
        raise ValueError(f"unusable rows in TEST would break trace alignment: {bad_test}")
    dropped = [r["qid"] for part in (train, dev) for r in part if not r["usable"]]
    if dropped:
        print(f"[data] dropping {len(dropped)} unusable train/dev rows (qids: {sorted(dropped)})")
    keep = lambda part: [to_example(r) for r in part if r["usable"]]
    return keep(train), keep(dev), keep(test)


def cap_per_subject(examples, n):
    """Keep at most n examples per subject (n<=0 keeps all)."""
    if n <= 0:
        return examples
    by, out = {}, []
    for e in examples:
        if len(by.setdefault(e.subject, [])) < n:
            by[e.subject].append(e); out.append(e)
    return out


PUBLIC_CSV = str(Path(__file__).resolve().parent.parent / "data" / "DTM2019_public.csv")
SUBJECT_MAP = {"math": "matematika", "physics": "fizika", "history": "tarix",
               "ona_tili": "ona_tili"}


def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).lower()


def load_public(csv_path: str = PUBLIC_CSV, benchmark: str = DEFAULT_DATASET,
                seed: int = 2026) -> list[dict]:
    """DTM2019 public items (complement of the benchmark): subject-normalized,
    deduped against the benchmark, options shuffled deterministically."""
    import csv as _csv
    bench_qs = {_norm_text(r["question"]) for r in json.load(open(benchmark))}
    rng = random.Random(seed)
    out = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            a = (row.get("correct_answer") or "").strip().upper()
            subj = SUBJECT_MAP.get((row.get("subject") or "").strip())
            opts = [row.get("option_A"), row.get("option_B"),
                    row.get("option_C"), row.get("option_D")]
            if not (len(a) == 1 and a in CHOICE_LETTERS) or subj is None or not all(o and o.strip() for o in opts):
                continue
            if _norm_text(row["question"]) in bench_qs:
                continue
            idx = list(range(4))
            rng.shuffle(idx)
            sh = [opts[i] for i in idx]
            out.append({"question": row["question"], "options": sh,
                        "answer": CHOICE_LETTERS[idx.index(CHOICE_LETTERS.index(a))],
                        "subject": subj, "qid": f"pub{row['question_id']}",
                        "usable": True})
    return out


def replication_onatili() -> list:
    """Frozen E4 replication set: all deduped public ona_tili items."""
    return [to_example(r) for r in load_public() if r["subject"] == "ona_tili"]


def replication_all(cap_nonnative: int | None = None, seed: int = 2026,
                    native: str = "ona_tili") -> list:
    """Frozen replication set across ALL FOUR subjects (E9).

    E4 used the ona_tili slice only, because a within-subject protocol has no use for
    the others. A powered CROSS-subject test needs both strata: the native subject for
    the absolute McNemar, and the non-native subjects for the differential odds ratio.
    The public corpus supplies 393 / 504 / 727 / 404 usable items
    (ona_tili / tarix / matematika / fizika).

    `cap_nonnative` caps each NON-native subject to that many items, so a run can be
    sized to the GPU budget without ever thinning the native subject the primary
    endpoint is measured on. Capping is a deterministic shuffle under `seed`, taken
    once over the whole subject so it does not depend on how many subjects are kept."""
    rows = load_public()
    keep, rng = [], random.Random(seed)
    by_subject: dict[str, list] = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(r)
    for subject in sorted(by_subject):
        items = by_subject[subject]
        if subject != native and cap_nonnative is not None and len(items) > cap_nonnative:
            items = sorted(items, key=lambda r: r["qid"])
            rng.shuffle(items)
            items = items[:cap_nonnative]
        keep += items
    keep.sort(key=lambda r: (r["subject"], r["qid"]))
    return [to_example(r) for r in keep]
