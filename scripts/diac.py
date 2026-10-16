#!/usr/bin/env python3
"""Vietnamese diacritics restoration with CANINE-s: prepare data, fine-tune, evaluate.

Task: input = Vietnamese text with every diacritic removed (ASCII letters), output = the original text.
The model predicts ONE label per input character (the combining marks that belong on it), so it can never
change a base letter: the output is rebuilt by composing each input character with its predicted marks.

Subcommands:  prepare | train | eval      (see run.sh for the full pipeline)
"""
import argparse, collections, hashlib, json, os, random, re, sys, time, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
SEED = 1234
DATASET = "wikimedia/wikipedia"
DATASET_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
DATASET_FILE = "20231101.vi/train-00000-of-00004.parquet"
MODEL = "google/canine-s"
MODEL_REV = "75d6d0b3f4d0bffed0ca7ddd0b73949977d4ca02"
N_TRAIN, N_VAL, N_TEST = 400_000, 2_000, 5_000
STROKE = "̵"   # we encode d-with-stroke as base 'd' + this pseudo-mark (it is not a real combining mark in NFD)
CLS, SEP, PAD = 0xE000, 0xE001, 0

# ---------- character <-> (base, marks) ----------
def split_char(c):
    """'ế' -> ('e', '̂́');  'đ' -> ('d', STROKE);  plain char -> (c, '')."""
    if c in "đĐ":
        return ("d" if c == "đ" else "D"), STROKE
    d = unicodedata.normalize("NFD", c)
    return d[0], d[1:]

def join_char(base, marks):
    if marks == STROKE:
        return {"d": "đ", "D": "Đ"}.get(base, base)
    r = unicodedata.normalize("NFC", base + marks)
    return r if len(r) == 1 else base   # impossible combination (e.g. a tone mark on 'b'): keep the plain letter

def strip_text(s):
    return "".join(split_char(c)[0] for c in s)

VI_LETTERS = set("abcdefghijklmnopqrstuvwxyz")  # Vietnamese uses no f j w z natively, but names/loanwords do; we keep ASCII letters only

def ok_sentence(s):
    """Keep only clean Vietnamese-looking sentences: NFC, letters are ASCII or Vietnamese, round-trips exactly."""
    if unicodedata.normalize("NFC", s) != s or not (40 <= len(s) <= 200):
        return False
    n_dia = 0
    for c in s:
        if c.isalpha():
            b, m = split_char(c)
            if b.lower() not in VI_LETTERS or join_char(b, m) != c:
                return False
            n_dia += bool(m)
        elif not (c.isdigit() or c in " .,;:!?()-\"'%/") :
            return False
    letters = sum(c.isalpha() for c in s)
    return letters >= 0.7 * len(s) and n_dia >= 5

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_of(title):
    h = int(hashlib.sha1(title.encode()).hexdigest(), 16) % 100
    return "test" if h < 5 else "val" if h < 8 else "train"   # by ARTICLE title: no article is in two splits

# ---------- prepare ----------
def prepare():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    path = hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset", revision=DATASET_REV, cache_dir=str(WORK / "hf"))
    pf = pq.ParquetFile(path)
    rng = random.Random(SEED)
    pools = {"train": [], "val": [], "test": []}
    n_art = collections.Counter()
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["id", "title", "text"]).to_pydict()
        for i, title, text in zip(t["id"], t["title"], t["text"]):
            sp = split_of(title)
            n_art[sp] += 1
            for para in text.split("\n"):
                for s in SENT_SPLIT.split(para.strip()):
                    if ok_sentence(s):
                        pools[sp].append((s, title, i))
    stats = {"articles": dict(n_art), "candidate_sentences": {k: len(v) for k, v in pools.items()}}
    # exact-duplicate sentences: one copy kept; anything in val/test is removed from train (no leakage by repetition)
    held = {s for k in ("val", "test") for s, _, _ in pools[k]}
    seen, train = set(), []
    for s, t_, i in pools["train"]:
        if s in held or s in seen:
            continue
        seen.add(s); train.append((s, t_, i))
    stats["train_removed_as_duplicate_or_in_heldout"] = len(pools["train"]) - len(train)
    out = {"train": train, "val": dedup(pools["val"]), "test": dedup(pools["test"])}
    out["val"] = [x for x in out["val"] if x[0] not in {s for s, _, _ in out["test"]}]
    for k, n in (("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST)):
        rng.shuffle(out[k]); out[k] = out[k][:n]
    assert not ({s for s, _, _ in out["train"]} & {s for s, _, _ in out["test"]}), "train/test sentence overlap"
    assert not ({t for _, t, _ in out["train"]} & {t for _, t, _ in out["test"]}), "train/test article overlap"
    for k, v in out.items():
        with open(WORK / f"{k}.jsonl", "w") as f:
            for s, t_, i in v:
                f.write(json.dumps({"text": s, "title": t_, "id": i}, ensure_ascii=False) + "\n")
    stats["used_sentences"] = {k: len(v) for k, v in out.items()}
    stats["used_articles"] = {k: len({t for _, t, _ in v}) for k, v in out.items()}
    stats["source"] = {"dataset": DATASET, "revision": DATASET_REV, "file": DATASET_FILE}
    stats["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(stats, open(ROOT / "results" / "data_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))

def dedup(lst):
    seen, o = set(), []
    for x in lst:
        if x[0] not in seen:
            seen.add(x[0]); o.append(x)
    return o

def load(split):
    return [json.loads(l) for l in open(WORK / f"{split}.jsonl")]

# ---------- labels / encoding ----------
def build_labels(train_texts):
    cnt = collections.Counter(split_char(c)[1] for s in train_texts for c in s)
    marks = [""] + sorted(m for m in cnt if m)
    return marks

def encode_batch(texts, label2id=None):
    import torch
    L = max(len(s) for s in texts) + 2
    ids = torch.zeros(len(texts), L, dtype=torch.long)
    att = torch.zeros(len(texts), L, dtype=torch.long)
    lab = torch.full((len(texts), L), -100, dtype=torch.long)
    for r, s in enumerate(texts):
        base = [split_char(c)[0] for c in s]
        row = [CLS] + [ord(b) for b in base] + [SEP]
        ids[r, :len(row)] = torch.tensor(row); att[r, :len(row)] = 1
        if label2id is not None:
            lab[r, 1:len(s) + 1] = torch.tensor([label2id[split_char(c)[1]] for c in s])
    return ids, att, lab

def make_model(n_labels):
    import torch
    from transformers import CanineForTokenClassification
    torch.manual_seed(SEED)
    return CanineForTokenClassification.from_pretrained(MODEL, revision=MODEL_REV, num_labels=n_labels,
                                                        cache_dir=str(WORK / "hf"))

@__import__("torch").no_grad()
def predict(model, texts, marks, bs=128):
    import torch
    model.eval(); dev = next(model.parameters()).device
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]; batch = [texts[i] for i in idx]
        ids, att, _ = encode_batch(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits
        pred = logits.argmax(-1).cpu()
        for j, i in enumerate(idx):
            out[i] = [marks[p] for p in pred[j, 1:len(batch[j]) + 1].tolist()]
    return out

def restore(plain, marks_seq):
    return "".join(join_char(b, m) if b.isalpha() else b for b, m in zip(plain, marks_seq))

# ---------- metrics ----------
def syllables(s):
    return [w.strip(".,;:!?()-\"'%/") for w in s.split()]

def score(golds, preds, train_vocab=None):
    """letters: every alphabetic character; syllables: whitespace tokens with punctuation stripped, only those
    containing a letter; unseen: syllables whose ASCII-folded form never occurs in the training set."""
    L = Lc = S = Sc = Sent = U = Uc = 0
    per_sent = []
    for g, p in zip(golds, preds):
        assert strip_text(g) == strip_text(p), "base letters changed"
        lc = sum(a == b for a, b in zip(g, p) if a.isalpha()); ln = sum(a.isalpha() for a in g)
        sc = sn = 0
        for gs, ps in zip(syllables(g), syllables(p)):
            if not any(ch.isalpha() for ch in gs):
                continue
            sn += 1; sc += gs == ps
            if train_vocab is not None and strip_text(gs).lower() not in train_vocab:
                U += 1; Uc += gs == ps
        L += ln; Lc += lc; S += sn; Sc += sc; Sent += g == p
        per_sent.append((lc, ln, sc, sn, g == p))
    r = {"letters": L, "letter_accuracy": Lc / L, "letter_error_rate": 1 - Lc / L,
         "syllables": S, "syllable_accuracy": Sc / S, "syllable_error_rate": 1 - Sc / S,
         "sentences": len(golds), "sentence_accuracy": Sent / len(golds)}
    if train_vocab is not None:
        r["unseen_syllables"] = U; r["unseen_syllable_accuracy"] = Uc / U if U else None
    return r, per_sent

def bootstrap_ci(per_sent, idx_c, idx_n, n=1000, seed=SEED):
    import numpy as np
    a = np.array([[x[idx_c], x[idx_n]] for x in per_sent], dtype=float)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        s = a[rng.integers(0, len(a), len(a))].sum(0); vals.append(s[0] / s[1])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

# ---------- train ----------
def train(args):
    import torch
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    tr, va = load("train"), load("val")
    marks = build_labels([x["text"] for x in tr]); label2id = {m: i for i, m in enumerate(marks)}
    json.dump(marks, open(WORK / "labels.json", "w"))
    model = make_model(len(marks)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    texts = [x["text"] for x in tr]
    steps_per_epoch = len(texts) // args.bs; total = steps_per_epoch * args.epochs; warm = int(0.05 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); step = 0
    log = open(ROOT / "results" / "train_log.jsonl", "w")
    rng = random.Random(SEED)
    for ep in range(args.epochs):
        order = list(range(len(texts))); rng.shuffle(order)
        for k in range(steps_per_epoch):
            batch = [texts[i] for i in order[k * args.bs:(k + 1) * args.bs]]
            ids, att, lab = encode_batch(batch, label2id)
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 100 == 0:
                rec = {"step": step, "of": total, "loss": round(loss.item(), 4), "elapsed_s": round(time.time() - t0, 1)}
                if step % 1000 == 0 or step == total:
                    v = va[:500]; p = predict(model, [strip_text(x["text"]) for x in v], marks)
                    pr = [restore(strip_text(x["text"]), m) for x, m in zip(v, p)]
                    rec["val500_syllable_accuracy"] = round(score([x["text"] for x in v], pr)[0]["syllable_accuracy"], 4)
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    wall = time.time() - t0
    model.save_pretrained(WORK / "model")
    json.dump({"train_seconds": wall, "steps": step, "batch_size": args.bs, "epochs": args.epochs, "lr": args.lr,
               "train_sentences": len(texts), "peak_vram_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
               "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
               "gpu": torch.cuda.get_device_name(0), "n_labels": len(marks)},
              open(ROOT / "results" / "train_stats.json", "w"), indent=1)

# ---------- eval ----------
def evaluate():
    import torch
    from transformers import CanineForTokenClassification
    te = load("test"); tr = load("train")
    golds = [x["text"] for x in te]; plain = [strip_text(g) for g in golds]
    marks = json.load(open(WORK / "labels.json"))
    vocab = {strip_text(s).lower() for x in tr for s in syllables(x["text"])}
    # baseline 1: copy the input (no diacritics at all)
    # baseline 2: unigram lookup, most frequent diacritised form of each ASCII-folded syllable in train
    cnt = collections.defaultdict(collections.Counter)
    for x in tr:
        for s in syllables(x["text"]):
            cnt[strip_text(s)][s] += 1
    best = {k: v.most_common(1)[0][0] for k, v in cnt.items()}
    def lookup(p):
        out = []
        for tok in re.split(r"(\s+)", p):
            core = tok.strip(".,;:!?()-\"'%/")
            out.append(tok.replace(core, best.get(core, core), 1) if core and not tok.isspace() else tok)
        return "".join(out)
    res = {"test_sentences": len(golds)}
    preds = {"copy_input": plain, "unigram_lookup": [lookup(p) for p in plain]}
    # before: the same CANINE-s with its classification head untrained (random, seeded) = "before fine-tuning"
    t0 = time.time()
    m0 = make_model(len(marks)).cuda()
    preds["canine_before_finetune"] = [restore(p, ms) for p, ms in zip(plain, predict(m0, plain, marks))]
    del m0
    m1 = CanineForTokenClassification.from_pretrained(WORK / "model").cuda()
    torch.cuda.reset_peak_memory_stats(); t1 = time.time()
    mk = predict(m1, plain, marks); torch.cuda.synchronize()
    inf_s = time.time() - t1
    preds["canine_finetuned"] = [restore(p, ms) for p, ms in zip(plain, mk)]
    res["inference"] = {"seconds_for_test_set": inf_s, "sentences_per_second": len(golds) / inf_s,
                        "peak_vram_allocated_gb": torch.cuda.max_memory_allocated() / 2**30}
    res["systems"] = {}
    for name, p in preds.items():
        r, per = score(golds, p, vocab)
        r["syllable_accuracy_ci95"] = bootstrap_ci(per, 2, 3); r["letter_accuracy_ci95"] = bootstrap_ci(per, 0, 1)
        res["systems"][name] = r
    # base letters can never change: asserted inside score(); record it
    res["base_letters_changed"] = 0
    json.dump(res, open(ROOT / "results" / "eval.json", "w"), indent=1)
    # examples
    ex = []
    for i, g in enumerate(golds):
        a, b = preds["canine_finetuned"][i], preds["unigram_lookup"][i]
        ex.append((i, g, plain[i], b, a))
    rng = random.Random(SEED); rng.shuffle(ex)
    def pick(cond, n): return [e for e in ex if cond(e)][:n]
    sel = {"model_right_baseline_wrong": pick(lambda e: e[4] == e[1] and e[3] != e[1], 5),
           "both_right": pick(lambda e: e[4] == e[1] and e[3] == e[1], 2),
           "model_wrong": pick(lambda e: e[4] != e[1], 8)}
    out = {k: [{"title": te[e[0]]["title"], "gold": e[1], "input": e[2], "unigram_lookup": e[3], "finetuned": e[4]} for e in v]
           for k, v in sel.items()}
    json.dump(out, open(ROOT / "results" / "examples.json", "w"), ensure_ascii=False, indent=1)
    # error breakdown: most frequent wrong syllable pairs
    errs = collections.Counter()
    for g, p in zip(golds, preds["canine_finetuned"]):
        for gs, ps in zip(syllables(g), syllables(p)):
            if gs != ps and any(c.isalpha() for c in gs): errs[(gs, ps)] += 1
    json.dump([{"gold": a, "predicted": b, "count": n} for (a, b), n in errs.most_common(25)],
              open(ROOT / "results" / "top_errors.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: {x: v[x] for x in ("letter_accuracy", "syllable_accuracy", "sentence_accuracy")} for k, v in res["systems"].items()}, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prepare", "train", "eval"])
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args(); WORK.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    {"prepare": prepare, "train": lambda: train(a), "eval": evaluate}[a.cmd]()
