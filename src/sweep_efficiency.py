"""M3 + M4: encoding-efficiency sweeps reproducing SuperBPE claims C2-C6 (tokenizer-only).

Trains BPE / SuperBPE / naive-no-pretok tokenizers on a small corpus and measures bytes-per-token
on a held-out shard. Writes a ledger + JSON for plotting. No model training.
"""

import json
import os
import time

from bpe import train

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
RUNS = os.path.join(HERE, "..", "runs")
os.makedirs(RUNS, exist_ok=True)


def load_docs(path):
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def bytes_per_token(tok, heldout_docs):
    # Encode in fixed-size slices: bounds the O(L^2) encoder cost and is a fine bytes/token estimate.
    nb = nt = 0
    for d in heldout_docs:
        for i in range(0, len(d), 400):
            chunk = d[i : i + 400]
            nb += len(chunk.encode("utf-8"))
            nt += len(tok.encode(chunk))
    return nb / nt, nt


def main():
    train_docs = load_docs(os.path.join(DATA, "corpus_train.txt"))
    held_docs = load_docs(os.path.join(DATA, "corpus_heldout.txt"))[:150]
    print(f"train docs: {len(train_docs)}  heldout docs: {len(held_docs)}", flush=True)

    VOCABS = [1000, 2000, 4000, 8000, 16000, 32000]
    SBPE_FRAC = 0.4  # transition point as fraction of vocab (paper's best ~80k/200k)

    results = {"vocab_sweep": [], "transition_sweep": []}

    # ---- M3: vocab sweep. BPE vs SuperBPE at every vocab; naive (no-pretok) once at the top (C6). ----
    print("\n=== M3: vocab sweep ===", flush=True)
    for V in VOCABS:
        row = {"vocab": V}
        variants = [("bpe", V), ("superbpe", int(V * SBPE_FRAC))]
        if V == VOCABS[-1]:
            variants.append(("naive", 256))
        for name, t in variants:
            s = time.time()
            tok = train(train_docs, vocab_size=V, transition_point=t)
            bpt, _ = bytes_per_token(tok, held_docs)
            supers = sum(tok.crosses_boundary(x) for x in tok.vocab)
            row[name] = round(bpt, 3)
            row[f"{name}_supers"] = supers
            print(f"  V={V:>6} {name:>9}: {bpt:.3f} bytes/tok  supers={supers:>5}  ({time.time()-s:.1f}s)", flush=True)
        results["vocab_sweep"].append(row)

    # ---- M4: transition-point sweep at fixed vocab ----
    V = 16000
    print(f"\n=== M4: transition-point sweep at vocab={V} ===")
    for frac in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        t = max(256, int(V * frac)) if frac > 0 else 256
        s = time.time()
        tok = train(train_docs, vocab_size=V, transition_point=t)
        bpt, _ = bytes_per_token(tok, held_docs)
        label = "naive(t=0)" if frac == 0 else ("bpe(t=T)" if frac == 1.0 else f"t={frac:.1f}T")
        results["transition_sweep"].append({"frac": frac, "t": t, "bytes_per_token": round(bpt, 3), "label": label})
        print(f"  {label:>11} (t={t:>6}): {bpt:.3f} bytes/tok  ({time.time()-s:.1f}s)", flush=True)

    with open(os.path.join(RUNS, "encoding_efficiency.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.join(RUNS, 'encoding_efficiency.json')}")


if __name__ == "__main__":
    main()
