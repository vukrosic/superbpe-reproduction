"""Fetch a small English corpus for tokenizer training + a held-out shard (A1/A8).

Streams cosmopedia-v2 (same source the harness already uses) and writes plain-text shards capped by
byte size. One document per line (newlines within docs stripped) so we can split train/held-out cleanly.
"""

import os
import sys

from datasets import load_dataset

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)

TRAIN_MB = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
HELDOUT_MB = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

ds = load_dataset(
    "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True
)

train_cap, held_cap = int(TRAIN_MB * 1e6), int(HELDOUT_MB * 1e6)
train_path = os.path.join(OUT, "corpus_train.txt")
held_path = os.path.join(OUT, "corpus_heldout.txt")

written_t = written_h = 0
ft = open(train_path, "w")
fh = open(held_path, "w")
try:
    for ex in ds:
        line = " ".join(ex["text"].split()) + "\n"  # collapse whitespace, one doc per line
        b = len(line.encode("utf-8"))
        if written_t < train_cap:
            ft.write(line)
            written_t += b
        elif written_h < held_cap:
            fh.write(line)
            written_h += b
        else:
            break
finally:
    ft.close()
    fh.close()

print(f"train:   {written_t/1e6:.2f} MB -> {train_path}")
print(f"heldout: {written_h/1e6:.2f} MB -> {held_path}")
