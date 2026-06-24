"""M5: plot the encoding-efficiency sweeps from runs/encoding_efficiency.json."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RUNS = os.path.join(HERE, "..", "runs")

with open(os.path.join(RUNS, "encoding_efficiency.json")) as f:
    R = json.load(f)

# --- Fig A: bytes/token vs vocab size (reproduces Fig 1 shape: BPE plateau, SuperBPE keeps rising) ---
vs = R["vocab_sweep"]
x = [r["vocab"] for r in vs]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x, [r["bpe"] for r in vs], "o-", label="BPE", color="#1f77b4")
ax.plot(x, [r["superbpe"] for r in vs], "s-", label="SuperBPE (t=0.4T)", color="#7b1f5c")
naive_pts = [(r["vocab"], r["naive"]) for r in vs if "naive" in r]
if naive_pts:
    ax.scatter([p[0] for p in naive_pts], [p[1] for p in naive_pts], marker="^",
               color="#2ca02c", s=80, zorder=5, label="BPE w/o pretok (naive)")
ax.set_xlabel("Vocabulary size")
ax.set_ylabel("Bytes per token (↑ = more efficient)")
ax.set_title("Encoding efficiency vs vocab size (reproduction of Fig 1)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RUNS, "fig1_vocab_sweep.png"), dpi=140)

# --- Fig B: bytes/token vs transition point (reproduces Fig 2: smooth, interior optimum) ---
ts = R["transition_sweep"]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([r["t"] for r in ts], [r["bytes_per_token"] for r in ts], "o-", color="#7b1f5c")
for r in ts:
    ax.annotate(r["label"], (r["t"], r["bytes_per_token"]), fontsize=7,
                textcoords="offset points", xytext=(0, 6), ha="center")
ax.set_xlabel("Transition point t (at fixed vocab=16k)")
ax.set_ylabel("Bytes per token (↑ = more efficient)")
ax.set_title("Encoding efficiency vs transition point (reproduction of Fig 2)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RUNS, "fig2_transition_sweep.png"), dpi=140)

print("wrote runs/fig1_vocab_sweep.png and runs/fig2_transition_sweep.png")
