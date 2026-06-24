# CLAIMS.md — SuperBPE (arXiv:2503.13423, COLM 2025)

All numbers read directly from the source PDF (`paper/superbpe-2503.13423.pdf`), not a summary.
★ = TARGET_CLAIM (reproducible at Mac scale). ☆ = out of scope here (documented, not targeted).

## Method (what we actually implement)

SuperBPE = standard byte-level BPE split into **two phases** by a *transition point* `t < T`
(`T` = final vocab size):
- **Phase 1 (vocab 0 → t):** ordinary BPE *with* whitespace pretokenization — merges cannot cross
  whitespace. Identical to regular BPE. Learns subwords.
- **Phase 2 (vocab t → T):** continue BPE from the phase-1 vocab but **disable** whitespace
  pretokenization, so pairs that bridge whitespace can merge → "superwords" (multi-word tokens).
- Edge cases: `t = T` ⇒ pure BPE; `t = 0` ⇒ naive no-pretokenization BPE.
- No change to model architecture, training, or decoding. (§2.2)

## Encoding-efficiency claims (tokenizer-only — PRIMARY TARGETS)

Measured in **bytes-per-token** (higher = more efficient) on a held-out subset; paper trains
tokenizers on a 10 GB subset of OLMo2's corpus. Final vocab `T = 200k`.

| ID | Claim | Paper number | Source | Target? |
|----|-------|--------------|--------|---------|
| C1 | SuperBPE encodes a fixed text in **up to 33% fewer tokens** than BPE at 200k vocab | up to −33% tokens | Abstract, §2.3, Fig 1 | ★ |
| C2 | BPE encoding efficiency **plateaus** (~50k vocab) and is bounded by avg word length | BPE → 4.45 bpt @200k; upper bound 4.68 | §2.3, Fig 1 | ★ |
| C3 | SuperBPE keeps improving with vocab size; **exceeds BPE's 4.68 upper bound at only ~12k vocab** | 5.55 bpt @50k; 6.63 bpt @200k | §2.3, Fig 1 | ★ |
| C4 | Encoding efficiency varies **smoothly** with transition point `t`, with an **interior optimum** near `t ≈ 80k` | Fig 2 curve, peak ~6.6 bpt @ t≈80k | §2.3, Fig 2 | ★ |
| C5 | Encoding efficiency by transition point @200k vocab | BPE 4.46 · t80k 6.63 · t160k 6.33 · t180k 6.09 (bpt) | Table 2 | ★ |
| C6 | Naive no-pretok BPE (`t=0`) is *worse* than SuperBPE (greedy makes bad early merges) | qualitative, Fig 1 "BPE w/o pretok" | §2.3, Fig 1 | ★ (cheap) |

## Small-model / loss claims (SECONDARY — attempt at reduced scale, BPB only)

Paper's small-scale runs use **680M and 1.9B** baselines and report **bits-per-byte (BPB)** only,
stating downstream evals are "too noisy for our small models." We go smaller still.

| ID | Claim | Paper number | Source | Target? |
|----|-------|--------------|--------|---------|
| C7 | At matched compute, 8B SuperBPE & BPE reach **very close BPB** (ranking ≠ downstream ranking) | BPE 0.7465 · SBPE-8B 0.7482 · SBPE-11B 0.7445 | §4.1 | ★ direction only |
| C8 | Scaling (680M/1.9B): **SuperBPE matching inference-compute achieves lower BPB at every size/budget**; in the under-trained regime both SuperBPE variants beat BPE | Fig 5 | §4.3, Fig 5 | ★ direction only |
| C9 | SuperBPE makes **fewer very-high and very-low loss predictions** (more uniform per-token difficulty); excluding `_the/_of/_to` flips BPB ranking by 0.02 | Fig 4; −0.02 BPB | §4.2 | ☆ stretch |

## Downstream claims (OUT OF SCOPE — meaningless at ~1M–10M params)

| ID | Claim | Paper number | Source | Target? |
|----|-------|--------------|--------|---------|
| C10 | SuperBPE-8B (t=180k) avg **+4.0** over BPE across 30 tasks, wins 25/30 | 39.8 → 43.8 | Table 1 | ☆ NO |
| C11 | **MMLU +8.2** (36.5 → 44.7) | +8.2 | Table 1 | ☆ NO |
| C12 | Inference compute reduced **27%** (t=180k) / **35%** (t=80k, with +3.1% task) | −27% / −35% | Abstract, §3.2 | ☆ NO (8B-scale) |

## Reference setup (paper, for context — Table 2)

8B models, vocab 200k, ~330B tokens, OLMo2-7B config, effective context fixed at ~18,262 bytes.
Context length in tokens: BPE 4096 · t80k 2756 · t160k 2884 · t180k 3000. Tokenizers train in a
few hours on 100 CPUs.

## What "done" looks like here

Reproduce C1–C6 (tokenizer mechanism + transition-point curve) with the right **shape and
direction** on our own corpus/vocab scale, and report C7–C8 honestly at reduced scale. The novel
contribution this repo adds beyond the paper: a **transition-point `t` sweep ablation at small
scale** with our own bytes/token + BPB ledger — data we own.
