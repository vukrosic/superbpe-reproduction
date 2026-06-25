# REPRO_REPORT — SuperBPE (arXiv:2503.13423, COLM 2025)

Independent small-scale reproduction. **Scale:** local Mac, CPU, $0. **Corpus:** 10 MB cosmopedia-v2
(train) + held-out shard. **Tokenizer:** from-scratch byte-level two-phase BPE (`src/bpe.py`).

Verdict key: ✅ reproduced (shape/direction) · 🟡 partial · ⬜ not attempted (out of scope at this scale).

## Per-claim results

| ID | Claim (paper) | Paper | Ours | Verdict | Evidence |
|----|---------------|-------|------|---------|----------|
| C2 | BPE encoding efficiency **plateaus** with vocab | 4.45 bpt @200k, flat after ~50k | gains decelerate: +0.61→+0.31 bpt per doubling, flattening by 32k | ✅ | `runs/fig1_vocab_sweep.png` |
| C3 | SuperBPE **keeps rising**, overtakes BPE | exceeds BPE upper bound ~12k vocab | SuperBPE > BPE at every vocab; gap widens +0.05→+1.43 bpt (1k→32k) | ✅ shape | `runs/encoding_efficiency.json` |
| C4 | Efficiency is **smooth in `t`** with an **interior optimum** | peak ~t=80k of 200k (Fig 2) | smooth; peak at **t=0.6T = 5.80 bpt**; t=T (BPE)=4.91, t=0 (naive)=5.59 — both worse | ✅ | `runs/fig2_transition_sweep.png` |
| C6 | Naive no-pretok BPE is **worse than SuperBPE** | qualitative (Fig 1) | naive@32k 6.41 < SuperBPE@32k 6.65; naive@16k 5.59 < interior 5.80 | ✅ | sweep |
| C1 | Up to **33% fewer tokens** at 200k vocab | −33% tokens | not measured at 200k; direction holds (fewer tokens, widening gap) | 🟡 | scale-limited (A2) |
| C5 | Exact bytes/token table @200k | 4.46 / 6.63 / 6.33 / 6.09 | different corpus + vocab — not targeted | ⬜ | A1/A2 |
| C7/C8 | Small-model **BPB** close / SuperBPE lower at matched compute | 0.7465 vs 0.7482 ; Fig 5 | not run (recommended skip: noisy at ~1M params) | ⬜ | A3 |
| C10–C12 | **+8.2 MMLU**, +4.0 avg, −27% inference (8B) | Table 1 | **not attempted** — meaningless at this scale | ⬜ | A3 |

## What it took
- One real bug, caught by the M1 toy gate: the phase-2 entry guard fired for pure BPE when phase 1
  exhausted its within-word pairs early. Fixed to gate on `t < vocab_size`.
- Three correctness-preserving speedups to make 32k-vocab tractable in pure Python: per-word
  re-encode cache at the phase transition, all-occurrences-per-rank encoder, lazy-deletion max-heap
  for best-pair selection, and a chunked bytes/token eval.

## Semantic diff vs official code (`github.com/PythonNut/superbpe`)
Checked our merge logic against the authors' `train_or_extend_tokenizer` (`utils.py`) and
`train_tokenizer.py`. **Same** where it matters: identical GPT-2 `bytes_to_unicode` map; phase 1 =
BPE with a whitespace-splitting pretok; phase 2 = continue from phase-1 merges with whitespace
pretok disabled; standard highest-frequency BPE merges. **Divergences (logged, not silent):**
1. **Engine.** They use a custom HF `tokenizers` Rust fork and "extend" by resuming `merges.txt`;
   we reimplement in pure Python and realize phase 2 by switching the training unit (word → document).
   Functionally equivalent; not the same codebase.
2. **Phase-2 boundaries.** Their stage-2 still applies a (non-whitespace) regex, which can constrain
   what merges across; we impose only the document boundary, so our superwords can in principle span
   further (visible on highly repetitive toy text). On natural corpora the learned superwords match
   the paper's character (2–4-word fixed expressions: `▁of▁course`, `▁by▁the▁way`, …).

## Honest bottom line
The **tokenizer-level mechanism and its qualitative claims (C2, C3, C4, C6) reproduce cleanly** at
small scale, $0. Absolute numbers differ from the paper by design (different corpus/vocab, A1/A2).
The **8B downstream/MMLU results are not reproduced and were never claimed** — they are not
measurable at this scale. Confidence the reproduction is correct-for-the-right-reason: **high** for
C2/C3/C4/C6 (matches paper shape + official-code semantics + sane learned superwords); the model-side
claims remain open by choice.
