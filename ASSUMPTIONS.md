# ASSUMPTIONS.md — SuperBPE reproduction

Every place the paper's setup is unstated, ambiguous, or deliberately diverged from because we run
on a Mac at small scale. Nothing here is filled silently — each is a logged choice for review.

## A1 — Corpus (DIVERGENCE)
Paper trains tokenizers on a 10 GB subset of OLMo2's pretraining corpus and evals bytes/token on a
held-out split. **Choice:** use a small on-disk corpus (`cosmopedia-v2`, ~10 MB train + 1 MB
held-out; regenerate with `src/fetch_corpus.py`) for the bytes/token eval.
**Why:** zero download, already validated in the harness. **Effect:** absolute bytes/token numbers
will differ from the paper; the *direction and shape* (C1–C6) should hold since they're properties
of the algorithm, not the corpus. This is the main reason "done" = qualitative match, not number match.

## A2 — Final vocab size (DIVERGENCE, scale-driven)
Paper fixes `T = 200k`. At ~1M–10M params, a 200k×d_model embedding table dwarfs the model
(200k×64 ≈ 12.8M params alone). **Choice:** two separate vocab regimes —
- **Tokenizer-only experiments (C1–C6):** sweep vocab up to ~64k so the BPE-plateau /
  SuperBPE-keeps-rising contrast is visible without needing 200k.
- **Model-training experiments (C7–C8):** small `T` (candidate 8k or 16k) so embeddings don't
  dominate; transition point `t` scaled proportionally.
**Why:** keeps the comparison about the *algorithm* at a scale the Mac can train. **Effect:** we
reproduce the trend, not the 200k absolute bpt values. Logged as the single biggest deviation.

## A3 — Model scale & metric (DIVERGENCE)
Paper's smallest models are 680M/1.9B, reporting BPB only. **Choice:** use the harness
`Tiny1M3MConfig` (~0.94M) for fast iteration and `Screen10M20MConfig` if time allows; primary
metric = **BPB / val loss**, never downstream accuracy. **Why:** matches budget. **Effect:** C10–C12
(downstream, MMLU) are not attempted; C7–C8 reported as direction-only and likely noisy.

## A4 — Tokenizer implementation (DECISION + diff plan)
Paper's curriculum lives in `PythonNut/superbpe` (built on HF `tokenizers`). **Choice:** implement
the two-phase trainer ourselves with HF `tokenizers` (byte-level BPE; phase 1 with whitespace
pretokenizer, phase 2 continuing from phase-1 merges with the pretokenizer disabled), then do a
**semantic diff** of our merge logic against the official repo and log any divergence (per Phase 3).
**Why:** fewer/cleaner deps, and reimplementing forces real understanding of the mechanism.
**Risk:** continuing BPE training from an existing merge list with pretok disabled is the subtle
part; if HF `tokenizers` can't cleanly resume, fall back to vendoring the official trainer.

## A5 — Byte-level vs char-level & special tokens
**Choice:** GPT-2-style **byte-level** BPE (matches modern tokenizers and the paper's framing),
standard special tokens. Pretokenization regex: GPT-2/GPT-4 style for phase 1.

## A6 — "Effective context fixed in bytes" (DIVERGENCE)
Paper fixes effective context in *bytes* and adjusts train steps so total bytes/FLOPs match across
tokenizers (a SuperBPE model sees fewer tokens but equal bytes). **Choice:** for model runs, match
**total training bytes** (equivalently: scale `train_tokens` by the bytes/token ratio) between BPE
and SuperBPE runs so the comparison is fair, mirroring the paper. Document the exact ratio used.

## A7 — Seeds
**Choice:** pin `seed=42` for tokenizer training (deterministic given data anyway) and for model
runs; repeat key model comparisons over seeds 42/123/7 because small-scale BPB is noisy (per A3).

## A8 — Held-out split for bytes/token
**Choice:** fixed held-out shard of the corpus not seen during tokenizer training; same shard across
all tokenizers so the bytes/token comparison is apples-to-apples.
