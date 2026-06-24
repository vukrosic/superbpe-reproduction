# RESULTS — SuperBPE reproduction

Status: **tokenizer half in progress.** Mechanism verified; encoding-efficiency sweep running.
Full per-claim verdicts land in `REPRO_REPORT.md` (Phase 5). This file is the live results log.

## Verified so far

**Mechanism (M1, toy corpus).** Two-phase BPE works: plain BPE learns 0 cross-word tokens;
SuperBPE learns real superwords (`_the_cat`, `_the_way`, …); naive no-pretok is worst. PASS.

**Encoding efficiency (M3, preliminary — cosmopedia-v2, 10 MB train / 1 MB held-out).**

| vocab | BPE (bytes/tok) | SuperBPE t=0.4T | naive | SuperBPE superwords |
|------:|----------------:|----------------:|------:|--------------------:|
| 2,000 | 3.23 | 3.34 | — | 140 |
| 8,000 | 4.51 | **5.00** | — | 1,352 |

Direction matches the paper (C2/C3): SuperBPE encodes more bytes per token than BPE, and the gap
**grows with vocab size** (+0.11 at 2k → +0.49 at 8k). Full 1k–64k sweep + transition-point sweep
+ naive baseline in progress.

## Honest scope note

Absolute bytes/token differ from the paper's (10 GB OLMo2 corpus, 200k vocab) — expected, see
`ASSUMPTIONS.md` A1/A2. We reproduce the **shape and direction**, not the 8B/200k numbers. The
headline **+8.2% MMLU** is **not attempted** — meaningless at this scale (C10–C12).

## Pending
- M3 full vocab sweep (1k–64k), M4 transition-point curve (interior optimum, C4/C5), plots.
- Decision with Vuk on small-model BPB runs (C7/C8).
