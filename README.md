# SuperBPE Reproduction

Reproduction of **SuperBPE: Space Travel for Language Models** (Liu et al., COLM 2025) on a
local Mac, following the verification-first workflow in `../../research-reproduction-agent/`.

This is the first repo in a planned series of paper reproductions. Layout below is the
template every future reproduction should copy.

## INPUTS

- **PAPER**: SuperBPE: Space Travel for Language Models — arXiv:2503.13423 — `paper/superbpe-2503.13423.pdf`
- **OFFICIAL_CODE**: https://github.com/PythonNut/superbpe ; released tokenizers: https://huggingface.co/collections/UW/superbpe-67db2338062faa07c7473ffa
- **TARGET_CLAIMS**: tokenizer-level encoding-efficiency claims (see `CLAIMS.md`, marked ★).
  The 8B downstream/MMLU claims are **out of scope at this scale** and explicitly *not* targeted.
- **COMPUTE_BUDGET**: local Mac only (Apple MPS / CPU), $0 cloud. Tokenizer training = CPU minutes–hours.
  Model runs = `Tiny1M`/`Screen10M`-scale on MPS, a handful of hours total. No Vast box.
- **DEFINITION_OF_DONE**: reproduce the *direction and qualitative shape* of each ★ claim
  (SuperBPE encodes more bytes/token than BPE at fixed vocab; BPE plateaus while SuperBPE keeps
  improving with vocab size; encoding efficiency is smooth in the transition point `t` with an
  interior optimum), plus an honest account of any small-model BPB result. Exact 200k-vocab/8B
  numbers are not the target — the *mechanism* is.

## Status

**Phase 1 (comprehension & claims ledger) — awaiting review.** No implementation code written yet.
See `CLAIMS.md` + `ASSUMPTIONS.md`. Per the reproduction-agent protocol, implementation does not
start until these are confirmed or corrected.

## Layout (template for future reproductions)

```
superbpe-reproduction/
  paper/            # the source PDF (read directly, never a summary)
  CLAIMS.md         # every reproducible claim, exact numbers, target-marked
  ASSUMPTIONS.md    # every gap the paper leaves + the choice we make and why
  PLAN.md           # Phase 2: milestones + the pass/fail check for each (not yet written)
  JOURNAL.md        # append-only running log; source of truth for "where are we"
  REPRO_REPORT.md   # Phase 5: per-claim deltas + verdicts (not yet written)
  src/              # clean, minimal implementation (not yet written)
  runs/             # results, ledgers (matches llm-research-kit runs/ format)
```

## Harness reuse

Model runs reuse the existing kit (`../../llm-research-kit/` or `../../../llm-research-kit-scaling/`):
the only change SuperBPE needs is swapping the tokenizer at `setup_tokenizer()` in `data/loader.py`.
The transition-point sweep is a natural autonomous-experiment target for VoidSpark. Wiring is
decided in `PLAN.md` (Phase 2), not duplicated here.
