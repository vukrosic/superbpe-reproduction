# METHODOLOGY — milestones & pass/fail gates

How this reproduction was structured: each milestone had a pass/fail check defined *before*
implementation, smallest scale first. This is the build plan the work followed, kept as a record of
methodology. For the per-claim results and verdicts, see [`REPRO_REPORT.md`](REPRO_REPORT.md).
Scope: the tokenizer-level claims (C1–C6). Model runs (C7–C8) are out of scope at this scale.

## M1 — Two-phase BPE trainer (the core mechanism)
**Build:** from-scratch byte-level BPE with a transition point `t`. Word boundaries encoded
GPT-2-style (leading-space marker). Phase 1 (vocab < t): pair-frequency counting **skips** pairs
that straddle a word boundary. Phase 2 (vocab ≥ t): all adjacent pairs counted, so superwords form.
`t = T` ⇒ plain BPE; `t = 0` ⇒ naive no-pretok BPE.
**Check (toy corpus):** (a) with `t=T`, no learned token contains an internal word boundary
(pure subwords); (b) with `t < T`, at least one learned token spans a whitespace boundary
(a real superword like `the_`+next); (c) encoding the toy text with the SuperBPE tokenizer uses
**strictly fewer tokens** than with the BPE tokenizer at the same vocab. If (b) or (c) fail, the
mechanism is wrong — stop and fix before scaling.

## M2 — Encoding-efficiency eval (bytes/token)
**Build:** encode a fixed held-out shard, report bytes-per-token = total UTF-8 bytes / token count.
**Check:** numbers are deterministic across reruns; BPE bytes/token on the held-out shard lands in a
sane range (~3–5 for English subword vocab). Same shard reused for every tokenizer (A8).

## M3 — Vocab-size sweep → C2/C3 (BPE plateau vs SuperBPE keeps rising)
**Build:** train BPE, SuperBPE (fixed `t` fraction), and naive-no-pretok across vocab sizes
{1k,2k,4k,8k,16k,32k,64k} on a small corpus; plot bytes/token vs vocab.
**Check (reproduces C2/C3/C6 in shape):** BPE curve **flattens** at high vocab; SuperBPE curve keeps
**rising** and overtakes BPE; naive-no-pretok is **worse** than SuperBPE. Direction match = pass
(absolute values will differ from the paper's 200k-vocab numbers — A1/A2).

## M4 — Transition-point sweep → C4/C5 (smooth curve, interior optimum)
**Build:** fix vocab `T`, sweep `t` ∈ {0, .1T, .2T, … T}; plot bytes/token vs `t`.
**Check (reproduces C4):** curve is smooth and has an **interior maximum** (best efficiency at some
`0 < t < T`), with `t=T` (=BPE) and `t=0` (=naive) both worse than the peak. This is the headline
qualitative result and the basis of our novel small-scale `t`-sweep ledger.

## M5 — Ledger + write-up
Save the sweep tables + plots under `runs/` and record per-claim deltas for C1–C6 in
[`REPRO_REPORT.md`](REPRO_REPORT.md).

## Semantic-diff gate
Before declaring M1 correct, diff the merge logic against `PythonNut/superbpe` and log any divergence
in [`REPRO_REPORT.md`](REPRO_REPORT.md). Reimplementation is allowed; silent divergence is not.
