# JOURNAL — SuperBPE reproduction

Append-only. Source of truth for "where are we."

---

### 2026-06-24 — Phase 1: comprehension & claims ledger
- Created repo under `research-repos/superbpe-reproduction/`, following the
  `research-reproduction-agent` template.
- Downloaded source PDF (`paper/superbpe-2503.13423.pdf`, COLM 2025 v3) and read all 10 pages
  directly (not a summary).
- Understood the method: two-phase BPE with a transition point `t` — phase 1 = BPE with whitespace
  pretok (subwords), phase 2 = continue BPE with pretok disabled (superwords). `t=T` ⇒ BPE, `t=0` ⇒
  naive no-pretok BPE.
- Wrote `CLAIMS.md` (C1–C12, target-marked) and `ASSUMPTIONS.md` (A1–A8 + 3 open questions).
- **Decision:** target the tokenizer-level encoding-efficiency claims (C1–C6) as primary — they're
  the heart of the method and fully reproducible on a Mac with no model training. Small-model BPB
  (C7–C8) attempted at reduced scale, direction-only. Downstream/MMLU (C10–C12) explicitly out of scope.
- **STOP — awaiting Vuk's review of CLAIMS.md + ASSUMPTIONS.md before writing any code** (per Phase 1).
- Next action after confirmation: write `PLAN.md` (Phase 2) — milestone 1 = two-phase tokenizer
  trainer + bytes/token eval, with the C2/C3 plateau-vs-rising contrast as its pass/fail check.

### 2026-06-24 — Phases 2 & 3 (tokenizer half)
- Vuk's decisions: pick vocab/param tradeoff myself (more compute OK for faithfulness); do
  **tokenizer claims first, then decide** on model runs; internal-first.
- Wrote `PLAN.md` (M1–M5).
- Implemented `src/bpe.py`: from-scratch byte-level two-phase BPE. Phase distinction comes from the
  training unit — phase 1 on word units (no cross-word pairs possible), phase 2 on document units
  (boundaries dissolved). Unified encoder applies ordered merges over whole text.
- **M1 verification (toy) caught a real bug:** phase-2 guard was `len(vocab) < vocab_size`, which
  fired even for pure BPE when phase 1 exhausted its within-word pairs early on the tiny corpus →
  BPE wrongly grew superwords. Fixed to `t < vocab_size`. Re-run: (a) BPE has 0 superwords, (b)
  SuperBPE forms superwords (`_the_cat`, …), (c) SuperBPE 4.36 > BPE 3.39 > naive 3.05 bytes/tok.
  PASS. The naive<BPE<SuperBPE ordering already previews C6.
- Speedups (correctness-preserving, re-verified on toy each time): (1) transition re-encode uses a
  per-word merge cache instead of slow whole-doc encode(); (2) encoder merges all occurrences of the
  lowest-rank pair per pass; (3) best-pair selection uses a lazy-deletion max-heap instead of an
  O(pairs) scan per merge — required to reach 64k vocab in pure Python.
- Fetched corpus from cosmopedia-v2 (A1): `data/corpus_train.txt` 10 MB, `data/corpus_heldout.txt` 1 MB.
- NEXT: confirm heap timing scales to 64k, then run `src/sweep_efficiency.py` (M3 vocab sweep +
  M4 transition sweep) and plot.

### 2026-06-24 — Heap timing + preliminary results + Paper Lab integration
- Heap fix confirmed: BPE V=8k trains in ~3s. Encoding (bytes/token eval) is now the slow part.
- **Preliminary M3 (cosmopedia, 10MB train / held-out):** V=2k → BPE 3.23, SuperBPE 3.34; V=8k →
  BPE 4.51, **SuperBPE 5.00**, superwords 140→1352. Direction matches C2/C3 and the gap widens with
  vocab — the reproduction is working. Logged to `RESULTS.md`.
- Launched full sweep `src/sweep_efficiency.py` (vocab 1k–64k + transition sweep at 16k) in the
  background → `runs/sweep.log` + `runs/encoding_efficiency.json`. Plot with `src/plot_efficiency.py`.
- **Paper Lab integration (Vuk's request):** added `reproduction.json` (+ `RESULTS.md`, `VIDEO.md`,
  `.gitignore`) so the repo is discovered by Command Center's `/api/reproductions` scanner.
  Verified live at http://localhost:4317/paper-lab — SuperBPE card renders (priority 1, in-progress,
  $0 compute ladder). No Command Center code changed; it stays a read-only consumer per the handoff.

### 2026-06-24 — Sweep complete, deck, semantic diff, report, GitHub
- Full sweep finished. **Vocab sweep (C2/C3):** BPE 2.60→5.22, SuperBPE 2.65→6.65 over 1k→32k; gap
  widens +0.05→+1.43; BPE gains decelerate (plateau). **Transition sweep at 16k (C4):** naive 5.59,
  t=0.1T 5.74, 0.2T 5.77, 0.4T 5.79, **0.6T 5.80 (peak)**, 0.8T 5.77, BPE(t=T) 4.91 — smooth interior
  optimum, both extremes worse. **C6:** naive@32k 6.41 < SuperBPE@32k 6.65. Plots: `runs/fig1`, `fig2`.
- Built `slides.html` (12-slide deck, inline-SVG diagrams + reproduced curve); slide 5 now carries the
  measured transition numbers.
- **Semantic diff vs `PythonNut/superbpe`** (Phase 3 gate): same byte map / phase structure / BPE rule;
  divergences = pure-Python-unit-switch vs their HF-fork resume, and document-boundary vs stage-2 regex.
  Logged in `REPRO_REPORT.md`.
- Wrote `REPRO_REPORT.md` (Phase 5 per-claim verdicts: C2/C3/C4/C6 ✅; C1 🟡; C5/C7/C8/C10–12 ⬜).
- Status → publishing to GitHub (public) as `superbpe-reproduction`.
