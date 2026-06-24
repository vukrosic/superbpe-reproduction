# VIDEO — SuperBPE reproduction

**Working title:** Tokenizers That Span Words: I Reproduced SuperBPE on a Laptop
**Status:** planned (internal research first; film after the tokenizer claims are clean)

## The hook
Every tokenizer you've used refuses to merge across a space. SuperBPE (COLM 2025) just... lifts that
rule, and a fixed text needs up to 33% fewer tokens. I reimplemented it from scratch and reproduced
the core result on a laptop — no GPU.

## Beat sheet
1. The one weird rule: BPE never merges across whitespace. Why?
2. SuperBPE's fix = a two-phase curriculum (subwords first, then superwords). ~30 lines of idea.
3. From-scratch code: the phase distinction falls out of the *training unit* (words → documents).
4. The reproduction: BPE's efficiency plateaus, SuperBPE's keeps climbing with vocab size (live curves).
5. The transition point `t`: sweep it, find the interior optimum (our own ablation, not in the tweet).
6. Honest verdict: what reproduces at small scale (encoding efficiency) and what doesn't (the 8B
   MMLU gain — and *why* you can't claim it on a laptop).

## Definition of done
Shows the mechanism, the code diff, the reproduced encoding-efficiency curves, the transition-point
sweep, and a clear reproduce-or-reject verdict with the scale caveat stated plainly.

## Notes
- Differentiator vs other "explain SuperBPE" content: original from-scratch code + our own
  transition-point ablation curve + an explicit honesty section on scale limits.
- Ties to the tools mission: the `t`-sweep is a clean VoidSpark autonomous-experiment demo.
