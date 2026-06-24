"""M1 verification: prove the two-phase mechanism on a tiny corpus before scaling (PLAN.md M1)."""

from bpe import train

# Toy corpus with heavy repetition of multi-word expressions, so superwords are learnable.
TEXT = (
    "by the way the cat sat on the mat . " * 200
    + "in the long run the cat and the dog ran . " * 200
    + "of course the cat is on the way home . " * 200
)
DOCS = [TEXT]
VOCAB = 400  # small target so phase-2 merges are reachable

bpe = train(DOCS, vocab_size=VOCAB, transition_point=VOCAB)          # t == T  -> plain BPE
sbpe = train(DOCS, vocab_size=VOCAB, transition_point=320)            # t < T   -> SuperBPE
naive = train(DOCS, vocab_size=VOCAB, transition_point=256)          # t == 0  -> naive no-pretok

bpe_supers = [t for t in bpe.vocab if bpe.crosses_boundary(t)]
sbpe_supers = [t for t in sbpe.vocab if sbpe.crosses_boundary(t)]

held_out = "the cat sat on the mat by the way of course in the long run ."
n_bytes = len(held_out.encode("utf-8"))
bpe_tok = bpe.encode(held_out)
sbpe_tok = sbpe.encode(held_out)

print("=== CHECK (a): plain BPE learns NO cross-word tokens ===")
print(f"  BPE superword tokens: {len(bpe_supers)}  (expect 0)")

print("\n=== CHECK (b): SuperBPE learns superwords ===")
print(f"  SuperBPE superword tokens: {len(sbpe_supers)}  (expect > 0)")
print(f"  examples: {[s.replace(chr(0x120), '_') for s in sbpe_supers[:8]]}")

print("\n=== CHECK (c): SuperBPE encodes the held-out text in fewer tokens ===")
print(f"  held-out bytes: {n_bytes}")
print(f"  BPE      tokens: {len(bpe_tok)}  -> {n_bytes/len(bpe_tok):.2f} bytes/token")
print(f"  SuperBPE tokens: {len(sbpe_tok)} -> {n_bytes/len(sbpe_tok):.2f} bytes/token")
print(f"  naive    tokens: {len(naive.encode(held_out))} -> {n_bytes/len(naive.encode(held_out)):.2f} bytes/token")

ok_a = len(bpe_supers) == 0
ok_b = len(sbpe_supers) > 0
ok_c = len(sbpe_tok) < len(bpe_tok)
print(f"\nRESULT: (a)={ok_a}  (b)={ok_b}  (c)={ok_c}  ->  {'PASS' if (ok_a and ok_b and ok_c) else 'FAIL'}")
