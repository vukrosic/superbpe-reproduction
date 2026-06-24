"""
From-scratch byte-level BPE with a SuperBPE transition point.

The whole SuperBPE method (Liu et al., 2025, arXiv:2503.13423 §2.2) is a two-phase curriculum:

  - Phase 1 (vocab 0 -> t):  ordinary BPE *with* whitespace pretokenization. Merges cannot cross
                             word boundaries -> the tokenizer learns subwords.
  - Phase 2 (vocab t -> T):  pretokenization is disabled. Merges may now bridge whitespace ->
                             the tokenizer learns "superwords" (multi-word tokens).

  t == T  -> plain BPE.    t == 0  -> naive no-pretokenization BPE.

We get the phase distinction for free from the *unit of training*:
  - Phase 1 trains on word units, so a cross-word pair never exists inside a unit.
  - At the transition we re-encode each document with the phase-1 merges and concatenate its words
    into one document unit, so phase-2 pairs can span the (now dissolved) word boundaries.

Byte-level throughout (GPT-2 byte<->unicode map); a leading space becomes the marker 'Ġ', so word
boundaries survive as part of the tokens. Encoding applies the ordered merge list greedily over the
whole text, which is equivalent to per-word merging when t == T and is required when t < T.
"""

from __future__ import annotations

import heapq
import json
import regex as re
from collections import Counter, defaultdict
from dataclasses import dataclass


# GPT-2 pretokenization pattern: keeps a leading space attached to the following word.
_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte<->unicode map: every byte -> a printable unicode char (space->'Ġ')."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


def _word_symbols(piece: str) -> tuple[str, ...]:
    """A pretoken piece -> tuple of single-byte symbol strings (the initial vocab units)."""
    return tuple(BYTE_ENCODER[b] for b in piece.encode("utf-8"))


@dataclass
class Tokenizer:
    vocab: list[str]                 # id -> token string (byte-level)
    merges: list[tuple[str, str]]    # ordered learned merges
    transition_point: int            # t (vocab size where phase 2 began); == len(vocab) means pure BPE

    def __post_init__(self):
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}

    def encode(self, text: str) -> list[str]:
        """Encode by applying merges in learned order over the whole text (no word isolation).

        Each iteration finds the lowest-rank pair currently present and merges ALL its occurrences,
        which is the standard BPE encoding and far cheaper than re-scanning per single merge.
        """
        symbols = [BYTE_ENCODER[b] for b in text.encode("utf-8")]
        while len(symbols) >= 2:
            best_rank, best_pair = None, None
            for pair in zip(symbols, symbols[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            a, b = best_pair
            merged = a + b
            out, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return symbols

    def crosses_boundary(self, token: str) -> bool:
        """True if a token spans a whitespace boundary (a superword), ignoring a leading space marker."""
        return "Ġ" in token[1:]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {"vocab": self.vocab, "merges": self.merges, "transition_point": self.transition_point},
                f,
            )

    @staticmethod
    def load(path: str) -> "Tokenizer":
        with open(path) as f:
            d = json.load(f)
        return Tokenizer(d["vocab"], [tuple(m) for m in d["merges"]], d["transition_point"])


def _count_pairs(units: list[tuple[list[str], int]]) -> tuple[Counter, dict]:
    """Initial pair frequencies and an index: pair -> set of unit indices containing it."""
    pair_counts: Counter = Counter()
    pair_to_units: dict = defaultdict(set)
    for idx, (syms, freq) in enumerate(units):
        for a, b in zip(syms, syms[1:]):
            pair_counts[(a, b)] += freq
            pair_to_units[(a, b)].add(idx)
    return pair_counts, pair_to_units


def _apply_merge(units, pair, new_sym, pair_counts, pair_to_units):
    """Merge `pair` -> `new_sym` everywhere, updating counts/index. Returns pairs whose count changed."""
    a, b = pair
    touched = set()
    for idx in list(pair_to_units.get(pair, ())):
        syms, freq = units[idx]
        i, out = 0, []
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                # neighbours lose their old pairs, gain new ones with new_sym
                if out:
                    left = out[-1]
                    pair_counts[(left, a)] -= freq
                    pair_counts[(left, new_sym)] += freq
                    pair_to_units[(left, new_sym)].add(idx)
                    touched.add((left, a))
                    touched.add((left, new_sym))
                if i + 2 < len(syms):
                    right = syms[i + 2]
                    pair_counts[(b, right)] -= freq
                    pair_counts[(new_sym, right)] += freq
                    pair_to_units[(new_sym, right)].add(idx)
                    touched.add((b, right))
                    touched.add((new_sym, right))
                out.append(new_sym)
                i += 2
            else:
                out.append(syms[i])
                i += 1
        units[idx] = (out, freq)
    pair_counts[pair] = 0
    pair_to_units.pop(pair, None)
    return touched


def _train_phase(units, vocab, vocab_target, merges):
    """Run incremental BPE merges on `units` until len(vocab) == vocab_target.

    Best-pair selection uses a max-heap with lazy deletion: stale entries (whose stored count no
    longer matches `pair_counts`) are discarded on pop, and every pair whose count changes is
    re-pushed with its current value.
    """
    pair_counts, pair_to_units = _count_pairs(units)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    while len(vocab) < vocab_target:
        pair = None
        while heap:
            neg_c, cand = heapq.heappop(heap)
            if pair_counts.get(cand, 0) == -neg_c and -neg_c > 0:
                pair = cand
                break
        if pair is None:
            break
        new_sym = pair[0] + pair[1]
        merges.append(pair)
        vocab.append(new_sym)
        for p in _apply_merge(units, pair, new_sym, pair_counts, pair_to_units):
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))


def train(documents: list[str], vocab_size: int, transition_point: int | None = None) -> Tokenizer:
    """
    Train a (Super)BPE tokenizer.

    transition_point t: vocab size at which to switch from within-word to cross-word merges.
      None or >= vocab_size -> plain BPE; 256 (== base vocab) -> naive no-pretok BPE.
    """
    t = vocab_size if transition_point is None else transition_point

    # Base vocab = all 256 byte symbols.
    vocab = [BYTE_ENCODER[b] for b in range(256)]
    merges: list[tuple[str, str]] = []

    # ---- Phase 1: within-word BPE up to min(t, vocab_size) ----
    word_freqs: Counter = Counter()
    doc_words: list[list[tuple[str, ...]]] = []  # documents as ordered lists of word keys (for phase 2)
    for doc in documents:
        words = [_word_symbols(m.group()) for m in _PAT.finditer(doc)]
        doc_words.append(words)
        word_freqs.update(words)
    word_keys = list(word_freqs.keys())
    units = [([*k], word_freqs[k]) for k in word_keys]
    _train_phase(units, vocab, min(t, vocab_size), merges)

    # ---- Phase 2: cross-word BPE up to vocab_size ----
    # Only when a transition was actually requested (t < vocab_size). For pure BPE (t == vocab_size)
    # we must never enter phase 2, even if phase 1 exhausted its within-word pairs early.
    if t < vocab_size and len(vocab) < vocab_size:
        # Phase-1 merges are within-word, so each unique word's merged form is just its unit now.
        # Re-encode documents by concatenating cached per-word merges (no slow whole-doc encode()).
        merged_word = {k: tuple(units[i][0]) for i, k in enumerate(word_keys)}
        doc_units: Counter = Counter()
        for words in doc_words:
            seq = tuple(tok for w in words for tok in merged_word[w])
            if seq:
                doc_units[seq] += 1
        units = [([*seq], freq) for seq, freq in doc_units.items()]
        _train_phase(units, vocab, vocab_size, merges)

    return Tokenizer(vocab, merges, min(t, vocab_size))
