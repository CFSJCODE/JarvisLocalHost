"""Corpus-native byte-pair tokenizer.

The tokenizer is deterministic, starts with no linguistic vocabulary and may
only learn merge rules from the operator-authorized corpus.  Its serialized
form carries the corpus digest used during training so lineage can be audited.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional


class JarvisTokenizer:
    """Small deterministic BPE tokenizer trained from random/empty state."""

    FORMAT_VERSION = 3
    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "<SEP>": 4,
    }
    # Citation markers are structural output syntax, not semantic vocabulary.
    # Reserving their delimiter and decimal pieces makes every [E<number>]
    # marker round-trip without depending on whether the source PDFs happened
    # to contain citations or digits.
    STRUCTURAL_TOKENS = {
        "[E": 5,
        "]": 6,
        **{str(digit): 7 + digit for digit in range(10)},
    }
    BASE_TOKENS = {**SPECIAL_TOKENS, **STRUCTURAL_TOKENS}
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    SEP_ID = 4

    def __init__(self, vocab_size: int = 8_000) -> None:
        if vocab_size < len(self.BASE_TOKENS):
            raise ValueError("vocab_size is smaller than the reserved-token set")
        self.vocab_size = int(vocab_size)
        self.token2id: dict[str, int] = dict(self.BASE_TOKENS)
        self.id2token: dict[int, str] = {
            value: key for key, value in self.BASE_TOKENS.items()
        }
        self.merges: list[tuple[str, str]] = []
        self._trained = False
        self.corpus_sha256 = ""

    @staticmethod
    def corpus_digest(corpus: str) -> str:
        return hashlib.sha256(corpus.encode("utf-8")).hexdigest()

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        # Unicode letters, decimal runs and individual punctuation marks.  No
        # hand-written language/domain vocabulary is introduced here. Citation
        # markers are recognized before case folding and canonicalized to the
        # syntax consumed by the grounding verifier.
        output: list[str] = []
        for token in re.findall(
            r"\[e\d+\]|[^\W\d_]+|\d+|[^\w\s]",
            text,
            re.IGNORECASE | re.UNICODE,
        ):
            if not token.strip():
                continue
            marker = re.fullmatch(r"\[e(\d+)\]", token, re.IGNORECASE)
            output.append(f"[E{marker.group(1)}]" if marker else token.casefold())
        return output

    @staticmethod
    def _pair_stats(
        vocabulary: dict[tuple[str, ...], int],
    ) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = collections.defaultdict(int)
        for symbols, frequency in vocabulary.items():
            for left, right in zip(symbols, symbols[1:]):
                counts[(left, right)] += frequency
        return counts

    @staticmethod
    def _merge_pair(
        pair: tuple[str, str], vocabulary: dict[tuple[str, ...], int]
    ) -> dict[tuple[str, ...], int]:
        merged: dict[tuple[str, ...], int] = {}
        compound = pair[0] + pair[1]
        for symbols, frequency in vocabulary.items():
            output: list[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and symbols[index] == pair[0]
                    and symbols[index + 1] == pair[1]
                ):
                    output.append(compound)
                    index += 2
                else:
                    output.append(symbols[index])
                    index += 1
            key = tuple(output)
            merged[key] = merged.get(key, 0) + frequency
        return merged

    def train(
        self,
        corpus: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if not corpus.strip():
            raise ValueError("cannot train a tokenizer on an empty corpus")

        # Retraining the same object must not retain semantic state from an
        # earlier corpus.
        self.token2id = dict(self.BASE_TOKENS)
        self.id2token = {value: key for key, value in self.BASE_TOKENS.items()}
        self.merges = []
        words = collections.Counter(
            token
            for token in self._pretokenize(corpus)
            if re.fullmatch(r"\[E\d+\]", token) is None
        )
        vocab_words = [
            ([*word, "</w>"], freq) for word, freq in words.items()
        ]

        symbols = sorted({symbol for word, _ in vocab_words for symbol in word})
        for symbol in symbols:
            if len(self.token2id) >= self.vocab_size:
                break
            if symbol not in self.token2id:
                token_id = len(self.token2id)
                self.token2id[symbol] = token_id
                self.id2token[token_id] = symbol

        pair_counts: dict[tuple[str, str], int] = collections.defaultdict(int)
        pair_where: dict[tuple[str, str], set[int]] = collections.defaultdict(set)

        for w_idx, (w_symbols, freq) in enumerate(vocab_words):
            for i in range(len(w_symbols) - 1):
                p = (w_symbols[i], w_symbols[i + 1])
                pair_counts[p] += freq
                pair_where[p].add(w_idx)

        while len(self.token2id) < self.vocab_size:
            if not pair_counts:
                break
            best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
            if pair_counts[best_pair] <= 0:
                break

            compound = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            if compound not in self.token2id:
                token_id = len(self.token2id)
                self.token2id[compound] = token_id
                self.id2token[token_id] = compound

            if progress_callback and (
                len(self.token2id) % 100 == 0 or len(self.token2id) >= self.vocab_size
            ):
                progress_callback(len(self.token2id), self.vocab_size)

            affected_indices = list(pair_where.pop(best_pair, ()))
            del pair_counts[best_pair]

            p0, p1 = best_pair
            for w_idx in affected_indices:
                w_symbols, freq = vocab_words[w_idx]
                new_symbols: list[str] = []
                i = 0
                changed = False
                while i < len(w_symbols):
                    if (
                        i + 1 < len(w_symbols)
                        and w_symbols[i] == p0
                        and w_symbols[i + 1] == p1
                    ):
                        new_symbols.append(compound)
                        i += 2
                        changed = True
                    else:
                        new_symbols.append(w_symbols[i])
                        i += 1

                if not changed:
                    continue

                for j in range(len(w_symbols) - 1):
                    old_p = (w_symbols[j], w_symbols[j + 1])
                    if old_p != best_pair:
                        pair_counts[old_p] -= freq
                        if pair_counts[old_p] <= 0:
                            pair_counts.pop(old_p, None)
                        pair_where[old_p].discard(w_idx)
                        if not pair_where[old_p]:
                            pair_where.pop(old_p, None)

                for j in range(len(new_symbols) - 1):
                    new_p = (new_symbols[j], new_symbols[j + 1])
                    pair_counts[new_p] = pair_counts.get(new_p, 0) + freq
                    if new_p not in pair_where:
                        pair_where[new_p] = set()
                    pair_where[new_p].add(w_idx)

                vocab_words[w_idx] = (new_symbols, freq)

        self.corpus_sha256 = self.corpus_digest(corpus)
        self._trained = True

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        if not hasattr(self, "_bpe_ranks") or len(self._bpe_ranks) != len(self.merges):
            self._bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        ranks = self._bpe_ranks
        current = list(symbols)
        if len(current) <= 1:
            return current
        while len(current) > 1:
            pairs = [(current[i], current[i + 1]) for i in range(len(current) - 1)]
            valid = [(ranks[p], p) for p in pairs if p in ranks]
            if not valid:
                break
            _, best = min(valid, key=lambda x: x[0])
            compound = best[0] + best[1]
            new_cur: list[str] = []
            i = 0
            while i < len(current):
                if (
                    i + 1 < len(current)
                    and current[i] == best[0]
                    and current[i + 1] == best[1]
                ):
                    new_cur.append(compound)
                    i += 2
                else:
                    new_cur.append(current[i])
                    i += 1
            current = new_cur
        return current

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        if not hasattr(self, "_word_cache"):
            self._word_cache: dict[str, list[str]] = {}
        cache = self._word_cache

        for word in self._pretokenize(text):
            marker = re.fullmatch(r"\[E(\d+)\]", word)
            if marker:
                tokens.extend(("[E", *marker.group(1), "]"))
            else:
                cached = cache.get(word)
                if cached is None:
                    cached = self._apply_merges([*word, "</w>"])
                    cache[word] = cached
                tokens.extend(cached)
        return tokens

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.token2id.get(token, self.UNK_ID) for token in self.tokenize(text)]
        return [self.BOS_ID, *ids, self.EOS_ID] if add_special else ids

    def encode_segments(self, segments: Iterable[str]) -> list[int]:
        """Encode corpus units with explicit, learnable boundaries.

        The segments themselves remain the only source of semantic tokens.
        BOS/EOS delimit each unit and SEP is a target only between units, so a
        decoder trained on this stream learns document/chunk transitions rather
        than treating the entire corpus as one accidental flat paragraph.
        """

        normalized = [segment.strip() for segment in segments if segment.strip()]
        output: list[int] = []
        for index, segment in enumerate(normalized):
            output.extend(self.encode(segment, add_special=True))
            if index + 1 < len(normalized):
                output.append(self.SEP_ID)
        return output

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        parts: list[str] = []
        for token_id in ids:
            token = self.id2token.get(int(token_id), "<UNK>")
            if skip_special and token in self.SPECIAL_TOKENS:
                continue
            parts.append(token)
        return "".join(parts).replace("</w>", " ").strip()

    @property
    def vocab_actual_size(self) -> int:
        return len(self.token2id)

    @property
    def trained(self) -> bool:
        return self._trained

    def pad_sequence(self, ids: list[int], max_len: int) -> list[int]:
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        return ids[:max_len] + [self.PAD_ID] * max(0, max_len - len(ids))

    def to_dict(self) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "kind": "jarvis-bpe",
            "initialized_from": "empty",
            "external_vocabulary": False,
            "corpus_sha256": self.corpus_sha256,
            "vocab_size": self.vocab_size,
            "token2id": self.token2id,
            "merges": [list(pair) for pair in self.merges],
            "trained": self._trained,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        descriptor, temporary = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_corpus_sha256: str | None = None,
    ) -> "JarvisTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("external_vocabulary", False):
            raise ValueError("external tokenizer vocabulary is not allowed")
        if data.get("initialized_from", "empty") != "empty":
            raise ValueError("tokenizer was not initialized from an empty vocabulary")
        corpus_sha256 = str(data.get("corpus_sha256", ""))
        if expected_corpus_sha256 and corpus_sha256 != expected_corpus_sha256:
            raise ValueError("tokenizer corpus digest does not match the active corpus")

        tokenizer = cls(vocab_size=int(data["vocab_size"]))
        if int(data.get("format_version", 0)) != cls.FORMAT_VERSION:
            raise ValueError(
                "unsupported tokenizer format; rebuild it to obtain the citation contract"
            )
        tokenizer.token2id = {str(k): int(v) for k, v in data["token2id"].items()}
        for token, token_id in cls.BASE_TOKENS.items():
            if tokenizer.token2id.get(token) != token_id:
                raise ValueError("tokenizer reserved-token contract is invalid")
        tokenizer.id2token = {value: key for key, value in tokenizer.token2id.items()}
        tokenizer.merges = [tuple(pair) for pair in data.get("merges", [])]
        tokenizer._trained = bool(data.get("trained", True))
        tokenizer.corpus_sha256 = corpus_sha256
        return tokenizer


__all__ = ["JarvisTokenizer"]
