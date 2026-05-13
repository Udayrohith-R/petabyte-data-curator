"""
Quality filtering for synthetic and web-scraped training data.

Implements a multi-stage filtering pipeline with configurable rules.
Each filter is stateless and composable — filters are applied in sequence.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    # Length filters
    min_chars: int = 100
    max_chars: int = 100_000
    min_words: int = 10
    max_words: int = 50_000

    # Quality heuristics
    min_avg_word_length: float = 3.0
    max_avg_word_length: float = 15.0
    max_symbol_to_word_ratio: float = 0.2
    max_digit_to_char_ratio: float = 0.4
    max_repeated_line_fraction: float = 0.3
    max_uppercase_fraction: float = 0.6

    # Language
    allowed_languages: Optional[List[str]] = None   # None = no lang filter
    min_language_score: float = 0.8

    # Content
    blocked_patterns: List[str] = field(default_factory=list)
    min_unique_word_fraction: float = 0.2

    # Perplexity (requires a KenLM model)
    max_perplexity: Optional[float] = None
    kenlm_model_path: Optional[str] = None


@dataclass
class FilterResult:
    passed: bool
    text: str
    doc_id: Optional[int] = None
    rejection_reason: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)


class BaseFilter(ABC):
    """Abstract base for composable document filters."""

    @abstractmethod
    def apply(self, text: str) -> Tuple[bool, Optional[str]]:
        """Return (passed, rejection_reason)."""
        ...

    def __call__(self, text: str) -> Tuple[bool, Optional[str]]:
        return self.apply(text)


class LengthFilter(BaseFilter):
    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def apply(self, text: str) -> Tuple[bool, Optional[str]]:
        n_chars = len(text)
        if n_chars < self.config.min_chars:
            return False, f"too_short: {n_chars} chars < {self.config.min_chars}"
        if n_chars > self.config.max_chars:
            return False, f"too_long: {n_chars} chars > {self.config.max_chars}"

        words = text.split()
        n_words = len(words)
        if n_words < self.config.min_words:
            return False, f"too_few_words: {n_words} < {self.config.min_words}"
        if n_words > self.config.max_words:
            return False, f"too_many_words: {n_words} > {self.config.max_words}"

        return True, None


class SymbolRatioFilter(BaseFilter):
    """Reject documents with excessive non-alphabetic symbols."""

    _SYMBOL_RE = re.compile(r"[^\w\s]")

    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def apply(self, text: str) -> Tuple[bool, Optional[str]]:
        words = text.split()
        if not words:
            return False, "empty_document"

        n_symbols = len(self._SYMBOL_RE.findall(text))
        ratio = n_symbols / max(len(words), 1)

        if ratio > self.config.max_symbol_to_word_ratio:
            return False, f"high_symbol_ratio: {ratio:.3f} > {self.config.max_symbol_to_word_ratio}"

        digits = sum(c.isdigit() for c in text)
        digit_ratio = digits / max(len(text), 1)
        if digit_ratio > self.config.max_digit_to_char_ratio:
            return False, f"high_digit_ratio: {digit_ratio:.3f}"

        return True, None


class WordLengthFilter(BaseFilter):
    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def apply(self, text: str) -> Tuple[bool, Optional[str]]:
        words = [w for w in text.split() if w.isalpha()]
        if not words:
            return False, "no_alpha_words"

        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len < self.config.min_avg_word_length:
            return False, f"avg_word_too_short: {avg_len:.2f}"
        if avg_len > self.config.max_avg_word_length:
            return False, f"avg_word_too_long: {avg_len:.2f}"

        return True, None


class RepeatedLineFilter(BaseFilter):
    """Reject documents with high fraction of repeated lines — common in scraped boilerplate."""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def apply(self, text: str) -> Tuple[bool, Optional[str]]:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 5:
            return True, None

        unique_lines = set(lines)
        repeated_fraction = 1.0 - (len(unique_lines) / len(lines))

        if repeated_fraction > self.config.max_repeated_line_fraction:
            return False, f"high_repeated_lines: {repeated_fraction:.3f}"

        return True, None


class UniqueWordFilter(BaseFilter):
    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def apply(self, text: str) -> Tuple[bool, Optional[str]]:
        words = text.lower().split()
        if not words:
            return False, "empty"
        unique_fraction = len(set(words)) / len(words)
        if unique_fraction < self.config.min_unique_word_fraction:
            return False, f"low_vocabulary: {unique_fraction:.3f}"
        return True, None


class BlockedPatternFilter(BaseFilter):
    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self._patterns = [re.compile(p, re.IGNORECASE) for p in config.blocked_patterns]

    def apply(self, text: str) -> Tuple[bool, Optional[str]]:
        for pattern in self._patterns:
            if pattern.search(text):
                return False, f"blocked_pattern: {pattern.pattern}"
        return True, None


class NormalizationFilter(BaseFilter):
    """Unicode normalization and whitespace cleanup. Always passes — just cleans text."""

    def apply(self, text: str) -> Tuple[bool, Optional[str]]:
        return True, None

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class QualityFilterPipeline:
    """
    Composable multi-stage quality filter pipeline.

    Applies filters in order, short-circuiting on first rejection.
    Tracks rejection statistics per filter stage.
    """

    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self._normalizer = NormalizationFilter()
        self._filters: List[BaseFilter] = self._build_filters()
        self._stats: Dict[str, int] = {"passed": 0, "rejected": 0}
        self._rejection_counts: Dict[str, int] = {}

    def _build_filters(self) -> List[BaseFilter]:
        filters: List[BaseFilter] = [
            LengthFilter(self.config),
            SymbolRatioFilter(self.config),
            WordLengthFilter(self.config),
            RepeatedLineFilter(self.config),
            UniqueWordFilter(self.config),
        ]
        if self.config.blocked_patterns:
            filters.append(BlockedPatternFilter(self.config))
        return filters

    def filter(self, text: str, doc_id: Optional[int] = None) -> FilterResult:
        """Apply all filters and return a FilterResult."""
        text = self._normalizer.normalize(text)

        for f in self._filters:
            passed, reason = f.apply(text)
            if not passed:
                self._stats["rejected"] += 1
                stage = type(f).__name__
                self._rejection_counts[stage] = self._rejection_counts.get(stage, 0) + 1
                return FilterResult(passed=False, text=text, doc_id=doc_id, rejection_reason=reason)

        self._stats["passed"] += 1
        return FilterResult(passed=True, text=text, doc_id=doc_id)

    def filter_batch(
        self, texts: List[str], doc_ids: Optional[List[int]] = None
    ) -> List[FilterResult]:
        ids = doc_ids or [None] * len(texts)
        return [self.filter(t, d) for t, d in zip(texts, ids)]

    def stats(self) -> Dict:
        total = self._stats["passed"] + self._stats["rejected"]
        return {
            "total": total,
            "passed": self._stats["passed"],
            "rejected": self._stats["rejected"],
            "pass_rate": self._stats["passed"] / max(total, 1),
            "rejection_by_stage": self._rejection_counts,
        }

    def reset_stats(self) -> None:
        self._stats = {"passed": 0, "rejected": 0}
        self._rejection_counts = {}
