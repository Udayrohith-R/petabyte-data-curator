"""
MinHash LSH deduplication for large-scale text datasets.

Implements exact-match and near-duplicate detection using MinHash signatures
and Locality-Sensitive Hashing. Designed for petabyte-scale datasets via
Ray distributed processing.

References:
    - Broder et al. (1997): On the Resemblance and Containment of Documents
    - Lee et al. (2022): Deduplicating Training Data Makes Language Models Better
"""

from __future__ import annotations

import hashlib
import logging
import struct
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np
import ray

logger = logging.getLogger(__name__)

# Mersenne prime for hash function family
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


@dataclass
class MinHashConfig:
    num_permutations: int = 128       # Signature length — higher = better precision
    ngram_size: int = 5               # Character n-gram size
    bands: int = 16                   # LSH bands (b * r = num_permutations)
    rows_per_band: int = 8            # Rows per band (r)
    jaccard_threshold: float = 0.7    # Similarity threshold for near-duplicates
    seed: int = 42

    def __post_init__(self) -> None:
        assert self.bands * self.rows_per_band == self.num_permutations, (
            f"bands ({self.bands}) * rows_per_band ({self.rows_per_band}) "
            f"must equal num_permutations ({self.num_permutations})"
        )
        # Theoretical threshold: (1/b)^(1/r)
        theoretical_threshold = (1.0 / self.bands) ** (1.0 / self.rows_per_band)
        if abs(theoretical_threshold - self.jaccard_threshold) > 0.15:
            logger.warning(
                f"Band/row config gives theoretical threshold {theoretical_threshold:.3f}, "
                f"but jaccard_threshold={self.jaccard_threshold}"
            )


class MinHashSignature:
    """
    Compute MinHash signatures for text documents.

    Uses a family of hash functions h(x) = (ax + b) % p to simulate
    min-wise independent permutations efficiently.
    """

    def __init__(self, config: MinHashConfig) -> None:
        self.config = config
        rng = np.random.RandomState(config.seed)
        self._a = rng.randint(1, _MERSENNE_PRIME, size=config.num_permutations, dtype=np.int64)
        self._b = rng.randint(0, _MERSENNE_PRIME, size=config.num_permutations, dtype=np.int64)

    def signature(self, text: str) -> np.ndarray:
        """Compute MinHash signature as array of uint32."""
        shingles = self._shingle(text, self.config.ngram_size)
        if not shingles:
            return np.full(self.config.num_permutations, _MAX_HASH, dtype=np.uint32)

        shingle_hashes = np.array(
            [self._hash_shingle(s) for s in shingles], dtype=np.int64
        )

        # Vectorized: compute min over all shingles for each hash function
        # shape: [num_permutations, num_shingles]
        hashed = (
            np.outer(self._a, shingle_hashes) + self._b[:, None]
        ) % _MERSENNE_PRIME

        return hashed.min(axis=1).astype(np.uint32)

    def lsh_bands(self, sig: np.ndarray) -> List[bytes]:
        """
        Split signature into bands and hash each band.
        Documents sharing any band hash are candidate duplicates.
        """
        bands = []
        for i in range(self.config.bands):
            start = i * self.config.rows_per_band
            end = start + self.config.rows_per_band
            band_sig = sig[start:end].tobytes()
            band_hash = hashlib.blake2b(band_sig, digest_size=8).digest()
            bands.append(band_hash)
        return bands

    @staticmethod
    def _shingle(text: str, n: int) -> FrozenSet[str]:
        """Character-level n-gram shingling."""
        text = text.lower()
        if len(text) < n:
            return frozenset([text])
        return frozenset(text[i: i + n] for i in range(len(text) - n + 1))

    @staticmethod
    def _hash_shingle(shingle: str) -> int:
        digest = hashlib.md5(shingle.encode("utf-8")).digest()
        return struct.unpack("<Q", digest[:8])[0] % _MERSENNE_PRIME


class LSHIndex:
    """
    In-memory LSH index for candidate pair retrieval.

    Each band maps band_hash -> set of document IDs.
    Documents sharing a band hash are candidate near-duplicates.
    """

    def __init__(self, n_bands: int) -> None:
        self.n_bands = n_bands
        self._buckets: List[Dict[bytes, Set[int]]] = [
            {} for _ in range(n_bands)
        ]
        self._n_docs = 0

    def insert(self, doc_id: int, band_hashes: List[bytes]) -> List[int]:
        """Insert a document and return candidate duplicate IDs."""
        assert len(band_hashes) == self.n_bands
        candidates: Set[int] = set()

        for band_idx, bh in enumerate(band_hashes):
            bucket = self._buckets[band_idx]
            if bh in bucket:
                candidates.update(bucket[bh])
            else:
                bucket[bh] = set()
            bucket[bh].add(doc_id)

        self._n_docs += 1
        return list(candidates)

    def __len__(self) -> int:
        return self._n_docs


def jaccard_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    return float((sig_a == sig_b).mean())


@ray.remote
class DeduplicationWorker:
    """
    Ray actor that maintains a partition of the LSH index and processes
    document batches concurrently with other workers.
    """

    def __init__(self, worker_id: int, config: MinHashConfig) -> None:
        self.worker_id = worker_id
        self.config = config
        self.hasher = MinHashSignature(config)
        self.index = LSHIndex(config.bands)
        self._seen: Set[int] = set()
        self._duplicate_count = 0

    def process_batch(
        self, doc_ids: List[int], texts: List[str]
    ) -> Tuple[List[int], List[int]]:
        """
        Process a batch of documents.

        Returns:
            unique_ids: document IDs determined to be unique
            duplicate_ids: document IDs that are near-duplicates
        """
        unique_ids = []
        duplicate_ids = []

        for doc_id, text in zip(doc_ids, texts):
            sig = self.hasher.signature(text)
            bands = self.hasher.lsh_bands(sig)
            candidates = self.index.insert(doc_id, bands)

            is_duplicate = False
            for cand_id in candidates:
                if cand_id in self._seen:
                    is_duplicate = True
                    break

            if is_duplicate:
                duplicate_ids.append(doc_id)
                self._duplicate_count += 1
            else:
                unique_ids.append(doc_id)
                self._seen.add(doc_id)

        return unique_ids, duplicate_ids

    def stats(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "n_docs": len(self.index),
            "n_duplicates": self._duplicate_count,
            "dedup_rate": self._duplicate_count / max(len(self.index), 1),
        }


class DistributedDeduplicator:
    """
    Orchestrates distributed deduplication across multiple Ray workers.

    Partitions documents by a hash of their content to ensure documents
    that could be duplicates are routed to the same worker.
    """

    def __init__(self, config: MinHashConfig, n_workers: int = 8) -> None:
        self.config = config
        self.n_workers = n_workers

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self._workers = [
            DeduplicationWorker.remote(i, config) for i in range(n_workers)
        ]
        logger.info(f"DistributedDeduplicator: {n_workers} workers initialized")

    def deduplicate(
        self,
        documents: Iterable[Tuple[int, str]],
        batch_size: int = 1024,
    ) -> Tuple[List[int], List[int]]:
        """
        Deduplicate a stream of (doc_id, text) pairs.

        Returns:
            all_unique: list of unique document IDs
            all_duplicates: list of duplicate document IDs
        """
        all_unique: List[int] = []
        all_duplicates: List[int] = []

        batches: List[List[Tuple[int, str]]] = []
        current_batch: List[Tuple[int, str]] = []

        for item in documents:
            current_batch.append(item)
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        futures = []
        for batch in batches:
            # Route to worker based on content hash for locality
            worker_batches: List[Tuple[List[int], List[str]]] = [
                ([], []) for _ in range(self.n_workers)
            ]
            for doc_id, text in batch:
                worker_idx = int(hashlib.md5(text[:64].encode()).hexdigest(), 16) % self.n_workers
                worker_batches[worker_idx][0].append(doc_id)
                worker_batches[worker_idx][1].append(text)

            for i, (ids, texts) in enumerate(worker_batches):
                if ids:
                    futures.append(self._workers[i].process_batch.remote(ids, texts))

        results = ray.get(futures)
        for unique_ids, dup_ids in results:
            all_unique.extend(unique_ids)
            all_duplicates.extend(dup_ids)

        total = len(all_unique) + len(all_duplicates)
        if total > 0:
            logger.info(
                f"Deduplication complete | total={total} "
                f"| unique={len(all_unique)} ({100*len(all_unique)/total:.1f}%) "
                f"| duplicates={len(all_duplicates)} ({100*len(all_duplicates)/total:.1f}%)"
            )

        return all_unique, all_duplicates

    def aggregate_stats(self) -> Dict:
        return ray.get([w.stats.remote() for w in self._workers])
