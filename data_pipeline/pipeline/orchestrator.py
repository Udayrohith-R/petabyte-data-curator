"""
Distributed data curation pipeline orchestrator.

Coordinates document ingestion, quality filtering, deduplication,
tokenization, and output serialization across a Ray cluster.
Kubernetes worker pods scale horizontally based on queue depth.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, Iterator, List, Optional, Tuple

import ray
from ray.util.queue import Queue as RayQueue

from data_pipeline.dedup.minhash import DistributedDeduplicator, MinHashConfig
from data_pipeline.quality.filters import FilterConfig, QualityFilterPipeline

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    # I/O
    input_glob: str = "./data/raw/**/*.jsonl"
    output_dir: str = "./data/processed"
    output_shard_size: int = 10_000

    # Parallelism
    n_filter_workers: int = 16
    n_dedup_workers: int = 8
    n_tokenizer_workers: int = 8
    batch_size: int = 512
    queue_maxsize: int = 4096

    # Quality
    filter: FilterConfig = field(default_factory=FilterConfig)

    # Dedup
    dedup: MinHashConfig = field(default_factory=MinHashConfig)

    # Tokenization
    tokenizer_name: str = "meta-llama/Llama-3-8b-hf"
    max_seq_len: int = 4096
    pack_sequences: bool = True


@ray.remote
class FilterWorker:
    """Ray actor for parallel quality filtering."""

    def __init__(self, config: FilterConfig) -> None:
        self.pipeline = QualityFilterPipeline(config)

    def process(self, batch: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        """Filter a batch, returning (doc_id, text) pairs that passed."""
        results = self.pipeline.filter_batch(
            [t for _, t in batch],
            doc_ids=[i for i, _ in batch],
        )
        return [(r.doc_id, r.text) for r in results if r.passed]

    def stats(self) -> Dict:
        return self.pipeline.stats()


@ray.remote
class TokenizationWorker:
    """
    Ray actor for distributed tokenization with sequence packing.

    Packs multiple short documents into fixed-length sequences to minimize
    padding waste — critical for training efficiency at scale.
    """

    def __init__(self, tokenizer_name: str, max_seq_len: int) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_len = max_seq_len
        self._buffer: List[int] = []
        self._packed: List[List[int]] = []

    def tokenize_batch(
        self, texts: List[str], pack: bool = True
    ) -> List[List[int]]:
        """Tokenize and optionally pack sequences."""
        all_ids: List[List[int]] = []

        for text in texts:
            ids = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_len,
            )
            if pack:
                self._buffer.extend(ids + [self.tokenizer.eos_token_id])
                while len(self._buffer) >= self.max_seq_len:
                    all_ids.append(self._buffer[:self.max_seq_len])
                    self._buffer = self._buffer[self.max_seq_len:]
            else:
                all_ids.append(ids)

        return all_ids

    def flush(self) -> List[List[int]]:
        """Flush remaining buffered tokens, padding to max_seq_len."""
        if not self._buffer:
            return []
        padded = self._buffer + [self.tokenizer.pad_token_id] * (
            self.max_seq_len - len(self._buffer)
        )
        self._buffer = []
        return [padded[:self.max_seq_len]]


@ray.remote
class ShardWriter:
    """Writes processed token sequences to sharded output files."""

    def __init__(self, output_dir: str, shard_size: int) -> None:
        import numpy as np
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self._current_shard: List[List[int]] = []
        self._shard_idx = 0
        self._total_sequences = 0

    def write(self, sequences: List[List[int]]) -> int:
        """Buffer sequences and flush when shard is full. Returns sequences written."""
        self._current_shard.extend(sequences)
        flushed = 0

        while len(self._current_shard) >= self.shard_size:
            self._flush_shard(self._current_shard[:self.shard_size])
            self._current_shard = self._current_shard[self.shard_size:]
            flushed += self.shard_size

        return flushed

    def finalize(self) -> Dict:
        """Flush remaining sequences and return summary stats."""
        if self._current_shard:
            self._flush_shard(self._current_shard)
            self._current_shard = []

        return {
            "total_shards": self._shard_idx,
            "total_sequences": self._total_sequences,
            "output_dir": str(self.output_dir),
        }

    def _flush_shard(self, sequences: List[List[int]]) -> None:
        import numpy as np
        arr = np.array(sequences, dtype=np.int32)
        shard_path = self.output_dir / f"shard_{self._shard_idx:06d}.npy"
        np.save(str(shard_path), arr)
        self._shard_idx += 1
        self._total_sequences += len(sequences)
        logger.debug(f"Flushed shard {self._shard_idx - 1}: {len(sequences)} sequences -> {shard_path}")


class DataCurationPipeline:
    """
    End-to-end distributed data curation pipeline.

    Stages:
        1. Ingest raw documents (streaming, memory-efficient)
        2. Quality filter (parallel, Ray workers)
        3. Deduplication (distributed MinHash LSH)
        4. Tokenization + sequence packing (parallel, Ray workers)
        5. Shard output to .npy files

    Designed for petabyte-scale datasets via horizontal Ray worker scaling.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self._filter_workers = [
            FilterWorker.remote(config.filter)
            for _ in range(config.n_filter_workers)
        ]
        self._deduplicator = DistributedDeduplicator(
            config=config.dedup,
            n_workers=config.n_dedup_workers,
        )
        self._tok_workers = [
            TokenizationWorker.remote(config.tokenizer_name, config.max_seq_len)
            for _ in range(config.n_tokenizer_workers)
        ]
        self._writer = ShardWriter.remote(config.output_dir, config.output_shard_size)

        logger.info(
            f"Pipeline initialized | filter_workers={config.n_filter_workers} "
            f"| dedup_workers={config.n_dedup_workers} "
            f"| tok_workers={config.n_tokenizer_workers}"
        )

    def run(self, documents: Iterable[Tuple[int, str]]) -> Dict:
        """
        Execute the full pipeline over a stream of (doc_id, text) pairs.
        Returns summary statistics.
        """
        t0 = time.perf_counter()
        stats: Dict = {}

        # Stage 1: Quality filtering (round-robin batching)
        batches = self._batch_iterator(documents, self.config.batch_size)
        filter_futures = []
        worker_idx = 0

        filtered_docs: List[Tuple[int, str]] = []
        for batch in batches:
            worker = self._filter_workers[worker_idx % len(self._filter_workers)]
            filter_futures.append(worker.process.remote(batch))
            worker_idx += 1

            if len(filter_futures) >= self.config.n_filter_workers * 2:
                results = ray.get(filter_futures[:self.config.n_filter_workers])
                for r in results:
                    filtered_docs.extend(r)
                filter_futures = filter_futures[self.config.n_filter_workers:]

        for r in ray.get(filter_futures):
            filtered_docs.extend(r)

        filter_stats = ray.get([w.stats.remote() for w in self._filter_workers])
        stats["filter"] = {
            "total_input": sum(s["total"] for s in filter_stats),
            "total_passed": sum(s["passed"] for s in filter_stats),
            "pass_rate": sum(s["passed"] for s in filter_stats) / max(
                sum(s["total"] for s in filter_stats), 1
            ),
        }
        logger.info(f"Stage 1 complete: {stats['filter']['total_passed']} docs passed filtering")

        # Stage 2: Deduplication
        unique_ids, dup_ids = self._deduplicator.deduplicate(iter(filtered_docs))
        unique_id_set = set(unique_ids)
        unique_docs = [(doc_id, text) for doc_id, text in filtered_docs if doc_id in unique_id_set]

        stats["dedup"] = {
            "input": len(filtered_docs),
            "unique": len(unique_docs),
            "duplicates": len(dup_ids),
            "dedup_rate": len(dup_ids) / max(len(filtered_docs), 1),
        }
        logger.info(
            f"Stage 2 complete: {len(unique_docs)} unique docs "
            f"({100*stats['dedup']['dedup_rate']:.1f}% duplicates removed)"
        )

        # Stage 3: Tokenization
        tok_futures = []
        tok_worker_idx = 0
        texts = [text for _, text in unique_docs]

        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i: i + self.config.batch_size]
            worker = self._tok_workers[tok_worker_idx % len(self._tok_workers)]
            tok_futures.append(
                worker.tokenize_batch.remote(batch_texts, self.config.pack_sequences)
            )
            tok_worker_idx += 1

        total_sequences = 0
        write_futures = []
        for sequences in ray.get(tok_futures):
            if sequences:
                write_futures.append(self._writer.write.remote(sequences))

        for n in ray.get(write_futures):
            total_sequences += n

        # Flush remaining packed sequences
        for worker in self._tok_workers:
            leftover = ray.get(worker.flush.remote())
            if leftover:
                ray.get(self._writer.write.remote(leftover))

        writer_stats = ray.get(self._writer.finalize.remote())
        stats["output"] = writer_stats

        elapsed = time.perf_counter() - t0
        stats["elapsed_seconds"] = elapsed
        stats["docs_per_second"] = len(unique_docs) / elapsed

        logger.info(
            f"Pipeline complete in {elapsed:.1f}s | "
            f"{stats['output']['total_sequences']} sequences written to "
            f"{stats['output']['total_shards']} shards"
        )

        return stats

    @staticmethod
    def _batch_iterator(
        iterable: Iterable, batch_size: int
    ) -> Iterator[List]:
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
