# Petabyte-Scale Synthetic Data Curation Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ray](https://img.shields.io/badge/Ray-2.10%2B-teal)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A distributed pipeline for generating, filtering, and deduplicating synthetic training data at petabyte scale via LLM orchestration and Ray.

## Pipeline Stages

```
Raw Documents (JSONL)
        │
        ▼
┌───────────────────┐
│  Quality Filtering │  ← 16 parallel Ray workers
│  - Length checks   │    Multi-stage, composable
│  - Symbol ratios   │    ~500K docs/min throughput
│  - Repeated lines  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Deduplication   │  ← 8 Ray actors, LSH index
│   MinHash LSH     │    128 permutations, 16 bands
│   Exact-match     │    ~70% Jaccard threshold
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Tokenization    │  ← 8 parallel workers
│   Sequence packing│    4096 token windows
│   BPE tokenizer   │    Zero padding waste
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Shard Output    │  ← .npy shards, 10K seqs each
│   10K seqs/shard  │    Reproducible, deterministic
└───────────────────┘
```

## MinHash LSH Configuration

The deduplication config balances precision and recall:

| Parameter | Value | Effect |
|---|---|---|
| `num_permutations` | 128 | Signature length — higher = better accuracy |
| `bands` | 16 | LSH bands |
| `rows_per_band` | 8 | Rows per band (16 × 8 = 128) |
| `jaccard_threshold` | ~0.7 | Theoretical: `(1/16)^(1/8) ≈ 0.72` |

False positive rate at threshold 0.7 is < 2% with this configuration.

## Quality Filter Pipeline

Filters are composable and applied in sequence, short-circuiting on first rejection:

```python
from data_pipeline.quality.filters import QualityFilterPipeline, FilterConfig

config = FilterConfig(
    min_chars=100,
    max_chars=100_000,
    max_symbol_to_word_ratio=0.2,
    max_repeated_line_fraction=0.3,
    blocked_patterns=["<script>", "onclick="],
)
pipeline = QualityFilterPipeline(config)
result = pipeline.filter(text, doc_id=42)
```

## Usage

```python
from data_pipeline.pipeline.orchestrator import DataCurationPipeline, PipelineConfig

config = PipelineConfig(
    input_glob="./raw/**/*.jsonl",
    output_dir="./processed",
    n_filter_workers=16,
    n_dedup_workers=8,
    tokenizer_name="meta-llama/Llama-3-8b-hf",
    max_seq_len=4096,
    pack_sequences=True,
)

pipeline = DataCurationPipeline(config)

# Stream (doc_id, text) pairs from your source
documents = ((i, line) for i, line in enumerate(open("data.jsonl")))
stats = pipeline.run(documents)
print(stats)
```

## Kubernetes Deployment

Worker pods scale automatically based on queue depth via HPA:

```bash
kubectl apply -f k8s/
kubectl get hpa data-pipeline-hpa  # Monitor autoscaler
```

## License

MIT
