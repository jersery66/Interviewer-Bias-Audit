from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reanalysis_v2.embedding_runner import (
    EmbeddingModelConfig,
    run_embedding_model_analysis,
)


MODELS = {
    "mpnet": EmbeddingModelConfig(
        name="sentence-transformers/all-mpnet-base-v2",
        revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
        slug="model_1_all_mpnet_base_v2",
        chunk_words=200,
        overlap=50,
        batch_size=64,
    ),
    "bge": EmbeddingModelConfig(
        name="BAAI/bge-large-en-v1.5",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        slug="model_2_bge_large_en_v1_5",
        chunk_words=200,
        overlap=50,
        batch_size=32,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen dense embedding model")
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--output", type=Path, default=Path("analysis_v2"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()
    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    manifest = run_embedding_model_analysis(
        args.output,
        MODELS[args.model],
        device=device,
        n_permutations=args.permutations,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "model": manifest["model"]["name"],
                "revision": manifest["model"]["revision"],
                "device": manifest["model"]["device"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
