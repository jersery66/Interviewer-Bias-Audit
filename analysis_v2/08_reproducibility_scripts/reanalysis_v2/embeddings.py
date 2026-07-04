from __future__ import annotations

import numpy as np
import pandas as pd


def word_chunks(text: str, *, chunk_words: int = 200, overlap: int = 50) -> list[str]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_words")
    words = str(text or "").split()
    if not words:
        return []
    step = chunk_words - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


def mean_pool_document(
    text: str,
    encoder,
    *,
    dim: int,
    chunk_words: int = 200,
    overlap: int = 50,
) -> np.ndarray:
    chunks = word_chunks(text, chunk_words=chunk_words, overlap=overlap)
    if not chunks:
        return np.zeros(dim, dtype=np.float32)
    encoded = np.asarray(
        encoder.encode(
            chunks,
            show_progress_bar=False,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    if encoded.shape != (len(chunks), dim):
        raise ValueError(
            f"encoder returned shape {encoded.shape}, expected {(len(chunks), dim)}"
        )
    pooled = encoded.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return pooled.astype(np.float32, copy=False)


def embed_documents(
    texts: list[str] | np.ndarray,
    encoder,
    *,
    dim: int,
    chunk_words: int = 200,
    overlap: int = 50,
    batch_size: int = 32,
) -> tuple[np.ndarray, pd.DataFrame]:
    texts = [str(text or "") for text in texts]
    all_chunks: list[str] = []
    document_slices: list[tuple[int, int]] = []
    audit_rows = []
    for document_index, text in enumerate(texts):
        chunks = word_chunks(text, chunk_words=chunk_words, overlap=overlap)
        start = len(all_chunks)
        all_chunks.extend(chunks)
        document_slices.append((start, len(all_chunks)))
        audit_rows.append(
            {
                "document_index": document_index,
                "word_count": len(text.split()),
                "chunk_count": len(chunks),
                "is_empty": len(chunks) == 0,
            }
        )
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    if all_chunks:
        encoded = np.asarray(
            encoder.encode(
                all_chunks,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        if encoded.shape != (len(all_chunks), dim):
            raise ValueError(
                f"encoder returned shape {encoded.shape}, expected {(len(all_chunks), dim)}"
            )
        for document_index, (start, stop) in enumerate(document_slices):
            if stop == start:
                continue
            pooled = encoded[start:stop].mean(axis=0)
            norm = float(np.linalg.norm(pooled))
            if norm > 0:
                pooled = pooled / norm
            matrix[document_index] = pooled
    return matrix, pd.DataFrame(audit_rows)
