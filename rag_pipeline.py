from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.logger import logging
from utils import (
    DEFAULT_CACHE_DIR,
    get_cache_path,
    hash_texts,
    load_cached_data,
    save_cached_data,
)


@dataclass
class RetrievalResult:
    retrieved_chunks: List[str]
    retrieved_indices: List[int]
    similarity_scores: List[float]


class RAGPipeline:
    """
    Encodes document chunks, stores them in FAISS, and retrieves relevant context.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = str(DEFAULT_CACHE_DIR / "embeddings"),
    ):
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks: List[str] = []
        self.embeddings = None
        logging.info("Loaded embedding model: %s", embedding_model_name)

    def _cache_key(self, chunks: List[str]) -> str:
        chunk_hash = hash_texts(chunks, prefix=self.embedding_model_name)
        return f"embeddings_{chunk_hash}.pkl"

    def build_index(self, chunks: List[str]):
        """
        Create or load chunk embeddings, then build a FAISS similarity index.
        """
        if not chunks:
            raise ValueError("Cannot build a vector store from empty chunks.")

        self.chunks = chunks
        cache_path = get_cache_path(self.cache_dir, self._cache_key(chunks))
        cached_payload = load_cached_data(cache_path)

        if cached_payload is not None:
            embeddings = cached_payload["embeddings"]
            logging.info("Loaded chunk embeddings from cache: %s", cache_path)
        else:
            embeddings = self.embedding_model.encode(
                chunks,
                batch_size=16,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            save_cached_data(cache_path, {"embeddings": embeddings, "chunks": chunks})
            logging.info("Saved chunk embeddings to cache: %s", cache_path)

        self.embeddings = embeddings
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info("Built FAISS index with %s chunks", len(chunks))

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """
        Retrieve the most relevant chunks for a query.
        """
        if self.index is None:
            raise ValueError("The FAISS index has not been built yet.")

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        retrieved_indices = indices[0].tolist()
        similarity_scores = scores[0].tolist()
        retrieved_chunks = [self.chunks[index] for index in retrieved_indices if index >= 0]

        return RetrievalResult(
            retrieved_chunks=retrieved_chunks,
            retrieved_indices=[index for index in retrieved_indices if index >= 0],
            similarity_scores=similarity_scores[: len(retrieved_chunks)],
        )

