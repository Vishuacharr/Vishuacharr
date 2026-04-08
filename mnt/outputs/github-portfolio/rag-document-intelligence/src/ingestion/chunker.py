"""
Semantic and recursive chunking for document ingestion.
Supports multiple strategies: semantic, recursive, fixed-size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    strategy: str = "semantic"          # "semantic" | "recursive" | "fixed"
    similarity_threshold: float = 0.75  # for semantic splitting
    min_chunk_size: int = 100
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


class DocumentChunker:
    """
    Intelligent document chunker combining semantic similarity and
    recursive character splitting for optimal retrieval granularity.
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self._embed_model: Optional[SentenceTransformer] = None

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk a list of LangChain Documents using the configured strategy."""
        if self.config.strategy == "semantic":
            return self._semantic_chunk(documents)
        elif self.config.strategy == "recursive":
            return self._recursive_chunk(documents)
        else:
            return self._fixed_chunk(documents)

    # ------------------------------------------------------------------
    # Chunking strategies
    # ------------------------------------------------------------------

    def _semantic_chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split on sentence boundaries and merge until cosine similarity
        between adjacent groups drops below threshold.
        """
        chunks: List[Document] = []
        for doc in documents:
            sentences = self._split_into_sentences(doc.page_content)
            if not sentences:
                continue

            embeddings = self.embed_model.encode(sentences, show_progress_bar=False)
            groups: List[List[str]] = [[sentences[0]]]

            for i, sent in enumerate(sentences[1:], start=1):
                sim = self._cosine_sim(embeddings[i - 1], embeddings[i])
                if sim >= self.config.similarity_threshold:
                    groups[-1].append(sent)
                else:
                    groups.append([sent])

            for group in groups:
                text = " ".join(group).strip()
                if len(text) >= self.config.min_chunk_size:
                    chunks.append(Document(page_content=text, metadata=doc.metadata.copy()))

        return chunks

    def _recursive_chunk(self, documents: List[Document]) -> List[Document]:
        """LangChain recursive character text splitting."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )
        return splitter.split_documents(documents)

    def _fixed_chunk(self, documents: List[Document]) -> List[Document]:
        """Simple fixed-size chunking with overlap."""
        chunks: List[Document] = []
        for doc in documents:
            text = doc.page_content
            step = self.config.chunk_size - self.config.chunk_overlap
            for start in range(0, len(text), step):
                piece = text[start: start + self.config.chunk_size].strip()
                if len(piece) >= self.config.min_chunk_size:
                    chunks.append(Document(page_content=piece, metadata=doc.metadata.copy()))
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
