from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import chromadb

from embeddings import EmbeddingService
from llmser import LLMService
from docprocessor import DocumentProcessor


@dataclass
class ChunkResult:
    text: str
    document: str
    similarity: float


class RAGSystem:
    """Minimal production-ready RAG pipeline."""

    def __init__(
        self,
        voyage_api_key: str,
        gemini_api_key: str,
        db_path: str = "./chroma_db",
    ):
        self.embedder = EmbeddingService(voyage_api_key)
        self.llm = LLMService(gemini_api_key)
        self.docs = DocumentProcessor()

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )


    @staticmethod
    def chunk_text(text: str, size: int = 800, overlap: int = 150) -> List[str]:
        words = text.split()
        step = size - overlap

        return [
            " ".join(words[i:i + size])
            for i in range(0, len(words), step)
            if words[i:i + size]
        ]


    def add_document(self, file_path: str, name: Optional[str] = None) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)

        doc_id = name or path.name
        text = self.docs.extract_text(path)

        chunks = self.chunk_text(text)
        if not chunks:
            raise ValueError("Document is empty")

        embeddings = [self.embedder.get_embedding(c).tolist() for c in chunks]

        self.collection.add(
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"document": doc_id, "chunk": i, "source": str(path)}
                for i in range(len(chunks))
            ],
        )


    def retrieve(self, question: str, k: int = 5) -> List[ChunkResult]:
        if self.collection.count() == 0:
            return []

        q_emb = self.embedder.get_embedding(question).tolist()

        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(k, self.collection.count()),
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            ChunkResult(
                text=docs[i],
                document=metas[i]["document"],
                similarity=round(1 - distances[i], 3),
            )
            for i in range(len(docs))
        ]


    def ask(self, question: str, k: int = 3) -> Dict:
        chunks = self.retrieve(question, k)

        if not chunks:
            return {"answer": "No relevant information found.", "sources": []}

        context = "\n\n".join(
            f"Fragment {i+1}:\n{c.text}"
            for i, c in enumerate(chunks)
        )

        answer = self.llm.generate_answer(question, context)

        return {
            "answer": answer,
            "sources": [
                {"document": c.document, "similarity": c.similarity}
                for c in chunks
            ],
        }


    def stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        count = self.collection.count()

        if count == 0:
            return {
                "chunks": 0,
                "documents": 0,
                "doc_list": []
            }

        try:
            sample = self.collection.get(limit=min(1000, count))
            
            if not sample or not sample.get("metadatas"):
                return {
                    "chunks": count,
                    "documents": 0,
                    "doc_list": []
                }
            
            docs = {m["document"] for m in sample["metadatas"] if m and "document" in m}
            
            return {
                "chunks": count,
                "documents": len(docs),
                "doc_list": sorted(docs),
            }
        except Exception as e:
            print(f"Error in stats: {e}")
            return {
                "chunks": count,
                "documents": 0,
                "doc_list": []
            }