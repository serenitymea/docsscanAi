"""
RAG system for document question answering
Based on architecture: ChromaDB + Voyage AI + Google Gemini
"""

from pathlib import Path
from typing import List, Dict, Optional
import chromadb

from embeddings import EmbeddingService
from llmser import LLMService
from docprocessor import DocumentProcessor


class RAGSystem:
    """Main RAG system"""
    
    def __init__(self, voyage_api_key: str, gemini_api_key: str, db_path: str = "./chroma_db"):
        if not voyage_api_key or not gemini_api_key:
            raise ValueError("Both API keys are required")
            
        try:
            self.embedding_service = EmbeddingService(voyage_api_key)
            self.llm_service = LLMService(gemini_api_key)
            self.doc_processor = DocumentProcessor()

            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"Database initialized: {db_path}")
            print(f"Documents in database: {self.collection.count()}")
            
        except Exception as e:
            raise Exception(f"RAG system initialization error: {str(e)}")
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split text into chunks with overlap"""
        if not text.strip():
            return []
            
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
            
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                
        return chunks
    
    def add_document(self, file_path: str, doc_name: Optional[str] = None) -> bool:
        """Add document to knowledge base"""
        try:
            print(f"Processing document: {file_path}")
            
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            text = self.doc_processor.extract_text_from_file(file_path)
            
            if not text.strip():
                raise ValueError("Document contains no text")

            chunks = self.chunk_text(text)
            if not chunks:
                raise ValueError("No valid chunks created from document")
                
            print(f"Split into {len(chunks)} chunks")

            doc_id = doc_name or Path(file_path).name
            embeddings = []
            chunk_ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.embedding_service.get_embedding(chunk)
                    embeddings.append(embedding.tolist())
                    chunk_ids.append(f"{doc_id}_{i}")
                    metadatas.append({
                        "document": doc_id,
                        "chunk_index": i,
                        "source": str(file_path)
                    })
                except Exception as e:
                    print(f"Error creating embedding for chunk {i}: {str(e)}")
                    continue
            
            if not embeddings:
                raise Exception("No embeddings could be created")

            self.collection.add(
                embeddings=embeddings,
                documents=chunks[:len(embeddings)],
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            print(f"Document added: {doc_id}")
            return True
            
        except Exception as e:
            print(f"Error adding document: {str(e)}")
            return False
    
    def search_relevant_chunks(self, question: str, n_results: int = 5) -> List[Dict]:
        """Search for most relevant text chunks"""
        try:
            if not question.strip():
                raise ValueError("Question cannot be empty")
                
            if self.collection.count() == 0:
                return []

            question_embedding = self.embedding_service.get_embedding(question)

            results = self.collection.query(
                query_embeddings=[question_embedding.tolist()],
                n_results=min(n_results, self.collection.count())
            )
            
            if not results['documents'][0]:
                return []
            
            chunks = []
            for i in range(len(results['documents'][0])):
                chunks.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i]
                })
            
            return chunks
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
    
    def ask_question(self, question: str, n_results: int = 3) -> Dict:
        """Answer question using RAG"""
        try:
            if not question.strip():
                raise ValueError("Question cannot be empty")
                
            print(f"Question: {question}")

            relevant_chunks = self.search_relevant_chunks(question, n_results)
            
            if not relevant_chunks:
                return {
                    'answer': 'Sorry, no relevant information found in the knowledge base.',
                    'sources': [],
                    'chunks_used': 0
                }

            context_parts = []
            sources = []
            
            for i, chunk in enumerate(relevant_chunks, 1):
                context_parts.append(f"Fragment {i}:\n{chunk['text']}")
                sources.append({
                    'document': chunk['metadata']['document'],
                    'similarity': round(chunk['similarity'], 3)
                })
            
            context = "\n\n".join(context_parts)
            
            print(f"Found {len(relevant_chunks)} relevant fragments")

            answer = self.llm_service.generate_answer(question, context)
            
            return {
                'answer': answer,
                'sources': sources,
                'chunks_used': len(relevant_chunks)
            }
            
        except Exception as e:
            print(f"Question processing error: {str(e)}")
            return {
                'answer': f'Error processing question: {str(e)}',
                'sources': [],
                'chunks_used': 0
            }
    
    def get_stats(self) -> Dict:
        """Return knowledge base statistics"""
        try:
            count = self.collection.count()
            documents = set()
            
            if count > 0:
                sample_size = min(1000, count)
                results = self.collection.get(limit=sample_size)
                documents = {meta['document'] for meta in results['metadatas'] if meta and 'document' in meta}
            
            return {
                'total_chunks': count,
                'total_documents': len(documents),
                'documents': list(documents)
            }
        except Exception as e:
            print(f"Stats error: {str(e)}")
            return {
                'total_chunks': 0,
                'total_documents': 0,
                'documents': []
            }