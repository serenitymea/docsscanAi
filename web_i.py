import os
import time
import argparse
import tempfile
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from rag_system import RAGSystem


ALLOWED_EXTENSIONS = {
    ".txt", ".pdf", ".docx",
    ".html", ".htm",
    ".csv",
    ".xlsx", ".xls",
    ".json"
}


class QuestionRequest(BaseModel):
    question: str
    n_results: int = 3


class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict]
    processing_time: float


class DocumentResponse(BaseModel):
    message: str
    document_name: str


class StatsResponse(BaseModel):
    chunks: int
    documents: int
    doc_list: List[str] = []


def load_template(path: str) -> str:
    """Load HTML template from file."""
    try:
        if not os.path.exists(path):
            print(f"Template not found: {path}")
            return "<h1>Template not found</h1><p>Please check the template path</p>"
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error loading template: {e}")
        return f"<h1>Error loading template</h1><p>{str(e)}</p>"


def create_app(voyage_key: str, gemini_key: str, db_path: str, template_path: str) -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(title="RAG Document QA")

    print("Initializing RAG system...")
    rag = RAGSystem(voyage_key, gemini_key, db_path)
    print("RAG system initialized successfully")

    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")
        print(f"Static files mounted from: {static_dir.absolute()}")
    else:
        print(f"WARNING: Static directory not found: {static_dir.absolute()}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def home():
        """Serve main HTML page."""
        return load_template(template_path)

    @app.post("/api/upload", response_model=DocumentResponse)
    async def upload(
        file: UploadFile = File(...),
        document_name: Optional[str] = Form(None),
    ):
        """Upload and process a document."""
        print(f"\n{'='*60}")
        print(f"Upload request received")
        print(f"Filename: {file.filename}")
        print(f"Content type: {file.content_type}")
        
        ext = Path(file.filename).suffix.lower()
        print(f"Extension: {ext}")

        if ext not in ALLOWED_EXTENSIONS:
            print(f"Unsupported extension: {ext}")
            raise HTTPException(400, f"Unsupported file format: {ext}")

        try:
            content = await file.read()
            print(f"File size: {len(content)} bytes")
        except Exception as e:
            print(f"Error reading file: {e}")
            raise HTTPException(500, f"Error reading file: {str(e)}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            print(f"Temp file: {tmp_path}")

        try:
            name = document_name or file.filename
            print(f"Processing document: {name}")

            rag.add_document(tmp_path, name)
            
            print(f"Document added successfully: {name}")
            print(f"{'='*60}\n")
            
            return DocumentResponse(
                message="Document added successfully",
                document_name=name,
            )
            
        except Exception as e:
            print(f"Error processing document: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(500, f"Error processing document: {str(e)}")
            
        finally:
            try:
                os.unlink(tmp_path)
                print(f"Temp file deleted: {tmp_path}")
            except Exception as e:
                print(f"Warning: Could not delete temp file: {e}")

    @app.post("/api/ask", response_model=QuestionResponse)
    async def ask(req: QuestionRequest):
        """Answer a question using the RAG system."""
        print(f"\n{'='*60}")
        print(f"Question: {req.question}")
        
        if not req.question.strip():
            raise HTTPException(400, "Question cannot be empty")

        start = time.time()
        
        try:
            result = rag.ask(req.question, req.n_results)
            elapsed = round(time.time() - start, 2)
            
            print(f"Answer generated in {elapsed}s")
            print(f"Sources: {len(result['sources'])}")
            print(f"{'='*60}\n")

            return QuestionResponse(
                answer=result["answer"],
                sources=result["sources"],
                processing_time=elapsed,
            )
            
        except Exception as e:
            print(f"Error processing question: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(500, f"Error: {str(e)}")

    @app.get("/api/stats", response_model=StatsResponse)
    async def stats():
        """Get knowledge base statistics."""
        try:
            stats_data = rag.stats()
            print(f"Stats: {stats_data}")

            if "doc_list" not in stats_data:
                stats_data["doc_list"] = []
            
            return StatsResponse(**stats_data)
        except Exception as e:
            print(f"Error getting stats: {e}")
            import traceback
            traceback.print_exc()
            return StatsResponse(chunks=0, documents=0, doc_list=[])

    @app.delete("/api/clear")
    async def clear():
        """Clear the knowledge base."""
        try:
            print("Clearing knowledge base...")
            rag.client.delete_collection("documents")
            rag.collection = rag.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
            print("Knowledge base cleared")
            return {"message": "Knowledge base cleared"}
        except Exception as e:
            print(f"Error clearing database: {e}")
            raise HTTPException(500, f"Error: {str(e)}")

    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Web Server")
    parser.add_argument("--voyage-key", required=True, help="Voyage API key")
    parser.add_argument("--gemini-key", required=True, help="Gemini API key")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB path")
    parser.add_argument("--template", default="./templates/index.html", help="HTML template path")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("RAG System Web Server")
    print("="*60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Template: {args.template}")
    print(f"Database: {args.db_path}")
    print("="*60 + "\n")

    if not Path("static").exists():
        print("WARNING: 'static' directory not found!")
        print("Please create the following structure:")
        print("  static/css/style.css")
        print("  static/js/script.js")
        print()
    
    if not Path("templates").exists():
        print("WARNING: 'templates' directory not found!")
        print("Please create: templates/index.html")
        print()

    uvicorn.run(
        create_app(args.voyage_key, args.gemini_key, args.db_path, args.template),
        host=args.host,
        port=args.port,
    )

if __name__ == "__main__":
    main()