import os
import time
import sys
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from rag_system import RAGSystem


class QuestionRequest(BaseModel):
    question: str
    n_results: Optional[int] = 3


class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict]
    chunks_used: int
    processing_time: float


class DocumentResponse(BaseModel):
    success: bool
    message: str
    document_name: Optional[str] = None


class StatsResponse(BaseModel):
    total_chunks: int
    total_documents: int
    documents: List[str]

rag_system: Optional[RAGSystem] = None


def load_html_template(template_path: str = "templates/index.html") -> str:
    """Load HTML template from file"""
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG System</title>
        </head>
        <body>
            <h1>RAG System for Documents</h1>
            <p>HTML template file 'index.html' not found. Please create the template file.</p>
        </body>
        </html>
        """
    except Exception as e:
        print(f"Error loading HTML template: {e}")
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG System - Error</title>
        </head>
        <body>
            <h1>Error loading template</h1>
            <p>Error: {str(e)}</p>
        </body>
        </html>
        """


def create_app(voyage_key: str, gemini_key: str, db_path: str = "./chroma_db", template_path: str = "index.html") -> FastAPI:
    """create fastAPI app"""
    
    global rag_system
    
    app = FastAPI(
        title="RAG system for docs",
        description="question-answering system for uploaded documents",
        version="1.0.0"
    )
    
    app.mount("/static", StaticFiles(directory="static"), name="static")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        rag_system = RAGSystem(voyage_key, gemini_key, db_path)
        print("initialized for web interface")
    except Exception as e:
        print(f"initialization error: {str(e)}")
        raise e
    
    @app.get("/", response_class=HTMLResponse)
    async def home():
        """Main page with web interface"""
        return load_html_template(template_path)
    
    @app.post("/api/upload", response_model=DocumentResponse)
    async def upload_document(
        file: UploadFile = File(...),
        document_name: Optional[str] = Form(None)
    ):
        """upload and process document"""
        
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")

        allowed_extensions = {'.txt', '.pdf', '.docx', '.html', '.htm', '.csv', '.xlsx', '.xls', '.json', '.md'}
        file_extension = Path(file.filename or "").suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {', '.join(allowed_extensions)}"
            )
        
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name

            doc_name = document_name or file.filename
            success = rag_system.add_document(temp_file_path, doc_name)
            
            if success:
                return DocumentResponse(
                    success=True,
                    message=f"Document '{doc_name}' successfully added to knowledge base",
                    document_name=doc_name
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to process document")
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
    
    @app.post("/api/ask", response_model=QuestionResponse)
    async def ask_question(request: QuestionRequest):
        """Ask question to the system"""
        
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        try:
            start_time = time.time()

            result = rag_system.ask_question(request.question, request.n_results)
            
            processing_time = time.time() - start_time
            
            return QuestionResponse(
                answer=result['answer'],
                sources=result['sources'],
                chunks_used=result['chunks_used'],
                processing_time=round(processing_time, 2)
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Question processing error: {str(e)}")
    
    @app.get("/api/stats", response_model=StatsResponse)
    async def get_stats():
        """get stats"""
        
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        try:
            stats = rag_system.get_stats()
            return StatsResponse(**stats)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"stat error: {str(e)}")
    
    @app.delete("/api/clear")
    async def clear_database():
        """clear knowledge base"""
        
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        try:
            rag_system.client.delete_collection("documents")
            rag_system.collection = rag_system.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            return {"success": True, "message": "Knowledge base cleared"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")
    
    return app


def run_web_server(voyage_key: str, gemini_key: str, host: str = "0.0.0.0", port: int = 8000, db_path: str = "./chroma_db", template_path: str = "index.html"):
    """Start web server"""
    
    print("starting interface...")
    print(f"URL: http://{host}:{port}")
    print(f"HTML template: {template_path}")
    print("-" * 60)

    if not os.path.exists(template_path):
        print(f"Warning: HTML template '{template_path}' not found!")
        print("Please make sure the HTML file exists in the same directory.")
        print("-" * 60)
    
    app = create_app(voyage_key, gemini_key, db_path, template_path)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    print("sys for docs")
    print("chromaDb + voyageAi + gemini")
    print("-" * 60)

    if len(sys.argv) > 1 and sys.argv[1] == 'web':
        parser = argparse.ArgumentParser(description='web interface')
        parser.add_argument('--voyage-key', required=True, help='voyage key')
        parser.add_argument('--gemini-key', required=True, help='gemini key')
        parser.add_argument('--host', default='0.0.0.0', help='server host')
        parser.add_argument('--port', type=int, default=8000, help='server port')
        parser.add_argument('--db-path', default='./chroma_db', help='database path')
        parser.add_argument('--template', default='./templates/index.html', help='HTML template file path')
        
        args = parser.parse_args(sys.argv[2:])
        
        run_web_server(args.voyage_key, args.gemini_key, args.host, args.port, args.db_path, args.template)
    else:
        print("usage: python web_i.py web --voyage-key YOUR_KEY --gemini-key YOUR_KEY")