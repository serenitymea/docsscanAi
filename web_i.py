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


def create_app(voyage_key: str, gemini_key: str, db_path: str = "./chroma_db") -> FastAPI:
    """create fastAPI app"""
    
    global rag_system
    
    app = FastAPI(
        title="RAG system for docs",
        description="question-answering system for uploaded documents",
        version="1.0.0"
    )

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
        return HTML_INTERFACE
    
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

HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System for Documents</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            color: #2d3748;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header p {
            color: #718096;
            font-size: 1.1rem;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
        }
        
        .card h3 {
            color: #2d3748;
            margin-bottom: 20px;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .icon {
            font-size: 1.5rem;
        }
        
        .file-upload {
            border: 2px dashed #cbd5e0;
            border-radius: 10px;
            padding: 30px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 15px;
        }
        
        .file-upload:hover {
            border-color: #667eea;
            background: #f7fafc;
        }
        
        .file-upload.dragover {
            border-color: #667eea;
            background: #ebf8ff;
        }
        
        input[type="file"] {
            display: none;
        }
        
        input[type="text"], textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
            font-family: inherit;
        }
        
        input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        textarea {
            resize: vertical;
            min-height: 120px;
        }
        
        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 15px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-secondary {
            background: #e2e8f0;
            color: #2d3748;
        }
        
        .btn-danger {
            background: #e53e3e;
        }
        
        .status {
            padding: 12px 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-weight: 500;
        }
        
        .status.success {
            background: #c6f6d5;
            color: #22543d;
            border: 1px solid #9ae6b4;
        }
        
        .status.error {
            background: #fed7d7;
            color: #742a2a;
            border: 1px solid #fc8181;
        }
        
        .status.loading {
            background: #bee3f8;
            color: #2a69ac;
            border: 1px solid #7dd3fc;
        }
        
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 10px;
        }
        
        .message.user {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        
        .message.assistant {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        
        .message .header {
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .message .content {
            line-height: 1.6;
            white-space: pre-wrap;
        }
        
        .message .sources {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(0,0,0,0.1);
            font-size: 0.85rem;
            color: #666;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            display: block;
        }
        
        .stat-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 1.5s infinite;
        }
        
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG System for Documents</h1>
            <p>Upload documents and ask questions based on their content</p>
        </div>
        
        <div class="main-grid">
            <!-- Left panel: document management -->
            <div class="card">
                <h3><span class="icon"></span> Document Management</h3>
                
                <div class="file-upload" id="fileUpload">
                    <p>Drag file here or click to select</p>
                    <p style="font-size: 0.9rem; color: #666; margin-top: 10px;">
                        Supported: PDF, DOCX, TXT, CSV, XLSX, HTML, JSON
                    </p>
                </div>
                
                <input type="file" id="fileInput" accept=".pdf,.docx,.txt,.csv,.xlsx,.html,.json,.md">
                <input type="text" id="docName" placeholder="Document name (optional)">
                <button class="btn" id="uploadBtn" disabled>Upload Document</button>
                
                <div id="uploadStatus"></div>
                
                <!-- Statistics -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-number" id="totalChunks">0</span>
                        <span class="stat-label">Fragments</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number" id="totalDocs">0</span>
                        <span class="stat-label">Documents</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number" id="avgTime">0.0s</span>
                        <span class="stat-label">Avg Time</span>
                    </div>
                </div>
                
                <button class="btn btn-secondary" id="refreshStats">Refresh Statistics</button>
                <button class="btn btn-danger" id="clearDb">Clear Database</button>
            </div>
            
            <!-- Right panel: chat -->
            <div class="card">
                <h3><span class="icon"></span> Ask Questions</h3>
                
                <div class="chat-container" id="chatContainer">
                    <div class="message assistant">
                        <div class="header">System</div>
                        <div class="content">
                            Welcome! Upload documents on the left and ask questions about their content.
                            I will find the most relevant information and provide accurate answers.
                        </div>
                    </div>
                </div>
                
                <textarea id="questionInput" placeholder="Enter your question..."></textarea>
                <button class="btn" id="askBtn">Ask Question</button>
                
                <div id="questionStatus"></div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let totalQuestions = 0;
        let totalTime = 0;
        
        // DOM elements
        const fileUpload = document.getElementById('fileUpload');
        const fileInput = document.getElementById('fileInput');
        const docName = document.getElementById('docName');
        const uploadBtn = document.getElementById('uploadBtn');
        const uploadStatus = document.getElementById('uploadStatus');
        const questionInput = document.getElementById('questionInput');
        const askBtn = document.getElementById('askBtn');
        const questionStatus = document.getElementById('questionStatus');
        const chatContainer = document.getElementById('chatContainer');
        const refreshStats = document.getElementById('refreshStats');
        const clearDb = document.getElementById('clearDb');
        
        // Statistics
        const totalChunks = document.getElementById('totalChunks');
        const totalDocs = document.getElementById('totalDocs');
        const avgTime = document.getElementById('avgTime');
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadStats();
        });
        
        // File upload handling
        fileUpload.addEventListener('click', () => fileInput.click());
        
        fileUpload.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUpload.classList.add('dragover');
        });
        
        fileUpload.addEventListener('dragleave', () => {
            fileUpload.classList.remove('dragover');
        });
        
        fileUpload.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUpload.classList.remove('dragover');
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        });
        
        fileInput.addEventListener('change', handleFileSelect);
        
        function handleFileSelect() {
            const file = fileInput.files[0];
            if (file) {
                uploadBtn.disabled = false;
                uploadBtn.textContent = `Upload ${file.name}`;
            }
        }
        
        // Document upload
        uploadBtn.addEventListener('click', async () => {
            const file = fileInput.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            if (docName.value.trim()) {
                formData.append('document_name', docName.value.trim());
            }
            
            uploadBtn.disabled = true;
            uploadBtn.classList.add('loading');
            showStatus(uploadStatus, 'loading', 'Processing document...');
            
            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showStatus(uploadStatus, 'success', result.message);
                    fileInput.value = '';
                    docName.value = '';
                    uploadBtn.textContent = 'Upload Document';
                    loadStats();
                } else {
                    showStatus(uploadStatus, 'error', result.detail || 'Upload error');
                }
            } catch (error) {
                showStatus(uploadStatus, 'error', `Error: ${error.message}`);
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.classList.remove('loading');
            }
        });
        
        // Ask question
        askBtn.addEventListener('click', askQuestion);
        questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                askQuestion();
            }
        });
        
        async function askQuestion() {
            const question = questionInput.value.trim();
            if (!question) return;
            
            // Add user question to chat
            addMessage('user', question);
            
            askBtn.disabled = true;
            askBtn.classList.add('loading');
            askBtn.textContent = 'Processing...';
            
            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        question: question,
                        n_results: 3
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    // Add system answer
                    addMessage('assistant', result.answer, result.sources, result.processing_time);
                    
                    // Update time statistics
                    totalQuestions++;
                    totalTime += result.processing_time;
                    avgTime.textContent = (totalTime / totalQuestions).toFixed(1) + 's';
                    
                    questionInput.value = '';
                } else {
                    showStatus(questionStatus, 'error', result.detail || 'Question processing error');
                }
            } catch (error) {
                showStatus(questionStatus, 'error', `Error: ${error.message}`);
            } finally {
                askBtn.disabled = false;
                askBtn.classList.remove('loading');
                askBtn.textContent = 'Ask Question';
            }
        }
        
        function addMessage(type, content, sources = null, time = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            let html = `
                <div class="header">${type === 'user' ? 'You' : 'System'}</div>
                <div class="content">${content}</div>
            `;
            
            if (sources && sources.length > 0) {
                html += `
                    <div class="sources">
                        <strong> Sources:</strong>
                        ${sources.map(s => `${s.document} (${(s.similarity * 100).toFixed(1)}%)`).join(', ')}
                        ${time ? `<br><strong> Processing time:</strong> ${time}s` : ''}
                    </div>
                `;
            }
            
            messageDiv.innerHTML = html;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Load statistics
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                totalChunks.textContent = stats.total_chunks;
                totalDocs.textContent = stats.total_documents;
            } catch (error) {
                console.error('Statistics loading error:', error);
            }
        }
        
        // Refresh statistics
        refreshStats.addEventListener('click', loadStats);
        
        // Clear database
        clearDb.addEventListener('click', async () => {
            if (!confirm('Are you sure you want to delete all documents from the knowledge base?')) {
                return;
            }
            
            try {
                const response = await fetch('/api/clear', { method: 'DELETE' });
                const result = await response.json();
                
                if (result.success) {
                    showStatus(uploadStatus, 'success', 'Knowledge base cleared');
                    loadStats();
                    
                    // Clear chat
                    chatContainer.innerHTML = `
                        <div class="message assistant">
                            <div class="header">System</div>
                            <div class="content">Knowledge base cleared. Upload new documents to start working.</div>
                        </div>
                    `;
                }
            } catch (error) {
                showStatus(uploadStatus, 'error', `Clear error: ${error.message}`);
            }
        });
        
        function showStatus(element, type, message) {
            element.innerHTML = `<div class="status ${type}">${message}</div>`;
            
            if (type !== 'loading') {
                setTimeout(() => {
                    element.innerHTML = '';
                }, 5000);
            }
        }
    </script>
</body>
</html>
"""


def run_web_server(voyage_key: str, gemini_key: str, host: str = "0.0.0.0", port: int = 8000, db_path: str = "./chroma_db"):
    """Start web server"""
    
    print("starting interface...")
    print(f"URL: http://{host}:{port}")
    print("-" * 60)
    
    app = create_app(voyage_key, gemini_key, db_path)
    
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
        
        args = parser.parse_args(sys.argv[2:])
        
        run_web_server(args.voyage_key, args.gemini_key, args.host, args.port, args.db_path)
    else:
        print("usage: python web_i.py web --voyage-key YOUR_KEY --gemini-key YOUR_KEY")