let totalQuestions = 0;
let totalTime = 0;

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

const totalChunks = document.getElementById('totalChunks');
const totalDocs = document.getElementById('totalDocs');
const avgTime = document.getElementById('avgTime');

document.addEventListener('DOMContentLoaded', function() {
    loadStats();
});

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

askBtn.addEventListener('click', askQuestion);
questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        askQuestion();
    }
});

async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

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
            addMessage('assistant', result.answer, result.sources, result.processing_time);

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
                <strong>sources:</strong>
                ${sources.map(s => `${s.document} (${(s.similarity * 100).toFixed(1)}%)`).join(', ')}
                ${time ? `<br><strong>processing time:</strong> ${time}s` : ''}
            </div>
        `;
    }
    
    messageDiv.innerHTML = html;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

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

refreshStats.addEventListener('click', loadStats);

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