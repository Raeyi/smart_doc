"""
Web界面 - 使用FastAPI
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import os
import json
import tempfile
import shutil
from pathlib import Path
import uvicorn

# 创建FastAPI应用
def create_app(platform=None):
    """创建FastAPI应用"""
    
    app = FastAPI(
        title="企业文档智能平台",
        description="基于AI的企业文档管理和智能问答系统",
        version="1.0.0"
    )
    
    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 获取平台实例
    if platform is None:
        from smart_doc import SmartDocPlatform
        platform = SmartDocPlatform()
        platform.init_components(init_llm=True)
    
    # API路由
    @app.get("/", response_class=HTMLResponse)
    async def read_root():
        """首页"""
        html_content = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>企业文档智能平台</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }
                .container {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                }
                .card {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .card h3 {
                    color: #333;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }
                .btn {
                    display: inline-block;
                    background: #667eea;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                    text-decoration: none;
                    margin-top: 10px;
                }
                .btn:hover {
                    background: #764ba2;
                }
                .api-list {
                    list-style: none;
                    padding: 0;
                }
                .api-list li {
                    padding: 10px;
                    border-bottom: 1px solid #eee;
                }
                .api-list li:last-child {
                    border-bottom: none;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📚 企业文档智能平台</h1>
                <p>基于AI的企业文档管理和智能问答系统</p>
            </div>
            
            <div class="container">
                <div class="card">
                    <h3>📁 文档管理</h3>
                    <p>上传、管理和检索企业文档</p>
                    <a href="/docs" class="btn">API文档</a>
                    <a href="/upload" class="btn">上传文档</a>
                </div>
                
                <div class="card">
                    <h3>🔍 智能搜索</h3>
                    <p>基于语义的文档搜索</p>
                    <a href="/search-ui" class="btn">搜索界面</a>
                </div>
                
                <div class="card">
                    <h3>💬 智能问答</h3>
                    <p>基于文档的智能问答</p>
                    <a href="/chat-ui" class="btn">问答界面</a>
                </div>
                
                <div class="card">
                    <h3>📊 系统状态</h3>
                    <p>查看系统运行状态和统计信息</p>
                    <a href="/api/stats" class="btn">查看状态</a>
                </div>
            </div>
            
            <div class="card" style="margin-top: 30px;">
                <h3>🛠️ API接口</h3>
                <ul class="api-list">
                    <li><strong>GET /api/stats</strong> - 系统状态</li>
                    <li><strong>POST /api/ingest</strong> - 摄取文档</li>
                    <li><strong>POST /api/search</strong> - 搜索文档</li>
                    <li><strong>POST /api/ask</strong> - 智能问答</li>
                    <li><strong>POST /api/chat</strong> - 对话模式</li>
                    <li><strong>POST /api/upload</strong> - 上传文档</li>
                </ul>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    @app.get("/api/stats")
    async def get_stats():
        """获取系统状态"""
        try:
            stats = platform.get_stats()
            return JSONResponse(content=stats)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/ingest")
    async def ingest_documents(
        directory: Optional[str] = Form(None),
        recursive: bool = Form(True)
    ):
        """摄取文档到向量存储"""
        try:
            if not directory:
                directory = platform.config.document.docs_dir
            
            result = platform.ingest_documents(directory, recursive)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/search")
    async def search_documents(
        query: str = Form(...),
        search_type: str = Form("similarity"),
        k: int = Form(4)
    ):
        """搜索文档"""
        try:
            result = platform.search_documents(query, search_type, k)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/ask")
    async def ask_question(
        question: str = Form(...),
        chain_type: str = Form("stuff"),
        prompt_template: str = Form("default"),
        k: int = Form(4)
    ):
        """智能问答"""
        try:
            if not platform.llm:
                raise HTTPException(status_code=400, detail="LLM未初始化，问答功能不可用")
            
            result = platform.ask_question(question, chain_type, prompt_template, k)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/chat")
    async def chat(
        question: str = Form(...),
        chat_history: Optional[str] = Form(None)
    ):
        """对话模式"""
        try:
            if not platform.llm:
                raise HTTPException(status_code=400, detail="LLM未初始化，对话功能不可用")
            
            # 解析聊天历史
            history = None
            if chat_history:
                try:
                    history = json.loads(chat_history)
                except:
                    history = None
            
            result = platform.chat(question, history)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/upload")
    async def upload_documents(
        files: List[UploadFile] = File(...),
        directory: Optional[str] = Form(None)
    ):
        """上传文档"""
        try:
            if not directory:
                directory = platform.config.document.docs_dir
            
            # 确保目录存在
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            uploaded_files = []
            for file in files:
                # 保存文件
                file_path = os.path.join(directory, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                uploaded_files.append({
                    "filename": file.filename,
                    "path": file_path,
                    "size": os.path.getsize(file_path)
                })
            
            # 摄取上传的文档
            result = platform.ingest_documents(directory, recursive=False)
            result["uploaded_files"] = uploaded_files
            
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/clear")
    async def clear_vector_store():
        """清空向量存储"""
        try:
            result = platform.clear_vector_store()
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/search-ui", response_class=HTMLResponse)
    async def search_ui():
        """搜索界面"""
        html_content = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>文档搜索</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .search-box {
                    display: flex;
                    gap: 10px;
                    margin-bottom: 20px;
                }
                .search-box input {
                    flex: 1;
                    padding: 10px;
                    font-size: 16px;
                }
                .search-box select, .search-box button {
                    padding: 10px 20px;
                }
                .result-item {
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                }
                .result-content {
                    margin: 10px 0;
                }
                .result-meta {
                    font-size: 12px;
                    color: #666;
                }
                .loading {
                    text-align: center;
                    padding: 20px;
                }
            </style>
        </head>
        <body>
            <h1>🔍 文档搜索</h1>
            <div class="search-box">
                <input type="text" id="query" placeholder="输入搜索关键词...">
                <select id="search-type">
                    <option value="similarity">语义搜索</option>
                    <option value="mmr">去重搜索</option>
                    <option value="hybrid">混合搜索</option>
                </select>
                <button onclick="search()">搜索</button>
            </div>
            <div id="results"></div>
            
            <script>
                async function search() {
                    const query = document.getElementById('query').value;
                    const searchType = document.getElementById('search-type').value;
                    
                    if (!query) {
                        alert('请输入搜索关键词');
                        return;
                    }
                    
                    document.getElementById('results').innerHTML = '<div class="loading">搜索中...</div>';
                    
                    const formData = new FormData();
                    formData.append('query', query);
                    formData.append('search_type', searchType);
                    formData.append('k', 10);
                    
                    try {
                        const response = await fetch('/api/search', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (data.success) {
                            let html = `<h3>找到 ${data.result_count} 个结果:</h3>`;
                            
                            data.results.forEach((result, index) => {
                                html += `
                                    <div class="result-item">
                                        <strong>${index + 1}. ${result.metadata.filename || '文档'}</strong>
                                        <div class="result-content">${result.content}</div>
                                        <div class="result-meta">
                                            来源: ${result.metadata.source || '未知'} |
                                            长度: ${result.length} 字符
                                        </div>
                                    </div>
                                `;
                            });
                            
                            document.getElementById('results').innerHTML = html;
                        } else {
                            document.getElementById('results').innerHTML = `<div class="error">搜索失败: ${data.error}</div>`;
                        }
                    } catch (error) {
                        document.getElementById('results').innerHTML = `<div class="error">请求失败: ${error}</div>`;
                    }
                }
                
                // 按回车键搜索
                document.getElementById('query').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        search();
                    }
                });
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    @app.get("/chat-ui", response_class=HTMLResponse)
    async def chat_ui():
        """聊天界面"""
        html_content = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>智能问答</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .chat-container {
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    padding: 20px;
                    height: 600px;
                    display: flex;
                    flex-direction: column;
                }
                .chat-messages {
                    flex: 1;
                    overflow-y: auto;
                    margin-bottom: 20px;
                }
                .message {
                    margin-bottom: 15px;
                    padding: 10px;
                    border-radius: 10px;
                    max-width: 80%;
                }
                .user-message {
                    background-color: #e3f2fd;
                    margin-left: auto;
                }
                .bot-message {
                    background-color: #f5f5f5;
                }
                .chat-input {
                    display: flex;
                    gap: 10px;
                }
                .chat-input input {
                    flex: 1;
                    padding: 10px;
                    font-size: 16px;
                }
                .chat-input button {
                    padding: 10px 20px;
                }
                .loading {
                    text-align: center;
                    padding: 20px;
                }
            </style>
        </head>
        <body>
            <h1>💬 智能问答</h1>
            <div class="chat-container">
                <div class="chat-messages" id="chat-messages">
                    <div class="message bot-message">
                        你好！我是企业文档智能助手。你可以问我关于文档内容的问题。
                    </div>
                </div>
                <div class="chat-input">
                    <input type="text" id="question" placeholder="输入你的问题..." autocomplete="off">
                    <button onclick="sendMessage()">发送</button>
                </div>
            </div>
            
            <script>
                let chatHistory = [];
                
                function addMessage(content, isUser = false) {
                    const messagesDiv = document.getElementById('chat-messages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                    messageDiv.textContent = content;
                    messagesDiv.appendChild(messageDiv);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }
                
                async function sendMessage() {
                    const question = document.getElementById('question').value.trim();
                    
                    if (!question) {
                        return;
                    }
                    
                    // 添加用户消息
                    addMessage(question, true);
                    
                    // 清空输入框
                    document.getElementById('question').value = '';
                    
                    // 添加加载提示
                    const loadingDiv = document.createElement('div');
                    loadingDiv.className = 'loading';
                    loadingDiv.id = 'loading';
                    loadingDiv.textContent = '思考中...';
                    document.getElementById('chat-messages').appendChild(loadingDiv);
                    
                    const formData = new FormData();
                    formData.append('question', question);
                    if (chatHistory.length > 0) {
                        formData.append('chat_history', JSON.stringify(chatHistory));
                    }
                    
                    try {
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        // 移除加载提示
                        document.getElementById('loading').remove();
                        
                        if (data.success) {
                            // 添加助手回复
                            addMessage(data.answer);
                            
                            // 更新聊天历史
                            chatHistory.push([question, data.answer]);
                            
                            // 如果历史太长，移除最早的记录
                            if (chatHistory.length > 10) {
                                chatHistory.shift();
                            }
                        } else {
                            addMessage(`抱歉，出错了: ${data.error}`);
                        }
                    } catch (error) {
                        document.getElementById('loading').remove();
                        addMessage(`请求失败: ${error}`);
                    }
                }
                
                // 按回车键发送
                document.getElementById('question').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    @app.get("/upload", response_class=HTMLResponse)
    async def upload_ui():
        """上传界面"""
        html_content = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>上传文档</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .upload-box {
                    border: 2px dashed #ddd;
                    padding: 40px;
                    text-align: center;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                .file-list {
                    margin-top: 20px;
                }
                .file-item {
                    padding: 10px;
                    border-bottom: 1px solid #eee;
                }
                .progress {
                    width: 100%;
                    height: 20px;
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    overflow: hidden;
                    margin-top: 10px;
                }
                .progress-bar {
                    height: 100%;
                    background-color: #4CAF50;
                    width: 0%;
                    transition: width 0.3s;
                }
            </style>
        </head>
        <body>
            <h1>📁 上传文档</h1>
            <div class="upload-box">
                <input type="file" id="file-input" multiple style="display: none;">
                <button onclick="document.getElementById('file-input').click()" style="padding: 15px 30px; font-size: 16px;">
                    选择文件
                </button>
                <p>支持格式: .txt, .md, .pdf, .docx, .pptx, .xlsx, .html 等</p>
                <div id="file-list" class="file-list"></div>
                <button onclick="uploadFiles()" style="padding: 15px 30px; font-size: 16px; background-color: #4CAF50; color: white; border: none;">
                    开始上传
                </button>
                <div id="progress" class="progress" style="display: none;">
                    <div class="progress-bar" id="progress-bar"></div>
                </div>
            </div>
            <div id="result"></div>
            
            <script>
                let selectedFiles = [];
                
                document.getElementById('file-input').addEventListener('change', function(e) {
                    selectedFiles = Array.from(e.target.files);
                    updateFileList();
                });
                
                function updateFileList() {
                    const fileListDiv = document.getElementById('file-list');
                    
                    if (selectedFiles.length === 0) {
                        fileListDiv.innerHTML = '<p>未选择文件</p>';
                        return;
                    }
                    
                    let html = '<h3>已选择文件:</h3>';
                    selectedFiles.forEach((file, index) => {
                        html += `
                            <div class="file-item">
                                <strong>${file.name}</strong> (${formatFileSize(file.size)})
                            </div>
                        `;
                    });
                    
                    fileListDiv.innerHTML = html;
                }
                
                function formatFileSize(bytes) {
                    if (bytes === 0) return '0 Bytes';
                    const k = 1024;
                    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                    const i = Math.floor(Math.log(bytes) / Math.log(k));
                    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
                }
                
                async function uploadFiles() {
                    if (selectedFiles.length === 0) {
                        alert('请先选择文件');
                        return;
                    }
                    
                    const formData = new FormData();
                    selectedFiles.forEach(file => {
                        formData.append('files', file);
                    });
                    
                    // 显示进度条
                    document.getElementById('progress').style.display = 'block';
                    document.getElementById('progress-bar').style.width = '0%';
                    
                    try {
                        const response = await fetch('/api/upload', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (data.success) {
                            document.getElementById('result').innerHTML = `
                                <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 5px; margin-top: 20px;">
                                    <h3>✅ 上传成功！</h3>
                                    <p>上传了 ${data.uploaded_files.length} 个文件，添加了 ${data.document_count} 个文档块到向量存储。</p>
                                    <ul>
                                        ${data.uploaded_files.map(file => `<li>${file.filename} (${formatFileSize(file.size)})</li>`).join('')}
                                    </ul>
                                </div>
                            `;
                        } else {
                            document.getElementById('result').innerHTML = `
                                <div style="background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 5px; margin-top: 20px;">
                                    <h3>❌ 上传失败</h3>
                                    <p>错误: ${data.error}</p>
                                </div>
                            `;
                        }
                    } catch (error) {
                        document.getElementById('result').innerHTML = `
                            <div style="background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 5px; margin-top: 20px;">
                                <h3>❌ 上传失败</h3>
                                <p>错误: ${error}</p>
                            </div>
                        `;
                    } finally {
                        // 隐藏进度条
                        document.getElementById('progress').style.display = 'none';
                        // 清空文件列表
                        selectedFiles = [];
                        document.getElementById('file-input').value = '';
                        updateFileList();
                    }
                }
                
                // 初始化文件列表
                updateFileList();
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    return app