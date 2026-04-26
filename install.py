"""
自动安装脚本，解决依赖冲突
"""

import subprocess
import sys
import os

def run_command(command):
    """运行命令并处理输出"""
    print(f"运行命令: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False
    print(f"成功: {result.stdout}")
    return True

def main():
    print("开始安装企业文档智能平台依赖...")
    
    # 1. 先安装基础依赖
    print("\n1. 安装基础依赖...")
    base_deps = [
        "pip install --upgrade pip",
        "pip install setuptools wheel",
    ]
    
    for cmd in base_deps:
        if not run_command(cmd):
            return False
    
    # 2. 安装核心依赖（不指定版本，让pip自动解决）
    print("\n2. 安装核心依赖...")
    core_deps = [
        "pip install langchain langchain-community langchain-text-splitters",
        "pip install langchain-chroma langchain-openai",
    ]
    
    for cmd in core_deps:
        if not run_command(cmd):
            return False
    
    # 3. 安装OpenAI（使用兼容版本）
    print("\n3. 安装OpenAI...")
    if not run_command("pip install 'openai>=1.10.0,<2.0.0'"):
        print("尝试安装特定版本...")
        if not run_command("pip install openai==1.12.0"):
            return False
    
    # 4. 安装向量数据库
    print("\n4. 安装向量数据库...")
    vector_deps = [
        "pip install chromadb==0.4.22",
        "pip install sentence-transformers==2.2.2",
        "pip install faiss-cpu==1.7.4",
    ]
    
    for cmd in vector_deps:
        if not run_command(cmd):
            return False
    
    # 5. 安装文档处理
    print("\n5. 安装文档处理...")
    doc_deps = [
        "pip install unstructured==0.10.30",
        "pip install pymupdf==1.23.8",
        "pip install pypdf==3.17.4",
        "pip install python-docx==0.8.11",
        "pip install markdown==3.5.1",
    ]
    
    for cmd in doc_deps:
        if not run_command(cmd):
            return False
    
    # 6. 安装Web框架（可选）
    print("\n6. 安装Web框架（可选）...")
    web_deps = [
        "pip install fastapi==0.104.1",
        "pip install uvicorn==0.24.0",
        "pip install jinja2==3.1.2",
        "pip install python-multipart==0.0.6",
    ]
    
    for cmd in web_deps:
        if not run_command(cmd):
            print(f"警告: {cmd} 安装失败，Web功能可能不可用")
    
    # 7. 安装其他工具
    print("\n7. 安装其他工具...")
    other_deps = [
        "pip install python-dotenv==1.0.0",
        "pip install tqdm==4.66.1",
        "pip install pandas==2.1.3",
        "pip install numpy==1.24.3",
    ]
    
    for cmd in other_deps:
        if not run_command(cmd):
            return False
    
    print("\n✅ 所有依赖安装完成！")
    print("\n可以运行以下命令测试：")
    print("python -c \"import langchain; print(f'LangChain版本: {langchain.__version__}')\"")
    print("python -c \"import openai; print(f'OpenAI版本: {openai.__version__}')\"")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)