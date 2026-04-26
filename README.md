# 企业文档智能平台 (SmartDoc)

基于AI的企业文档管理和智能问答系统，支持文档上传、智能搜索、问答对话等功能。

## 功能特点

- 📁 **多格式文档支持**: 支持txt、md、pdf、docx、pptx、xlsx、html等多种格式
- 🔍 **智能语义搜索**: 基于向量嵌入的语义搜索，理解用户查询意图
- 💬 **智能问答**: 基于RAG的智能问答，准确回答文档相关问题
- 🗣️ **多轮对话**: 支持上下文感知的多轮对话
- 📊 **向量存储**: 使用ChromaDB/FAISS进行高效向量检索
- 🌐 **Web界面**: 提供友好的Web界面，支持文档上传、搜索、问答
- 🔧 **命令行工具**: 提供完整的命令行接口
- 🔌 **可扩展架构**: 模块化设计，支持自定义模型和插件

## 系统架构

企业文档智能平台

├── 视图表现层 (Web界面/CLI)

├── 后端服务层 (FastAPI/业务逻辑)

├── LLM服务层 (OpenAI/本地LLM)

├── 检索增强层 (向量检索/RAG)

└── 数据存储层 (文档存储/向量数据库)

## 快速开始

### 1. 环境准备

bash

克隆项目

git clone <repository-url>

cd smart_doc

创建虚拟环境

python -m venv venv

source venv/bin/activate  # Linux/Mac

或

venv\Scripts\activate  # Windows

安装依赖

pip install -r requirements.txt

### 2. 配置

编辑 `config.py` 文件，配置以下参数：

python

OpenAI配置（如果使用）

config.model.openai_api_key = "your-api-key"

config.model.openai_model = "gpt-3.5-turbo"

或使用本地模型

config.model.llm_provider = "local"

config.model.local_model_path = "./models/your-model.bin"

文档目录

config.document.docs_dir = "./docs"

向量存储

config.vector_store.vector_store_type = "chroma"  # 或 "faiss"

也可以通过环境变量配置：

bash

export OPENAI_API_KEY="your-api-key"

export EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

### 3. 准备文档

将企业文档放入 `docs/` 目录，支持子目录结构：

docs/

├── 公司制度/

│   ├── 员工手册.md

│   └── 考勤制度.pdf

├── 技术文档/

│   ├── API文档.md

│   └── 架构设计.docx

└── 项目资料/

└── 项目计划.pptx

### 4. 运行程序

#### 交互模式（推荐）

bash

python main.py

#### 命令行模式

bash

摄取文档

python main.py --mode ingest --directory ./docs

搜索文档

python main.py --mode search --query "请假流程"

提问

python main.py --mode ask --question "公司的年假有多少天？"

查看系统状态

python main.py --stats

清空向量存储

python main.py --clear

#### Web服务器模式

bash

python main.py --mode web --port 8000

然后在浏览器中访问：http://localhost:8000

## API接口

### RESTful API

- `GET /api/stats` - 获取系统状态
- `POST /api/ingest` - 摄取文档
- `POST /api/search` - 搜索文档
- `POST /api/ask` - 智能问答
- `POST /api/chat` - 对话模式
- `POST /api/upload` - 上传文档
- `POST /api/clear` - 清空向量存储

### Web界面

- `/` - 首页
- `/search-ui` - 搜索界面
- `/chat-ui` - 聊天界面
- `/upload` - 上传界面

## 配置说明

### 模型配置

支持多种LLM提供商：

1. **OpenAI** (默认)
python

config.model.llm_provider = "openai"

config.model.openai_api_key = "your-api-key"

config.model.openai_model = "gpt-3.5-turbo"  # 或 "gpt-4"

2. **Azure OpenAI**

python

config.model.llm_provider = "azure"

config.model.azure_api_key = "your-key"

config.model.azure_endpoint = "https://your-resource.openai.azure.com/"

3. **本地模型**

python

config.model.llm_provider = "local"

config.model.local_model_path = "./models/llama-7b.bin"

config.model.local_model_type = "llama"  # 或 "chatglm", "qwen"

### 向量存储配置

支持两种向量数据库：

1. **ChromaDB** (默认，推荐)

python

config.vector_store.vector_store_type = "chroma"

config.vector_store.chroma_persist_dir = "./chroma_db"

2. **FAISS**

python

config.vector_store.vector_store_type = "faiss"

config.vector_store.faiss_index_path = "./faiss_index"

### 嵌入模型配置

python

使用HuggingFace嵌入模型（默认）

config.model.embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

或使用OpenAI嵌入模型

config.model.embedding_model = "text-embedding-ada-002"

## 高级功能

### 自定义提示模板

python

from src.qa_chain import get_qa_system

qa_system = get_qa_system()

qa_system.add_custom_prompt_template(

name="technical",

template="""你是一个技术专家。请基于以下技术文档回答问题。

{context}

问题: {question}

技术答案:"""

)

### 混合检索

系统支持多种检索方式：

1. **向量检索** - 基于语义相似度
2. **BM25检索** - 基于关键词匹配
3. **混合检索** - 结合两者优势
4. **MMR检索** - 最大化边际相关性，减少重复

### 文档分割策略

支持多种文档分割方式：

1. **按字符分割** - 通用分割
2. **按Markdown标题分割** - 保持文档结构
3. **按代码结构分割** - 针对代码文档
4. **按段落分割** - 针对长文档

## 部署建议

### 生产环境配置

1. **数据库持久化**

python

config.database.metadata_db_type = "mysql"

config.database.mysql_host = "localhost"

config.database.mysql_port = 3306

config.database.mysql_database = "smart_doc"

2. **启用认证**

python

config.api.enable_auth = True

config.api.jwt_secret_key = "your-strong-secret-key"

3. **性能优化**

python

config.model.embedding_device = "cuda"  # 使用GPU加速

config.vector_store.k = 10  # 增加检索数量

config.document.chunk_size = 500  # 减小分块大小

### Docker部署

dockerfile

FROM python:3.9-slim

WORKDIR /app

安装系统依赖

RUN apt-get update && apt-get install -y \

gcc \

g++ \

&& rm -rf /var/lib/apt/lists/*

复制项目文件

COPY requirements.txt .

COPY . .

安装Python依赖

RUN pip install --no-cache-dir -r requirements.txt

创建必要目录

RUN mkdir -p docs chroma_db logs

运行应用

CMD ["python", "main.py", "--mode", "web", "--host", "0.0.0.0", "--port", "8000"]

## 故障排除

### 常见问题

1. **导入错误**

pip install --upgrade langchain langchain-community langchain-text-splitters

2. **文档加载失败**

pip install "unstructured[all-docs]"

3. **内存不足**

- 减小 `config.document.chunk_size`
- 使用更小的嵌入模型
- 分批处理文档

4. **LLM回答质量差**

- 调整提示模板
- 增加检索文档数量
- 使用更高质量的文档

### 日志查看

bash

查看日志

tail -f logs/smart_doc.log

设置日志级别

export LOG_LEVEL=DEBUG

## 性能优化

1. **缓存机制**: 对常见查询结果进行缓存
2. **批量处理**: 批量处理文档，减少IO操作
3. **异步处理**: 使用异步API提高并发性能
4. **索引优化**: 定期优化向量索引

## 安全考虑

1. **文档权限**: 实现基于角色的文档访问控制
2. **API认证**: 启用JWT认证保护API
3. **输入验证**: 对所有用户输入进行验证和清理
4. **审计日志**: 记录所有用户操作

## 许可证

本项目采用 MIT 许可证。

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件至: <surrayi@163.com>

---

**注意**: 本项目处于开发阶段，API可能会发生变化。建议在生产环境中进行充分测试。
