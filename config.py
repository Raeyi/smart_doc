"""
配置文件 - 企业文档智能平台
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class AppConfig:
    """应用配置"""
    app_name: str = "企业文档智能平台"
    version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"
    log_file: str = "./logs/smart_doc.log"

@dataclass
class DocumentConfig:
    """文档处理配置"""
    # 支持的文档类型
    supported_extensions: List[str] = field(default_factory=lambda: [
        '.txt', '.md', '.pdf', '.docx', '.doc', '.ppt', '.pptx', 
        '.xls', '.xlsx', '.html', '.htm'
    ])
    
    # 文档目录
    docs_dir: str = "./docs"
    
    # 文档处理配置
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # 元数据字段
    metadata_fields: List[str] = field(default_factory=lambda: [
        'source', 'title', 'author', 'created_date', 'modified_date',
        'document_type', 'department', 'category'
    ])

@dataclass
class ModelConfig:
    """模型配置"""
    # 嵌入模型配置
    embedding_provider: str = "huggingface"  # 嵌入模型提供商
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 嵌入模型名称
    embedding_device: str = "cpu"  # 设备类型: "cpu" 或 "cuda"
    
    # LLM配置（可切换不同模型）
    llm_provider: str = "deepseek"  # 可选: "openai", "azure", "anthropic", "local"
    llm_model: str = "deepseek-chat"  # DeepSeek模型名称
    
    # OpenAI配置
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.1
    
     # DeepSeek配置
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    
    # 本地模型配置
    local_model_path: Optional[str] = None
    local_model_type: str = "llama"  # 可选: "llama", "chatglm", "qwen"
    
    # 模型参数
    temperature: float = 0.1
    max_tokens: int = 2000
    top_p: float = 0.9

@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    # 向量数据库类型
    vector_store_type: str = "chroma"  # 可选: "chroma", "faiss", "weaviate", "pinecone"
    
    # Chroma配置
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "enterprise_documents"
    
    # FAISS配置
    faiss_index_path: str = "./faiss_index"
    
    # 检索配置
    search_type: str = "similarity"  # 可选: "similarity", "mmr" (最大边际相关)
    k: int = 4  # 检索文档数量
    score_threshold: float = 0.5  # 相似度阈值

@dataclass
class DatabaseConfig:
    """数据库配置"""
    # 元数据数据库（存储文档信息）
    metadata_db_type: str = "sqlite"  # 可选: "sqlite", "mysql", "postgresql"
    
    # SQLite配置
    sqlite_db_path: str = "./smart_doc.db"
    
    # MySQL配置
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "smart_doc"

@dataclass
class APIConfig:
    """API配置"""
    # API服务器配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True
    
    # 认证配置
    enable_auth: bool = False
    jwt_secret_key: str = "your-secret-key-change-this"
    
    # 速率限制
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 3600  # 秒

@dataclass
class Config:
    """总配置"""
    app: AppConfig = AppConfig()
    document: DocumentConfig = DocumentConfig()
    model: ModelConfig = ModelConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    database: DatabaseConfig = DatabaseConfig()
    api: APIConfig = APIConfig()
    
    def __post_init__(self):
        """初始化后创建必要目录"""
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        dirs = [
            self.document.docs_dir,
            self.vector_store.chroma_persist_dir,
            os.path.dirname(self.database.sqlite_db_path),
            os.path.dirname(self.app.log_file),
        ]
        
        for dir_path in dirs:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_env_variables(self):
        """从环境变量加载配置"""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()  # 加载.env文件
        
        # DeepSeek配置
        if os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"):
            # 优先使用DEEPSEEK_API_KEY，其次使用OPENAI_API_KEY
            self.model.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if os.getenv("OPENAI_BASE_URL"):
            self.model.deepseek_base_url = os.getenv("OPENAI_BASE_URL")
        
        # 嵌入模型配置
        if os.getenv("EMBEDDING_MODEL"):
            self.model.embedding_model = os.getenv("EMBEDDING_MODEL")
        
        if os.getenv("EMBEDDING_PROVIDER"):
            self.model.embedding_provider = os.getenv("EMBEDDING_PROVIDER")
        
        if os.getenv("DEBUG"):
            self.app.debug = os.getenv("DEBUG").lower() == "true"

# 全局配置实例
config = Config()