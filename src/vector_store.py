"""
向量存储管理
"""

import os
import shutil
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
import logging

from .utils import logger, timer, handle_exceptions
from .document_loader import DocumentLoader, get_document_loader

class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(self, config=None):
        """初始化向量存储管理器"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import config as default_config
        
        self.config = config or default_config
        self.embedding_model = None
        self.vector_store = None
        self.document_loader = get_document_loader(config)
        
    def init_embedding_model(self) -> Embeddings:
        """初始化嵌入模型"""
        try:
            if self.config.model.embedding_model.startswith("text-embedding-"):
                # 使用OpenAI嵌入模型
                from langchain_openai import OpenAIEmbeddings
                
                if not self.config.model.openai_api_key:
                    raise ValueError("OpenAI API Key未配置")
                
                self.embedding_model = OpenAIEmbeddings(
                    model=self.config.model.embedding_model,
                    openai_api_key=self.config.model.openai_api_key
                )
                logger.info(f"使用OpenAI嵌入模型: {self.config.model.embedding_model}")
                
            else:
                # 使用HuggingFace嵌入模型
                from langchain_community.embeddings import HuggingFaceEmbeddings
                
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.config.model.embedding_model,
                    model_kwargs={'device': self.config.model.embedding_device},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"使用HuggingFace嵌入模型: {self.config.model.embedding_model}")
            
            return self.embedding_model
            
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {e}")
            raise
    
    def init_vector_store(self, force_recreate: bool = False):
        """初始化向量存储"""
        vector_store_type = self.config.vector_store.vector_store_type.lower()
        
        if force_recreate and vector_store_type == "chroma":
            # 如果强制重建，删除已有的Chroma数据库
            chroma_dir = self.config.vector_store.chroma_persist_dir
            if os.path.exists(chroma_dir):
                shutil.rmtree(chroma_dir)
                logger.info(f"已删除旧的Chroma数据库: {chroma_dir}")
        
        if vector_store_type == "chroma":
            return self._init_chroma_vector_store()
        elif vector_store_type == "faiss":
            return self._init_faiss_vector_store()
        else:
            raise ValueError(f"不支持的向量存储类型: {vector_store_type}")
    
    def _init_chroma_vector_store(self):
        """初始化Chroma向量存储"""
        try:
            from langchain_chroma import Chroma
            
            # 初始化嵌入模型
            if self.embedding_model is None:
                self.init_embedding_model()
            
            # 检查是否已存在向量存储
            if os.path.exists(self.config.vector_store.chroma_persist_dir):
                logger.info(f"加载已有的Chroma数据库: {self.config.vector_store.chroma_persist_dir}")
                self.vector_store = Chroma(
                    persist_directory=self.config.vector_store.chroma_persist_dir,
                    embedding_function=self.embedding_model,
                    collection_name=self.config.vector_store.chroma_collection_name
                )
            else:
                logger.info("创建新的Chroma数据库")
                self.vector_store = Chroma(
                    persist_directory=self.config.vector_store.chroma_persist_dir,
                    embedding_function=self.embedding_model,
                    collection_name=self.config.vector_store.chroma_collection_name
                )
            
            logger.info("Chroma向量存储初始化完成")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"初始化Chroma向量存储失败: {e}")
            raise
    
    def _init_faiss_vector_store(self):
        """初始化FAISS向量存储"""
        try:
            from langchain_community.vectorstores import FAISS
            
            # 初始化嵌入模型
            if self.embedding_model is None:
                self.init_embedding_model()
            
            # 检查是否已存在FAISS索引
            if os.path.exists(self.config.vector_store.faiss_index_path):
                logger.info(f"加载已有的FAISS索引: {self.config.vector_store.faiss_index_path}")
                self.vector_store = FAISS.load_local(
                    self.config.vector_store.faiss_index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            else:
                logger.info("创建新的FAISS索引")
                # 创建空的FAISS索引
                from langchain_community.vectorstores.utils import DistanceStrategy
                
                self.vector_store = FAISS.from_documents(
                    [],  # 空文档列表
                    self.embedding_model
                )
            
            logger.info("FAISS向量存储初始化完成")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"初始化FAISS向量存储失败: {e}")
            raise
    
    @timer
    def add_documents(self, documents: List[Document], 
                      ids: Optional[List[str]] = None) -> List[str]:
        """添加文档到向量存储"""
        if not self.vector_store:
            self.init_vector_store()
        
        if not documents:
            logger.warning("没有文档可添加")
            return []
        
        try:
            # 添加文档
            doc_ids = self.vector_store.add_documents(documents, ids=ids)
            
            # 持久化存储
            if self.config.vector_store.vector_store_type == "chroma":
                self.vector_store.persist()
            elif self.config.vector_store.vector_store_type == "faiss":
                self.vector_store.save_local(self.config.vector_store.faiss_index_path)
            
            logger.info(f"成功添加 {len(doc_ids)} 个文档到向量存储")
            return doc_ids
            
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {e}")
            raise
    
    @timer
    def add_documents_from_directory(self, directory_path: str, 
                                     recursive: bool = True) -> List[str]:
        """从目录添加文档到向量存储"""
        # 加载文档
        documents = self.document_loader.load_documents_from_directory(
            directory_path, recursive=recursive
        )
        
        if not documents:
            logger.warning(f"目录中没有找到文档: {directory_path}")
            return []
        
        # 分割文档
        split_docs = self.document_loader.split_documents(documents)
        
        # 添加到向量存储
        return self.add_documents(split_docs)
    
    @timer
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """相似性搜索"""
        if not self.vector_store:
            self.init_vector_store()
        
        k = k or self.config.vector_store.k
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"相似性搜索完成: 查询='{query[:50]}...', 返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"相似性搜索失败: {e}")
            return []
    
    @timer
    def similarity_search_with_score(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """带相似度得分的搜索"""
        if not self.vector_store:
            self.init_vector_store()
        
        k = k or self.config.vector_store.k
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"带得分的相似性搜索完成: 查询='{query[:50]}...', 返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"带得分的相似性搜索失败: {e}")
            return []
    
    @timer
    def max_marginal_relevance_search(self, query: str, k: Optional[int] = None, 
                                     fetch_k: Optional[int] = None) -> List[Document]:
        """最大边际相关性搜索（减少重复）"""
        if not self.vector_store:
            self.init_vector_store()
        
        k = k or self.config.vector_store.k
        fetch_k = fetch_k or k * 2
        
        try:
            results = self.vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
            logger.info(f"MMR搜索完成: 查询='{query[:50]}...', 返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"MMR搜索失败: {e}")
            return []
    
    def delete_documents(self, ids: List[str]) -> None:
        """从向量存储删除文档"""
        if not self.vector_store:
            self.init_vector_store()
        
        try:
            # Chroma支持删除
            if hasattr(self.vector_store, '_collection') and hasattr(self.vector_store._collection, 'delete'):
                self.vector_store._collection.delete(ids=ids)
                logger.info(f"从向量存储删除 {len(ids)} 个文档")
            else:
                logger.warning("当前向量存储不支持删除操作")
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        if not self.vector_store:
            self.init_vector_store()
        
        stats = {
            "vector_store_type": self.config.vector_store.vector_store_type,
            "total_documents": 0,
            "embedding_model": self.config.model.embedding_model
        }
        
        try:
            if self.config.vector_store.vector_store_type == "chroma":
                if hasattr(self.vector_store, '_collection'):
                    collection = self.vector_store._collection
                    stats["total_documents"] = collection.count()
                    
            elif self.config.vector_store.vector_store_type == "faiss":
                if hasattr(self.vector_store, 'index') and hasattr(self.vector_store.index, 'ntotal'):
                    stats["total_documents"] = self.vector_store.index.ntotal
        
        except Exception as e:
            logger.error(f"获取向量存储统计信息失败: {e}")
        
        return stats
    
    def clear_vector_store(self) -> bool:
        """清空向量存储"""
        try:
            if self.config.vector_store.vector_store_type == "chroma":
                if os.path.exists(self.config.vector_store.chroma_persist_dir):
                    shutil.rmtree(self.config.vector_store.chroma_persist_dir)
                    logger.info(f"已清空Chroma数据库: {self.config.vector_store.chroma_persist_dir}")
            
            elif self.config.vector_store.vector_store_type == "faiss":
                index_files = [
                    self.config.vector_store.faiss_index_path + ".pkl",
                    self.config.vector_store.faiss_index_path + ".faiss"
                ]
                for file_path in index_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"已删除FAISS索引文件: {file_path}")
            
            # 重新初始化向量存储
            self.vector_store = None
            self.init_vector_store(force_recreate=False)
            
            return True
            
        except Exception as e:
            logger.error(f"清空向量存储失败: {e}")
            return False

# 单例模式
_vector_store_manager_instance = None

def get_vector_store_manager(config=None) -> VectorStoreManager:
    """获取向量存储管理器单例"""
    global _vector_store_manager_instance
    if _vector_store_manager_instance is None:
        _vector_store_manager_instance = VectorStoreManager(config)
    return _vector_store_manager_instance