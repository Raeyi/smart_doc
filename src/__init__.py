"""
企业文档智能平台
"""

__version__ = "1.0.0"
__author__ = "SmartDoc Team"
__description__ = "基于AI的企业文档管理和智能问答系统"

# 导出主要类和函数
from .document_loader import DocumentLoader, get_document_loader
from .vector_store import VectorStoreManager, get_vector_store_manager
from .retriever import SmartRetriever, get_retriever
from .qa_chain import DocumentQA, get_qa_system
from .utils import setup_logger, logger

__all__ = [
    "SmartDocPlatform",
    "main",
    "config",
    "DocumentLoader",
    "get_document_loader",
    "VectorStoreManager", 
    "get_vector_store_manager",
    "SmartRetriever",
    "get_retriever",
    "DocumentQA",
    "get_qa_system",
    "setup_logger",
    "logger",
]