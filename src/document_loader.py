"""
文档加载器 - 支持多种文档格式
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
import logging

from .utils import logger, is_supported_document, extract_document_metadata, timer

class DocumentLoader:
    """文档加载器"""
    
    def __init__(self, config=None):
        """初始化文档加载器"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import config as default_config
        
        self.config = config or default_config
        self.supported_extensions = self.config.document.supported_extensions
        
        # 初始化加载器映射
        self.loader_map = self._init_loader_map()
        
    def _init_loader_map(self) -> Dict[str, Any]:
        """初始化文档加载器映射"""
        loader_map = {}
        
        try:
            # Markdown加载器
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            loader_map['.md'] = UnstructuredMarkdownLoader
        except ImportError as e:
            logger.warning(f"Markdown加载器不可用: {e}")
        
        try:
            # PDF加载器
            from langchain_community.document_loaders import PyPDFLoader
            loader_map['.pdf'] = PyPDFLoader
        except ImportError as e:
            logger.warning(f"PDF加载器不可用: {e}")
        
        try:
            # Word文档加载器
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader_map['.docx'] = UnstructuredWordDocumentLoader
            loader_map['.doc'] = UnstructuredWordDocumentLoader
        except ImportError as e:
            logger.warning(f"Word加载器不可用: {e}")
        
        try:
            # 纯文本加载器
            from langchain_community.document_loaders import TextLoader
            loader_map['.txt'] = TextLoader
        except ImportError as e:
            logger.warning(f"文本加载器不可用: {e}")
        
        try:
            # HTML加载器
            from langchain_community.document_loaders import UnstructuredHTMLLoader
            loader_map['.html'] = UnstructuredHTMLLoader
            loader_map['.htm'] = UnstructuredHTMLLoader
        except ImportError as e:
            logger.warning(f"HTML加载器不可用: {e}")
        
        try:
            # Excel加载器
            from langchain_community.document_loaders import UnstructuredExcelLoader
            loader_map['.xlsx'] = UnstructuredExcelLoader
            loader_map['.xls'] = UnstructuredExcelLoader
        except ImportError as e:
            logger.warning(f"Excel加载器不可用: {e}")
        
        try:
            # PowerPoint加载器
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            loader_map['.pptx'] = UnstructuredPowerPointLoader
            loader_map['.ppt'] = UnstructuredPowerPointLoader
        except ImportError as e:
            logger.warning(f"PowerPoint加载器不可用: {e}")
        
        return loader_map
    
    def load_document(self, file_path: str) -> List[Document]:
        """加载单个文档"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        ext = Path(file_path).suffix.lower()
        
        if ext not in self.supported_extensions:
            raise ValueError(f"不支持的文档格式: {ext}")
        
        if ext not in self.loader_map:
            raise ValueError(f"没有可用的加载器处理 {ext} 格式")
        
        try:
            # 使用对应的加载器加载文档
            loader_class = self.loader_map[ext]
            
            # 特殊处理：某些加载器需要不同的参数
            if ext in ['.txt', '.md']:
                loader = loader_class(file_path, encoding='utf-8')
            else:
                loader = loader_class(file_path)
            
            documents = loader.load()
            
            # 为每个文档添加元数据
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                
                # 添加文件元数据
                file_metadata = extract_document_metadata(file_path)
                doc.metadata.update(file_metadata)
                
                # 添加内容哈希
                from ..utils import calculate_text_md5
                doc.metadata['content_hash'] = calculate_text_md5(doc.page_content)
            
            logger.info(f"成功加载文档: {file_path}, 共 {len(documents)} 页")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {e}")
            raise
    
    @timer
    def load_documents_from_directory(self, directory_path: str, 
                                    recursive: bool = True, 
                                    pattern: str = "**/*") -> List[Document]:
        """从目录加载所有文档"""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        all_documents = []
        file_count = 0
        error_files = []
        
        # 构建搜索模式
        if recursive:
            search_pattern = os.path.join(directory_path, pattern)
        else:
            search_pattern = os.path.join(directory_path, "*")
        
        # 获取所有文件
        all_files = glob.glob(search_pattern, recursive=recursive)
        
        logger.info(f"在目录 {directory_path} 中找到 {len(all_files)} 个文件")
        
        for file_path in all_files:
            # 检查是否为文件
            if not os.path.isfile(file_path):
                continue
            
            # 检查文件格式是否支持
            if not is_supported_document(file_path, self.supported_extensions):
                continue
            
            try:
                # 加载文档
                documents = self.load_document(file_path)
                all_documents.extend(documents)
                file_count += 1
                
                logger.debug(f"已处理: {file_path} ({len(documents)} 页)")
                
            except Exception as e:
                error_files.append((file_path, str(e)))
                logger.warning(f"处理文件失败 {file_path}: {e}")
        
        logger.info(f"成功加载 {file_count} 个文件，共 {len(all_documents)} 页文档")
        
        if error_files:
            logger.warning(f"有 {len(error_files)} 个文件处理失败")
            for file_path, error in error_files[:5]:  # 只显示前5个错误
                logger.warning(f"  - {file_path}: {error}")
        
        return all_documents
    
    def split_documents(self, documents: List[Document], 
                       text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        """分割文档为小块"""
        if not documents:
            return []
        
        if text_splitter is None:
            # 使用默认的分割器
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.document.chunk_size,
                chunk_overlap=self.config.document.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]
            )
        
        try:
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"文档分割完成: {len(documents)} -> {len(split_docs)} 块")
            return split_docs
        except Exception as e:
            logger.error(f"文档分割失败: {e}")
            raise
    
    def get_markdown_splitter(self) -> TextSplitter:
        """获取Markdown专用的分割器"""
        try:
            from langchain_text_splitters import MarkdownTextSplitter
            
            return MarkdownTextSplitter(
                chunk_size=self.config.document.chunk_size,
                chunk_overlap=self.config.document.chunk_overlap
            )
        except ImportError:
            logger.warning("MarkdownTextSplitter不可用，使用默认分割器")
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.document.chunk_size,
                chunk_overlap=self.config.document.chunk_overlap
            )
    
    def get_code_splitter(self) -> TextSplitter:
        """获取代码专用的分割器"""
        try:
            from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
            
            # 这里可以根据需要配置不同语言的分隔符
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.config.document.chunk_size,
                chunk_overlap=self.config.document.chunk_overlap
            )
        except ImportError:
            logger.warning("代码分割器不可用，使用默认分割器")
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.document.chunk_size,
                chunk_overlap=self.config.document.chunk_overlap
            )

# 单例模式
_document_loader_instance = None

def get_document_loader(config=None) -> DocumentLoader:
    """获取文档加载器单例"""
    global _document_loader_instance
    if _document_loader_instance is None:
        _document_loader_instance = DocumentLoader(config)
    return _document_loader_instance