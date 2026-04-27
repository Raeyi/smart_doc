"""
检索器 - 文档检索功能
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import numpy as np
import logging

from .utils import logger, timer
from .vector_store import get_vector_store_manager

class SmartRetriever:
    """智能检索器"""
    
    def __init__(self, config=None, llm=None):
        """初始化检索器"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import config as default_config
        
        self.config = config or default_config
        self.llm = llm
        self.vector_store_manager = get_vector_store_manager(config)
        self.vector_store = None
        self.retriever = None
        self.bm25_retriever = None
        
    def init_retriever(self, search_type: str = None, k: int = None):
        """初始化检索器"""
        if not self.vector_store:
            self.vector_store = self.vector_store_manager.init_vector_store()
        
        search_type = search_type or self.config.vector_store.search_type
        k = k or self.config.vector_store.k
        
        # 创建向量检索器
        if search_type == "similarity":
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        elif search_type == "mmr":
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 2}
            )
        else:
            raise ValueError(f"不支持的搜索类型: {search_type}")
        
        logger.info(f"初始化检索器: 类型={search_type}, k={k}")
        return self.retriever
    
    def init_bm25_retriever(self, documents: List[Document]):
        """初始化BM25检索器（基于关键词的检索）"""
        try:
            from langchain.retrievers import BM25Retriever
            
            # 提取文档文本
            texts = [doc.page_content for doc in documents]
            
            # 创建BM25检索器
            self.bm25_retriever = BM25Retriever.from_texts(
                texts,
                metadatas=[doc.metadata for doc in documents]
            )
            self.bm25_retriever.k = self.config.vector_store.k
            
            logger.info(f"初始化BM25检索器: 文档数={len(documents)}")
            return self.bm25_retriever
            
        except Exception as e:
            logger.error(f"初始化BM25检索器失败: {e}")
            return None
    
    def init_ensemble_retriever(self, documents: List[Document], 
                                weights: List[float] = None):
        """初始化集成检索器（结合向量检索和BM25）"""
        if not self.retriever:
            self.init_retriever()
        
        if not self.bm25_retriever:
            self.init_bm25_retriever(documents)
        
        if not self.bm25_retriever:
            logger.warning("BM25检索器初始化失败，仅使用向量检索")
            return self.retriever
        
        # 设置权重
        if weights is None:
            weights = [0.5, 0.5]  # 默认各占50%
        
        # 创建集成检索器
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.retriever, self.bm25_retriever],
            weights=weights
        )
        
        logger.info(f"初始化集成检索器: 权重={weights}")
        return ensemble_retriever
    
    def init_compression_retriever(self):
        """初始化压缩检索器（使用LLM压缩检索结果）"""
        if not self.llm:
            logger.warning("未提供LLM，无法初始化压缩检索器")
            return self.retriever
        
        if not self.retriever:
            self.init_retriever()
        
        # 创建压缩器
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        # 创建压缩检索器
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )
        
        logger.info("初始化压缩检索器")
        return compression_retriever
    
    @timer
    def retrieve(self, query: str, 
                 retriever_type: str = "vector",
                 k: int = None) -> List[Document]:
        """检索文档
        
        Args:
            query: 查询文本
            retriever_type: 检索器类型，可选: "vector", "bm25", "ensemble", "compression"
            k: 返回结果数量
            
        Returns:
            检索到的文档列表
        """
        k = k or self.config.vector_store.k
        
        try:
            if retriever_type == "vector":
                if not self.retriever:
                    self.init_retriever()
                results = self.retriever.get_relevant_documents(query)
                
            elif retriever_type == "bm25":
                if not self.bm25_retriever:
                    # 需要先初始化BM25检索器
                    logger.warning("BM25检索器未初始化，使用向量检索")
                    return self.retriever.get_relevant_documents(query)
                results = self.bm25_retriever.get_relevant_documents(query)
                
            elif retriever_type == "ensemble":
                # 创建临时集成检索器
                if not self.retriever:
                    self.init_retriever()
                if not self.bm25_retriever:
                    logger.warning("BM25检索器未初始化，使用向量检索")
                    return self.retriever.get_relevant_documents(query)
                
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.retriever, self.bm25_retriever],
                    weights=[0.5, 0.5]
                )
                results = ensemble_retriever.get_relevant_documents(query)
                
            elif retriever_type == "compression":
                if not self.llm:
                    logger.warning("未提供LLM，使用向量检索")
                    return self.retriever.get_relevant_documents(query)
                    
                compression_retriever = self.init_compression_retriever()
                results = compression_retriever.get_relevant_documents(query)
                
            else:
                raise ValueError(f"不支持的检索器类型: {retriever_type}")
            
            logger.info(f"检索完成: 查询='{query[:50]}...', 类型={retriever_type}, 结果数={len(results)}")
            return results[:k]  # 确保不超过k个结果
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    @timer
    def retrieve_with_scores(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """带相似度得分的检索"""
        k = k or self.config.vector_store.k
        
        try:
            # 使用向量存储的相似性搜索
            results_with_scores = self.vector_store_manager.similarity_search_with_score(query, k=k)
            
            # 过滤低分结果
            filtered_results = [
                (doc, score) for doc, score in results_with_scores
                if score >= self.config.vector_store.score_threshold
            ]
            
            logger.info(f"带得分检索完成: 查询='{query[:50]}...', 结果数={len(filtered_results)}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"带得分检索失败: {e}")
            return []
    
    def hybrid_retrieve(self, query: str, 
                        vector_weight: float = 0.7,
                        bm25_weight: float = 0.3,
                        k: int = None) -> List[Document]:
        """混合检索（向量 + BM25）"""
        k = k or self.config.vector_store.k
        
        try:
            # 向量检索结果
            vector_results = self.retrieve(query, retriever_type="vector", k=k*2)
            
            # BM25检索结果
            bm25_results = []
            if self.bm25_retriever:
                bm25_results = self.bm25_retriever.get_relevant_documents(query)
            
            # 合并结果
            all_results = {}
            
            # 添加向量结果
            for i, doc in enumerate(vector_results):
                doc_id = doc.metadata.get("content_hash", str(hash(doc.page_content)))
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        "doc": doc,
                        "score": vector_weight * (1.0 - i/len(vector_results)) if vector_results else 0
                    }
            
            # 添加BM25结果
            for i, doc in enumerate(bm25_results):
                doc_id = doc.metadata.get("content_hash", str(hash(doc.page_content)))
                if doc_id in all_results:
                    all_results[doc_id]["score"] += bm25_weight * (1.0 - i/len(bm25_results)) if bm25_results else 0
                else:
                    all_results[doc_id] = {
                        "doc": doc,
                        "score": bm25_weight * (1.0 - i/len(bm25_results)) if bm25_results else 0
                    }
            
            # 按分数排序
            sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
            
            # 返回前k个结果
            final_results = [item["doc"] for item in sorted_results[:k]]
            
            logger.info(f"混合检索完成: 查询='{query[:50]}...', 结果数={len(final_results)}")
            return final_results
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return self.retrieve(query, k=k)
    
    def rerank_results(self, query: str, documents: List[Document], 
                      reranker_type: str = "simple") -> List[Document]:
        """对检索结果进行重排序"""
        if not documents:
            return documents
        
        if reranker_type == "simple":
            # 简单的基于查询长度的重排序
            query_len = len(query)
            return sorted(documents, 
                         key=lambda x: len(x.page_content) / (abs(len(x.page_content) - query_len) + 1), 
                         reverse=True)
        
        elif reranker_type == "bm25":
            # 使用BM25分数重排序
            if not self.bm25_retriever:
                return documents
            
            # 重新计算BM25分数
            texts = [doc.page_content for doc in documents]
            scores = []
            
            for text in texts:
                # 这里简化处理，实际可以使用更复杂的BM25计算
                query_terms = set(query.lower().split())
                text_terms = set(text.lower().split())
                score = len(query_terms.intersection(text_terms)) / len(query_terms) if query_terms else 0
                scores.append(score)
            
            # 按分数排序
            sorted_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
            return sorted_docs
        
        else:
            logger.warning(f"不支持的reranker类型: {reranker_type}")
            return documents

# 单例模式
_retriever_instance = None

def get_retriever(config=None, llm=None) -> SmartRetriever:
    """获取检索器单例"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = SmartRetriever(config, llm)
    return _retriever_instance