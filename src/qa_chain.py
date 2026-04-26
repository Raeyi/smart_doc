"""
问答链 - 基于文档的智能问答
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import logging

from .utils import logger, timer
from .retriever import SmartRetriever

class DocumentQA:
    """文档问答系统"""
    
    def __init__(self, config=None, llm=None, retriever=None):
        """初始化问答系统"""
        from ..config import config as default_config
        
        self.config = config or default_config
        self.llm = llm
        self.retriever = retriever or get_retriever(config, llm)
        self.qa_chain = None
        self.conversation_chain = None
        self.memory = None
        
        # 初始化提示模板
        self.prompt_templates = self._init_prompt_templates()
    
    def _init_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """初始化提示模板"""
        # 默认提示模板
        default_template = """基于以下上下文，请用中文回答最后的问题。如果你不知道答案，就说你不知道，不要编造答案。

上下文:
{context}

问题: {question}
答案:"""
        
        # 详细回答模板
        detailed_template = """你是一个专业的企业知识库助手。请基于提供的上下文信息，用中文详细、准确地回答用户的问题。

上下文信息:
{context}

请根据以上上下文，回答以下问题。如果上下文没有提供足够的信息，请如实说明。

问题: {question}
回答:"""
        
        # 简洁回答模板
        concise_template = """请根据以下上下文，用一句话简洁地回答这个问题。

上下文: {context}

问题: {question}
回答:"""
        
        # 代码相关模板
        code_template = """你是一个编程专家。请基于以下代码示例和文档，回答问题。

{context}

问题: {question}
答案（请提供代码示例和解释）:"""
        
        templates = {
            "default": PromptTemplate(
                template=default_template,
                input_variables=["context", "question"]
            ),
            "detailed": PromptTemplate(
                template=detailed_template,
                input_variables=["context", "question"]
            ),
            "concise": PromptTemplate(
                template=concise_template,
                input_variables=["context", "question"]
            ),
            "code": PromptTemplate(
                template=code_template,
                input_variables=["context", "question"]
            )
        }
        
        return templates
    
    def init_qa_chain(self, chain_type: str = "stuff", 
                      prompt_template: str = "default") -> RetrievalQA:
        """初始化问答链"""
        if not self.llm:
            raise ValueError("未提供LLM，无法初始化问答链")
        
        if not self.retriever.retriever:
            self.retriever.init_retriever()
        
        # 选择提示模板
        if prompt_template in self.prompt_templates:
            prompt = self.prompt_templates[prompt_template]
        else:
            prompt = self.prompt_templates["default"]
        
        # 创建问答链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info(f"初始化问答链: chain_type={chain_type}, prompt_template={prompt_template}")
        return self.qa_chain
    
    def init_conversation_chain(self, memory_length: int = 5) -> ConversationalRetrievalChain:
        """初始化对话链（支持多轮对话）"""
        if not self.llm:
            raise ValueError("未提供LLM，无法初始化对话链")
        
        if not self.retriever.retriever:
            self.retriever.init_retriever()
        
        # 初始化记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # 对话提示模板
        condense_question_template = """给定以下对话和后续问题，将后续问题重写为独立的问题。

聊天历史:
{chat_history}

后续输入: {question}
独立的问题:"""
        
        condense_question_prompt = PromptTemplate.from_template(condense_question_template)
        
        # 创建对话链
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever.retriever,
            memory=self.memory,
            condense_question_prompt=condense_question_prompt,
            return_source_documents=True,
            verbose=self.config.app.debug
        )
        
        logger.info(f"初始化对话链: memory_length={memory_length}")
        return self.conversation_chain
    
    @timer
    def ask(self, question: str, 
            chain_type: str = "stuff",
            prompt_template: str = "default",
            k: int = None) -> Dict[str, Any]:
        """提问并获取答案
        
        Args:
            question: 问题
            chain_type: 链类型，可选: "stuff", "map_reduce", "refine", "map_rerank"
            prompt_template: 提示模板，可选: "default", "detailed", "concise", "code"
            k: 检索文档数量
            
        Returns:
            包含答案和来源的字典
        """
        if not self.qa_chain:
            self.init_qa_chain(chain_type, prompt_template)
        
        try:
            # 设置检索数量
            if k and hasattr(self.retriever.retriever, 'search_kwargs'):
                self.retriever.retriever.search_kwargs['k'] = k
            
            # 执行问答
            result = self.qa_chain({"query": question})
            
            # 提取结果
            answer = result.get("result", "抱歉，我无法回答这个问题。")
            source_docs = result.get("source_documents", [])
            
            # 处理来源文档
            sources = []
            for doc in source_docs:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            response = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "source_count": len(sources)
            }
            
            logger.info(f"问答完成: 问题='{question[:50]}...', 答案长度={len(answer)}")
            return response
            
        except Exception as e:
            logger.error(f"问答失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，处理问题时发生错误: {str(e)}",
                "sources": [],
                "source_count": 0,
                "error": str(e)
            }
    
    @timer
    def ask_with_context(self, question: str, 
                        context: List[Document] = None,
                        prompt_template: str = "default") -> Dict[str, Any]:
        """基于给定上下文提问（不进行检索）"""
        if not self.llm:
            raise ValueError("未提供LLM，无法回答问题")
        
        if not context:
            # 如果没有提供上下文，则进行检索
            context = self.retriever.retrieve(question)
        
        if not context:
            return {
                "question": question,
                "answer": "抱歉，我没有找到相关的信息来回答这个问题。",
                "sources": [],
                "source_count": 0
            }
        
        try:
            # 选择提示模板
            if prompt_template in self.prompt_templates:
                prompt = self.prompt_templates[prompt_template]
            else:
                prompt = self.prompt_templates["default"]
            
            # 创建QA链
            qa_chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
            
            # 准备上下文
            context_text = "\n\n".join([doc.page_content for doc in context])
            
            # 执行问答
            result = qa_chain(
                {"input_documents": context, "question": question},
                return_only_outputs=True
            )
            
            answer = result.get("output_text", "抱歉，我无法回答这个问题。")
            
            # 处理来源文档
            sources = []
            for doc in context:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            response = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "source_count": len(sources)
            }
            
            logger.info(f"基于上下文问答完成: 问题='{question[:50]}...', 上下文文档数={len(context)}")
            return response
            
        except Exception as e:
            logger.error(f"基于上下文问答失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，处理问题时发生错误: {str(e)}",
                "sources": [],
                "source_count": 0,
                "error": str(e)
            }
    
    @timer
    def chat(self, question: str, 
             chat_history: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """多轮对话"""
        if not self.conversation_chain:
            self.init_conversation_chain()
        
        try:
            # 如果有聊天历史，设置到记忆中
            if chat_history and self.memory:
                self.memory.clear()
                for human_msg, ai_msg in chat_history[-5:]:  # 只保留最近5轮
                    self.memory.save_context({"input": human_msg}, {"output": ai_msg})
            
            # 执行对话
            result = self.conversation_chain({"question": question})
            
            # 提取结果
            answer = result.get("answer", "抱歉，我无法回答这个问题。")
            source_docs = result.get("source_documents", [])
            
            # 处理来源文档
            sources = []
            for doc in source_docs:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            response = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "source_count": len(sources),
                "chat_history": chat_history or []
            }
            
            logger.info(f"对话完成: 问题='{question[:50]}...'")
            return response
            
        except Exception as e:
            logger.error(f"对话失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，处理对话时发生错误: {str(e)}",
                "sources": [],
                "source_count": 0,
                "error": str(e)
            }
    
    def clear_memory(self):
        """清除对话记忆"""
        if self.memory:
            self.memory.clear()
            logger.info("已清除对话记忆")
    
    def get_prompt_templates(self) -> Dict[str, str]:
        """获取可用的提示模板"""
        return {name: template.template[:100] + "..." for name, template in self.prompt_templates.items()}
    
    def add_custom_prompt_template(self, name: str, template: str):
        """添加自定义提示模板"""
        self.prompt_templates[name] = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        logger.info(f"已添加自定义提示模板: {name}")

# 单例模式
_qa_instance = None

def get_qa_system(config=None, llm=None, retriever=None) -> DocumentQA:
    """获取问答系统单例"""
    global _qa_instance
    if _qa_instance is None:
        _qa_instance = DocumentQA(config, llm, retriever)
    return _qa_instance