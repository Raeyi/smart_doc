"""
企业文档智能平台 - 主程序
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.utils import setup_logger, logger, timer
from src.document_loader import DocumentLoader, get_document_loader
from src.vector_store import VectorStoreManager, get_vector_store_manager
from src.retriever import SmartRetriever, get_retriever
from src.qa_chain import DocumentQA, get_qa_system

# 初始化LLM
def init_llm(config):
    """初始化语言模型"""
    try:
        if config.model.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            
            if not config.model.openai_api_key:
                raise ValueError("OpenAI API Key未配置，请在config.py中设置或设置环境变量OPENAI_API_KEY")
            
            llm = ChatOpenAI(
                model=config.model.openai_model,
                temperature=config.model.openai_temperature,
                max_tokens=config.model.max_tokens,
                openai_api_key=config.model.openai_api_key
            )
            logger.info(f"使用OpenAI模型: {config.model.openai_model}")
            
        elif config.model.llm_provider == "azure":
            from langchain_openai import AzureChatOpenAI
            
            llm = AzureChatOpenAI(
                deployment_name=config.model.azure_deployment_name,
                openai_api_version=config.model.azure_api_version,
                openai_api_key=config.model.azure_api_key,
                azure_endpoint=config.model.azure_endpoint,
                temperature=config.model.openai_temperature,
                max_tokens=config.model.max_tokens
            )
            logger.info(f"使用Azure OpenAI模型: {config.model.azure_deployment_name}")
            
        elif config.model.llm_provider == "local":
            # 这里可以根据具体模型类型进行初始化
            if config.model.local_model_type == "llama":
                from langchain_community.llms import LlamaCpp
                
                llm = LlamaCpp(
                    model_path=config.model.local_model_path,
                    temperature=config.model.openai_temperature,
                    max_tokens=config.model.max_tokens,
                    n_ctx=2048,  # 上下文长度
                    verbose=False
                )
                logger.info(f"使用本地Llama模型: {config.model.local_model_path}")
                
            elif config.model.local_model_type == "chatglm":
                from langchain_community.llms import ChatGLM
                
                llm = ChatGLM(
                    endpoint_url=config.model.local_model_path,
                    max_token=config.model.max_tokens,
                    temperature=config.model.openai_temperature
                )
                logger.info(f"使用ChatGLM模型: {config.model.local_model_path}")
                
            else:
                raise ValueError(f"不支持的本地模型类型: {config.model.local_model_type}")
                
        else:
            raise ValueError(f"不支持的LLM提供商: {config.model.llm_provider}")
        
        return llm
        
    except Exception as e:
        logger.error(f"初始化LLM失败: {e}")
        logger.info("将使用检索功能，问答功能不可用")
        return None

class SmartDocPlatform:
    """企业文档智能平台主类"""
    
    def __init__(self, config_path: str = None):
        """初始化平台"""
        # 加载配置
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            # 这里可以添加配置合并逻辑
            logger.info(f"从 {config_path} 加载配置")
        
        # 从环境变量加载配置
        config.load_env_variables()
        
        # 设置日志
        global logger
        logger = setup_logger("smart_doc", config.app.log_file, config.app.log_level)
        
        self.config = config
        self.llm = None
        self.document_loader = None
        self.vector_store_manager = None
        self.retriever = None
        self.qa_system = None
        
        logger.info(f"初始化 {config.app.app_name} v{config.app.version}")
    
    def init_components(self, init_llm: bool = True):
        """初始化所有组件"""
        logger.info("初始化组件...")
        
        # 1. 初始化文档加载器
        self.document_loader = get_document_loader(self.config)
        logger.info("文档加载器初始化完成")
        
        # 2. 初始化向量存储
        self.vector_store_manager = get_vector_store_manager(self.config)
        self.vector_store_manager.init_vector_store()
        logger.info("向量存储初始化完成")
        
        # 3. 初始化LLM（可选）
        if init_llm:
            self.llm = init_llm(self.config)
            if self.llm:
                logger.info("LLM初始化完成")
            else:
                logger.warning("LLM初始化失败，问答功能将不可用")
        
        # 4. 初始化检索器
        self.retriever = get_retriever(self.config, self.llm)
        self.retriever.init_retriever()
        logger.info("检索器初始化完成")
        
        # 5. 初始化问答系统
        if self.llm:
            self.qa_system = get_qa_system(self.config, self.llm, self.retriever)
            self.qa_system.init_qa_chain()
            logger.info("问答系统初始化完成")
        
        logger.info("所有组件初始化完成")
    
    @timer
    def ingest_documents(self, directory_path: str = None, recursive: bool = True) -> Dict[str, Any]:
        """摄取文档到向量存储"""
        if not directory_path:
            directory_path = self.config.document.docs_dir
        
        if not os.path.exists(directory_path):
            logger.error(f"文档目录不存在: {directory_path}")
            return {"success": False, "error": f"目录不存在: {directory_path}"}
        
        try:
            logger.info(f"开始摄取文档: {directory_path}")
            
            # 添加文档到向量存储
            doc_ids = self.vector_store_manager.add_documents_from_directory(
                directory_path, recursive
            )
            
            # 获取统计信息
            stats = self.vector_store_manager.get_stats()
            
            result = {
                "success": True,
                "directory": directory_path,
                "document_count": len(doc_ids),
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"文档摄取完成: 添加了 {len(doc_ids)} 个文档块")
            return result
            
        except Exception as e:
            logger.error(f"文档摄取失败: {e}")
            return {"success": False, "error": str(e)}
    
    @timer
    def search_documents(self, query: str, 
                        search_type: str = "similarity",
                        k: int = None) -> Dict[str, Any]:
        """搜索文档"""
        try:
            logger.info(f"搜索文档: '{query}'")
            
            if search_type == "similarity":
                results = self.retriever.retrieve(query, "vector", k)
            elif search_type == "mmr":
                results = self.retriever.max_marginal_relevance_search(query, k)
            elif search_type == "hybrid":
                results = self.retriever.hybrid_retrieve(query, k=k)
            else:
                results = self.retriever.retrieve(query, "vector", k)
            
            # 格式化结果
            formatted_results = []
            for i, doc in enumerate(results):
                formatted_result = {
                    "rank": i + 1,
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata,
                    "length": len(doc.page_content)
                }
                formatted_results.append(formatted_result)
            
            result = {
                "success": True,
                "query": query,
                "search_type": search_type,
                "result_count": len(results),
                "results": formatted_results,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"搜索完成: 找到 {len(results)} 个结果")
            return result
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return {"success": False, "error": str(e)}
    
    @timer
    def ask_question(self, question: str, 
                     chain_type: str = "stuff",
                     prompt_template: str = "default",
                     k: int = None) -> Dict[str, Any]:
        """提问"""
        if not self.qa_system:
            return {
                "success": False,
                "error": "问答系统未初始化，请先初始化LLM"
            }
        
        try:
            logger.info(f"提问: '{question}'")
            
            result = self.qa_system.ask(question, chain_type, prompt_template, k)
            result["success"] = True
            result["timestamp"] = datetime.now().isoformat()
            
            logger.info(f"问答完成: 答案长度={len(result.get('answer', ''))}")
            return result
            
        except Exception as e:
            logger.error(f"提问失败: {e}")
            return {"success": False, "error": str(e)}
    
    @timer
    def chat(self, question: str, 
             chat_history: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """对话"""
        if not self.qa_system:
            return {
                "success": False,
                "error": "问答系统未初始化，请先初始化LLM"
            }
        
        try:
            logger.info(f"对话: '{question}'")
            
            result = self.qa_system.chat(question, chat_history)
            result["success"] = True
            result["timestamp"] = datetime.now().isoformat()
            
            logger.info(f"对话完成")
            return result
            
        except Exception as e:
            logger.error(f"对话失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            vector_stats = self.vector_store_manager.get_stats() if self.vector_store_manager else {}
            
            stats = {
                "app": {
                    "name": self.config.app.app_name,
                    "version": self.config.app.version
                },
                "vector_store": vector_stats,
                "llm_available": self.llm is not None,
                "llm_provider": self,
                "llm_available": self.llm is not None,
                "llm_provider": self.config.model.llm_provider if self.llm else None,
                "timestamp": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"success": False, "error": str(e)}
    
    def clear_vector_store(self) -> Dict[str, Any]:
        """清空向量存储"""
        try:
            success = self.vector_store_manager.clear_vector_store()
            
            if success:
                result = {
                    "success": True,
                    "message": "向量存储已清空",
                    "timestamp": datetime.now().isoformat()
                }
                logger.info("向量存储已清空")
            else:
                result = {
                    "success": False,
                    "error": "清空向量存储失败",
                    "timestamp": datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"清空向量存储失败: {e}")
            return {"success": False, "error": str(e)}
    
    def export_documents(self, output_dir: str = "./exports") -> Dict[str, Any]:
        """导出文档信息"""
        try:
            import json
            from pathlib import Path
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # 获取所有文档
            if hasattr(self.vector_store_manager.vector_store, '_collection'):
                collection = self.vector_store_manager.vector_store._collection
                
                # 获取所有文档
                results = collection.get(include=["metadatas", "documents"])
                
                documents = []
                for i, (metadata, content) in enumerate(zip(results["metadatas"], results["documents"])):
                    doc_info = {
                        "id": i + 1,
                        "content_preview": content[:200] + "..." if len(content) > 200 else content,
                        "content_length": len(content),
                        "metadata": metadata
                    }
                    documents.append(doc_info)
                
                # 保存为JSON
                output_file = os.path.join(output_dir, f"documents_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(documents, f, ensure_ascii=False, indent=2)
                
                result = {
                    "success": True,
                    "export_file": output_file,
                    "document_count": len(documents),
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"文档导出完成: {output_file}, 共 {len(documents)} 个文档")
                return result
            
            else:
                return {
                    "success": False,
                    "error": "当前向量存储不支持导出功能",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"导出文档失败: {e}")
            return {"success": False, "error": str(e)}
    
    def interactive_mode(self):
        """交互模式"""
        print(f"\n{'='*60}")
        print(f"欢迎使用 {self.config.app.app_name} v{self.config.app.version}")
        print(f"{'='*60}\n")
        
        # 检查LLM是否可用
        if not self.llm:
            print("⚠️  警告: LLM未初始化，问答功能不可用")
            print("   请在config.py中配置LLM或设置环境变量\n")
        
        while True:
            print("\n请选择操作:")
            print("1. 搜索文档")
            print("2. 提问（需要LLM）")
            print("3. 对话（需要LLM）")
            print("4. 摄取文档")
            print("5. 查看系统状态")
            print("6. 清空向量存储")
            print("7. 退出")
            
            try:
                choice = input("\n请输入选项 (1-7): ").strip()
                
                if choice == "1":
                    query = input("请输入搜索关键词: ").strip()
                    if query:
                        search_type = input("搜索类型 (similarity/mmr/hybrid, 默认similarity): ").strip() or "similarity"
                        k = input("返回结果数量 (默认4): ").strip()
                        k = int(k) if k.isdigit() else 4
                        
                        result = self.search_documents(query, search_type, k)
                        
                        if result["success"]:
                            print(f"\n🔍 找到 {result['result_count']} 个结果:")
                            for i, res in enumerate(result["results"]):
                                print(f"\n{i+1}. {'-'*50}")
                                print(f"内容: {res['content']}")
                                print(f"来源: {res['metadata'].get('source', '未知')}")
                                print(f"长度: {res['length']} 字符")
                        else:
                            print(f"❌ 搜索失败: {result.get('error', '未知错误')}")
                
                elif choice == "2":
                    if not self.llm:
                        print("❌ 问答功能需要LLM支持，请先配置LLM")
                        continue
                    
                    question = input("请输入问题: ").strip()
                    if question:
                        result = self.ask_question(question)
                        
                        if result["success"]:
                            print(f"\n🤖 答案: {result['answer']}")
                            
                            if result["sources"]:
                                print(f"\n📚 参考来源 ({result['source_count']} 个):")
                                for i, source in enumerate(result["sources"]):
                                    print(f"\n{i+1}. 内容: {source['content']}")
                                    print(f"   文件: {source['metadata'].get('source', '未知')}")
                        else:
                            print(f"❌ 提问失败: {result.get('error', '未知错误')}")
                
                elif choice == "3":
                    if not self.llm:
                        print("❌ 对话功能需要LLM支持，请先配置LLM")
                        continue
                    
                    print("\n💬 对话模式 (输入 '退出' 结束对话)")
                    
                    chat_history = []
                    while True:
                        question = input("\n你: ").strip()
                        
                        if question.lower() in ["退出", "exit", "quit"]:
                            break
                        
                        if question:
                            result = self.chat(question, chat_history)
                            
                            if result["success"]:
                                print(f"\n助手: {result['answer']}")
                                # 添加到历史
                                chat_history.append((question, result["answer"]))
                            else:
                                print(f"❌ 对话失败: {result.get('error', '未知错误')}")
                
                elif choice == "4":
                    directory = input("请输入文档目录路径 (默认: ./docs): ").strip() or "./docs"
                    recursive = input("是否递归处理子目录? (y/n, 默认y): ").strip().lower() != "n"
                    
                    result = self.ingest_documents(directory, recursive)
                    
                    if result["success"]:
                        print(f"\n✅ 文档摄取完成: 添加了 {result['document_count']} 个文档块")
                        print(f"   向量存储统计: {json.dumps(result['stats'], ensure_ascii=False, indent=2)}")
                    else:
                        print(f"❌ 文档摄取失败: {result.get('error', '未知错误')}")
                
                elif choice == "5":
                    stats = self.get_stats()
                    
                    if isinstance(stats, dict) and "success" not in stats:
                        print(f"\n📊 系统状态:")
                        print(f"应用: {stats.get('app', {}).get('name')} v{stats.get('app', {}).get('version')}")
                        print(f"向量存储类型: {stats.get('vector_store', {}).get('vector_store_type')}")
                        print(f"文档数量: {stats.get('vector_store', {}).get('total_documents', 0)}")
                        print(f"嵌入模型: {stats.get('vector_store', {}).get('embedding_model')}")
                        print(f"LLM可用: {'是' if stats.get('llm_available') else '否'}")
                        if stats.get('llm_available'):
                            print(f"LLM提供商: {stats.get('llm_provider')}")
                    else:
                        print(f"❌ 获取系统状态失败: {stats.get('error', '未知错误')}")
                
                elif choice == "6":
                    confirm = input("⚠️  警告: 这将清空所有向量数据，确定吗? (y/n): ").strip().lower()
                    if confirm == "y":
                        result = self.clear_vector_store()
                        if result["success"]:
                            print("✅ 向量存储已清空")
                        else:
                            print(f"❌ 清空失败: {result.get('error', '未知错误')}")
                    else:
                        print("操作已取消")
                
                elif choice == "7":
                    print("\n👋 再见！")
                    break
                
                else:
                    print("❌ 无效选项，请重新选择")
            
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                logger.error(f"交互模式错误: {e}")
                print(f"❌ 发生错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description=f"{config.app.app_name} - 企业文档智能平台")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--mode", "-m", choices=["cli", "web", "ingest", "search", "ask", "interactive"], 
                       default="interactive", help="运行模式")
    parser.add_argument("--directory", "-d", help="文档目录路径")
    parser.add_argument("--query", "-q", help="搜索或提问的内容")
    parser.add_argument("--question", help="提问的问题（同--query）")
    parser.add_argument("--search-type", choices=["similarity", "mmr", "hybrid"], default="similarity", 
                       help="搜索类型")
    parser.add_argument("--k", type=int, default=4, help="返回结果数量")
    parser.add_argument("--chain-type", choices=["stuff", "map_reduce", "refine", "map_rerank"], 
                       default="stuff", help="问答链类型")
    parser.add_argument("--prompt", default="default", help="提示模板")
    parser.add_argument("--port", type=int, default=8000, help="Web服务器端口")
    parser.add_argument("--host", default="0.0.0.0", help="Web服务器主机")
    parser.add_argument("--clear", action="store_true", help="清空向量存储")
    parser.add_argument("--stats", action="store_true", help="显示系统状态")
    parser.add_argument("--export", help="导出文档到指定目录")
    parser.add_argument("--recursive", action="store_true", default=True, help="递归处理子目录")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive", 
                       help="不递归处理子目录")
    
    args = parser.parse_args()
    
    # 初始化平台
    platform = SmartDocPlatform(args.config)
    
    # 初始化组件
    platform.init_components(init_llm=True)
    
    # 根据模式执行相应操作
    if args.clear:
        result = platform.clear_vector_store()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.stats:
        stats = platform.get_stats()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    elif args.export:
        result = platform.export_documents(args.export)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.mode == "ingest":
        if args.directory:
            result = platform.ingest_documents(args.directory, args.recursive)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("错误: 请使用 --directory 指定文档目录")
    
    elif args.mode == "search":
        if args.query:
            result = platform.search_documents(args.query, args.search_type, args.k)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("错误: 请使用 --query 指定搜索内容")
    
    elif args.mode == "ask":
        question = args.query or args.question
        if question:
            result = platform.ask_question(question, args.chain_type, args.prompt, args.k)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("错误: 请使用 --query 或 --question 指定问题")
    
    elif args.mode == "web":
        # 启动Web服务器
        from web.app import create_app
        app = create_app(platform)
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    
    elif args.mode == "cli" or args.mode == "interactive":
        platform.interactive_mode()

if __name__ == "__main__":
    main()