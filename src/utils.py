"""
工具函数
"""

import os
import sys
import logging
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import functools
from tqdm import tqdm

# 配置日志
def setup_logger(name: str = "smart_doc", log_file: str = None, level=logging.INFO):
    """设置日志记录器"""
    
    # 创建日志目录
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 创建文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_handler.setFormatter(formatter)
    if log_file:
        file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    if log_file:
        logger.addHandler(file_handler)
    
    return logger

# 计算文件MD5
def calculate_file_md5(file_path: str) -> str:
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# 计算文本MD5
def calculate_text_md5(text: str) -> str:
    """计算文本的MD5哈希值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# 文件扩展名判断
def get_file_extension(file_path: str) -> str:
    """获取文件扩展名（小写）"""
    return Path(file_path).suffix.lower()

# 文档类型判断
def is_supported_document(file_path: str, supported_extensions: List[str]) -> bool:
    """检查文件是否为支持的文档类型"""
    ext = get_file_extension(file_path)
    return ext in supported_extensions

# 读取文档文本
def read_text_file(file_path: str) -> str:
    """读取文本文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 写入JSON文件
def write_json(data: Any, file_path: str, indent: int = 2):
    """写入JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

# 读取JSON文件
def read_json(file_path: str) -> Any:
    """读取JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 计时装饰器
def timer(func):
    """函数计时装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger = logging.getLogger(func.__module__)
        logger.debug(f"函数 {func.__name__} 执行时间: {elapsed_time:.4f}秒")
        return result
    return wrapper

# 批量处理进度显示
def batch_process_with_progress(items: List[Any], process_func, desc: str = "处理中"):
    """批量处理并显示进度"""
    results = []
    for item in tqdm(items, desc=desc):
        try:
            result = process_func(item)
            results.append(result)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"处理失败: {e}")
            results.append(None)
    return results

# 清理文本
def clean_text(text: str) -> str:
    """清理文本，移除多余空白字符"""
    if not text:
        return ""
    
    # 替换各种空白字符为单个空格
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # 移除首尾空白
    text = text.strip()
    
    return text

# 提取文档元数据
def extract_document_metadata(file_path: str) -> Dict[str, Any]:
    """从文件路径提取元数据"""
    path = Path(file_path)
    
    metadata = {
        'source': str(path.absolute()),
        'filename': path.name,
        'extension': path.suffix.lower(),
        'directory': str(path.parent),
        'size': path.stat().st_size if path.exists() else 0,
        'created_time': datetime.fromtimestamp(path.stat().st_ctime).isoformat() if path.exists() else None,
        'modified_time': datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
    }
    
    return metadata

# 配置文件加载
def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置文件"""
    if config_path and os.path.exists(config_path):
        return read_json(config_path)
    else:
        # 返回默认配置
        return {
            "app": {
                "name": "SmartDoc",
                "version": "1.0.0"
            },
            "model": {
                "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            }
        }

# 异常处理装饰器
def handle_exceptions(default_return=None):
    """异常处理装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = logging.getLogger(func.__module__)
                logger.error(f"函数 {func.__name__} 执行错误: {e}")
                if default_return is not None:
                    return default_return
                raise
        return wrapper
    return decorator

# 创建全局日志记录器
logger = setup_logger("smart_doc", level=logging.INFO)