# src/agent/tools.py
from langchain_core.tools import tool
from src.api.rag_chain import rag_query


@tool
def search_pdf(query: str) -> str:
    """当用户的问题涉及强化学习、马尔可夫决策过程、价值函数、
    策略迭代、占用度量等 PDF 内容时，使用此工具检索 PDF 知识库并生成答案。
    
    Args:
        query: 用户提出的问题，保持原样传入即可。
    
    Returns:
        基于 PDF 内容生成的回答。
    """
    result = rag_query(query)
    return result["answer"]