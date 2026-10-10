# src/routing/router.py
"""
Day 2 路由链路实现

- skip:      跳过检索（闲聊、问候、超范围）
- exact:     精确查询（术语/符号/公式，走 BM25）
- semantic:  语义查询（概念理解，走向量）
- hybrid:    混合查询（对比/步骤/综合，走 BM25+向量）
"""

import time
from enum import Enum

from src.routing.query_classifier import classify_query
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


class RouteTarget(str, Enum):
    """路由目标枚举：4 条路线"""
    SKIP = "skip"
    EXACT = "exact"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


# 分类结果 -> 路由目标 的映射表
# 注意：分类器有 5 种类别，但路由只有 4 条，out-of-scope 也走 skip
ROUTING_TABLE = {
    "skip": RouteTarget.SKIP,
    "out-of-scope": RouteTarget.SKIP,   # 超范围，直接拒答
    "exact": RouteTarget.EXACT,
    "semantic": RouteTarget.SEMANTIC,
    "hybrid": RouteTarget.HYBRID,
}


class Router:
    def __init__(self, collection):
        """
        collection: Chroma collection 对象
        在初始化时创建三个检索器，避免每次请求重复加载模型/建索引
        """
        self.bm25 = BM25Retriever(collection)          # 默认 config_name="fixed_512_0"
        self.vector = VectorRetriever(collection)      # 默认 config_name="recursive_512_0"
        self.hybrid = HybridRetriever(collection)      # 默认 config_name="recursive_512_0"

    # ---------- 4 条路由方法 ----------

    def route_skip(self, query: str):
        """跳过检索：闲聊、问候、超范围问题，直接返回固定回复"""
        return {
            "route": "skip",
            "query": query,
            "answer": "这个问题不需要查资料，我只是个检索助手哦。",
            "results": [],
        }

    def route_exact(self, query: str, top_k: int = 5):
        """精确查询：走 BM25"""
        results = self.bm25.retrieve(query, top_k=top_k)
        return {
            "route": "exact",
            "query": query,
            "results": results,
        }

    def route_semantic(self, query: str, top_k: int = 5):
        """语义查询：走向量检索"""
        results = self.vector.retrieve(query, top_k=top_k)
        return {
            "route": "semantic",
            "query": query,
            "results": results,
        }

    def route_hybrid(self, query: str, top_k: int = 5):
        """混合查询：走 RRF 混合检索"""
        results = self.hybrid.retrieve(query, top_k=top_k)
        return {
            "route": "hybrid",
            "query": query,
            "results": results,
        }

    # ---------- 主函数：分类 -> 查表 -> 分发 -> 返回 ----------

    def route(self, query: str, top_k: int = 5, force_route: str = None):
        """
        force_route: 若不为 None，跳过分类器，强制走指定路由
                 取值: "skip" / "exact" / "semantic" / "hybrid"
        """
        t0 = time.perf_counter()

        # 1. 分类 或 强制路由
        if force_route is not None:
            category = "forced"
            target = ROUTING_TABLE.get(force_route, RouteTarget.SKIP)
        else:
            classification = classify_query(query)
            category = classification.query_type
            target = ROUTING_TABLE.get(category, RouteTarget.SKIP)

        # 路由决策耗时
        t1 = time.perf_counter()
        routing_latency_ms = (t1 - t0) * 1000

       # 2. 分发
        if target == RouteTarget.SKIP:
            result = self.route_skip(query)
        elif target == RouteTarget.EXACT:
            result = self.route_exact(query, top_k=top_k)
        elif target == RouteTarget.SEMANTIC:
            result = self.route_semantic(query, top_k=top_k)
        elif target == RouteTarget.HYBRID:
            result = self.route_hybrid(query, top_k=top_k)
        else:
            result = self.route_skip(query)

        t2 = time.perf_counter()
        total_latency_ms = (t2 - t0) * 1000

        # 3. 附加日志字段
        result["category"] = category
        result["routing_latency_ms"] = routing_latency_ms
        result["total_latency_ms"] = total_latency_ms

        return result