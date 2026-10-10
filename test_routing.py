# test_routing.py
import chromadb
from src.routing.router import Router

# 1. 连接 Chroma
client = chromadb.PersistentClient(path="chroma_db")

# 2. 打印所有集合名（帮你确认集合名）
collections = client.list_collections()
print("可用的集合有：")
for c in collections:
    print(" -", c.name)

# 3. 拿到你的 collection（把 "你的集合名" 改成上面打印出来的名字）
collection = client.get_collection("day4_chunking_eval")

# 4. 创建 Router
router = Router(collection)

# 5. 准备 12 个测试 query（4 条路由 × 3 个）
test_queries = [
    # skip 类
    "你好，你是谁？",
    "今天天气怎么样？",
    "谢谢你的帮助",
    # exact 类
    "MDP 的五元组是什么？",
    "折扣因子 γ 代表什么？",
    "价值函数的公式如何表示？",
    # semantic 类
    "什么是马尔可夫过程？",
    "为什么要引入折扣因子？",
    "如何理解贝尔曼方程的意义？",
    # hybrid 类
    "价值迭代和策略迭代有哪些区别？",
    "描述 MDP 的动态过程。",
    "价值迭代和策略迭代的关系是什么？",
]

# 6. 跑测试并打印日志
print("\n===== 路由测试开始 =====")
for q in test_queries:
    result = router.route(q)
    print(f"\nquery: {q}")
    print(f"  category: {result['category']}")
    print(f"  route: {result['route']}")
    print(f"  routing_latency_ms: {result['routing_latency_ms']:.2f} ms")
    print(f"  total_latency_ms: {result['total_latency_ms']:.2f} ms")
    print(f"  results_count: {len(result['results'])}")

# 5. 变式测试：强制路由
print("\n===== 强制路由测试 =====")
result = router.route("什么是马尔可夫过程？", force_route="exact")
print(f"query: 什么是马尔可夫过程？ (force_route=exact)")
print(f"  category: {result['category']}")
print(f"  route: {result['route']}")
print(f"  routing_latency_ms: {result['routing_latency_ms']:.2f} ms")
print(f"  results_count: {len(result['results'])}")