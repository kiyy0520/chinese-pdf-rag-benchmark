import os
import time
import json
import logging
from typing import Dict, Any, Tuple, List
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# 【修复1】静音 transformers 和 huggingface 的 HTTP 请求日志
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from src.retrieval.hybrid_retriever import HybridRetriever

# 只保留我们自己的 INFO 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING) # 屏蔽 openai 的 http 日志
logger = logging.getLogger(__name__)

load_dotenv()

class MultiTurnRAGPipeline:
    def __init__(self, db_path: str = "chroma_db", collection_name: str = "day4_chunking_eval"):
        self.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未在 .env 中找到 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
            
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        
        chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = chroma_client.get_collection(name=collection_name)
        
        self.retriever = HybridRetriever(
            self.collection, 
            config_name="recursive_512_0",
            rrf_k=60,
            weights=(1.0, 1.0)
        )
        
        self.history = []
        self.current_state = {}
        
        self.retrieval_threshold = 0.01 
        self.generation_threshold = 0.6

    def process_turn(self, user_query: str) -> Dict[str, Any]:
        start_time = time.time()

        self.current_state = self._update_state(user_query)
        rewritten_query = self._rewrite_query(self.current_state)

        retrieval_start = time.time()
        top_docs, retrieval_score = self._retrieve_and_map(rewritten_query)
        retrieval_latency = time.time() - retrieval_start

        llm_output = self._generate_answer(rewritten_query, top_docs)

        is_refused, reason = self._should_refuse(
            self.current_state, top_docs, retrieval_score, llm_output
        )

        total_latency = time.time() - start_time
        final_answer = "根据现有资料无法回答。" if is_refused else llm_output.get("answer", "")

        self.history.append({"user": user_query, "assistant": final_answer})
        
        return {
            "answer": final_answer,
            "rewritten_query": rewritten_query,
            "is_refused": is_refused,
            "refuse_reason": reason,
            "retrieval_latency": round(retrieval_latency, 4),
            "total_latency": round(total_latency, 4),
            "retrieval_score": retrieval_score,
            "current_state": self.current_state
        }

    def _retrieve_and_map(self, query: str) -> Tuple[List[Dict], float]:
        try:
            results = self.retriever.retrieve(query, top_k=5, candidate_k=20)
            if not results:
                return [], 0.0
                
            doc_ids = [doc_id for doc_id, score in results]
            top_score = results[0][1]
            
            chroma_results = self.collection.get(ids=doc_ids)
            
            docs = []
            if chroma_results and chroma_results.get("documents"):
                for i, doc_id in enumerate(doc_ids):
                    try:
                        idx = chroma_results["ids"].index(doc_id)
                        text = chroma_results["documents"][idx]
                        docs.append({"text": text, "id": doc_id, "score": results[i][1]})
                    except ValueError:
                        continue
            
            return docs, top_score
            
        except Exception as e:
            logger.error(f"检索链路发生异常: {e}")
            return [], 0.0

    def _update_state(self, query: str) -> Dict[str, Any]:
        prompt = f"""
        历史对话: {self.history[-2:] if self.history else "无"}
        用户新问题: {query}
        请分析当前意图和必须满足的硬约束。
        注意1：search_query 必须是高度浓缩的检索关键词（用于去向量库搜索），绝对不能包含指令，只需保留核心实体和属性。
        注意2：hard_constraints 必须是极简的名词或实体（比如"马尔可夫决策过程", "奖励函数"），绝对不要写成长句指令。
        
        请严格输出JSON格式：
        {{
            "intent": "意图描述", 
            "hard_constraints": ["条件词1", "条件词2"],
            "search_query": "核心实体 + 核心属性"
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"状态更新 JSON 解析失败: {e}")
            return {"intent": query, "hard_constraints": [], "search_query": query}

    def _rewrite_query(self, state: Dict[str, Any]) -> str:
        return state.get("search_query", state.get("intent", ""))

    def _generate_answer(self, query: str, docs: List[Dict]) -> Dict[str, Any]:
        if not docs:
            return {"answer": "", "confidence": 0.0}
            
        context = "\n".join([d["text"] for d in docs])
        prompt = f"""
        根据以下资料回答问题。
        资料: {context}
        问题: {query}
        请严格输出JSON格式：{{"answer": "你的回答", "confidence": 0.0到1.0的分数}}
        """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"生成答案 JSON 解析失败: {e}")
            return {"answer": "生成解析失败", "confidence": 0.0}

    def _should_refuse(self, state: Dict[str, Any], docs: List[Dict], ret_score: float, llm_out: Dict[str, Any]) -> Tuple[bool, str]:
        if ret_score < self.retrieval_threshold:
            return True, f"检索置信度过低 ({ret_score} < {self.retrieval_threshold})"
            
        hard_constraints = state.get("hard_constraints", [])
        if hard_constraints and docs:
            top_text = docs[0]["text"]
            # 【修复3】宽松匹配：只要文档里出现了任意一个硬约束实体词，就认为满足
            if not any(c in top_text for c in hard_constraints):
                return True, f"检索结果不满足硬约束: {hard_constraints}"
                
        gen_conf = llm_out.get("confidence", 0.0)
        if gen_conf < self.generation_threshold:
            return True, f"生成置信度过低 ({gen_conf} < {self.generation_threshold})"
            
        return False, "正常回答"