import json
from pathlib import Path
from src.conversation.state import ConversationState

class StateTracker:
    def __init__(self, llm, state_dir="./data/states"):
        self.llm = llm
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_file_path(self, session_id: str) -> Path:
        return self.state_dir / f"{session_id}.json"

    def load_state(self, session_id: str) -> ConversationState:
        path = self._get_file_path(session_id)
        if not path.exists():
            return ConversationState()
        try:
            return ConversationState.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ 发现损坏的旧状态，已自动重置: {e}")
            return ConversationState()

    def save_state(self, state: ConversationState, session_id: str):
        path = self._get_file_path(session_id)
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    def update_state_with_llm(self, state: ConversationState, user_message: str, ai_message: str) -> ConversationState:
        llm_json = self.llm.bind(response_format={"type": "json_object"})
        
        prompt = f"""请从以下对话中提取状态更新信息，并以 JSON 格式输出。
必须包含以下字段：
- new_entities: 字符串列表，例如 ["时间:明天", "地点:朝阳区"]，禁止使用嵌套字典！
- new_confirmed: 字典，例如 {{"预算": "500元"}}
- new_questions: 字符串列表
- topic_switch: 布尔值
- summary_of_turn: 一句话总结本轮对话的核心信息

对话：
用户：{user_message}
助手：{ai_message}
"""
        try:
            response = llm_json.invoke(prompt)
            result = json.loads(response.content)
            
            # 极致数据清洗（防止大模型乱返回嵌套字典导致崩溃）
            raw_entities = result.get("new_entities", [])
            cleaned_entities = []
            if isinstance(raw_entities, list):
                for ent in raw_entities:
                    if isinstance(ent, dict):
                        val = ent.get("value") or ent.get("name") or ent.get("entity") or str(ent)
                        cleaned_entities.append(str(val))
                    else:
                        cleaned_entities.append(str(ent))
            elif isinstance(raw_entities, dict):
                for k, v in raw_entities.items():
                    cleaned_entities.append(f"{k}:{v}")
            else:
                cleaned_entities.append(str(raw_entities))

            raw_confirmed = result.get("new_confirmed", {})
            cleaned_confirmed = {}
            if isinstance(raw_confirmed, dict):
                cleaned_confirmed = {str(k): str(v) for k, v in raw_confirmed.items()}
            
            raw_questions = result.get("new_questions", [])
            cleaned_questions = []
            if isinstance(raw_questions, list):
                cleaned_questions = [str(q) for q in raw_questions]
            
            # 更新 Pydantic 状态
            state.apply_turn_update(
                new_entities=cleaned_entities,
                new_confirmed=cleaned_confirmed,
                new_questions=cleaned_questions,
                topic_switch=bool(result.get("topic_switch", False))
            )
            
            # 摘要减肥
            turn_summary = result.get("summary_of_turn", "")
            if turn_summary:
                combined = (state.history_summary + " " + turn_summary).strip()
                state.history_summary = combined[-300:]  # 强制截断，最多保留最后 300 个字符
            
        except Exception as e:
            print(f"状态提取失败: {e}")
        return state