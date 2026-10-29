import json
from langchain_core.messages import SystemMessage, trim_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from src.conversation.state_tracker import StateTracker

class MemoryManager:
    def __init__(self, llm, token_counter, state_tracker: StateTracker):
        self.llm = llm
        self.token_counter = token_counter
        self.state_tracker = state_tracker

    def get_trimmed_history(self, session_id: str, full_history: ChatMessageHistory, max_tokens: int = 300) -> ChatMessageHistory:
        """核心职责：把全量历史裁剪、压缩，并拼上状态注入给 LLM"""
        state = self.state_tracker.load_state(session_id)
        
        # 1. 摘要压缩（超过 20 条触发）
        if len(full_history.messages) > 20:
            to_summarize = full_history.messages[:10]
            rest = full_history.messages[10:]
            text = "\n".join([f"{m.type}: {m.content}" for m in to_summarize])
            
            summary_prompt = f"将以下对话压缩为简短摘要，保留所有实体和时间地点：{text}"
            summary = self.llm.invoke(summary_prompt).content
            
            # 更新状态里的摘要（硬截断防膨胀）
            state.history_summary = (state.history_summary + " " + summary).strip()[-300:]
            self.state_tracker.save_state(state, session_id)
            
            # 物理移除已压缩的历史
            full_history.messages = [SystemMessage(content=f"历史摘要：{state.history_summary}")] + rest

        # 2. 滑动窗口 Token 截断
        trimmer = trim_messages(
            max_tokens=max_tokens, strategy="last", token_counter=self.token_counter,
            include_system=True, start_on="human", allow_partial=False
        )
        trimmed = trimmer.invoke(full_history.messages)
        
        # 3. 降维注入：只取最精简的状态拼入 SystemMessage
        core_state = {
            "已确认信息": state.confirmed_info,
            "历史摘要": state.history_summary
        }
        state_text = json.dumps(core_state, ensure_ascii=False, indent=2)

        new_history = ChatMessageHistory()
        new_history.add_message(SystemMessage(content=f"以下是用户当前的关键状态，请在回答时严格遵守：\n{state_text}"))
        new_history.messages.extend(trimmed)
        return new_history