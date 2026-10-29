from pydantic import BaseModel, Field
from typing import List, Dict

class ConversationState(BaseModel):
    current_entities: List[str] = Field(default_factory=list, description="当前讨论实体")
    confirmed_info: Dict[str, str] = Field(default_factory=dict, description="已确认信息")
    pending_questions: List[str] = Field(default_factory=list, description="待澄清问题")
    topic_switch_flag: bool = Field(default=False, description="话题切换标记")
    history_summary: str = Field(default="", description="历史轮次摘要")

    def apply_turn_update(
        self,
        new_entities: List[str],
        new_confirmed: Dict[str, str],
        new_questions: List[str],
        topic_switch: bool
    ) -> None:
        """
        一轮对话结束后，根据新提取的信息更新整个状态。
        """
        # 1. 更新话题切换标记
        self.topic_switch_flag = topic_switch
        
        # 2. 更新当前讨论实体：切换则替换，不切换则追加去重
        if topic_switch:
            self.current_entities = new_entities
        else:
            for ent in new_entities:
                if ent not in self.current_entities:
                    self.current_entities.append(ent)
                    
        # 3. 更新已确认信息（字典合并）
        self.confirmed_info.update(new_confirmed)
        
        # 4. 更新待澄清问题（去重追加）
        for q in new_questions:
            if q not in self.pending_questions:
                self.pending_questions.append(q)
                
        # 5. 更新历史轮次摘要（简单拼接示意）
        self.history_summary = f"当前话题: {self.current_entities}; 已确认: {self.confirmed_info}"

    def reset(self) -> None:
        """显式重置状态，通常由用户主动触发（如“重新开始”）"""
        self.current_entities.clear()
        self.confirmed_info.clear()
        self.pending_questions.clear()
        self.topic_switch_flag = False
        self.history_summary = ""