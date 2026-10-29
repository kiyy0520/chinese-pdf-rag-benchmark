# src/conversation/topic_switch.py
from typing import List, Set

def calculate_jaccard_similarity(entities_a: List[str], entities_b: List[str]) -> float:
    """计算两个实体列表的Jaccard相似度（重叠度）"""
    set_a = set(entities_a)
    set_b = set(entities_b)
    
    # 如果两个都是空的，视为没有变化（不触发切换）
    if not set_a and not set_b:
        return 1.0
    
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0

def detect_topic_switch(
    new_entities: List[str], 
    current_entities: List[str], 
    threshold: float = 0.3
) -> bool:
    """检测是否发生话题切换"""
    overlap = calculate_jaccard_similarity(new_entities, current_entities)
    return overlap < threshold