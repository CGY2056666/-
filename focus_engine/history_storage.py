# focus_engine/history_storage.py
import json
import os
from pathlib import Path
from typing import List, Dict

# 历史数据存储文件路径，和app.py同目录
HISTORY_FILE = Path(__file__).parent.parent / "focus_history.json"

def init_history_file():
    """初始化历史数据文件，不存在则创建"""
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

def save_session_history(session_data: Dict):
    """保存单次会话的历史数据"""
    init_history_file()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    # 给会话加唯一标识，避免重复
    session_data["session_id"] = f"{session_data.get('goal', '')}_{session_data.get('end_time', '')}"
    history.append(session_data)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_all_history() -> List[Dict]:
    """加载所有历史会话数据"""
    init_history_file()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_history_by_goal_type(goal_type: str) -> List[Dict]:
    """按目标类型加载历史数据，用于同类型目标的推荐修正"""
    all_history = load_all_history()
    return [s for s in all_history if s.get("goal_type") == goal_type]