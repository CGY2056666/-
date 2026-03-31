# focus_engine/duration_recommender.py
import os
import numpy as np
from typing import List, Dict
from openai import OpenAI
from .goal_profiles import normalize_goal_input
from .history_storage import load_all_history, load_history_by_goal_type

class FocusDurationRecommender:
    def __init__(self, config):
        self.config = config
        self.history_sessions = []
        # 自主初始化DeepSeek客户端，复用项目.env配置
        self.llm_client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        )
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        # 启动时自动加载已有历史数据
        self.load_history(load_all_history())

    def load_history(self, sessions: List[Dict]):
        """加载用户历史会话数据"""
        self.history_sessions = sessions

    def _classify_goal(self, goal: str) -> str:
        """调用DeepSeek分类目标类型，加异常兜底"""
        prompt = f"将以下专注目标分为三类：认知学习类、创意创作类、机械执行类，只输出分类结果，不要额外内容。目标：{goal}"
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"目标分类失败，默认使用认知学习类: {e}")
            # 兜底关键词匹配
            goal = goal.lower()
            if any(keyword in goal for keyword in ["学习", "背", "刷题", "考试", "高数", "python", "编程", "英语", "课程", "考研", "考证"]):
                return "认知学习类"
            elif any(keyword in goal for keyword in ["写", "创作", "设计", "策划", "小说", "文案", "画图", "视频", "剪辑", "PPT"]):
                return "创意创作类"
            else:
                return "机械执行类"

    def _calculate_baseline(self, goal_type: str = None) -> float:
        """计算基准专注时长：优先同类型历史数据，无则冷启动"""
        # 有同类型历史数据，用同类型数据计算
        if goal_type:
            type_history = load_history_by_goal_type(goal_type)
            if len(type_history) >= 1:
                valid_durations = []
                for s in type_history:
                    focus_sequence = s.get("focus_sequence", [])
                    max_valid_window = 0
                    current_window = 0
                    for minute_score in focus_sequence:
                        if minute_score >= 40:
                            current_window += 1
                            max_valid_window = max(max_valid_window, current_window)
                        else:
                            current_window = 0
                    if max_valid_window > 0:
                        valid_durations.append(max_valid_window)
                if valid_durations:
                    return float(np.median(valid_durations))
        # 无历史数据，冷启动默认25分钟
        return 25.0

    def _calculate_decay_coeff(self, goal_type: str = None) -> float:
        """拟合用户注意力衰减系数：优先同类型历史数据"""
        if goal_type:
            type_history = load_history_by_goal_type(goal_type)
            if len(type_history) >= 1:
                all_slopes = []
                for s in type_history:
                    focus_sequence = s.get("focus_sequence", [])
                    if len(focus_sequence) < 5:
                        continue
                    t = np.arange(len(focus_sequence))
                    slope, _ = np.polyfit(t, focus_sequence, 1)
                    all_slopes.append(abs(slope))
                if all_slopes:
                    return float(np.median(all_slopes))
        # 无历史数据，默认中衰减
        return 0.5

    def _calculate_next_recommend(self, current_session: Dict) -> Dict:
        """【核心】基于本次会话表现，计算下一次的推荐时长"""
        # 提取本次会话核心数据
        goal = current_session.get("goal", "")
        goal_type = current_session.get("goal_type", "认知学习类")
        plan_duration = current_session.get("plan_duration", 25)
        actual_duration = current_session.get("actual_duration", 0)
        focus_sequence = current_session.get("focus_sequence", [])
        completion_score = current_session.get("completion_score", 0)

        # 1. 计算本次完成率
        completion_rate = actual_duration / plan_duration if plan_duration > 0 else 0

        # 2. 计算本次有效专注时长
        max_valid_window = 0
        current_window = 0
        for minute_score in focus_sequence:
            if minute_score >= 40:
                current_window += 1
                max_valid_window = max(max_valid_window, current_window)
            else:
                current_window = 0

        # 3. 计算本次注意力衰减情况
        decay_coeff = 0.5
        if len(focus_sequence) >= 5:
            t = np.arange(len(focus_sequence))
            slope, _ = np.polyfit(t, focus_sequence, 1)
            decay_coeff = abs(slope)

        # 4. 核心修正逻辑（基于本次表现）
        base_baseline = self._calculate_baseline(goal_type)
        # 正强化：完成率≥90%，且有效时长≥计划时长，下次推荐上浮10%
        if completion_rate >= 0.9 and max_valid_window >= plan_duration * 0.8:
            next_base = base_baseline * 1.1
            feedback = "本次专注表现优秀，注意力保持稳定，下次推荐适当延长时长，挑战更高专注深度。"
        # 负修正：完成率<60%，或有效时长不足计划的50%，下次推荐下浮20%
        elif completion_rate < 0.6 or max_valid_window < plan_duration * 0.5:
            next_base = base_baseline * 0.8
            feedback = "本次专注完成度较低，注意力衰减较快，下次推荐缩短时长，降低目标门槛，帮你建立专注正反馈。"
        # 平稳状态：保持基准，微调
        else:
            next_base = base_baseline
            feedback = "本次专注表现平稳，下次推荐保持基准时长，你可以根据自身状态选择挑战或轻松模式。"

        # 5. 边界约束
        min_dur = self.config.session_min_duration_minutes
        max_dur = self.config.session_max_duration_minutes
        next_base = max(min_dur, min(next_base, max_dur))

        # 6. 生成分级推荐
        return {
            "current_summary": {
                "plan_duration": plan_duration,
                "actual_duration": actual_duration,
                "max_valid_focus_minutes": max_valid_window,
                "completion_rate": round(completion_rate * 100, 1),
                "completion_score": completion_score
            },
            "next_recommend": {
                "base_recommend": round(next_base),
                "challenge_recommend": round(min(next_base * 1.2, max_dur)),
                "easy_recommend": round(max(next_base * 0.7, min_dur)),
                "rest_recommend": round(max(next_base / 5, 5))
            },
            "goal_type": goal_type,
            "feedback_text": feedback
        }

    def recommend(self, goal: str) -> Dict:
        """核心推荐入口：基于目标+历史数据，返回推荐结果"""
        goal = normalize_goal_input(goal)
        # 1. 目标分类
        goal_type = self._classify_goal(goal)
        # 2. 基于同类型历史数据，计算基线与衰减系数
        baseline = self._calculate_baseline(goal_type)
        alpha = self._calculate_decay_coeff(goal_type)
        # 3. 目标类型修正
        beta_map = {"认知学习类": 0.9, "创意创作类": 1.2, "机械执行类": 1.0}
        beta = beta_map.get(goal_type, 1.0)
        # 4. 历史完成率修正
        gamma = 1.0
        type_history = load_history_by_goal_type(goal_type)
        if len(type_history) >= 3:
            completion_rates = [
                s.get("actual_duration", 0) / s.get("plan_duration", 1)
                for s in type_history
            ]
            cr_mean = np.mean(completion_rates)
            if cr_mean >= 0.9:
                gamma = 1.1
            elif cr_mean < 0.6:
                gamma = 0.7
        # 5. 高效时长修正
        delta = 1.0
        if len(type_history) >= 3:
            high_eff_sessions = [s for s in type_history if s.get("completion_score", 0) >= 80]
            if high_eff_sessions:
                high_eff_mean = np.mean([s.get("max_valid_focus_minutes", baseline) for s in high_eff_sessions])
                if high_eff_mean > baseline:
                    delta = 1.15
        # 6. 边界约束与最终输出
        t_corrected = baseline * beta * gamma * delta
        min_dur = self.config.session_min_duration_minutes
        max_dur = self.config.session_max_duration_minutes
        t_final = max(min_dur, min(t_corrected, max_dur))

        return {
            "base_recommend": round(t_final),
            "challenge_recommend": round(min(t_final * 1.2, max_dur)),
            "easy_recommend": round(max(t_final * 0.7, min_dur)),
            "rest_recommend": round(max(t_final / 5, 5)),
            "goal_type": goal_type,
            "decay_type": "低衰减型" if alpha <=0.3 else "中衰减型" if alpha <=0.8 else "高衰减型"
        }