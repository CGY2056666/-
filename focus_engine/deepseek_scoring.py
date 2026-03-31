from __future__ import annotations

import hashlib
import json
import time
from typing import Any
from urllib import error, request

from .config import AnalyzerConfig, DEFAULT_CONFIG
from .models import GoalProfile
from .utils import LRUCache, clamp, normalize_spaces


SYSTEM_PROMPT = """你是“目标相关截图评分器”。
你需要根据用户目标、关键词、截图文件名、OCR 文本和 OCR 质量，对当前截图进行目标相关性评分。

规则：
1. 只能依据输入内容判断，不能脑补截图中未出现的信息。
2. 如果 OCR 质量较低或文本很短，优先降低 confidence；证据不足时倾向 neutral，不要武断下结论。
3. 购物、旅游攻略、产品对比、地图、酒店、机票、社交、聊天、视频等内容，是否相关完全取决于用户当前目标，不能因为内容类别本身就判为偏离。
4. distraction_score 表示“与当前目标的偏离风险”，只有当内容与目标明显无关、支持证据很弱或出现明显跑题线索时才提高。
5. 与用户目标直接相关的教材、题目、公式、单词、论文、阅读材料、笔记、商品信息、攻略内容、检索结果等，都应提高 relevance_score 和 focus_score。
6. 只输出 JSON，不要输出 Markdown，不要补充额外解释。
7. semantic_match_score、keyword_hit_score、strong_hit_score、coverage_score、structure_score、app_context_score、relevance_score、distraction_score、confidence 取值范围均为 0 到 1，保留 3 位小数。
8. focus_score 取值范围为 0 到 100。
9. category_type 只能是 focus、neutral。
10. status 只能是 专注、轻微偏离、分心。
11. matched_keywords、strong_hits、scene_hits、support_hits、negative_hits、matched_context_patterns 都必须返回数组，即使为空数组。
"""


class DeepSeekScorer:
    def __init__(self, config: AnalyzerConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.cache = LRUCache(config.llm_cache_size)

    def status(self) -> dict[str, object]:
        return {
            "provider": "deepseek",
            "configured": bool(self.config.deepseek_api_key),
            "model": self.config.deepseek_model,
            "base_url": self.config.deepseek_base_url,
        }

    def score(self, profile: GoalProfile, filename: str, text: str, ocr_quality: float) -> dict[str, object]:
        cleaned_text = normalize_spaces(text)
        if not self.config.deepseek_api_key:
            return self._review_result("DeepSeek API Key 未配置，请先设置 DEEPSEEK_API_KEY。", ocr_quality)
        if not cleaned_text:
            return self._review_result("OCR 未提取到可用文本，当前截图无法提交 DeepSeek 评分。", ocr_quality)

        limited_text = cleaned_text[: self.config.deepseek_max_ocr_chars]
        truncated = len(cleaned_text) > len(limited_text)
        cache_key = self._build_cache_key(profile, filename, limited_text, ocr_quality)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        system_prompt, user_prompt = self._build_prompts(profile, filename, limited_text, ocr_quality)
        started = time.perf_counter()

        try:
            response_payload = self._request_completion(system_prompt, user_prompt)
            result = self._normalize_response(response_payload, ocr_quality)
        except Exception as exc:
            return self._review_result(f"DeepSeek 评分失败：{exc}", ocr_quality)

        result["fallback_reason"] = "OCR 文本过长，已截断后提交模型评分。" if truncated else ""
        result["score_breakdown"]["input_truncated"] = 1.0 if truncated else 0.0
        result["score_breakdown"]["llm_latency_ms"] = float(int((time.perf_counter() - started) * 1000))
        self.cache.set(cache_key, dict(result))
        return dict(result)

    def _build_cache_key(self, profile: GoalProfile, filename: str, text: str, ocr_quality: float) -> str:
        payload = {
            "goal": profile.normalized_goal,
            "filename": filename,
            "text": text,
            "ocr_quality": round(clamp(ocr_quality), 3),
            "model": self.config.deepseek_model,
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _build_prompts(
        self,
        profile: GoalProfile,
        filename: str,
        text: str,
        ocr_quality: float,
    ) -> tuple[str, str]:
        output_schema = {
            "category_type": "focus",
            "category_label": "目标相关页面",
            "semantic_match_score": 0.832,
            "keyword_hit_score": 0.774,
            "strong_hit_score": 0.801,
            "coverage_score": 0.756,
            "structure_score": 0.689,
            "app_context_score": 0.743,
            "relevance_score": 0.812,
            "distraction_score": 0.103,
            "focus_score": 84.5,
            "confidence": 0.861,
            "status": "专注",
            "matched_keywords": ["目标关键词", "页面主体内容"],
            "strong_hits": ["目标关键词命中", "页面主体信息"],
            "scene_hits": ["正文或列表结构"],
            "support_hits": ["任务相关步骤"],
            "negative_hits": [],
            "matched_context_patterns": ["目标相关内容页"],
            "positive_rule": "页面内容与当前学习目标直接相关。",
            "negative_rule": "",
            "reason": "截图主体与当前目标高度相关，偏离风险较低。",
        }

        lines = [
            f"当前目标：{profile.raw_goal}",
            "目标模式：用户自定义目标",
            f"OCR 质量：{round(clamp(ocr_quality), 3)}",
            f"截图文件名：{filename}",
            f"核心关键词：{json.dumps(profile.core_keywords[:16], ensure_ascii=False)}",
            f"场景关键词：{json.dumps(profile.scene_keywords[:16], ensure_ascii=False)}",
            f"辅助关键词：{json.dumps(profile.support_keywords[:16], ensure_ascii=False)}",
            f"语义关键词：{json.dumps(profile.semantic_keywords[:16], ensure_ascii=False)}",
            f"负向关键词：{json.dumps(profile.negative_keywords[:16], ensure_ascii=False)}",
            "输出 JSON Schema：",
            json.dumps(output_schema, ensure_ascii=False),
            "OCR 文本：",
            text,
            "请严格按照 Schema 返回 JSON。",
        ]
        return SYSTEM_PROMPT, "\n".join(lines)

    def _request_completion(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        endpoint = self._build_endpoint()
        payload = {
            "model": self.config.deepseek_model,
            "temperature": self.config.deepseek_temperature,
            "max_tokens": self.config.deepseek_max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.deepseek_timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {error_body[:240]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"网络请求失败：{exc.reason}") from exc

        body = json.loads(raw_response)
        message = ((body.get("choices") or [{}])[0].get("message") or {})
        content = str(message.get("content") or "").strip()
        if not content:
            raise RuntimeError("模型返回了空内容。")
        return json.loads(self._extract_json_payload(content))

    def _build_endpoint(self) -> str:
        base_url = self.config.deepseek_base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def _normalize_response(self, payload: dict[str, Any], ocr_quality: float) -> dict[str, object]:
        category_type = self._normalize_category(payload.get("category_type"))
        category_label = str(payload.get("category_label") or self._default_category_label(category_type)).strip()

        strong_hits = self._coerce_list(payload.get("strong_hits"), 8)
        scene_hits = self._coerce_list(payload.get("scene_hits"), 8)
        support_hits = self._coerce_list(payload.get("support_hits"), 8)
        negative_hits = self._coerce_list(payload.get("negative_hits"), 8)
        matched_keywords = self._coerce_list(payload.get("matched_keywords"), 12)
        if not matched_keywords:
            matched_keywords = self._coerce_list(strong_hits + scene_hits + support_hits, 12)
        matched_context_patterns = self._coerce_list(payload.get("matched_context_patterns"), 8)

        semantic_match_score = self._coerce_score(payload.get("semantic_match_score"))
        keyword_hit_score = self._coerce_score(payload.get("keyword_hit_score"))
        strong_hit_score = self._coerce_score(payload.get("strong_hit_score"))
        coverage_score = self._coerce_score(payload.get("coverage_score"))
        structure_score = self._coerce_score(payload.get("structure_score"))
        app_context_score = self._coerce_score(payload.get("app_context_score"))
        relevance_score = self._coerce_score(payload.get("relevance_score"))
        distraction_score = self._coerce_score(payload.get("distraction_score"))

        if keyword_hit_score == 0.0 and matched_keywords:
            keyword_hit_score = round(clamp(len(matched_keywords) / 6.0), 3)
        if strong_hit_score == 0.0 and strong_hits:
            strong_hit_score = round(clamp(len(strong_hits) / 4.0), 3)
        if coverage_score == 0.0 and matched_keywords:
            coverage_score = round(clamp(len(matched_keywords) / 8.0), 3)
        if app_context_score == 0.0:
            app_context_score = round(relevance_score, 3)

        focus_score = self._coerce_focus_score(payload.get("focus_score"))
        if focus_score == 0.0 and (relevance_score > 0.0 or distraction_score > 0.0):
            focus_score = round(clamp(0.72 * relevance_score + 0.28 * (1 - distraction_score)) * 100, 1)
        confidence = self._coerce_score(payload.get("confidence"))
        status = self._normalize_status(payload.get("status"), focus_score)

        positive_rule = str(payload.get("positive_rule") or "").strip()
        negative_rule = str(payload.get("negative_rule") or "").strip()
        if not positive_rule and strong_hits:
            positive_rule = f"主要正向证据：{strong_hits[0]}"
        if not negative_rule and negative_hits:
            negative_rule = f"主要偏离信号：{negative_hits[0]}"

        reason = str(payload.get("reason") or positive_rule or negative_rule or "DeepSeek 已完成目标相关性评分。").strip()
        focus_probability = round(clamp(focus_score / 100.0), 3)

        return {
            "category_type": category_type,
            "category_label": category_label,
            "semantic_match_score": semantic_match_score,
            "keyword_hit_score": keyword_hit_score,
            "strong_hit_score": strong_hit_score,
            "coverage_score": coverage_score,
            "structure_score": structure_score,
            "app_context_score": app_context_score,
            "relevance_score": relevance_score,
            "distraction_score": distraction_score,
            "focus_score": focus_score,
            "focus_probability": focus_probability,
            "confidence": confidence,
            "status": status,
            "matched_keywords": matched_keywords,
            "strong_hits": strong_hits,
            "scene_hits": scene_hits,
            "support_hits": support_hits,
            "negative_hits": negative_hits,
            "matched_context_patterns": matched_context_patterns,
            "positive_rule": positive_rule,
            "negative_rule": negative_rule,
            "reason": reason,
            "fallback_reason": "",
            "review_required": False,
            "score_breakdown": {
                "semantic_score": semantic_match_score,
                "strong_hit_score": strong_hit_score,
                "coverage_score": coverage_score,
                "keyword_hit_score": keyword_hit_score,
                "structure_score": structure_score,
                "app_context_score": app_context_score,
                "ocr_quality_score": round(clamp(ocr_quality), 3),
                "base_relevance_score": relevance_score,
                "distraction_score": distraction_score,
                "confidence": confidence,
                "llm_scored": 1.0,
            },
        }

    def _review_result(self, message: str, ocr_quality: float) -> dict[str, object]:
        return {
            "category_type": "neutral",
            "category_label": "待复核场景",
            "semantic_match_score": 0.0,
            "keyword_hit_score": 0.0,
            "strong_hit_score": 0.0,
            "coverage_score": 0.0,
            "structure_score": 0.0,
            "app_context_score": 0.0,
            "relevance_score": 0.0,
            "distraction_score": 0.0,
            "focus_score": 0.0,
            "focus_probability": 0.0,
            "confidence": 0.0,
            "status": "待复核",
            "matched_keywords": [],
            "strong_hits": [],
            "scene_hits": [],
            "support_hits": [],
            "negative_hits": [],
            "matched_context_patterns": [],
            "positive_rule": "",
            "negative_rule": "",
            "reason": message,
            "fallback_reason": message,
            "review_required": True,
            "score_breakdown": {
                "semantic_score": 0.0,
                "strong_hit_score": 0.0,
                "coverage_score": 0.0,
                "keyword_hit_score": 0.0,
                "structure_score": 0.0,
                "app_context_score": 0.0,
                "ocr_quality_score": round(clamp(ocr_quality), 3),
                "base_relevance_score": 0.0,
                "distraction_score": 0.0,
                "confidence": 0.0,
                "llm_scored": 0.0,
            },
        }

    def _extract_json_payload(self, content: str) -> str:
        stripped = content.strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise RuntimeError("模型返回内容不是有效 JSON。")
        return stripped[start : end + 1]

    def _normalize_category(self, value: Any) -> str:
        text = str(value or "neutral").strip().lower()
        if text in {"focus", "study", "learning", "goal_related", "related"}:
            return "focus"
        return "neutral"

    def _normalize_status(self, value: Any, focus_score: float) -> str:
        text = str(value or "").strip()
        if text in {"专注", "轻微偏离", "分心"}:
            return text
        if focus_score >= 74:
            return "专注"
        if focus_score >= 50:
            return "轻微偏离"
        return "分心"

    def _default_category_label(self, category_type: str) -> str:
        if category_type == "focus":
            return "目标相关场景"
        return "待确认场景"

    def _coerce_score(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if number > 1.0 and number <= 100.0:
            number = number / 100.0
        return round(clamp(number), 3)

    def _coerce_focus_score(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if 0.0 <= number <= 1.0:
            number = number * 100.0
        return round(max(0.0, min(number, 100.0)), 1)

    def _coerce_list(self, value: Any, limit: int) -> list[str]:
        if isinstance(value, str):
            source = [value]
        elif isinstance(value, list):
            source = value
        else:
            source = []

        items: list[str] = []
        seen: set[str] = set()
        for raw in source:
            text = str(raw or "").strip()
            if not text:
                continue
            normalized = text.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            items.append(text)
            if len(items) >= limit:
                break
        return items

