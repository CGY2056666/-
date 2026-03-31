from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import ContextMatch, FrameAnalysis, GoalProfile
from .utils import clamp, extract_meaningful_tokens, normalize_spaces, normalize_text


CONTEXT_RULES = [
    {
        "key": "dictionary",
        "label": "词典/解释页",
        "category_type": "focus",
        "intent_score": 0.88,
        "risk_score": 0.08,
        "patterns": [
            "dictionary", "translation", "meaning", "definition", "pronunciation", "example",
            "example sentence", "词典", "释义", "词义", "翻译", "例句", "发音",
            "phrase", "collins", "oxford", "cambridge", "youdao",
        ],
    },
    {
        "key": "course_material",
        "label": "课程/资料页",
        "category_type": "focus",
        "intent_score": 0.84,
        "risk_score": 0.12,
        "patterns": [
            "lesson", "lecture", "course", "chapter", "section", "worksheet", "handout",
            "passage", "课程", "章节", "资料", "讲义", "课件", "学习内容", "阅读材料",
        ],
    },
    {
        "key": "notes",
        "label": "笔记/整理页",
        "category_type": "focus",
        "intent_score": 0.86,
        "risk_score": 0.10,
        "patterns": ["note", "notes", "obsidian", "onenote", "markdown", "notebook", "笔记", "整理", "总结", "提纲"],
    },
    {
        "key": "document",
        "label": "文档/正文页",
        "category_type": "focus",
        "intent_score": 0.82,
        "risk_score": 0.14,
        "patterns": ["pdf", "document", "word", "chapter", "section", "doc", "正文", "文档", "资料", "章节", "说明"],
    },
    {
        "key": "search_comparison",
        "label": "检索/对比页",
        "category_type": "focus",
        "intent_score": 0.78,
        "risk_score": 0.16,
        "patterns": [
            "search", "results", "compare", "comparison", "spec", "price", "hotel", "flight", "route",
            "搜索结果", "比价", "对比", "参数", "预算", "酒店", "机票", "路线", "攻略", "商品详情",
        ],
    },
    {
        "key": "research_analysis",
        "label": "研究/分析页",
        "category_type": "focus",
        "intent_score": 0.82,
        "risk_score": 0.14,
        "patterns": [
            "abstract", "method", "result", "discussion", "hypothesis", "anova", "regression",
            "analysis", "dashboard", "report", "summary", "摘要", "方法", "结果", "讨论", "分析", "报表",
        ],
    },
    {
        "key": "work_tools",
        "label": "工具/工作台",
        "category_type": "focus",
        "intent_score": 0.72,
        "risk_score": 0.18,
        "patterns": ["jupyter", "notebook", "excel", "pandas", "jamovi", "spss", "table", "sheet", "表格", "工作台"],
    },
    {
        "key": "chat",
        "label": "即时沟通页",
        "category_type": "neutral",
        "intent_score": 0.34,
        "risk_score": 0.78,
        "patterns": ["微信", "qq", "message", "chat", "weixin", "im", "私信", "消息"],
    },
    {
        "key": "short_video",
        "label": "信息流/短视频页",
        "category_type": "neutral",
        "intent_score": 0.22,
        "risk_score": 0.92,
        "patterns": ["douyin", "xhs", "reels", "直播", "推荐流", "短视频", "feed"],
    },
    {
        "key": "video",
        "label": "视频播放页",
        "category_type": "neutral",
        "intent_score": 0.28,
        "risk_score": 0.86,
        "patterns": ["bilibili", "youtube", "video", "播放", "up 主", "片段"],
    },
]

OCR_REPLACEMENTS = {
    "1earn": "learn",
    "w0rd": "word",
    "examp1e": "example",
    "translatlon": "translation",
    "transiation": "translation",
    "defination": "definition",
    "definitlon": "definition",
    "meanlng": "meaning",
    "pronunciatlon": "pronunciation",
    "sentencc": "sentence",
    "gramrnar": "grammar",
    "1istening": "listening",
    "0f": "of",
    "th1s": "this",
}


def _dedupe_terms(terms: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = term.strip()
        if len(cleaned) < 2:
            continue
        normalized = normalize_text(cleaned)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(cleaned)
    return unique


def clean_ocr_text(text: str) -> str:
    text = normalize_spaces(text.replace("\r", "\n"))
    lines: list[str] = []
    seen: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if len(line) < 2:
            continue
        if re.fullmatch(r"[\W_]+", line):
            continue
        for source, target in OCR_REPLACEMENTS.items():
            line = re.sub(source, target, line, flags=re.IGNORECASE)
        line = re.sub(r"[^\S\n]{2,}", " ", line)
        if len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", line)) < 2:
            continue
        if line not in seen:
            seen.add(line)
            lines.append(line)

    merged: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if len(line) <= 6 and len(lines) > 1:
            buffer.append(line)
            continue
        if buffer:
            merged.append(" ".join(buffer + [line]).strip())
            buffer = []
        else:
            merged.append(line)
    if buffer:
        merged.append(" ".join(buffer).strip())

    return "\n".join(item for item in merged if item)


def score_ocr_quality(text: str) -> float:
    if not text:
        return 0.0

    meaningful_chars = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))
    total_chars = max(len(text), 1)
    meaningful_ratio = meaningful_chars / total_chars
    symbol_ratio = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text)) / total_chars
    useful_lines = sum(1 for line in text.splitlines() if len(line.strip()) >= 4)
    duplicate_lines = len(text.splitlines()) - len({line.strip() for line in text.splitlines() if line.strip()})

    score = (
        0.33 * clamp(meaningful_chars / 90.0)
        + 0.33 * clamp(meaningful_ratio)
        + 0.22 * clamp(useful_lines / 4.0)
        + 0.12 * clamp(1 - symbol_ratio - min(duplicate_lines / 6.0, 0.2))
    )
    return round(clamp(score), 3)


def _latin_tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    return set(re.findall(r"[a-z][a-z0-9_\-+#./]{1,}", normalized))


def _contains_term(term: str, haystack: str, latin_tokens: set[str]) -> bool:
    normalized_term = normalize_text(term)
    if len(normalized_term) < 2:
        return False
    if normalized_term in haystack:
        return True
    if re.search(r"[一-鿿]", normalized_term):
        return False
    if " " in normalized_term:
        return False

    for token in latin_tokens:
        if token == normalized_term:
            return True
        if len(normalized_term) >= 5 and token and token[0] == normalized_term[0]:
            if abs(len(token) - len(normalized_term)) <= 3:
                if SequenceMatcher(None, token, normalized_term).ratio() >= 0.82:
                    return True
    return False


def _match_terms(terms: list[str], haystack: str, latin_tokens: set[str]) -> list[str]:
    matched: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = normalize_text(term)
        if normalized in seen or len(normalized) < 2:
            continue
        if _contains_term(term, haystack, latin_tokens):
            matched.append(term)
            seen.add(normalized)
    return matched


def _tokenize_for_similarity(text: str) -> str:
    tokens = extract_meaningful_tokens(text)
    return " ".join(_dedupe_terms(tokens))


def _collect_signal_matches(profile: GoalProfile, text: str) -> dict[str, object]:
    haystack = normalize_text(text)
    latin_tokens = _latin_tokens(text)

    core_hits = _match_terms(profile.core_keywords, haystack, latin_tokens)
    scene_hits = _dedupe_terms(
        _match_terms(profile.scene_keywords, haystack, latin_tokens)
        + _match_terms(profile.aliases, haystack, latin_tokens)
    )
    support_hits = _match_terms(profile.support_keywords, haystack, latin_tokens)
    semantic_hits = _match_terms(profile.semantic_keywords, haystack, latin_tokens)

    strong_negative_hits = _match_terms(profile.strong_negative_keywords, haystack, latin_tokens)
    medium_negative_hits = _match_terms(profile.medium_negative_keywords, haystack, latin_tokens)
    weak_negative_hits = _match_terms(profile.weak_negative_keywords, haystack, latin_tokens)

    core_weight = min(len(core_hits), 5) * 1.15
    scene_weight = min(len(scene_hits), 5) * 0.95
    semantic_weight = min(len(semantic_hits), 5) * 0.75
    support_weight = min(len(support_hits), 5) * 0.55
    coverage_cap = (
        max(1, min(len(profile.core_keywords), 5)) * 1.15
        + max(1, min(len(profile.scene_keywords), 5)) * 0.95
        + max(1, min(len(profile.semantic_keywords), 5)) * 0.75
        + max(1, min(len(profile.support_keywords), 5)) * 0.55
    )
    coverage_score = clamp((core_weight + scene_weight + semantic_weight + support_weight) / max(coverage_cap, 1.0))

    if core_hits:
        strong_hit_score = clamp(
            0.62
            + 0.15 * min(len(core_hits) - 1, 3)
            + 0.05 * min(len(scene_hits), 2)
            + 0.04 * min(len(semantic_hits), 2)
        )
    elif len(scene_hits) >= 2:
        strong_hit_score = 0.54 if semantic_hits else 0.50
    elif len(scene_hits) == 1 and (semantic_hits or len(support_hits) >= 1):
        strong_hit_score = 0.42
    elif len(semantic_hits) >= 2:
        strong_hit_score = 0.34
    elif scene_hits or semantic_hits:
        strong_hit_score = 0.28
    else:
        strong_hit_score = 0.16 if len(support_hits) >= 3 else 0.0

    negative_penalty = clamp(
        0.85 * (1.0 if strong_negative_hits else 0.0)
        + 0.42 * clamp(len(medium_negative_hits) / 2.0)
        + 0.18 * clamp(len(weak_negative_hits) / 3.0)
    )

    return {
        "core_hits": core_hits,
        "scene_hits": scene_hits,
        "support_hits": support_hits,
        "semantic_hits": semantic_hits,
        "strong_negative_hits": strong_negative_hits,
        "medium_negative_hits": medium_negative_hits,
        "weak_negative_hits": weak_negative_hits,
        "matched_keywords": _dedupe_terms(core_hits + scene_hits + semantic_hits + support_hits),
        "negative_hits": _dedupe_terms(strong_negative_hits + medium_negative_hits + weak_negative_hits),
        "strong_hit_score": round(clamp(strong_hit_score), 3),
        "coverage_score": round(clamp(coverage_score), 3),
        "negative_penalty": round(negative_penalty, 3),
    }


def semantic_similarity(profile: GoalProfile, text: str) -> float:
    goal_bundle = " ".join(
        _dedupe_terms(
            [profile.raw_goal]
            + profile.aliases[:8]
            + profile.core_keywords[:14]
            + profile.scene_keywords[:12]
            + profile.support_keywords[:8]
            + profile.semantic_keywords[:16]
        )
    )
    cleaned_text = normalize_text(text)
    normalized_goal_bundle = normalize_text(goal_bundle)
    if len(normalized_goal_bundle) < 2 or len(cleaned_text) < 2:
        return 0.0

    try:
        char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
        char_matrix = char_vectorizer.fit_transform([normalized_goal_bundle, cleaned_text])
        char_score = cosine_similarity(char_matrix[0:1], char_matrix[1:2])[0][0]
    except ValueError:
        char_score = 0.0

    token_goal = _tokenize_for_similarity(goal_bundle)
    token_text = _tokenize_for_similarity(text)
    if token_goal and token_text:
        try:
            word_vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")
            word_matrix = word_vectorizer.fit_transform([token_goal, token_text])
            word_score = cosine_similarity(word_matrix[0:1], word_matrix[1:2])[0][0]
        except ValueError:
            word_score = 0.0
    else:
        word_score = 0.0

    goal_terms = set(_dedupe_terms(extract_meaningful_tokens(goal_bundle)))
    text_terms = set(_dedupe_terms(extract_meaningful_tokens(text)))
    overlap_score = clamp(len(goal_terms & text_terms) / max(1, min(len(goal_terms), 12)))

    priority_terms = set(_dedupe_terms(profile.core_keywords[:12] + profile.scene_keywords[:8] + profile.semantic_keywords[:8]))
    priority_overlap = clamp(len(priority_terms & text_terms) / max(1, min(len(priority_terms), 10)))

    return round(clamp(0.30 * char_score + 0.24 * word_score + 0.20 * overlap_score + 0.26 * priority_overlap), 3)



def detect_structure_score(profile: GoalProfile, text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0

    line_count = len(lines)
    short_line_ratio = sum(2 <= len(line) <= 28 for line in lines) / line_count
    bilingual_lines = sum(
        bool(re.search(r"[A-Za-z]", line)) and bool(re.search(r"[\u4e00-\u9fff]", line))
        for line in lines
    )
    delimiter_lines = sum(any(symbol in line for symbol in (":", "：", "-", "->", "=>", "|")) for line in lines)
    list_lines = sum(bool(re.search(r"^(\d+[\.\)]|[-•·]|第.+[章节部分步])", line)) for line in lines)
    formula_lines = sum(
        bool(re.search(r"(定理|证明|推导|公式|函数|极限|导数|积分|矩阵|方程)", line))
        or (bool(re.search(r"[=+\-*/^]", line)) and bool(re.search(r"\d|x|y|f", line, flags=re.IGNORECASE)))
        for line in lines
    )
    document_lines = sum(
        bool(re.search(r"(摘要|目录|结论|参考文献|abstract|introduction|chapter|section|summary)", line, flags=re.IGNORECASE))
        for line in lines
    )
    stats_lines = sum(
        bool(re.search(r"(anova|regression|correlation|t-test|p\s*[<=>]|cronbach|信度|效度|样本|方差分析|指标)", line, flags=re.IGNORECASE))
        for line in lines
    )
    comparison_lines = sum(
        bool(re.search(r"(参数|预算|价格|型号|优缺点|规格|酒店|机票|路线|compare|comparison|spec|price|option|plan|hotel|flight|route)", line, flags=re.IGNORECASE))
        for line in lines
    )

    goal_terms = [
        normalize_text(term)
        for term in _dedupe_terms(profile.structure_patterns[:8] + profile.aliases[:10] + profile.scene_keywords[:8])
        if len(normalize_text(term)) >= 2
    ]
    keyword_lines = 0
    for line in lines:
        normalized_line = normalize_text(line)
        if any(term in normalized_line for term in goal_terms):
            keyword_lines += 1

    score = 0.10 * clamp(line_count / 5.0)
    score += 0.12 * clamp(short_line_ratio)
    score += 0.12 * clamp(bilingual_lines / 2.0)
    score += 0.10 * clamp(delimiter_lines / 2.0)
    score += 0.16 * clamp(list_lines / 3.0)
    score += 0.14 * clamp(document_lines / 2.0)
    score += 0.12 * clamp(formula_lines / 2.0)
    score += 0.10 * clamp(stats_lines / 2.0)
    score += 0.14 * clamp(comparison_lines / 2.0)
    score += 0.18 * clamp(keyword_lines / max(2.0, min(len(goal_terms), 6) or 2.0))
    return round(clamp(score), 3)



def _context_goal_alignment(profile: GoalProfile, matched_terms: list[str]) -> float:
    if not matched_terms:
        return 0.5

    goal_haystack = normalize_text(
        " ".join(
            profile.core_keywords[:18]
            + profile.scene_keywords[:18]
            + profile.aliases[:16]
            + profile.semantic_keywords[:16]
        )
    )
    overlap = 0
    for term in matched_terms:
        normalized = normalize_text(term)
        if normalized and normalized in goal_haystack:
            overlap += 1

    return round(
        clamp(0.42 + 0.16 * min(len(matched_terms), 3) + 0.42 * (overlap / max(len(matched_terms), 1))),
        3,
    )



def classify_context(filename: str, text: str, profile: GoalProfile) -> ContextMatch:
    haystack = normalize_text(f"{filename}\n{text}")
    latin_tokens = _latin_tokens(f"{filename}\n{text}")
    best_match = ContextMatch(
        key="unknown",
        label="待确认场景",
        category_type="neutral",
        intent_score=0.48,
        risk_score=0.42,
        goal_boost=0.50,
        matches=[],
    )

    best_score = -1.0
    for rule in CONTEXT_RULES:
        matched = _match_terms(rule["patterns"], haystack, latin_tokens)
        if not matched:
            continue
        goal_boost = _context_goal_alignment(profile, matched)
        score = max(rule["intent_score"], goal_boost) + 0.16 * min(len(matched), 3)
        if rule["category_type"] == "focus":
            score += 0.08
            if goal_boost >= 0.72:
                score += 0.06
        if score > best_score:
            best_score = score
            best_match = ContextMatch(
                key=rule["key"],
                label=rule["label"],
                category_type=rule["category_type"],
                intent_score=rule["intent_score"],
                risk_score=rule["risk_score"],
                goal_boost=goal_boost,
                matches=matched,
            )

    return best_match

def compute_base_relevance(
    profile: GoalProfile,
    text: str,
    ocr_quality: float,
    context: ContextMatch,
) -> dict[str, object]:
    semantic_score = semantic_similarity(profile, text)
    signal_matches = _collect_signal_matches(profile, text)
    structure_score = detect_structure_score(profile, text)
    app_context_score = round(clamp((context.intent_score + context.goal_boost) / 2.0), 3)

    strong_hit_score = float(signal_matches["strong_hit_score"])
    coverage_score = float(signal_matches["coverage_score"])
    keyword_score = round(clamp(0.72 * strong_hit_score + 0.28 * coverage_score), 3)
    negative_penalty = float(signal_matches["negative_penalty"])

    if ocr_quality < 0.25:
        relevance = (
            0.16 * semantic_score
            + 0.36 * strong_hit_score
            + 0.18 * coverage_score
            + 0.24 * app_context_score
            + 0.06 * structure_score
        )
    else:
        relevance = (
            0.22 * semantic_score
            + 0.33 * strong_hit_score
            + 0.19 * coverage_score
            + 0.20 * app_context_score
            + 0.06 * structure_score
        )

    positive_rule = ""
    negative_rule = ""
    fallback_reason = ""

    strong_negative_hits = signal_matches["strong_negative_hits"]
    medium_negative_hits = signal_matches["medium_negative_hits"]
    weak_negative_hits = signal_matches["weak_negative_hits"]
    core_hits = signal_matches["core_hits"]
    scene_hits = signal_matches["scene_hits"]
    semantic_hits = signal_matches["semantic_hits"]

    if len(core_hits) >= 2 and not strong_negative_hits:
        relevance = max(relevance, 0.84)
        positive_rule = "???????????????????"
    elif core_hits and (scene_hits or semantic_hits) and context.category_type != "distract":
        relevance = max(relevance, 0.78)
        positive_rule = "?????????????????????"
    elif len(scene_hits) >= 2 and context.category_type == "focus" and not strong_negative_hits:
        relevance = max(relevance, 0.74)
        positive_rule = "???????????????????"
    elif app_context_score >= 0.84 and max(strong_hit_score, coverage_score, structure_score) >= 0.30 and not strong_negative_hits:
        relevance = max(relevance, 0.72)
        positive_rule = "??????????????????"
    elif scene_hits and structure_score >= 0.40 and context.category_type == "focus":
        relevance = max(relevance, 0.68)
        positive_rule = "??????????????????"

    if ocr_quality < 0.28 and context.category_type == "focus" and (strong_hit_score >= 0.32 or structure_score >= 0.36):
        relevance = max(relevance, 0.64)
        fallback_reason = "OCR ???????????????????"

    if strong_negative_hits:
        if strong_hit_score >= 0.74 or (context.category_type == "focus" and structure_score >= 0.48):
            relevance = max(0.48, relevance - 0.14)
            negative_rule = "?????????????????????"
        else:
            relevance = min(relevance, 0.35)
            negative_rule = "???????????????????"
    elif medium_negative_hits:
        penalty = 0.06 if strong_hit_score >= 0.70 else 0.12
        relevance = max(0.0, relevance - penalty)
        negative_rule = "????????????????"
    elif weak_negative_hits:
        relevance = max(0.0, relevance - 0.03)
        negative_rule = "????????????????"

    score_breakdown = {
        "semantic_score": round(semantic_score, 3),
        "strong_hit_score": round(strong_hit_score, 3),
        "coverage_score": round(coverage_score, 3),
        "keyword_hit_score": round(keyword_score, 3),
        "structure_score": round(structure_score, 3),
        "app_context_score": round(app_context_score, 3),
        "ocr_quality_score": round(ocr_quality, 3),
        "negative_penalty": round(negative_penalty, 3),
        "base_relevance_score": round(clamp(relevance), 3),
    }

    return {
        "relevance": round(clamp(relevance), 3),
        "semantic_score": round(semantic_score, 3),
        "keyword_score": keyword_score,
        "strong_hit_score": round(strong_hit_score, 3),
        "coverage_score": round(coverage_score, 3),
        "structure_score": round(structure_score, 3),
        "app_context_score": round(app_context_score, 3),
        "matched_keywords": signal_matches["matched_keywords"],
        "strong_hits": core_hits,
        "scene_hits": scene_hits,
        "support_hits": signal_matches["support_hits"],
        "negative_hits": signal_matches["negative_hits"],
        "negative_penalty": round(negative_penalty, 3),
        "positive_rule": positive_rule,
        "negative_rule": negative_rule,
        "fallback_reason": fallback_reason,
        "score_breakdown": score_breakdown,
    }


def window_consistency(items: list[FrameAnalysis], index: int, radius: int = 3) -> float:
    current = items[index]
    neighbors = [
        items[position]
        for position in range(max(0, index - radius), min(len(items), index + radius + 1))
        if position != index and items[position].image_valid
    ]
    if not neighbors:
        return 0.55

    focus_like_neighbors = [
        neighbor
        for neighbor in neighbors
        if neighbor.category_type == "focus"
        or (neighbor.base_relevance_score or neighbor.relevance_score) >= 0.64
    ]
    stable_focus_ratio = len(focus_like_neighbors) / len(neighbors)
    same_category_ratio = (
        sum(neighbor.category_label == current.category_label for neighbor in focus_like_neighbors)
        / max(len(focus_like_neighbors), 1)
    )

    overlaps: list[float] = []
    current_keywords = set(current.matched_keywords)
    for neighbor in neighbors:
        neighbor_keywords = set(neighbor.matched_keywords)
        union = current_keywords | neighbor_keywords
        if union:
            overlaps.append(len(current_keywords & neighbor_keywords) / len(union))
    keyword_overlap_score = sum(overlaps) / len(overlaps) if overlaps else 0.0

    current_base = current.base_relevance_score or current.relevance_score
    relevance_band_score = sum(
        clamp(1 - abs(current_base - (neighbor.base_relevance_score or neighbor.relevance_score)))
        for neighbor in neighbors
    ) / len(neighbors)

    score = (
        0.34 * stable_focus_ratio
        + 0.24 * same_category_ratio
        + 0.22 * keyword_overlap_score
        + 0.20 * relevance_band_score
    )
    if current.category_type == "focus":
        score += 0.06
    return round(clamp(score), 3)


def _build_decision_reason(item: FrameAnalysis) -> str:
    reasons: list[str] = []
    if item.base_decision_reason:
        reasons.append(item.base_decision_reason)
    elif item.positive_rule:
        reasons.append(item.positive_rule)
    elif item.strong_hit_score >= 0.72:
        reasons.append("命中了高置信度目标关键词")
    elif item.keyword_hit_score >= 0.38:
        reasons.append("命中了多组相关关键词")
    elif item.semantic_match_score >= 0.38:
        reasons.append("OCR 文本与目标语义接近")
    else:
        reasons.append("直接文本证据偏弱")

    if item.structure_score >= 0.55:
        reasons.append("页面结构与目标场景高度一致")

    if item.category_label:
        reasons.append(f"当前场景为“{item.category_label}”")

    if item.window_consistency_score >= 0.72:
        reasons.append("前后截图保持连续一致")
    elif item.window_consistency_score < 0.40 and not item.review_required:
        reasons.append("前后截图切换较频繁")

    if item.negative_rule:
        reasons.append(item.negative_rule)
    elif item.distraction_score >= 0.80:
        reasons.append("与当前目标偏离较明显")

    return "，".join(_dedupe_terms(reasons))


def _scored_items(items: list[FrameAnalysis]) -> list[FrameAnalysis]:
    return [item for item in items if item.image_valid and not item.review_required and item.status != "待复核"]


def finalize_frame_scores(items: list[FrameAnalysis]) -> list[FrameAnalysis]:
    if not items:
        return items

    for index, item in enumerate(items):
        if not item.image_valid:
            item.status = "无法分析"
            continue

        if item.review_required:
            item.window_consistency_score = 0.0
            item.focus_probability = round(clamp(item.focus_score / 100.0), 3) if item.focus_score else 0.0
            item.status = "待复核"
            if not item.decision_reason:
                item.decision_reason = item.fallback_reason or "模型评分失败，请检查 DeepSeek 配置后重试。"
            item.score_breakdown.update(
                {
                    "window_consistency_score": 0.0,
                    "final_relevance_score": round(item.relevance_score, 3),
                    "focus_probability": round(item.focus_probability, 3),
                }
            )
            continue

        item.window_consistency_score = window_consistency(items, index)

        if item.scoring_source == "deepseek":
            item.relevance_score = round(clamp(item.base_relevance_score or item.relevance_score), 3)
            raw_focus_probability = item.focus_probability or clamp(item.focus_score / 100.0) or item.relevance_score
            if item.score_confidence < 0.35:
                smoothed_probability = raw_focus_probability
            else:
                smoothed_probability = clamp(0.88 * raw_focus_probability + 0.12 * item.window_consistency_score)
            item.focus_probability = round(smoothed_probability, 3)
            item.focus_score = round(item.focus_probability * 100, 1)
        else:
            base_relevance = item.base_relevance_score or item.relevance_score
            relevance = (
                0.76 * base_relevance
                + 0.17 * item.window_consistency_score
                + 0.07 * (1 - item.distraction_score)
            )

            neighbors = [
                neighbor
                for position, neighbor in enumerate(items)
                if abs(position - index) <= 3 and position != index and neighbor.image_valid and not neighbor.review_required
            ]
            focused_neighbors = [
                neighbor
                for neighbor in neighbors
                if neighbor.category_type == "focus"
                and (neighbor.base_relevance_score or neighbor.relevance_score) >= 0.64
            ]
            strong_negative_override = item.category_type == "distract" and item.distraction_score >= 0.85

            if neighbors and not strong_negative_override:
                focused_neighbor_ratio = len(focused_neighbors) / len(neighbors)
                if focused_neighbor_ratio >= 0.67 and item.category_type != "distract":
                    floor = 0.64 if max(item.strong_hit_score, item.structure_score, item.app_context_score) >= 0.40 else 0.60
                    if relevance < floor:
                        relevance = floor
                        if not item.positive_rule:
                            item.positive_rule = "连续多帧保持同类专注场景"

            item.relevance_score = round(clamp(relevance), 3)
            item.focus_probability = round(
                clamp(
                    0.58 * item.relevance_score
                    + 0.24 * item.app_context_score
                    + 0.12 * item.strong_hit_score
                    + 0.06 * (1 - item.distraction_score)
                ),
                3,
            )
            item.focus_score = round(clamp(item.focus_probability) * 100, 1)

        if item.focus_score >= 74:
            item.status = "专注"
        elif item.focus_score >= 50:
            item.status = "轻微偏离"
        else:
            item.status = "分心"

        item.decision_reason = _build_decision_reason(item)
        item.score_breakdown.update(
            {
                "window_consistency_score": round(item.window_consistency_score, 3),
                "final_relevance_score": round(item.relevance_score, 3),
                "focus_probability": round(item.focus_probability, 3),
            }
        )

    return items


def summary_suggestions(profile: GoalProfile, items: list[FrameAnalysis]) -> list[str]:
    image_items = [item for item in items if item.image_valid]
    scored_items = _scored_items(items)
    if not image_items:
        return ["请先上传清晰截图，以便系统提取有效文本和场景证据。"]
    if not scored_items:
        return ["截图已完成 OCR，但 DeepSeek 评分暂不可用，请检查 API Key、网络或模型配额后重试。"]

    suggestions: list[str] = []
    focus_ratio = sum(item.status == "专注" for item in scored_items) / max(len(scored_items), 1)
    average_quality = sum(item.ocr_quality_score for item in scored_items) / max(len(scored_items), 1)
    average_strong_hit = sum(item.strong_hit_score for item in scored_items) / max(len(scored_items), 1)
    average_structure = sum(item.structure_score for item in scored_items) / max(len(scored_items), 1)

    if focus_ratio < 0.5:
        suggestions.append("尽量上传连续截图，并保留标题、章节名或页面主体内容，减少只截局部界面的情况。")
    if average_quality < 0.35:
        suggestions.append("优先上传正文区域更完整、更清晰的截图，这会明显提升 OCR 和评分稳定性。")
    if average_strong_hit < 0.35:
        suggestions.append(f"尽量让截图里出现与“{profile.raw_goal}”直接相关的标题、正文、步骤、结果或对比信息，减少只截局部按钮和工具栏。")
    if average_structure < 0.28:
        suggestions.append("尽量避免只截滚动过渡帧或局部弹窗，保留页面结构能帮助系统更准确判断场景。")
    if any(
        item.status != "专注"
        and (item.distraction_score >= 0.58 or item.relevance_score < 0.55 or item.category_type != "focus")
        for item in scored_items
    ):
        suggestions.append("可以回看那些与当前目标偏离较明显的截图，判断这类页面是否需要单独安排到别的时段处理。")
    if not suggestions:
        suggestions.append("当前任务线索比较稳定，可以继续沿用这组学习环境和截图方式。")
    return suggestions

def build_summary(profile: GoalProfile, items: list[FrameAnalysis], processing_ms: int) -> dict:
    image_items = [item for item in items if item.image_valid]
    scored_items = _scored_items(items)
    review_count = sum(item.review_required for item in image_items)
    focus_count = sum(item.status == "专注" for item in scored_items)
    drift_count = sum(item.status == "轻微偏离" for item in scored_items)
    distract_count = sum(item.status == "分心" for item in scored_items)
    cache_hits = sum(item.cache_hit for item in image_items)
    fallback_count = sum(item.used_fallback for item in image_items)
    unique_hashes = len({item.thumbnail_hash for item in image_items if item.thumbnail_hash})

    top_context = Counter(item.category_label for item in scored_items).most_common(1)
    top_distractor = Counter(
        item.category_label
        for item in scored_items
        if item.status != "专注"
        and (item.distraction_score >= 0.58 or item.relevance_score < 0.55 or item.category_type != "focus")
    ).most_common(1)

    return {
        "goal": profile.raw_goal,
        "goal_id": profile.goal_type,
        "goal_label": profile.raw_goal,
        "goal_type": profile.goal_type,
        "keywords": profile.keywords[:10],
        "total_images": len(items),
        "valid_images": len(image_items),
        "scored_images": len(scored_items),
        "review_count": review_count,
        "focus_count": focus_count,
        "drift_count": drift_count,
        "distract_count": distract_count,
        "focus_ratio": round((focus_count / max(len(scored_items), 1)) * 100, 1),
        "avg_focus_score": round(sum(item.focus_score for item in scored_items) / max(len(scored_items), 1), 1),
        "avg_relevance_score": round(sum(item.relevance_score for item in scored_items) / max(len(scored_items), 1), 3),
        "avg_ocr_quality": round(sum(item.ocr_quality_score for item in scored_items) / max(len(scored_items), 1), 3),
        "avg_keyword_score": round(sum(item.keyword_hit_score for item in scored_items) / max(len(scored_items), 1), 3),
        "avg_semantic_score": round(sum(item.semantic_match_score for item in scored_items) / max(len(scored_items), 1), 3),
        "avg_strong_hit_score": round(sum(item.strong_hit_score for item in scored_items) / max(len(scored_items), 1), 3),
        "avg_coverage_score": round(sum(item.coverage_score for item in scored_items) / max(len(scored_items), 1), 3),
        "avg_structure_score": round(sum(item.structure_score for item in scored_items) / max(len(scored_items), 1), 3),
        "top_context": top_context[0][0] if top_context else ("待复核" if review_count else "暂无明显场景"),
        "top_distractor": top_distractor[0][0] if top_distractor else ("待复核" if review_count and not scored_items else "无明显高频偏离场景"),
        "cache_hits": cache_hits,
        "fallback_count": fallback_count,
        "unique_hashes": unique_hashes,
        "processing_ms": processing_ms,
        "throughput_images_per_sec": round(len(scored_items) / max(processing_ms / 1000, 0.001), 2),
        "suggestions": summary_suggestions(profile, items),
    }



