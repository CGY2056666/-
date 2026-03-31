from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace

from .config import AnalyzerConfig, DEFAULT_CONFIG
from .deepseek_scoring import DeepSeekScorer
from .goal_profiles import build_goal_profile
from .models import FrameAnalysis, GoalProfile
from .ocr import FastOCREngine, average_hash, decode_image, hash_distance
from .scoring import build_summary, finalize_frame_scores


@dataclass
class PreparedFrame:
    index: int
    filename: str
    image_valid: bool
    image: object | None = None
    width: int = 0
    height: int = 0
    thumbnail_hash: str = ""
    duplicate_of: int | None = None


class FocusAnalyzer:
    def __init__(self, config: AnalyzerConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.ocr_engine = FastOCREngine(config)
        self.scorer = DeepSeekScorer(config)

    def configure_tesseract(self) -> tuple[bool, str]:
        return self.ocr_engine.status()

    def scoring_status(self) -> dict[str, object]:
        return self.scorer.status()

    def allowed_file(self, filename: str) -> bool:
        return filename.lower().endswith(self.config.supported_extensions)

    def analyze_uploads(self, goal: str, uploads: list[tuple[str, bytes]]) -> dict:
        profile, items, summary = self.analyze_objects(goal, uploads)
        ready, engine = self.configure_tesseract()
        scoring = self.scoring_status()
        return {
            "goal_profile": profile.to_dict(),
            "summary": summary,
            "items": [item.to_dict() for item in items],
            "runtime": {
                "ocr_ready": ready,
                "ocr_engine": engine,
                "workers": self.config.max_workers,
                "llm_ready": bool(scoring["configured"]),
                "llm_provider": str(scoring["provider"]),
                "llm_model": str(scoring["model"]),
            },
        }

    def analyze_objects(self, goal: str, uploads: list[tuple[str, bytes]]) -> tuple[GoalProfile, list[FrameAnalysis], dict]:
        started = time.perf_counter()
        profile = build_goal_profile(goal)
        prepared = self._prepare_frames(uploads[: self.config.max_uploads])
        analyzed = self._analyze_prepared_frames(profile, prepared)
        finalize_frame_scores(analyzed)
        summary = build_summary(profile, analyzed, int((time.perf_counter() - started) * 1000))
        return profile, analyzed, summary

    def analyze_frame(self, profile: GoalProfile, filename: str, file_bytes: bytes) -> FrameAnalysis:
        prepared = self._prepare_frames([(filename, file_bytes)])
        frame = prepared[0]
        if not frame.image_valid:
            return FrameAnalysis(
                index=0,
                filename=filename,
                image_valid=False,
                status="无法分析",
                decision_reason="图片无法解码，请重新共享更清晰的窗口后再试。",
            )
        return self._analyze_unique_frame(profile, frame)

    def _prepare_frames(self, uploads: list[tuple[str, bytes]]) -> list[PreparedFrame]:
        prepared: list[PreparedFrame] = []
        last_unique: PreparedFrame | None = None

        for index, (filename, file_bytes) in enumerate(uploads):
            image = decode_image(file_bytes)
            if image is None:
                prepared.append(PreparedFrame(index=index, filename=filename, image_valid=False))
                continue

            height, width = image.shape[:2]
            thumbnail_hash = average_hash(image, self.config.thumbnail_hash_size)
            duplicate_of = None

            if last_unique and hash_distance(thumbnail_hash, last_unique.thumbnail_hash) <= self.config.duplicate_distance_threshold:
                duplicate_of = last_unique.index

            frame = PreparedFrame(
                index=index,
                filename=filename,
                image_valid=True,
                image=image,
                width=width,
                height=height,
                thumbnail_hash=thumbnail_hash,
                duplicate_of=duplicate_of,
            )
            prepared.append(frame)
            if duplicate_of is None:
                last_unique = frame

        return prepared

    def _analyze_prepared_frames(self, profile: GoalProfile, prepared: list[PreparedFrame]) -> list[FrameAnalysis]:
        by_index: dict[int, FrameAnalysis] = {}
        unique_frames = [frame for frame in prepared if frame.image_valid and frame.duplicate_of is None]

        if unique_frames:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._analyze_unique_frame, profile, frame): frame.index
                    for frame in unique_frames
                }
                for future in as_completed(futures):
                    frame_index = futures[future]
                    by_index[frame_index] = future.result()

        items: list[FrameAnalysis] = []
        for frame in prepared:
            if not frame.image_valid:
                items.append(
                    FrameAnalysis(
                        index=frame.index,
                        filename=frame.filename,
                        image_valid=False,
                        status="无法分析",
                        decision_reason="图片无法解码，请重新上传更完整的截图。",
                    )
                )
                continue

            if frame.duplicate_of is not None and frame.duplicate_of in by_index:
                source = by_index[frame.duplicate_of]
                clone = replace(
                    source,
                    index=frame.index,
                    filename=frame.filename,
                    duplicate_of=source.filename,
                    cache_hit=True,
                    processing_ms=1,
                    fallback_reason="与上一张截图高度相似，已复用识别和评分结果。",
                )
                items.append(clone)
                continue

            items.append(by_index[frame.index])

        return items

    def _analyze_unique_frame(self, profile: GoalProfile, frame: PreparedFrame) -> FrameAnalysis:
        assert frame.image is not None
        started = time.perf_counter()
        ocr_result = self.ocr_engine.extract(frame.image, frame.thumbnail_hash)
        score_data = self.scorer.score(
            profile=profile,
            filename=frame.filename,
            text=ocr_result.text,
            ocr_quality=ocr_result.quality_score,
        )

        fallback_reason = str(score_data["fallback_reason"])
        if ocr_result.warning:
            fallback_reason = f"{fallback_reason}；{ocr_result.warning}".strip("；") if fallback_reason else ocr_result.warning

        return FrameAnalysis(
            index=frame.index,
            filename=frame.filename,
            image_valid=True,
            width=frame.width,
            height=frame.height,
            thumbnail_hash=frame.thumbnail_hash,
            ocr_text=ocr_result.text,
            ocr_quality_score=ocr_result.quality_score,
            semantic_match_score=float(score_data["semantic_match_score"]),
            keyword_hit_score=float(score_data["keyword_hit_score"]),
            strong_hit_score=float(score_data["strong_hit_score"]),
            coverage_score=float(score_data["coverage_score"]),
            structure_score=float(score_data["structure_score"]),
            app_context_score=float(score_data["app_context_score"]),
            base_relevance_score=float(score_data["relevance_score"]),
            relevance_score=float(score_data["relevance_score"]),
            focus_score=float(score_data["focus_score"]),
            focus_probability=float(score_data["focus_probability"]),
            score_confidence=float(score_data["confidence"]),
            distraction_score=float(score_data["distraction_score"]),
            category_label=str(score_data["category_label"]),
            category_type=str(score_data["category_type"]),
            matched_keywords=list(score_data["matched_keywords"]),
            strong_hits=list(score_data["strong_hits"]),
            scene_hits=list(score_data["scene_hits"]),
            support_hits=list(score_data["support_hits"]),
            negative_hits=list(score_data["negative_hits"]),
            matched_context_patterns=list(score_data["matched_context_patterns"]),
            status=str(score_data["status"]),
            decision_reason=str(score_data["reason"]),
            base_decision_reason=str(score_data["reason"]),
            cache_hit=ocr_result.cache_hit,
            used_fallback=ocr_result.used_fallback,
            processing_ms=max(int((time.perf_counter() - started) * 1000), ocr_result.processing_ms),
            warning=ocr_result.warning,
            fallback_reason=fallback_reason,
            positive_rule=str(score_data["positive_rule"]),
            negative_rule=str(score_data["negative_rule"]),
            score_breakdown=dict(score_data["score_breakdown"]),
            scoring_source="deepseek",
            review_required=bool(score_data["review_required"]),
        )


