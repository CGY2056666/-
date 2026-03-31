from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from .config import AnalyzerConfig, DEFAULT_CONFIG
from .models import OcrResult
from .scoring import clean_ocr_text, score_ocr_quality
from .utils import LRUCache, clamp


def decode_image(file_bytes: bytes) -> np.ndarray | None:
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


@lru_cache(maxsize=1)
def _configure_tesseract_cached(candidates: tuple[str, ...]) -> tuple[bool, str]:
    for candidate in candidates:
        if not candidate:
            continue
        if Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            tessdata = Path(candidate).parent / "tessdata"
            if tessdata.exists():
                import os

                os.environ.setdefault("TESSDATA_PREFIX", str(tessdata))
            return True, candidate
    return False, "未找到 Tesseract，可安装后启用 OCR"


def configure_tesseract(config: AnalyzerConfig = DEFAULT_CONFIG) -> tuple[bool, str]:
    return _configure_tesseract_cached(tuple(config.tesseract_candidates))


def crop_focus_region(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    top = int(height * 0.04)
    bottom = int(height * 0.96)
    left = int(width * 0.02)
    right = int(width * 0.98)
    return image[top:bottom, left:right]


def resize_for_ocr(image: np.ndarray, config: AnalyzerConfig) -> np.ndarray:
    height, width = image.shape[:2]
    max_edge = max(height, width)
    min_edge = min(height, width)

    if max_edge > config.max_image_edge:
        scale = config.max_image_edge / max_edge
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if min_edge < config.min_image_edge:
        scale = config.min_image_edge / max(min_edge, 1)
        scale = min(scale, 1.8)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return image


def preprocess_primary(image: np.ndarray, config: AnalyzerConfig) -> np.ndarray:
    cropped = crop_focus_region(image)
    resized = resize_for_ocr(cropped, config)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_fallback(image: np.ndarray, config: AnalyzerConfig) -> np.ndarray:
    cropped = crop_focus_region(image)
    resized = resize_for_ocr(cropped, config)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )


def average_hash(image: np.ndarray, hash_size: int = 16) -> str:
    gray = cv2.cvtColor(crop_focus_region(image), cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = float(thumb.mean())
    bits = "".join("1" if pixel > mean else "0" for pixel in thumb.flatten())
    width = hash_size * hash_size // 4
    return hex(int(bits, 2))[2:].zfill(width)


def hash_distance(left: str, right: str) -> int:
    if not left or not right:
        return 999
    left_bits = bin(int(left, 16))[2:].zfill(len(left) * 4)
    right_bits = bin(int(right, 16))[2:].zfill(len(right) * 4)
    return sum(bit_left != bit_right for bit_left, bit_right in zip(left_bits, right_bits))


class FastOCREngine:
    def __init__(self, config: AnalyzerConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.cache = LRUCache(max_size=config.ocr_cache_size)

    def status(self) -> tuple[bool, str]:
        return configure_tesseract(self.config)

    def _run_tesseract(self, image: np.ndarray, config_string: str) -> tuple[str, str | None]:
        try:
            text = pytesseract.image_to_string(image, lang="chi_sim+eng", config=config_string)
            return text, None
        except pytesseract.TesseractError:
            text = pytesseract.image_to_string(image, lang="eng", config=config_string)
            return text, "未检测到中文语言包，已降级为英文 OCR"

    def extract(self, image: np.ndarray, thumbnail_hash: str) -> OcrResult:
        cached = self.cache.get(thumbnail_hash)
        if cached:
            return OcrResult(
                text=cached.text,
                quality_score=cached.quality_score,
                engine=cached.engine,
                cache_hit=True,
                used_fallback=cached.used_fallback,
                warning=cached.warning,
                thumbnail_hash=thumbnail_hash,
                processing_ms=1,
            )

        ready, engine = self.status()
        if not ready:
            return OcrResult(text="", quality_score=0.0, engine=engine, warning=engine, thumbnail_hash=thumbnail_hash)

        started = time.perf_counter()
        primary_image = preprocess_primary(image, self.config)
        primary_text_raw, primary_warning = self._run_tesseract(primary_image, self.config.primary_ocr_config)
        primary_text = clean_ocr_text(primary_text_raw)
        primary_quality = score_ocr_quality(primary_text)

        best_text = primary_text
        best_quality = primary_quality
        warning = primary_warning
        used_fallback = False

        if primary_quality < 0.32 or len(primary_text) < 24:
            fallback_image = preprocess_fallback(image, self.config)
            fallback_text_raw, fallback_warning = self._run_tesseract(fallback_image, self.config.fallback_ocr_config)
            fallback_text = clean_ocr_text(fallback_text_raw)
            fallback_quality = score_ocr_quality(fallback_text)
            if fallback_quality >= best_quality + 0.05 or len(fallback_text) > len(best_text) + 20:
                best_text = fallback_text
                best_quality = fallback_quality
                warning = fallback_warning or warning
                used_fallback = True

        elapsed = int((time.perf_counter() - started) * 1000)
        result = OcrResult(
            text=best_text,
            quality_score=round(clamp(best_quality), 3),
            engine=engine,
            cache_hit=False,
            used_fallback=used_fallback,
            warning=warning,
            thumbnail_hash=thumbnail_hash,
            processing_ms=elapsed,
        )
        self.cache.set(thumbnail_hash, result)
        return result
