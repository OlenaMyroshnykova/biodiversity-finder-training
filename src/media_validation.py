"""Utilities for validating media URLs before publishing image artifacts.

The training pipeline should put only safe, renderable image URLs into image_url.
Suspicious media is kept in unverified_media_url for diagnostics, but the app should
not try to render it as a species image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".jfif"}
INVALID_MEDIA_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".ogg", ".oga", ".flac",
    ".mp4", ".mov", ".avi", ".webm", ".mkv",
    ".pdf", ".svg", ".txt", ".html", ".htm",
}
INVALID_MARKERS = {
    "placeholder", "no_image", "noimage", "missing", "logo", "icon", "map",
}


@dataclass(frozen=True)
class MediaValidationResult:
    original_url: str
    image_url: str
    unverified_media_url: str
    media_type: str
    image_validation_status: str
    has_image: bool


def clean_media_url(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split()[0].strip("'\"")


def _url_extension(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    return PurePosixPath(path).suffix.lower()


def validate_media_url(value: object) -> MediaValidationResult:
    url = clean_media_url(value)
    lowered = url.lower()
    if not url:
        return MediaValidationResult("", "", "", "missing", "empty", False)
    if not lowered.startswith(("http://", "https://")):
        return MediaValidationResult(url, "", url, "unknown", "invalid_scheme", False)
    if any(marker in lowered for marker in INVALID_MARKERS):
        return MediaValidationResult(url, "", url, "unknown", "invalid_placeholder", False)

    ext = _url_extension(url)
    if ext in VALID_IMAGE_EXTENSIONS:
        return MediaValidationResult(url, url, "", "image", "valid_extension", True)
    if ext in INVALID_MEDIA_EXTENSIONS:
        media_type = "audio" if ext in {".mp3", ".m4a", ".wav", ".ogg", ".oga", ".flac"} else "non_image"
        return MediaValidationResult(url, "", url, media_type, f"invalid_extension:{ext}", False)
    if not ext:
        return MediaValidationResult(url, "", url, "unknown", "unknown_no_extension", False)
    return MediaValidationResult(url, "", url, "unknown", f"unknown_extension:{ext}", False)


def is_valid_image_url(value: object) -> bool:
    return validate_media_url(value).has_image
