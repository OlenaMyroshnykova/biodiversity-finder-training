"""Media URL validation for Biodiversity Finder artifacts.

This module keeps the public contract used by older tests while supporting the
new training pipeline contract. The key rule is simple: ``image_url`` must contain
only render-safe still images. Suspicious media can be kept in
``unverified_media_url`` for diagnostics, but the app should not render it as a
species image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable
from urllib.parse import unquote, urlparse

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".jfif"}
INVALID_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".oga", ".flac", ".aac"}
INVALID_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
INVALID_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".csv", ".json", ".xml", ".html", ".htm"}
INVALID_IMAGE_EXTENSIONS = {".svg"}
BAD_IMAGE_MARKERS = {
    "placeholder", "no_image", "noimage", "no-photo", "no_photo",
    "logo", "icon", "range_map", "distribution_map", "map_",
    "blank", "transparent",
}


@dataclass(frozen=True)
class MediaValidation:
    """Validation decision for one media URL.

    The primary fields match the v35 contract. The properties below provide the
    v36 contract used by image enrichment code.
    """

    url: str
    is_valid_image: bool
    media_type: str
    status: str
    extension: str
    reason: str

    @property
    def original_url(self) -> str:
        return self.url

    @property
    def image_url(self) -> str:
        return self.url if self.is_valid_image else ""

    @property
    def unverified_media_url(self) -> str:
        return "" if self.is_valid_image or not self.url else self.url

    @property
    def image_validation_status(self) -> str:
        return self.status

    @property
    def has_image(self) -> bool:
        return self.is_valid_image


MediaValidationResult = MediaValidation


def clean_url(value: object) -> str:
    """Return a safe stripped URL string."""
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split()[0].strip("'\"")


def clean_media_url(value: object) -> str:
    """Backward-compatible alias used by image_enrichment."""
    return clean_url(value)


def get_url_extension(value: object) -> str:
    """Extract a lower-case extension from a URL path.

    Query strings are ignored. Wikimedia ``Special:FilePath`` URLs still expose
    the original filename in the path, so ``.../FilePath/Foo.jpg`` returns
    ``.jpg``.
    """
    url = clean_url(value)
    if not url:
        return ""
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    return PurePosixPath(path).suffix.lower()


def has_bad_image_marker(value: object, markers: Iterable[str] = BAD_IMAGE_MARKERS) -> bool:
    """Return True when a URL looks like a placeholder/map/icon."""
    lower = clean_url(value).lower()
    return any(marker in lower for marker in markers)


def classify_media_url(value: object) -> MediaValidation:
    """Classify a media URL without performing network requests.

    This intentionally prefers precision over recall: URLs without a clear image
    extension are not promoted to ``image_url``. They can still be kept in
    ``unverified_media_url`` for later manual/HEAD validation.
    """
    url = clean_url(value)
    if not url:
        return MediaValidation(url, False, "empty", "empty", "", "empty value")

    if not url.startswith(("http://", "https://")):
        return MediaValidation(url, False, "invalid", "invalid_non_http", "", "not an HTTP URL")

    extension = get_url_extension(url)

    if has_bad_image_marker(url):
        return MediaValidation(url, False, "invalid", "invalid_bad_marker", extension, "placeholder/map/icon marker")

    if extension in VALID_IMAGE_EXTENSIONS:
        return MediaValidation(url, True, "image", "valid_image_extension", extension, "supported image extension")

    if extension in INVALID_IMAGE_EXTENSIONS:
        return MediaValidation(url, False, "image", "invalid_svg", extension, "SVG is not used for species cards")

    if extension in INVALID_AUDIO_EXTENSIONS:
        return MediaValidation(url, False, "audio", "invalid_audio", extension, "audio file")

    if extension in INVALID_VIDEO_EXTENSIONS:
        return MediaValidation(url, False, "video", "invalid_video", extension, "video file")

    if extension in INVALID_DOCUMENT_EXTENSIONS:
        return MediaValidation(url, False, "document", "invalid_document", extension, "document/text file")

    if extension:
        return MediaValidation(url, False, "unknown", "invalid_unknown_extension", extension, "unsupported media extension")

    return MediaValidation(url, False, "unknown", "unknown_no_extension", "", "URL has no verifiable image extension")


def validate_media_url(value: object) -> MediaValidation:
    """v36-compatible name for classify_media_url."""
    return classify_media_url(value)


def is_valid_image_url(value: object) -> bool:
    """Return True only for URLs safe to render as species card images."""
    return classify_media_url(value).is_valid_image


def is_unverified_media_url(value: object) -> bool:
    """Return True for HTTP media that is not render-safe but may be useful to keep."""
    decision = classify_media_url(value)
    return decision.status in {"unknown_no_extension", "invalid_unknown_extension"}


def validate_media_series(values) -> list[MediaValidation]:
    """Validate a pandas-like series/list and return decisions."""
    return [classify_media_url(value) for value in values]
