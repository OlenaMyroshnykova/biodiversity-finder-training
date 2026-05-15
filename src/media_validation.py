"""Media URL validation for Biodiversity Finder artifacts.

The training pipeline can receive media links from GBIF, Wikidata and
Wikimedia. Some public biodiversity records contain audio, video, PDF files or
URLs without an image extension. The Streamlit app should only receive confirmed
still-image URLs in ``image_url``; everything uncertain stays out of the visual
card and can be kept as technical metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable
from urllib.parse import unquote, urlparse

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".jfif",
}

INVALID_AUDIO_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".wav",
    ".ogg",
    ".oga",
    ".flac",
    ".aac",
}

INVALID_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".webm",
    ".mkv",
    ".m4v",
}

INVALID_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".csv",
    ".json",
    ".xml",
    ".html",
    ".htm",
}

INVALID_IMAGE_EXTENSIONS = {
    ".svg",  # often maps/icons/logos; Streamlit can render but cards look bad.
}

BAD_IMAGE_MARKERS = {
    "placeholder",
    "no_image",
    "noimage",
    "no-photo",
    "no_photo",
    "logo",
    "icon",
    "range_map",
    "distribution_map",
    "map_",
    "blank",
    "transparent",
}


@dataclass(frozen=True)
class MediaValidation:
    """Validation decision for one media URL."""

    url: str
    is_valid_image: bool
    media_type: str
    status: str
    extension: str
    reason: str


def clean_url(value: object) -> str:
    """Return a safe stripped URL string."""
    return str(value or "").strip()


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
    suffix = PurePosixPath(path).suffix.lower()
    return suffix


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
