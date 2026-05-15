import pandas as pd

from src.image_enrichment import commons_file_url, ensure_image_columns, first_valid_existing_image_url, is_valid_image_url
from src.media_validation import classify_media_url, validate_media_url


def test_v37_preserves_image_enrichment_public_contract() -> None:
    assert commons_file_url("Foo bar.jpg").endswith("Foo_bar.jpg")
    assert is_valid_image_url("https://example.org/photo.webp")
    assert not is_valid_image_url("https://example.org/audio.m4a")


def test_v37_ensure_image_columns_moves_bad_media() -> None:
    df = pd.DataFrame([
        {"scientific_name": "A", "image_url": "https://example.org/photo.jpg"},
        {"scientific_name": "B", "image_url": "https://example.org/audio.mp3"},
        {"scientific_name": "C", "image_url": "https://example.org/media/12345"},
    ])
    result = ensure_image_columns(df)
    assert bool(result.loc[0, "has_image"])
    assert result.loc[1, "image_url"] == ""
    assert result.loc[1, "media_type"] == "audio"
    assert result.loc[2, "image_url"] == ""
    assert result.loc[2, "unverified_media_url"] == "https://example.org/media/12345"


def test_v37_first_valid_existing_image_url_skips_audio() -> None:
    row = pd.Series({"image_url": "https://example.org/sound.wav", "thumbnail_url": "https://example.org/species.png"})
    assert first_valid_existing_image_url(row) == "https://example.org/species.png"


def test_v37_media_validation_has_both_old_and_new_contracts() -> None:
    decision = classify_media_url("https://example.org/photo.jpg")
    assert decision.is_valid_image
    assert decision.has_image
    assert decision.image_url.endswith("photo.jpg")
    assert validate_media_url("https://example.org/media/12345").image_validation_status == "unknown_no_extension"
