import pandas as pd

from src.image_enrichment import ensure_image_columns, first_valid_existing_image_url
from src.media_validation import classify_media_url, is_valid_image_url


def test_audio_urls_are_not_valid_images() -> None:
    decision = classify_media_url("https://example.org/species-call.m4a")

    assert not decision.is_valid_image
    assert decision.media_type == "audio"
    assert decision.status == "invalid_audio"
    assert not is_valid_image_url("https://example.org/species-call.m4a")


def test_urls_without_extension_are_not_rendered_as_card_images() -> None:
    decision = classify_media_url("https://api.gbif.org/v1/image/12345")

    assert not decision.is_valid_image
    assert decision.status == "unknown_no_extension"


def test_standard_image_extensions_are_valid() -> None:
    assert is_valid_image_url("https://example.org/photos/crocodile.webp")
    assert is_valid_image_url("https://commons.wikimedia.org/wiki/Special:FilePath/Foo_bar.jpg")


def test_ensure_image_columns_moves_unverified_media_out_of_image_url() -> None:
    df = pd.DataFrame(
        [
            {"scientific_name": "A", "image_url": "https://example.org/photo.jpg"},
            {"scientific_name": "B", "image_url": "https://example.org/audio.mp3"},
            {"scientific_name": "C", "image_url": "https://example.org/media/12345"},
        ]
    )

    result = ensure_image_columns(df)

    assert result.loc[0, "has_image"] is True or bool(result.loc[0, "has_image"])
    assert result.loc[1, "image_url"] == ""
    assert result.loc[1, "media_type"] == "audio"
    assert result.loc[2, "image_url"] == ""
    assert result.loc[2, "unverified_media_url"] == "https://example.org/media/12345"


def test_first_valid_existing_image_url_skips_audio_and_uses_next_image() -> None:
    row = pd.Series(
        {
            "image_url": "https://example.org/sound.wav",
            "thumbnail_url": "https://example.org/species.png",
        }
    )

    assert first_valid_existing_image_url(row) == "https://example.org/species.png"
