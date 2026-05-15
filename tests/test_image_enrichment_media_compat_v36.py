import pandas as pd

from src.image_enrichment import add_images_to_encyclopedia, commons_file_url, is_valid_image_url


def test_commons_file_url_stays_public_contract():
    assert commons_file_url("Brown bear.jpg").endswith("/Brown_bear.jpg")


def test_audio_url_is_removed_from_image_url_without_losing_diagnostics():
    df = pd.DataFrame(
        {
            "scientific_name": ["Species audio"],
            "image_url": ["https://example.org/audio/species.m4a"],
            "image_source": ["GBIF media"],
        }
    )

    result, records = add_images_to_encyclopedia(df, use_api=False)

    assert result.loc[0, "image_url"] == ""
    assert result.loc[0, "unverified_media_url"].endswith("species.m4a")
    assert result.loc[0, "has_image"] is False or not result.loc[0, "has_image"]
    assert records.loc[0, "image_url"] == ""


def test_validation_rejects_unknown_no_extension_urls():
    assert is_valid_image_url("https://example.org/photo.jpg")
    assert not is_valid_image_url("https://example.org/media/12345")
