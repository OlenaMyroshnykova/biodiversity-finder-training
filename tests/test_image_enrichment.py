import pandas as pd

from src.image_enrichment import add_images_to_encyclopedia, commons_file_url, is_valid_image_url


def test_commons_file_url_builds_special_filepath_url():
    url = commons_file_url("Brown bear.jpg")
    assert url == "https://commons.wikimedia.org/wiki/Special:FilePath/Brown_bear.jpg"


def test_image_enrichment_preserves_existing_images_without_api():
    df = pd.DataFrame(
        {
            "scientific_name": ["Ursus arctos"],
            "image_url": ["https://example.org/bear.jpg"],
            "image_source": ["GBIF occurrence media"],
        }
    )

    enriched_df, image_df = add_images_to_encyclopedia(df, use_api=False)

    assert enriched_df.loc[0, "image_url"] == "https://example.org/bear.jpg"
    assert image_df.loc[0, "image_source"] == "GBIF occurrence media"


def test_image_url_validation_rejects_placeholders():
    assert is_valid_image_url("https://example.org/species.jpg")
    assert not is_valid_image_url("https://example.org/placeholder.jpg")
    assert not is_valid_image_url("")
