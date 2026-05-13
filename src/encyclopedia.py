"""Construcción de la enciclopedia inteligente de especies."""

from __future__ import annotations

import pandas as pd


def build_species_encyclopedia(features_df: pd.DataFrame) -> pd.DataFrame:
    """Construye una enciclopedia agregada: una fila por especie.

    Acepta columnas originales de GBIF y columnas normalizadas por el pipeline.
    """
    if features_df.empty:
        return pd.DataFrame()

    df = features_df.copy()

    df["scientific_name"] = get_first_existing_column(
        df,
        ["scientific_name", "scientificName"],
        default="Unknown species",
    )
    df["taxon_class"] = get_first_existing_column(
        df,
        ["taxon_class", "class"],
        default="Unknown class",
    )
    df["taxon_order"] = get_first_existing_column(
        df,
        ["taxon_order", "order"],
        default="Unknown order",
    )

    df["country_code"] = get_first_existing_column(
        df,
        ["country_code", "countryCode", "country"],
        default="Unknown country",
    )
    df["basis_of_record"] = get_first_existing_column(
        df,
        ["basis_of_record", "basisOfRecord"],
        default="Unknown basis",
    )
    df["decimal_latitude"] = get_first_existing_numeric_column(
        df,
        ["decimal_latitude", "decimalLatitude", "latitude", "lat"],
    )
    df["decimal_longitude"] = get_first_existing_numeric_column(
        df,
        ["decimal_longitude", "decimalLongitude", "longitude", "lon", "lng"],
    )

    required_columns = {
        "kingdom": "Unknown kingdom",
        "phylum": "Unknown phylum",
        "family": "Unknown family",
        "genus": "Unknown genus",
        "species": "Unknown species",
        "season": "desconocida",
        "source_query": "unknown_source",
    }

    for column, default_value in required_columns.items():
        if column not in df.columns:
            df[column] = default_value

    if "year" not in df.columns:
        df["year"] = pd.NA

    encyclopedia_df = (
        df.groupby("scientific_name", as_index=False)
        .agg(
            kingdom=("kingdom", most_common_value),
            phylum=("phylum", most_common_value),
            taxon_class=("taxon_class", most_common_value),
            taxon_order=("taxon_order", most_common_value),
            family=("family", most_common_value),
            genus=("genus", most_common_value),
            species=("species", most_common_value),
            observations=("scientific_name", "count"),
            countries=("country_code", join_unique_values),
            first_year=("year", "min"),
            last_year=("year", "max"),
            avg_latitude=("decimal_latitude", "mean"),
            avg_longitude=("decimal_longitude", "mean"),
            most_common_basis=("basis_of_record", most_common_value),
            most_common_season=("season", most_common_value),
            source_queries=("source_query", join_unique_values),
            iucn_status=(
                "iucn_red_list_category", most_common_value
            ) if "iucn_red_list_category" in df.columns else (
                "scientific_name", lambda x: "Unknown"
            ),
        )
    )

    encyclopedia_df["profile_text"] = encyclopedia_df.apply(build_profile_text, axis=1)
    encyclopedia_df["search_document"] = encyclopedia_df.apply(build_search_document, axis=1)

    return (
        encyclopedia_df
        .sort_values(
            ["observations", "scientific_name"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def get_first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    default: str,
) -> pd.Series:
    """Devuelve la primera columna existente de una lista de candidatas."""
    for column in candidates:
        if column in df.columns:
            return df[column].fillna(default)

    return pd.Series([default] * len(df), index=df.index)


def get_first_existing_numeric_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> pd.Series:
    """Devuelve la primera columna numérica existente de una lista."""
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")

    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")


def most_common_value(values: pd.Series) -> str:
    """Devuelve el valor más frecuente no nulo."""
    clean_values = values.dropna().astype(str)
    clean_values = clean_values[clean_values.str.strip() != ""]

    if clean_values.empty:
        return "Unknown"

    mode_values = clean_values.mode()

    if mode_values.empty:
        return clean_values.iloc[0]

    return mode_values.iloc[0]


def join_unique_values(values: pd.Series) -> str:
    """Une valores únicos de una columna en texto."""
    clean_values = values.dropna().astype(str)
    clean_values = clean_values[clean_values.str.strip() != ""]

    unique_values = sorted(set(clean_values))

    if not unique_values:
        return "Unknown"

    return ", ".join(unique_values)


def build_profile_text(row: pd.Series) -> str:
    """Crea un texto descriptivo breve para una especie."""
    return (
        f"{row['scientific_name']} pertenece a la clase {row['taxon_class']}, "
        f"al orden {row['taxon_order']} y a la familia {row['family']}. "
        f"En el dataset tiene {row['observations']} observaciones "
        f"entre {row['first_year']} y {row['last_year']}."
    )


def build_search_document(row: pd.Series) -> str:
    """Construye el documento de búsqueda con términos científicos y humanos."""
    scientific_text = " ".join(
        [
            str(row.get("scientific_name", "")),
            str(row.get("kingdom", "")),
            str(row.get("phylum", "")),
            str(row.get("taxon_class", "")),
            str(row.get("taxon_order", "")),
            str(row.get("family", "")),
            str(row.get("genus", "")),
            str(row.get("species", "")),
            str(row.get("countries", "")),
            str(row.get("most_common_basis", "")),
            str(row.get("source_queries", "")),
            str(row.get("profile_text", "")),
        ]
    )

    human_terms = build_human_search_terms(row)

    return f"{scientific_text} {human_terms}".strip()


def build_human_search_terms(row: pd.Series) -> str:
    """Añade vocabulario humano según taxonomía y fuente de descarga.

    Esto no sustituye los datos científicos, solo ayuda a que el buscador
    entienda términos cotidianos.
    """
    combined_text = " ".join(
        [
            str(row.get("scientific_name", "")),
            str(row.get("kingdom", "")),
            str(row.get("taxon_class", "")),
            str(row.get("taxon_order", "")),
            str(row.get("family", "")),
            str(row.get("genus", "")),
            str(row.get("source_queries", "")),
        ]
    ).lower()

    terms: list[str] = []

    if "animalia" in combined_text:
        terms.extend(["animal", "animales", "fauna", "organismo", "especie"])

    if "plantae" in combined_text or "magnoliopsida" in combined_text:
        terms.extend(
            [
                "planta",
                "plantas",
                "vegetal",
                "flor",
                "flores",
                "flower",
                "flowers",
                "flowering plant",
                "plant",
            ]
        )

    if "aves" in combined_text:
        terms.extend(
            [
                "ave",
                "aves",
                "pajaro",
                "pájaro",
                "pajaros",
                "pájaros",
                "bird",
                "birds",
                "alas",
                "plumas",
                "vuela",
                "volador",
            ]
        )

    if "phoenicopterus roseus" in combined_text or "flamingo_pink_bird" in combined_text:
        terms.extend(
            [
                "flamenco",
                "flamingo",
                "pajaro rosa",
                "pájaro rosa",
                "ave rosa",
                "pink bird",
                "humedal",
                "laguna",
            ]
        )

    if "mammalia" in combined_text:
        terms.extend(
            [
                "mamifero",
                "mamífero",
                "mamiferos",
                "mamíferos",
                "mammal",
                "mammals",
            ]
        )

    if "ursus maritimus" in combined_text or "polar_bear" in combined_text:
        terms.extend(
            [
                "oso",
                "oso polar",
                "polar bear",
                "animal polar",
                "hielo",
                "ice",
                "nieve",
                "snow",
                "arctico",
                "ártico",
                "arctic",
            ]
        )

    if "insecta" in combined_text:
        terms.extend(
            [
                "insecto",
                "insectos",
                "insect",
                "insects",
                "bicho",
                "bichos",
            ]
        )

    if "lepidoptera" in combined_text or "butterflies_lepidoptera" in combined_text:
        terms.extend(
            [
                "mariposa",
                "mariposas",
                "butterfly",
                "butterflies",
                "polilla",
                "polillas",
                "moth",
                "moths",
                "alas",
                "colores",
            ]
        )

    if "amphibia" in combined_text or "amphibians" in combined_text:
        terms.extend(
            [
                "rana",
                "ranas",
                "anfibio",
                "anfibios",
                "frog",
                "frogs",
                "agua",
                "rio",
                "río",
                "humedo",
                "húmedo",
                "charca",
            ]
        )

    if "accipitridae" in combined_text or "raptors_accipitridae" in combined_text:
        terms.extend(
            [
                "ave rapaz",
                "rapaz",
                "rapaces",
                "aguila",
                "águila",
                "eagle",
                "hawk",
                "halcon",
                "halcón",
                "montaña",
                "cazador",
            ]
        )

    if "actinopterygii" in combined_text or "bony_fish" in combined_text:
        terms.extend([
            "pez", "peces", "pescado", "fish", "agua", "acuatico", "acuático",
            "río", "lago", "рыба", "рыбы",
        ])

    if "reptilia" in combined_text or "reptiles_crocodylia" in combined_text or "crocodylia" in combined_text:
        terms.extend([
            "reptil", "reptiles", "cocodrilo", "cocodrilos", "crocodile",
            "caiman", "lagarto", "serpiente", "snake", "iguana",
            "крокодил", "рептилия", "ящерица", "змея", "escamas",
        ])

    if "chondrichthyes" in combined_text or "sharks_rays" in combined_text:
        terms.extend([
            "tiburon", "tiburón", "tiburones", "shark", "sharks",
            "raya", "rayas", "ray", "акула", "акулы", "скат",
            "mar", "oceano", "marino",
        ])

    if "arachnida" in combined_text or "arachnids" in combined_text:
        terms.extend([
            "araña", "arañas", "spider", "escorpion", "escorpión", "scorpion",
            "aracnido", "арácnido", "паук", "скорпион",
        ])

    if "fungi" in combined_text or "fungi_mushrooms" in combined_text:
        terms.extend([
            "hongo", "hongos", "seta", "setas", "mushroom", "mushrooms",
            "гриб", "грибы", "descomponedor",
        ])

    return " ".join(sorted(set(terms)))
