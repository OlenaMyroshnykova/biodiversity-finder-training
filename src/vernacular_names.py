"""Nombres comunes / vernacular names para especies.

Este módulo enriquece la enciclopedia con nombres humanos usando fuentes externas:

1. GBIF Species API / vernacularNames.
2. Wikidata SPARQL por GBIF taxon ID P846.
3. Wikidata SPARQL por taxon name P225.
4. Wikidata Search API como fallback.
5. Fallback por scientific_name.

Después combina las fuentes con `pd.concat()` y une el resumen final con la
enciclopedia mediante `df.merge()`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import pandas as pd
import requests


GBIF_SPECIES_URL = "https://api.gbif.org/v1/species"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"

REQUEST_HEADERS = {
    "User-Agent": (
        "BiodiversityFinder/1.0 "
        "(educational project; contact: biodiversity-finder@example.com)"
    )
}

LANGUAGE_MAP_2_TO_3 = {
    "es": "spa",
    "en": "eng",
    "ru": "rus",
    "uk": "ukr",
    "pt": "por",
    "it": "ita",
    "ca": "cat",
    "fr": "fra",
    "de": "deu",
}

SUPPORTED_WIKIDATA_LANGUAGES = ["es", "en", "ru", "uk", "pt", "it", "ca", "fr", "de"]

SUPPORTED_LANGUAGE_PRIORITY = [
    "spa",
    "eng",
    "rus",
    "ukr",
    "por",
    "ita",
    "cat",
    "fra",
    "deu",
    "lat",
    "",
]


@dataclass(frozen=True)
class VernacularNameRecord:
    """Registro individual de nombre común."""

    scientific_name: str
    canonical_scientific_name: str
    species_key: str
    language: str
    vernacular_name: str
    source: str


def add_vernacular_names_to_encyclopedia(
    encyclopedia_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    max_species: int = 1200,
    use_api: bool = True,
    use_wikidata: bool = True,
    pause_seconds: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Añade nombres comunes a la enciclopedia usando fuentes externas."""
    if encyclopedia_df.empty:
        empty_names_df = build_empty_vernacular_names()
        return encyclopedia_df.copy(), empty_names_df

    working_encyclopedia_df = encyclopedia_df.copy()
    working_encyclopedia_df["canonical_scientific_name"] = (
        working_encyclopedia_df["scientific_name"].apply(canonicalize_scientific_name)
    )

    species_lookup_df = build_species_lookup(
        encyclopedia_df=working_encyclopedia_df,
        features_df=features_df,
    )

    limited_lookup_df = species_lookup_df.head(max_species).copy()

    gbif_names_df = build_gbif_vernacular_names_table(
        species_lookup_df=limited_lookup_df,
        use_api=use_api,
        pause_seconds=pause_seconds,
    )

    wikidata_names_df = build_wikidata_vernacular_names_table(
        species_lookup_df=limited_lookup_df,
        use_wikidata=use_wikidata,
        pause_seconds=pause_seconds,
    )

    fallback_names_df = build_fallback_names_table(limited_lookup_df)

    all_names_df = combine_vernacular_sources(
        gbif_names_df=gbif_names_df,
        wikidata_names_df=wikidata_names_df,
        fallback_names_df=fallback_names_df,
    )

    vernacular_summary_df = summarize_vernacular_names(all_names_df)

    enriched_df = working_encyclopedia_df.merge(
        vernacular_summary_df,
        on="canonical_scientific_name",
        how="left",
        validate="many_to_one",
    )

    enriched_df["vernacular_names"] = enriched_df["vernacular_names"].fillna("")
    enriched_df["vernacular_languages"] = enriched_df["vernacular_languages"].fillna("")
    enriched_df["vernacular_sources"] = enriched_df["vernacular_sources"].fillna("")

    enriched_df = enrich_search_document_with_vernacular_names(enriched_df)

    return enriched_df, all_names_df


def canonicalize_scientific_name(scientific_name: object) -> str:
    """Convierte nombres científicos con autoría en nombres canónicos simples."""
    text = str(scientific_name or "").strip()
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_species_lookup(
    encyclopedia_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Crea tabla species -> key para consultar fuentes externas."""
    key_candidates = [
        "speciesKey",
        "acceptedTaxonKey",
        "taxonKey",
        "key",
        "species_key",
        "taxon_key",
    ]

    available_keys = [column for column in key_candidates if column in features_df.columns]

    lookup_df = (
        encyclopedia_df[["scientific_name", "canonical_scientific_name"]]
        .drop_duplicates()
        .copy()
    )

    if available_keys:
        key_column = available_keys[0]
        feature_keys_df = features_df[["scientific_name", key_column]].dropna().copy()
        feature_keys_df["canonical_scientific_name"] = (
            feature_keys_df["scientific_name"].apply(canonicalize_scientific_name)
        )
        feature_keys_df = feature_keys_df.drop_duplicates(subset=["canonical_scientific_name"])

        lookup_df = lookup_df.merge(
            feature_keys_df[["canonical_scientific_name", key_column]],
            on="canonical_scientific_name",
            how="left",
            validate="many_to_one",
        )
        lookup_df = lookup_df.rename(columns={key_column: "species_key"})
    else:
        lookup_df["species_key"] = ""

    lookup_df["species_key"] = lookup_df["species_key"].fillna("").astype(str)
    lookup_df["species_key"] = lookup_df["species_key"].apply(clean_species_key)

    return lookup_df


def clean_species_key(value: object) -> str:
    """Limpia species_key para poder compararlo con Wikidata P846."""
    text = str(value or "").strip()

    if not text or text.lower() == "nan":
        return ""

    if re.fullmatch(r"\d+\.0", text):
        return text.split(".")[0]

    return text


def build_empty_vernacular_names() -> pd.DataFrame:
    """Devuelve tabla vacía de nombres comunes."""
    return pd.DataFrame(
        columns=[
            "scientific_name",
            "canonical_scientific_name",
            "species_key",
            "language",
            "vernacular_name",
            "source",
        ]
    )


def build_gbif_vernacular_names_table(
    species_lookup_df: pd.DataFrame,
    *,
    use_api: bool = True,
    pause_seconds: float = 0.05,
) -> pd.DataFrame:
    """Construye tabla larga de nombres comunes desde GBIF."""
    if species_lookup_df.empty or not use_api:
        return build_empty_vernacular_names()

    records: list[VernacularNameRecord] = []

    for _, row in species_lookup_df.iterrows():
        scientific_name = str(row.get("scientific_name", "")).strip()
        canonical_name = str(row.get("canonical_scientific_name", "")).strip()
        species_key = clean_species_key(row.get("species_key", ""))

        if not scientific_name or not species_key:
            continue

        records.extend(
            fetch_gbif_vernacular_names(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_name,
                species_key=species_key,
            )
        )

        sleep_if_needed(pause_seconds)

    return records_to_dataframe(records)


def fetch_gbif_vernacular_names(
    *,
    scientific_name: str,
    canonical_scientific_name: str,
    species_key: str,
    timeout: int = 20,
) -> list[VernacularNameRecord]:
    """Descarga nombres comunes desde GBIF Species API."""
    url = f"{GBIF_SPECIES_URL}/{species_key}/vernacularNames"

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    results = payload.get("results", [])
    records: list[VernacularNameRecord] = []

    for item in results:
        if not isinstance(item, dict):
            continue

        vernacular_name = clean_vernacular_name(item.get("vernacularName", ""))
        language = normalize_language_code(item.get("language", ""))

        if not is_good_vernacular_name(vernacular_name, canonical_scientific_name):
            continue

        records.append(
            VernacularNameRecord(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_scientific_name,
                species_key=species_key,
                language=language,
                vernacular_name=vernacular_name,
                source="GBIF Species API",
            )
        )

    return sort_vernacular_records(records)


def build_wikidata_vernacular_names_table(
    species_lookup_df: pd.DataFrame,
    *,
    use_wikidata: bool = True,
    pause_seconds: float = 0.05,
) -> pd.DataFrame:
    """Construye tabla larga de nombres comunes desde Wikidata."""
    if species_lookup_df.empty or not use_wikidata:
        return build_empty_vernacular_names()

    records: list[VernacularNameRecord] = []

    for _, row in species_lookup_df.iterrows():
        scientific_name = str(row.get("scientific_name", "")).strip()
        canonical_name = str(row.get("canonical_scientific_name", "")).strip()
        species_key = clean_species_key(row.get("species_key", ""))

        if not canonical_name:
            continue

        records.extend(
            fetch_wikidata_vernacular_names(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_name,
                species_key=species_key,
            )
        )

        sleep_if_needed(pause_seconds)

    return records_to_dataframe(records)


def fetch_wikidata_vernacular_names(
    *,
    scientific_name: str,
    canonical_scientific_name: str,
    species_key: str,
    timeout: int = 30,
) -> list[VernacularNameRecord]:
    """Descarga nombres comunes desde Wikidata por varios caminos."""
    records: list[VernacularNameRecord] = []

    if species_key:
        records.extend(
            fetch_wikidata_sparql_names(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_scientific_name,
                species_key=species_key,
                query=build_wikidata_query_by_gbif_key(species_key),
                timeout=timeout,
            )
        )

    if not records:
        records.extend(
            fetch_wikidata_sparql_names(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_scientific_name,
                species_key=species_key,
                query=build_wikidata_query_by_scientific_name(canonical_scientific_name),
                timeout=timeout,
            )
        )

    if not records:
        records.extend(
            fetch_wikidata_search_names(
                scientific_name=scientific_name,
                canonical_scientific_name=canonical_scientific_name,
                species_key=species_key,
                timeout=timeout,
            )
        )

    return sort_vernacular_records(records)


def fetch_wikidata_sparql_names(
    *,
    scientific_name: str,
    canonical_scientific_name: str,
    species_key: str,
    query: str,
    timeout: int = 30,
) -> list[VernacularNameRecord]:
    """Ejecuta una consulta SPARQL y parsea nombres."""
    try:
        response = requests.get(
            WIKIDATA_SPARQL_URL,
            params={"query": query, "format": "json"},
            headers=REQUEST_HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    bindings = payload.get("results", {}).get("bindings", [])

    return parse_wikidata_bindings(
        bindings=bindings,
        scientific_name=scientific_name,
        canonical_scientific_name=canonical_scientific_name,
        species_key=species_key,
    )


def parse_wikidata_bindings(
    *,
    bindings: list[dict],
    scientific_name: str,
    canonical_scientific_name: str,
    species_key: str,
) -> list[VernacularNameRecord]:
    """Parsea bindings de Wikidata SPARQL."""
    records: list[VernacularNameRecord] = []

    for binding in bindings:
        for variable_name, source_name in [
            ("commonName", "Wikidata P1843"),
            ("label", "Wikidata label"),
            ("alias", "Wikidata alias"),
        ]:
            value_data = binding.get(variable_name)

            if not isinstance(value_data, dict):
                continue

            vernacular_name = clean_vernacular_name(value_data.get("value", ""))
            language = normalize_language_code(value_data.get("xml:lang", ""))

            if not language:
                language = normalize_language_code(value_data.get("lang", ""))

            if not is_good_vernacular_name(vernacular_name, canonical_scientific_name):
                continue

            records.append(
                VernacularNameRecord(
                    scientific_name=scientific_name,
                    canonical_scientific_name=canonical_scientific_name,
                    species_key=species_key,
                    language=language,
                    vernacular_name=vernacular_name,
                    source=source_name,
                )
            )

    return records


def build_wikidata_query_by_gbif_key(species_key: str) -> str:
    """Construye SPARQL por GBIF taxon ID / P846."""
    escaped_key = clean_species_key(species_key).replace("\\", "\\\\").replace('"', '\\"')
    language_filter = build_language_filter()

    return f"""
    SELECT ?item ?commonName ?label ?alias WHERE {{
      ?item wdt:P846 "{escaped_key}".
      OPTIONAL {{
        ?item wdt:P1843 ?commonName.
        FILTER(LANG(?commonName) IN ({language_filter}))
      }}
      OPTIONAL {{
        ?item rdfs:label ?label.
        FILTER(LANG(?label) IN ({language_filter}))
      }}
      OPTIONAL {{
        ?item skos:altLabel ?alias.
        FILTER(LANG(?alias) IN ({language_filter}))
      }}
    }}
    LIMIT 200
    """


def build_wikidata_query_by_scientific_name(canonical_scientific_name: str) -> str:
    """Construye consulta SPARQL para Wikidata por nombre científico P225."""
    escaped_name = canonical_scientific_name.replace("\\", "\\\\").replace('"', '\\"')
    language_filter = build_language_filter()

    return f"""
    SELECT ?item ?commonName ?label ?alias WHERE {{
      ?item wdt:P225 "{escaped_name}".
      OPTIONAL {{
        ?item wdt:P1843 ?commonName.
        FILTER(LANG(?commonName) IN ({language_filter}))
      }}
      OPTIONAL {{
        ?item rdfs:label ?label.
        FILTER(LANG(?label) IN ({language_filter}))
      }}
      OPTIONAL {{
        ?item skos:altLabel ?alias.
        FILTER(LANG(?alias) IN ({language_filter}))
      }}
    }}
    LIMIT 200
    """


def build_wikidata_query(canonical_scientific_name: str) -> str:
    """Mantiene compatibilidad con tests anteriores."""
    return build_wikidata_query_by_scientific_name(canonical_scientific_name)


def build_language_filter() -> str:
    """Devuelve lista de idiomas para SPARQL."""
    return ", ".join(f'"{language}"' for language in SUPPORTED_WIKIDATA_LANGUAGES)


def fetch_wikidata_search_names(
    *,
    scientific_name: str,
    canonical_scientific_name: str,
    species_key: str,
    timeout: int = 30,
) -> list[VernacularNameRecord]:
    """Usa Wikidata Search API como fallback si SPARQL no devuelve nombres."""
    records: list[VernacularNameRecord] = []

    for language in SUPPORTED_WIKIDATA_LANGUAGES:
        params = {
            "action": "wbsearchentities",
            "search": canonical_scientific_name,
            "language": language,
            "uselang": language,
            "format": "json",
            "limit": 5,
        }

        try:
            response = requests.get(
                WIKIDATA_SEARCH_URL,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            continue

        for item in payload.get("search", []):
            label = clean_vernacular_name(item.get("label", ""))
            description = clean_vernacular_name(item.get("description", ""))

            if not is_probable_taxon_result(
                label=label,
                description=description,
                canonical_scientific_name=canonical_scientific_name,
            ):
                continue

            language_code = normalize_language_code(language)

            if is_good_vernacular_name(label, canonical_scientific_name):
                records.append(
                    VernacularNameRecord(
                        scientific_name=scientific_name,
                        canonical_scientific_name=canonical_scientific_name,
                        species_key=species_key,
                        language=language_code,
                        vernacular_name=label,
                        source="Wikidata search label",
                    )
                )

            aliases = item.get("aliases", [])

            if isinstance(aliases, list):
                for alias in aliases:
                    alias_name = clean_vernacular_name(alias)

                    if is_good_vernacular_name(alias_name, canonical_scientific_name):
                        records.append(
                            VernacularNameRecord(
                                scientific_name=scientific_name,
                                canonical_scientific_name=canonical_scientific_name,
                                species_key=species_key,
                                language=language_code,
                                vernacular_name=alias_name,
                                source="Wikidata search alias",
                            )
                        )

    return sort_vernacular_records(records)


def is_probable_taxon_result(
    *,
    label: str,
    description: str,
    canonical_scientific_name: str,
) -> bool:
    """Filtra resultados de búsqueda de Wikidata."""
    combined_text = f"{label} {description}".lower()
    canonical_lower = canonical_scientific_name.lower()

    taxon_markers = [
        "taxon",
        "species",
        "specie",
        "especie",
        "animal",
        "plant",
        "planta",
        "mammal",
        "ave",
        "bird",
        "insect",
        "genus",
        "género",
    ]

    if canonical_lower in combined_text:
        return True

    return any(marker in combined_text for marker in taxon_markers)


def build_fallback_names_table(species_lookup_df: pd.DataFrame) -> pd.DataFrame:
    """Crea fallback de nombre científico para todas las especies."""
    if species_lookup_df.empty:
        return build_empty_vernacular_names()

    records = [
        VernacularNameRecord(
            scientific_name=str(row.get("scientific_name", "")).strip(),
            canonical_scientific_name=str(row.get("canonical_scientific_name", "")).strip(),
            species_key=clean_species_key(row.get("species_key", "")),
            language="lat",
            vernacular_name=str(row.get("scientific_name", "")).strip(),
            source="scientific_name_fallback",
        )
        for _, row in species_lookup_df.iterrows()
        if str(row.get("scientific_name", "")).strip()
    ]

    return records_to_dataframe(records)


def combine_vernacular_sources(
    *,
    gbif_names_df: pd.DataFrame,
    wikidata_names_df: pd.DataFrame,
    fallback_names_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combina fuentes de nombres comunes con pd.concat()."""
    all_names_df = pd.concat(
        [
            gbif_names_df,
            wikidata_names_df,
            fallback_names_df,
        ],
        ignore_index=True,
    )

    if all_names_df.empty:
        return build_empty_vernacular_names()

    all_names_df["vernacular_name"] = all_names_df["vernacular_name"].astype(str).str.strip()
    all_names_df = all_names_df[all_names_df["vernacular_name"] != ""]
    all_names_df = all_names_df.drop_duplicates(
        subset=["canonical_scientific_name", "language", "vernacular_name", "source"]
    )

    return all_names_df.reset_index(drop=True)


def records_to_dataframe(records: list[VernacularNameRecord]) -> pd.DataFrame:
    """Convierte registros en DataFrame."""
    if not records:
        return build_empty_vernacular_names()

    names_df = pd.DataFrame([record.__dict__ for record in records])
    names_df["vernacular_name"] = names_df["vernacular_name"].astype(str).str.strip()
    names_df = names_df[names_df["vernacular_name"] != ""]
    names_df = names_df.drop_duplicates(
        subset=["canonical_scientific_name", "language", "vernacular_name", "source"]
    )

    return names_df.reset_index(drop=True)


def summarize_vernacular_names(vernacular_names_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa nombres comunes por especie canónica para poder hacer merge."""
    if vernacular_names_df.empty:
        return pd.DataFrame(
            columns=[
                "canonical_scientific_name",
                "vernacular_names",
                "vernacular_languages",
                "vernacular_sources",
            ]
        )

    sorted_df = vernacular_names_df.copy()
    sorted_df["language_priority"] = sorted_df["language"].map(
        {language: index for index, language in enumerate(SUPPORTED_LANGUAGE_PRIORITY)}
    ).fillna(999)
    sorted_df = sorted_df.sort_values(
        ["canonical_scientific_name", "language_priority", "vernacular_name"]
    )

    grouped_df = (
        sorted_df
        .groupby("canonical_scientific_name", as_index=False)
        .agg(
            vernacular_names=(
                "vernacular_name",
                lambda values: " | ".join(unique_preserve_order(values)),
            ),
            vernacular_languages=(
                "language",
                lambda values: " | ".join(unique_preserve_order(values)),
            ),
            vernacular_sources=(
                "source",
                lambda values: " | ".join(unique_preserve_order(values)),
            ),
        )
    )

    return grouped_df


def enrich_search_document_with_vernacular_names(
    encyclopedia_df: pd.DataFrame,
) -> pd.DataFrame:
    """Añade nombres comunes a search_document y profile_text."""
    enriched_df = encyclopedia_df.copy()

    if "search_document" not in enriched_df.columns:
        enriched_df["search_document"] = ""

    enriched_df["search_document"] = (
        enriched_df["search_document"].fillna("").astype(str)
        + " "
        + enriched_df["vernacular_names"].fillna("").astype(str)
    )

    if "profile_text" in enriched_df.columns:
        enriched_df["profile_text"] = enriched_df.apply(
            add_common_names_to_profile_text,
            axis=1,
        )

    return enriched_df


def add_common_names_to_profile_text(row: pd.Series) -> str:
    """Añade una frase corta con nombres comunes al perfil de especie."""
    profile_text = str(row.get("profile_text", "") or "")
    vernacular_names = str(row.get("vernacular_names", "") or "").strip()

    if not vernacular_names:
        return profile_text

    public_names = [
        name.strip()
        for name in vernacular_names.split("|")
        if name.strip()
        and str(row.get("scientific_name", "")).strip() != name.strip()
    ]

    if not public_names:
        return profile_text

    short_names = " / ".join(public_names[:8])

    if "Nombres comunes:" in profile_text:
        return profile_text

    return f"{profile_text} Nombres comunes: {short_names}."


def clean_vernacular_name(value: object) -> str:
    """Limpia un nombre común."""
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)

    return text


def is_good_vernacular_name(
    vernacular_name: str,
    canonical_scientific_name: str,
) -> bool:
    """Filtra valores vacíos o claramente inútiles."""
    if not vernacular_name:
        return False

    if len(vernacular_name) < 2:
        return False

    lower_name = vernacular_name.lower()
    lower_scientific_name = canonical_scientific_name.lower()

    bad_markers = [
        "http://",
        "https://",
        "wikidata",
        "undefined",
        "unknown",
    ]

    if any(marker in lower_name for marker in bad_markers):
        return False

    if lower_name == lower_scientific_name:
        return False

    return True


def normalize_language_code(language: object) -> str:
    """Convierte código de idioma a ISO 639-3 cuando sea posible."""
    code = str(language or "").strip().lower()

    if not code:
        return ""

    if code in LANGUAGE_MAP_2_TO_3:
        return LANGUAGE_MAP_2_TO_3[code]

    return code


def sort_vernacular_records(
    records: list[VernacularNameRecord],
) -> list[VernacularNameRecord]:
    """Ordena nombres comunes priorizando idiomas útiles para el proyecto."""
    priority = {
        language: index
        for index, language in enumerate(SUPPORTED_LANGUAGE_PRIORITY)
    }

    return sorted(
        records,
        key=lambda record: (
            priority.get(record.language, 999),
            record.vernacular_name.lower(),
        ),
    )


def unique_preserve_order(values: pd.Series) -> list[str]:
    """Devuelve valores únicos manteniendo orden."""
    result = []
    seen = set()

    for value in values:
        text = str(value).strip()

        if not text:
            continue

        key = text.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


def sleep_if_needed(seconds: float) -> None:
    """Pausa pequeña para respetar APIs externas."""
    if seconds > 0:
        time.sleep(seconds)
