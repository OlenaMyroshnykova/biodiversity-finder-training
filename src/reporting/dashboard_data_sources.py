"""Secciones informativas sobre fuentes de datos para el dashboard técnico."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def build_data_sources_table() -> pd.DataFrame:
    """Devuelve una tabla estable con las fuentes usadas por el proyecto."""
    return pd.DataFrame(
        [
            {
                "Fuente": "GBIF Occurrence API",
                "Tipo": "Observaciones de biodiversidad",
                "Qué aporta": "Registros de presencia: especie, coordenadas, fecha, país, taxonomía y basis_of_record.",
                "Uso en el proyecto": "Base principal para entrenamiento, mapas Folium y conteo de observaciones.",
                "Archivo / salida": "data/raw/*.csv, data/processed/species_encyclopedia.parquet",
                "Riesgo / limitación": "Puede contener ruido, duplicados, sesgo geográfico y clases desbalanceadas.",
            },
            {
                "Fuente": "GBIF Species API",
                "Tipo": "Catálogo taxonómico y nombres comunes",
                "Qué aporta": "Nombres comunes, nombres canónicos y taxonomía complementaria.",
                "Uso en el proyecto": "Mejora búsquedas por lenguaje natural y nombres no científicos.",
                "Archivo / salida": "data/interim/vernacular_names.csv",
                "Riesgo / limitación": "No todas las especies tienen nombres comunes en español o inglés.",
            },
            {
                "Fuente": "Wikidata / Wikipedia / Wikimedia Commons",
                "Tipo": "Conocimiento abierto e imágenes",
                "Qué aporta": "Imágenes, etiquetas multilingües y enlaces de referencia.",
                "Uso en el proyecto": "Enriquecimiento visual de la enciclopedia y apoyo a search_document.",
                "Archivo / salida": "data/interim/image_enrichment.csv, image_url en encyclopedia",
                "Riesgo / limitación": "Cobertura irregular; algunas imágenes pueden no existir o no estar bien enlazadas.",
            },
            {
                "Fuente": "NASA POWER API",
                "Tipo": "Variables climáticas",
                "Qué aporta": "Temperatura, precipitación y otras variables aproximadas por coordenadas.",
                "Uso en el proyecto": "Capa climática del hábitat y demostración de pd.merge().",
                "Archivo / salida": "data/interim/climate_features.csv",
                "Riesgo / limitación": "Es una aproximación espacial y temporal, no una medición exacta del microhábitat.",
            },
            {
                "Fuente": "IUCN Red List API v4",
                "Tipo": "Estado de conservación",
                "Qué aporta": "Categorías oficiales: LC, NT, VU, EN, CR, EW, EX, DD.",
                "Uso en el proyecto": "Bordes/avisos visuales para especies amenazadas y conciencia ética.",
                "Archivo / salida": "data/interim/conservation_status.csv, iucn_category en encyclopedia",
                "Riesgo / limitación": "Requiere token; puede haber especies sin datos o límites de tiempo/rate limit.",
            },
            {
                "Fuente": "Pipeline propio de training",
                "Tipo": "Feature engineering y contrato de artefactos",
                "Qué aporta": "tags_de_busqueda, search_document, size_tag, color_tag, habitat_tag y exports light/offline.",
                "Uso en el proyecto": "Convierte datos brutos en una enciclopedia buscable y ligera para la app.",
                "Archivo / salida": "data/processed/species_encyclopedia*.parquet",
                "Riesgo / limitación": "Las etiquetas de color/tamaño/hábitat son inferencias educativas, no verdad biológica absoluta.",
            },
            {
                "Fuente": "Hugging Face Datasets",
                "Tipo": "Publicación de artefactos",
                "Qué aporta": "Repositorio central para parquet, CSV, métricas y reports.",
                "Uso en el proyecto": "La app y este dashboard leen artefactos publicados tras GitHub Actions.",
                "Archivo / salida": "selenamir/biodiversity-finder-artifacts",
                "Riesgo / limitación": "Si el último pipeline falla, puede seguir mostrando artefactos anteriores.",
            },
        ]
    )


def render_data_sources_section() -> None:
    """Renderiza una explicación clara de las fuentes usadas por el proyecto."""
    st.header("Fuentes de datos del proyecto")
    st.write(
        "El dashboard no solo enseña métricas del modelo. También documenta de dónde "
        "sale la información que alimenta la enciclopedia: observaciones, taxonomía, "
        "clima, nombres comunes, imágenes y estado de conservación."
    )

    sources_df = build_data_sources_table()
    st.dataframe(sources_df, width="stretch", hide_index=True)

    with st.expander("Cómo se unen las fuentes en el pipeline", expanded=False):
        st.markdown(
            """
            1. **GBIF Occurrence API** descarga registros por grupos taxonómicos.
            2. El pipeline limpia columnas, elimina ruido y crea features básicas.
            3. Varias descargas se combinan con **`pd.concat()`** para construir un dataset único.
            4. Las capas externas —clima, nombres comunes, imágenes e IUCN— se añaden con **`pd.merge()`**.
            5. `tags_de_busqueda` se mantiene como una columna ligera para filtros rápidos.
            6. `search_document` reúne nombres científicos, nombres comunes y taxonomía para búsqueda semántica simple.
            7. Los artefactos finales se publican en **Hugging Face Datasets** para que la app y el dashboard usen la misma fuente.
            """
        )

    with st.expander("Limitaciones éticas y de calidad de los datos", expanded=False):
        st.markdown(
            """
            - **Sesgo geográfico:** hay más observaciones en zonas con más usuarios de ciencia ciudadana.
            - **Sesgo taxonómico:** aves y mamíferos suelen estar mejor documentados que insectos o anfibios.
            - **Datos incompletos:** no todas las especies tienen imagen, nombre común o categoría IUCN.
            - **Inferencias educativas:** color, tamaño y hábitat ayudan a buscar, pero no sustituyen una fuente científica oficial.
            - **Uso responsable:** la app debe explicar incertidumbre, especialmente en especies amenazadas o invasoras.
            """
        )
