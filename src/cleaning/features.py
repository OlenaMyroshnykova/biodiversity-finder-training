"""Creación de variables para análisis y entrenamiento."""
from __future__ import annotations
import pandas as pd

def create_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables adicionales para análisis y modelo."""
    features_df = clean_df.copy()
    features_df['observation_decade'] = (features_df['year'] // 10) * 10
    features_df['season'] = features_df['month'].apply(get_season)
    features_df['coordinate_precision'] = features_df.apply(calculate_coordinate_precision, axis=1)
    features_df['scientific_text'] = features_df.apply(build_scientific_text, axis=1)
    features_df['taxon_label'] = features_df['taxon_class'].astype(str)
    return features_df

def get_season(month: int) -> str:
    """Convierte un mes en estación aproximada."""
    if month in [12,1,2]: return 'invierno'
    if month in [3,4,5]: return 'primavera'
    if month in [6,7,8]: return 'verano'
    return 'otoño'

def calculate_coordinate_precision(row: pd.Series) -> str:
    """Clasifica la precisión de coordenadas."""
    uncertainty = row.get('coordinate_uncertainty_meters')
    if pd.isna(uncertainty): return 'desconocida'
    try: uncertainty_value = float(uncertainty)
    except ValueError: return 'desconocida'
    if uncertainty_value <= 100: return 'alta'
    if uncertainty_value <= 1000: return 'media'
    return 'baja'

def build_scientific_text(row: pd.Series) -> str:
    """Construye texto científico para el vectorizador TF-IDF.

    IMPORTANTE: taxon_class está excluido deliberadamente.
    taxon_label = taxon_class, por lo que incluirlo causaría data leakage:
    el modelo simplemente aprendería a buscar 'Mammalia' en el texto
    para predecir Mammalia, en lugar de aprender patrones reales.

    El modelo aprende la clase taxonómica a partir de:
    - Nombre científico y género (contienen señales filogenéticas)
    - Reino y filo (jerarquía taxonómica legítima)
    - Orden y familia (correlacionan con la clase pero no la repiten literalmente)
    - Especie y basis_of_record (contexto del registro)
    """
    text_parts = [
        row.get('scientific_name', ''),
        row.get('kingdom', ''),
        row.get('phylum', ''),
        # taxon_class EXCLUIDO — es el target (taxon_label)
        row.get('taxon_order', ''),
        row.get('family', ''),
        row.get('genus', ''),
        row.get('species', ''),
        row.get('basis_of_record', ''),
    ]
    return ' '.join(str(part) for part in text_parts if part and not pd.isna(part))
