"""Construcción de una enciclopedia desde datos reales."""
from __future__ import annotations
import pandas as pd

def build_species_encyclopedia(features_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa ocurrencias por especie para crear perfiles."""
    grouped_df = (features_df.groupby(['scientific_name','kingdom','phylum','taxon_class','family','genus','species'], dropna=False)
        .agg(observations=('key','count'), countries=('country_code', lambda values: ', '.join(sorted(set(values.astype(str)))[:5])), first_year=('year','min'), last_year=('year','max'), avg_latitude=('decimal_latitude','mean'), avg_longitude=('decimal_longitude','mean'), most_common_basis=('basis_of_record', most_common_value), most_common_season=('season', most_common_value))
        .reset_index().sort_values('observations', ascending=False))
    grouped_df['profile_text'] = grouped_df.apply(build_profile_text, axis=1)
    grouped_df['search_document'] = grouped_df.apply(build_search_document, axis=1)
    return grouped_df

def most_common_value(values: pd.Series) -> str:
    """Devuelve el valor más frecuente de una serie."""
    if values.empty: return 'desconocido'
    mode_values = values.mode(dropna=True)
    if mode_values.empty: return 'desconocido'
    return str(mode_values.iloc[0])

def build_profile_text(row: pd.Series) -> str:
    """Construye una descripción breve de la especie."""
    return f"{row['scientific_name']} pertenece a la clase {row['taxon_class']} y a la familia {row['family']}. En el dataset tiene {row['observations']} observaciones entre {row['first_year']} y {row['last_year']}."

def build_search_document(row: pd.Series) -> str:
    """Construye un documento de búsqueda para la app online."""
    return ' '.join([str(row.get('scientific_name','')), str(row.get('kingdom','')), str(row.get('phylum','')), str(row.get('taxon_class','')), str(row.get('family','')), str(row.get('genus','')), str(row.get('species','')), str(row.get('profile_text',''))])
