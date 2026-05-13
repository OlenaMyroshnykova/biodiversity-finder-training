"""Tests de limpieza de datos."""
import pandas as pd
from src.data_cleaning import clean_occurrences

def test_clean_occurrences_removes_invalid_coordinates() -> None:
    """La limpieza debe eliminar coordenadas fuera de rango."""
    raw_df = pd.DataFrame([
        {'key':1,'scientificName':'Species validus','kingdom':'Animalia','phylum':'Chordata','class':'Aves','family':'Testidae','countryCode':'ES','decimalLatitude':40.0,'decimalLongitude':-3.0,'year':2020,'month':5,'basisOfRecord':'HUMAN_OBSERVATION'},
        {'key':2,'scientificName':'Species invalidus','kingdom':'Animalia','phylum':'Chordata','class':'Aves','family':'Testidae','countryCode':'ES','decimalLatitude':200.0,'decimalLongitude':-3.0,'year':2020,'month':5,'basisOfRecord':'HUMAN_OBSERVATION'},
    ])
    result_df = clean_occurrences(raw_df, min_class_records=1)
    assert len(result_df) == 1
    assert result_df.iloc[0]['scientific_name'] == 'Species validus'


def test_remove_rare_classes_protects_anchor_queries() -> None:
    """Los registros de consultas ancla no deben eliminarse aunque tengan pocos registros."""
    raw_df = pd.DataFrame([
        # Reptil de consulta ancla — solo 1 registro, normalmente sería eliminado
        {
            "key": 10, "scientificName": "Crocodylus niloticus", "kingdom": "Animalia",
            "phylum": "Chordata", "class": "Reptilia", "family": "Crocodylidae",
            "countryCode": "KE", "decimalLatitude": -1.0, "decimalLongitude": 36.0,
            "year": 2020, "month": 5, "basisOfRecord": "HUMAN_OBSERVATION",
            "source_query": "reptiles_crocodylia",
        },
        # Especie común — 2 registros, supera min_class_records=1
        {
            "key": 11, "scientificName": "Aves commonus", "kingdom": "Animalia",
            "phylum": "Chordata", "class": "Aves", "family": "Testidae",
            "countryCode": "ES", "decimalLatitude": 40.0, "decimalLongitude": -3.0,
            "year": 2020, "month": 5, "basisOfRecord": "HUMAN_OBSERVATION",
            "source_query": "general_global",
        },
        {
            "key": 12, "scientificName": "Aves commonus", "kingdom": "Animalia",
            "phylum": "Chordata", "class": "Aves", "family": "Testidae",
            "countryCode": "ES", "decimalLatitude": 41.0, "decimalLongitude": -3.0,
            "year": 2021, "month": 6, "basisOfRecord": "HUMAN_OBSERVATION",
            "source_query": "general_global",
        },
    ])
    result_df = clean_occurrences(raw_df, min_class_records=2)
    # Reptilia solo tiene 1 registro pero es ancla — debe sobrevivir
    assert "Crocodylus niloticus" in result_df["scientific_name"].values
    # Aves tiene 2 registros — también sobrevive
    assert "Aves commonus" in result_df["scientific_name"].values
