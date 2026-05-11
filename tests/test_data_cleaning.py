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
