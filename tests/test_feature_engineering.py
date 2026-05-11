"""Tests de creación de variables."""
import pandas as pd
from src.feature_engineering import create_features, get_season

def test_get_season_returns_expected_values() -> None:
    """Debe convertir meses a estaciones."""
    assert get_season(1) == 'invierno'
    assert get_season(4) == 'primavera'
    assert get_season(7) == 'verano'
    assert get_season(10) == 'otoño'

def test_create_features_adds_required_columns() -> None:
    """Debe añadir columnas nuevas para el modelo."""
    clean_df = pd.DataFrame([{'key':1,'scientific_name':'Test species','kingdom':'Animalia','phylum':'Chordata','taxon_class':'Aves','taxon_order':'Passeriformes','family':'Testidae','genus':'Test','species':'Test species','basis_of_record':'HUMAN_OBSERVATION','country_code':'ES','decimal_latitude':40.0,'decimal_longitude':-3.0,'year':2021,'month':4,'coordinate_uncertainty_meters':50}])
    result_df = create_features(clean_df)
    assert 'season' in result_df.columns
    assert 'scientific_text' in result_df.columns
    assert 'taxon_label' in result_df.columns
    assert result_df.iloc[0]['season'] == 'primavera'
