"""Tests de entrenamiento de modelo."""
from pathlib import Path
import pandas as pd
from src.model_training import train_model

def test_train_model_creates_files(tmp_path: Path) -> None:
    """El entrenamiento debe crear modelo y métricas."""
    rows = []
    for index in range(30):
        rows.append({'decimal_latitude':40+index*0.01,'decimal_longitude':-3,'year':2020,'month':5,'observation_decade':2020,'kingdom':'Animalia','phylum':'Chordata','family':'Birdidae','country_code':'ES','basis_of_record':'HUMAN_OBSERVATION','season':'primavera','coordinate_precision':'alta','scientific_text':'bird aves chordata feathers','taxon_label':'Aves'})
    for index in range(30):
        rows.append({'decimal_latitude':41+index*0.01,'decimal_longitude':-4,'year':2021,'month':8,'observation_decade':2020,'kingdom':'Animalia','phylum':'Chordata','family':'Mammalidae','country_code':'ES','basis_of_record':'HUMAN_OBSERVATION','season':'verano','coordinate_precision':'media','scientific_text':'mammal mammalia fur','taxon_label':'Mammalia'})
    features_df = pd.DataFrame(rows)
    model_path = tmp_path / 'model.joblib'; metrics_path = tmp_path / 'metrics.json'; report_path = tmp_path / 'report.csv'
    metrics = train_model(features_df=features_df, model_path=model_path, metrics_path=metrics_path, report_path=report_path)
    assert model_path.exists()
    assert metrics_path.exists()
    assert report_path.exists()
    assert metrics['accuracy'] >= 0
