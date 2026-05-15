"""Reportes EDA rápidos para documentar hallazgos."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def generate_eda_reports(
    encyclopedia_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Genera tablas EDA rápidas y hallazgos en JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: dict[str, Path] = {}

    class_distribution_path = output_dir / "eda_class_distribution.csv"
    build_class_distribution(encyclopedia_df).to_csv(class_distribution_path, index=False)
    generated_files["eda_class_distribution"] = class_distribution_path

    conservation_summary_path = output_dir / "eda_conservation_summary.csv"
    build_conservation_summary(encyclopedia_df).to_csv(conservation_summary_path, index=False)
    generated_files["eda_conservation_summary"] = conservation_summary_path

    habitat_summary_path = output_dir / "eda_habitat_summary.csv"
    build_habitat_summary(encyclopedia_df).to_csv(habitat_summary_path, index=False)
    generated_files["eda_habitat_summary"] = habitat_summary_path

    findings_path = output_dir / "eda_findings.json"
    findings = build_eda_findings(encyclopedia_df, features_df)
    findings_path.write_text(
        json.dumps(findings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    generated_files["eda_findings"] = findings_path

    return generated_files


def build_class_distribution(encyclopedia_df: pd.DataFrame) -> pd.DataFrame:
    """Distribución de especies y observaciones por clase."""
    if encyclopedia_df.empty or "taxon_class" not in encyclopedia_df.columns:
        return pd.DataFrame(columns=["taxon_class", "species_count", "observations"])

    return (
        encyclopedia_df
        .groupby("taxon_class", as_index=False)
        .agg(
            species_count=("scientific_name", "nunique"),
            observations=("observations", "sum"),
        )
        .sort_values("observations", ascending=False)
    )


def build_conservation_summary(encyclopedia_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de conservación."""
    if encyclopedia_df.empty or "conservation_status" not in encyclopedia_df.columns:
        return pd.DataFrame(columns=["conservation_status", "species_count"])

    return (
        encyclopedia_df
        .groupby("conservation_status", as_index=False)
        .agg(species_count=("scientific_name", "nunique"))
        .sort_values("species_count", ascending=False)
    )


def build_habitat_summary(encyclopedia_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por habitat_tag."""
    if encyclopedia_df.empty or "habitat_tag" not in encyclopedia_df.columns:
        return pd.DataFrame(columns=["habitat_tag", "species_count"])

    return (
        encyclopedia_df
        .groupby("habitat_tag", as_index=False)
        .agg(species_count=("scientific_name", "nunique"))
        .sort_values("species_count", ascending=False)
    )


def build_eda_findings(
    encyclopedia_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> dict:
    """Genera hallazgos automáticos para documentación."""
    findings = {
        "total_species": int(len(encyclopedia_df)),
        "total_occurrences": int(len(features_df)),
        "ethical_impact": (
            "El proyecto ayuda a visualizar biodiversidad, conservación y distribución "
            "de especies, pero no debe utilizarse como sustituto de evaluaciones "
            "científicas oficiales."
        ),
        "limitations": [
            "Los datos dependen de observaciones disponibles en fuentes abiertas.",
            "Algunas especies pueden estar sobrerrepresentadas o infrarrepresentadas.",
            "Los tags de color, tamaño y hábitat son inferencias educativas.",
            "El estado de conservación debe validarse con fuentes oficiales como IUCN.",
        ],
    }

    if "taxon_class" in encyclopedia_df.columns and not encyclopedia_df.empty:
        top_class = (
            encyclopedia_df
            .groupby("taxon_class")["observations"]
            .sum()
            .sort_values(ascending=False)
            .head(1)
        )

        if not top_class.empty:
            findings["top_observed_class"] = {
                "taxon_class": str(top_class.index[0]),
                "observations": int(top_class.iloc[0]),
            }

    if "is_threatened" in encyclopedia_df.columns:
        findings["threatened_species_count"] = int(
            encyclopedia_df["is_threatened"].fillna(False).sum()
        )

    return findings
