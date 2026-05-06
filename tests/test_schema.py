"""Tests for the ClinicalSummary output schema."""

import pytest
from pydantic import ValidationError

from fhir_summarizer.schema import (
    ActiveCondition,
    Allergy,
    ClinicalSummary,
    CurrentMedication,
    LabTrend,
    PatientDemographics,
    RecentEncounter,
)


def test_minimal_summary_valid() -> None:
    """A summary with only required fields should validate."""
    summary = ClinicalSummary(
        patient_demographics=PatientDemographics(),
        summary_narrative="Patient is a 45-year-old with no significant active conditions.",
    )
    assert summary.active_conditions == []
    assert summary.recent_lab_trends == []


def test_demographics_accepts_known_sex_values() -> None:
    PatientDemographics(sex="male")
    PatientDemographics(sex="female")
    PatientDemographics(sex="other")
    PatientDemographics(sex="unknown")
    PatientDemographics(sex=None)


def test_demographics_rejects_unknown_sex() -> None:
    with pytest.raises(ValidationError):
        PatientDemographics(sex="something-weird")


def test_lab_trend_enforces_known_trend_values() -> None:
    valid = LabTrend(
        lab_name="Hemoglobin A1c",
        latest_value="7.1 %",
        trend="worsening",
        source_resource_ids=["obs-1", "obs-2", "obs-3"],
    )
    assert valid.trend == "worsening"

    with pytest.raises(ValidationError):
        LabTrend(
            lab_name="Hemoglobin A1c",
            latest_value="7.1 %",
            trend="getting-worse",  # not a literal value
            source_resource_ids=["obs-1"],
        )


def test_lab_trend_insufficient_data_is_valid() -> None:
    """Critical: insufficient_data is a first-class trend value."""
    trend = LabTrend(
        lab_name="LDL Cholesterol",
        latest_value="142 mg/dL",
        trend="insufficient_data",
        source_resource_ids=["obs-99"],
    )
    assert trend.trend == "insufficient_data"


def test_active_condition_requires_source_id() -> None:
    """Citation enforcement: source_resource_id is required."""
    with pytest.raises(ValidationError):
        ActiveCondition(condition_text="Hypertension")


def test_summary_narrative_length_bounds() -> None:
    with pytest.raises(ValidationError):
        ClinicalSummary(
            patient_demographics=PatientDemographics(),
            summary_narrative="too short",
        )


def test_full_summary_validates_end_to_end() -> None:
    """A realistic summary with all sections populated should validate."""
    summary = ClinicalSummary(
        patient_demographics=PatientDemographics(age_years=58, sex="female"),
        active_conditions=[
            ActiveCondition(
                condition_text="Type 2 diabetes mellitus",
                snomed_code="44054006",
                onset_date="2019-03-15",
                source_resource_id="cond-001",
            )
        ],
        current_medications=[
            CurrentMedication(
                medication_text="Metformin 500 mg oral tablet",
                dose="500 mg twice daily",
                start_date="2019-04-02",
                source_resource_id="med-001",
            )
        ],
        recent_lab_trends=[
            LabTrend(
                lab_name="Hemoglobin A1c",
                loinc_code="4548-4",
                latest_value="7.4 %",
                latest_date="2025-09-12",
                trend="stable",
                source_resource_ids=["obs-101", "obs-102", "obs-103"],
            )
        ],
        allergies=[
            Allergy(
                substance="Penicillin",
                reaction="Hives",
                severity="moderate",
                source_resource_id="allergy-001",
            )
        ],
        recent_encounters=[
            RecentEncounter(
                encounter_type="Outpatient",
                date="2025-09-12",
                reason="Diabetes follow-up",
                source_resource_id="enc-001",
            )
        ],
        summary_narrative=(
            "58-year-old female with type 2 diabetes on metformin. "
            "A1c trending stable at 7.4%. Penicillin allergy on file."
        ),
        data_quality_flags=[],
    )
    assert len(summary.active_conditions) == 1
    assert summary.recent_lab_trends[0].trend == "stable"
