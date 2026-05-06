"""Tests for the citation validator.

Covers:
- Happy path: validator passes when all citations match
- Orphan detection in each section (conditions, meds, trends, allergies, encounters)
- Counting and message formatting
- Multiple-IDs-per-entry case for lab trends
"""

from pathlib import Path

import pytest

from fhir_summarizer.loader import PatientRecord, load_bundle
from fhir_summarizer.schema import (
    ActiveCondition,
    Allergy,
    ClinicalSummary,
    CurrentMedication,
    LabTrend,
    PatientDemographics,
    RecentEncounter,
)
from fhir_summarizer.validator import (
    CitationIssue,
    ValidationResult,
    validate_citations,
)

FIXTURES = Path(__file__).parent.parent / "data" / "fixtures"


def fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("patient_*.json"))


# --- Helpers -------------------------------------------------------------


def _empty_summary() -> ClinicalSummary:
    return ClinicalSummary(
        patient_demographics=PatientDemographics(),
        summary_narrative="Nothing notable on this patient at this time.",
    )


def _real_record() -> PatientRecord:
    """Use the first fixture as a realistic source record for tests."""
    files = fixture_files()
    if not files:
        pytest.skip("No fixtures available")
    return load_bundle(files[0])


# --- Happy path ----------------------------------------------------------


def test_empty_summary_is_valid() -> None:
    """A summary with no citations passes trivially."""
    record = _real_record()
    result = validate_citations(_empty_summary(), record)
    assert result.is_valid is True
    assert result.issue_count == 0
    assert result.total_citations_checked == 0


def test_summary_with_real_ids_is_valid() -> None:
    """A summary that cites real IDs from the record should validate."""
    record = _real_record()

    if not record.conditions or not record.medications:
        pytest.skip("Fixture lacks conditions or medications")

    real_cond_id = record.conditions[0].id
    real_med_id = record.medications[0].id

    summary = ClinicalSummary(
        patient_demographics=PatientDemographics(),
        active_conditions=[
            ActiveCondition(
                condition_text="Test condition",
                source_resource_id=real_cond_id,
            )
        ],
        current_medications=[
            CurrentMedication(
                medication_text="Test medication",
                source_resource_id=real_med_id,
            )
        ],
        summary_narrative="Patient has a condition and is on a medication.",
    )

    result = validate_citations(summary, record)
    assert result.is_valid is True
    assert result.issue_count == 0
    assert result.total_citations_checked == 2


# --- Orphan detection ----------------------------------------------------


def test_orphan_condition_id_caught() -> None:
    record = _real_record()
    summary = ClinicalSummary(
        patient_demographics=PatientDemographics(),
        active_conditions=[
            ActiveCondition(
                condition_text="Made-up condition",
                source_resource_id="this-id-does-not-exist",
            )
        ],
        summary_narrative="Patient supposedly has something, citation is fake.",
    )
    result = validate_citations(summary, record)
    assert result.is_valid is False
    assert result.issue_count == 1
    assert result.issues[0].cited_id == "this-id-does-not-exist"
    assert result.issues[0].expected_resource_type == "Condition"
    assert "active_conditions[0]" in result.issues[0].section


def test_orphan_medication_id_caught() -> None:
    record = _real_record()
    summary = ClinicalSummary(
        patient_demographics=PatientDemographics(),
        current_medications=[
            CurrentMedication(
                medication_text="Made-up medication",
                source_resource_id="fake-med-id",
            )
        ],
        summary_narrative="Patient is allegedly on a medication, citation is fake.",
    )
    result = validate_citations(summary, record)
    assert result.is_valid is False
    assert result.issues[0].expected_resource_type == "MedicationRequest"


def test_orphan_observation_id_in_lab_trend_caught() -> None:
    """Lab trends use source_resource_ids (plural). Each ID is checked."""
    record = _real_record()
    if not record.observations:
        pytest.skip("Fixture lacks observations")

    real_obs_id = record.observations[0].id
    summary = ClinicalSummary(
        patient_demographics=PatientDemographics(),
        recent_lab_trends=[
            LabTrend(
                lab_name="Test Lab",
                latest_value="42",
                trend="stable",
                source_resource_ids=[real_obs_id, "fake-obs-id-1", "fake-obs-id-2"],
            )
        ],
        summary_narrative="Lab trend with one real and two fake supporting observations.",
    )
    result = validate_citations(summary, record)
    assert result.is_valid is False
    assert result.issue_count == 2
    assert result.total_citations_checked == 3


def test_orphan_allergy_id_caught() -> None:
    record = _real_record()
    summary = ClinicalSummary(
        patient_demographics=PatientDemographics(),
        allergies=[
            Allergy(substance="Made-up", source_resource_id="fake-allergy-id"),
        ],
        summary_narrative="Patient supposedly has an allergy, citation is fake.",
    )
    result = validate_citations(summary, record)
    assert result.is_valid is False
    assert result.issues[0].expected_resource_type == "AllergyIntolerance"


def test_orphan_encounter_id_caught() -> None:
    record = _real_record()
    summary = ClinicalSummary(
        patient_demographics=PatientDemographics(),
        recent_encounters=[
            RecentEncounter(
                encounter_type="Outpatient",
                source_resource_id="fake-encounter-id",
            )
        ],
        summary_narrative="Patient supposedly had an encounter, citation is fake.",
    )
    result = validate_citations(summary, record)
    assert result.is_valid is False
    assert result.issues[0].expected_resource_type == "Encounter"


# --- Mixed valid + invalid -----------------------------------------------


def test_partial_valid_partial_orphan() -> None:
    """A summary mixing real and fake citations should report only the fakes."""
    record = _real_record()
    if not record.conditions:
        pytest.skip("Fixture lacks conditions")

    real_cond_id = record.conditions[0].id

    summary = ClinicalSummary(
        patient_demographics=PatientDemographics(),
        active_conditions=[
            ActiveCondition(
                condition_text="Real",
                source_resource_id=real_cond_id,
            ),
            ActiveCondition(
                condition_text="Fake",
                source_resource_id="not-a-real-id",
            ),
        ],
        summary_narrative="Patient has one real and one fake condition cited.",
    )
    result = validate_citations(summary, record)
    assert result.is_valid is False
    assert result.issue_count == 1
    assert result.total_citations_checked == 2
    # The orphan should be the second one
    assert result.issues[0].section == "active_conditions[1]"


# --- Message formatting --------------------------------------------------


def test_summary_message_for_valid_result() -> None:
    record = _real_record()
    result = validate_citations(_empty_summary(), record)
    msg = result.summary_message()
    assert "validated" in msg.lower()
    assert "0" in msg


def test_summary_message_for_invalid_result() -> None:
    record = _real_record()
    summary = ClinicalSummary(
        patient_demographics=PatientDemographics(),
        active_conditions=[
            ActiveCondition(
                condition_text="Fake",
                source_resource_id="orphan-id",
            )
        ],
        summary_narrative="Has a fake citation that should fail validation.",
    )
    result = validate_citations(summary, record)
    msg = result.summary_message()
    assert "1 orphan" in msg.lower() or "1 orphan" in msg
    assert "orphan-id" in msg


# --- Smoke: validator runs cleanly across all fixtures with empty summaries


@pytest.mark.parametrize("fixture", fixture_files())
def test_validator_runs_on_all_fixtures(fixture: Path) -> None:
    record = load_bundle(fixture)
    result = validate_citations(_empty_summary(), record)
    assert result.is_valid is True
