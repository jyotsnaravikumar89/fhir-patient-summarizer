"""Tests for the agent tools.

These verify that each tool returns LLM-friendly dicts with the expected
keys, that source_resource_id is always present, and that filters work.
Tests run across all curated fixtures.
"""

from pathlib import Path

import pytest

from fhir_summarizer.loader import load_bundle
from fhir_summarizer.tools import (
    get_allergies,
    get_conditions,
    get_encounters,
    get_medications,
    get_observations,
    get_patient_demographics,
)

FIXTURES = Path(__file__).parent.parent / "data" / "fixtures"


def fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("patient_*.json"))


# --- Demographics --------------------------------------------------------


@pytest.mark.parametrize("fixture", fixture_files())
def test_demographics_has_required_keys(fixture: Path) -> None:
    record = load_bundle(fixture)
    demo = get_patient_demographics(record)
    assert set(demo.keys()) >= {
        "age_years",
        "sex",
        "primary_language",
        "source_resource_id",
    }
    assert demo["source_resource_id"] is not None


@pytest.mark.parametrize("fixture", fixture_files())
def test_demographics_sex_is_known_value(fixture: Path) -> None:
    record = load_bundle(fixture)
    demo = get_patient_demographics(record)
    if demo["sex"] is not None:
        assert demo["sex"] in {"male", "female", "other", "unknown"}


# --- Conditions ----------------------------------------------------------


@pytest.mark.parametrize("fixture", fixture_files())
def test_conditions_has_clean_keys(fixture: Path) -> None:
    record = load_bundle(fixture)
    conds = get_conditions(record)
    assert isinstance(conds, list)
    for cond in conds:
        assert set(cond.keys()) >= {
            "condition_text",
            "snomed_code",
            "onset_date",
            "clinical_status",
            "source_resource_id",
        }
        assert cond["source_resource_id"] is not None
        assert cond["condition_text"]  # non-empty


@pytest.mark.parametrize("fixture", fixture_files())
def test_conditions_status_filter(fixture: Path) -> None:
    record = load_bundle(fixture)
    all_conds = get_conditions(record)
    if not all_conds:
        pytest.skip("Patient has no conditions")
    statuses = {c["clinical_status"] for c in all_conds if c["clinical_status"]}
    if not statuses:
        pytest.skip("No clinical statuses present")
    pick = next(iter(statuses))
    filtered = get_conditions(record, status_filter=pick)
    assert all(c["clinical_status"] == pick for c in filtered)


# --- Medications --------------------------------------------------------


@pytest.mark.parametrize("fixture", fixture_files())
def test_medications_has_clean_keys(fixture: Path) -> None:
    record = load_bundle(fixture)
    meds = get_medications(record)
    assert isinstance(meds, list)
    for med in meds:
        assert set(med.keys()) >= {
            "medication_text",
            "rxnorm_code",
            "dose",
            "status",
            "start_date",
            "source_resource_id",
        }
        assert med["source_resource_id"] is not None


# --- Observations -------------------------------------------------------


@pytest.mark.parametrize("fixture", fixture_files())
def test_observations_has_clean_keys(fixture: Path) -> None:
    record = load_bundle(fixture)
    obs = get_observations(record)
    assert isinstance(obs, list)
    if not obs:
        pytest.skip("Patient has no observations")
    for o in obs[:5]:  # sample to avoid spending forever on rich fixtures
        assert set(o.keys()) >= {
            "name",
            "loinc_code",
            "value",
            "unit",
            "effective_date",
            "source_resource_id",
        }
        assert o["source_resource_id"] is not None


@pytest.mark.parametrize("fixture", fixture_files())
def test_observations_category_filter(fixture: Path) -> None:
    record = load_bundle(fixture)
    labs = get_observations(record, category="laboratory")
    vitals = get_observations(record, category="vital-signs")
    # We can't assert exact counts but sanity: filtering should not increase count.
    all_obs = get_observations(record)
    assert len(labs) <= len(all_obs)
    assert len(vitals) <= len(all_obs)


# --- Encounters ---------------------------------------------------------


@pytest.mark.parametrize("fixture", fixture_files())
def test_encounters_sorted_most_recent_first(fixture: Path) -> None:
    record = load_bundle(fixture)
    encs = get_encounters(record, limit=20)
    if len(encs) < 2:
        pytest.skip("Need at least 2 encounters to test ordering")
    dates = [e["date"] for e in encs if e["date"]]
    if len(dates) < 2:
        pytest.skip("Encounters lack dates")
    # Most-recent-first means a non-increasing date sequence
    assert dates == sorted(dates, reverse=True)


@pytest.mark.parametrize("fixture", fixture_files())
def test_encounters_respects_limit(fixture: Path) -> None:
    record = load_bundle(fixture)
    encs = get_encounters(record, limit=3)
    assert len(encs) <= 3


# --- Allergies ----------------------------------------------------------


@pytest.mark.parametrize("fixture", fixture_files())
def test_allergies_has_clean_keys(fixture: Path) -> None:
    record = load_bundle(fixture)
    allergies = get_allergies(record)
    assert isinstance(allergies, list)
    for a in allergies:
        assert set(a.keys()) >= {
            "substance",
            "reaction",
            "severity",
            "source_resource_id",
        }
        assert a["source_resource_id"] is not None


# --- Smoke test: every tool runs cleanly on every fixture ---------------


@pytest.mark.parametrize("fixture", fixture_files())
def test_all_tools_run_without_error(fixture: Path) -> None:
    """Cross-check that no tool throws on any fixture."""
    record = load_bundle(fixture)
    get_patient_demographics(record)
    get_conditions(record)
    get_medications(record)
    get_observations(record)
    get_encounters(record)
    get_allergies(record)
