"""Tests for the FHIR loader."""

import json
from pathlib import Path

import pytest

from fhir_summarizer.loader import SUPPORTED_TYPES, load_bundle

FIXTURES = Path(__file__).parent.parent / "data" / "fixtures"


def fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("patient_*.json"))


@pytest.mark.parametrize("fixture", fixture_files())
def test_loads_without_error(fixture: Path) -> None:
    record = load_bundle(fixture)
    assert record.patient is not None
    assert record.patient.id is not None


@pytest.mark.parametrize("fixture", fixture_files())
def test_has_some_clinical_data(fixture: Path) -> None:
    """Synthea patients should have at least conditions and observations."""
    record = load_bundle(fixture)
    assert len(record.conditions) > 0, "Expected at least one Condition"
    assert len(record.observations) > 0, "Expected at least one Observation"


@pytest.mark.parametrize("fixture", fixture_files())
def test_resource_count_is_sane(fixture: Path) -> None:
    record = load_bundle(fixture)
    counts = record.resource_count
    assert counts["Patient"] == 1
    assert all(c >= 0 for c in counts.values())


def test_unsupported_types_filtered(tmp_path: Path) -> None:
    """Resources outside SUPPORTED_TYPES should be silently dropped."""
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "test-patient-1",
                }
            },
            {
                "resource": {
                    "resourceType": "Practitioner",
                    "id": "should-be-ignored",
                }
            },
        ],
    }
    fp = tmp_path / "minimal.json"
    fp.write_text(json.dumps(bundle))

    record = load_bundle(fp)
    assert record.patient.id == "test-patient-1"
    assert len(record.conditions) == 0
    assert len(record.medications) == 0


def test_supported_types_set() -> None:
    """Sanity check that SUPPORTED_TYPES has exactly the six expected types."""
    assert SUPPORTED_TYPES == {
        "Patient",
        "Condition",
        "MedicationRequest",
        "Observation",
        "Encounter",
        "AllergyIntolerance",
    }
