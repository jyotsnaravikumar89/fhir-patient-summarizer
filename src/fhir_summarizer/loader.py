"""Load a Synthea-generated FHIR R4 bundle into typed, filtered Python objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fhir.resources.R4B.allergyintolerance import AllergyIntolerance
from fhir.resources.R4B.bundle import Bundle
from fhir.resources.R4B.condition import Condition
from fhir.resources.R4B.encounter import Encounter
from fhir.resources.R4B.medicationrequest import MedicationRequest
from fhir.resources.R4B.observation import Observation
from fhir.resources.R4B.patient import Patient

# The six resource types we care about. Everything else is filtered out.
SUPPORTED_TYPES = {
    "Patient",
    "Condition",
    "MedicationRequest",
    "Observation",
    "Encounter",
    "AllergyIntolerance",
}


@dataclass
class PatientRecord:
    """A typed, filtered view of a single patient's FHIR data."""

    patient: Patient
    conditions: list[Condition] = field(default_factory=list)
    medications: list[MedicationRequest] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    encounters: list[Encounter] = field(default_factory=list)
    allergies: list[AllergyIntolerance] = field(default_factory=list)

    @property
    def resource_count(self) -> dict[str, int]:
        return {
            "Patient": 1,
            "Condition": len(self.conditions),
            "MedicationRequest": len(self.medications),
            "Observation": len(self.observations),
            "Encounter": len(self.encounters),
            "AllergyIntolerance": len(self.allergies),
        }


_RESOURCE_TO_ATTR = {
    "Condition": "conditions",
    "MedicationRequest": "medications",
    "Observation": "observations",
    "Encounter": "encounters",
    "AllergyIntolerance": "allergies",
}


def load_bundle(path: str | Path) -> PatientRecord:
    """Load a Synthea FHIR R4 bundle JSON file into a PatientRecord."""
    path = Path(path)
    with path.open() as f:
        raw = json.load(f)

    bundle = Bundle.model_validate(raw)

    patient: Patient | None = None
    record_kwargs: dict[str, list] = {
        "conditions": [],
        "medications": [],
        "observations": [],
        "encounters": [],
        "allergies": [],
    }

    if not bundle.entry:
        raise ValueError(f"Bundle at {path} has no entries")

    for entry in bundle.entry:
        resource = entry.resource
        if resource is None:
            continue

        rtype = type(resource).__name__
        if rtype not in SUPPORTED_TYPES:
            continue

        if rtype == "Patient":
            if patient is None:
                patient = resource
        else:
            attr = _RESOURCE_TO_ATTR[rtype]
            record_kwargs[attr].append(resource)

    if patient is None:
        raise ValueError(f"No Patient resource found in bundle at {path}")

    return PatientRecord(patient=patient, **record_kwargs)
