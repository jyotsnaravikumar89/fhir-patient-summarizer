"""Output schema for the patient summarizer agent.

Every clinical claim references a FHIR resource ID for citation enforcement.
Trend annotations are constrained to prevent the most common hallucination mode
(declaring a trend with insufficient data).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --- Sub-models for each summary section ----------------------------------


class PatientDemographics(BaseModel):
    age_years: int | None = Field(
        None, description="Computed age in years; None if birthdate unavailable"
    )
    sex: Literal["male", "female", "other", "unknown"] | None = None
    primary_language: str | None = Field(
        None, description="ISO language code or display string if present"
    )


class ActiveCondition(BaseModel):
    condition_text: str = Field(..., description="Human-readable condition name")
    snomed_code: str | None = Field(
        None, description="SNOMED CT code if present in coding"
    )
    onset_date: str | None = Field(
        None, description="Onset date in ISO 8601 format if available"
    )
    source_resource_id: str = Field(
        ..., description="FHIR Condition resource ID this claim is grounded in"
    )


class CurrentMedication(BaseModel):
    medication_text: str
    rxnorm_code: str | None = None
    dose: str | None = Field(
        None, description="Dose text as a single string, e.g. '10 mg daily'"
    )
    start_date: str | None = None
    indication_if_known: str | None = Field(
        None, description="Indication only if explicitly stated in the resource"
    )
    source_resource_id: str


class LabTrend(BaseModel):
    """A lab trend can only be reported with sufficient evidence.

    The agent must use trend='insufficient_data' when fewer than 3 observations
    over fewer than 90 days are available. This is enforced by prompt + by
    eval rather than by schema, but the literal type at least narrows hallucination.
    """

    lab_name: str
    loinc_code: str | None = None
    latest_value: str = Field(
        ..., description="Most recent value as a string (e.g. '7.2 mmol/L')"
    )
    latest_date: str | None = None
    trend: Literal["improving", "worsening", "stable", "insufficient_data"]
    source_resource_ids: list[str] = Field(
        ..., description="All Observation IDs supporting this trend assessment"
    )


class Allergy(BaseModel):
    substance: str
    reaction: str | None = None
    severity: Literal["mild", "moderate", "severe", "unknown"] | None = None
    source_resource_id: str


class RecentEncounter(BaseModel):
    encounter_type: str = Field(
        ..., description="e.g. 'Outpatient', 'Emergency', 'Inpatient'"
    )
    date: str | None = None
    reason: str | None = Field(
        None, description="Reason for visit if recorded; otherwise None"
    )
    source_resource_id: str


# --- Top-level summary ---------------------------------------------------


class ClinicalSummary(BaseModel):
    """Structured clinical summary of a single patient.

    Every list field can be empty, but None is never used as a placeholder for
    a missing list (use []). All resource IDs in source_resource_id(s) fields
    must exist in the input bundle; this is verified by the citation validator
    after the agent run, not by the schema itself.
    """

    patient_demographics: PatientDemographics
    active_conditions: list[ActiveCondition] = Field(default_factory=list)
    current_medications: list[CurrentMedication] = Field(default_factory=list)
    recent_lab_trends: list[LabTrend] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    recent_encounters: list[RecentEncounter] = Field(
        default_factory=list,
        description="Most recent 3 encounters, ordered most recent first",
    )
    summary_narrative: str = Field(
        ...,
        description="2-4 sentence prose summary suitable for a clinician at a glance",
        min_length=20,
        max_length=1000,
    )
    data_quality_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Notes on missing, contradictory, or suspicious data in the bundle. "
            "Empty list if no concerns."
        ),
    )
