"""Agent tools for retrieving patient data.

These functions wrap a PatientRecord and return flat, LLM-friendly dicts.
Raw FHIR JSON is never exposed to the model - that's a deliberate design
choice to reduce hallucination from confusing nested structures.

Every returned dict preserves source_resource_id (or source_resource_ids
for tools that aggregate). This is the foundation for the citation
validator in validator.py.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from fhir_summarizer.loader import PatientRecord


# --- Helpers --------------------------------------------------------------


def _coding_to_code(coding_list: list | None, system_substring: str) -> str | None:
    """Find the first code in a list of FHIR Codings whose system contains the
    given substring. Returns None if not found.

    SNOMED CT system: 'http://snomed.info/sct'
    LOINC system: 'http://loinc.org'
    RxNorm system: 'http://www.nlm.nih.gov/research/umls/rxnorm'
    """
    if not coding_list:
        return None
    for coding in coding_list:
        system = getattr(coding, "system", None) or ""
        if system_substring in system:
            return getattr(coding, "code", None)
    return None


def _codeable_concept_text(cc: Any) -> str | None:
    """Extract a human-readable text from a CodeableConcept.

    Prefers the .text field, falls back to the first coding's .display.
    """
    if cc is None:
        return None
    text = getattr(cc, "text", None)
    if text:
        return text
    coding = getattr(cc, "coding", None) or []
    for c in coding:
        display = getattr(c, "display", None)
        if display:
            return display
    return None


def _format_date(d: Any) -> str | None:
    """Convert a FHIR date / dateTime / instance to ISO string. Returns None
    if no usable value.
    """
    if d is None:
        return None
    if isinstance(d, (date, datetime)):
        return d.isoformat()
    return str(d)


def _compute_age_years(birth_date: Any, as_of: date | None = None) -> int | None:
    if birth_date is None:
        return None
    if as_of is None:
        as_of = date.today()
    if isinstance(birth_date, datetime):
        birth_date = birth_date.date()
    if not isinstance(birth_date, date):
        return None
    years = as_of.year - birth_date.year
    if (as_of.month, as_of.day) < (birth_date.month, birth_date.day):
        years -= 1
    return years if years >= 0 else None


# --- The five tools ------------------------------------------------------


def get_patient_demographics(record: PatientRecord) -> dict[str, Any]:
    """Return basic demographic fields plus the patient resource ID."""
    p = record.patient
    return {
        "age_years": _compute_age_years(getattr(p, "birthDate", None)),
        "sex": getattr(p, "gender", None),
        "primary_language": _extract_primary_language(p),
        "source_resource_id": p.id,
    }


def _extract_primary_language(patient: Any) -> str | None:
    """Pull a primary language from Patient.communication. Returns None if
    not specified.
    """
    comms = getattr(patient, "communication", None) or []
    for comm in comms:
        if getattr(comm, "preferred", False):
            return _codeable_concept_text(getattr(comm, "language", None))
    if comms:
        return _codeable_concept_text(getattr(comms[0], "language", None))
    return None


def get_conditions(
    record: PatientRecord, status_filter: str | None = None
) -> list[dict[str, Any]]:
    """Return conditions, optionally filtered by clinical status.

    status_filter values: 'active', 'recurrence', 'relapse', 'inactive',
    'remission', 'resolved'. None returns all.
    """
    out: list[dict[str, Any]] = []
    for cond in record.conditions:
        clinical_status = _codeable_concept_text(
            getattr(cond, "clinicalStatus", None)
        )
        if status_filter and clinical_status != status_filter:
            continue

        code = getattr(cond, "code", None)
        snomed = _coding_to_code(getattr(code, "coding", None) or [], "snomed")
        out.append(
            {
                "condition_text": _codeable_concept_text(code) or "Unknown",
                "snomed_code": snomed,
                "onset_date": _format_date(getattr(cond, "onsetDateTime", None)),
                "clinical_status": clinical_status,
                "source_resource_id": cond.id,
            }
        )
    return out


def get_medications(
    record: PatientRecord, status_filter: str | None = None
) -> list[dict[str, Any]]:
    """Return medications, optionally filtered by status.

    status values include 'active', 'completed', 'stopped'. None returns all.
    """
    out: list[dict[str, Any]] = []
    for med in record.medications:
        status = getattr(med, "status", None)
        if status_filter and status != status_filter:
            continue

        med_concept = getattr(med, "medicationCodeableConcept", None)
        text = _codeable_concept_text(med_concept) or "Unknown medication"
        rxnorm = _coding_to_code(
            (getattr(med_concept, "coding", None) or []) if med_concept else [],
            "rxnorm",
        )

        # Dose extraction is best-effort; FHIR dosage is deeply nested.
        dose_text = None
        dosage_inst = getattr(med, "dosageInstruction", None) or []
        if dosage_inst:
            dose_text = getattr(dosage_inst[0], "text", None)

        # Synthea uses authoredOn as the prescription date.
        start_date = _format_date(getattr(med, "authoredOn", None))

        out.append(
            {
                "medication_text": text,
                "rxnorm_code": rxnorm,
                "dose": dose_text,
                "status": status,
                "start_date": start_date,
                "source_resource_id": med.id,
            }
        )
    return out


def get_observations(
    record: PatientRecord,
    category: str | None = None,
    since_date: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return observations, optionally filtered by category and date.

    category values include 'laboratory', 'vital-signs', 'survey'.
    since_date is an ISO date string; observations on or after this date are
    returned. If since_date is None, defaults to the last 2 years to bound
    the result set for patients with very long histories.

    limit caps the number of returned observations (most recent first).
    Default 200 is sufficient for trend analysis without blowing the
    LLM context window.
    """
    from datetime import timedelta
    since_dt: date | None = None
    if since_date:
        try:
            since_dt = date.fromisoformat(since_date)
        except ValueError:
            since_dt = None
    if since_dt is None:
        # Default to last 2 years to bound the result set
        since_dt = date.today() - timedelta(days=730)
    if since_dt is None:
        # Default to last 2 years to bound the result set
        since_dt = date.today() - timedelta(days=730)

    out: list[dict[str, Any]] = []
    for obs in record.observations:
        if category:
            cats = getattr(obs, "category", None) or []
            cat_codes = []
            for cat in cats:
                for c in getattr(cat, "coding", None) or []:
                    code = getattr(c, "code", None)
                    if code:
                        cat_codes.append(code)
            if category not in cat_codes:
                continue

        effective = getattr(obs, "effectiveDateTime", None)
        if since_dt and effective:
            obs_date = effective.date() if isinstance(effective, datetime) else None
            if obs_date and obs_date < since_dt:
                continue

        code = getattr(obs, "code", None)
        loinc = _coding_to_code(getattr(code, "coding", None) or [], "loinc")

        # Value extraction handles the most common cases. FHIR has many
        # value[x] variants; we cover Quantity and string here.
        value_str = "no value"
        unit = None
        value_quantity = getattr(obs, "valueQuantity", None)
        if value_quantity is not None:
            v = getattr(value_quantity, "value", None)
            unit = getattr(value_quantity, "unit", None)
            if v is not None:
                value_str = f"{v}"
        else:
            value_string = getattr(obs, "valueString", None)
            if value_string:
                value_str = value_string

        out.append(
            {
                "name": _codeable_concept_text(code) or "Unknown observation",
                "loinc_code": loinc,
                "value": value_str,
                "unit": unit,
                "effective_date": _format_date(effective),
                "source_resource_id": obs.id,
            }
        )
    # Sort most-recent-first and cap at limit
    out.sort(key=lambda o: o["effective_date"] or "", reverse=True)
    return out[:limit]


def get_encounters(record: PatientRecord, limit: int = 10) -> list[dict[str, Any]]:
    """Return the most recent encounters, ordered most recent first."""
    encounters = list(record.encounters)

    def sort_key(enc: Any) -> str:
        period = getattr(enc, "period", None)
        start = getattr(period, "start", None) if period else None
        return _format_date(start) or ""

    encounters.sort(key=sort_key, reverse=True)

    out: list[dict[str, Any]] = []
    for enc in encounters[:limit]:
        period = getattr(enc, "period", None)
        start = getattr(period, "start", None) if period else None

        # Encounter type extraction
        enc_type = "Unknown"
        types = getattr(enc, "type", None) or []
        if types:
            enc_type = _codeable_concept_text(types[0]) or "Unknown"
        else:
            class_ = getattr(enc, "class_fhir", None) or getattr(enc, "class", None)
            display = getattr(class_, "display", None) if class_ else None
            if display:
                enc_type = display

        # Reason extraction
        reason = None
        reasons = (
            getattr(enc, "reasonCode", None)
            or getattr(enc, "reason", None)
            or []
        )
        if reasons:
            reason = _codeable_concept_text(reasons[0])

        out.append(
            {
                "encounter_type": enc_type,
                "date": _format_date(start),
                "reason": reason,
                "source_resource_id": enc.id,
            }
        )
    return out


def get_allergies(record: PatientRecord) -> list[dict[str, Any]]:
    """Return all allergies. No filtering since allergies are typically few
    and always relevant.
    """
    out: list[dict[str, Any]] = []
    for allergy in record.allergies:
        substance = _codeable_concept_text(getattr(allergy, "code", None)) or "Unknown"

        reaction_text = None
        severity = None
        reactions = getattr(allergy, "reaction", None) or []
        if reactions:
            r = reactions[0]
            manifestations = getattr(r, "manifestation", None) or []
            if manifestations:
                reaction_text = _codeable_concept_text(manifestations[0])
            severity = getattr(r, "severity", None)

        out.append(
            {
                "substance": substance,
                "reaction": reaction_text,
                "severity": severity,
                "source_resource_id": allergy.id,
            }
        )
    return out
