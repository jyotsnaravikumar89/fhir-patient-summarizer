"""Citation validator.

Walks a ClinicalSummary and verifies that every source_resource_id (or
source_resource_ids) actually exists in the input bundle. This is the
project's grounding guarantee: the agent cannot invent resource IDs and
have them slip through to the user.

This is deterministic, fast, and runs as a separate post-processing step
after the agent finishes. If validation fails, the agent's output is
rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from fhir_summarizer.loader import PatientRecord
from fhir_summarizer.schema import ClinicalSummary


@dataclass
class CitationIssue:
    """One orphan citation: a source_resource_id that doesn't exist."""

    section: str  # e.g. "active_conditions[2]"
    field: str  # e.g. "source_resource_id"
    cited_id: str
    expected_resource_type: str  # e.g. "Condition"


@dataclass
class ValidationResult:
    """Result of running the citation validator over a summary.

    `is_valid` is True only when there are no orphan citations. `issues` is
    a list of every problem found, suitable for displaying to a developer
    or feeding back into a retry prompt.
    """

    is_valid: bool
    issues: list[CitationIssue]
    total_citations_checked: int

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def summary_message(self) -> str:
        if self.is_valid:
            return (
                f"All {self.total_citations_checked} citations validated "
                f"against the source bundle."
            )
        lines = [
            f"Found {self.issue_count} orphan citation(s) "
            f"out of {self.total_citations_checked} total:"
        ]
        for issue in self.issues:
            lines.append(
                f"  - {issue.section}.{issue.field}={issue.cited_id!r} "
                f"(expected to find {issue.expected_resource_type} with this ID)"
            )
        return "\n".join(lines)


def _build_id_index(record: PatientRecord) -> dict[str, set[str]]:
    """Build a map from resource type -> set of valid IDs for that type."""
    return {
        "Patient": {record.patient.id} if record.patient.id else set(),
        "Condition": {c.id for c in record.conditions if c.id},
        "MedicationRequest": {m.id for m in record.medications if m.id},
        "Observation": {o.id for o in record.observations if o.id},
        "Encounter": {e.id for e in record.encounters if e.id},
        "AllergyIntolerance": {a.id for a in record.allergies if a.id},
    }


def validate_citations(
    summary: ClinicalSummary, record: PatientRecord
) -> ValidationResult:
    """Verify every source_resource_id in the summary exists in the record.

    Each clinical entry has an expected resource type:
        active_conditions -> Condition
        current_medications -> MedicationRequest
        recent_lab_trends -> Observation
        allergies -> AllergyIntolerance
        recent_encounters -> Encounter
    """
    index = _build_id_index(record)
    issues: list[CitationIssue] = []
    total = 0

    # Active conditions
    for i, cond in enumerate(summary.active_conditions):
        total += 1
        if cond.source_resource_id not in index["Condition"]:
            issues.append(
                CitationIssue(
                    section=f"active_conditions[{i}]",
                    field="source_resource_id",
                    cited_id=cond.source_resource_id,
                    expected_resource_type="Condition",
                )
            )

    # Current medications
    for i, med in enumerate(summary.current_medications):
        total += 1
        if med.source_resource_id not in index["MedicationRequest"]:
            issues.append(
                CitationIssue(
                    section=f"current_medications[{i}]",
                    field="source_resource_id",
                    cited_id=med.source_resource_id,
                    expected_resource_type="MedicationRequest",
                )
            )

    # Lab trends - aggregate field, can have multiple IDs per entry
    for i, trend in enumerate(summary.recent_lab_trends):
        for j, obs_id in enumerate(trend.source_resource_ids):
            total += 1
            if obs_id not in index["Observation"]:
                issues.append(
                    CitationIssue(
                        section=f"recent_lab_trends[{i}].source_resource_ids[{j}]",
                        field="source_resource_ids",
                        cited_id=obs_id,
                        expected_resource_type="Observation",
                    )
                )

    # Allergies
    for i, allergy in enumerate(summary.allergies):
        total += 1
        if allergy.source_resource_id not in index["AllergyIntolerance"]:
            issues.append(
                CitationIssue(
                    section=f"allergies[{i}]",
                    field="source_resource_id",
                    cited_id=allergy.source_resource_id,
                    expected_resource_type="AllergyIntolerance",
                )
            )

    # Encounters
    for i, enc in enumerate(summary.recent_encounters):
        total += 1
        if enc.source_resource_id not in index["Encounter"]:
            issues.append(
                CitationIssue(
                    section=f"recent_encounters[{i}]",
                    field="source_resource_id",
                    cited_id=enc.source_resource_id,
                    expected_resource_type="Encounter",
                )
            )

    return ValidationResult(
        is_valid=(len(issues) == 0),
        issues=issues,
        total_citations_checked=total,
    )
