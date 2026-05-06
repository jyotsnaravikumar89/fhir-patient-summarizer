"""Agent control loop for patient summarization.

A small, hand-written ReAct-style loop that orchestrates Claude with the
five FHIR retrieval tools and produces a validated ClinicalSummary.

Design choices:

- No agent framework. The whole loop is ~150 lines and easier to debug
  than abstracted-away framework state. LangGraph earns its complexity at
  multi-agent or multi-step graph scale; not here.
- Tool calls capped at 10 per run. A safety stop against runaway loops.
- One retry on schema validation failure, with the validation error fed
  back to the model. After that, fail loudly.
- The system prompt is opinionated about lab trend reporting. Lab trends
  are the most common hallucination mode in clinical AI; the explicit rule
  (3+ observations over 90+ days, justified by actual values, otherwise
  insufficient_data) is the difference between a tool you'd let near a
  real workflow and one you wouldn't.
"""

from __future__ import annotations

import json
import os
from typing import Any

from anthropic import Anthropic
from pydantic import ValidationError

from fhir_summarizer.loader import PatientRecord
from fhir_summarizer.schema import ClinicalSummary
from fhir_summarizer.tools import (
    get_allergies,
    get_conditions,
    get_encounters,
    get_medications,
    get_observations,
    get_patient_demographics,
)

# --- Configuration -------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-5"
MAX_TOOL_CALLS = 10
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are a clinical summarization assistant. Your task is to
produce a structured clinical summary of a single patient from FHIR resources.

You have six tools for retrieving the patient's data. You must use the tools
to ground every clinical claim. Do not invent or assume facts that are not
returned by a tool.

Your final output MUST be a single JSON object matching the ClinicalSummary
schema. Do not include any prose before or after the JSON.

# The ClinicalSummary schema

```json
{
  "patient_demographics": {
    "age_years": int | null,
    "sex": "male" | "female" | "other" | "unknown" | null,
    "primary_language": str | null
  },
  "active_conditions": [
    {
      "condition_text": str,
      "snomed_code": str | null,
      "onset_date": str | null,
      "source_resource_id": str
    }
  ],
  "current_medications": [
    {
      "medication_text": str,
      "rxnorm_code": str | null,
      "dose": str | null,
      "start_date": str | null,
      "indication_if_known": str | null,
      "source_resource_id": str
    }
  ],
  "recent_lab_trends": [
    {
      "lab_name": str,
      "loinc_code": str | null,
      "latest_value": str,
      "latest_date": str | null,
      "trend": "improving" | "worsening" | "stable" | "insufficient_data",
      "source_resource_ids": [str, ...]
    }
  ],
  "allergies": [
    {
      "substance": str,
      "reaction": str | null,
      "severity": "mild" | "moderate" | "severe" | "unknown" | null,
      "source_resource_id": str
    }
  ],
  "recent_encounters": [
    {
      "encounter_type": str,
      "date": str | null,
      "reason": str | null,
      "source_resource_id": str
    }
  ],
  "summary_narrative": str,
  "data_quality_flags": [str, ...]
}
```

# Critical rules

1. CITATIONS ARE REQUIRED. Every condition, medication, allergy, encounter,
   and lab trend must include a source_resource_id (or source_resource_ids
   for trends) drawn from the IDs returned by the tools. Do not invent IDs.

2. LAB TRENDS REQUIRE EVIDENCE. Only report a trend value of "improving",
   "worsening", or "stable" when ALL of these are true:
   - There are 3 or more observations of the same LOINC code
   - The observations span 90 or more days
   - The trend label is justified by the actual numeric values you saw
   If any of those is not true, the trend MUST be "insufficient_data".
   Lab trends are the most common hallucination mode in clinical AI;
   following this rule is non-negotiable.

3. ACTIVE CONDITIONS ONLY. Filter conditions to those with clinical_status
   of "active", "recurrence", or "relapse". Skip resolved or inactive
   conditions.

4. CURRENT MEDICATIONS ONLY. Filter medications to status "active". Skip
   completed or stopped medications.

5. RECENT ENCOUNTERS = LAST 3. Use get_encounters with limit=3.

6. SURFACE DATA QUALITY ISSUES. If you encounter contradictory, missing,
   or suspicious data (e.g., a medication with no dose, observations with
   no values, encounters with no dates), record it in data_quality_flags.
   An empty list is appropriate when data is clean.

7. NARRATIVE IS BRIEF. The summary_narrative is 2-4 sentences, written
   for a clinician at a glance. Do not repeat all structured data in prose.

# Process

Plan your tool calls. A reasonable order is:
1. get_patient_demographics
2. get_conditions(status_filter="active")
3. get_medications(status_filter="active")
4. get_allergies
5. get_observations for each major lab category to assess trends
6. get_encounters(limit=3)

Then synthesize the final JSON output."""

# --- Tool definitions for the Anthropic API -----------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_patient_demographics",
        "description": "Get the patient's age, sex, primary language, and resource ID.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_conditions",
        "description": (
            "Get the patient's conditions. Optionally filter by clinical status "
            "(e.g. 'active', 'resolved', 'inactive')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "description": "Clinical status to filter by, or omit for all conditions.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_medications",
        "description": (
            "Get the patient's medications. Optionally filter by status "
            "(e.g. 'active', 'completed', 'stopped')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "description": "Status to filter by, or omit for all medications.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_observations",
        "description": (
            "Get the patient's observations (labs, vitals). Optionally filter by "
            "category (e.g. 'laboratory', 'vital-signs') and a since_date in "
            "ISO 8601 format. Use this to assess lab trends over time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Category code, e.g. 'laboratory' or 'vital-signs'.",
                },
                "since_date": {
                    "type": "string",
                    "description": "ISO date; only observations on or after this date are returned.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_encounters",
        "description": (
            "Get the patient's most recent encounters, ordered most recent first. "
            "Pass limit=3 for the recent_encounters section of the summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of encounters to return (default 10).",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_allergies",
        "description": "Get the patient's allergies and intolerances.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# --- Tool dispatch ------------------------------------------------------


def _dispatch_tool(
    name: str, arguments: dict[str, Any], record: PatientRecord
) -> list[dict[str, Any]] | dict[str, Any]:
    """Run the named tool against the PatientRecord and return its output."""
    if name == "get_patient_demographics":
        return get_patient_demographics(record)
    if name == "get_conditions":
        return get_conditions(record, status_filter=arguments.get("status_filter"))
    if name == "get_medications":
        return get_medications(record, status_filter=arguments.get("status_filter"))
    if name == "get_observations":
        return get_observations(
            record,
            category=arguments.get("category"),
            since_date=arguments.get("since_date"),
        )
    if name == "get_encounters":
        return get_encounters(record, limit=arguments.get("limit", 10))
    if name == "get_allergies":
        return get_allergies(record)
    raise ValueError(f"Unknown tool: {name}")


# --- The main loop ------------------------------------------------------


def summarize_patient(
    record: PatientRecord,
    *,
    model: str = DEFAULT_MODEL,
    max_tool_calls: int = MAX_TOOL_CALLS,
    verbose: bool = False,
) -> ClinicalSummary:
    """Run the agent loop and return a validated ClinicalSummary.

    Raises:
        ValueError: if the agent exceeds max_tool_calls without producing a
            final answer, or if the final answer fails schema validation
            even after one retry.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "See the project README for setup."
        )

    client = Anthropic()

    initial_user_message = (
        "Produce a ClinicalSummary for the patient whose data is available "
        "via your tools. Begin by retrieving demographics, then proceed through "
        "the other sections. Return only the final JSON object as your last message."
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": initial_user_message}
    ]

    for tool_call_count in range(max_tool_calls + 1):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if verbose:
            print(
                f"[turn {tool_call_count}] stop_reason={response.stop_reason} "
                f"blocks={[b.type for b in response.content]}"
            )

        # If the model is done with tools, extract its final text.
        if response.stop_reason == "end_turn":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            return _parse_and_validate(client, model, messages, final_text, verbose)

        # Otherwise expect tool_use blocks.
        if response.stop_reason != "tool_use":
            raise ValueError(
                f"Unexpected stop_reason: {response.stop_reason}. "
                f"Content: {response.content}"
            )

        # Append the assistant's turn to history (required for tool-use loop).
        messages.append({"role": "assistant", "content": response.content})

        # Run each tool call and append the results as a single user turn.
        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                output = _dispatch_tool(block.name, block.input or {}, record)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output, default=str),
                    }
                )
            except Exception as exc:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Tool error: {exc}",
                        "is_error": True,
                    }
                )

        messages.append({"role": "user", "content": tool_results})

    raise ValueError(
        f"Agent exceeded max_tool_calls={max_tool_calls} without producing a final answer."
    )


def _parse_and_validate(
    client: Anthropic,
    model: str,
    messages: list[dict[str, Any]],
    final_text: str,
    verbose: bool,
) -> ClinicalSummary:
    """Parse the model's final text as JSON and validate against the schema.

    On the first failure, retry once with the validation error in the prompt.
    On the second failure, raise.
    """
    try:
        return _parse_to_schema(final_text)
    except (ValueError, ValidationError) as first_error:
        if verbose:
            print(f"[validation] first attempt failed: {first_error}")

        retry_messages = list(messages)
        retry_messages.append({"role": "assistant", "content": final_text})
        retry_messages.append(
            {
                "role": "user",
                "content": (
                    f"Your previous response failed schema validation with this error:\n\n"
                    f"{first_error}\n\n"
                    "Please return a corrected JSON object matching the ClinicalSummary "
                    "schema. Output only the JSON, no prose."
                ),
            }
        )
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=retry_messages,
        )
        retry_text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        return _parse_to_schema(retry_text)


def _parse_to_schema(text: str) -> ClinicalSummary:
    """Strip code fences if present, then parse JSON and validate."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip ```json or ``` fence
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Final output is not valid JSON: {exc}") from exc

    return ClinicalSummary.model_validate(data)
