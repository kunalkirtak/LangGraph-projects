"""
validators.py

Validates raw lead input before it is allowed to enter the LangGraph
workflow. Malformed input must never reach the qualification LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

REQUIRED_FIELDS = ("lead_name", "company", "role", "need")
OPTIONAL_FIELDS = ("industry", "company_size", "budget", "urgency")


@dataclass
class ValidationResult:
    """Outcome of validating a raw lead dict."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def validate_lead(raw_lead: dict) -> ValidationResult:
    """Validate a raw lead dictionary before normalization.

    This checks presence/shape only -- it does not make any judgment about
    whether the lead is a *good* one. That judgment belongs to the
    qualification node, not here.
    """

    errors: list[str] = []

    if not isinstance(raw_lead, dict):
        return ValidationResult(is_valid=False, errors=["Lead input must be a dictionary."])

    # Required string fields must be present and non-blank.
    for field_name in REQUIRED_FIELDS:
        value = raw_lead.get(field_name)
        if _is_blank(value):
            errors.append(f"Missing required field: '{field_name}'.")
        elif not isinstance(value, str):
            errors.append(f"Field '{field_name}' must be a string.")

    # lead_name minimum plausibility check.
    lead_name = raw_lead.get("lead_name")
    if isinstance(lead_name, str) and 0 < len(lead_name.strip()) < 2:
        errors.append("Field 'lead_name' is too short to be a valid name.")

    # company_size, if present, must be a positive integer (or numeric string).
    if "company_size" in raw_lead and not _is_blank(raw_lead.get("company_size")):
        company_size = raw_lead["company_size"]
        try:
            size_int = int(company_size)
            if size_int <= 0:
                errors.append("Field 'company_size' must be a positive integer.")
        except (TypeError, ValueError):
            errors.append("Field 'company_size' must be a valid integer.")

    # Optional string fields, if present, must actually be strings.
    for field_name in ("industry", "budget", "urgency"):
        if field_name in raw_lead and not _is_blank(raw_lead.get(field_name)):
            if not isinstance(raw_lead[field_name], str):
                errors.append(f"Field '{field_name}' must be a string if provided.")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
