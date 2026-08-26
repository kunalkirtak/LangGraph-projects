from __future__ import annotations

from validators import validate_lead

VALID_LEAD = {
    "lead_name": "Sarah Chen",
    "company": "Acme Technologies",
    "role": "CTO",
    "need": "Automating internal AI workflows",
    "industry": "SaaS",
    "company_size": 500,
    "budget": "$75000",
    "urgency": "High",
}


def test_valid_lead_passes():
    result = validate_lead(VALID_LEAD)
    assert result.is_valid
    assert result.errors == []


def test_missing_name_fails():
    lead = {**VALID_LEAD}
    del lead["lead_name"]
    result = validate_lead(lead)
    assert not result.is_valid
    assert any("lead_name" in e for e in result.errors)


def test_missing_company_fails():
    lead = {**VALID_LEAD}
    del lead["company"]
    result = validate_lead(lead)
    assert not result.is_valid
    assert any("company" in e for e in result.errors)


def test_missing_need_fails():
    lead = {**VALID_LEAD}
    del lead["need"]
    result = validate_lead(lead)
    assert not result.is_valid
    assert any("need" in e for e in result.errors)


def test_missing_role_fails():
    lead = {**VALID_LEAD}
    del lead["role"]
    result = validate_lead(lead)
    assert not result.is_valid
    assert any("role" in e for e in result.errors)


def test_invalid_company_size_fails():
    lead = {**VALID_LEAD, "company_size": "a lot of people"}
    result = validate_lead(lead)
    assert not result.is_valid
    assert any("company_size" in e for e in result.errors)


def test_negative_company_size_fails():
    lead = {**VALID_LEAD, "company_size": -5}
    result = validate_lead(lead)
    assert not result.is_valid


def test_blank_string_field_fails():
    lead = {**VALID_LEAD, "lead_name": "   "}
    result = validate_lead(lead)
    assert not result.is_valid


def test_optional_fields_can_be_omitted():
    lead = {
        "lead_name": "Jordan",
        "company": "Small Co",
        "role": "Owner",
        "need": "Basic automation",
    }
    result = validate_lead(lead)
    assert result.is_valid


def test_non_dict_input_fails():
    result = validate_lead("not a dict")  # type: ignore[arg-type]
    assert not result.is_valid
