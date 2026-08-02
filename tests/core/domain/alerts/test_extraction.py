from __future__ import annotations

from core.domain.alerts.extraction import (
    fallback_details,
    needs_full_json_prompt,
)


def test_needs_full_json_prompt_for_alertmanager_shape() -> None:
    payload = {
        "commonLabels": {"alertname": "HighCPU"},
        "alerts": [{"startsAt": "2026-01-01T00:00:00Z"}],
    }
    assert needs_full_json_prompt(payload) is True


def test_fallback_details_reads_alertmanager_labels() -> None:
    raw_alert = {
        "commonLabels": {"alertname": "DiskFull", "severity": "critical", "service": "api"},
        "commonAnnotations": {"summary": "disk pressure"},
    }
    details = fallback_details({}, raw_alert)
    assert details.alert_name == "DiskFull"
    assert details.severity == "critical"
    assert details.is_noise is True
    assert "pipeline_name" not in details.model_fields


def test_fallback_details_defaults_to_noise_on_failure() -> None:
    """Fallback must classify as noise when LLM extraction is unavailable."""
    details = fallback_details({}, "plain text alert")
    assert details.is_noise is True
    assert details.alert_name == "unknown"
    assert details.severity == "unknown"


def test_fallback_details_noise_default_even_with_structured_payload() -> None:
    """Even a structured payload defaults to noise in fallback — the LLM
    is the reliable classifier and fallback cannot distinguish noise from alert."""
    details = fallback_details(
        {"alert_name": "SuspectedIncident", "severity": "high"},
        {"text": "ERROR: disk full on node-3"},
    )
    assert details.is_noise is True
    assert details.alert_name == "SuspectedIncident"
