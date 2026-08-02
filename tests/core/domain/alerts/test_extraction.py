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
    """Structured alert retains is_noise=False — safer to investigate than drop."""
    raw_alert = {
        "commonLabels": {"alertname": "DiskFull", "severity": "critical", "service": "api"},
        "commonAnnotations": {"summary": "disk pressure"},
    }
    details = fallback_details({}, raw_alert)
    assert details.alert_name == "DiskFull"
    assert details.severity == "critical"
    assert details.is_noise is False
    assert "pipeline_name" not in details.model_fields


def test_fallback_details_plain_string_also_investigates() -> None:
    """Even an unstructured string default to investigate when LLM is unavailable."""
    details = fallback_details({}, "ERROR: something broke")
    assert details.is_noise is False
    assert details.alert_name == "unknown"
    assert details.severity == "unknown"


def test_fallback_details_preserves_state_values() -> None:
    """State-supplied alert_name and severity are forwarded through fallback."""
    details = fallback_details(
        {"alert_name": "SuspectedIncident", "severity": "high"},
        {"text": "ERROR: disk full on node-3"},
    )
    assert details.is_noise is False
    assert details.alert_name == "SuspectedIncident"
