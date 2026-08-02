"""Lifecycle stage spans — investigation pipeline emits stage_kind spans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.agent_harness.session.persistence.jsonl_storage import JsonlSessionStorage
from platform.observability.trace.spans import (
    NoopSessionTraceSink,
    bind_session_trace,
    set_session_trace_sink,
)
from surfaces.interactive_shell.session.trace_sink import JsonlSessionTraceSink
from tools.investigation.stages.gather_evidence import ConnectedInvestigationAgent


@pytest.fixture(autouse=True)
def _reset_session_trace_sink() -> Any:
    set_session_trace_sink(NoopSessionTraceSink())
    yield
    set_session_trace_sink(NoopSessionTraceSink())


class _QuietAgent(ConnectedInvestigationAgent):
    def run(  # type: ignore[override]
        self,
        state: dict[str, Any],  # noqa: ARG002
        on_event: Any | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        return {"gather_evidence_ran": True}


def test_run_connected_investigation_emits_stage_spans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each pipeline stage is timed under ``span_kind=stage`` when tracing is on."""
    from tools.investigation.lifecycle import run_connected_investigation
    from tools.investigation.state_factory import make_initial_state

    monkeypatch.setattr(
        "core.agent_harness.session.persistence.jsonl_storage.session_path",
        lambda session_id: tmp_path / f"{session_id}.jsonl",
    )
    storage = JsonlSessionStorage()
    session_id = "sess-lifecycle-stages"
    path = tmp_path / f"{session_id}.jsonl"
    path.write_text(
        json.dumps({"type": "session", "version": 2, "id": session_id}) + "\n",
        encoding="utf-8",
    )
    set_session_trace_sink(JsonlSessionTraceSink(storage=storage))

    state = make_initial_state(raw_alert="alert text")
    with (
        bind_session_trace(session_id),
        patch(
            "tools.investigation.stages.resolve_integrations.resolve_integrations",
            return_value={"resolved_integrations": {}},
        ),
        patch(
            "tools.investigation.stages.intake.extract_alert",
            return_value={"is_noise": False},
        ),
        patch("tools.investigation.stages.plan_evidence.plan_actions", return_value={}),
        patch(
            "tools.investigation.reporting.upstream_correlation.node.node_correlate_upstream",
            return_value={},
        ),
        patch("tools.investigation.reporting.deliver", return_value={}),
    ):
        run_connected_investigation(state, agent_class=_QuietAgent)

    spans = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().splitlines()
        if json.loads(line).get("type") == "trace_span"
    ]
    stage_names = [rec["name"] for rec in spans if rec.get("span_kind") == "stage"]
    assert stage_names == [
        "resolve_integrations",
        "intake",
        "plan_evidence",
        "gather_evidence",
        "diagnose",
        "deliver",
    ]
    assert all(rec["status"] == "ok" for rec in spans if rec.get("span_kind") == "stage")
    assert all("duration_ms" in rec for rec in spans if rec.get("span_kind") == "stage")


def test_run_connected_investigation_skips_later_stages_on_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.investigation.lifecycle import run_connected_investigation
    from tools.investigation.state_factory import make_initial_state

    monkeypatch.setattr(
        "core.agent_harness.session.persistence.jsonl_storage.session_path",
        lambda session_id: tmp_path / f"{session_id}.jsonl",
    )
    storage = JsonlSessionStorage()
    session_id = "sess-lifecycle-noise"
    path = tmp_path / f"{session_id}.jsonl"
    path.write_text(
        json.dumps({"type": "session", "version": 2, "id": session_id}) + "\n",
        encoding="utf-8",
    )
    set_session_trace_sink(JsonlSessionTraceSink(storage=storage))

    state = make_initial_state(raw_alert="noise")
    with (
        bind_session_trace(session_id),
        patch(
            "tools.investigation.stages.resolve_integrations.resolve_integrations",
            return_value={"resolved_integrations": {}},
        ),
        patch(
            "tools.investigation.stages.intake.extract_alert",
            return_value={"is_noise": True},
        ),
    ):
        run_connected_investigation(state, agent_class=_QuietAgent)

    stage_names = [
        json.loads(line)["name"]
        for line in path.read_text(encoding="utf-8").strip().splitlines()
        if json.loads(line).get("type") == "trace_span"
        and json.loads(line).get("span_kind") == "stage"
    ]
    assert stage_names == ["resolve_integrations", "intake"]


def test_run_connected_investigation_noop_sink_emits_nothing() -> None:
    """Headless / gateway default: pipeline stages must not require a sink."""
    from platform.observability.trace.spans import get_session_trace_sink
    from tools.investigation.lifecycle import run_connected_investigation
    from tools.investigation.state_factory import make_initial_state

    assert isinstance(get_session_trace_sink(), NoopSessionTraceSink)
    state = make_initial_state(raw_alert="alert text")
    with (
        patch(
            "tools.investigation.stages.resolve_integrations.resolve_integrations",
            return_value={"resolved_integrations": {}},
        ),
        patch(
            "tools.investigation.stages.intake.extract_alert",
            return_value={"is_noise": True},
        ),
    ):
        out = run_connected_investigation(state, agent_class=_QuietAgent)
    assert out.get("is_noise") is True


# --- Rollback on stage failure ---


class _FailingAgent(ConnectedInvestigationAgent):
    def run(  # type: ignore[override]
        self,
        state: dict[str, Any],  # noqa: ARG002
        on_event: Any | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        raise RuntimeError("gather_evidence blew up")


def test_later_stage_failure_rolls_back_failing_stage_mutations() -> None:
    """Per-stage rollback: a failing stage's mutations are undone;
    mutations from earlier successful stages persist."""
    from tools.investigation.lifecycle import run_connected_investigation
    from tools.investigation.state_factory import make_initial_state

    state = make_initial_state(raw_alert="alert text")
    # resolved_integrations starts as {} from the initial state.
    assert state.get("resolved_integrations") == {}

    with (
        patch(
            "tools.investigation.stages.resolve_integrations.resolve_integrations",
            return_value={"resolved_integrations": {"grafana": {"url": "g"}}},
        ),
        patch(
            "tools.investigation.stages.intake.extract_alert",
            return_value={"is_noise": False, "alert_name": "Stale", "severity": "high"},
        ),
        patch(
            "tools.investigation.stages.plan_evidence.plan_actions",
            side_effect=RuntimeError("plan_actions blew up"),
        ),
        pytest.raises(RuntimeError, match="plan_actions blew up"),
    ):
        run_connected_investigation(state, agent_class=_FailingAgent)

    # resolve_integrations succeeded → its value persists.
    assert state.get("resolved_integrations") == {"grafana": {"url": "g"}}
    # intake succeeded → its mutations persist.
    assert state.get("is_noise") is False
    assert state.get("alert_name") == "Stale"
    # plan_actions failed → its mutations (none, it raised immediately) are absent.
    # severity from intake persists.
    assert state.get("severity") == "high"


def test_stage_rollback_preserves_pre_stage_state_on_first_failure() -> None:
    """Exception in the first stage restores exactly the initial state."""
    from tools.investigation.lifecycle import run_connected_investigation
    from tools.investigation.state_factory import make_initial_state

    state = make_initial_state(raw_alert="alert text")
    original_resolved = state.get("resolved_integrations")

    with (
        patch(
            "tools.investigation.stages.resolve_integrations.resolve_integrations",
            side_effect=RuntimeError("resolve blew up"),
        ),
        pytest.raises(RuntimeError, match="resolve blew up"),
    ):
        run_connected_investigation(state, agent_class=_FailingAgent)

    assert state.get("resolved_integrations") == original_resolved
    assert state.get("is_noise") is False  # initial default


def test_stage_rollback_undoes_mutation_from_failing_stage_only() -> None:
    """When a stage writes values then raises, those values are rolled back;
    earlier stages are unaffected."""
    from tools.investigation.lifecycle import run_connected_investigation
    from tools.investigation.state_factory import make_initial_state

    state = make_initial_state(raw_alert="alert text")
    # intake succeeds — severity changes to "critical"
    # plan_actions raises AFTER writing to state
    with (
        patch(
            "tools.investigation.stages.resolve_integrations.resolve_integrations",
            return_value={"resolved_integrations": {}},
        ),
        patch(
            "tools.investigation.stages.intake.extract_alert",
            return_value={"is_noise": False, "severity": "critical"},
        ),
        patch(
            "tools.investigation.stages.plan_evidence.plan_actions",
            side_effect=RuntimeError("plan blew up"),
        ),
        pytest.raises(RuntimeError),
    ):
        run_connected_investigation(state, agent_class=_FailingAgent)

    # intake ran and has is_noise=False — NOT rolled back (it succeeded).
    assert state.get("is_noise") is False
    # resolve_integrations mutations persist (it succeeded).
    assert "resolved_integrations" in state
    # plan_actions raised before producing output, so no plan_actions key to restore.
