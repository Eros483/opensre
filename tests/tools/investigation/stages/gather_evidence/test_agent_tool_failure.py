"""Tests that execute_tools failure preserves collected evidence."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from config.constants.investigation import MAX_INVESTIGATION_LOOPS
from core.state.evidence import EvidenceEntry
from tools.investigation.stages.gather_evidence.agent import ConnectedInvestigationAgent
from tools.investigation.stages.gather_evidence.loop import (
    degraded_investigation_from_tool_failure,
)


class _AssertiveAgent(ConnectedInvestigationAgent):
    """Agent that accepts the conclusion immediately so the loop exits clean."""

    def _should_accept_conclusion(
        self,
        *,
        evidence_count: int,  # noqa: ARG002
        iteration: int,  # noqa: ARG002
        final_text: str = "",  # noqa: ARG002
    ) -> tuple[bool, str | None]:
        return True, None


class TestDegradedInvestigationFromToolFailure:
    def test_preserves_evidence_after_tool_failure(self) -> None:
        """D.A1: collected evidence survives a tool-execution error."""
        evidence = {"db": {"rows": 5}}
        evidence_entries = [
            EvidenceEntry(
                key="db",
                data={"rows": 5},
                tool_name="query_db",
                tool_args={"q": "errors"},
                source="datadog",
                loop_iteration=0,
            )
        ]
        messages = [{"role": "assistant", "content": "checking..."}]
        hypotheses = [{"hypothesis": "DB error", "actions": ["query_db"], "loop_iteration": 0}]
        mock_emit = mock.MagicMock()
        mock_tracker = mock.MagicMock()

        result = degraded_investigation_from_tool_failure(
            RuntimeError("connection refused"),
            tracker=mock_tracker,
            _emit=mock_emit,
            evidence=evidence,
            evidence_entries=evidence_entries,
            messages=messages,
            executed_hypotheses=hypotheses,
            tool_context={"connected_integrations": []},
            investigation_loop_count=1,
        )

        assert result["evidence"] == evidence
        assert result["evidence_entries"] == [e.model_dump() for e in evidence_entries]
        assert result["agent_messages"] == messages
        assert result["executed_hypotheses"] == hypotheses
        assert result["root_cause_category"] == "tool_execution_error"
        assert result["validity_score"] == 0.0
        assert result["investigation_loop_count"] == 1

    def test_empty_evidence_survives_tool_failure(self) -> None:
        """D.A2: no-crash when there was no prior evidence to preserve."""
        result = degraded_investigation_from_tool_failure(
            RuntimeError("timeout"),
            tracker=mock.MagicMock(),
            _emit=mock.MagicMock(),
            evidence={},
            evidence_entries=[],
            messages=[],
            executed_hypotheses=[],
            tool_context={"connected_integrations": []},
            investigation_loop_count=0,
        )

        assert result["evidence"] == {}
        assert result["evidence_entries"] == []
        assert result["agent_messages"] == []
        assert result["root_cause_category"] == "tool_execution_error"


@pytest.mark.live_llm  # ponytail: needs fixture refactor to go without LLM; ok for now
class TestAgentToolFailureIntegration:
    def test_execute_tools_iter_raises_returns_degraded_state(self, monkeypatch: Any) -> None:
        """Tool iteration failure returns degraded state with evidence intact."""
        from core.messages import MessageMapper

        agent = _AssertiveAgent()
        agent._on_tuple_event = None
        agent._on_runtime_event = None
        agent._tracker = mock.MagicMock()

        # Reach in and set the internal loop state the agent would accumulate.
        agent._redacted_inputs = {}
        agent._emit = mock.MagicMock()
        agent._build_system_prompt = lambda _s: "system"
        agent._filter_tools = lambda tools: tools

        def _fake_tool(_run_id: str = "") -> dict[str, Any]:
            return {"ok": True}

        from core.domain.types.retrieval import RetrievalControls
        from core.tool_framework.registered_tool import RegisteredTool

        fake_tool = RegisteredTool(
            name="fake_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            source="datadog",  # type: ignore[arg-type]
            run=_fake_tool,
            use_cases=[],
            retrieval_controls=RetrievalControls(),
        )

        mock_llm = mock.MagicMock()
        mock_llm.invoke.return_value = mock.MagicMock(
            has_tool_calls=True,
            tool_calls=[
                mock.MagicMock(
                    id="call_1",
                    name="fake_tool",
                    input={"run_id": "1"},
                )
            ],
            content="",
        )
        mock_llm.tool_schemas = lambda _tools: []

        msg_mapper = MessageMapper(mock_llm)
        msg_mapper.to_assistant_provider_message = lambda r: {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": "{}"}}
                for tc in r.tool_calls
            ],
        }
        msg_mapper.to_tool_result_provider_messages = lambda _tcs, _results: []

        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.get_llm",
            lambda _role: mock_llm,
        )
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.build_investigation_system_prompt",
            lambda _s: "system",
        )
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.get_available_tools",
            lambda _r: [fake_tool],
        )
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.select_investigation_tools",
            lambda tools, _s: tools,
        )
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.build_connected_tool_context",
            lambda _r, _t: {"connected_integrations": ["datadog"], "available_action_names": []},
        )
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.build_seed_calls",
            lambda _s, _t, _l: [],
        )
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.tool_source",
            lambda _tbyn, _name: "datadog",
        )
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.format_alert_context",
            lambda _ps, _tools: "Investigate this.",
        )
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.system_and_tools_overhead",
            lambda _sys, _t: 0,
        )
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.estimate_message_tokens",
            lambda _m, **_kw: 0,
        )
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.context_budget_ceiling_for_model",
            lambda _m: MAX_INVESTIGATION_LOOPS * 1000,
        )
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.enforce_context_budget",
            lambda _m, **_kw: None,
        )

        # Make execute_tools raise
        monkeypatch.setattr(
            "tools.investigation.stages.gather_evidence.agent.execute_tools",
            lambda _calls, _tools, _res: _raise_connection_refused(),
        )

        state: dict[str, Any] = {
            "raw_alert": {"text": "test alert"},
            "alert_name": "TestAlert",
            "severity": "warning",
            "resolved_integrations": {"datadog": {"api_key": "x"}},
        }
        result = agent.run(state)

        assert result["root_cause_category"] == "tool_execution_error"
        assert result["validity_score"] == 0.0
        assert "remediation_steps" in result


def _raise_connection_refused() -> None:
    raise RuntimeError("connection refused")
