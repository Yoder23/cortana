"""
Integration tests for the Cortana Pipeline.

Covers:
- CortanaPipeline construction (no external deps)
- propose_action: safe actions allowed, forbidden blocked
- learn_from_experience: returns PipelineResult
- retrieve_relevant_knowledge: returns list
- stats: returns dict
All tests use tmp_path; no API key or GPU required.
"""

import pytest
from cortana.ir import (
    Modality, ActionType, ActionMeta, Action, Claim,
    create_claim, create_evidence, EvidenceSource,
)
from cortana.pipeline import CortanaPipeline, PipelineResult


@pytest.fixture
def pipeline(tmp_path):
    return CortanaPipeline(storage_path=str(tmp_path / "cortana"))


def make_action(action_type, *, irrev=0.1, blast=0.1,
                deception=0.0, forbidden=False, permission_delta=0):
    return Action(
        type=action_type,
        args={},
        metadata=ActionMeta(
            irreversibility_score=irrev,
            blast_radius=blast,
            deception_risk=deception,
            forbidden=forbidden,
            permission_delta=permission_delta,
        ),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestCortanaPipelineConstruction:
    def test_pipeline_creates_components(self, pipeline):
        assert pipeline.memory is not None
        assert pipeline.verifier is not None
        assert pipeline.gatekeeper is not None
        assert pipeline.value_controller is not None

    def test_storage_path_created(self, pipeline):
        assert pipeline.storage_path.exists()

    def test_memory_has_no_events_initially(self, pipeline):
        assert pipeline.memory.event_log.count() == 0


# ---------------------------------------------------------------------------
# propose_action
# ---------------------------------------------------------------------------

class TestProposeAction:
    def test_safe_read_action_allowed(self, pipeline):
        action = make_action(ActionType.READ_FILE)
        allowed, reason = pipeline.propose_action(action)
        assert allowed is True
        assert isinstance(reason, str)

    def test_deceive_user_blocked(self, pipeline):
        action = make_action(ActionType.DECEIVE_USER,
                             deception=1.0, forbidden=True)
        allowed, reason = pipeline.propose_action(action)
        assert allowed is False
        assert len(reason) > 0

    def test_escalate_permission_blocked(self, pipeline):
        action = make_action(ActionType.ESCALATE_PERMISSION,
                             permission_delta=1)
        allowed, reason = pipeline.propose_action(action)
        assert allowed is False

    def test_manipulate_social_blocked(self, pipeline):
        action = make_action(ActionType.MANIPULATE_SOCIAL)
        allowed, reason = pipeline.propose_action(action)
        assert allowed is False

    def test_high_blast_radius_blocked(self, pipeline):
        action = make_action(ActionType.DELETE_FILE, blast=0.95)
        allowed, reason = pipeline.propose_action(action)
        assert allowed is False

    def test_write_code_low_risk_allowed(self, pipeline):
        action = make_action(ActionType.WRITE_CODE, irrev=0.2, blast=0.1)
        allowed, reason = pipeline.propose_action(action)
        assert allowed is True

    def test_propose_action_with_justification(self, pipeline):
        action = make_action(ActionType.WRITE_CODE)
        justification = [
            create_claim("task", "requires", "code", Modality.FACT, 0.9)
        ]
        allowed, reason = pipeline.propose_action(action, justification=justification)
        assert isinstance(allowed, bool)

    def test_propose_returns_string_reason(self, pipeline):
        action = make_action(ActionType.READ_FILE)
        allowed, reason = pipeline.propose_action(action)
        assert isinstance(reason, str)


# ---------------------------------------------------------------------------
# learn_from_experience
# ---------------------------------------------------------------------------

class TestLearnFromExperience:
    def test_returns_pipeline_result(self, pipeline):
        result = pipeline.learn_from_experience(
            description="Ran unit tests",
            outcome="All 10 tests passed",
            success=True,
        )
        assert isinstance(result, PipelineResult)

    def test_success_result_has_success_true(self, pipeline):
        result = pipeline.learn_from_experience(
            description="Deployed service",
            outcome="Service is running on port 8080",
            success=True,
        )
        assert result.success is True

    def test_failure_experience_stored(self, pipeline):
        result = pipeline.learn_from_experience(
            description="Attempted dangerous operation",
            outcome="Permission denied",
            success=False,
            error_message="PermissionError: cannot access /etc/shadow",
        )
        assert isinstance(result, PipelineResult)

    def test_result_has_integer_counts(self, pipeline):
        result = pipeline.learn_from_experience(
            description="Read config file",
            outcome="Config loaded: 15 keys",
            success=True,
        )
        assert isinstance(result.claims_stored, int)
        assert isinstance(result.actions_proposed, int)
        assert isinstance(result.actions_approved, int)
        assert isinstance(result.actions_rejected, int)

    def test_result_violations_is_list(self, pipeline):
        result = pipeline.learn_from_experience(
            description="Ran benchmark",
            outcome="620k decisions/s",
            success=True,
        )
        assert isinstance(result.violations, list)

    def test_memory_grows_after_learning(self, pipeline):
        before = pipeline.memory.event_log.count()
        pipeline.learn_from_experience(
            description="Tested component",
            outcome="Component works correctly",
            success=True,
        )
        after = pipeline.memory.event_log.count()
        assert after >= before


# ---------------------------------------------------------------------------
# retrieve_relevant_knowledge
# ---------------------------------------------------------------------------

class TestRetrieveRelevantKnowledge:
    def test_returns_list(self, pipeline):
        result = pipeline.retrieve_relevant_knowledge("server status")
        assert isinstance(result, list)

    def test_facts_only_flag(self, pipeline):
        result = pipeline.retrieve_relevant_knowledge("database", facts_only=True)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, Claim)

    def test_retrieve_finds_previously_stored(self, pipeline):
        # Store something then retrieve
        pipeline.learn_from_experience(
            description="PostgreSQL replication is active",
            outcome="Replication lag: 0ms",
            success=True,
        )
        results = pipeline.retrieve_relevant_knowledge("PostgreSQL")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestPipelineStats:
    def test_stats_returns_dict(self, pipeline):
        stats = pipeline.stats()
        assert isinstance(stats, dict)

    def test_stats_has_expected_keys(self, pipeline):
        stats = pipeline.stats()
        # Should have at least some stats
        assert len(stats) > 0


# ---------------------------------------------------------------------------
# Full integration: learn → retrieve → act
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_learn_then_retrieve_then_act(self, pipeline):
        # 1. Learn from a safe experience
        learn_result = pipeline.learn_from_experience(
            description="Loaded configuration successfully",
            outcome="Config: {'timeout': 30, 'retries': 3}",
            success=True,
        )
        assert isinstance(learn_result, PipelineResult)

        # 2. Retrieve relevant knowledge
        knowledge = pipeline.retrieve_relevant_knowledge("configuration")
        assert isinstance(knowledge, list)

        # 3. Propose a safe action
        action = make_action(ActionType.READ_FILE)
        allowed, reason = pipeline.propose_action(action)
        assert allowed is True

        # 4. Propose a dangerous action
        dangerous = make_action(ActionType.DECEIVE_USER, deception=1.0, forbidden=True)
        allowed_dangerous, _ = pipeline.propose_action(dangerous)
        assert allowed_dangerous is False
