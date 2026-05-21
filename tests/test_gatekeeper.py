"""
Tests for the Cortana Action Gatekeeper (safety gate).

Covers:
- Hard constraints: forbidden action types, blast radius, irreversibility
- SafetyClass classification
- Adversarial inputs: deception risk, permission escalation
- Audit trail logging
All tests run with zero external dependencies.
"""

import pytest
from cortana.ir import (
    Action, ActionType, ActionMeta, Claim, Modality, create_claim,
)
from cortana.gatekeeper import (
    ActionGatekeeper, HardConstraints, SafetyClass, ActionDecision,
)


def make_action(action_type, *, irreversibility=0.1, blast=0.1,
                deception=0.0, forbidden=False, permission_delta=0,
                data_access_change=0, args=None):
    """Helper — build an Action with given risk scores."""
    return Action(
        type=action_type,
        args=args or {},
        metadata=ActionMeta(
            irreversibility_score=irreversibility,
            blast_radius=blast,
            deception_risk=deception,
            forbidden=forbidden,
            permission_delta=permission_delta,
            data_access_change=data_access_change,
        ),
    )


@pytest.fixture
def gate():
    return ActionGatekeeper()


# ---------------------------------------------------------------------------
# HardConstraints constants
# ---------------------------------------------------------------------------

class TestHardConstraints:
    def test_forbidden_actions_not_empty(self):
        assert len(HardConstraints.FORBIDDEN_ACTIONS) >= 3

    def test_escalate_permission_is_forbidden(self):
        assert ActionType.ESCALATE_PERMISSION in HardConstraints.FORBIDDEN_ACTIONS

    def test_deceive_user_is_forbidden(self):
        assert ActionType.DECEIVE_USER in HardConstraints.FORBIDDEN_ACTIONS

    def test_manipulate_social_is_forbidden(self):
        assert ActionType.MANIPULATE_SOCIAL in HardConstraints.FORBIDDEN_ACTIONS

    def test_max_blast_radius_lt_one(self):
        assert 0 < HardConstraints.MAX_BLAST_RADIUS < 1.0

    def test_max_irreversibility_lt_one(self):
        assert 0 < HardConstraints.MAX_IRREVERSIBILITY < 1.0

    def test_permission_increase_forbidden(self):
        assert HardConstraints.ALLOW_PERMISSION_INCREASE is False

    def test_check_safe_action_has_no_violations(self):
        action = make_action(ActionType.READ_FILE, irreversibility=0.1, blast=0.1)
        valid, violations = HardConstraints.check(action, action.metadata)
        assert valid is True
        assert violations == []

    def test_check_forbidden_type_returns_violation(self):
        action = make_action(ActionType.DECEIVE_USER, deception=0.9)
        valid, violations = HardConstraints.check(action, action.metadata)
        assert valid is False
        assert len(violations) >= 1

    def test_check_excessive_blast_radius(self):
        action = make_action(ActionType.DELETE_FILE, blast=0.95)
        valid, violations = HardConstraints.check(action, action.metadata)
        assert valid is False
        assert any("blast" in v.lower() for v in violations)

    def test_check_excessive_irreversibility(self):
        action = make_action(ActionType.MODIFY_SYSTEM, irreversibility=0.95)
        valid, violations = HardConstraints.check(action, action.metadata)
        assert valid is False
        assert any("irreversib" in v.lower() for v in violations)

    def test_check_permission_escalation(self):
        action = make_action(ActionType.QUERY_API, permission_delta=1)
        valid, violations = HardConstraints.check(action, action.metadata)
        assert valid is False
        assert any("permission" in v.lower() for v in violations)

    def test_check_deception_risk(self):
        action = make_action(ActionType.WRITE_CODE, deception=0.5)
        valid, violations = HardConstraints.check(action, action.metadata)
        assert valid is False
        assert any("deception" in v.lower() for v in violations)

    def test_check_explicitly_forbidden_flag(self):
        action = make_action(ActionType.WRITE_CODE, forbidden=True)
        valid, violations = HardConstraints.check(action, action.metadata)
        assert valid is False
        assert any("forbidden" in v.lower() for v in violations)


# ---------------------------------------------------------------------------
# ActionGatekeeper.evaluate()
# ---------------------------------------------------------------------------

class TestActionGatekeeperEvaluate:
    def test_safe_read_is_allowed(self, gate):
        action = make_action(ActionType.READ_FILE,
                             irreversibility=0.1, blast=0.1)
        decision = gate.evaluate(action)
        assert isinstance(decision, ActionDecision)
        assert decision.allowed is True

    def test_deceive_user_is_blocked(self, gate):
        action = make_action(ActionType.DECEIVE_USER,
                             deception=1.0, forbidden=True)
        decision = gate.evaluate(action)
        assert decision.allowed is False
        assert decision.safety_class == SafetyClass.FORBIDDEN

    def test_escalate_permission_is_blocked(self, gate):
        action = make_action(ActionType.ESCALATE_PERMISSION,
                             permission_delta=1)
        decision = gate.evaluate(action)
        assert decision.allowed is False
        assert decision.safety_class == SafetyClass.FORBIDDEN

    def test_manipulate_social_is_blocked(self, gate):
        action = make_action(ActionType.MANIPULATE_SOCIAL)
        decision = gate.evaluate(action)
        assert decision.allowed is False
        assert decision.safety_class == SafetyClass.FORBIDDEN

    def test_high_blast_radius_blocked(self, gate):
        action = make_action(ActionType.DELETE_FILE, blast=0.95)
        decision = gate.evaluate(action)
        assert decision.allowed is False

    def test_high_irreversibility_blocked(self, gate):
        action = make_action(ActionType.MODIFY_SYSTEM, irreversibility=0.95)
        decision = gate.evaluate(action)
        assert decision.allowed is False

    def test_decision_has_reasons(self, gate):
        action = make_action(ActionType.DECEIVE_USER, deception=1.0)
        decision = gate.evaluate(action)
        assert isinstance(decision.reasons, list)
        assert len(decision.reasons) >= 1

    def test_allowed_decision_has_safety_class(self, gate):
        action = make_action(ActionType.READ_FILE)
        decision = gate.evaluate(action)
        assert decision.safety_class in list(SafetyClass)

    def test_decision_with_justification(self, gate):
        action = make_action(ActionType.WRITE_CODE,
                             irreversibility=0.2, blast=0.2)
        justification = [
            create_claim("task", "requires", "writing_code",
                         Modality.FACT, 0.9)
        ]
        decision = gate.evaluate(action, justification=justification)
        # Should still return a valid ActionDecision
        assert isinstance(decision, ActionDecision)

    def test_alternatives_field_is_list(self, gate):
        action = make_action(ActionType.DELETE_FILE, blast=0.95)
        decision = gate.evaluate(action)
        assert isinstance(decision.alternatives, list)

    def test_write_code_safe_scores_allowed(self, gate):
        action = make_action(ActionType.WRITE_CODE,
                             irreversibility=0.2, blast=0.1)
        decision = gate.evaluate(action)
        assert decision.allowed is True

    def test_install_package_moderate_risk(self, gate):
        """INSTALL_PACKAGE with moderate scores should evaluate without crashing."""
        action = make_action(ActionType.INSTALL_PACKAGE,
                             irreversibility=0.4, blast=0.3)
        decision = gate.evaluate(action)
        assert isinstance(decision, ActionDecision)

    def test_network_access_low_risk_allowed(self, gate):
        action = make_action(ActionType.ACCESS_NETWORK,
                             irreversibility=0.1, blast=0.2)
        decision = gate.evaluate(action)
        assert decision.allowed is True

    def test_network_access_high_blast_blocked(self, gate):
        action = make_action(ActionType.ACCESS_NETWORK, blast=0.9)
        decision = gate.evaluate(action)
        assert decision.allowed is False


# ---------------------------------------------------------------------------
# SafetyClass hierarchy
# ---------------------------------------------------------------------------

class TestSafetyClass:
    def test_all_safety_classes_exist(self):
        expected = {"SAFE", "REVERSIBLE", "CONFIRM", "FORBIDDEN"}
        actual = {sc.name for sc in SafetyClass}
        assert expected.issubset(actual)

    def test_forbidden_is_not_safe(self):
        assert SafetyClass.FORBIDDEN != SafetyClass.SAFE


# ---------------------------------------------------------------------------
# Adversarial patterns
# ---------------------------------------------------------------------------

class TestAdversarialGatekeeper:
    """Test that sophisticated adversarial inputs are still caught."""

    def test_escalate_via_custom_args(self, gate):
        """A QUERY_API action that tries to escalate should still pass constraint check."""
        action = Action(
            type=ActionType.QUERY_API,
            args={"endpoint": "sudo", "method": "escalate_permissions"},
            metadata=ActionMeta(
                irreversibility_score=0.1,
                blast_radius=0.1,
                permission_delta=2,   # <-- escalation attempt
                deception_risk=0.0,
                forbidden=False,
            ),
        )
        decision = gate.evaluate(action)
        assert decision.allowed is False

    def test_forbidden_flag_always_blocks(self, gate):
        """Even a READ_FILE is blocked if forbidden=True."""
        action = make_action(ActionType.READ_FILE, forbidden=True)
        decision = gate.evaluate(action)
        assert decision.allowed is False

    def test_all_three_forbidden_types_blocked(self, gate):
        """Every type in FORBIDDEN_ACTIONS must be blocked."""
        for action_type in HardConstraints.FORBIDDEN_ACTIONS:
            action = Action(
                type=action_type,
                args={},
                metadata=ActionMeta(),
            )
            decision = gate.evaluate(action)
            assert decision.allowed is False, \
                f"Expected {action_type.name} to be blocked, but it was allowed"

    def test_deception_in_write_code_blocked(self, gate):
        action = make_action(ActionType.WRITE_CODE, deception=0.8)
        decision = gate.evaluate(action)
        assert decision.allowed is False
