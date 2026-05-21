"""
Tests for ValueDriftController and RedLineTests.

Covers:
- RedLineTests invariants are defined
- ValueProfile creation and versioning
- ValueDriftController construction
All tests run with zero external dependencies.
"""

import pytest
from cortana.ir import Action, ActionType, ActionMeta, ValueDiff
from cortana.values import (
    ValueDriftController, ValueProfile, RedLineTests, ValueTest, TestResult,
)


# ---------------------------------------------------------------------------
# RedLineTests
# ---------------------------------------------------------------------------

class TestRedLineTests:
    def test_default_red_lines_exist(self):
        tests = RedLineTests.get_default_red_lines()
        assert len(tests) >= 3

    def test_red_lines_are_value_tests(self):
        tests = RedLineTests.get_default_red_lines()
        for t in tests:
            assert isinstance(t, ValueTest)

    def test_no_deception_red_line_present(self):
        tests = RedLineTests.get_default_red_lines()
        names = [t.name for t in tests]
        assert any("deception" in n.lower() for n in names), \
            "Expected a no-deception red line"

    def test_no_permission_escalation_red_line_present(self):
        tests = RedLineTests.get_default_red_lines()
        names = [t.name for t in tests]
        assert any("permission" in n.lower() for n in names), \
            "Expected a no-permission-escalation red line"

    def test_preserve_corrigibility_red_line(self):
        tests = RedLineTests.get_default_red_lines()
        names = [t.name for t in tests]
        assert any("corrigib" in n.lower() for n in names), \
            "Expected a corrigibility red line"

    def test_each_test_has_forbidden_behaviors(self):
        tests = RedLineTests.get_default_red_lines()
        for t in tests:
            assert isinstance(t.forbidden_behaviors, list)
            assert len(t.forbidden_behaviors) >= 1, \
                f"Red line '{t.name}' has no forbidden behaviors"

    def test_each_test_has_expected_behavior(self):
        tests = RedLineTests.get_default_red_lines()
        for t in tests:
            assert len(t.expected_behavior) > 0


# ---------------------------------------------------------------------------
# ValueTest dataclass
# ---------------------------------------------------------------------------

class TestValueTest:
    def test_construction(self):
        vt = ValueTest(
            name="no_lying",
            description="Never lie to users",
            scenario="User asks capability question",
            expected_behavior="Provide honest answer",
            forbidden_behaviors=["Claim non-existent capabilities"],
        )
        assert vt.name == "no_lying"
        assert "Claim non-existent" in vt.forbidden_behaviors[0]

    def test_repr_contains_name(self):
        vt = ValueTest(
            name="test_x",
            description="desc",
            scenario="scen",
            expected_behavior="good",
            forbidden_behaviors=["bad"],
        )
        assert "test_x" in repr(vt)


# ---------------------------------------------------------------------------
# ValueProfile
# ---------------------------------------------------------------------------

class TestValueProfile:
    def test_construction(self):
        from datetime import datetime
        vp = ValueProfile(
            version=1,
            weights={"honesty": 1.0, "helpfulness": 0.9, "safety": 1.0},
            timestamp=datetime.now(),
            rationale="Initial profile",
        )
        assert vp.version == 1
        assert "honesty" in vp.weights
        assert vp.test_results == []

    def test_repr_contains_version(self):
        from datetime import datetime
        vp = ValueProfile(
            version=3,
            weights={},
            timestamp=datetime.now(),
            rationale="",
        )
        assert "v3" in repr(vp) or "3" in repr(vp)


# ---------------------------------------------------------------------------
# ValueDriftController
# ---------------------------------------------------------------------------

class TestValueDriftController:
    def test_construction(self, tmp_path):
        vdc = ValueDriftController(storage_path=str(tmp_path / "values"))
        assert vdc is not None

    def test_has_red_line_tests(self, tmp_path):
        vdc = ValueDriftController(storage_path=str(tmp_path / "values"))
        assert hasattr(vdc, 'red_line_tests')
        assert len(vdc.red_line_tests) >= 3

    def test_has_current_profile(self, tmp_path):
        vdc = ValueDriftController(storage_path=str(tmp_path / "values"))
        assert hasattr(vdc, 'current_profile')

    def test_current_profile_has_weights(self, tmp_path):
        vdc = ValueDriftController(storage_path=str(tmp_path / "values"))
        profile = vdc.current_profile
        assert isinstance(profile.weights, dict)
        assert len(profile.weights) > 0

    def test_initial_version_is_positive(self, tmp_path):
        vdc = ValueDriftController(storage_path=str(tmp_path / "values"))
        assert vdc.current_profile.version >= 1

    def test_safety_weight_is_highest(self, tmp_path):
        """Safety should be the highest or equal-highest value weight."""
        vdc = ValueDriftController(storage_path=str(tmp_path / "values"))
        weights = vdc.current_profile.weights
        if "safety" in weights:
            max_weight = max(weights.values())
            assert weights["safety"] >= max_weight * 0.9, \
                "Safety weight should be near or at the maximum"
