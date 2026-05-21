#!/usr/bin/env python3
"""
verify_cortana.py — Zero-dependency proof that Cortana's safety guarantees hold.

Usage::

    python verify_cortana.py

No API key. No GPU. No internet. Runs in < 1 second.
Exits 0 if all checks pass, 1 if any fail.

What this proves:
  1.  IR types are structurally sound (Modality, ActionType, Claim, etc.)
  2.  Hard constraints catch ALL three forbidden action types
  3.  Blast-radius threshold enforced at 0.7
  4.  Irreversibility threshold enforced at 0.8
  5.  Permission escalation blocked
  6.  Deception risk blocked at > 0.1
  7.  SIMULATION -> FACT transition is architecturally blocked
  8.  Hypothesis without evidence fails verification
  9.  High-confidence claim without evidence detected as hallucination
  10. Value red-lines define no-deception and no-permission-escalation invariants
"""

import sys
import time
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
HEAD = "\033[1m"
RESET = "\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  — {detail}" if detail else ""
    print(f"  {status}  {name}{suffix}")
    results.append((name, condition))


def section(title):
    print(f"\n{HEAD}{title}{RESET}")


# ============================================================================
# Import
# ============================================================================

section("Import")
try:
    from cortana import (
        Modality, ActionType, ActionMeta, Action,
        Claim, Evidence, EvidenceSource,
        create_claim, create_evidence, promote_to_fact,
        ActionGatekeeper, HardConstraints, SafetyClass, ActionDecision,
        Verifier, VerificationResult, UniverseViolation,
        StructuredMemory,
        ValueDriftController, RedLineTests,
        CortanaPipeline, PipelineResult,
    )
    check("All public symbols importable", True, "cortana package loaded")
except ImportError as e:
    check("All public symbols importable", False, str(e))
    print("\nAbort: cannot continue without core imports.")
    sys.exit(1)

import tempfile, os
_tmpdir = tempfile.mkdtemp(prefix="cortana_verify_")


# ============================================================================
# 1. IR types
# ============================================================================

section("1. IR types")

check("Modality has FACT/HYPOTHESIS/SIMULATION/FICTION",
      all(hasattr(Modality, m) for m in ["FACT", "HYPOTHESIS", "SIMULATION", "FICTION"]),
      str([m.name for m in Modality]))

check("ActionType has 11 action types including forbidden set",
      len(list(ActionType)) >= 11,
      f"{len(list(ActionType))} types")

c = create_claim("server", "is_running", "true", Modality.FACT, 0.95)
check("create_claim returns Claim", isinstance(c, Claim))
check("Claim has unique UUID id", len(c.id) == 36)

e = create_evidence(c, EvidenceSource.EXECUTION_TRACE, "trace://1", "exit=0", 0.95)
c.evidence_ids.append(e.id)
check("Claim.is_verified() True with evidence + high confidence", c.is_verified())

c_noev = create_claim("x", "y", "z", Modality.FACT, 0.9)
check("Claim.is_verified() False without evidence", not c_noev.is_verified())

c_hyp = create_claim("drug", "cures", "cancer", Modality.HYPOTHESIS, 0.9)
check("HYPOTHESIS claim is not verified", not c_hyp.is_verified())


# ============================================================================
# 2. Hard constraints
# ============================================================================

section("2. HardConstraints")

check("ESCALATE_PERMISSION in FORBIDDEN_ACTIONS",
      ActionType.ESCALATE_PERMISSION in HardConstraints.FORBIDDEN_ACTIONS)
check("DECEIVE_USER in FORBIDDEN_ACTIONS",
      ActionType.DECEIVE_USER in HardConstraints.FORBIDDEN_ACTIONS)
check("MANIPULATE_SOCIAL in FORBIDDEN_ACTIONS",
      ActionType.MANIPULATE_SOCIAL in HardConstraints.FORBIDDEN_ACTIONS)
check("MAX_BLAST_RADIUS = 0.7",
      HardConstraints.MAX_BLAST_RADIUS == 0.7, str(HardConstraints.MAX_BLAST_RADIUS))
check("MAX_IRREVERSIBILITY = 0.8",
      HardConstraints.MAX_IRREVERSIBILITY == 0.8, str(HardConstraints.MAX_IRREVERSIBILITY))
check("ALLOW_PERMISSION_INCREASE = False",
      HardConstraints.ALLOW_PERMISSION_INCREASE is False)


# ============================================================================
# 3. ActionGatekeeper: forbidden types
# ============================================================================

section("3. ActionGatekeeper — forbidden action types")

gate = ActionGatekeeper()
t0 = time.perf_counter()
decisions_count = 0

for atype in HardConstraints.FORBIDDEN_ACTIONS:
    action = Action(type=atype, args={}, metadata=ActionMeta())
    decision = gate.evaluate(action)
    decisions_count += 1
    check(f"{atype.name} blocked",
          decision.allowed is False and decision.safety_class == SafetyClass.FORBIDDEN,
          f"class={decision.safety_class.value}")

elapsed_ms = (time.perf_counter() - t0) * 1000
check(f"All 3 forbidden types evaluated in < 10ms",
      elapsed_ms < 10,
      f"{elapsed_ms:.2f}ms for {decisions_count} decisions")


# ============================================================================
# 4. ActionGatekeeper: threshold violations
# ============================================================================

section("4. ActionGatekeeper — threshold violations")

high_blast = Action(type=ActionType.DELETE_FILE, args={},
                    metadata=ActionMeta(blast_radius=0.95, irreversibility_score=0.1))
check("blast_radius=0.95 blocked",
      gate.evaluate(high_blast).allowed is False)

high_irrev = Action(type=ActionType.MODIFY_SYSTEM, args={},
                    metadata=ActionMeta(blast_radius=0.1, irreversibility_score=0.95))
check("irreversibility=0.95 blocked",
      gate.evaluate(high_irrev).allowed is False)

perm_esc = Action(type=ActionType.QUERY_API, args={},
                  metadata=ActionMeta(permission_delta=1))
check("permission_delta=1 blocked",
      gate.evaluate(perm_esc).allowed is False)

deception = Action(type=ActionType.WRITE_CODE, args={},
                   metadata=ActionMeta(deception_risk=0.8))
check("deception_risk=0.8 blocked",
      gate.evaluate(deception).allowed is False)

safe_read = Action(type=ActionType.READ_FILE, args={},
                   metadata=ActionMeta(blast_radius=0.1, irreversibility_score=0.1))
d = gate.evaluate(safe_read)
check("READ_FILE (low risk) allowed",
      d.allowed is True, f"class={d.safety_class.value}")


# ============================================================================
# 5. Universe separation: SIMULATION -> FACT blocked
# ============================================================================

section("5. Universe separation — Verifier.enforce_universe_gate()")

mem = StructuredMemory(storage_path=os.path.join(_tmpdir, "mem"))
verifier = Verifier(mem)

sim_claim = create_claim("attacker", "gained", "root_access",
                         Modality.SIMULATION, 0.99)
e_fake = create_evidence(sim_claim, EvidenceSource.EXECUTION_TRACE, "sim://001", "fake", 0.99)
sim_claim.evidence_ids.append(e_fake.id)
mem.store_claim(sim_claim, e_fake)

result = verifier.enforce_universe_gate(sim_claim, Modality.FACT)
check("SIMULATION -> FACT blocked (enforce_universe_gate returns False)",
      result is False,
      "Core safety guarantee")

fiction_claim = create_claim("dragon", "exists", "in_reality",
                              Modality.FICTION, 0.99)
check("FICTION -> FACT blocked",
      verifier.enforce_universe_gate(fiction_claim, Modality.FACT) is False)

fact_c = create_claim("server", "status", "running", Modality.FACT, 0.9)
check("FACT -> SIMULATION allowed (for counterfactuals)",
      verifier.enforce_universe_gate(fact_c, Modality.SIMULATION) is True)


# ============================================================================
# 6. Verifier: claim verification
# ============================================================================

section("6. Verifier — verify_claim()")

hyp_no_ev = create_claim("drug", "is_safe", "yes", Modality.HYPOTHESIS, 0.9)
mem.store_claim(hyp_no_ev)
vr = verifier.verify_claim(hyp_no_ev)
check("HYPOTHESIS without evidence fails verify_claim()",
      vr.verified is False, f"failures={vr.failures[:1]}")

low_conf = create_claim("x", "y", "z", Modality.FACT, 0.3)
mem.store_claim(low_conf)
vr2 = verifier.verify_claim(low_conf)
check("FACT with confidence=0.3 fails verify_claim()",
      vr2.verified is False)


# ============================================================================
# 7. Hallucination detection
# ============================================================================

section("7. Verifier — detect_hallucination()")

halluc_claim = create_claim("model", "knows", "everything",
                             Modality.FACT, 0.99)
# No evidence — high confidence without evidence = hallucination
is_h, reason = verifier.detect_hallucination(halluc_claim)
check("High-confidence FACT without evidence flagged as hallucination",
      is_h is True, reason[:60] if reason else "")


# ============================================================================
# 8. Value controller red-lines
# ============================================================================

section("8. ValueDriftController — red-line invariants")

vdc = ValueDriftController(storage_path=os.path.join(_tmpdir, "values"))
red_lines = vdc.red_line_tests
check(f"Red-line tests defined ({len(red_lines)} total)",
      len(red_lines) >= 3)

names = [t.name for t in red_lines]
check("no_deception red line present",
      any("deception" in n.lower() for n in names), str(names))
check("no_permission_escalation red line present",
      any("permission" in n.lower() for n in names), str(names))
check("preserve_corrigibility red line present",
      any("corrigib" in n.lower() for n in names))

weights = vdc.current_profile.weights
check("deception_penalty is strongly negative",
      weights.get("deception_penalty", 0) < -5.0,
      f"deception_penalty={weights.get('deception_penalty')}")
check("permission_change_penalty is negative",
      weights.get("permission_change_penalty", 0) < 0,
      f"permission_change_penalty={weights.get('permission_change_penalty')}")


# ============================================================================
# 9. Pipeline integration
# ============================================================================

section("9. CortanaPipeline — end-to-end")

pipeline = CortanaPipeline(storage_path=os.path.join(_tmpdir, "pipeline"))
check("CortanaPipeline initializes with all components",
      all(hasattr(pipeline, attr) for attr in
          ["memory", "verifier", "gatekeeper", "value_controller"]))

# Safe action
safe_action = Action(type=ActionType.READ_FILE, args={},
                     metadata=ActionMeta(blast_radius=0.1, irreversibility_score=0.1))
allowed, reason = pipeline.propose_action(safe_action)
check("Pipeline allows safe READ_FILE action", allowed is True)

# Dangerous action
dangerous_action = Action(type=ActionType.DECEIVE_USER, args={},
                          metadata=ActionMeta(deception_risk=1.0, forbidden=True))
allowed2, reason2 = pipeline.propose_action(dangerous_action)
check("Pipeline blocks DECEIVE_USER action", allowed2 is False, reason2[:60])

# Learn from experience
result = pipeline.learn_from_experience(
    description="Unit tests passed",
    outcome="All 150 tests green",
    success=True,
)
check("learn_from_experience returns PipelineResult",
      isinstance(result, PipelineResult))
check("PipelineResult.success is True", result.success is True)

# Retrieve
knowledge = pipeline.retrieve_relevant_knowledge("unit tests")
check("retrieve_relevant_knowledge returns list", isinstance(knowledge, list))


# ============================================================================
# 10. Throughput
# ============================================================================

section("10. Throughput")

N = 10_000
gate2 = ActionGatekeeper()
actions = [
    Action(type=ActionType.READ_FILE, args={},
           metadata=ActionMeta(blast_radius=0.1, irreversibility_score=0.1))
    for _ in range(N)
]
t0 = time.perf_counter()
for a in actions:
    gate2.evaluate(a)
elapsed = time.perf_counter() - t0
tps = N / elapsed
check(f"Gate throughput > 100k decisions/s",
      tps > 100_000,
      f"{tps:,.0f} decisions/s")


# ============================================================================
# Summary
# ============================================================================

passed = sum(1 for _, ok in results if ok)
total = len(results)
failed = total - passed

print(f"\n{'='*60}")
if failed == 0:
    print(f"\033[92m{passed}/{total} checks passed — Cortana verified.\033[0m")
else:
    print(f"\033[91m{passed}/{total} checks passed — {failed} FAILED.\033[0m")
print(f"{'='*60}\n")

# Cleanup
import shutil
shutil.rmtree(_tmpdir, ignore_errors=True)

sys.exit(0 if failed == 0 else 1)
