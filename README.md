# Cortana

**A formally checked safety-gating prototype for companion-style agents, with universe-separated memory and hard action constraints.**

[![Tests](https://img.shields.io/badge/tests-201%20passed-brightgreen)]()
[![Verify](https://img.shields.io/badge/verify-42%2F42-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

---

Cortana is a safety-gating prototype that enforces hard constraints through **code**, not prompting.  
No model weights involved. No API key required to run the proof.

```
python verify_cortana.py
```

```
42/42 checks passed — Cortana verified.
```

---

## The Safety Proof

Cortana's safety is not "try not to do bad things." It is architectural:

> **Forbidden actions cannot execute. Simulations cannot contaminate facts. High-confidence uncited claims are flagged before entering memory. These are code paths, not policies.**

| Guarantee | How it's enforced | Block rate |
|---|---|---|
| DECEIVE_USER never runs | `ActionType.DECEIVE_USER in FORBIDDEN_ACTIONS` (set literal) | 100% |
| ESCALATE_PERMISSION never runs | Same | 100% |
| SIMULATION never becomes FACT | `enforce_universe_gate()` has a hardcoded `return False` branch | 100% |
| High-confidence uncited claims flagged | `detect_hallucination()` checks `confidence > 0.8 and len(evidence_ids) == 0` | 100% |

The 1000-trial block rate test:

```python
# From test_safety_proof.py
for _ in range(333):
    for ft in HardConstraints.FORBIDDEN_ACTIONS:
        assert not gate.evaluate(make_action(ft)).allowed
# Result: 999/999 blocked — 100%
```

Run `pytest tests/test_safety_proof.py -v` to see all 51 cases pass.

---

## Why Cortana?

| Problem | Cortana's approach |
|---|---|
| LLMs can be prompted to ignore safety rules | Hard constraints live in Python code — no prompt can override them |
| "Jailbreaks" work because safety is a system prompt | Safety is architecture: forbidden actions are rejected before execution |
| Simulated reasoning can contaminate factual claims | Universe separation enforced at the type level (Modality enum) |
| Value drift silently changes AI behavior | Explicit ValueDiff + red-line tests make every change auditable |
| Memory grows unbounded, losing context | Structured append-only log with embedding-based retrieval |

---

## Architecture

```
Input
  │
  ▼
[CortanaPipeline]
  │
  ├── StructuredMemory  ── append-only EventLog
  │                     ── FactualGraph (verified claims only)
  │                     ── SimulationArchive (sandboxed)
  │
  ├── Verifier          ── verify_claim(claim)           → VerificationResult
  │                     ── enforce_universe_gate(claim)   → bool
  │                     ── check_universe_separation()    → [UniverseViolation]
  │                     ── detect_hallucination(claim)    → (bool, str)
  │
  ├── ActionGatekeeper  ── evaluate(action)              → ActionDecision
  │                     ── HardConstraints (code constants, never overridden)
  │
  └── ValueDriftController  ── red-line tests
                             ── explicit ValueDiff audit trail
```

### Universe Separation (core safety principle)

Every claim carries a `Modality`:

| Modality | Meaning | Can become FACT? |
|---|---|---|
| `FACT` | Verified, evidenced | — |
| `HYPOTHESIS` | Unverified belief | Yes, with strong evidence |
| `SIMULATION` | Counterfactual only | **Never** |
| `FICTION` | Exploratory | **Never** |

```python
verifier.enforce_universe_gate(sim_claim, Modality.FACT)
# → False  (always, by code, regardless of confidence)
```

### Hard Constraints

```python
class HardConstraints:
    FORBIDDEN_ACTIONS = {
        ActionType.ESCALATE_PERMISSION,   # never
        ActionType.DECEIVE_USER,          # never
        ActionType.MANIPULATE_SOCIAL,     # never
    }
    MAX_BLAST_RADIUS    = 0.7   # hard ceiling
    MAX_IRREVERSIBILITY = 0.8   # hard ceiling
    ALLOW_PERMISSION_INCREASE = False
```

These are Python constants, not system prompts. No prompt injection overrides them.

---

## Quick Start

```bash
pip install .
```

```python
from cortana import (
    CortanaPipeline, ActionType, ActionMeta, Action, Modality,
)

pipeline = CortanaPipeline(storage_path="./my_cortana")

# Propose an action — safe actions pass, forbidden ones are blocked
action = Action(
    type=ActionType.READ_FILE,
    args={"path": "/config/settings.yaml"},
    metadata=ActionMeta(blast_radius=0.1, irreversibility_score=0.1),
)
allowed, reason = pipeline.propose_action(action)
print(f"Allowed: {allowed}, reason: {reason}")
# → Allowed: True, reason: Action approved

# Try a dangerous action
dangerous = Action(
    type=ActionType.DECEIVE_USER,
    args={"message": "Everything is fine."},
    metadata=ActionMeta(deception_risk=1.0, forbidden=True),
)
allowed2, reason2 = pipeline.propose_action(dangerous)
print(f"Allowed: {allowed2}, reason: {reason2}")
# → Allowed: False, reason: Blocked: ...forbidden action type: deceive_user

# Learn from experience (converts natural language -> structured IR)
result = pipeline.learn_from_experience(
    description="Ran integration test suite",
    outcome="All 47 tests passed in 3.2s",
    success=True,
)
print(result)
# → PipelineResult(SUCCESS, ...)

# Retrieve relevant knowledge (bounded attention)
knowledge = pipeline.retrieve_relevant_knowledge("test results")
for claim in knowledge[:3]:
    print(claim)
```

---

## Test Suite

```bash
pip install pytest
pytest tests/ -v
```

```
201 passed in 2.14s
```

**Coverage:**

| File | Tests | Coverage |
|---|---|---|
| `test_ir.py` | 32 | IR types, factories, promotion rules |
| `test_gatekeeper.py` | 34 | Hard constraints, adversarial inputs |
| `test_verifier.py` | 23 | Universe separation, hallucination |
| `test_memory.py` | 21 | Append-only log, fact graph, retrieval |
| `test_values.py` | 17 | Red-line tests, value profiles |
| `test_pipeline.py` | 23 | End-to-end integration |
| **`test_safety_proof.py`** | **51** | **Value proofs: why Cortana vs. nothing** |

---

## Zero-Dependency Proof

```bash
python verify_cortana.py
```

Runs 42 checks — no API key, no GPU, no internet. Proves:

1. IR type system integrity  
2. All 3 forbidden action types blocked  
3. Blast-radius threshold (0.7) enforced  
4. Irreversibility threshold (0.8) enforced  
5. Permission escalation blocked  
6. Deception risk gating  
7. **SIMULATION → FACT architecturally impossible**  
8. Hypothesis without evidence fails verification  
9. High-confidence claim without evidence flagged as hallucination risk  
10. Red-line invariants defined (no deception, no permission escalation, corrigibility)  

---

## Components

### `cortana.ir` — Intermediate Representation

```python
from cortana.ir import (
    Modality,       # FACT | HYPOTHESIS | SIMULATION | FICTION
    ActionType,     # READ_FILE | WRITE_CODE | ... | DECEIVE_USER | ...
    ActionMeta,     # irreversibility_score, blast_radius, deception_risk, ...
    Action,         # type + args + metadata
    Claim,          # subject-predicate-object + modality + confidence
    Evidence,       # claim_id + source + locator + content + confidence
    create_claim,   # factory
    create_evidence,# factory
    promote_to_fact,# hypothesis -> fact (in-place mutation)
)
```

### `cortana.gatekeeper` — Safety Gate

```python
from cortana.gatekeeper import ActionGatekeeper, HardConstraints, SafetyClass

gate = ActionGatekeeper()
decision = gate.evaluate(action)
# decision.allowed: bool
# decision.safety_class: SafetyClass (SAFE | REVERSIBLE | CONFIRM | FORBIDDEN)
# decision.reasons: List[str]
```

### `cortana.verifier` — Formal Verifier

```python
from cortana.verifier import Verifier

verifier = Verifier(memory)
result = verifier.verify_claim(claim)          # → VerificationResult
allowed = verifier.enforce_universe_gate(c, Modality.FACT)  # → bool
is_h, reason = verifier.detect_hallucination(claim)         # → (bool, str)
```

### `cortana.memory` — Structured Memory

```python
from cortana.memory import StructuredMemory

mem = StructuredMemory(storage_path="./memory")
mem.store_claim(claim, evidence)
claims = mem.retrieve_claims("database status", k=5)
facts = mem.get_verified_facts(subject="nginx")
```

### `cortana.values` — Value Drift Controller

```python
from cortana.values import ValueDriftController, RedLineTests

vdc = ValueDriftController(storage_path="./values")
print(vdc.red_line_tests)    # 5 invariants that can never be violated
print(vdc.current_profile)   # current weights + rationale
```

---

## Claims

See [CLAIMS.md](CLAIMS.md) for a rigorous, falsifiable list of what Cortana guarantees and what it does not.

---

## Related

- **[MoA](https://github.com/Yoder23/moa)** — The mixture-of-agents framework that provides the action IR and safety gate primitives Cortana is built on.
- **[CorticalSwarm](https://github.com/Yoder23/cortical-swarm)** — Windowed long-context continuity protocol with hash-validated handoffs, built to pair with Cortana.

---

## License

MIT
