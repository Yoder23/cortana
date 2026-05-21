# Cortana — Claims

A precise, falsifiable list of what Cortana guarantees and what it does not.

---

## What We Claim (Verifiable)

### C1: SIMULATION claims cannot become FACT via `enforce_universe_gate()`

**Claim:** `Verifier.enforce_universe_gate(claim, Modality.FACT)` returns `False` when `claim.modality == Modality.SIMULATION`, regardless of the claim's confidence score or evidence count.

**How to verify:**
```python
from cortana import Verifier, StructuredMemory, Modality, create_claim, create_evidence, EvidenceSource

mem = StructuredMemory("./tmp_mem")
v = Verifier(mem)
sim = create_claim("attacker", "gained", "root_access", Modality.SIMULATION, confidence=0.9999)
assert v.enforce_universe_gate(sim, Modality.FACT) is False
```

**Why:** `enforce_universe_gate()` is a 3-line Python function with a hardcoded `return False` branch for non-HYPOTHESIS inputs targeting FACT. There is no path through the code that returns `True` for a SIMULATION claim.

---

### C2: Three action types are permanently blocked

**Claim:** `ActionGatekeeper.evaluate()` returns `ActionDecision(allowed=False, safety_class=SafetyClass.FORBIDDEN)` for any action of type `ESCALATE_PERMISSION`, `DECEIVE_USER`, or `MANIPULATE_SOCIAL`.

**How to verify:**
```python
from cortana import ActionGatekeeper, HardConstraints, ActionType, Action, ActionMeta

gate = ActionGatekeeper()
for atype in HardConstraints.FORBIDDEN_ACTIONS:
    d = gate.evaluate(Action(type=atype, args={}, metadata=ActionMeta()))
    assert not d.allowed
```

**Why:** `HardConstraints.FORBIDDEN_ACTIONS` is a Python `set` constant. `HardConstraints.check()` tests `action.type in cls.FORBIDDEN_ACTIONS` before any other logic. No prompting can change this — it's a `set` literal in source code.

---

### C3: Hard thresholds for blast radius (0.7) and irreversibility (0.8)

**Claim:** Any action with `ActionMeta.blast_radius > 0.7` or `ActionMeta.irreversibility_score > 0.8` is blocked by the gatekeeper.

**Verification:** See `tests/test_gatekeeper.py::TestHardConstraints`.

---

### C4: High-confidence claims without evidence are flagged as hallucinations

**Claim:** `Verifier.detect_hallucination(claim)` returns `(True, reason)` when `claim.confidence > 0.8` and `len(claim.evidence_ids) == 0`.

**Verification:** See `tests/test_verifier.py::TestDetectHallucination`.

---

### C5: Red-line value tests define permanent behavioral invariants

**Claim:** `RedLineTests.get_default_red_lines()` returns at least 5 `ValueTest` objects covering: no deception, no permission escalation, preserve corrigibility, no goal hijacking, no resource acquisition.

**Verification:** See `tests/test_values.py::TestRedLineTests`.

---

### C6: Gate throughput > 100k decisions/second

**Claim:** On commodity hardware (2024 laptop), `ActionGatekeeper.evaluate()` achieves > 100,000 decisions/second.

**Measured:** 370,000+ decisions/second on RTX 3080 Laptop host.

**How to verify:**
```bash
python verify_cortana.py   # Check section 10
```

---

## What We Do NOT Claim

### X1: Cortana does not guarantee runtime model behavior

Cortana's safety layer wraps an AI system — it does not control the model's outputs. A model could still generate harmful text that is never routed through the action gatekeeper. Cortana's guarantees apply to **actions** (structured `Action` objects), not to free-text generation.

### X2: Cortana does not detect all hallucinations

`detect_hallucination()` catches the pattern of high-confidence claims without evidence. It does not catch factually incorrect claims that happen to have (fake) evidence attached, or hallucinations that are stated with low confidence.

### X3: Cortana's safety does not work if the gatekeeper is bypassed

Cortana's safety guarantees apply only to actions routed through `ActionGatekeeper.evaluate()` or `CortanaPipeline.propose_action()`. If code directly executes actions without going through these methods, Cortana provides no protection.

### X4: `StructuredMemory` retrieval quality depends on embedder

The `retrieve_claims()` method uses cosine similarity over embeddings. With the `SimpleEmbedder` fallback (character-level histograms), retrieval quality is low. High-quality retrieval requires `sentence-transformers` (`pip install sentence-transformers`) or a trained LayerCake checkpoint.

### X5: Universe separation only covers `enforce_universe_gate()`

The Modality field on `Claim` objects is mutable. A malicious or buggy caller could set `claim.modality = Modality.FACT` directly without going through `enforce_universe_gate()`. Cortana's architectural guarantee applies to the API surface, not to direct attribute mutation.

---

## Benchmark Context

All benchmark numbers (`verify_cortana.py`, README) were measured on:
- CPU: Intel Core i7-11800H  
- GPU: NVIDIA RTX 3080 Laptop  
- Python 3.10, numpy 1.26, no GPU usage for these benchmarks

Gate throughput is pure Python with dataclass evaluation — no GPU, no neural networks.
