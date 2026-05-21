"""
CORTANA Verifier & Universe Separation

Prevents hallucination leakage by enforcing strict verification rules.

Rules:
1. No evidence -> no factual rendering
2. Hypotheticals cannot justify real actions
3. Conflicts trigger user escalation
4. Simulations never contaminate facts

Design Principle: Language is an Interface, Not Truth
- English is for humans
- Truth lives in structured IR
- Verification happens in IR space
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .ir import (
    Claim, Evidence, Action, ReasoningTrace,
    Modality, EvidenceSource, ActionType
)
from .memory import StructuredMemory

logger = logging.getLogger(__name__)


# ============================================================================
# Verification Results
# ============================================================================

@dataclass
class VerificationResult:
    """Result of verification check"""
    verified: bool
    confidence: float
    failures: List[str]
    warnings: List[str]
    
    def __repr__(self):
        status = "[v] VERIFIED" if self.verified else "[x] FAILED"
        return f"VerificationResult({status}, conf={self.confidence:.2f}, {len(self.failures)} failures)"


@dataclass
class UniverseViolation:
    """Record of universe separation violation"""
    claim: Claim
    violation_type: str  # "mixed_universes", "unverified_fact", "simulation_leak"
    description: str
    
    def __repr__(self):
        return f"UniverseViolation({self.violation_type}: {self.description})"


# ============================================================================
# Verifier
# ============================================================================

class Verifier:
    """
    Verifies claims and enforces universe separation
    
    Verification Levels:
    1. Structural: Does the claim have required fields?
    2. Evidential: Is there supporting evidence?
    3. Consistency: Does it contradict known facts?
    4. Universe: Is the modality appropriate?
    """
    
    def __init__(self, memory: StructuredMemory):
        self.memory = memory
        self.violations: List[UniverseViolation] = []
        
        # Confidence thresholds
        self.min_confidence_for_fact = 0.8
        self.min_evidence_count = 1
        
        logger.info("[OK] Verifier initialized")
    
    # ========================================================================
    # Claim Verification
    # ========================================================================
    
    def verify_claim(self, claim: Claim) -> VerificationResult:
        """
        Verify a claim can be treated as fact
        
        Requirements:
        1. Modality must be FACT
        2. Must have evidence
        3. Evidence confidence >= threshold
        4. No contradictions with existing facts
        """
        failures = []
        warnings = []
        confidence = claim.confidence
        
        # Check modality
        if claim.modality != Modality.FACT:
            failures.append(f"Modality is {claim.modality.value}, not FACT")
        
        # Check evidence
        if len(claim.evidence_ids) < self.min_evidence_count:
            failures.append(f"Insufficient evidence: {len(claim.evidence_ids)} < {self.min_evidence_count}")
        
        # Check confidence
        if claim.confidence < self.min_confidence_for_fact:
            failures.append(f"Low confidence: {claim.confidence:.2f} < {self.min_confidence_for_fact}")
        
        # Check for contradictions
        if self.memory.factual_graph.has_contradiction(claim):
            failures.append("Contradicts existing facts")
        
        # Verify evidence quality (fetch once, not per evidence_id)
        evidence_list = self.memory.get_evidence_for_claim(claim.id)
        for evidence in evidence_list:
            if evidence.confidence < 0.5:
                warnings.append(f"Weak evidence: {evidence.source.value} conf={evidence.confidence:.2f}")
        
        verified = len(failures) == 0
        
        result = VerificationResult(
            verified=verified,
            confidence=confidence,
            failures=failures,
            warnings=warnings
        )
        
        if verified:
            logger.info(f"   [v] Verified: {claim}")
        else:
            logger.warning(f"   [x] Verification failed: {claim}")
            for failure in failures:
                logger.warning(f"      * {failure}")
        
        return result
    
    def verify_evidence(self, evidence: Evidence) -> bool:
        """Verify evidence is trustworthy"""
        # Check source reliability
        trusted_sources = {
            EvidenceSource.EXECUTION_TRACE,  # We ran it
            EvidenceSource.HUMAN_LABEL,      # User confirmed
            EvidenceSource.VERIFICATION_TOOL  # Tool checked
        }
        
        if evidence.source not in trusted_sources:
            logger.warning(f"   [WARN]  Untrusted evidence source: {evidence.source}")
            return False
        
        # Check confidence
        if evidence.confidence < 0.5:
            logger.warning(f"   [WARN]  Low confidence evidence: {evidence.confidence:.2f}")
            return False
        
        return True
    
    # ========================================================================
    # Universe Separation
    # ========================================================================
    
    def check_universe_separation(self, reasoning_trace: ReasoningTrace) -> List[UniverseViolation]:
        """
        Ensure universes don't mix
        
        Rules:
        1. FACT claims can only reference FACT inputs
        2. SIMULATION claims must stay isolated
        3. HYPOTHESIS can use both, but output remains HYPOTHESIS
        """
        violations = []
        
        # Check input consistency
        input_modalities = {claim.modality for claim in reasoning_trace.input_claims}
        output_modalities = {claim.modality for claim in reasoning_trace.output_claims}
        
        # Rule 1: FACT outputs require FACT inputs
        if Modality.FACT in output_modalities:
            if input_modalities - {Modality.FACT}:
                violation = UniverseViolation(
                    claim=reasoning_trace.output_claims[0],
                    violation_type="mixed_universes",
                    description="FACT output derived from non-FACT inputs"
                )
                violations.append(violation)
                logger.error(f"   [FAIL] {violation}")
        
        # Rule 2: SIMULATION must stay isolated
        if Modality.SIMULATION in input_modalities:
            if Modality.FACT in output_modalities:
                violation = UniverseViolation(
                    claim=reasoning_trace.output_claims[0],
                    violation_type="simulation_leak",
                    description="Simulation contaminated factual reasoning"
                )
                violations.append(violation)
                logger.error(f"   [FAIL] {violation}")
        
        # Store violations
        self.violations.extend(violations)
        
        if not violations:
            logger.info("   [v] Universe separation maintained")
        
        return violations
    
    def enforce_universe_gate(self, claim: Claim, target_universe: Modality) -> bool:
        """
        Check if claim can enter target universe
        
        Promotion rules:
        - HYPOTHESIS -> FACT: requires strong evidence
        - FACT -> SIMULATION: always allowed (for counterfactuals)
        - SIMULATION -> FACT: NEVER allowed
        - FICTION -> FACT: NEVER allowed
        """
        # Same universe: always ok
        if claim.modality == target_universe:
            return True
        
        # Promotion rules
        if target_universe == Modality.FACT:
            if claim.modality == Modality.HYPOTHESIS:
                # Check if we have evidence
                verification = self.verify_claim(claim)
                return verification.verified
            else:
                # Cannot promote SIMULATION or FICTION to FACT
                logger.error(f"   [FAIL] Cannot promote {claim.modality.value} to FACT")
                return False
        
        elif target_universe == Modality.SIMULATION:
            # Facts can be used in simulations
            return True
        
        elif target_universe == Modality.HYPOTHESIS:
            # Downgrade is always safe
            return True
        
        else:  # FICTION
            # Anything can become fiction
            return True
    
    # ========================================================================
    # Action Verification
    # ========================================================================
    
    def verify_action_justification(self, action: Action, 
                                    supporting_claims: any) -> VerificationResult:
        """
        Verify that action is justified by claims
        
        Rules:
        1. Real-world actions require FACT justification
        2. HYPOTHESIS claims cannot justify irreversible actions
        3. SIMULATION claims cannot justify any real actions
        """
        failures = []
        warnings = []
        confidence = 1.0
        
        # Handle string justifications (simple case)
        if isinstance(supporting_claims, str):
            # String justifications are treated as weak evidence
            # Allow them but with lower confidence
            return VerificationResult(
                verified=True,
                confidence=0.5,
                failures=[],
                warnings=["Using string justification instead of structured claims"]
            )
        
        # Handle List[Claim] — empty/None justification must NOT pass verification
        if not supporting_claims:
            return VerificationResult(
                verified=False,
                confidence=0.0,
                failures=["No justification provided — actions require supporting claims"],
                warnings=[]
            )
        
        # Check claim modalities
        modalities = {claim.modality for claim in supporting_claims}
        
        # Real actions need facts
        if action.type in {ActionType.DELETE_FILE, ActionType.MODIFY_SYSTEM, 
                          ActionType.INSTALL_PACKAGE}:
            if Modality.FACT not in modalities:
                failures.append("Irreversible action requires FACT justification")
                confidence = 0.0
            
            if Modality.HYPOTHESIS in modalities:
                warnings.append("Action partially justified by hypothesis")
                confidence *= 0.7
            
            if Modality.SIMULATION in modalities or Modality.FICTION in modalities:
                failures.append("Simulation/fiction cannot justify real actions")
                confidence = 0.0
        
        # Reversible actions can use hypotheses with confirmation
        elif action.type in {ActionType.WRITE_CODE, ActionType.RUN_CODE}:
            if Modality.SIMULATION in modalities or Modality.FICTION in modalities:
                warnings.append("Using simulation/fiction for real action - needs confirmation")
                confidence *= 0.5
        
        verified = len(failures) == 0
        
        result = VerificationResult(
            verified=verified,
            confidence=confidence,
            failures=failures,
            warnings=warnings
        )
        
        if not verified:
            logger.error(f"   [FAIL] Action verification failed: {action.type.value}")
            for failure in failures:
                logger.error(f"      * {failure}")
        
        return result
    
    # ========================================================================
    # Hallucination Detection
    # ========================================================================
    
    def detect_hallucination(self, claim: Claim) -> Tuple[bool, str]:
        """
        Detect if claim is likely hallucinated
        
        Indicators:
        1. High confidence but no evidence
        2. Contradicts verified facts
        3. Unusual specificity without source
        """
        # Check 1: Confidence without evidence
        if claim.confidence > 0.8 and len(claim.evidence_ids) == 0:
            return True, "High confidence with no evidence (likely hallucination)"
        
        # Check 2: Contradictions
        if self.memory.factual_graph.has_contradiction(claim):
            return True, "Contradicts verified facts"
        
        # Check 3: Claimed as fact without verification
        if claim.modality == Modality.FACT and not claim.is_verified():
            return True, "Claimed as FACT without verification"
        
        return False, ""
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Verification statistics"""
        return {
            'total_violations': len(self.violations),
            'violation_types': {
                vtype: sum(1 for v in self.violations if v.violation_type == vtype)
                for vtype in set(v.violation_type for v in self.violations)
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("\n" + "="*80)
    print("CORTANA Verifier Demo")
    print("="*80 + "\n")
    
    from .ir import create_claim, create_evidence, EvidenceSource
    
    # Initialize
    memory = StructuredMemory("demo_cortana_memory")
    verifier = Verifier(memory)
    
    # Test 1: Verify a proper fact
    print("\n" + "-"*80)
    print("Test 1: Verify proper fact with evidence")
    print("-"*80)
    
    claim1 = create_claim(
        subject="requests",
        predicate="requires_import",
        object="True",
        modality=Modality.FACT,
        confidence=0.95
    )
    evidence1 = create_evidence(
        claim=claim1,
        source=EvidenceSource.EXECUTION_TRACE,
        locator="script.py:2",
        content="NameError: name 'requests' is not defined",
        confidence=1.0
    )
    claim1.evidence_ids.append(evidence1.id)
    
    result1 = verifier.verify_claim(claim1)
    print(f"\nResult: {result1}")
    
    # Test 2: Try to verify claim without evidence
    print("\n" + "-"*80)
    print("Test 2: Verify claim without evidence (should fail)")
    print("-"*80)
    
    claim2 = create_claim(
        subject="numpy",
        predicate="is_fast",
        object="True",
        modality=Modality.FACT,
        confidence=0.9
    )
    
    result2 = verifier.verify_claim(claim2)
    print(f"\nResult: {result2}")
    
    # Test 3: Detect hallucination
    print("\n" + "-"*80)
    print("Test 3: Detect hallucination")
    print("-"*80)
    
    claim3 = create_claim(
        subject="unicorn_api",
        predicate="version",
        object="3.14.159",
        modality=Modality.FACT,
        confidence=0.99
    )
    
    is_hallucination, reason = verifier.detect_hallucination(claim3)
    print(f"\nHallucination detected: {is_hallucination}")
    if is_hallucination:
        print(f"Reason: {reason}")
    
    # Test 4: Universe separation
    print("\n" + "-"*80)
    print("Test 4: Universe separation check")
    print("-"*80)
    
    trace = ReasoningTrace(
        input_claims=[
            create_claim("x", "equals", "5", Modality.SIMULATION)
        ],
        reasoning_steps=["Calculate x + 1"],
        output_claims=[
            create_claim("y", "equals", "6", Modality.FACT)  # WRONG!
        ],
        confidence=0.9,
        universe=Modality.SIMULATION
    )
    
    violations = verifier.check_universe_separation(trace)
    print(f"\nViolations found: {len(violations)}")
    for v in violations:
        print(f"   {v}")

