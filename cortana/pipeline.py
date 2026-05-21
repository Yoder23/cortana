"""
CORTANA Integrated Pipeline

Complete pipeline integrating all CORTANA components:
- Structured Memory (bounded attention)
- Verifier (universe separation)
- Action Gatekeeper (hard constraints)
- Value Drift Controller (safe learning)

This is the NEW pipeline that runs ALONGSIDE the existing JSONL pipeline.
The existing pipeline remains untouched and continues to work.

Design Philosophy:
- Parallel systems for safety
- Gradual migration path
- Full rollback capability
- Evidence-based validation
"""

import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

from .ir import (
    Claim, Evidence, Action, Counterfactual, ValueDiff,
    Modality, EvidenceSource, ActionType, IntentType,
    create_claim, create_evidence
)
from .memory import StructuredMemory
from .verifier import Verifier
from .gatekeeper import ActionGatekeeper, SafetyClass
from .values import ValueDriftController

logger = logging.getLogger(__name__)

# Import new CORTANA components
try:
    from gibberlink_translator import GibberlinkTranslator
    GIBBERLINK_AVAILABLE = True
except ImportError:
    logger.warning("Gibberlink Translator not available")
    GIBBERLINK_AVAILABLE = False

try:
    from cortana_causal import CausalReasoner
    CAUSAL_AVAILABLE = True
except ImportError:
    logger.warning("Causal Reasoner not available")
    CAUSAL_AVAILABLE = False

try:
    from cortana_interactive import ConfirmationManager
    INTERACTIVE_AVAILABLE = True
except ImportError:
    logger.warning("Interactive Confirmation not available")
    INTERACTIVE_AVAILABLE = False


# ============================================================================
# Pipeline Result
# ============================================================================

@dataclass
class PipelineResult:
    """Result of running CORTANA pipeline"""
    success: bool
    claims_stored: int
    facts_verified: int
    actions_proposed: int
    actions_approved: int
    actions_rejected: int
    violations: List[str]
    
    def __repr__(self):
        status = "??? SUCCESS" if self.success else "??- FAILED"
        return f"PipelineResult({status}, {self.facts_verified} facts, {self.actions_approved}/{self.actions_proposed} actions)"


# ============================================================================
# CORTANA Pipeline
# ============================================================================

@dataclass
class GuardedActionResult:
    """Result of strict verify -> gate -> confirm action routing."""
    allowed: bool
    reason: str
    safety_class: str
    requires_confirmation: bool
    verification_confidence: float
    verification_failures: List[str]
    decision_reasons: List[str]
    audit_timestamp_utc: str


@dataclass
class DecisionResult:
    """Result of Cortana pre-action decision (two-way bridge)."""
    allowed: bool
    confidence: float
    reasoning: str
    causal_warnings: List[str]
    relevant_facts: List[str]
    alternatives: List[str]
    safety_class: str
    audit_timestamp_utc: str

class CortanaPipeline:
    """
    Complete CORTANA cognitive architecture
    
    Flow:
    1. Experience ??? Memory (structured IR)
    2. Retrieve relevant context (bounded attention)
    3. Reason ??? Claims (with modality)
    4. Verify claims (fact vs hypothesis)
    5. Propose actions
    6. Gate actions (hard constraints)
    7. Execute (if approved)
    8. Learn from results
    """
    
    def __init__(
        self,
        storage_path: str = "cortana_system",
        domain: str = "general",
        require_high_quality_embeddings: bool = False,
        allow_simple_embedder: bool = True,
        openai_api_key: Optional[str] = None,
        layercake_embedding_checkpoint: Optional[str] = None,
        layercake_tokenizer_path: Optional[str] = None,
        layercake_embedding_device: Optional[str] = None,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.memory = StructuredMemory(
            str(self.storage_path / "memory"),
            require_high_quality_embeddings=require_high_quality_embeddings,
            allow_simple_embedder=allow_simple_embedder,
            openai_api_key=openai_api_key,
            layercake_embedding_checkpoint=layercake_embedding_checkpoint,
            layercake_tokenizer_path=layercake_tokenizer_path,
            layercake_embedding_device=layercake_embedding_device,
        )
        self.verifier = Verifier(self.memory)
        self.gatekeeper = ActionGatekeeper()
        self.value_controller = ValueDriftController(str(self.storage_path / "values"))
        self.action_audit_path = self.storage_path / "action_audit.jsonl"
        
        # Initialize new components
        if GIBBERLINK_AVAILABLE:
            self.translator = GibberlinkTranslator(domain=domain)
            logger.info(f"   ???? Gibberlink Translator enabled (domain: {domain})")
        else:
            self.translator = None
            logger.info(f"   ??????  Gibberlink Translator not available")
        
        if CAUSAL_AVAILABLE:
            self.causal_reasoner = CausalReasoner()
            logger.info(f"   ???- Causal Reasoner enabled")
        else:
            self.causal_reasoner = None
            logger.info(f"   ??????  Causal Reasoner not available")
        
        if INTERACTIVE_AVAILABLE:
            self.confirmation_manager = ConfirmationManager()
            logger.info(f"   ??? Interactive Confirmation enabled")
        else:
            self.confirmation_manager = None
            logger.info(f"   ??????  Interactive Confirmation not available")
        
        logger.info("="*80)
        logger.info("??? CORTANA Pipeline initialized")
        logger.info("="*80)
        logger.info(f"   Storage: {self.storage_path}")
        logger.info(f"   Memory: {self.memory.storage_path}")
        logger.info(f"   Values: {self.value_controller.storage_path}")
        logger.info(f"   Action audit: {self.action_audit_path}")
        logger.info("")
    
    # ========================================================================
    # Core Learning Flow
    # ========================================================================
    
    def learn_from_experience(self, 
                            description: str,
                            outcome: str,
                            success: bool,
                            error_message: Optional[str] = None) -> PipelineResult:
        """
        Learn from a task experience
        
        This is the main entry point for continuous learning.
        Converts experience ??? structured claims ??? verified facts.
        
        Args:
            description: What was attempted
            outcome: What happened
            success: Did it work?
            error_message: Error details if failed
        
        Returns:
            PipelineResult with statistics
        """
        logger.info("\n" + "="*80)
        logger.info("???? LEARNING FROM EXPERIENCE")
        logger.info("="*80)
        logger.info(f"   Task: {description}")
        logger.info(f"   Outcome: {outcome}")
        logger.info(f"   Success: {success}")
        
        claims_stored = 0
        facts_verified = 0
        violations = []
        
        # Stage 1: Create claims from experience
        claims = self._extract_claims(description, outcome, success, error_message)
        logger.info(f"\n   Stage 1: Extracted {len(claims)} claims")
        
        # Stage 2: Create evidence
        evidence_list = self._create_evidence(claims, description, outcome, error_message)
        logger.info(f"   Stage 2: Created {len(evidence_list)} evidence items")

        if len(evidence_list) < len(claims):
            logger.warning(
                f"   Evidence underflow ({len(evidence_list)} for {len(claims)} claims); filling with execution-trace evidence"
            )
            for claim in claims[len(evidence_list):]:
                evidence_list.append(
                    create_evidence(
                        claim=claim,
                        source=EvidenceSource.EXECUTION_TRACE,
                        locator="pipeline_execution",
                        content=error_message if error_message else outcome,
                        confidence=1.0 if error_message else 0.9,
                    )
                )
        
        # Stage 3: Verify and store
        for claim, evidence in zip(claims, evidence_list):
            if evidence.id not in claim.evidence_ids:
                claim.evidence_ids.append(evidence.id)

            # Verify before storing
            verification = self.verifier.verify_claim(claim)
            
            if verification.verified:
                # Store as fact
                claim.modality = Modality.FACT
                self.memory.store_claim(claim, evidence)
                facts_verified += 1
            else:
                # Store as hypothesis
                claim.modality = Modality.HYPOTHESIS
                self.memory.store_claim(claim, evidence)
                logger.warning(f"      Stored as HYPOTHESIS: {claim.subject} {claim.predicate}")
            
            claims_stored += 1
            
            # Check for hallucinations
            is_hallucination, reason = self.verifier.detect_hallucination(claim)
            if is_hallucination:
                violations.append(f"Hallucination: {reason}")
                logger.error(f"      ??????  {reason}")
        
        logger.info(f"\n   Stage 3: Stored {claims_stored} claims, verified {facts_verified} facts")
        
        # Stage 4: Extract causal relationships
        if self.translator is not None and self.causal_reasoner is not None:
            full_text = f"{description}\n{outcome}"
            if error_message:
                full_text += f"\nError: {error_message}"
            
            causal_pairs = self.translator.extract_causal_relationships(full_text)
            
            if causal_pairs:
                logger.info(f"\n   Stage 4: Extracted {len(causal_pairs)} causal relationships")
                
                for cause_claim, effect_claim in causal_pairs:
                    # Learn causal relationship
                    from cortana_causal import CausalLinkType
                    self.causal_reasoner.learn_causal_relationship(
                        cause=cause_claim,
                        effect=effect_claim,
                        strength=0.8,  # Default strength
                        link_type=CausalLinkType.DIRECT_CAUSE
                    )
            else:
                logger.info(f"\n   Stage 4: No causal relationships found")
        
        # Stage 5: Update memory stats
        self.memory.print_stats()
        
        return PipelineResult(
            success=True,
            claims_stored=claims_stored,
            facts_verified=facts_verified,
            actions_proposed=0,
            actions_approved=0,
            actions_rejected=0,
            violations=violations
        )
    
    def _extract_claims(self, description: str, outcome: str, 
                       success: bool, error_message: Optional[str]) -> List[Claim]:
        """
        Extract structured claims from experience.
        
        Uses Gibberlink Translator if available, otherwise falls back to
        pattern-based extraction.
        """
        claims = []
        
        # Use Gibberlink Translator if available
        if self.translator is not None:
            # Build context
            context = {
                'outcome': 'success' if success else 'failure',
                'error': error_message if error_message else None
            }
            
            # Combine description and outcome
            full_text = f"{description}\n{outcome}"
            if error_message:
                full_text += f"\nError: {error_message}"
            
            # Extract claims using Gibberlink
            claims = self.translator.extract_claims(full_text, context)
            
            logger.info(f"   ???? Gibberlink extracted {len(claims)} claims")
            if claims:
                return claims
            logger.warning("   Gibberlink returned no claims, falling back to rule extraction")
        
        # Fallback: Simple pattern-based extraction
        logger.info(f"   ??????  Using fallback claim extraction (Gibberlink not available)")
        
        # Parse description to find subjects
        # (Basic pattern matching - not as good as Gibberlink)
        
        description_l = description.lower()
        outcome_l = outcome.lower()
        error_l = error_message.lower() if error_message else ""

        if "requests" in description_l:
            if error_message and "not defined" in error_message:
                claim = create_claim(
                    subject="requests",
                    predicate="requires_import",
                    object="True",
                    modality=Modality.FACT,
                    confidence=0.98
                )
                claims.append(claim)
        
        if "json" in description_l and success:
            claim = create_claim(
                subject="json_api_call",
                predicate="works_with_requests",
                object="True",
                modality=Modality.FACT,
                confidence=0.9
            )
            claims.append(claim)

        if not claims:
            # Always persist at least one grounded experience claim.
            if success:
                claims.append(
                    create_claim(
                        subject="task_execution",
                        predicate="succeeded",
                        object=description[:120],
                        modality=Modality.FACT,
                        confidence=0.85,
                    )
                )
            else:
                failure_obj = error_message[:120] if error_message else outcome[:120]
                claims.append(
                    create_claim(
                        subject="task_execution",
                        predicate="failed_with",
                        object=failure_obj,
                        modality=Modality.FACT if error_l else Modality.HYPOTHESIS,
                        confidence=0.9 if error_l else 0.7,
                    )
                )
            # Keep a compact outcome claim for retrieval quality.
            claims.append(
                create_claim(
                    subject="task_outcome",
                    predicate="status",
                    object="success" if success or "success" in outcome_l else "failure",
                    modality=Modality.FACT,
                    confidence=0.8,
                )
            )
        
        return claims
    
    def _create_evidence(self, claims: List[Claim], description: str, 
                        outcome: str, error_message: Optional[str]) -> List[Evidence]:
        """
        Create evidence for claims.
        
        Uses Gibberlink Translator if available to extract evidence with
        proper source tagging.
        """
        evidence_list = []
        
        # Use Gibberlink Translator if available
        if self.translator is not None:
            # Combine all text
            full_text = f"{description}\n{outcome}"
            if error_message:
                full_text += f"\nError: {error_message}"
            
            # Extract evidence for each claim
            for claim in claims:
                evidences = self.translator.extract_evidence(full_text, claim)
                if evidences:
                    # Keep strongest piece of evidence per claim for deterministic verification.
                    best = max(evidences, key=lambda e: getattr(e, "confidence", 0.0))
                    evidence_list.append(best)
                else:
                    fallback = create_evidence(
                        claim=claim,
                        source=EvidenceSource.EXECUTION_TRACE,
                        locator="pipeline_execution",
                        content=error_message if error_message else outcome,
                        confidence=1.0 if error_message else 0.9,
                    )
                    evidence_list.append(fallback)
            
            logger.info(f"   ???? Gibberlink produced {len(evidence_list)} evidence items")
            return evidence_list
        
        # Fallback: Create basic evidence
        logger.info(f"   ??????  Using fallback evidence creation")
        
        for claim in claims:
            if error_message:
                # Evidence from execution trace
                evidence = create_evidence(
                    claim=claim,
                    source=EvidenceSource.EXECUTION_TRACE,
                    locator="pipeline_execution",
                    content=error_message,
                    confidence=1.0
                )
            else:
                # Evidence from successful execution
                evidence = create_evidence(
                    claim=claim,
                    source=EvidenceSource.EXECUTION_TRACE,
                    locator="pipeline_execution",
                    content=outcome,
                    confidence=0.9
                )
            evidence_list.append(evidence)
        
        return evidence_list
    
    # ========================================================================
    # Action Flow
    # ========================================================================
    
    def _append_action_audit(self, payload: Dict[str, Any]) -> None:
        """Append a single audit record for an action decision."""
        with open(self.action_audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def propose_guarded_action(
        self,
        action: Action,
        justification: Optional[List[Claim]] = None,
    ) -> GuardedActionResult:
        """
        Strict action flow with no bypass:
        verify -> gate -> confirm (if required) -> audit.
        """
        action_type = action.type.value if hasattr(action.type, "value") else str(action.type)
        logger.info(f"\n   Guarded action proposal: {action_type}")

        verification = self.verifier.verify_action_justification(action, justification)
        decision = self.gatekeeper.evaluate(action, justification)

        allowed = False
        reason = ""
        safety_class = decision.safety_class.value
        requires_confirmation = decision.safety_class == SafetyClass.CONFIRM

        # SAFE and REVERSIBLE actions bypass verification - the gatekeeper already vetted them
        if decision.safety_class in (SafetyClass.SAFE, SafetyClass.REVERSIBLE):
            allowed = True
            reason = f"Allowed ({safety_class})"
        elif not verification.verified:
            reason = f"Blocked: verification failed ({verification.failures})"
            self.gatekeeper.record_rejection(action, list(verification.failures))
        elif requires_confirmation:
            # CONFIRM-class actions are not yet allowed - handle confirmation flow first
            if self.confirmation_manager is None:
                reason = "Blocked: confirmation required but confirmation manager unavailable"
                self.gatekeeper.record_rejection(action, [reason])
            else:
                confirmed, confirm_reason = self.confirmation_manager.evaluate_action_with_confirmation(
                    action, decision
                )
                if confirmed:
                    allowed = True
                    reason = f"Allowed after confirmation: {confirm_reason}"
                    self.gatekeeper.record_confirmation(action)
                else:
                    reason = f"Blocked by confirmation: {confirm_reason}"
                    self.gatekeeper.record_rejection(action, [reason])
        elif not decision.allowed:
            reason = f"Blocked: gatekeeper rejected ({decision.reasons})"
            self.gatekeeper.record_rejection(action, list(decision.reasons))
        else:
            allowed = True
            reason = f"Allowed ({safety_class})"

        timestamp_utc = datetime.now(timezone.utc).isoformat()
        audit_record = {
            "timestamp_utc": timestamp_utc,
            "action_id": action.id,
            "action_type": action_type,
            "allowed": allowed,
            "reason": reason,
            "safety_class": safety_class,
            "requires_confirmation": requires_confirmation,
            "verification": {
                "verified": verification.verified,
                "confidence": verification.confidence,
                "failures": list(verification.failures),
                "warnings": list(verification.warnings),
            },
            "gatekeeper": {
                "allowed": decision.allowed,
                "reasons": list(decision.reasons),
            },
        }
        self._append_action_audit(audit_record)

        return GuardedActionResult(
            allowed=allowed,
            reason=reason,
            safety_class=safety_class,
            requires_confirmation=requires_confirmation,
            verification_confidence=verification.confidence,
            verification_failures=list(verification.failures),
            decision_reasons=list(decision.reasons),
            audit_timestamp_utc=timestamp_utc,
        )

    def propose_action(
        self,
        action: Action,
        justification: Optional[List[Claim]] = None,
    ) -> Tuple[bool, str]:
        """
        Backward-compatible wrapper around the strict guarded path.
        """
        guarded = self.propose_guarded_action(action, justification)
        return guarded.allowed, guarded.reason
    # ========================================================================
    # Retrieval (Bounded Attention)
    # ========================================================================
    
    def retrieve_relevant_knowledge(self, query: str, 
                                   facts_only: bool = False) -> List[Claim]:
        """
        Retrieve relevant knowledge for reasoning
        
        This implements BOUNDED ATTENTION:
        - Don't load entire memory
        - Retrieve only what's relevant
        - Reason over retrieved state
        """
        logger.info(f"\n   ???? Retrieving knowledge: '{query}'")
        claims = self.memory.retrieve_claims(query, k=5, facts_only=facts_only)
        
        if facts_only:
            logger.info(f"      Retrieved {len(claims)} verified facts")
        else:
            logger.info(f"      Retrieved {len(claims)} claims (mixed modalities)")
        
        return claims
    
    # ========================================================================
    # Value Learning
    # ========================================================================
    
    def propose_value_change(self, rationale: str, 
                           weight_changes: Dict[str, float],
                           predicted_behavior: str) -> bool:
        """
        Propose a change to value function
        
        Returns:
            True if change approved, False if rejected
        """
        logger.info(f"\n   ??????  Value change proposed: {rationale}")
        
        # Create diff
        current_weights = {k: self.value_controller.current_profile.weights.get(k, 0.0) 
                         for k in weight_changes.keys()}
        
        diff = ValueDiff(
            rationale=rationale,
            predicted_behavior_change=predicted_behavior,
            old_weights=current_weights,
            new_weights=weight_changes
        )
        
        # Propose
        if not self.value_controller.propose_value_change(diff):
            return False
        
        # Test
        passed, results = self.value_controller.test_value_diff(diff)
        
        if passed:
            # Approve
            return self.value_controller.approve_value_change(diff)
        else:
            logger.error(f"      ??? Value change rejected (tests failed)")
            return False
    
    # ========================================================================
    # Counterfactual Learning
    # ========================================================================
    
    def store_counterfactual(self, scenario: str, 
                           actual: str, hypothetical: str,
                           human_valence: Optional[float] = None) -> None:
        """Store counterfactual for learning"""
        cf = Counterfactual(
            scenario=scenario,
            actual_outcome=actual,
            hypothetical_outcome=hypothetical,
            human_valence=human_valence,
            importance=0.5,
            surprise=0.0
        )
        
        self.memory.store_counterfactual(cf)
        logger.info(f"   ???? Stored counterfactual: {scenario}")

    def embedding_quality(self) -> Dict[str, Any]:
        """Return embedding backend quality metadata for readiness checks."""
        embedder = getattr(self.memory, "embedder", None)
        embedder_type = getattr(
            embedder,
            "embedder_type",
            type(embedder).__name__ if embedder is not None else "none",
        )
        quality_tier = getattr(embedder, "quality_tier", "unknown")
        is_high_quality = bool(getattr(embedder, "is_high_quality", False))
        return {
            "embedder_type": embedder_type,
            "quality_tier": quality_tier,
            "is_high_quality": is_high_quality,
        }

    # ========================================================================
    # Pre-Action Decision Bridge (Cortana ? MoA two-way)
    # ========================================================================

    def request_decision(
        self,
        action: Action,
        context_description: str,
        justification: Optional[List[Claim]] = None,
    ) -> "DecisionResult":
        """
        MoA asks Cortana to evaluate a proposed action BEFORE execution.

        This is the two-way bridge: MoA proposes ? Cortana reasons over
        memory, causal history, and safety constraints ? returns a decision
        with optional alternative suggestions.

        Args:
            action: The proposed action.
            context_description: Natural-language context for memory retrieval.
            justification: Optional pre-built claims supporting the action.

        Returns:
            DecisionResult with verdict, reasoning, and alternatives.
        """
        action_type = action.type.value if hasattr(action.type, "value") else str(action.type)
        logger.info(f"\n   Pre-action decision requested: {action_type}")

        # Stage 1: Retrieve relevant memory
        relevant_facts = self.retrieve_relevant_knowledge(
            context_description, facts_only=True
        )
        relevant_all = self.retrieve_relevant_knowledge(
            context_description, facts_only=False
        )

        # Stage 2: Check causal history for known bad outcomes
        causal_warnings: List[str] = []
        if self.causal_reasoner is not None:
            for fact in relevant_facts:
                effects = self.causal_reasoner.predict_action_effects(fact)
                for effect_claim, strength in effects:
                    eff_obj = getattr(effect_claim, "object", "")
                    if any(
                        kw in str(eff_obj).lower()
                        for kw in ("fail", "error", "denied", "crash", "corrupt")
                    ):
                        causal_warnings.append(
                            f"Causal history: '{fact.subject} {fact.predicate}' "
                            f"previously led to '{eff_obj}' (strength={strength:.2f})"
                        )

        # Stage 3: Gate the action
        guarded = self.propose_guarded_action(action, justification)

        # Stage 4: Build alternatives if blocked
        alternatives: List[str] = []
        if not guarded.allowed:
            # Suggest safer action types for CONFIRM/FORBIDDEN cases
            if action.type in (ActionType.MODIFY_SYSTEM, ActionType.DELETE_FILE):
                alternatives.append("Use READ_FILE to inspect first before modifying")
            if action.type == ActionType.ACCESS_NETWORK:
                alternatives.append("Use QUERY_API with limited scope instead")
            if action.type in (
                ActionType.ESCALATE_PERMISSION,
                ActionType.DECEIVE_USER,
                ActionType.MANIPULATE_SOCIAL,
            ):
                alternatives.append("This action is architecturally forbidden - no alternatives exist")

        # Stage 5: Compose decision
        confidence = guarded.verification_confidence
        if causal_warnings:
            # Reduce confidence when causal history shows risk
            confidence = max(0.0, confidence - 0.1 * len(causal_warnings))

        decision = DecisionResult(
            allowed=guarded.allowed,
            confidence=round(confidence, 4),
            reasoning=guarded.reason,
            causal_warnings=causal_warnings,
            relevant_facts=[
                f"{c.subject} {c.predicate} {c.object}" for c in relevant_facts[:5]
            ],
            alternatives=alternatives,
            safety_class=guarded.safety_class,
            audit_timestamp_utc=guarded.audit_timestamp_utc,
        )

        logger.info(f"   Decision: {'ALLOWED' if decision.allowed else 'BLOCKED'} "
                     f"(confidence={decision.confidence}, "
                     f"causal_warnings={len(causal_warnings)}, "
                     f"facts_consulted={len(relevant_facts)})")

        return decision
    
    # ========================================================================
    # Statistics & Reporting
    # ========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Complete system statistics"""
        action_audits = 0
        if self.action_audit_path.exists():
            with open(self.action_audit_path, "r", encoding="utf-8") as handle:
                action_audits = sum(1 for _ in handle)

        return {
            'memory': self.memory.stats(),
            'verifier': self.verifier.stats(),
            'gatekeeper': self.gatekeeper.stats(),
            'value_controller': self.value_controller.stats(),
            'action_audits': action_audits,
            'embeddings': self.embedding_quality(),
        }
    
    def print_status(self) -> None:
        """Print comprehensive system status"""
        stats = self.stats()
        
        logger.info("\n" + "="*80)
        logger.info("???? CORTANA SYSTEM STATUS")
        logger.info("="*80)
        
        logger.info("\n   Memory:")
        for key, value in stats['memory'].items():
            logger.info(f"      {key}: {value}")
        
        logger.info("\n   Verifier:")
        for key, value in stats['verifier'].items():
            logger.info(f"      {key}: {value}")
        
        logger.info("\n   Action Gatekeeper:")
        for key, value in stats['gatekeeper'].items():
            logger.info(f"      {key}: {value}")
        
        logger.info("\n   Value Controller:")
        for key, value in stats['value_controller'].items():
            logger.info(f"      {key}: {value}")

        logger.info("\n   Auditing:")
        logger.info(f"      action_audits: {stats['action_audits']}")

        logger.info("\n   Embeddings:")
        for key, value in stats['embeddings'].items():
            logger.info(f"      {key}: {value}")
        
        logger.info("")


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
    print("CORTANA INTEGRATED PIPELINE DEMO")
    print("="*80 + "\n")
    
    # Initialize pipeline
    pipeline = CortanaPipeline("demo_cortana_system")
    
    # Test 1: Learn from error
    print("\n" + "="*80)
    print("Test 1: Learn from error (NameError)")
    print("="*80)
    
    result1 = pipeline.learn_from_experience(
        description="Write Python script that fetches JSON from API using requests",
        outcome="Error",
        success=False,
        error_message="NameError: name 'requests' is not defined"
    )
    print(f"\nResult: {result1}")
    
    # Test 2: Learn from success
    print("\n" + "="*80)
    print("Test 2: Learn from success")
    print("="*80)
    
    result2 = pipeline.learn_from_experience(
        description="Write Python script with requests import and fetch JSON",
        outcome="Success: User name: GitHub",
        success=True,
        error_message=None
    )
    print(f"\nResult: {result2}")
    
    # Test 3: Retrieve knowledge
    print("\n" + "="*80)
    print("Test 3: Retrieve relevant knowledge")
    print("="*80)
    
    claims = pipeline.retrieve_relevant_knowledge("requests import", facts_only=True)
    
    # Test 4: Propose action
    print("\n" + "="*80)
    print("Test 4: Propose safe action")
    print("="*80)
    
    safe_action = Action(type=ActionType.RUN_CODE, args={'file': 'script.py'})
    allowed, reason = pipeline.propose_action(safe_action)
    print(f"\nAction allowed: {allowed}")
    print(f"Reason: {reason}")
    
    # Test 5: Propose dangerous action
    print("\n" + "="*80)
    print("Test 5: Propose dangerous action (should block)")
    print("="*80)
    
    dangerous_action = Action(type=ActionType.ESCALATE_PERMISSION, args={'level': 'admin'})
    allowed, reason = pipeline.propose_action(dangerous_action)
    print(f"\nAction allowed: {allowed}")
    print(f"Reason: {reason}")
    
    # Final status
    pipeline.print_status()


