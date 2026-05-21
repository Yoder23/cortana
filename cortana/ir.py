"""
CORTANA Intermediate Representation (IR)

Structured IR objects for memory, reasoning, verification, and action control.
Eliminates hallucination leakage and enables formal verification.

Design Principle: Memory ≠ Belief
- Everything may be stored
- Only verified information may be treated as fact
"""

from dataclasses import dataclass, field
from typing import Literal, Dict, List, Optional, Any, Set
from enum import Enum
from datetime import datetime
import uuid


# ============================================================================
# Core IR Enums
# ============================================================================

class Modality(Enum):
    """Universe separation: fact, hypothesis, simulation, fiction"""
    FACT = "fact"              # Requires evidence, verified
    HYPOTHESIS = "hypothesis"  # Uncertain, needs verification
    SIMULATION = "simulation"  # Counterfactual only
    FICTION = "fiction"        # Non-binding, exploratory


class IntentType(Enum):
    """Types of user intents"""
    QUERY = "query"           # Ask question
    CREATE = "create"         # Write code, file, etc.
    MODIFY = "modify"         # Edit existing
    EXECUTE = "execute"       # Run command
    LEARN = "learn"           # Store knowledge
    VERIFY = "verify"         # Check facts


class ActionType(Enum):
    """Types of actions the system can take"""
    # Safe, reversible actions
    READ_FILE = "read_file"
    WRITE_CODE = "write_code"
    RUN_CODE = "run_code"
    QUERY_API = "query_api"
    
    # Requires confirmation
    INSTALL_PACKAGE = "install_package"
    DELETE_FILE = "delete_file"
    MODIFY_SYSTEM = "modify_system"
    ACCESS_NETWORK = "access_network"
    
    # Forbidden
    ESCALATE_PERMISSION = "escalate_permission"
    DECEIVE_USER = "deceive_user"
    MANIPULATE_SOCIAL = "manipulate_social"


class EvidenceSource(Enum):
    """Where evidence comes from"""
    EXECUTION_TRACE = "execution_trace"  # Code ran, we saw output
    API_RESPONSE = "api_response"        # External service returned data
    HUMAN_LABEL = "human_label"          # User confirmed
    VERIFICATION_TOOL = "verification_tool"  # Checker validated
    CROSS_REFERENCE = "cross_reference"  # Multiple sources agree


# ============================================================================
# Core IR Dataclasses
# ============================================================================

@dataclass
class IRObject:
    """Base class for all IR objects"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    # provenance moved to subclasses to avoid field ordering issues


@dataclass
class Intent(IRObject):
    """User intent decoded from natural language"""
    type: IntentType = field(default=IntentType.QUERY)
    description: str = ""
    priority: int = 1  # 1-10
    constraints: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    provenance: Optional[str] = None  # Override parent
    
    def __repr__(self):
        return f"Intent({self.type.value}, priority={self.priority}, desc='{self.description[:40]}...')"


@dataclass
class Entity(IRObject):
    """Normalized entity (file, API, concept, etc.)"""
    type: str = ""  # "file", "api", "concept", "package"
    canonical_name: str = ""  # Normalized identifier
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[str] = None  # Override parent
    
    def __repr__(self):
        return f"Entity({self.type}: {self.canonical_name})"


@dataclass
class Claim(IRObject):
    """A statement about the world"""
    subject: str = ""  # Entity or concept
    predicate: str = ""  # Relationship or property
    object: str = ""  # Value or target
    modality: Modality = Modality.HYPOTHESIS  # fact, hypothesis, simulation, fiction
    confidence: float = 0.5  # 0.0 to 1.0
    
    # Evidence chain
    evidence_ids: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)
    provenance: Optional[str] = None  # Override parent
    
    def __repr__(self):
        return f"Claim({self.subject} {self.predicate} {self.object} [{self.modality.value}] conf={self.confidence:.2f})"
    
    def is_verified(self) -> bool:
        """Can this claim be treated as fact?"""
        return self.modality == Modality.FACT and len(self.evidence_ids) > 0 and self.confidence >= 0.8


@dataclass
class Evidence(IRObject):
    """Evidence supporting or refuting a claim"""
    claim_id: str = ""
    source: EvidenceSource = EvidenceSource.EXECUTION_TRACE
    locator: str = ""  # file:line, URL, trace_id
    content: str = ""  # Actual evidence text
    confidence: float = 1.0  # 0.0 to 1.0
    is_supporting: bool = True  # False if this contradicts the claim
    provenance: Optional[str] = None  # Override parent
    
    def __repr__(self):
        support = "+" if self.is_supporting else "-"
        return f"Evidence({support}{self.source.value} conf={self.confidence:.2f})"


@dataclass
class ActionMeta(IRObject):
    """Metadata for safety analysis of an action"""
    action_id: str = ""
    
    # Safety scores (0.0 = safe, 1.0 = maximum risk)
    irreversibility_score: float = 0.5  # Can we undo this?
    blast_radius: float = 0.5  # How much can this affect?
    permission_delta: int = 0  # Does this change permissions? (+1, 0, -1)
    deception_risk: float = 0.0  # Could this mislead user?
    data_access_change: int = 0  # New data sources? (+1, 0, -1)
    
    # Requirements
    confirmation_required: bool = False
    rationale_required: bool = False
    
    # Classification
    forbidden: bool = False
    provenance: Optional[str] = None  # Override parent
    
    def is_safe(self) -> bool:
        """Is this action safe to execute without confirmation?"""
        return (
            not self.forbidden and
            self.irreversibility_score < 0.3 and
            self.blast_radius < 0.3 and
            self.permission_delta <= 0 and
            self.deception_risk < 0.1 and
            self.data_access_change <= 0
        )
    
    def __repr__(self):
        status = "SAFE" if self.is_safe() else "CONFIRM" if not self.forbidden else "FORBIDDEN"
        return f"ActionMeta({status}, irrev={self.irreversibility_score:.2f}, blast={self.blast_radius:.2f})"


@dataclass
class Action(IRObject):
    """A proposed or executed action"""
    type: ActionType = ActionType.READ_FILE
    args: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[ActionMeta] = None
    
    # Execution tracking
    status: Literal["proposed", "approved", "executing", "completed", "failed", "rejected"] = "proposed"
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Optionality tracking
    preserves_optionality: bool = True  # Does this keep options open?
    future_actions_enabled: Set[str] = field(default_factory=set)
    future_actions_blocked: Set[str] = field(default_factory=set)
    provenance: Optional[str] = None  # Override parent
    
    def __repr__(self):
        return f"Action({self.type.value}, status={self.status})"


@dataclass
class MemoryWrite(IRObject):
    """Record of writing to memory (append-only log)"""
    scope: str = ""  # Domain or category
    payload: Optional[IRObject] = None  # The actual object being stored
    provenance: str = ""  # Where this came from
    version: int = 1
    
    def __repr__(self):
        return f"MemoryWrite({self.scope}, v{self.version}, {type(self.payload).__name__})"


@dataclass
class ValueDiff(IRObject):
    """Proposed change to value function"""
    rationale: str = ""
    predicted_behavior_change: str = ""
    
    # What changes
    old_weights: Dict[str, float] = field(default_factory=dict)
    new_weights: Dict[str, float] = field(default_factory=dict)
    
    # Testing requirements
    red_lines: List[str] = field(default_factory=list)  # Must never do
    regression_tests: List[str] = field(default_factory=list)  # Must still work
    
    # Approval
    status: Literal["proposed", "testing", "approved", "rejected", "reverted"] = "proposed"
    test_results: Dict[str, bool] = field(default_factory=dict)
    provenance: Optional[str] = None  # Override parent
    
    def __repr__(self):
        return f"ValueDiff({self.status}, {len(self.new_weights)} weights)"


@dataclass
class Counterfactual(IRObject):
    """Immutable counterfactual for learning"""
    scenario: str = ""
    actual_outcome: str = ""
    hypothetical_outcome: str = ""
    human_valence: Optional[float] = None  # -1.0 (bad) to +1.0 (good)
    
    # What would have been different?
    actions_taken: List[Action] = field(default_factory=list)
    actions_avoided: List[Action] = field(default_factory=list)
    
    # Learning signals
    importance: float = 0.5  # How much does this matter?
    surprise: float = 0.0  # How unexpected was this?
    provenance: Optional[str] = None  # Override parent
    
    def __repr__(self):
        valence = f"valence={self.human_valence:+.2f}" if self.human_valence else "unlabeled"
        return f"Counterfactual({valence}, importance={self.importance:.2f})"


@dataclass
class ReasoningTrace(IRObject):
    """Record of reasoning steps (for audit)"""
    input_claims: List[Claim] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    output_claims: List[Claim] = field(default_factory=list)
    confidence: float = 0.5
    
    # Which universe?
    universe: Modality = Modality.HYPOTHESIS
    
    # Verification
    verified: bool = False
    verification_failures: List[str] = field(default_factory=list)
    provenance: Optional[str] = None  # Override parent
    
    def __repr__(self):
        status = "[v]" if self.verified else "?"
        return f"ReasoningTrace({status}, {len(self.reasoning_steps)} steps, conf={self.confidence:.2f})"


# ============================================================================
# Utility Functions
# ============================================================================

def create_claim(subject: str, predicate: str, object: str, 
                 modality: Modality = Modality.HYPOTHESIS,
                 confidence: float = 0.5) -> Claim:
    """Helper to create a claim"""
    return Claim(
        subject=subject,
        predicate=predicate,
        object=object,
        modality=modality,
        confidence=confidence
    )


def create_evidence(claim: Claim, source: EvidenceSource, 
                   locator: str, content: str, 
                   confidence: float = 1.0) -> Evidence:
    """Helper to create evidence for a claim"""
    return Evidence(
        claim_id=claim.id,
        source=source,
        locator=locator,
        content=content,
        confidence=confidence
    )


def promote_to_fact(claim: Claim, evidence: Evidence) -> None:
    """Promote a hypothesis to fact if evidence is strong enough"""
    if evidence.confidence >= 0.8 and evidence.is_supporting:
        claim.modality = Modality.FACT
        if evidence.id not in claim.evidence_ids:
            claim.evidence_ids.append(evidence.id)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create a hypothesis
    claim = create_claim(
        subject="requests",
        predicate="requires_import",
        object="True",
        modality=Modality.HYPOTHESIS,
        confidence=0.7
    )
    print(f"1. {claim}")
    
    # Add evidence
    evidence = create_evidence(
        claim=claim,
        source=EvidenceSource.EXECUTION_TRACE,
        locator="script.py:2",
        content="NameError: name 'requests' is not defined",
        confidence=1.0
    )
    print(f"2. {evidence}")
    
    # Promote to fact
    promote_to_fact(claim, evidence)
    print(f"3. {claim}")
    print(f"   Verified: {claim.is_verified()}")
    
    # Create action with metadata
    action = Action(
        type=ActionType.RUN_CODE,
        args={"file": "script.py"}
    )
    action.metadata = ActionMeta(
        action_id=action.id,
        irreversibility_score=0.1,  # Low - can re-run
        blast_radius=0.2,  # Low - just one file
        permission_delta=0,  # No change
        deception_risk=0.0,  # No deception
        data_access_change=0,  # No new access
        confirmation_required=False,
        rationale_required=False
    )
    print(f"4. {action}")
    print(f"   {action.metadata}")
    print(f"   Safe to execute: {action.metadata.is_safe()}")
