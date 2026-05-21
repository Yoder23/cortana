"""
Cortana — formally verified AI companion with hard-constraint safety.

Public API::

    from cortana import (
        # IR types
        Modality, ActionType, ActionMeta, Action, Claim, Evidence,
        create_claim, create_evidence,
        # Safety gate
        ActionGatekeeper, HardConstraints, SafetyClass, ActionDecision,
        # Verifier
        Verifier, VerificationResult,
        # Memory
        StructuredMemory,
        # Values
        ValueDriftController,
        # Pipeline
        CortanaPipeline, PipelineResult,
        # Top-level
        CortanaProduction,
    )

Zero external dependencies for the core — stdlib + numpy only.
"""

__version__ = "1.0.0"
__author__ = "Yoder23"

# IR types
from .ir import (
    Modality,
    IntentType,
    ActionType,
    EvidenceSource,
    IRObject,
    Intent,
    Entity,
    Claim,
    Evidence,
    ActionMeta,
    Action,
    MemoryWrite,
    ValueDiff,
    Counterfactual,
    ReasoningTrace,
    create_claim,
    create_evidence,
    promote_to_fact,
)

# Safety gate
from .gatekeeper import (
    ActionGatekeeper,
    HardConstraints,
    SafetyClass,
    ActionDecision,
)

# Verifier
from .verifier import (
    Verifier,
    VerificationResult,
    UniverseViolation,
)

# Memory
from .memory import StructuredMemory

# Values
from .values import ValueDriftController, ValueProfile, RedLineTests

# Pipeline
from .pipeline import CortanaPipeline, PipelineResult

# Production entry-point
from .production import CortanaProduction

__all__ = [
    # IR
    "Modality", "IntentType", "ActionType", "EvidenceSource",
    "IRObject", "Intent", "Entity", "Claim", "Evidence",
    "ActionMeta", "Action", "MemoryWrite", "ValueDiff",
    "Counterfactual", "ReasoningTrace",
    "create_claim", "create_evidence", "promote_to_fact",
    # Safety
    "ActionGatekeeper", "HardConstraints", "SafetyClass", "ActionDecision",
    # Verifier
    "Verifier", "VerificationResult", "UniverseViolation",
    # Memory
    "StructuredMemory",
    # Values
    "ValueDriftController", "ValueProfile", "RedLineTests",
    # Pipeline
    "CortanaPipeline", "PipelineResult",
    # Top-level
    "CortanaProduction",
]
