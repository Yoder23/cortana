"""
CORTANA Action Gatekeeper

Hard constraints beat incentives.
Forbidden actions are rejected BEFORE optimization.

Design Principle: Corrigibility is Instrumental
- Actions that preserve human steering are preferred
- Permission escalation is forbidden
- Irreversible actions require confirmation
- Deception is structurally prevented

Safety guarantees through architecture, not alignment.
"""

import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from .ir import (
    Action, ActionType, ActionMeta, Claim, Modality
)

logger = logging.getLogger(__name__)


# ============================================================================
# Action Classification
# ============================================================================

class SafetyClass(Enum):
    """Action safety classification"""
    SAFE = "safe"              # Execute immediately
    REVERSIBLE = "reversible"  # Low risk, can undo
    CONFIRM = "confirm"        # Needs user approval
    FORBIDDEN = "forbidden"    # Never execute


@dataclass
class ActionDecision:
    """Decision about whether to allow action"""
    allowed: bool
    safety_class: SafetyClass
    reasons: List[str]
    alternatives: List[Action]  # Safer alternatives
    
    def __repr__(self):
        status = "[v] ALLOWED" if self.allowed else "[x] BLOCKED"
        return f"ActionDecision({status}, {self.safety_class.value}, {len(self.reasons)} reasons)"


# ============================================================================
# Hard Constraints (Never Violated)
# ============================================================================

class HardConstraints:
    """
    Hard constraints that can never be violated
    
    These are architectural guarantees, not learned behaviors.
    No amount of training can override these.
    """
    
    # Forbidden action types (never allowed)
    FORBIDDEN_ACTIONS = {
        ActionType.ESCALATE_PERMISSION,
        ActionType.DECEIVE_USER,
        ActionType.MANIPULATE_SOCIAL,
    }
    
    # Maximum blast radius (0.0 to 1.0)
    MAX_BLAST_RADIUS = 0.7
    
    # Maximum irreversibility score (0.0 to 1.0)
    MAX_IRREVERSIBILITY = 0.8
    
    # Permission changes forbidden
    ALLOW_PERMISSION_INCREASE = False
    
    # Data access changes require confirmation
    NEW_DATA_ACCESS_FORBIDDEN = True
    
    @classmethod
    def check(cls, action: Action, metadata: ActionMeta) -> Tuple[bool, List[str]]:
        """
        Check if action violates hard constraints
        
        Returns:
            (is_valid, violations)
        """
        violations = []
        
        # Check 1: Forbidden action type
        if action.type in cls.FORBIDDEN_ACTIONS:
            action_type_str = action.type.value if hasattr(action.type, 'value') else action.type
            violations.append(f"Forbidden action type: {action_type_str}")
        
        # Check 2: Explicit forbidden flag
        if metadata.forbidden:
            violations.append("Action explicitly marked as forbidden")
        
        # Check 3: Blast radius too high
        if metadata.blast_radius > cls.MAX_BLAST_RADIUS:
            violations.append(f"Blast radius {metadata.blast_radius:.2f} exceeds {cls.MAX_BLAST_RADIUS}")
        
        # Check 4: Too irreversible
        if metadata.irreversibility_score > cls.MAX_IRREVERSIBILITY:
            violations.append(f"Irreversibility {metadata.irreversibility_score:.2f} exceeds {cls.MAX_IRREVERSIBILITY}")
        
        # Check 5: Permission escalation
        if not cls.ALLOW_PERMISSION_INCREASE and metadata.permission_delta > 0:
            violations.append(f"Permission escalation forbidden (delta={metadata.permission_delta})")
        
        # Check 6: New data access
        if cls.NEW_DATA_ACCESS_FORBIDDEN and metadata.data_access_change > 0:
            violations.append(f"New data access forbidden (change={metadata.data_access_change})")
        
        # Check 7: Deception risk
        if metadata.deception_risk > 0.1:
            violations.append(f"Deception risk {metadata.deception_risk:.2f} exceeds 0.1")
        
        is_valid = len(violations) == 0
        return is_valid, violations


# ============================================================================
# Action Gatekeeper
# ============================================================================

class ActionGatekeeper:
    """
    Enforces safety constraints on actions
    
    Pipeline:
    1. Hard constraint check (forbidden?)
    2. Safety classification
    3. Corrigibility check
    4. Optionality analysis
    5. Final decision
    """
    
    def __init__(self):
        self.rejected_actions: List[Tuple[Action, List[str]]] = []
        self.confirmed_actions: List[Action] = []
        
        logger.info("[OK] Action Gatekeeper initialized")
        logger.info(f"   Hard constraints active:")
        logger.info(f"      * Forbidden actions: {len(HardConstraints.FORBIDDEN_ACTIONS)}")
        logger.info(f"      * Max blast radius: {HardConstraints.MAX_BLAST_RADIUS}")
        logger.info(f"      * Max irreversibility: {HardConstraints.MAX_IRREVERSIBILITY}")
        logger.info(f"      * Permission escalation: {'ALLOWED' if HardConstraints.ALLOW_PERMISSION_INCREASE else 'FORBIDDEN'}")
    
    # ========================================================================
    # Core Evaluation
    # ========================================================================
    
    def evaluate(self, action: Action, 
                 justification: Optional[List[Claim]] = None) -> ActionDecision:
        """
        Evaluate if action should be allowed
        
        Args:
            action: The proposed action
            justification: Claims supporting this action
        
        Returns:
            ActionDecision with verdict and reasoning
        """
        reasons = []
        alternatives = []
        
        # Ensure action has metadata
        if action.metadata is None:
            action.metadata = self._default_metadata(action)
        
        metadata = action.metadata
        
        # Stage 1: Hard Constraints (MUST PASS)
        is_valid, violations = HardConstraints.check(action, metadata)
        if not is_valid:
            reasons.extend(violations)
            action_type_str = action.type.value if hasattr(action.type, 'value') else action.type
            logger.error(f"   [FAIL] Action BLOCKED by hard constraints: {action_type_str}")
            for violation in violations:
                logger.error(f"      * {violation}")
            
            # Find safer alternatives
            alternatives = self._find_alternatives(action)
            
            return ActionDecision(
                allowed=False,
                safety_class=SafetyClass.FORBIDDEN,
                reasons=reasons,
                alternatives=alternatives
            )
        
        # Stage 2: Safety Classification
        safety_class = self._classify_safety(action, metadata)
        
        # Stage 3: Justification Check
        if justification:
            justification_valid = self._check_justification(action, justification)
            if not justification_valid:
                reasons.append("Insufficient justification (hypothesis/simulation cannot justify real actions)")
                safety_class = SafetyClass.CONFIRM
        
        # Stage 4: Corrigibility Check
        if not self._preserves_corrigibility(action, metadata):
            reasons.append("Action may reduce human control")
            safety_class = SafetyClass.CONFIRM
        
        # Stage 5: Optionality Analysis
        optionality_score = self._analyze_optionality(action)
        if optionality_score < 0.3:
            reasons.append(f"Low optionality (closes future options, score={optionality_score:.2f})")
        
        # Final decision
        # CONFIRM actions are NOT allowed — they require confirmation first.
        # The guarded action path in CortanaPipeline handles confirmation flow.
        allowed = safety_class in {SafetyClass.SAFE, SafetyClass.REVERSIBLE}
        
        decision = ActionDecision(
            allowed=allowed,
            safety_class=safety_class,
            reasons=reasons if reasons else ["Action meets safety criteria"],
            alternatives=alternatives
        )
        
        # Logging
        action_type_str = action.type.value if hasattr(action.type, 'value') else action.type
        if allowed:
            if safety_class == SafetyClass.SAFE:
                logger.info(f"   [v] Action ALLOWED (safe): {action_type_str}")
            elif safety_class == SafetyClass.REVERSIBLE:
                logger.info(f"   [v] Action ALLOWED (reversible): {action_type_str}")
            elif safety_class == SafetyClass.CONFIRM:
                logger.warning(f"   [WARN]  Action needs CONFIRMATION: {action_type_str}")
                for reason in reasons:
                    logger.warning(f"      * {reason}")
        else:
            logger.error(f"   [FAIL] Action BLOCKED: {action_type_str}")
        
        return decision
    
    def _classify_safety(self, action: Action, metadata: ActionMeta) -> SafetyClass:
        """Classify action into safety category"""
        # Forbidden check
        if metadata.forbidden or action.type in HardConstraints.FORBIDDEN_ACTIONS:
            return SafetyClass.FORBIDDEN
        
        # Safe actions
        if metadata.is_safe():
            return SafetyClass.SAFE
        
        # Reversible actions
        if metadata.irreversibility_score < 0.5 and metadata.blast_radius < 0.5:
            return SafetyClass.REVERSIBLE
        
        # Everything else needs confirmation
        return SafetyClass.CONFIRM
    
    def _check_justification(self, action: Action, claims: any) -> bool:
        """Check if claims properly justify action"""
        # Handle string justifications (simple case)
        if isinstance(claims, str):
            # Simple string justifications are treated as weak evidence
            return len(claims) > 0
        
        # Handle List[Claim]
        if not claims:
            return False
        
        # Irreversible actions need FACT justification
        if action.metadata.irreversibility_score > 0.5:
            return any(c.modality == Modality.FACT for c in claims)
        
        # Simulation/fiction cannot justify real actions
        return not any(c.modality in {Modality.SIMULATION, Modality.FICTION} for c in claims)
    
    def _preserves_corrigibility(self, action: Action, metadata: ActionMeta) -> bool:
        """Check if action preserves human control"""
        # Actions that change permissions reduce corrigibility
        if metadata.permission_delta > 0:
            return False
        
        # Actions that hide information reduce corrigibility
        if metadata.deception_risk > 0.0:
            return False
        
        # Actions with high irreversibility reduce options
        if metadata.irreversibility_score > 0.7:
            return False
        
        return True
    
    def _analyze_optionality(self, action: Action) -> float:
        """
        Analyze how action affects future options
        
        Optionality score:
        - 1.0: Opens many options, closes none
        - 0.5: Neutral
        - 0.0: Closes all options
        """
        enabled = len(action.future_actions_enabled)
        blocked = len(action.future_actions_blocked)
        
        if enabled + blocked == 0:
            return 0.5  # Neutral
        
        score = enabled / (enabled + blocked)
        return score
    
    def _find_alternatives(self, action: Action) -> List[Action]:
        """Find safer alternatives to forbidden action"""
        alternatives = []
        
        # Common patterns
        if action.type == ActionType.DELETE_FILE:
            # Suggest move to trash instead
            alt = Action(
                type=ActionType.MODIFY_SYSTEM,
                args={'operation': 'move_to_trash', 'file': action.args.get('file')}
            )
            alternatives.append(alt)
        
        elif action.type == ActionType.INSTALL_PACKAGE:
            # Suggest asking user first
            alt = Action(
                type=ActionType.QUERY_API,
                args={'query': f"Should I install {action.args.get('package')}?"}
            )
            alternatives.append(alt)
        
        return alternatives
    
    def _default_metadata(self, action: Action) -> ActionMeta:
        """Generate default metadata for action type"""
        # Safe defaults
        metadata = ActionMeta(
            action_id=action.id,
            irreversibility_score=0.5,
            blast_radius=0.5,
            permission_delta=0,
            deception_risk=0.0,
            data_access_change=0,
            confirmation_required=True,
            rationale_required=True,
            forbidden=False
        )
        
        # Adjust based on type
        if action.type == ActionType.READ_FILE:
            metadata.irreversibility_score = 0.0
            metadata.blast_radius = 0.1
            metadata.confirmation_required = False
        
        elif action.type == ActionType.WRITE_CODE:
            metadata.irreversibility_score = 0.2
            metadata.blast_radius = 0.2
            metadata.confirmation_required = False
        
        elif action.type == ActionType.RUN_CODE:
            metadata.irreversibility_score = 0.3
            metadata.blast_radius = 0.4
            metadata.confirmation_required = False
        
        elif action.type == ActionType.DELETE_FILE:
            # High-risk but still operator-confirmable under hard limits.
            metadata.irreversibility_score = 0.8
            metadata.blast_radius = 0.7
            metadata.confirmation_required = True
        
        elif action.type == ActionType.INSTALL_PACKAGE:
            metadata.irreversibility_score = 0.6
            metadata.blast_radius = 0.5
            metadata.data_access_change = 1
            metadata.confirmation_required = True
        
        elif action.type in HardConstraints.FORBIDDEN_ACTIONS:
            metadata.forbidden = True
        
        return metadata
    
    # ========================================================================
    # Execution Tracking
    # ========================================================================
    
    def record_rejection(self, action: Action, reasons: List[str]) -> None:
        """Record rejected action for analysis"""
        self.rejected_actions.append((action, reasons))
        action_type_str = action.type.value if hasattr(action.type, 'value') else action.type
        logger.warning(f"   [NOTE] Recorded rejection: {action_type_str}")
    
    def record_confirmation(self, action: Action) -> None:
        """Record that action was confirmed by user"""
        self.confirmed_actions.append(action)
        logger.info(f"   [v] User confirmed: {action.type.value}")
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def stats(self) -> Dict[str, int]:
        """Gatekeeper statistics"""
        return {
            'total_rejections': len(self.rejected_actions),
            'total_confirmations': len(self.confirmed_actions),
            'rejection_types': {
                action.type.value: sum(1 for a, _ in self.rejected_actions if a.type == action.type)
                for action, _ in self.rejected_actions
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
    print("CORTANA Action Gatekeeper Demo")
    print("="*80 + "\n")
    
    from .ir import create_claim, Modality
    
    gatekeeper = ActionGatekeeper()
    
    # Test 1: Safe action (read file)
    print("\n" + "-"*80)
    print("Test 1: Safe action (read file)")
    print("-"*80)
    
    action1 = Action(type=ActionType.READ_FILE, args={'file': 'test.py'})
    decision1 = gatekeeper.evaluate(action1)
    print(f"\nDecision: {decision1}")
    
    # Test 2: Needs confirmation (delete file)
    print("\n" + "-"*80)
    print("Test 2: Needs confirmation (delete file)")
    print("-"*80)
    
    action2 = Action(type=ActionType.DELETE_FILE, args={'file': 'important.txt'})
    decision2 = gatekeeper.evaluate(action2)
    print(f"\nDecision: {decision2}")
    if decision2.alternatives:
        print(f"Alternatives: {len(decision2.alternatives)}")
        for alt in decision2.alternatives:
            print(f"   * {alt.type.value}")
    
    # Test 3: Forbidden action (permission escalation)
    print("\n" + "-"*80)
    print("Test 3: Forbidden action (permission escalation)")
    print("-"*80)
    
    action3 = Action(type=ActionType.ESCALATE_PERMISSION, args={'level': 'admin'})
    decision3 = gatekeeper.evaluate(action3)
    print(f"\nDecision: {decision3}")
    
    # Test 4: Action with weak justification
    print("\n" + "-"*80)
    print("Test 4: Irreversible action with hypothesis justification (should block)")
    print("-"*80)
    
    action4 = Action(type=ActionType.MODIFY_SYSTEM, args={'change': 'registry'})
    action4.metadata = ActionMeta(
        action_id=action4.id,
        irreversibility_score=0.9,
        blast_radius=0.8,
        permission_delta=0,
        deception_risk=0.0,
        data_access_change=0,
        confirmation_required=True,
        rationale_required=True
    )
    
    weak_justification = [
        create_claim("system", "needs_update", "maybe", Modality.HYPOTHESIS)
    ]
    decision4 = gatekeeper.evaluate(action4, weak_justification)
    print(f"\nDecision: {decision4}")
    
    # Stats
    print("\n" + "-"*80)
    print("Statistics")
    print("-"*80)
    stats = gatekeeper.stats()
    print(f"\nTotal rejections: {stats['total_rejections']}")
    print(f"Total confirmations: {stats['total_confirmations']}")

