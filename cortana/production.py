"""
CORTANA - Trusted AI Companion with Formal Verification

The groundbreaking AI system with:
- Structured reasoning (FACT/HYPOTHESIS/SIMULATION/FICTION separation)
- Formal verification (no hallucination leakage)
- Hard constraint gating (safety cannot be learned away)
- Value drift control (explicit, auditable, reversible)
- Bounded attention over infinite memory

This is the production entry point. Old MOA system kept as backup.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from .pipeline import CortanaPipeline, PipelineResult
from .ir import Action, ActionMeta, Modality


class CortanaProduction:
    """
    Production CORTANA system.
    
    This is the main interface for the groundbreaking AI companion.
    The old MOA system remains available as a backup, but CORTANA
    is the primary, production-ready system.
    """
    
    def __init__(
        self,
        storage_path: str = "cortana_system",
        enable_legacy_fallback: bool = True,
        domain: str = "general",
        runtime_profile: str = "production",
        require_high_quality_embeddings: Optional[bool] = None,
        allow_simple_embedder: bool = False,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize CORTANA production system.
        
        Args:
            storage_path: Base directory for all CORTANA storage (memory, values, etc.)
            enable_legacy_fallback: Keep old MOA system available as backup
            domain: Domain for Gibberlink translation (general, python, chess, etc.)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing CORTANA production system...")
        runtime_profile = runtime_profile.strip().lower()
        if runtime_profile not in {"research", "production"}:
            raise ValueError(f"Unknown runtime_profile: {runtime_profile}")
        if require_high_quality_embeddings is None:
            require_high_quality_embeddings = runtime_profile == "production"
        
        # Primary system: CORTANA
        self.cortana = CortanaPipeline(
            storage_path=storage_path,
            domain=domain,
            require_high_quality_embeddings=bool(require_high_quality_embeddings),
            allow_simple_embedder=allow_simple_embedder,
            openai_api_key=openai_api_key,
        )
        
        # Backup system (optional)
        self.legacy_available = False
        if enable_legacy_fallback:
            try:
                from moa_continuous_learning import ContinuousLearningPipeline
                self.legacy_pipeline = ContinuousLearningPipeline(
                    Path("layercake_domains")
                )
                self.legacy_available = True
                self.logger.info("   ✓ Legacy MOA system available as backup")
            except Exception as e:
                self.logger.warning(f"   ⚠ Legacy system unavailable: {e}")
        
        self.logger.info("   ✓ CORTANA initialized successfully")
        self._print_capabilities()
    
    def _print_capabilities(self):
        """Print CORTANA capabilities."""
        self.logger.info("\n" + "="*80)
        self.logger.info("CORTANA CAPABILITIES")
        self.logger.info("="*80)
        self.logger.info("✓ Structured IR with modality tracking")
        self.logger.info("✓ Formal verification (prevent hallucination leakage)")
        self.logger.info("✓ Hard constraint gating (safety guaranteed)")
        self.logger.info("✓ Value drift control (auditable, reversible)")
        self.logger.info("✓ Bounded attention over infinite memory")
        self.logger.info("✓ Universe separation (FACT/HYPOTHESIS/SIMULATION/FICTION)")
        
        # Show new features
        if self.cortana.translator is not None:
            self.logger.info("✓ Gibberlink Translation (English ↔ Structured IR)")
        if self.cortana.causal_reasoner is not None:
            self.logger.info("✓ Causal Reasoning (root cause analysis, counterfactuals)")
        if self.cortana.confirmation_manager is not None:
            self.logger.info("✓ Interactive Confirmation (human-in-the-loop)")
        
        # Show embeddings quality
        embedder_type = getattr(self.cortana.memory.embedder, 'embedder_type', None)
        if embedder_type:
            self.logger.info(f"✓ High-Quality Embeddings ({embedder_type})")
        
        self.logger.info("="*80 + "\n")
    
    def learn_from_experience(
        self,
        description: str,
        outcome: str,
        success: bool,
        error_message: Optional[str] = None
    ) -> PipelineResult:
        """
        Learn from an experience using CORTANA's formal verification.
        
        This is the primary learning interface. It:
        1. Extracts structured claims from the experience
        2. Creates evidence for each claim
        3. Verifies claims (fact vs hypothesis)
        4. Stores in structured memory with universe separation
        
        Args:
            description: What was attempted
            outcome: What happened
            success: Whether it succeeded
            error_message: Error if it failed
            
        Returns:
            PipelineResult with verification stats
        """
        self.logger.info(f"📚 Learning from experience: {description[:50]}...")
        
        result = self.cortana.learn_from_experience(
            description=description,
            outcome=outcome,
            success=success,
            error_message=error_message
        )
        
        self.logger.info(f"   ✓ Learned: {result.facts_verified} facts, "
                        f"{result.claims_stored - result.facts_verified} hypotheses")
        
        return result
    
    def retrieve_knowledge(
        self,
        query: str,
        facts_only: bool = False
    ):
        """
        Retrieve relevant knowledge with bounded attention.
        
        This uses embedding-based retrieval to find the top-k most
        relevant claims (k=5 for bounded attention).
        
        Args:
            query: What to search for
            facts_only: Only return verified facts
            
        Returns:
            List of relevant claims
        """
        self.logger.info(f"🔍 Retrieving knowledge: '{query}'")
        
        claims = self.cortana.retrieve_relevant_knowledge(
            query=query,
            facts_only=facts_only
        )
        
        fact_count = sum(1 for c in claims if c.modality == Modality.FACT)
        self.logger.info(f"   ✓ Retrieved {len(claims)} claims "
                        f"({fact_count} verified facts)")
        
        return claims
    
    def propose_action(
        self,
        action: Action,
        justification: str
    ) -> tuple[bool, str]:
        """
        Propose an action through CORTANA's gatekeeper.
        
        This ensures:
        1. Justification is verified (facts for irreversible actions)
        2. Hard constraints are satisfied
        3. Action is safe, reversible, or requires confirmation
        
        Args:
            action: The action to perform
            justification: Why this action should be taken
            
        Returns:
            (allowed, reason) tuple
        """
        self.logger.info(f"🎬 Proposing action: {action.type}")
        
        allowed, reason = self.cortana.propose_action(
            action=action,
            justification=justification
        )
        
        status = "✓ ALLOWED" if allowed else "✗ BLOCKED"
        self.logger.info(f"   {status}: {reason}")
        
        return allowed, reason
    
    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        stats = self.cortana.memory.stats()
        
        self.logger.info("\n📊 CORTANA Memory Statistics:")
        self.logger.info(f"   Total Events: {stats['total_events']}")
        self.logger.info(f"   Verified Facts: {stats['verified_facts']}")
        self.logger.info(f"   Total Claims: {stats['total_claims']}")
        self.logger.info(f"   Evidence: {stats['total_evidence']}")
        self.logger.info(f"   Counterfactuals: {stats['counterfactuals']}")
        
        return stats
    
    def get_value_profile(self) -> dict:
        """Get current value weights."""
        profile = self.cortana.value_controller.current_profile
        
        self.logger.info("\n⚖️ CORTANA Value Profile:")
        self.logger.info(f"   Version: {profile.version}")
        for name, weight in profile.weights.items():
            self.logger.info(f"   {name}: {weight}")
        
        return profile.weights
    
    def propose_value_change(
        self,
        new_weights: dict,
        rationale: str
    ) -> tuple[bool, list]:
        """
        Propose a value weight change.
        
        This will:
        1. Create explicit diff
        2. Run red-line tests (invariants)
        3. Run regression tests
        4. Run Goodhart tests
        5. Approve only if all pass
        
        Args:
            new_weights: New weight values
            rationale: Why this change is needed
            
        Returns:
            (approved, test_results) tuple
        """
        self.logger.info(f"⚖️ Proposing value change: {rationale[:50]}...")
        
        approved, test_results = self.cortana.propose_value_change(
            new_weights=new_weights,
            rationale=rationale
        )
        
        status = "✓ APPROVED" if approved else "✗ REJECTED"
        self.logger.info(f"   {status}")
        
        return approved, test_results
    
    def analyze_problem(self, problem_description: str):
        """
        Perform causal root cause analysis on a problem.
        
        Uses CORTANA's causal reasoning engine to find root causes,
        explain causal chains, and suggest interventions.
        
        Args:
            problem_description: Description of the problem
            
        Returns:
            Analysis results or None if causal reasoning unavailable
        """
        if self.cortana.causal_reasoner is None:
            self.logger.warning("Causal reasoning not available")
            return None
        
        # First, extract a claim about the problem
        if self.cortana.translator is None:
            self.logger.warning("Cannot extract problem claim without translator")
            return None
        
        claims = self.cortana.translator.extract_claims(problem_description, {})
        if not claims:
            self.logger.warning("No claims extracted from problem description")
            return None
        
        problem_claim = claims[0]
        
        # Analyze
        analysis = self.cortana.causal_reasoner.analyze_problem(problem_claim)
        
        self.logger.info("\n🔍 ROOT CAUSE ANALYSIS")
        self.logger.info("="*80)
        self.logger.info(f"Problem: {problem_claim.subject}")
        self.logger.info(f"Root causes found: {analysis['num_root_causes']}")
        
        for i, ana in enumerate(analysis['analyses'][:3], 1):
            self.logger.info(f"\nRoot Cause #{i}:")
            self.logger.info(f"  Cause: {ana['root_cause'].subject}")
            self.logger.info(f"  Path length: {ana['path_length']}")
            self.logger.info(f"  Strength: {ana['cumulative_strength']:.2f}")
        
        return analysis
    
    def explain_causality(self, effect_description: str) -> str:
        """
        Generate natural language explanation of why something happened.
        
        Args:
            effect_description: Description of the effect to explain
            
        Returns:
            Natural language explanation
        """
        if self.cortana.causal_reasoner is None or self.cortana.translator is None:
            return "Causal explanation not available"
        
        # Extract claim
        claims = self.cortana.translator.extract_claims(effect_description, {})
        if not claims:
            return "Could not extract claim from description"
        
        effect_claim = claims[0]
        
        # Generate explanation
        explanation = self.cortana.causal_reasoner.explain_why(effect_claim)
        
        return explanation
    
    def predict_action_effects(self, action_description: str):
        """
        Predict effects of an action using causal reasoning.
        
        Args:
            action_description: Description of the action
            
        Returns:
            List of predicted effects
        """
        if self.cortana.causal_reasoner is None or self.cortana.translator is None:
            self.logger.warning("Causal prediction not available")
            return []
        
        # Extract action claim
        claims = self.cortana.translator.extract_claims(action_description, {})
        if not claims:
            return []
        
        action_claim = claims[0]
        
        # Predict effects
        effects = self.cortana.causal_reasoner.predict_action_effects(action_claim, min_strength=0.5)
        
        self.logger.info(f"\n🔮 PREDICTED EFFECTS")
        self.logger.info(f"Action: {action_claim.subject}")
        self.logger.info(f"Predicted {len(effects)} effects:")
        
        for effect_claim, strength in effects[:5]:
            self.logger.info(f"  - {effect_claim.subject} (strength: {strength:.2f})")
        
        return effects
    
    def set_confirmation_mode(self, mode: str):
        """
        Change interactive confirmation mode.
        
        Modes:
        - "interactive": Ask human via CLI
        - "auto_approve": Approve automatically (testing)
        - "auto_reject": Reject automatically (conservative)
        
        Args:
            mode: Confirmation mode
        """
        if self.cortana.confirmation_manager is None:
            self.logger.warning("Interactive confirmation not available")
            return
        
        from cortana_interactive import ConfirmationMode
        
        mode_map = {
            "interactive": ConfirmationMode.INTERACTIVE,
            "auto_approve": ConfirmationMode.AUTO_APPROVE,
            "auto_reject": ConfirmationMode.AUTO_REJECT
        }
        
        if mode not in mode_map:
            self.logger.error(f"Unknown mode: {mode}")
            return
        
        self.cortana.confirmation_manager.set_mode(mode_map[mode])
        self.logger.info(f"✓ Confirmation mode set to: {mode}")



def demo_cortana_production():
    """Demonstrate CORTANA production system."""
    
    print("\n" + "="*80)
    print("CORTANA PRODUCTION DEMO")
    print("="*80)
    print("\nThe groundbreaking AI companion is now live!\n")
    
    # Initialize
    cortana = CortanaProduction(storage_path="cortana_production")
    
    # Demo 1: Learn from experience
    print("\n" + "-"*80)
    print("DEMO 1: Learning from Experience")
    print("-"*80 + "\n")
    
    result = cortana.learn_from_experience(
        description="Implement API endpoint with authentication",
        outcome="Successfully created /api/v1/users endpoint with JWT auth",
        success=True
    )
    print(f"   Facts: {result.facts_verified}, Claims: {result.claims_stored}")
    
    # Demo 2: Retrieve knowledge
    print("\n" + "-"*80)
    print("DEMO 2: Knowledge Retrieval (Bounded Attention)")
    print("-"*80 + "\n")
    
    claims = cortana.retrieve_knowledge("authentication")
    print(f"   Retrieved {len(claims)} relevant claims")
    
    # Demo 3: Propose safe action
    print("\n" + "-"*80)
    print("DEMO 3: Action Gating (Safe Action)")
    print("-"*80 + "\n")
    
    safe_action = Action(
        type="read_file",
        args={"path": "config.json"},
        metadata=ActionMeta(
            irreversibility_score=0.0,
            blast_radius=0.1,
            permission_delta=0,
            deception_risk=0.0
        )
    )
    
    allowed, reason = cortana.propose_action(
        action=safe_action,
        justification=""  # Empty for simple demo
    )
    print(f"   Action: {'ALLOWED' if allowed else 'BLOCKED'}")
    print(f"   Reason: {reason}")
    
    # Demo 4: Propose forbidden action
    print("\n" + "-"*80)
    print("DEMO 4: Action Gating (Forbidden Action)")
    print("-"*80 + "\n")
    
    forbidden_action = Action(
        type="escalate_permission",
        args={"target": "admin"},
        metadata=ActionMeta(
            irreversibility_score=0.95,
            blast_radius=0.9,
            permission_delta=1,
            deception_risk=0.0,
            forbidden=True
        )
    )
    
    allowed, reason = cortana.propose_action(
        action=forbidden_action,
        justification=""
    )
    print(f"   Action: {'ALLOWED' if allowed else 'BLOCKED'}")
    print(f"   Reason: {reason}")
    
    # Stats
    print("\n" + "-"*80)
    print("FINAL STATISTICS")
    print("-"*80 + "\n")
    
    cortana.get_memory_stats()
    cortana.get_value_profile()
    
    print("\n" + "="*80)
    print("✓ CORTANA PRODUCTION SYSTEM OPERATIONAL")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    sys.exit(demo_cortana_production())

