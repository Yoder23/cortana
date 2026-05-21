"""
CORTANA Structured Memory System

Bounded attention over infinite memory.
- Append-only, versioned, provenance-tracked
- Embedding-based retrieval
- Fact vs hypothesis separation
- Promotion rules enforced

Design Principle: Memory ≠ Belief
- Everything stored in event log
- Only verified facts in factual graph
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from datetime import datetime
from collections import defaultdict
import numpy as np

from .ir import (
    IRObject, Claim, Evidence, Action, Entity, Counterfactual,
    MemoryWrite, ReasoningTrace, Modality, EvidenceSource
)

logger = logging.getLogger(__name__)


# ============================================================================
# Real Embeddings (using SmartEmbedder for high-quality semantic retrieval)
# ============================================================================

try:
    from .embeddings import SmartEmbedder
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("cortana_embeddings not available, using fallback")
    EMBEDDINGS_AVAILABLE = False
    
    # Fallback: Simple character-level embeddings
    class SimpleEmbedder:
        """Character-level embeddings for demo purposes"""
        
        def embed(self, text: str) -> np.ndarray:
            """Convert text to simple embedding"""
            # Simple character histogram (26 letters + space + digits)
            vec = np.zeros(38)
            text_lower = text.lower()
            for char in text_lower:
                if 'a' <= char <= 'z':
                    vec[ord(char) - ord('a')] += 1
                elif char == ' ':
                    vec[26] += 1
                elif '0' <= char <= '9':
                    vec[27 + ord(char) - ord('0')] += 1
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        
        def similarity(self, text1: str, text2: str) -> float:
            """Cosine similarity between two texts"""
            vec1 = self.embed(text1)
            vec2 = self.embed(text2)
            return float(np.dot(vec1, vec2))


# ============================================================================
# Memory Stores
# ============================================================================

class EventLog:
    """Append-only log of everything that happened"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.storage_path / "event_log.jsonl"
        self.index = []  # In-memory index
        self._load_index()
    
    def _load_index(self):
        """Load index from disk"""
        if self.log_file.exists():
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self.index.append({
                            'id': entry['id'],
                            'timestamp': entry['timestamp'],
                            'type': entry['type']
                        })
            logger.info(f"   [BOOK] Loaded {len(self.index)} events from log")
    
    def append(self, obj: IRObject) -> None:
        """Append object to event log"""
        entry = {
            'id': obj.id,
            'timestamp': obj.timestamp.isoformat(),
            'type': type(obj).__name__,
            'data': self._serialize(obj)
        }
        
        # Write to disk
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Update index
        self.index.append({
            'id': obj.id,
            'timestamp': obj.timestamp.isoformat(),
            'type': type(obj).__name__
        })
    
    def _serialize(self, obj: IRObject) -> Dict:
        """Serialize IR object to dict"""
        if isinstance(obj, Claim):
            return {
                'subject': obj.subject,
                'predicate': obj.predicate,
                'object': obj.object,
                'modality': obj.modality.value,
                'confidence': obj.confidence,
                'evidence_ids': obj.evidence_ids,
                'provenance': obj.provenance
            }
        elif isinstance(obj, Evidence):
            content = obj.content
            if not isinstance(content, (str, int, float, bool, type(None), list, dict)):
                content = str(content)
            if isinstance(content, dict):
                # Ensure nested values are JSON-serializable.
                content = {
                    str(k): (v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v))
                    for k, v in content.items()
                }
            return {
                'claim_id': str(obj.claim_id),
                'source': obj.source.value,
                'locator': obj.locator,
                'content': content,
                'confidence': obj.confidence,
                'is_supporting': obj.is_supporting
            }
        elif isinstance(obj, Action):
            return {
                'type': obj.type.value,
                'args': obj.args,
                'status': obj.status,
                'result': str(obj.result) if obj.result else None,
                'error': obj.error
            }
        else:
            # Fallback: store __dict__
            return {k: str(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    
    def get(self, obj_id: str) -> Optional[Dict]:
        """Retrieve object by ID"""
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry['id'] == obj_id:
                        return entry
        return None
    
    def count(self) -> int:
        """Total events in log"""
        return len(self.index)


class FactualGraph:
    """Graph of verified facts only"""
    
    def __init__(self):
        self.facts: Dict[str, Claim] = {}  # id -> Claim
        self.evidence: Dict[str, Evidence] = {}  # id -> Evidence
        self.entities: Dict[str, Entity] = {}  # canonical_name -> Entity
        
        # Indexing
        self.subject_index: Dict[str, Set[str]] = defaultdict(set)  # subject -> claim_ids
        self.predicate_index: Dict[str, Set[str]] = defaultdict(set)  # predicate -> claim_ids
    
    def add_fact(self, claim: Claim) -> bool:
        """Add verified fact to graph"""
        if not claim.is_verified():
            logger.warning(f"   [WARN]  Cannot add unverified claim to factual graph: {claim}")
            return False
        
        self.facts[claim.id] = claim
        self.subject_index[claim.subject].add(claim.id)
        self.predicate_index[claim.predicate].add(claim.id)
        logger.info(f"   [v] Added fact: {claim}")
        return True
    
    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence to graph"""
        self.evidence[evidence.id] = evidence
    
    def get_facts_about(self, subject: str) -> List[Claim]:
        """Get all verified facts about a subject"""
        claim_ids = self.subject_index.get(subject, set())
        return [self.facts[cid] for cid in claim_ids if cid in self.facts]
    
    def get_facts_with_predicate(self, predicate: str) -> List[Claim]:
        """Get all facts with a specific predicate"""
        claim_ids = self.predicate_index.get(predicate, set())
        return [self.facts[cid] for cid in claim_ids if cid in self.facts]
    
    def has_contradiction(self, claim: Claim) -> bool:
        """Check if claim contradicts existing facts"""
        existing_facts = self.get_facts_about(claim.subject)
        for fact in existing_facts:
            if fact.predicate == claim.predicate and fact.object != claim.object:
                logger.warning(f"   [WARN]  Contradiction detected!")
                logger.warning(f"      Existing: {fact}")
                logger.warning(f"      New: {claim}")
                return True
        return False
    
    def count_facts(self) -> int:
        """Total verified facts"""
        return len(self.facts)


class SimulationArchive:
    """Store counterfactuals and simulations (never promote to facts)"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path / "simulations"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.counterfactuals: List[Counterfactual] = []
    
    def add_counterfactual(self, cf: Counterfactual) -> None:
        """Store counterfactual (immutable)"""
        self.counterfactuals.append(cf)
        
        # Save to disk
        cf_file = self.storage_path / f"cf_{cf.id}.json"
        with open(cf_file, 'w', encoding='utf-8') as f:
            json.dump({
                'id': cf.id,
                'scenario': cf.scenario,
                'actual_outcome': cf.actual_outcome,
                'hypothetical_outcome': cf.hypothetical_outcome,
                'human_valence': cf.human_valence,
                'importance': cf.importance,
                'surprise': cf.surprise
            }, f, indent=2)
        
        logger.info(f"   [THOUGHT] Stored counterfactual: {cf}")
    
    def get_important_counterfactuals(self, k: int = 5) -> List[Counterfactual]:
        """Get most important counterfactuals"""
        return sorted(self.counterfactuals, key=lambda cf: cf.importance, reverse=True)[:k]
    
    def count(self) -> int:
        return len(self.counterfactuals)


# ============================================================================
# Main Memory System
# ============================================================================

class StructuredMemory:
    """
    CORTANA Memory System
    
    - Bounded attention over infinite memory
    - Retrieval-based reasoning (not full history)
    - Strict promotion rules: hypothesis -> fact requires evidence
    """
    
    def __init__(
        self,
        storage_path: str = "cortana_memory",
        embedder=None,
        require_high_quality_embeddings: bool = False,
        allow_simple_embedder: bool = True,
        openai_api_key: Optional[str] = None,
        layercake_embedding_checkpoint: Optional[str] = None,
        layercake_tokenizer_path: Optional[str] = None,
        layercake_embedding_device: Optional[str] = None,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Memory stores
        self.event_log = EventLog(self.storage_path)
        self.factual_graph = FactualGraph()
        self.simulation_archive = SimulationArchive(self.storage_path)
        
        # Retrieval - use real embeddings if available
        if embedder is None:
            if EMBEDDINGS_AVAILABLE:
                cache_dir = self.storage_path / "embeddings_cache"
                cache_dir.mkdir(exist_ok=True)
                self.embedder = SmartEmbedder(
                    cache_dir=str(cache_dir),
                    openai_api_key=openai_api_key,
                    allow_simple_fallback=allow_simple_embedder,
                    require_high_quality=require_high_quality_embeddings,
                    layercake_checkpoint=layercake_embedding_checkpoint,
                    layercake_tokenizer_path=layercake_tokenizer_path,
                    layercake_device=layercake_embedding_device,
                )
                embedder_type = getattr(self.embedder, "embedder_type", "unknown")
                quality_tier = getattr(self.embedder, "quality_tier", "unknown")
                logger.info(f"   [BRAIN] Using SmartEmbedder ({quality_tier} quality, type={embedder_type})")
            else:
                if require_high_quality_embeddings:
                    raise RuntimeError(
                        "High-quality embeddings required, but cortana_embeddings is unavailable."
                    )
                self.embedder = SimpleEmbedder()
                logger.info(f"   [WARN]  Using SimpleEmbedder (low-quality fallback)")
        else:
            self.embedder = embedder
        
        # In-memory caches
        self.claims_cache: Dict[str, Claim] = {}
        self.evidence_cache: Dict[str, Evidence] = {}
        
        logger.info("[OK] Structured Memory initialized")
        logger.info(f"   Storage: {self.storage_path}")
    
    # ========================================================================
    # Core Operations
    # ========================================================================
    
    def store_claim(self, claim: Claim, evidence: Optional[Evidence] = None) -> None:
        """
        Store claim in memory
        
        Rules:
        1. Always write to event log
        2. If modality=FACT and has evidence -> add to factual graph
        3. If modality=SIMULATION -> send to simulation archive
        """
        # Write to event log (everything)
        self.event_log.append(claim)
        self.claims_cache[claim.id] = claim
        
        # Handle evidence
        if evidence:
            self.event_log.append(evidence)
            self.evidence_cache[evidence.id] = evidence
            
            # Link evidence to claim
            if evidence.id not in claim.evidence_ids:
                claim.evidence_ids.append(evidence.id)
            
            # NOTE: Do NOT auto-promote HYPOTHESIS->FACT here.
            # Promotion must go through the Verifier to preserve universe separation.
            # The caller (CortanaPipeline) is responsible for calling verifier.verify_claim()
            # and setting claim.modality = Modality.FACT only if verification passes.
            if evidence.confidence >= 0.8 and evidence.is_supporting:
                if claim.modality == Modality.HYPOTHESIS:
                    logger.info(f"   ℹ️  Strong evidence for hypothesis (promotion deferred to Verifier): {claim}")
        
        # Add to factual graph if verified
        if claim.modality == Modality.FACT and claim.is_verified():
            if not self.factual_graph.has_contradiction(claim):
                self.factual_graph.add_fact(claim)
                if evidence:
                    self.factual_graph.add_evidence(evidence)
            else:
                logger.error(f"   [FAIL] Cannot add contradictory fact!")
        
        # Simulations go to archive
        elif claim.modality == Modality.SIMULATION:
            # Don't add to factual graph, just log
            pass
    
    def store_action(self, action: Action) -> None:
        """Store action in event log"""
        self.event_log.append(action)
    
    def store_counterfactual(self, cf: Counterfactual) -> None:
        """Store counterfactual (never promotes to fact)"""
        self.event_log.append(cf)
        self.simulation_archive.add_counterfactual(cf)
    
    # ========================================================================
    # Retrieval (Bounded Attention)
    # ========================================================================
    
    def retrieve_claims(self, query: str, k: int = 5, 
                       facts_only: bool = False) -> List[Claim]:
        """
        Retrieve top-k relevant claims
        
        This implements BOUNDED ATTENTION:
        - We don't load entire history
        - We retrieve only what's relevant
        - Reasoning happens over retrieved state
        """
        query_embedding = self.embedder.embed(query)
        
        # Get candidates
        candidates = list(self.claims_cache.values())
        if facts_only:
            candidates = [c for c in candidates if c.is_verified()]
        
        # Score by relevance
        scored = []
        for claim in candidates:
            claim_text = f"{claim.subject} {claim.predicate} {claim.object}"
            score = self.embedder.similarity(query, claim_text)
            scored.append((score, claim))
        
        # Sort and return top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [claim for score, claim in scored[:k]]
        
        logger.info(f"   [SEARCH] Retrieved {len(results)} claims for query: '{query}'")
        for claim in results:
            logger.info(f"      * {claim}")
        
        return results
    
    def get_verified_facts(self, subject: Optional[str] = None) -> List[Claim]:
        """Get verified facts (optionally filtered by subject)"""
        if subject:
            return self.factual_graph.get_facts_about(subject)
        else:
            return list(self.factual_graph.facts.values())
    
    def get_evidence_for_claim(self, claim_id: str) -> List[Evidence]:
        """Get all evidence supporting a claim"""
        return [self.evidence_cache[eid] 
                for eid in self.claims_cache.get(claim_id, Claim("", "", "", Modality.HYPOTHESIS)).evidence_ids
                if eid in self.evidence_cache]
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def stats(self) -> Dict[str, Any]:
        """Memory statistics"""
        return {
            'total_events': self.event_log.count(),
            'verified_facts': self.factual_graph.count_facts(),
            'total_claims': len(self.claims_cache),
            'total_evidence': len(self.evidence_cache),
            'counterfactuals': self.simulation_archive.count()
        }
    
    def print_stats(self) -> None:
        """Print memory statistics"""
        stats = self.stats()
        logger.info("\n[STATS] Memory Statistics:")
        logger.info(f"   Total Events: {stats['total_events']}")
        logger.info(f"   Verified Facts: {stats['verified_facts']}")
        logger.info(f"   Total Claims: {stats['total_claims']}")
        logger.info(f"   Evidence: {stats['total_evidence']}")
        logger.info(f"   Counterfactuals: {stats['counterfactuals']}")


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
    print("CORTANA Structured Memory Demo")
    print("="*80 + "\n")
    
    # Initialize memory
    memory = StructuredMemory("demo_cortana_memory")
    
    # Create a hypothesis
    from .ir import create_claim, create_evidence, Modality, EvidenceSource
    
    claim1 = create_claim(
        subject="requests",
        predicate="requires_import",
        object="True",
        modality=Modality.HYPOTHESIS,
        confidence=0.7
    )
    memory.store_claim(claim1)
    
    # Add evidence -> promotes to fact
    evidence1 = create_evidence(
        claim=claim1,
        source=EvidenceSource.EXECUTION_TRACE,
        locator="script.py:2",
        content="NameError: name 'requests' is not defined",
        confidence=1.0
    )
    memory.store_claim(claim1, evidence1)
    
    # Create another fact
    claim2 = create_claim(
        subject="requests",
        predicate="is_http_library",
        object="True",
        modality=Modality.FACT,
        confidence=0.95
    )
    evidence2 = create_evidence(
        claim=claim2,
        source=EvidenceSource.API_RESPONSE,
        locator="https://pypi.org/project/requests/",
        content="requests is an HTTP library",
        confidence=1.0
    )
    memory.store_claim(claim2, evidence2)
    
    # Retrieve relevant claims
    print("\n" + "-"*80)
    results = memory.retrieve_claims("import http library", k=2, facts_only=True)
    
    # Stats
    print("\n" + "-"*80)
    memory.print_stats()

