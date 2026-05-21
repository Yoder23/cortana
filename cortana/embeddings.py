"""
Real Semantic Embeddings for CORTANA

Replaces SimpleEmbedder with actual semantic embeddings for high-quality retrieval.

Options:
1. Sentence-Transformers (local, free, good quality)
2. LayerCake Semantic Embedder (local, checkpoint-based)
3. OpenAI Embeddings (API-based, best quality, requires key)
4. Custom domain-specific embeddings (fine-tuned)

We implement all three with automatic fallback.
"""

import logging
import hashlib
import pickle
from typing import List, Optional, Dict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Base Embedder Interface
# ============================================================================

class BaseEmbedder:
    """Base class for all embedders"""
    
    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        raise NotImplementedError
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Convert batch of texts to embeddings"""
        return np.array([self.embed(text) for text in texts])
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


# ============================================================================
# Sentence-Transformers Embedder (LOCAL, HIGH QUALITY)
# ============================================================================

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Use sentence-transformers for high-quality local embeddings.
    
    Models:
    - all-MiniLM-L6-v2: Fast, 384-dim, good quality
    - all-mpnet-base-v2: Slower, 768-dim, best quality
    - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A retrieval
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize sentence-transformers embedder.
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Optional cache directory for embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
            self.cache_dir = Path(cache_dir) if cache_dir else None
            self.cache: Dict[str, np.ndarray] = {}
            
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._load_cache()
            
            logger.info(f"✅ Sentence-Transformers Embedder initialized")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Dimensions: {self.dim}")
            logger.info(f"   Cache: {self.cache_dir or 'disabled'}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text (with caching)"""
        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Compute embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Cache it
        self.cache[cache_key] = embedding
        if self.cache_dir:
            self._save_to_cache(cache_key, embedding)
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding (more efficient than individual)"""
        # Check which are cached
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                results[i] = self.cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Compute uncached
        if uncached_texts:
            embeddings = self.model.encode(uncached_texts, convert_to_numpy=True, show_progress_bar=False)
            for idx, text, emb in zip(uncached_indices, uncached_texts, embeddings):
                results[idx] = emb
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = emb
                if self.cache_dir:
                    self._save_to_cache(cache_key, emb)
        
        return np.array(results)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk"""
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"   Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"   Failed to load cache: {e}")
    
    def _save_to_cache(self, key: str, embedding: np.ndarray):
        """Save single embedding to cache"""
        # Periodically save full cache (every 100 items)
        if len(self.cache) % 100 == 0:
            cache_file = self.cache_dir / "embeddings_cache.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                logger.warning(f"   Failed to save cache: {e}")


# ============================================================================
# OpenAI Embedder (API-BASED, BEST QUALITY)
# ============================================================================

class OpenAIEmbedder(BaseEmbedder):
    """
    Use OpenAI's embeddings API for best quality.
    
    Models:
    - text-embedding-3-small: 1536-dim, fast, cheap
    - text-embedding-3-large: 3072-dim, best quality
    - text-embedding-ada-002: 1536-dim, older model
    """
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize OpenAI embedder.
        
        Args:
            model: OpenAI model name
            api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        """
        try:
            import openai
            if api_key:
                openai.api_key = api_key
            
            self.model = model
            self.client = openai
            
            # Test API
            test_emb = self.embed("test")
            self.dim = len(test_emb)
            
            logger.info(f"✅ OpenAI Embedder initialized")
            logger.info(f"   Model: {model}")
            logger.info(f"   Dimensions: {self.dim}")
            
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install with: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API initialization failed: {e}")
    
    def embed(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API"""
        response = self.client.Embedding.create(
            model=self.model,
            input=text
        )
        return np.array(response['data'][0]['embedding'])
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding (more efficient)"""
        response = self.client.Embedding.create(
            model=self.model,
            input=texts
        )
        return np.array([item['embedding'] for item in response['data']])


# ============================================================================
# LayerCake Semantic Embedder (LOCAL, HIGH QUALITY)
# ============================================================================

class LayerCakeSemanticEmbedder(BaseEmbedder):
    """
    Use a trained LayerCake core checkpoint to produce semantic ABI embeddings.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
        max_seq_len: int = 256,
        cache_dir: Optional[str] = None,
    ):
        try:
            import torch
            import sentencepiece as spm
        except Exception as e:
            raise RuntimeError(f"LayerCake embedder requires torch + sentencepiece: {e}")

        from layercake_model_fixed_abi import LayerCakeLMFixedABI

        self._torch = torch
        self.max_seq_len = int(max_seq_len)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache: Dict[str, np.ndarray] = {}

        base_dir = Path(__file__).resolve().parent

        if checkpoint_path is None:
            candidates = [
                base_dir / "runs/fluent_core_250k/v6_core_seed6000/core_v6.pt",
                base_dir / "runs/c4_fluent_core/v6_core_seed6000/core_v6.pt",
            ]
            checkpoint_path = next((str(p) for p in candidates if p.exists()), None)
        if tokenizer_path is None:
            tok_candidate = base_dir / "tokenizer/layercake_sp.model"
            tokenizer_path = str(tok_candidate) if tok_candidate.exists() else None

        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise RuntimeError(
                "LayerCakeSemanticEmbedder could not locate a valid core checkpoint."
            )
        if not tokenizer_path or not Path(tokenizer_path).exists():
            raise RuntimeError(
                "LayerCakeSemanticEmbedder could not locate a valid SentencePiece tokenizer."
            )

        self.checkpoint_path = str(Path(checkpoint_path))
        self.tokenizer_path = str(Path(tokenizer_path))

        self.tokenizer = spm.SentencePieceProcessor()
        if not self.tokenizer.load(self.tokenizer_path):
            raise RuntimeError(f"Failed to load tokenizer: {self.tokenizer_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        d_model = int(checkpoint.get("d_model", 512))
        d_abi = int(checkpoint.get("d_abi", 512))
        vocab_size = int(checkpoint.get("vocab_size", 16000))
        seq_len = int(checkpoint.get("seq_len", 256))
        state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        self.model = LayerCakeLMFixedABI(
            vocab_size=vocab_size,
            d_model=d_model,
            d_abi=d_abi,
            n_core_layers=6,
            n_heads=8,
            d_ff=2048,
            max_seq_len=seq_len,
            domain_names=("general",),
            use_router=False,
        ).to(self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.dim = d_abi

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

        logger.info("✅ LayerCake Semantic Embedder initialized")
        logger.info(f"   Checkpoint: {self.checkpoint_path}")
        logger.info(f"   Tokenizer:  {self.tokenizer_path}")
        logger.info(f"   Dimensions: {self.dim}")
        logger.info(f"   Device:     {self.device}")

    def embed(self, text: str) -> np.ndarray:
        key = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        if key in self.cache:
            return self.cache[key]

        ids = self.tokenizer.encode(text)
        if not ids:
            unk = int(self.tokenizer.unk_id())
            ids = [unk if unk >= 0 else 0]
        ids = ids[: self.max_seq_len]

        with self._torch.no_grad():
            input_ids = self._torch.tensor([ids], dtype=self._torch.long, device=self.device)
            _, h_abi = self.model.encode_core(input_ids)
            emb = h_abi.mean(dim=1).squeeze(0).float().cpu().numpy()

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        emb = emb.astype(np.float32)

        self.cache[key] = emb
        if self.cache_dir:
            self._save_to_cache()
        return emb

    def _load_cache(self):
        cache_file = self.cache_dir / "layercake_embeddings_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                logger.info(f"   Loaded {len(self.cache)} LayerCake cached embeddings")
            except Exception as e:
                logger.warning(f"   Failed to load LayerCake cache: {e}")

    def _save_to_cache(self):
        if len(self.cache) % 100 != 0:
            return
        cache_file = self.cache_dir / "layercake_embeddings_cache.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"   Failed to save LayerCake cache: {e}")


# ============================================================================
# Simple Character-Level Embedder (FALLBACK)
# ============================================================================

class SimpleEmbedder(BaseEmbedder):
    """
    Fallback embedder using character-level hashing.
    
    This is what we had before. It works but has low quality.
    Only used if sentence-transformers and OpenAI are unavailable.
    """
    
    def __init__(self, dim: int = 128):
        """
        Initialize simple embedder.
        
        Args:
            dim: Embedding dimension
        """
        self.dim = dim
        logger.info(f"⚠️  Simple Embedder initialized (FALLBACK)")
        logger.info(f"   Quality: LOW - consider installing sentence-transformers")
        logger.info(f"   Dimensions: {dim}")
    
    def embed(self, text: str) -> np.ndarray:
        """Simple character-level embedding"""
        # Deterministic hash-based embedding
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dim)
        
        # Add character features
        for i, char in enumerate(text[:self.dim]):
            embedding[i % self.dim] += ord(char) / 128.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


# ============================================================================
# Smart Embedder (AUTO-SELECTS BEST AVAILABLE)
# ============================================================================

class SmartEmbedder(BaseEmbedder):
    """
    Automatically selects the best available embedder.
    
    Priority:
    1. Sentence-Transformers (if installed)
    2. LayerCake semantic embedder (if checkpoint/tokenizer available)
    3. OpenAI (if API key available)
    4. Simple (fallback)
    """
    
    def __init__(self, 
                 prefer_local: bool = True,
                 openai_api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 allow_simple_fallback: bool = True,
                 require_high_quality: bool = False,
                 layercake_checkpoint: Optional[str] = None,
                 layercake_tokenizer_path: Optional[str] = None,
                 layercake_device: Optional[str] = None):
        """
        Initialize smart embedder with automatic selection.
        
        Args:
            prefer_local: Prefer local models over API
            openai_api_key: Optional OpenAI API key
            cache_dir: Optional cache directory
            allow_simple_fallback: Allow low-quality fallback embedder
            require_high_quality: Fail closed unless high-quality embedder is available
        """
        self.embedder = None
        self.embedder_type = None
        
        # Try Sentence-Transformers first if preferred
        if prefer_local:
            try:
                self.embedder = SentenceTransformerEmbedder(cache_dir=cache_dir)
                self.embedder_type = "sentence-transformers"
                self.dim = self.embedder.dim
                logger.info("✅ Using Sentence-Transformers (local, high quality)")
                return
            except (ImportError, Exception) as e:
                logger.warning(f"   Sentence-Transformers unavailable: {e}")

            try:
                self.embedder = LayerCakeSemanticEmbedder(
                    checkpoint_path=layercake_checkpoint,
                    tokenizer_path=layercake_tokenizer_path,
                    device=layercake_device,
                    cache_dir=cache_dir,
                )
                self.embedder_type = "layercake-semantic"
                self.dim = self.embedder.dim
                logger.info("✅ Using LayerCake Semantic Embedder (local, high quality)")
                return
            except Exception as e:
                logger.warning(f"   LayerCake semantic embedder unavailable: {e}")
        
        # Try OpenAI if API key provided
        if openai_api_key:
            try:
                self.embedder = OpenAIEmbedder(api_key=openai_api_key)
                self.embedder_type = "openai"
                self.dim = self.embedder.dim
                logger.info("✅ Using OpenAI Embeddings (API, best quality)")
                return
            except (ImportError, Exception) as e:
                logger.warning(f"   OpenAI unavailable: {e}")
        
        if require_high_quality:
            raise RuntimeError(
                "High-quality embeddings required, but neither sentence-transformers "
                "nor OpenAI embeddings are available."
            )

        if not allow_simple_fallback:
            raise RuntimeError(
                "Simple embedder fallback disabled and no high-quality embedder available."
            )

        # Fallback to simple embedder
        self.embedder = SimpleEmbedder()
        self.embedder_type = "simple"
        self.dim = self.embedder.dim
        logger.warning("⚠️  Using Simple Embedder (low quality fallback)")
        logger.warning("   Install sentence-transformers for better results:")
        logger.warning("   pip install sentence-transformers")
    
    def embed(self, text: str) -> np.ndarray:
        """Delegate to selected embedder"""
        return self.embedder.embed(text)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Delegate to selected embedder"""
        return self.embedder.embed_batch(texts)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Delegate to selected embedder"""
        return self.embedder.similarity(text1, text2)

    @property
    def quality_tier(self) -> str:
        """Return embedding quality tier for runtime readiness checks."""
        if self.embedder_type in {"sentence-transformers", "openai", "layercake-semantic"}:
            return "high"
        return "low"

    @property
    def is_high_quality(self) -> bool:
        """True when semantic retrieval quality is production-grade."""
        return self.quality_tier == "high"


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("REAL EMBEDDINGS DEMO")
    print("="*80 + "\n")
    
    # Test smart embedder
    embedder = SmartEmbedder(cache_dir="embeddings_cache")
    
    print("\nDemo 1: Single Text Embedding")
    print("-"*80)
    text1 = "The API was deployed to production"
    emb1 = embedder.embed(text1)
    print(f"Text: {text1}")
    print(f"Embedding shape: {emb1.shape}")
    print(f"Embedding norm: {np.linalg.norm(emb1):.4f}")
    
    print("\nDemo 2: Batch Embedding")
    print("-"*80)
    texts = [
        "NumPy is a Python library",
        "Knights move in L-shape",
        "The test passed successfully"
    ]
    embs = embedder.embed_batch(texts)
    print(f"Batch size: {len(texts)}")
    print(f"Embeddings shape: {embs.shape}")
    
    print("\nDemo 3: Semantic Similarity")
    print("-"*80)
    text_a = "The deployment was successful"
    text_b = "Successfully deployed the system"
    text_c = "Knights move in chess"
    
    sim_ab = embedder.similarity(text_a, text_b)
    sim_ac = embedder.similarity(text_a, text_c)
    
    print(f"Text A: {text_a}")
    print(f"Text B: {text_b}")
    print(f"Text C: {text_c}")
    print(f"\nSimilarity A-B (similar meaning): {sim_ab:.4f}")
    print(f"Similarity A-C (different meaning): {sim_ac:.4f}")
    
    print("\n" + "="*80)
    print("✓ REAL EMBEDDINGS OPERATIONAL")
    print("="*80 + "\n")
