"""
E5 Embeddings Microservice

Servizio indipendente per generare embeddings semantici usando E5.
Vettori a 384 dimensioni per semantic search e similarity.

Port: 5002
Model: agentlans/multilingual-e5-small-aligned-sentiment
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import logging
import os
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="E5 Embeddings Microservice",
    description="Servizio di embeddings semantici a 384 dimensioni",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class EmbeddingRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Testo da convertire in embedding"
    )
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Il testo non può essere vuoto')
        return v.strip()


class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Lista di testi da convertire (max 100)"
    )
    
    @validator('texts')
    def texts_not_empty(cls, v):
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError('Almeno un testo deve essere non vuoto')
        return cleaned


class SimilarityRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Testo query")
    candidates: List[str] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="Lista di testi candidati (max 1000)"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Numero di risultati da ritornare"
    )


class EmbeddingResponse(BaseModel):
    embedding: List[float] = Field(..., description="Vettore 384-dim normalizzato")
    dimension: int = Field(default=384, description="Dimensione embedding")
    processing_time_ms: Optional[float] = Field(None, description="Tempo elaborazione")


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int = Field(default=384)
    total_processed: int
    total_time_ms: float


class SimilarityResult(BaseModel):
    text: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)


class SimilarityResponse(BaseModel):
    query: str
    results: List[SimilarityResult]
    total_candidates: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str
    embedding_dimension: int


class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    task: str
    embedding_dimension: int
    languages: str
    parameters: str
    device: str
    max_input_length: int
    use_cases: List[str]

# ============================================
# E5 EMBEDDINGS MODEL MANAGER (Singleton)
# ============================================

class E5EmbeddingsModel:
    """
    Singleton per gestire il modello E5.
    
    Features:
    - Lazy loading
    - Mean pooling + L2 normalization
    - Batch processing ottimizzato
    - Cosine similarity search
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model_name = os.getenv(
            'MODEL_NAME',
            'agentlans/multilingual-e5-small-aligned-sentiment'
        )
        
        # Determina device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("="*60)
        logger.info("🚀 Inizializzazione E5 Embeddings Service")
        logger.info(f"📦 Model: {self.model_name}")
        logger.info(f"🖥️  Device: {self.device}")
        
        try:
            # Carica tokenizer
            logger.info("📥 Caricamento tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Carica model
            logger.info("📥 Caricamento modello...")
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            
            # Modalità evaluation
            self.model.eval()
            
            # Embedding dimension
            self.embedding_dim = 384
            
            self._initialized = True
            
            logger.info("✅ Modello caricato con successo!")
            logger.info(f"📊 Embedding dimension: {self.embedding_dim}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento modello: {e}")
            raise
    
    @staticmethod
    def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling per ottenere sentence embedding da token embeddings.
        
        Args:
            token_embeddings: Output layer del transformer [batch, seq_len, hidden]
            attention_mask: Maschera attenzione [batch, seq_len]
            
        Returns:
            Sentence embeddings [batch, hidden]
        """
        # Espandi attention mask
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        
        # Sum over tokens, weighted by mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum of mask (evita divisione per zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Mean pooling
        return sum_embeddings / sum_mask
    
    def encode(self, text: str) -> np.ndarray:
        """
        Genera embedding per un singolo testo.
        
        Args:
            text: Testo da codificare
            
        Returns:
            Numpy array [384] normalizzato
        """
        start_time = time.time()
        
        try:
            # E5 richiede il prefisso "query: " per le query
            # (o "passage: " per i documenti, ma usiamo query per uniformità)
            prefixed_text = f"query: {text}"
            
            # Tokenization
            inputs = self.tokenizer(
                prefixed_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Mean pooling
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs['attention_mask']
                )
                
                # L2 normalization (per cosine similarity)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            processing_time = (time.time() - start_time) * 1000
            
            return embeddings[0].cpu().numpy(), processing_time
            
        except Exception as e:
            logger.error(f"Errore durante encoding: {e}")
            raise
    
    def batch_encode(self, texts: List[str]) -> tuple[np.ndarray, float]:
        """
        Genera embeddings per batch di testi.
        
        Args:
            texts: Lista di testi
            
        Returns:
            Tuple (embeddings array [N, 384], processing_time_ms)
        """
        start_time = time.time()
        
        try:
            # Aggiungi prefisso
            prefixed_texts = [f"query: {text}" for text in texts]
            
            # Batch tokenization
            inputs = self.tokenizer(
                prefixed_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Mean pooling
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs['attention_mask']
                )
                
                # L2 normalization
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            processing_time = (time.time() - start_time) * 1000
            
            return embeddings.cpu().numpy(), processing_time
            
        except Exception as e:
            logger.error(f"Errore durante batch encoding: {e}")
            raise
    
    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Trova i candidati più simili alla query usando cosine similarity.
        
        Args:
            query: Testo query
            candidates: Lista di testi candidati
            top_k: Numero di risultati
            
        Returns:
            Tuple (results list, processing_time_ms)
        """
        start_time = time.time()
        
        try:
            # Embedding query
            query_emb, _ = self.encode(query)
            
            # Embeddings candidati (batch per efficienza)
            candidate_embs, _ = self.batch_encode(candidates)
            
            # Calcola cosine similarity
            # (embeddings già normalizzati, quindi dot product = cosine similarity)
            similarities = np.dot(candidate_embs, query_emb)
            
            # Top-k indices (ordine decrescente)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Costruisci risultati
            results = []
            for rank, idx in enumerate(top_indices, 1):
                results.append({
                    'text': candidates[idx],
                    'similarity': float(similarities[idx]),
                    'rank': rank
                })
            
            processing_time = (time.time() - start_time) * 1000
            
            return results, processing_time
            
        except Exception as e:
            logger.error(f"Errore durante similarity search: {e}")
            raise

# ============================================
# GLOBAL MODEL INSTANCE
# ============================================

e5_model: Optional[E5EmbeddingsModel] = None

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Inizializza il modello all'avvio"""
    global e5_model
    
    logger.info("🚀 Starting E5 Embeddings Microservice...")
    
    try:
        e5_model = E5EmbeddingsModel()
        logger.info("✅ Service ready and listening on port 5002")
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup al shutdown"""
    logger.info("🛑 Shutting down E5 Embeddings Microservice...")
    
    # Cleanup GPU memory
    if e5_model and e5_model.device == "cuda":
        torch.cuda.empty_cache()
    
    logger.info("✅ Shutdown complete")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", tags=["Health"])
def root():
    """Root endpoint - info sul servizio"""
    return {
        "service": "E5 Embeddings Microservice",
        "version": "1.0.0",
        "model": e5_model.model_name if e5_model else "Not loaded",
        "device": e5_model.device if e5_model else "Unknown",
        "embedding_dimension": 384,
        "status": "running" if e5_model and e5_model._initialized else "initializing",
        "endpoints": {
            "embed": "POST /embed",
            "batch_embed": "POST /batch-embed",
            "similarity": "POST /similarity",
            "health": "GET /health",
            "info": "GET /info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check per monitoring"""
    if not e5_model or not e5_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=e5_model.device,
        model_name=e5_model.model_name,
        embedding_dimension=e5_model.embedding_dim
    )


@app.post("/embed", response_model=EmbeddingResponse, tags=["Embeddings"])
def generate_embedding(request: EmbeddingRequest):
    """
    Genera embedding per un singolo testo.
    
    **Output:**
    - embedding: Vettore 384-dim normalizzato (L2 norm = 1)
    - dimension: 384
    - processing_time_ms: Tempo elaborazione
    
    **Use case:**
    - Convertire testo in rappresentazione vettoriale
    - Preparare query per similarity search
    
    **Example:**
    ```json
    {
      "text": "Hello world"
    }
    ```
    
    **Response:**
    ```json
    {
      "embedding": [0.234, -0.456, 0.123, ...],
      "dimension": 384,
      "processing_time_ms": 28.5
    }
    ```
    """
    if not e5_model or not e5_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    try:
        embedding, proc_time = e5_model.encode(request.text)
        
        return EmbeddingResponse(
            embedding=embedding.tolist(),
            dimension=len(embedding),
            processing_time_ms=round(proc_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during embedding generation: {str(e)}"
        )


@app.post("/batch-embed", response_model=BatchEmbeddingResponse, tags=["Embeddings"])
def batch_generate_embeddings(request: BatchEmbeddingRequest):
    """
    Genera embeddings per batch di testi (più efficiente).
    
    **Limiti:**
    - Minimo: 1 testo
    - Massimo: 100 testi
    
    **Performance:**
    - Batch processing è ~8x più veloce
    
    **Example:**
    ```json
    {
      "texts": ["Hello", "World", "AI is amazing"]
    }
    ```
    
    **Response:**
    ```json
    {
      "embeddings": [
        [0.1, 0.2, ...],
        [0.3, 0.4, ...],
        [0.5, 0.6, ...]
      ],
      "dimension": 384,
      "total_processed": 3,
      "total_time_ms": 95.2
    }
    ```
    """
    if not e5_model or not e5_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    try:
        embeddings, proc_time = e5_model.batch_encode(request.texts)
        
        return BatchEmbeddingResponse(
            embeddings=embeddings.tolist(),
            dimension=embeddings.shape[1],
            total_processed=len(embeddings),
            total_time_ms=round(proc_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error in batch embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during batch embedding: {str(e)}"
        )


@app.post("/similarity", response_model=SimilarityResponse, tags=["Search"])
def find_similar(request: SimilarityRequest):
    """
    Trova i testi più simili alla query usando cosine similarity.
    
    **Come funziona:**
    1. Converte query in embedding
    2. Converte tutti i candidati in embeddings
    3. Calcola cosine similarity tra query e candidati
    4. Ritorna top-k risultati ordinati
    
    **Limiti:**
    - Max 1000 candidati
    - Max 50 risultati (top_k)
    
    **Use cases:**
    - Semantic search in documenti
    - "Find similar messages"
    - Recommendation systems
    
    **Example:**
    ```json
    {
      "query": "We need to make a decision",
      "candidates": [
        "Let's finalize this by Friday",
        "Great work everyone",
        "I think we should move forward"
      ],
      "top_k": 2
    }
    ```
    
    **Response:**
    ```json
    {
      "query": "We need to make a decision",
      "results": [
        {
          "text": "I think we should move forward",
          "similarity": 0.87,
          "rank": 1
        },
        {
          "text": "Let's finalize this by Friday",
          "similarity": 0.82,
          "rank": 2
        }
      ],
      "total_candidates": 3,
      "processing_time_ms": 145.3
    }
    ```
    """
    if not e5_model or not e5_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    try:
        results, proc_time = e5_model.find_similar(
            query=request.query,
            candidates=request.candidates,
            top_k=request.top_k
        )
        
        return SimilarityResponse(
            query=request.query,
            results=[SimilarityResult(**r) for r in results],
            total_candidates=len(request.candidates),
            processing_time_ms=round(proc_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during similarity search: {str(e)}"
        )


@app.get("/info", response_model=ModelInfoResponse, tags=["Info"])
def model_info():
    """Informazioni dettagliate sul modello"""
    if not e5_model or not e5_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    return ModelInfoResponse(
        model_name=e5_model.model_name,
        architecture="E5-small (12 layers, 384 hidden)",
        task="Semantic Embeddings",
        embedding_dimension=384,
        languages="100+ languages",
        parameters="~117M",
        device=e5_model.device,
        max_input_length=512,
        use_cases=[
            "Semantic search",
            "Text similarity",
            "Clustering",
            "Recommendations",
            "Cross-lingual retrieval"
        ]
    )

# ============================================
# MAIN (per test locali)
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 5002))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )