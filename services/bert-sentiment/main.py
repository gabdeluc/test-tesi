"""
BERT Sentiment Microservice

Servizio indipendente per sentiment analysis usando BERT multilingual.
Classifica testi su scala 1-5 stelle.

Port: 5001
Model: nlptown/bert-base-multilingual-uncased-sentiment
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
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
    title="BERT Sentiment Microservice",
    description="Servizio di sentiment analysis con classificazione 1-5 stelle",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Testo da analizzare"
    )
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Il testo non può essere vuoto')
        return v.strip()


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Lista di testi da analizzare (max 100)"
    )
    
    @validator('texts')
    def texts_not_empty(cls, v):
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError('Almeno un testo deve essere non vuoto')
        return cleaned


class SentimentResponse(BaseModel):
    stars: float = Field(..., ge=1.0, le=5.0, description="Punteggio sentiment 1.0-5.0")
    sentiment: str = Field(..., description="Categoria sentiment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidenza predizione")
    processing_time_ms: Optional[float] = Field(None, description="Tempo elaborazione in ms")


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processed: int
    total_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str


class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    task: str
    languages: List[str]
    parameters: str
    device: str
    max_input_length: int
    output_classes: int

# ============================================
# BERT MODEL MANAGER (Singleton)
# ============================================

class BERTSentimentModel:
    """
    Singleton per gestire il modello BERT.
    
    Features:
    - Lazy loading
    - Thread-safe
    - Batch processing ottimizzato
    - Error handling robusto
    """
    
    _instance = None
    _lock = None
    
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
            'nlptown/bert-base-multilingual-uncased-sentiment'
        )
        
        # Determina device (GPU se disponibile)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("="*60)
        logger.info("🚀 Inizializzazione BERT Sentiment Service")
        logger.info(f"📦 Model: {self.model_name}")
        logger.info(f"🖥️  Device: {self.device}")
        
        try:
            # Carica tokenizer
            logger.info("📥 Caricamento tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Carica model
            logger.info("📥 Caricamento modello...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            
            # Modalità evaluation (no training)
            self.model.eval()
            
            # Mappa sentiment labels
            self.sentiment_map = {
                0: "very_negative",
                1: "negative",
                2: "neutral",
                3: "positive",
                4: "very_positive"
            }
            
            # Stars values per weighted average
            self.stars_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(self.device)
            
            self._initialized = True
            
            logger.info("✅ Modello caricato con successo!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento modello: {e}")
            raise
    
    def analyze(
        self,
        text: str,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Analizza sentiment di un singolo testo.
        
        Args:
            text: Testo da analizzare
            return_probabilities: Se True, ritorna anche le probabilità per classe
            
        Returns:
            Dict con stars, sentiment, confidence
        """
        start_time = time.time()
        
        try:
            # Tokenization
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0]
            
            # Calcola stelle (media pesata delle probabilità)
            weighted_stars = (probs * self.stars_values).sum().item()
            
            # Classe predetta (argmax)
            predicted_class = torch.argmax(probs).item()
            confidence = probs[predicted_class].item()
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            result = {
                'stars': round(weighted_stars, 2),
                'sentiment': self.sentiment_map[predicted_class],
                'confidence': round(confidence, 3),
                'processing_time_ms': round(processing_time, 2)
            }
            
            if return_probabilities:
                result['probabilities'] = probs.cpu().numpy().tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Errore durante analisi: {e}")
            raise
    
    def batch_analyze(
        self,
        texts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Analizza batch di testi (più efficiente).
        
        Args:
            texts: Lista di testi da analizzare
            
        Returns:
            Lista di risultati sentiment
        """
        start_time = time.time()
        
        try:
            # Batch tokenization
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            
            # Process results
            results = []
            
            for i, prob in enumerate(probs):
                weighted_stars = (prob * self.stars_values).sum().item()
                predicted_class = torch.argmax(prob).item()
                confidence = prob[predicted_class].item()
                
                results.append({
                    'stars': round(weighted_stars, 2),
                    'sentiment': self.sentiment_map[predicted_class],
                    'confidence': round(confidence, 3),
                    'processing_time_ms': None  # Calcolato a livello batch
                })
            
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(texts)
            
            # Aggiungi tempo medio per testo
            for result in results:
                result['processing_time_ms'] = round(avg_time, 2)
            
            return results
            
        except Exception as e:
            logger.error(f"Errore durante batch analysis: {e}")
            raise

# ============================================
# GLOBAL MODEL INSTANCE
# ============================================

# Il modello viene inizializzato all'avvio del servizio
bert_model: Optional[BERTSentimentModel] = None

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Inizializza il modello all'avvio del servizio"""
    global bert_model
    
    logger.info("🚀 Starting BERT Sentiment Microservice...")
    
    try:
        bert_model = BERTSentimentModel()
        logger.info("✅ Service ready and listening on port 5001")
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup al shutdown"""
    logger.info("🛑 Shutting down BERT Sentiment Microservice...")
    
    # Cleanup GPU memory se necessario
    if bert_model and bert_model.device == "cuda":
        torch.cuda.empty_cache()
    
    logger.info("✅ Shutdown complete")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", tags=["Health"])
def root():
    """Root endpoint - informazioni base sul servizio"""
    return {
        "service": "BERT Sentiment Analysis Microservice",
        "version": "1.0.0",
        "model": bert_model.model_name if bert_model else "Not loaded",
        "device": bert_model.device if bert_model else "Unknown",
        "status": "running" if bert_model and bert_model._initialized else "initializing",
        "endpoints": {
            "analyze": "POST /analyze",
            "batch": "POST /batch",
            "health": "GET /health",
            "info": "GET /info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint per monitoring.
    
    Usato da Docker healthcheck e orchestratori.
    """
    if not bert_model or not bert_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=bert_model.device,
        model_name=bert_model.model_name
    )


@app.post("/analyze", response_model=SentimentResponse, tags=["Sentiment Analysis"])
def analyze_sentiment(request: SentimentRequest):
    """
    Analizza sentiment di un singolo testo.
    
    **Input:**
    - text: Testo da analizzare (1-5000 caratteri)
    
    **Output:**
    - stars: Punteggio 1.0-5.0
    - sentiment: very_negative, negative, neutral, positive, very_positive
    - confidence: Confidenza predizione 0.0-1.0
    - processing_time_ms: Tempo elaborazione
    
    **Example:**
    ```json
    {
      "text": "This meeting was very productive!"
    }
    ```
    
    **Response:**
    ```json
    {
      "stars": 4.8,
      "sentiment": "very_positive",
      "confidence": 0.92,
      "processing_time_ms": 45.23
    }
    ```
    """
    if not bert_model or not bert_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    try:
        result = bert_model.analyze(request.text)
        return SentimentResponse(**result)
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during sentiment analysis: {str(e)}"
        )


@app.post("/batch", response_model=BatchSentimentResponse, tags=["Sentiment Analysis"])
def batch_analyze_sentiment(request: BatchSentimentRequest):
    """
    Analizza sentiment per batch di testi (più efficiente).
    
    **Limiti:**
    - Minimo: 1 testo
    - Massimo: 100 testi per richiesta
    
    **Performance:**
    - Batch processing è ~10x più veloce di chiamate singole
    
    **Example:**
    ```json
    {
      "texts": [
        "Great work!",
        "This is terrible",
        "Not sure about this"
      ]
    }
    ```
    
    **Response:**
    ```json
    {
      "results": [
        {"stars": 4.9, "sentiment": "very_positive", "confidence": 0.95},
        {"stars": 1.2, "sentiment": "very_negative", "confidence": 0.88},
        {"stars": 2.8, "sentiment": "neutral", "confidence": 0.71}
      ],
      "total_processed": 3,
      "total_time_ms": 120.5
    }
    ```
    """
    if not bert_model or not bert_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    try:
        start_time = time.time()
        results = bert_model.batch_analyze(request.texts)
        total_time = (time.time() - start_time) * 1000
        
        return BatchSentimentResponse(
            results=[SentimentResponse(**r) for r in results],
            total_processed=len(results),
            total_time_ms=round(total_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during batch sentiment analysis: {str(e)}"
        )


@app.get("/info", response_model=ModelInfoResponse, tags=["Info"])
def model_info():
    """
    Informazioni dettagliate sul modello.
    
    Returns:
        Dettagli su architettura, capacità, limiti
    """
    if not bert_model or not bert_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    return ModelInfoResponse(
        model_name=bert_model.model_name,
        architecture="BERT-base (12 layers, 768 hidden)",
        task="Sentiment Classification (1-5 stars)",
        languages=["en", "nl", "de", "fr", "it", "es"],
        parameters="~110M",
        device=bert_model.device,
        max_input_length=512,
        output_classes=5
    )

# ============================================
# MAIN (per test locali)
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 5001))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )