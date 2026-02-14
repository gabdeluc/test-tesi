"""
BERT Toxicity Microservice

Servizio indipendente per toxicity analysis usando BERT.
Classifica testi su scala di tossicità 0-1.

Port: 5003
Model: gravitee-io/bert-small-toxicity
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
    title="BERT Toxicity Microservice",
    description="Servizio di toxicity detection con classificazione 0-1",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class ToxicityRequest(BaseModel):
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


class BatchToxicityRequest(BaseModel):
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


class ToxicityResponse(BaseModel):
    toxicity_score: float = Field(..., ge=0.0, le=1.0, description="Score tossicità 0.0-1.0")
    is_toxic: bool = Field(..., description="True se tossico (score > 0.5)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidenza predizione")
    label: str = Field(..., description="toxic/non-toxic")
    processing_time_ms: Optional[float] = Field(None, description="Tempo elaborazione in ms")


class BatchToxicityResponse(BaseModel):
    results: List[ToxicityResponse]
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
    output_range: str
    device: str
    max_input_length: int
    threshold: float

# ============================================
# BERT TOXICITY MODEL MANAGER (Singleton)
# ============================================

class BERTToxicityModel:
    """
    Singleton per gestire il modello BERT Toxicity.
    
    Features:
    - Lazy loading
    - Thread-safe
    - Batch processing ottimizzato
    - Binary classification (toxic/non-toxic)
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
            'gravitee-io/bert-small-toxicity'
        )
        
        # Determina device (GPU se disponibile)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Threshold per classificazione binaria
        self.threshold = 0.5
        
        logger.info("="*60)
        logger.info("🚀 Inizializzazione BERT Toxicity Service")
        logger.info(f"📦 Model: {self.model_name}")
        logger.info(f"🖥️  Device: {self.device}")
        logger.info(f"🎯 Threshold: {self.threshold}")
        
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
            
            # Label mapping (0=non-toxic, 1=toxic)
            self.id2label = {0: "non-toxic", 1: "toxic"}
            
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
        Analizza tossicità di un singolo testo.
        
        Args:
            text: Testo da analizzare
            return_probabilities: Se True, ritorna anche le probabilità per classe
            
        Returns:
            Dict con toxicity_score, is_toxic, confidence, label
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
            
            # Score tossicità (probabilità classe "toxic")
            toxicity_score = probs[1].item()
            
            # Classificazione binaria
            is_toxic = toxicity_score > self.threshold
            predicted_class = 1 if is_toxic else 0
            
            # Confidence = probabilità della classe predetta
            confidence = probs[predicted_class].item()
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            result = {
                'toxicity_score': round(toxicity_score, 3),
                'is_toxic': is_toxic,
                'confidence': round(confidence, 3),
                'label': self.id2label[predicted_class],
                'processing_time_ms': round(processing_time, 2)
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    'non-toxic': round(probs[0].item(), 3),
                    'toxic': round(probs[1].item(), 3)
                }
            
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
            Lista di risultati toxicity
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
                toxicity_score = prob[1].item()
                is_toxic = toxicity_score > self.threshold
                predicted_class = 1 if is_toxic else 0
                confidence = prob[predicted_class].item()
                
                results.append({
                    'toxicity_score': round(toxicity_score, 3),
                    'is_toxic': is_toxic,
                    'confidence': round(confidence, 3),
                    'label': self.id2label[predicted_class],
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

toxicity_model: Optional[BERTToxicityModel] = None

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Inizializza il modello all'avvio del servizio"""
    global toxicity_model
    
    logger.info("🚀 Starting BERT Toxicity Microservice...")
    
    try:
        toxicity_model = BERTToxicityModel()
        logger.info("✅ Service ready and listening on port 5003")
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup al shutdown"""
    logger.info("🛑 Shutting down BERT Toxicity Microservice...")
    
    # Cleanup GPU memory se necessario
    if toxicity_model and toxicity_model.device == "cuda":
        torch.cuda.empty_cache()
    
    logger.info("✅ Shutdown complete")

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", tags=["Health"])
def root():
    """Root endpoint - informazioni base sul servizio"""
    return {
        "service": "BERT Toxicity Analysis Microservice",
        "version": "1.0.0",
        "model": toxicity_model.model_name if toxicity_model else "Not loaded",
        "device": toxicity_model.device if toxicity_model else "Unknown",
        "status": "running" if toxicity_model and toxicity_model._initialized else "initializing",
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
    if not toxicity_model or not toxicity_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=toxicity_model.device,
        model_name=toxicity_model.model_name
    )


@app.post("/analyze", response_model=ToxicityResponse, tags=["Toxicity Analysis"])
def analyze_toxicity(request: ToxicityRequest):
    """
    Analizza tossicità di un singolo testo.
    
    **Input:**
    - text: Testo da analizzare (1-5000 caratteri)
    
    **Output:**
    - toxicity_score: Score 0.0-1.0 (maggiore = più tossico)
    - is_toxic: True se score > 0.5
    - confidence: Confidenza predizione 0.0-1.0
    - label: "toxic" o "non-toxic"
    - processing_time_ms: Tempo elaborazione
    
    **Example:**
    ```json
    {
      "text": "You are stupid and useless!"
    }
    ```
    
    **Response:**
    ```json
    {
      "toxicity_score": 0.89,
      "is_toxic": true,
      "confidence": 0.89,
      "label": "toxic",
      "processing_time_ms": 42.15
    }
    ```
    """
    if not toxicity_model or not toxicity_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    try:
        result = toxicity_model.analyze(request.text)
        return ToxicityResponse(**result)
        
    except Exception as e:
        logger.error(f"Error analyzing toxicity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during toxicity analysis: {str(e)}"
        )


@app.post("/batch", response_model=BatchToxicityResponse, tags=["Toxicity Analysis"])
def batch_analyze_toxicity(request: BatchToxicityRequest):
    """
    Analizza tossicità per batch di testi (più efficiente).
    
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
        "This is terrible and you are awful",
        "Thank you for your help"
      ]
    }
    ```
    
    **Response:**
    ```json
    {
      "results": [
        {"toxicity_score": 0.05, "is_toxic": false, "label": "non-toxic"},
        {"toxicity_score": 0.92, "is_toxic": true, "label": "toxic"},
        {"toxicity_score": 0.03, "is_toxic": false, "label": "non-toxic"}
      ],
      "total_processed": 3,
      "total_time_ms": 115.8
    }
    ```
    """
    if not toxicity_model or not toxicity_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    try:
        start_time = time.time()
        results = toxicity_model.batch_analyze(request.texts)
        total_time = (time.time() - start_time) * 1000
        
        return BatchToxicityResponse(
            results=[ToxicityResponse(**r) for r in results],
            total_processed=len(results),
            total_time_ms=round(total_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during batch toxicity analysis: {str(e)}"
        )


@app.get("/info", response_model=ModelInfoResponse, tags=["Info"])
def model_info():
    """
    Informazioni dettagliate sul modello.
    
    Returns:
        Dettagli su architettura, capacità, limiti
    """
    if not toxicity_model or not toxicity_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    return ModelInfoResponse(
        model_name=toxicity_model.model_name,
        architecture="BERT-small (6 layers, 512 hidden)",
        task="Binary Toxicity Classification",
        output_range="0.0 (non-toxic) - 1.0 (toxic)",
        device=toxicity_model.device,
        max_input_length=512,
        threshold=toxicity_model.threshold
    )

# ============================================
# MAIN (per test locali)
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 5003))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )