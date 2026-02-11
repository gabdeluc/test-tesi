"""
Toxicity Detection Microservice

Servizio indipendente per rilevare toxicity nei testi.
Usa il modello gravitee-io/bert-small-toxicity da HuggingFace.

Port: 5002
Model: gravitee-io/bert-small-toxicity
Output: toxic/non_toxic (normalizzato tramite BaseMLModel)
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
    title="Toxicity Detection Microservice",
    description="Servizio di rilevamento toxicity con classificazione toxic/non_toxic",
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
    """Output normalizzato secondo BaseMLModel"""
    text: str = Field(..., description="Testo analizzato")
    prediction: str = Field(..., description="Classe predetta: toxic/non_toxic")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidenza predizione")
    raw_scores: Dict[str, Any] = Field(..., description="Score originali del modello")
    metadata: Dict[str, Any] = Field(..., description="Informazioni sul modello")
    processing_time_ms: Optional[float] = Field(None, description="Tempo elaborazione")


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
    output_classes: List[str]
    threshold: float
    device: str
    max_input_length: int

# ============================================
# TOXICITY MODEL MANAGER (Singleton)
# ============================================

class ToxicityModel:
    """
    Singleton per gestire il modello Toxicity.
    
    Features:
    - Lazy loading
    - Thread-safe
    - Batch processing ottimizzato
    - Normalizzazione output secondo BaseMLModel
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
        
        # Threshold per classificazione toxic/non_toxic
        self.threshold = 0.5
        
        logger.info("="*60)
        logger.info("🚀 Inizializzazione Toxicity Detection Service")
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
            
            # Labels del modello
            # gravitee-io/bert-small-toxicity output: [non-toxic, toxic]
            self.id2label = self.model.config.id2label
            
            self._initialized = True
            
            logger.info("✅ Modello caricato con successo!")
            logger.info(f"📊 Labels: {self.id2label}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento modello: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Analizza toxicity di un singolo testo.
        
        Output normalizzato secondo BaseMLModel:
        - prediction: "toxic" o "non_toxic"
        - confidence: probabilità della classe predetta
        - raw_scores: score originali del modello
        
        Args:
            text: Testo da analizzare
            
        Returns:
            Dict con formato normalizzato
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
            
            # Estrai probabilità
            # Il modello ha 2 classi: [non-toxic, toxic]
            # Quindi probs[0] = P(non-toxic), probs[1] = P(toxic)
            prob_non_toxic = probs[0].item()
            prob_toxic = probs[1].item()
            
            # NORMALIZZAZIONE secondo BaseMLModel
            # Se P(toxic) >= threshold → toxic, altrimenti non_toxic
            if prob_toxic >= self.threshold:
                prediction = "toxic"
                confidence = prob_toxic
            else:
                prediction = "non_toxic"
                confidence = prob_non_toxic
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Output normalizzato
            result = {
                'text': text,
                'prediction': prediction,
                'confidence': round(confidence, 3),
                'raw_scores': {
                    'toxicity_score': round(prob_toxic, 3),
                    'non_toxicity_score': round(prob_non_toxic, 3),
                    'logits': {
                        'non_toxic': round(logits[0][0].item(), 3),
                        'toxic': round(logits[0][1].item(), 3)
                    }
                },
                'metadata': {
                    'model': self.model_name,
                    'threshold': self.threshold,
                    'normalization': '2-class',
                    'device': self.device
                },
                'processing_time_ms': round(processing_time, 2)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Errore durante analisi: {e}")
            raise
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analizza batch di testi (più efficiente).
        
        Args:
            texts: Lista di testi da analizzare
            
        Returns:
            Lista di risultati normalizzati
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
            
            for i, (text, prob) in enumerate(zip(texts, probs)):
                prob_non_toxic = prob[0].item()
                prob_toxic = prob[1].item()
                
                # Normalizzazione
                if prob_toxic >= self.threshold:
                    prediction = "toxic"
                    confidence = prob_toxic
                else:
                    prediction = "non_toxic"
                    confidence = prob_non_toxic
                
                results.append({
                    'text': text,
                    'prediction': prediction,
                    'confidence': round(confidence, 3),
                    'raw_scores': {
                        'toxicity_score': round(prob_toxic, 3),
                        'non_toxicity_score': round(prob_non_toxic, 3),
                        'logits': {
                            'non_toxic': round(logits[i][0].item(), 3),
                            'toxic': round(logits[i][1].item(), 3)
                        }
                    },
                    'metadata': {
                        'model': self.model_name,
                        'threshold': self.threshold,
                        'normalization': '2-class',
                        'device': self.device
                    },
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

toxicity_model: Optional[ToxicityModel] = None

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Inizializza il modello all'avvio del servizio"""
    global toxicity_model
    
    logger.info("🚀 Starting Toxicity Detection Microservice...")
    
    try:
        toxicity_model = ToxicityModel()
        logger.info("✅ Service ready and listening on port 5002")
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup al shutdown"""
    logger.info("🛑 Shutting down Toxicity Detection Microservice...")
    
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
        "service": "Toxicity Detection Microservice",
        "version": "1.0.0",
        "model": toxicity_model.model_name if toxicity_model else "Not loaded",
        "device": toxicity_model.device if toxicity_model else "Unknown",
        "threshold": toxicity_model.threshold if toxicity_model else 0.5,
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


@app.post("/analyze", response_model=ToxicityResponse, tags=["Toxicity Detection"])
def analyze_toxicity(request: ToxicityRequest):
    """
    Analizza toxicity di un singolo testo.
    
    **Input:**
    - text: Testo da analizzare (1-5000 caratteri)
    
    **Output normalizzato (BaseMLModel):**
    - prediction: "toxic" o "non_toxic"
    - confidence: Probabilità 0.0-1.0
    - raw_scores: Score originali del modello
    - metadata: Info sul modello
    
    **Example Request:**
    ```json
    {
      "text": "You are stupid and I hate you!"
    }
    ```
    
    **Example Response:**
    ```json
    {
      "text": "You are stupid and I hate you!",
      "prediction": "toxic",
      "confidence": 0.892,
      "raw_scores": {
        "toxicity_score": 0.892,
        "non_toxicity_score": 0.108
      },
      "metadata": {
        "model": "gravitee-io/bert-small-toxicity",
        "threshold": 0.5,
        "normalization": "2-class"
      },
      "processing_time_ms": 35.2
    }
    ```
    """
    if not toxicity_model or not toxicity_model._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    try:
        result = toxicity_model.predict(request.text)
        return ToxicityResponse(**result)
        
    except Exception as e:
        logger.error(f"Error analyzing toxicity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during toxicity analysis: {str(e)}"
        )


@app.post("/batch", response_model=BatchToxicityResponse, tags=["Toxicity Detection"])
def batch_analyze_toxicity(request: BatchToxicityRequest):
    """
    Analizza toxicity per batch di testi (più efficiente).
    
    **Limiti:**
    - Minimo: 1 testo
    - Massimo: 100 testi per richiesta
    
    **Performance:**
    - Batch processing è ~8x più veloce di chiamate singole
    
    **Example Request:**
    ```json
    {
      "texts": [
        "Have a nice day!",
        "You are an idiot!",
        "I disagree with your point"
      ]
    }
    ```
    
    **Example Response:**
    ```json
    {
      "results": [
        {
          "text": "Have a nice day!",
          "prediction": "non_toxic",
          "confidence": 0.98
        },
        {
          "text": "You are an idiot!",
          "prediction": "toxic",
          "confidence": 0.87
        },
        {
          "text": "I disagree with your point",
          "prediction": "non_toxic",
          "confidence": 0.92
        }
      ],
      "total_processed": 3,
      "total_time_ms": 95.3
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
        results = toxicity_model.batch_predict(request.texts)
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
        task="Binary Text Classification (Toxicity Detection)",
        output_classes=["non_toxic", "toxic"],
        threshold=toxicity_model.threshold,
        device=toxicity_model.device,
        max_input_length=512
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