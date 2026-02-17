"""
Abstract Model Predictor + Standalone Toxicity Detector

SENTIMENT: Abstract class per normalizzare output in positive/neutral/negative
TOXICITY: Standalone detector con output dedicato (is_toxic, severity)

Design rationale:
- Sentiment è multiclass → abstract pattern ha senso
- Toxicity è binary → standalone più chiaro e semantico
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


# ============================================
# ENUMS
# ============================================

class SentimentLabel(str, Enum):
    """Etichette sentiment normalizzate"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ToxicitySeverity(str, Enum):
    """Livelli di gravità tossicità"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ============================================
# SENTIMENT RESPONSE MODELS
# ============================================

class NormalizedPrediction(BaseModel):
    """
    Output normalizzato per SENTIMENT analysis.
    
    Converte stelle 1-5 in categorie positive/neutral/negative.
    """
    label: SentimentLabel = Field(..., description="Categoria: positive, neutral, negative")
    score: float = Field(..., ge=0.0, le=1.0, description="Score normalizzato 0-1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidenza predizione")
    raw_output: Dict[str, Any] = Field(..., description="Output originale del modello")
    model_type: str = Field(..., description="Tipo di modello")


class BatchNormalizedPrediction(BaseModel):
    """Batch di predizioni sentiment normalizzate"""
    predictions: List[NormalizedPrediction]
    total_processed: int
    avg_score: float = Field(..., ge=0.0, le=1.0)
    label_distribution: Dict[str, int] = Field(..., description="Conteggio per label")


# ============================================
# TOXICITY RESPONSE MODELS (Dedicated)
# ============================================

class ToxicityResult(BaseModel):
    """
    Output dedicato per TOXICITY detection.
    
    Non usa positive/neutral/negative perché semanticamente non ha senso.
    Usa invece is_toxic boolean + severity levels.
    """
    is_toxic: bool = Field(..., description="True se tossico (score > 0.5)")
    toxicity_score: float = Field(..., ge=0.0, le=1.0, description="Score tossicità 0-1")
    severity: ToxicitySeverity = Field(..., description="Livello gravità: low/medium/high")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidenza predizione")
    raw_output: Dict[str, Any] = Field(..., description="Output originale del modello")


class BatchToxicityResult(BaseModel):
    """Batch di predizioni toxicity"""
    results: List[ToxicityResult]
    total_processed: int
    toxic_count: int = Field(..., description="Numero di messaggi tossici")
    toxic_ratio: float = Field(..., ge=0.0, le=1.0, description="Percentuale tossici")
    avg_toxicity_score: float = Field(..., ge=0.0, le=1.0)


# ============================================
# ABSTRACT BASE CLASS (Solo per Sentiment)
# ============================================

class ModelPredictor(ABC):
    """
    Abstract base class per SENTIMENT predictors.
    
    Normalizza output di modelli sentiment (stelle, scores, ecc.)
    in formato uniforme: positive/neutral/negative.
    
    NOTA: NON usata per toxicity - quella è standalone.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_type = self._get_model_type()
    
    @abstractmethod
    def _get_model_type(self) -> str:
        """Ritorna il tipo di modello (es: 'sentiment')"""
        pass
    
    @abstractmethod
    async def _raw_predict(self, text: str) -> Dict[str, Any]:
        """Chiamata raw al modello ML"""
        pass
    
    @abstractmethod
    async def _raw_batch_predict(self, texts: List[str]) -> Dict[str, Any]:
        """Batch prediction raw"""
        pass
    
    @abstractmethod
    def _normalize_output(self, raw_output: Dict[str, Any]) -> NormalizedPrediction:
        """Normalizza l'output del modello in formato standard"""
        pass
    
    async def predict(self, text: str) -> NormalizedPrediction:
        """API uniforme per predizione singola"""
        raw_output = await self._raw_predict(text)
        return self._normalize_output(raw_output)
    
    async def predict_batch(self, texts: List[str]) -> BatchNormalizedPrediction:
        """API uniforme per batch prediction"""
        raw_batch = await self._raw_batch_predict(texts)
        
        predictions = [
            self._normalize_output(result)
            for result in raw_batch.get('results', [])
        ]
        
        total = len(predictions)
        avg_score = sum(p.score for p in predictions) / total if total > 0 else 0.0
        
        label_distribution = {
            "positive": sum(1 for p in predictions if p.label == SentimentLabel.POSITIVE),
            "neutral": sum(1 for p in predictions if p.label == SentimentLabel.NEUTRAL),
            "negative": sum(1 for p in predictions if p.label == SentimentLabel.NEGATIVE)
        }
        
        return BatchNormalizedPrediction(
            predictions=predictions,
            total_processed=total,
            avg_score=round(avg_score, 3),
            label_distribution=label_distribution
        )


# ============================================
# SENTIMENT ADAPTER
# ============================================

class SentimentPredictor(ModelPredictor):
    """
    Adapter per BERT Sentiment (1-5 stelle).
    
    Normalizza:
    - 1.0-2.5 stelle → NEGATIVE
    - 2.5-3.5 stelle → NEUTRAL
    - 3.5-5.0 stelle → POSITIVE
    """
    
    def __init__(self, service_url: str, http_client):
        super().__init__(model_name="bert-sentiment")
        self.service_url = service_url
        self.http_client = http_client
    
    def _get_model_type(self) -> str:
        return "sentiment"
    
    async def _raw_predict(self, text: str) -> Dict[str, Any]:
        """Chiama il microservizio BERT Sentiment"""
        response = await self.http_client.post(
            f"{self.service_url}/analyze",
            json={"text": text},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
    
    async def _raw_batch_predict(self, texts: List[str]) -> Dict[str, Any]:
        """Batch prediction per sentiment"""
        response = await self.http_client.post(
            f"{self.service_url}/batch",
            json={"texts": texts},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
    
    def _normalize_output(self, raw_output: Dict[str, Any]) -> NormalizedPrediction:
        """
        Normalizza stelle (1-5) in categorie (positive, neutral, negative).
        """
        stars = raw_output.get('stars', 3.0)
        confidence = raw_output.get('confidence', 0.0)
        
        # Normalizza stelle in score 0-1
        normalized_score = (stars - 1) / 4  # 1-5 → 0-1
        
        # Determina label
        if stars < 2.5:
            label = SentimentLabel.NEGATIVE
        elif stars <= 3.5:
            label = SentimentLabel.NEUTRAL
        else:
            label = SentimentLabel.POSITIVE
        
        return NormalizedPrediction(
            label=label,
            score=round(normalized_score, 3),
            confidence=round(confidence, 3),
            raw_output=raw_output,
            model_type=self.model_type
        )


# ============================================
# TOXICITY DETECTOR (Standalone - NO Abstract)
# ============================================

class ToxicityDetector:
    """
    Standalone Toxicity Detector.
    
    NON eredita da ModelPredictor perché:
    1. Toxicity è binary (toxic/non-toxic), non multiclass
    2. Output dedicato più semantico di positive/neutral/negative
    3. Usa metodo 'detect()' invece di 'predict()' per chiarezza
    
    Severity levels:
    - LOW: score < 0.3 (safe)
    - MEDIUM: score 0.3-0.6 (borderline)
    - HIGH: score > 0.6 (toxic)
    """
    
    def __init__(self, service_url: str, http_client):
        self.service_url = service_url
        self.http_client = http_client
        self.threshold = 0.5  # Soglia per is_toxic
    
    def _get_severity(self, score: float) -> ToxicitySeverity:
        """Determina severity level da score"""
        if score < 0.3:
            return ToxicitySeverity.LOW
        elif score < 0.6:
            return ToxicitySeverity.MEDIUM
        else:
            return ToxicitySeverity.HIGH
    
    async def detect(self, text: str) -> ToxicityResult:
        """
        Rileva tossicità di un singolo testo.
        
        Returns:
            ToxicityResult con is_toxic, severity, score
        """
        response = await self.http_client.post(
            f"{self.service_url}/analyze",
            json={"text": text},
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        
        toxicity_score = data['toxicity_score']
        
        return ToxicityResult(
            is_toxic=data['is_toxic'],
            toxicity_score=round(toxicity_score, 3),
            severity=self._get_severity(toxicity_score),
            confidence=round(data['confidence'], 3),
            raw_output=data
        )
    
    async def detect_batch(self, texts: List[str]) -> BatchToxicityResult:
        """
        Rileva tossicità per batch di testi.
        
        Returns:
            BatchToxicityResult con statistiche aggregate
        """
        response = await self.http_client.post(
            f"{self.service_url}/batch",
            json={"texts": texts},
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        toxic_count = 0
        total_toxicity = 0.0
        
        for item in data['results']:
            toxicity_score = item['toxicity_score']
            is_toxic = item['is_toxic']
            
            result = ToxicityResult(
                is_toxic=is_toxic,
                toxicity_score=round(toxicity_score, 3),
                severity=self._get_severity(toxicity_score),
                confidence=round(item['confidence'], 3),
                raw_output=item
            )
            
            results.append(result)
            if is_toxic:
                toxic_count += 1
            total_toxicity += toxicity_score
        
        total = len(results)
        
        return BatchToxicityResult(
            results=results,
            total_processed=total,
            toxic_count=toxic_count,
            toxic_ratio=round(toxic_count / total, 3) if total > 0 else 0.0,
            avg_toxicity_score=round(total_toxicity / total, 3) if total > 0 else 0.0
        )


# ============================================
# FACTORY
# ============================================

class PredictorFactory:
    """
    Factory per creare predictors e detectors.
    """
    
    def __init__(self, http_client):
        self.http_client = http_client
    
    def create_sentiment_predictor(self, service_url: str) -> SentimentPredictor:
        """Crea sentiment predictor (abstract pattern)"""
        return SentimentPredictor(service_url, self.http_client)
    
    def create_toxicity_detector(self, service_url: str) -> ToxicityDetector:
        """Crea toxicity detector (standalone)"""
        return ToxicityDetector(service_url, self.http_client)