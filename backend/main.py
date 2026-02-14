"""
Backend Gateway - API Orchestrator

Questo backend orchestra i microservizi ML senza caricare modelli.
Mantiene la logica business e i dati, delega ML a servizi specializzati.

Architecture:
- BERT Sentiment Service: http://bert-sentiment:5001
- BERT Toxicity Service: http://bert-toxicity:5003
"""

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import httpx
import random
import os
from config.config_loader import config_loader

# ============================================
# FASTAPI APP SETUP
# ============================================

app = FastAPI(
    title="Meeting Transcript API Gateway",
    description="Backend orchestrator per microservizi di sentiment e toxicity analysis",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MICROSERVICES CONFIGURATION
# ============================================

BERT_SERVICE_URL = os.getenv("BERT_SERVICE_URL", "http://bert-sentiment:5001")
TOXICITY_SERVICE_URL = os.getenv("TOXICITY_SERVICE_URL", "http://bert-toxicity:5003")

# HTTP client with timeout for service calls
http_client = httpx.AsyncClient(timeout=30.0)

# ============================================
# PYDANTIC MODELS
# ============================================

class TranscriptEntry(BaseModel):
    uid: str
    nickname: str
    text: str
    from_field: str = Field(..., alias="from", description="Timestamp HH:MM:SS.mmm")
    to: str = Field(..., description="Timestamp HH:MM:SS.mmm")

class Participant(BaseModel):
    id: str
    name: str

class MeetingMetadata(BaseModel):
    participants: List[Participant]
    date: str

class TranscriptMetadata(BaseModel):
    language: str

class TranscriptResponse(BaseModel):
    transcript: List[TranscriptEntry]
    metadata: TranscriptMetadata

class MeetingResponse(BaseModel):
    metadata: MeetingMetadata

# ============================================
# SENTIMENT & TOXICITY MODELS
# ============================================

class SentimentResult(BaseModel):
    stars: float
    sentiment: str
    confidence: float

class ToxicityResult(BaseModel):
    toxicity_score: float
    is_toxic: bool
    confidence: float
    label: str

class SentimentAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class ToxicityAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100)

class BatchToxicityRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100)

# ============================================
# LOAD CONFIGURATION
# ============================================

SAMPLE_PHRASES = config_loader.get_sample_phrases()
PARTICIPANTS_CONFIG = config_loader.get_participants()
MEETINGS_CONFIG = config_loader.get_meetings()
GENERATION_CONFIG = config_loader.get_generation_config()

PARTICIPANTS = [Participant(**p) for p in PARTICIPANTS_CONFIG]

# ============================================
# MOCK DATA GENERATION
# ============================================

def generate_mock_transcript(num_entries: int = 20) -> List[TranscriptEntry]:
    """Genera transcript mock usando configurazione"""
    transcript = []
    current_time = 0
    
    min_duration = GENERATION_CONFIG['min_duration_seconds']
    max_pause = GENERATION_CONFIG['max_pause_seconds']
    chars_per_sec = GENERATION_CONFIG['chars_per_second']
    
    for i in range(num_entries):
        participant = random.choice(PARTICIPANTS)
        text = random.choice(SAMPLE_PHRASES)
        duration = max(min_duration, len(text) // chars_per_sec)
        
        from_time = f"{current_time // 3600:02d}:{(current_time % 3600) // 60:02d}:{current_time % 60:02d}.000"
        current_time += duration
        to_time = f"{current_time // 3600:02d}:{(current_time % 3600) // 60:02d}:{current_time % 60:02d}.000"
        
        transcript.append(TranscriptEntry(
            uid=str(12345 + i),
            nickname=participant.name,
            text=text,
            **{"from": from_time},
            to=to_time
        ))
        
        current_time += random.randint(1, max_pause)
    
    return transcript

# Initialize mock database
MOCK_MEETINGS = {}
for meeting_config in MEETINGS_CONFIG:
    meeting_id = meeting_config['id']
    MOCK_MEETINGS[meeting_id] = {
        "metadata": MeetingMetadata(
            participants=PARTICIPANTS,
            date=meeting_config['date']
        ),
        "transcript": generate_mock_transcript(meeting_config['num_entries'])
    }

# ============================================
# MICROSERVICES COMMUNICATION HELPERS
# ============================================

async def call_bert_service(endpoint: str, payload: dict) -> dict:
    """Chiama il servizio BERT per sentiment analysis"""
    try:
        response = await http_client.post(
            f"{BERT_SERVICE_URL}/{endpoint}",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="BERT service timeout"
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BERT service error: {str(e)}"
        )

async def call_toxicity_service(endpoint: str, payload: dict) -> dict:
    """Chiama il servizio Toxicity per toxicity analysis"""
    try:
        response = await http_client.post(
            f"{TOXICITY_SERVICE_URL}/{endpoint}",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Toxicity service timeout"
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Toxicity service error: {str(e)}"
        )

async def check_service_health(service_url: str) -> bool:
    """Verifica se un servizio è online"""
    try:
        response = await http_client.get(f"{service_url}/health", timeout=5.0)
        return response.status_code == 200
    except:
        return False

# ============================================
# MAIN ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint con informazioni sul gateway e status dei servizi"""
    bert_healthy = await check_service_health(BERT_SERVICE_URL)
    toxicity_healthy = await check_service_health(TOXICITY_SERVICE_URL)
    
    return {
        "status": "ok",
        "version": "3.0.0",
        "architecture": "microservices",
        "gateway": "FastAPI Backend",
        "services": {
            "bert_sentiment": {
                "url": BERT_SERVICE_URL,
                "healthy": bert_healthy,
                "port": 5001
            },
            "bert_toxicity": {
                "url": TOXICITY_SERVICE_URL,
                "healthy": toxicity_healthy,
                "port": 5003
            }
        },
        "endpoints": {
            "meeting": [
                "GET /meeting/{meetingId}",
                "GET /meeting/{meetingId}/transcript/",
                "GET /meeting/{meetingId}/analysis"
            ],
            "sentiment": [
                "POST /sentiment/analyze",
                "POST /sentiment/batch"
            ],
            "toxicity": [
                "POST /toxicity/analyze",
                "POST /toxicity/batch"
            ],
            "utility": [
                "GET /participants",
                "GET /meetings",
                "GET /services/status"
            ]
        }
    }

@app.get("/health")
def health_check():
    """Health check per il gateway"""
    return {"status": "healthy", "service": "backend-gateway"}

# ============================================
# MEETING ENDPOINTS
# ============================================

@app.get("/meeting/{meetingId}", response_model=MeetingResponse)
def get_meeting(meetingId: str):
    """Ottieni metadata del meeting"""
    meeting = MOCK_MEETINGS.get(meetingId)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meetingId} not found"
        )
    
    return MeetingResponse(metadata=meeting["metadata"])

@app.get("/meeting/{meetingId}/transcript/", response_model=TranscriptResponse)
def get_transcript_full(meetingId: str):
    """Ottieni transcript completo"""
    meeting = MOCK_MEETINGS.get(meetingId)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meetingId} not found"
        )
    
    return TranscriptResponse(
        transcript=meeting["transcript"],
        metadata=TranscriptMetadata(language="en")
    )

@app.get("/meeting/{meetingId}/transcript")
def get_transcript_filtered(
    meetingId: str,
    participant_id: Optional[str] = None
):
    """Ottieni transcript con filtro opzionale per partecipante"""
    meeting = MOCK_MEETINGS.get(meetingId)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meetingId} not found"
        )
    
    transcript = meeting["transcript"]
    
    if participant_id:
        participant_name = next(
            (p.name for p in PARTICIPANTS if p.id == participant_id),
            None
        )
        if participant_name:
            transcript = [e for e in transcript if e.nickname == participant_name]
    
    return {
        "transcript": transcript,
        "metadata": {"language": "en"}
    }

# ============================================
# ANALYSIS ENDPOINT (SENTIMENT + TOXICITY)
# ============================================

@app.get("/meeting/{meetingId}/analysis")
async def get_transcript_with_analysis(
    meetingId: str,
    participant_id: Optional[str] = None
):
    """
    Ottieni transcript arricchito con SENTIMENT e TOXICITY analysis.
    Questo è l'endpoint principale per il widget board.
    """
    # 1. Ottieni meeting
    meeting = MOCK_MEETINGS.get(meetingId)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meetingId} not found"
        )
    
    transcript = meeting["transcript"]
    
    # 2. Filtra per partecipante se richiesto
    if participant_id:
        participant_name = next(
            (p.name for p in PARTICIPANTS if p.id == participant_id),
            None
        )
        if participant_name:
            transcript = [e for e in transcript if e.nickname == participant_name]
    
    # 3. Estrai testi
    texts = [entry.text for entry in transcript]
    
    if not texts:
        return {
            "transcript": [],
            "metadata": {
                "language": "en",
                "stats": {
                    "total_messages": 0,
                    "sentiment": {},
                    "toxicity": {}
                }
            }
        }
    
    # 4. Chiama BERT Sentiment (batch)
    sentiment_response = await call_bert_service("batch", {"texts": texts})
    sentiments = sentiment_response["results"]
    
    # 5. Chiama BERT Toxicity (batch)
    toxicity_response = await call_toxicity_service("batch", {"texts": texts})
    toxicities = toxicity_response["results"]
    
    # 6. Combina transcript con analisi
    enriched_transcript = []
    
    # Stats accumulatori
    total_stars = 0
    positive_count = 0
    toxic_count = 0
    total_toxicity = 0
    
    for i, entry in enumerate(transcript):
        entry_dict = entry.dict(by_alias=True)
        entry_dict['sentiment'] = sentiments[i]
        entry_dict['toxicity'] = toxicities[i]
        
        enriched_transcript.append(entry_dict)
        
        # Accumula stats
        total_stars += sentiments[i]['stars']
        if sentiments[i]['stars'] >= 3.5:
            positive_count += 1
        if toxicities[i]['is_toxic']:
            toxic_count += 1
        total_toxicity += toxicities[i]['toxicity_score']
    
    # 7. Calcola statistiche aggregate
    num_messages = len(sentiments)
    avg_stars = total_stars / num_messages
    positive_ratio = positive_count / num_messages
    toxic_ratio = toxic_count / num_messages
    avg_toxicity = total_toxicity / num_messages
    
    return {
        "transcript": enriched_transcript,
        "metadata": {
            "language": "en",
            "stats": {
                "total_messages": num_messages,
                "sentiment": {
                    "average_stars": round(avg_stars, 2),
                    "positive_ratio": round(positive_ratio, 2),
                    "positive_count": positive_count
                },
                "toxicity": {
                    "toxic_ratio": round(toxic_ratio, 2),
                    "toxic_count": toxic_count,
                    "average_toxicity": round(avg_toxicity, 3)
                }
            }
        }
    }

# ============================================
# SENTIMENT ENDPOINTS
# ============================================

@app.post("/sentiment/analyze")
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analizza sentiment di un singolo testo"""
    result = await call_bert_service("analyze", {"text": request.text})
    return result

@app.post("/sentiment/batch")
async def analyze_sentiment_batch(request: BatchSentimentRequest):
    """Analizza sentiment per batch di testi"""
    result = await call_bert_service("batch", {"texts": request.texts})
    return result

# ============================================
# TOXICITY ENDPOINTS
# ============================================

@app.post("/toxicity/analyze")
async def analyze_toxicity(request: ToxicityAnalysisRequest):
    """Analizza toxicity di un singolo testo"""
    result = await call_toxicity_service("analyze", {"text": request.text})
    return result

@app.post("/toxicity/batch")
async def analyze_toxicity_batch(request: BatchToxicityRequest):
    """Analizza toxicity per batch di testi"""
    result = await call_toxicity_service("batch", {"texts": request.texts})
    return result

# ============================================
# UTILITY ENDPOINTS
# ============================================

@app.get("/participants")
def get_participants():
    """Lista tutti i partecipanti disponibili"""
    return {"participants": PARTICIPANTS}

@app.get("/meetings")
def get_all_meetings():
    """Lista tutti i meeting disponibili con metadata"""
    return {
        "meetings": [
            {
                "id": meeting_id,
                "date": meeting["metadata"].date,
                "participants_count": len(meeting["metadata"].participants),
                "messages_count": len(meeting["transcript"])
            }
            for meeting_id, meeting in MOCK_MEETINGS.items()
        ]
    }

@app.get("/services/status")
async def get_services_status():
    """Status dettagliato di tutti i microservizi"""
    bert_healthy = await check_service_health(BERT_SERVICE_URL)
    toxicity_healthy = await check_service_health(TOXICITY_SERVICE_URL)
    
    bert_info = None
    toxicity_info = None
    
    if bert_healthy:
        try:
            response = await http_client.get(f"{BERT_SERVICE_URL}/info", timeout=5.0)
            bert_info = response.json()
        except:
            pass
    
    if toxicity_healthy:
        try:
            response = await http_client.get(f"{TOXICITY_SERVICE_URL}/info", timeout=5.0)
            toxicity_info = response.json()
        except:
            pass
    
    return {
        "bert_sentiment": {
            "healthy": bert_healthy,
            "url": BERT_SERVICE_URL,
            "port": 5001,
            "info": bert_info
        },
        "bert_toxicity": {
            "healthy": toxicity_healthy,
            "url": TOXICITY_SERVICE_URL,
            "port": 5003,
            "info": toxicity_info
        }
    }

@app.get("/config")
def get_config():
    """Visualizza configurazione corrente (debug)"""
    return {
        "sample_phrases": SAMPLE_PHRASES,
        "participants": [p.dict() for p in PARTICIPANTS],
        "meetings": MEETINGS_CONFIG,
        "generation": GENERATION_CONFIG
    }

# ============================================
# SHUTDOWN HANDLER
# ============================================

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup al shutdown"""
    await http_client.aclose()