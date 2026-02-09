"""
Backend Gateway - API Orchestrator

Questo backend orchestra i microservizi ML senza caricare modelli.
Mantiene la logica business e i dati, delega ML a servizi specializzati.

Architecture:
- BERT Sentiment Service: http://bert-sentiment:5001
- E5 Embeddings Service: http://e5-embeddings:5002
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
    description="Backend orchestrator per microservizi di sentiment analysis",
    version="2.0.0"
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
E5_SERVICE_URL = os.getenv("E5_SERVICE_URL", "http://e5-embeddings:5002")

# HTTP client with timeout for service calls
http_client = httpx.AsyncClient(timeout=30.0)

# ============================================
# PYDANTIC MODELS (ORIGINALI)
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
# SENTIMENT MODELS
# ============================================

class SentimentResult(BaseModel):
    stars: float
    sentiment: str
    confidence: float

class SentimentAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100)

class SimilaritySearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

# ============================================
# LOAD CONFIGURATION (ORIGINALE)
# ============================================

SAMPLE_PHRASES = config_loader.get_sample_phrases()
PARTICIPANTS_CONFIG = config_loader.get_participants()
MEETINGS_CONFIG = config_loader.get_meetings()
GENERATION_CONFIG = config_loader.get_generation_config()

PARTICIPANTS = [Participant(**p) for p in PARTICIPANTS_CONFIG]

# ============================================
# MOCK DATA GENERATION (ORIGINALE)
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
    """
    Chiama il servizio BERT per sentiment analysis
    
    Args:
        endpoint: L'endpoint da chiamare (es: "analyze", "batch")
        payload: Dati da inviare
        
    Returns:
        Risposta JSON dal servizio BERT
        
    Raises:
        HTTPException: Se il servizio non è disponibile
    """
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

async def call_e5_service(endpoint: str, payload: dict) -> dict:
    """
    Chiama il servizio E5 per embeddings/similarity
    
    Args:
        endpoint: L'endpoint da chiamare (es: "embed", "similarity")
        payload: Dati da inviare
        
    Returns:
        Risposta JSON dal servizio E5
        
    Raises:
        HTTPException: Se il servizio non è disponibile
    """
    try:
        response = await http_client.post(
            f"{E5_SERVICE_URL}/{endpoint}",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="E5 service timeout"
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"E5 service error: {str(e)}"
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
    """
    Root endpoint con informazioni sul gateway e status dei servizi
    """
    # Check services health
    bert_healthy = await check_service_health(BERT_SERVICE_URL)
    e5_healthy = await check_service_health(E5_SERVICE_URL)
    
    return {
        "status": "ok",
        "version": "2.0.0",
        "architecture": "microservices",
        "gateway": "FastAPI Backend",
        "services": {
            "bert_sentiment": {
                "url": BERT_SERVICE_URL,
                "healthy": bert_healthy,
                "port": 5001
            },
            "e5_embeddings": {
                "url": E5_SERVICE_URL,
                "healthy": e5_healthy,
                "port": 5002
            }
        },
        "endpoints": {
            "original": [
                "GET /meeting/{meetingId}",
                "GET /meeting/{meetingId}/transcript/",
                "GET /meeting/{meetingId}/character-count"
            ],
            "sentiment": [
                "POST /sentiment/analyze",
                "POST /sentiment/batch",
                "GET /meeting/{meetingId}/sentiment"
            ],
            "similarity": [
                "POST /meeting/{meetingId}/similarity"
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
# ORIGINAL ENDPOINTS (INVARIATI)
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
    
    # Filtra per partecipante se richiesto
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

@app.get("/meeting/{meetingId}/character-count")
def get_character_count(
    meetingId: str,
    participant_id: Optional[str] = None
):
    """
    Conta caratteri, parole e messaggi nel transcript
    
    Query params:
    - participant_id: filtra per partecipante (opzionale)
    """
    response = get_transcript_filtered(meetingId, participant_id)
    transcript = response["transcript"]
    
    total_chars = sum(len(entry.text) for entry in transcript)
    total_words = sum(len(entry.text.split()) for entry in transcript)
    
    result = {
        "meeting_id": meetingId,
        "total_characters": total_chars,
        "total_words": total_words,
        "total_messages": len(transcript)
    }
    
    # Aggiungi info partecipante se filtrato
    if participant_id:
        participant = next((p for p in PARTICIPANTS if p.id == participant_id), None)
        if participant:
            result["participant"] = {
                "id": participant_id,
                "name": participant.name
            }
    
    return result

# ============================================
# SENTIMENT ANALYSIS ENDPOINTS (NUOVI)
# Orchestrano chiamate a BERT microservice
# ============================================

@app.post("/sentiment/analyze")
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """
    Analizza sentiment di un singolo testo tramite BERT microservice
    
    Body:
        - text: Testo da analizzare
    
    Returns:
        - stars: float 1.0-5.0
        - sentiment: str (very_negative, negative, neutral, positive, very_positive)
        - confidence: float 0.0-1.0
    
    Example:
        POST /sentiment/analyze
        {"text": "This meeting was very productive!"}
        
        Response:
        {"stars": 4.8, "sentiment": "very_positive", "confidence": 0.92}
    """
    result = await call_bert_service("analyze", {"text": request.text})
    return result

@app.post("/sentiment/batch")
async def analyze_sentiment_batch(request: BatchSentimentRequest):
    """
    Analizza sentiment per batch di testi tramite BERT microservice
    
    Più efficiente di chiamate singole (~10x faster)
    
    Body:
        - texts: Lista di testi (max 100)
    
    Example:
        POST /sentiment/batch
        {"texts": ["Great work!", "This is terrible", "Not sure"]}
    """
    result = await call_bert_service("batch", {"texts": request.texts})
    return result

@app.get("/meeting/{meetingId}/sentiment")
async def get_transcript_with_sentiment(
    meetingId: str,
    participant_id: Optional[str] = None,
    include_embeddings: bool = Query(
        False,
        description="Includi embeddings E5 (384-dim) - computazionalmente costoso"
    )
):
    """
    Ottieni transcript arricchito con sentiment analysis
    
    Query params:
    - participant_id: filtra per partecipante (opzionale)
    - include_embeddings: se True, include anche embeddings E5
    
    Workflow:
    1. Ottieni transcript dal database
    2. Filtra per partecipante (se richiesto)
    3. Chiama BERT microservice per sentiment (batch)
    4. Opzionalmente chiama E5 microservice per embeddings
    5. Combina risultati e calcola statistiche
    
    Returns:
        - transcript: Lista messaggi con sentiment
        - metadata: Stats aggregate (avg_stars, positive_ratio, etc.)
    
    Example:
        GET /meeting/mtg001/sentiment
        
        Response:
        {
          "transcript": [
            {
              "uid": "12345",
              "nickname": "Alice",
              "text": "Great meeting!",
              "from": "00:00:01.000",
              "to": "00:00:05.000",
              "sentiment": {
                "stars": 4.5,
                "sentiment": "very_positive",
                "confidence": 0.89
              }
            }
          ],
          "metadata": {
            "language": "en",
            "sentiment_stats": {
              "average_stars": 3.8,
              "positive_ratio": 0.65,
              "total_analyzed": 20
            }
          }
        }
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
                "sentiment_stats": {
                    "average_stars": 0,
                    "positive_ratio": 0,
                    "total_analyzed": 0
                }
            }
        }
    
    # 4. Chiama BERT microservice per sentiment (batch efficiente)
    sentiment_response = await call_bert_service("batch", {"texts": texts})
    sentiments = sentiment_response["results"]
    
    # 5. Opzionalmente chiama E5 microservice per embeddings
    embeddings = None
    if include_embeddings:
        embedding_response = await call_e5_service("batch-embed", {"texts": texts})
        embeddings = embedding_response["embeddings"]
    
    # 6. Combina transcript con sentiment (e embeddings)
    enriched_transcript = []
    total_stars = 0
    positive_count = 0
    
    for i, entry in enumerate(transcript):
        entry_dict = entry.dict(by_alias=True)
        entry_dict['sentiment'] = sentiments[i]
        
        if embeddings:
            entry_dict['embedding'] = embeddings[i]
        
        enriched_transcript.append(entry_dict)
        
        # Accumula per stats
        total_stars += sentiments[i]['stars']
        if sentiments[i]['stars'] >= 3.5:
            positive_count += 1
    
    # 7. Calcola statistiche aggregate
    avg_stars = total_stars / len(sentiments)
    positive_ratio = positive_count / len(sentiments)
    
    return {
        "transcript": enriched_transcript,
        "metadata": {
            "language": "en",
            "sentiment_stats": {
                "average_stars": round(avg_stars, 2),
                "positive_ratio": round(positive_ratio, 2),
                "total_analyzed": len(sentiments)
            }
        }
    }

# ============================================
# SIMILARITY SEARCH ENDPOINTS (NUOVI)
# Orchestrano chiamate a E5 microservice
# ============================================

@app.post("/meeting/{meetingId}/similarity")
async def find_similar_messages(
    meetingId: str,
    request: SimilaritySearchRequest
):
    """
    Trova messaggi semanticamente simili usando E5 microservice
    
    Non cerca keywords esatte, ma significato semantico!
    
    Body:
        - query: Testo di ricerca
        - top_k: Numero di risultati (default: 5, max: 20)
    
    Workflow:
    1. Ottieni tutti i messaggi del transcript
    2. Chiama E5 microservice per similarity search
    3. Arricchisci risultati con metadata (speaker, timestamp)
    
    Example:
        POST /meeting/mtg001/similarity
        {
          "query": "We need to make a decision",
          "top_k": 3
        }
        
        Response:
        {
          "query": "We need to make a decision",
          "similar_messages": [
            {
              "text": "We need to make a decision on this by the end of the week.",
              "similarity": 0.92,
              "rank": 1,
              "speaker": "Alice",
              "timestamp": "00:02:15.000"
            },
            {
              "text": "I think we should move forward with this approach.",
              "similarity": 0.85,
              "rank": 2,
              "speaker": "Bob",
              "timestamp": "00:05:30.000"
            }
          ]
        }
    """
    # 1. Ottieni meeting
    meeting = MOCK_MEETINGS.get(meetingId)
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meetingId} not found"
        )
    
    transcript = meeting["transcript"]
    
    # 2. Estrai testi candidati
    candidate_texts = [entry.text for entry in transcript]
    
    if not candidate_texts:
        return {
            "query": request.query,
            "similar_messages": []
        }
    
    # 3. Chiama E5 microservice per similarity search
    similarity_response = await call_e5_service("similarity", {
        "query": request.query,
        "candidates": candidate_texts,
        "top_k": request.top_k
    })
    
    # 4. Arricchisci risultati con metadata del transcript
    results = similarity_response["results"]
    similar_messages = []
    
    for result in results:
        # Trova entry originale per ottenere metadata
        entry = next(e for e in transcript if e.text == result["text"])
        
        similar_messages.append({
            "text": result["text"],
            "similarity": result["similarity"],
            "rank": result["rank"],
            "speaker": entry.nickname,
            "timestamp": entry.from_field
        })
    
    return {
        "query": request.query,
        "similar_messages": similar_messages
    }

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
    """
    Status dettagliato di tutti i microservizi
    
    Verifica health e ottiene info da ogni servizio
    """
    # Check health
    bert_healthy = await check_service_health(BERT_SERVICE_URL)
    e5_healthy = await check_service_health(E5_SERVICE_URL)
    
    # Ottieni info dettagliate se servizi online
    bert_info = None
    e5_info = None
    
    if bert_healthy:
        try:
            response = await http_client.get(f"{BERT_SERVICE_URL}/info", timeout=5.0)
            bert_info = response.json()
        except:
            pass
    
    if e5_healthy:
        try:
            response = await http_client.get(f"{E5_SERVICE_URL}/info", timeout=5.0)
            e5_info = response.json()
        except:
            pass
    
    return {
        "bert_sentiment": {
            "healthy": bert_healthy,
            "url": BERT_SERVICE_URL,
            "port": 5001,
            "info": bert_info
        },
        "e5_embeddings": {
            "healthy": e5_healthy,
            "url": E5_SERVICE_URL,
            "port": 5002,
            "info": e5_info
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