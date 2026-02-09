import { useState, useEffect } from 'react'

const API_URL = 'http://localhost:8000'

function App() {
  // ============================================
  // STATE MANAGEMENT
  // ============================================
  
  // UI State
  const [activeTab, setActiveTab] = useState('counter')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  // Character Counter State
  const [characterCount, setCharacterCount] = useState(null)
  const [participants, setParticipants] = useState([])
  const [selectedParticipant, setSelectedParticipant] = useState('')
  
  // Sentiment Analysis State
  const [sentimentText, setSentimentText] = useState('')
  const [sentimentResult, setSentimentResult] = useState(null)
  const [transcriptSentiment, setTranscriptSentiment] = useState(null)
  
  // Similarity Search State
  const [similarityQuery, setSimilarityQuery] = useState('')
  const [similarResults, setSimilarResults] = useState(null)
  const [topK, setTopK] = useState(5)

  // ============================================
  // EFFECTS
  // ============================================
  
  useEffect(() => {
    loadParticipants()
  }, [])

  // ============================================
  // API CALLS
  // ============================================
  
  const loadParticipants = async () => {
    try {
      const response = await fetch(`${API_URL}/participants`)
      const data = await response.json()
      setParticipants(data.participants)
    } catch (err) {
      console.error('Errore caricamento partecipanti:', err)
    }
  }

  const countCharacters = async () => {
    setLoading(true)
    setError(null)
    
    try {
      let url = `${API_URL}/meeting/mtg001/character-count`
      if (selectedParticipant) {
        url += `?participant_id=${selectedParticipant}`
      }
      
      const response = await fetch(url)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      
      const data = await response.json()
      setCharacterCount(data)
    } catch (err) {
      setError(err.message)
      setCharacterCount(null)
    } finally {
      setLoading(false)
    }
  }

  const analyzeSingleSentiment = async () => {
    if (!sentimentText.trim()) {
      setError('Inserisci un testo da analizzare')
      return
    }
    
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${API_URL}/sentiment/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: sentimentText })
      })
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      
      const data = await response.json()
      setSentimentResult(data)
    } catch (err) {
      setError(err.message)
      setSentimentResult(null)
    } finally {
      setLoading(false)
    }
  }

  const analyzeTranscriptSentiment = async () => {
    setLoading(true)
    setError(null)
    
    try {
      let url = `${API_URL}/meeting/mtg001/sentiment?include_embeddings=false`
      if (selectedParticipant) {
        url += `&participant_id=${selectedParticipant}`
      }
      
      const response = await fetch(url)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      
      const data = await response.json()
      setTranscriptSentiment(data)
    } catch (err) {
      setError(err.message)
      setTranscriptSentiment(null)
    } finally {
      setLoading(false)
    }
  }

  const searchSimilarMessages = async () => {
    if (!similarityQuery.trim()) {
      setError('Inserisci una query di ricerca')
      return
    }
    
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${API_URL}/meeting/mtg001/similarity`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: similarityQuery,
          top_k: topK 
        })
      })
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      
      const data = await response.json()
      setSimilarResults(data)
    } catch (err) {
      setError(err.message)
      setSimilarResults(null)
    } finally {
      setLoading(false)
    }
  }

  // ============================================
  // RENDER
  // ============================================

  return (
    <div style={styles.container}>
      {/* Header */}
      <Header />

      {/* Tab Navigation */}
      <TabNavigation 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
      />

      {/* Main Content */}
      <div style={styles.content}>
        {/* Error Display */}
        {error && <ErrorMessage message={error} onClose={() => setError(null)} />}

        {/* Tab Content */}
        {activeTab === 'counter' && (
          <CharacterCounterTab
            participants={participants}
            selectedParticipant={selectedParticipant}
            setSelectedParticipant={setSelectedParticipant}
            loading={loading}
            countCharacters={countCharacters}
            characterCount={characterCount}
          />
        )}

        {activeTab === 'sentiment' && (
          <SentimentTab
            sentimentText={sentimentText}
            setSentimentText={setSentimentText}
            loading={loading}
            analyzeSingleSentiment={analyzeSingleSentiment}
            sentimentResult={sentimentResult}
            participants={participants}
            selectedParticipant={selectedParticipant}
            setSelectedParticipant={setSelectedParticipant}
            analyzeTranscriptSentiment={analyzeTranscriptSentiment}
            transcriptSentiment={transcriptSentiment}
          />
        )}

        {activeTab === 'similarity' && (
          <SimilarityTab
            similarityQuery={similarityQuery}
            setSimilarityQuery={setSimilarityQuery}
            topK={topK}
            setTopK={setTopK}
            loading={loading}
            searchSimilarMessages={searchSimilarMessages}
            similarResults={similarResults}
          />
        )}
      </div>

      {/* Footer */}
      <Footer />
    </div>
  )
}

// ============================================
// COMPONENTS
// ============================================

function Header() {
  return (
    <div style={styles.header}>
      <h1 style={styles.title}>🤖 AI-Powered Meeting Analytics</h1>
      <p style={styles.subtitle}>
        Sentiment Analysis + Semantic Search con BERT & E5 (Microservices)
      </p>
    </div>
  )
}

function TabNavigation({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'counter', label: '📊 Character Counter', icon: '📊' },
    { id: 'sentiment', label: '😊 Sentiment Analysis', icon: '😊' },
    { id: 'similarity', label: '🔍 Similarity Search', icon: '🔍' }
  ]

  return (
    <div style={styles.tabContainer}>
      {tabs.map(tab => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          style={{
            ...styles.tab,
            ...(activeTab === tab.id ? styles.activeTab : {})
          }}
        >
          {tab.icon} {tab.label}
        </button>
      ))}
    </div>
  )
}

function ErrorMessage({ message, onClose }) {
  return (
    <div style={styles.error}>
      <span>⚠️ {message}</span>
      <button onClick={onClose} style={styles.errorClose}>✕</button>
    </div>
  )
}

// ============================================
// TAB COMPONENTS
// ============================================

function CharacterCounterTab({ 
  participants, 
  selectedParticipant, 
  setSelectedParticipant,
  loading, 
  countCharacters, 
  characterCount 
}) {
  return (
    <Card>
      <h2 style={styles.cardTitle}>📊 Conteggio Caratteri Transcript</h2>
      
      <ParticipantFilter
        participants={participants}
        selectedParticipant={selectedParticipant}
        setSelectedParticipant={setSelectedParticipant}
      />

      <ActionButton onClick={countCharacters} loading={loading}>
        Conta Caratteri
      </ActionButton>

      {characterCount && (
        <ResultBox>
          <h3 style={styles.resultTitle}>Risultato</h3>
          {characterCount.participant && (
            <InfoBadge>
              📌 Partecipante: <strong>{characterCount.participant.name}</strong>
            </InfoBadge>
          )}
          <MetricsGrid>
            <MetricCard 
              label="Totale Caratteri" 
              value={characterCount.total_characters.toLocaleString()} 
              icon="🔤"
            />
            <MetricCard 
              label="Totale Parole" 
              value={characterCount.total_words.toLocaleString()} 
              icon="📝"
            />
            <MetricCard 
              label="Totale Messaggi" 
              value={characterCount.total_messages} 
              icon="💬"
            />
          </MetricsGrid>
        </ResultBox>
      )}
    </Card>
  )
}

function SentimentTab({
  sentimentText, 
  setSentimentText, 
  loading, 
  analyzeSingleSentiment,
  sentimentResult, 
  participants, 
  selectedParticipant, 
  setSelectedParticipant,
  analyzeTranscriptSentiment, 
  transcriptSentiment
}) {
  return (
    <div style={styles.tabGrid}>
      {/* Single Text Analysis */}
      <Card>
        <h2 style={styles.cardTitle}>😊 Analizza Sentiment Testo</h2>
        <p style={styles.cardDescription}>
          Usa <strong>BERT microservice</strong> (port 5001) per classificare sentiment 1-5 stelle
        </p>
        
        <textarea
          value={sentimentText}
          onChange={(e) => setSentimentText(e.target.value)}
          placeholder="Scrivi un testo da analizzare... (es: 'This meeting was very productive!')"
          style={styles.textarea}
        />

        <ActionButton onClick={analyzeSingleSentiment} loading={loading}>
          🚀 Analizza Sentiment
        </ActionButton>

        {sentimentResult && (
          <ResultBox>
            <SentimentDisplay result={sentimentResult} />
          </ResultBox>
        )}
      </Card>

      {/* Transcript Analysis */}
      <Card>
        <h2 style={styles.cardTitle}>📊 Sentiment Analysis Transcript</h2>
        <p style={styles.cardDescription}>
          Analizza sentiment di tutti i messaggi del meeting tramite <strong>batch API</strong>
        </p>

        <ParticipantFilter
          participants={participants}
          selectedParticipant={selectedParticipant}
          setSelectedParticipant={setSelectedParticipant}
        />

        <ActionButton onClick={analyzeTranscriptSentiment} loading={loading}>
          📈 Analizza Transcript
        </ActionButton>

        {transcriptSentiment && (
          <div>
            <ResultBox>
              <h3 style={styles.resultTitle}>Statistiche Generali</h3>
              <MetricsGrid>
                <MetricCard 
                  label="Sentiment Medio" 
                  value={`⭐ ${transcriptSentiment.metadata.sentiment_stats.average_stars}/5.0`}
                  icon="📈"
                />
                <MetricCard 
                  label="Messaggi Positivi" 
                  value={`${(transcriptSentiment.metadata.sentiment_stats.positive_ratio * 100).toFixed(0)}%`}
                  icon="😊"
                />
                <MetricCard 
                  label="Totale Messaggi" 
                  value={transcriptSentiment.metadata.sentiment_stats.total_analyzed}
                  icon="💬"
                />
              </MetricsGrid>
            </ResultBox>

            <div style={{ marginTop: '1.5rem' }}>
              <h3 style={styles.sectionTitle}>Messaggi con Sentiment</h3>
              <div style={styles.scrollContainer}>
                {transcriptSentiment.transcript.map((entry, idx) => (
                  <MessageCard key={idx} entry={entry} />
                ))}
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}

function SimilarityTab({
  similarityQuery, 
  setSimilarityQuery, 
  topK, 
  setTopK,
  loading, 
  searchSimilarMessages, 
  similarResults
}) {
  return (
    <Card>
      <h2 style={styles.cardTitle}>🔍 Ricerca Semantica con E5</h2>
      <p style={styles.cardDescription}>
        Usa <strong>E5 microservice</strong> (port 5002) per trovare messaggi simili per significato
      </p>

      <textarea
        value={similarityQuery}
        onChange={(e) => setSimilarityQuery(e.target.value)}
        placeholder="Inserisci una frase di ricerca... (es: 'We need to make a decision')"
        style={styles.textareaSmall}
      />

      <div style={styles.sliderContainer}>
        <label style={styles.sliderLabel}>
          Numero di risultati: <strong>{topK}</strong>
        </label>
        <input
          type="range"
          min="1"
          max="10"
          value={topK}
          onChange={(e) => setTopK(parseInt(e.target.value))}
          style={styles.slider}
        />
      </div>

      <ActionButton onClick={searchSimilarMessages} loading={loading}>
        🔎 Cerca Messaggi Simili
      </ActionButton>

      {similarResults && (
        <ResultBox>
          <h3 style={styles.resultTitle}>
            Risultati per: "<em>{similarResults.query}</em>"
          </h3>
          {similarResults.similar_messages.length === 0 ? (
            <p style={styles.noResults}>Nessun risultato trovato</p>
          ) : (
            similarResults.similar_messages.map((msg, idx) => (
              <SimilarMessageCard key={idx} message={msg} />
            ))
          )}
        </ResultBox>
      )}

      {/* Info Box */}
      <InfoBox>
        <strong>💡 Come funziona la ricerca semantica:</strong>
        <ul style={styles.infoList}>
          <li>Non cerca keywords esatte</li>
          <li>Confronta il <em>significato</em> dei testi</li>
          <li>Query "deadline" → trova anche "finish by Friday"</li>
          <li>Usa embeddings a 384 dimensioni</li>
        </ul>
      </InfoBox>
    </Card>
  )
}

function Footer() {
  return (
    <div style={styles.footer}>
      <div style={styles.footerTitle}>🤖 Powered by:</div>
      <div style={styles.footerContent}>
        <span style={styles.footerItem}>
          <strong>BERT Sentiment</strong> (nlptown) - Port 5001
        </span>
        <span style={styles.footerItem}>
          <strong>E5 Embeddings</strong> (agentlans) - Port 5002
        </span>
      </div>
      <div style={styles.footerTech}>
        Backend: <code>FastAPI Gateway</code> | Frontend: <code>React + Vite</code>
      </div>
      <div style={styles.footerLinks}>
        <a href={`${API_URL}/docs`} target="_blank" style={styles.footerLink}>
          📚 API Docs
        </a>
        {' | '}
        <a href={`${API_URL}/services/status`} target="_blank" style={styles.footerLink}>
          🔧 Services Status
        </a>
      </div>
    </div>
  )
}

// ============================================
// UI COMPONENTS
// ============================================

function Card({ children }) {
  return <div style={styles.card}>{children}</div>
}

function ResultBox({ children }) {
  return <div style={styles.resultBox}>{children}</div>
}

function InfoBox({ children }) {
  return <div style={styles.infoBox}>{children}</div>
}

function InfoBadge({ children }) {
  return <div style={styles.infoBadge}>{children}</div>
}

function MetricsGrid({ children }) {
  return <div style={styles.metricsGrid}>{children}</div>
}

function MetricCard({ label, value, icon }) {
  return (
    <div style={styles.metricCard}>
      <span style={styles.metricLabel}>
        {icon} {label}
      </span>
      <span style={styles.metricValue}>{value}</span>
    </div>
  )
}

function ParticipantFilter({ participants, selectedParticipant, setSelectedParticipant }) {
  return (
    <div style={styles.filterContainer}>
      <label style={styles.filterLabel}>
        Filtra per partecipante (opzionale):
      </label>
      <select
        value={selectedParticipant}
        onChange={(e) => setSelectedParticipant(e.target.value)}
        style={styles.select}
      >
        <option value="">Tutti i partecipanti</option>
        {participants.map(p => (
          <option key={p.id} value={p.id}>{p.name}</option>
        ))}
      </select>
    </div>
  )
}

function ActionButton({ onClick, loading, children }) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      style={{
        ...styles.button,
        ...(loading ? styles.buttonDisabled : {})
      }}
    >
      {loading ? '⏳ Caricamento...' : children}
    </button>
  )
}

function SentimentDisplay({ result }) {
  const getEmoji = (sentiment) => {
    const map = {
      'very_negative': '😢',
      'negative': '😕',
      'neutral': '😐',
      'positive': '😊',
      'very_positive': '😁'
    }
    return map[sentiment] || '😐'
  }

  const getColor = (stars) => {
    if (stars >= 4.5) return '#4caf50'
    if (stars >= 3.5) return '#8bc34a'
    if (stars >= 2.5) return '#ff9800'
    if (stars >= 1.5) return '#ff5722'
    return '#f44336'
  }

  return (
    <div style={styles.sentimentDisplay}>
      <div style={styles.sentimentEmoji}>{getEmoji(result.sentiment)}</div>
      <div style={{ ...styles.sentimentStars, color: getColor(result.stars) }}>
        ⭐ {result.stars}/5.0
      </div>
      <div style={styles.sentimentLabel}>
        {result.sentiment.replace('_', ' ')}
      </div>
      <div style={styles.sentimentConfidence}>
        Confidenza: {(result.confidence * 100).toFixed(1)}%
      </div>
    </div>
  )
}

function MessageCard({ entry }) {
  const getColor = (stars) => {
    if (stars >= 4.0) return '#e8f5e9'
    if (stars >= 3.0) return '#fff9e6'
    return '#ffebee'
  }

  const getBorderColor = (stars) => {
    if (stars >= 4.0) return '#4caf50'
    if (stars >= 3.0) return '#ff9800'
    return '#f44336'
  }

  return (
    <div style={{
      ...styles.messageCard,
      backgroundColor: getColor(entry.sentiment.stars),
      borderLeft: `4px solid ${getBorderColor(entry.sentiment.stars)}`
    }}>
      <div style={styles.messageHeader}>
        <strong>{entry.nickname}</strong>
        <span style={styles.messageTime}>{entry.from}</span>
      </div>
      <p style={styles.messageText}>"{entry.text}"</p>
      <div style={styles.messageFooter}>
        <span>⭐ {entry.sentiment.stars}/5.0</span>
        <span style={styles.messageSentiment}>
          {entry.sentiment.sentiment.replace('_', ' ')}
        </span>
      </div>
    </div>
  )
}

function SimilarMessageCard({ message }) {
  const getColor = (score) => {
    if (score >= 0.8) return '#4caf50'
    if (score >= 0.6) return '#8bc34a'
    if (score >= 0.4) return '#ff9800'
    return '#ff5722'
  }

  return (
    <div style={{
      ...styles.similarCard,
      borderLeft: `4px solid ${getColor(message.similarity)}`
    }}>
      <div style={styles.similarHeader}>
        <div style={{ ...styles.similarRank, color: getColor(message.similarity) }}>
          #{message.rank}
        </div>
        <div style={styles.similarScore}>
          Similarity: <strong>{(message.similarity * 100).toFixed(1)}%</strong>
        </div>
      </div>
      <p style={styles.similarText}>"{message.text}"</p>
      <div style={styles.similarMeta}>
        👤 {message.speaker} • 🕐 {message.timestamp}
      </div>
    </div>
  )
}

// ============================================
// STYLES
// ============================================

const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#f5f5f5',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    padding: '2rem'
  },
  header: {
    textAlign: 'center',
    marginBottom: '2rem'
  },
  title: {
    fontSize: '2rem',
    color: '#333',
    marginBottom: '0.5rem',
    fontWeight: '700'
  },
  subtitle: {
    color: '#666',
    fontSize: '0.95rem'
  },
  tabContainer: {
    display: 'flex',
    justifyContent: 'center',
    gap: '1rem',
    marginBottom: '2rem',
    flexWrap: 'wrap'
  },
  tab: {
    padding: '0.75rem 1.5rem',
    fontSize: '0.95rem',
    fontWeight: '500',
    color: '#666',
    backgroundColor: 'white',
    border: '2px solid #e0e0e0',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s'
  },
  activeTab: {
    color: 'white',
    backgroundColor: '#1976d2',
    borderColor: '#1976d2',
    fontWeight: '600'
  },
  content: {
    maxWidth: '1200px',
    margin: '0 auto'
  },
  error: {
    marginBottom: '1rem',
    padding: '1rem',
    backgroundColor: '#fee',
    border: '1px solid #fcc',
    borderRadius: '6px',
    color: '#c33',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  errorClose: {
    background: 'none',
    border: 'none',
    fontSize: '1.2rem',
    cursor: 'pointer',
    color: '#c33'
  },
  tabGrid: {
    display: 'grid',
    gap: '1.5rem',
    gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))'
  },
  card: {
    backgroundColor: 'white',
    borderRadius: '12px',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
    padding: '2rem'
  },
  cardTitle: {
    fontSize: '1.5rem',
    marginBottom: '0.5rem',
    color: '#333'
  },
  cardDescription: {
    color: '#666',
    fontSize: '0.9rem',
    marginBottom: '1.5rem',
    lineHeight: '1.5'
  },
  textarea: {
    width: '100%',
    minHeight: '100px',
    padding: '0.75rem',
    fontSize: '1rem',
    borderRadius: '6px',
    border: '1px solid #ddd',
    marginBottom: '1rem',
    fontFamily: 'inherit',
    resize: 'vertical'
  },
  textareaSmall: {
    width: '100%',
    minHeight: '80px',
    padding: '0.75rem',
    fontSize: '1rem',
    borderRadius: '6px',
    border: '1px solid #ddd',
    marginBottom: '1rem',
    fontFamily: 'inherit',
    resize: 'vertical'
  },
  filterContainer: {
    marginBottom: '1.5rem'
  },
  filterLabel: {
    display: 'block',
    marginBottom: '0.5rem',
    fontSize: '0.9rem',
    fontWeight: '500',
    color: '#666'
  },
  select: {
    width: '100%',
    padding: '0.75rem',
    fontSize: '1rem',
    borderRadius: '6px',
    border: '1px solid #ddd',
    backgroundColor: 'white'
  },
  button: {
    width: '100%',
    padding: '1rem',
    fontSize: '1rem',
    fontWeight: '600',
    color: 'white',
    backgroundColor: '#1976d2',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'background-color 0.2s'
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
    cursor: 'not-allowed'
  },
  resultBox: {
    marginTop: '1.5rem',
    padding: '1.5rem',
    backgroundColor: '#e3f2fd',
    borderRadius: '8px',
    border: '2px solid #1976d2'
  },
  resultTitle: {
    margin: '0 0 1rem 0',
    fontSize: '1.25rem',
    color: '#1976d2'
  },
  infoBadge: {
    marginBottom: '1rem',
    padding: '0.75rem',
    backgroundColor: 'white',
    borderRadius: '6px',
    fontSize: '0.9rem',
    color: '#666'
  },
  metricsGrid: {
    display: 'grid',
    gap: '1rem',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))'
  },
  metricCard: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    padding: '1rem',
    backgroundColor: 'white',
    borderRadius: '6px'
  },
  metricLabel: {
    color: '#666',
    fontSize: '0.9rem'
  },
  metricValue: {
    fontSize: '1.75rem',
    fontWeight: 'bold',
    color: '#1976d2'
  },
  sentimentDisplay: {
    textAlign: 'center'
  },
  sentimentEmoji: {
    fontSize: '4rem',
    marginBottom: '1rem'
  },
  sentimentStars: {
    fontSize: '2.5rem',
    fontWeight: 'bold',
    marginBottom: '0.5rem'
  },
  sentimentLabel: {
    fontSize: '1.2rem',
    color: '#666',
    textTransform: 'capitalize',
    marginBottom: '1rem'
  },
  sentimentConfidence: {
    fontSize: '0.9rem',
    color: '#999'
  },
  scrollContainer: {
    maxHeight: '400px',
    overflowY: 'auto'
  },
  sectionTitle: {
    fontSize: '1.1rem',
    marginBottom: '1rem',
    color: '#333'
  },
  messageCard: {
    padding: '1rem',
    marginBottom: '0.75rem',
    borderRadius: '8px'
  },
  messageHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '0.5rem'
  },
  messageTime: {
    fontSize: '0.85rem',
    color: '#666'
  },
  messageText: {
    margin: '0.5rem 0',
    color: '#333'
  },
  messageFooter: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.85rem',
    color: '#666'
  },
  messageSentiment: {
    textTransform: 'capitalize'
  },
  sliderContainer: {
    marginBottom: '1.5rem'
  },
  sliderLabel: {
    display: 'block',
    marginBottom: '0.5rem',
    fontSize: '0.9rem',
    color: '#666'
  },
  slider: {
    width: '100%'
  },
  similarCard: {
    padding: '1rem',
    marginBottom: '0.75rem',
    backgroundColor: 'white',
    borderRadius: '6px'
  },
  similarHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.5rem'
  },
  similarRank: {
    fontSize: '1.5rem',
    fontWeight: 'bold'
  },
  similarScore: {
    fontSize: '0.9rem',
    color: '#666'
  },
  similarText: {
    margin: '0.5rem 0',
    color: '#333'
  },
  similarMeta: {
    fontSize: '0.85rem',
    color: '#999'
  },
  infoBox: {
    marginTop: '1.5rem',
    padding: '1rem',
    backgroundColor: '#f9f9f9',
    borderRadius: '6px',
    fontSize: '0.85rem',
    color: '#666',
    lineHeight: '1.6'
  },
  infoList: {
    margin: '0.5rem 0 0 0',
    paddingLeft: '1.5rem'
  },
  noResults: {
    textAlign: 'center',
    color: '#999',
    padding: '2rem'
  },
  footer: {
    marginTop: '3rem',
    padding: '1.5rem',
    textAlign: 'center',
    fontSize: '0.85rem',
    color: '#666',
    backgroundColor: 'white',
    borderRadius: '8px'
  },
  footerTitle: {
    marginBottom: '0.5rem',
    fontWeight: '600'
  },
  footerContent: {
    marginBottom: '1rem',
    display: 'flex',
    justifyContent: 'center',
    gap: '2rem',
    flexWrap: 'wrap'
  },
  footerItem: {
    fontSize: '0.9rem'
  },
  footerTech: {
    fontSize: '0.8rem',
    marginBottom: '0.5rem'
  },
  footerLinks: {
    marginTop: '0.5rem'
  },
  footerLink: {
    color: '#1976d2',
    textDecoration: 'none'
  }
}

export default App