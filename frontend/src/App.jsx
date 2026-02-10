import { useState, useEffect } from 'react'

const API_URL = 'http://localhost:8000'

function App() {
  // ============================================
  // STATE MANAGEMENT
  // ============================================
  
  const [isPanelOpen, setIsPanelOpen] = useState(false)
  const [activeView, setActiveView] = useState('overview')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [transcriptSentiment, setTranscriptSentiment] = useState(null)
  const [similarResults, setSimilarResults] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')

  // ============================================
  // LOGICA CALCOLO DATI (MESSAGGI, PARTECIPANTI, DURATA)
  // ============================================

  // Calcoliamo i dati per la dashboard principale solo se transcriptSentiment esiste
  const dashboardStats = transcriptSentiment ? {
    messages: transcriptSentiment.transcript.length,
    participants: new Set(transcriptSentiment.transcript.map(t => t.nickname)).size,
    // Prendiamo il timestamp dell'ultimo messaggio come durata approssimativa
    duration: transcriptSentiment.transcript.length > 0 
      ? transcriptSentiment.transcript[transcriptSentiment.transcript.length - 1].from 
      : "00:00",
    // Stima grezza dei token (caratteri / 4 è una media standard per l'inglese/italiano)
    tokens: Math.round(transcriptSentiment.transcript.reduce((acc, curr) => acc + curr.text.length, 0) / 4)
  } : {
    messages: '-',
    participants: '-',
    duration: '-',
    tokens: '-'
  }

  // ============================================
  // EFFECTS
  // ============================================
  
  useEffect(() => {
    if (isPanelOpen && !transcriptSentiment && !loading) {
      loadTranscriptSentiment()
    }
  }, [isPanelOpen])

  // ============================================
  // API CALLS
  // ============================================
  
  const loadTranscriptSentiment = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${API_URL}/meeting/mtg001/sentiment?include_embeddings=false`)
      if (!response.ok) throw new Error(`Status ${response.status}`)
      const data = await response.json()
      setTranscriptSentiment(data)
    } catch (err) {
      setError('Impossibile sincronizzare i dati della sessione.')
      setTranscriptSentiment(null)
    } finally {
      setLoading(false)
    }
  }

  const searchSimilar = async () => {
    if (!searchQuery.trim()) return
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${API_URL}/meeting/mtg001/similarity`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, top_k: 5 })
      })
      if (!response.ok) throw new Error(`Status ${response.status}`)
      const data = await response.json()
      setSimilarResults(data)
      setActiveView('similarity')
    } catch (err) {
      setError('Errore nel motore di ricerca semantica.')
    } finally {
      setLoading(false)
    }
  }

  const handleSearchKeyPress = (e) => {
    if (e.key === 'Enter') searchSimilar()
  }

  // ============================================
  // RENDER
  // ============================================

  return (
    <div style={styles.appContainer}>
      {/* SIDEBAR - Dark & Professional */}
      <div style={styles.sidebar}>
        <div style={styles.sidebarHeader}>
          <div style={styles.logoCircle}>M</div>
          <span style={styles.logoText}>MEETING<br/>INTELLIGENCE</span>
        </div>

        <div style={styles.sidebarNav}>
          <SidebarItem 
            label="Dashboard Operativa"
            onClick={() => setIsPanelOpen(false)}
            active={!isPanelOpen}
          />
          <SidebarItem 
            label="Analisi AI"
            onClick={() => setIsPanelOpen(!isPanelOpen)}
            active={isPanelOpen}
            badge={transcriptSentiment ? 'Ready' : null}
          />
          <SidebarItem 
            label="Configurazione"
            onClick={() => alert('Funzionalità riservata agli amministratori')}
          />
        </div>

        <div style={styles.sidebarFooter}>
          <div style={styles.userProfile}>
            <div style={styles.userAvatar}>AD</div>
            <div style={styles.userInfo}>
              <span style={styles.userName}>Admin User</span>
              <span style={styles.userRole}>Enterprise Plan</span>
            </div>
          </div>
        </div>
      </div>

      {/* SLIDE-OUT PANEL */}
      {isPanelOpen && (
        <div style={styles.panel}>
          <div style={styles.panelHeader}>
            <div>
              <h2 style={styles.panelTitle}>Analisi Riunione</h2>
              <span style={styles.panelSubtitle}>ID: MTG-001 • Elaborazione Completata</span>
            </div>
            <button
              onClick={() => setIsPanelOpen(false)}
              style={styles.panelClose}
            >
              ✕
            </button>
          </div>

          <div style={styles.searchContainer}>
            <input
              type="text"
              placeholder="Cerca insights nel transcript..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={handleSearchKeyPress}
              style={styles.searchInput}
            />
          </div>

          <div style={styles.tabContainer}>
            <TabButton
              label="Overview"
              active={activeView === 'overview'}
              onClick={() => setActiveView('overview')}
            />
            <TabButton
              label="Transcript"
              active={activeView === 'messages'}
              onClick={() => setActiveView('messages')}
            />
            {similarResults && (
              <TabButton
                label={`Risultati (${similarResults.similar_messages.length})`}
                active={activeView === 'similarity'}
                onClick={() => setActiveView('similarity')}
              />
            )}
          </div>

          <div style={styles.panelContent}>
            {error && (
              <div style={styles.errorBanner}>
                <div style={styles.errorDot}></div>
                {error}
              </div>
            )}

            {loading && (
              <div style={styles.loadingContainer}>
                <div style={styles.loaderLine}></div>
                <p>Elaborazione NLP in corso...</p>
              </div>
            )}

            {!loading && transcriptSentiment && activeView === 'overview' && (
              <OverviewView data={transcriptSentiment} />
            )}

            {!loading && transcriptSentiment && activeView === 'messages' && (
              <MessagesView data={transcriptSentiment} />
            )}

            {!loading && similarResults && activeView === 'similarity' && (
              <SimilarityView data={similarResults} />
            )}
          </div>
        </div>
      )}

      {/* MAIN CONTENT AREA */}
      <div style={{
        ...styles.mainContent,
        marginLeft: isPanelOpen ? '680px' : '280px' // Sidebar (280) + Panel (400)
      }}>
        <div style={styles.topNav}>
          <span style={styles.breadcrumb}>Home / Meeting / <strong>MTG-001</strong></span>
          <div style={styles.statusBadge}>
            {transcriptSentiment ? '● Online' : '○ Offline'}
          </div>
        </div>

        <div style={styles.contentWrapper}>
          <div style={styles.heroSection}>
            <h1 style={styles.pageTitle}>Meeting Board</h1>
            <p style={styles.pageSubtitle}>Piattaforma centralizzata per la gestione dei verbali.</p>
          </div>
          
          <div style={styles.mainCard}>
            <div style={styles.cardHeaderBorder}>
              <h2 style={styles.cardTitle}>Dettagli Sessione</h2>
            </div>
            
            <div style={styles.cardBody}>
              <p style={{lineHeight: '1.6', color: '#475569', marginBottom: '2rem'}}>
                {transcriptSentiment 
                  ? "I dati visualizzati di seguito sono calcolati in tempo reale dal motore NLP basato sul transcript recuperato." 
                  : "Nessun dato caricato. Apri il pannello 'Analisi AI' sulla sinistra per inizializzare il caricamento e popolare le statistiche."}
              </p>

              <div style={styles.statsRow}>
                <FeatureBox 
                  title="Messaggi Totali" 
                  value={dashboardStats.messages} 
                  icon="💬"
                />
                <FeatureBox 
                  title="Partecipanti" 
                  value={dashboardStats.participants} 
                  icon="👥"
                />
                <FeatureBox 
                  title="Durata Stimata" 
                  value={dashboardStats.duration} 
                  icon="⏱️"
                />
                <FeatureBox 
                  title="Token (Est.)" 
                  value={dashboardStats.tokens} 
                  icon="🔢"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ============================================
// COMPONENTS
// ============================================

function SidebarItem({ label, onClick, active, badge }) {
  return (
    <div
      onClick={onClick}
      style={{
        ...styles.sidebarItem,
        ...(active ? styles.sidebarItemActive : {})
      }}
    >
      <div style={styles.sidebarLabelContainer}>
        {active && <div style={styles.activeIndicator}></div>}
        <span style={{...styles.sidebarLabel, fontWeight: active ? '600' : '400'}}>
          {label}
        </span>
      </div>
      {badge && <span style={styles.sidebarBadge}>{badge}</span>}
    </div>
  )
}

function TabButton({ label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        ...styles.tabButton,
        ...(active ? styles.tabButtonActive : {})
      }}
    >
      {label}
    </button>
  )
}

function OverviewView({ data }) {
  const stats = data.metadata.sentiment_stats

  return (
    <div style={styles.viewContainer}>
      <h3 style={styles.sectionHeader}>KPI Sentiment</h3>
      
      <div style={styles.kpiGrid}>
        <KpiCard label="Average Score" value={stats.average_stars.toFixed(1)} sub="/ 5.0" color="#3b82f6" />
        <KpiCard label="Positivity Rate" value={(stats.positive_ratio * 100).toFixed(0)} sub="%" color="#10b981" />
        <KpiCard label="Total Messages" value={stats.total_analyzed} sub="" color="#6366f1" />
      </div>

      <div style={styles.chartSection}>
        <h4 style={styles.chartTitle}>Distribuzione Tono</h4>
        <SentimentBar data={data.transcript} />
        <div style={styles.chartLegend}>
          <span>Negative</span>
          <span>Neutral</span>
          <span>Positive</span>
        </div>
      </div>

      <div style={styles.insightCard}>
        <div style={styles.insightHeader}>AUTO-INSIGHTS</div>
        <ul style={styles.insightList}>
          <li>Il sentiment complessivo è <strong>{stats.average_stars >= 3 ? 'Positivo' : 'Critico'}</strong>.</li>
          <li>Rilevata predominanza di interventi {stats.positive_ratio > 0.5 ? 'costruttivi' : 'neutri o critici'}.</li>
        </ul>
      </div>
    </div>
  )
}

function MessagesView({ data }) {
  return (
    <div style={styles.viewContainer}>
      <h3 style={styles.sectionHeader}>Transcript Log</h3>
      <div style={styles.messageStream}>
        {data.transcript.map((entry, idx) => (
          <MessageBubble key={idx} entry={entry} />
        ))}
      </div>
    </div>
  )
}

function SimilarityView({ data }) {
  return (
    <div style={styles.viewContainer}>
      <h3 style={styles.sectionHeader}>Risultati Semantici</h3>
      <div style={styles.queryBadge}>Query: {data.query}</div>
      <div style={styles.resultsStack}>
        {data.similar_messages.map((msg, idx) => (
          <SimilarResultRow key={idx} message={msg} />
        ))}
      </div>
    </div>
  )
}

function KpiCard({ label, value, sub, color }) {
  return (
    <div style={styles.kpiCard}>
      <div style={styles.kpiLabel}>{label}</div>
      <div style={{...styles.kpiValue, color}}>{value}<span style={styles.kpiSub}>{sub}</span></div>
    </div>
  )
}

function SentimentBar({ data }) {
  const counts = { very_positive: 0, positive: 0, neutral: 0, negative: 0, very_negative: 0 }
  data.forEach(entry => counts[entry.sentiment.sentiment]++)
  const total = data.length
  
  const colors = {
    very_positive: '#059669', positive: '#34d399', neutral: '#94a3b8', negative: '#f87171', very_negative: '#dc2626'
  }

  return (
    <div style={styles.barTrack}>
      {Object.keys(counts).map(key => {
        const pct = (counts[key] / total) * 100
        return pct > 0 ? (
          <div key={key} style={{...styles.barFill, width: `${pct}%`, backgroundColor: colors[key]}} />
        ) : null
      })}
    </div>
  )
}

function MessageBubble({ entry }) {
  const sentimentColor = {
    'very_positive': '#10b981', 'positive': '#34d399', 'neutral': '#cbd5e1', 'negative': '#f87171', 'very_negative': '#ef4444'
  }[entry.sentiment.sentiment]

  return (
    <div style={{...styles.msgBubble, borderLeft: `4px solid ${sentimentColor}`}}>
      <div style={styles.msgMeta}>
        <span style={styles.msgAuthor}>{entry.nickname}</span>
        <span style={styles.msgScore}>{entry.sentiment.stars.toFixed(1)}</span>
      </div>
      <p style={styles.msgText}>{entry.text}</p>
      <div style={styles.msgTime}>{entry.from}</div>
    </div>
  )
}

function SimilarResultRow({ message }) {
  const opacity = Math.max(0.4, message.similarity); 
  return (
    <div style={{...styles.resultRow, opacity}}>
      <div style={styles.resultRank}>#{message.rank}</div>
      <div style={styles.resultContent}>
        <div style={styles.resultText}>"{message.text}"</div>
        <div style={styles.resultMeta}>
          <span style={styles.resultMatch}>{(message.similarity * 100).toFixed(0)}% Match</span>
          <span> • {message.speaker}</span>
        </div>
      </div>
    </div>
  )
}

function FeatureBox({ title, value, icon }) {
  return (
    <div style={styles.featureBox}>
      <div style={styles.fbIcon}>{icon}</div>
      <div style={styles.fbValue}>{value}</div>
      <div style={styles.fbTitle}>{title}</div>
    </div>
  )
}

// ============================================
// STYLES SYSTEM
// ============================================

const styles = {
  appContainer: {
    display: 'flex',
    minHeight: '100vh',
    backgroundColor: '#f1f5f9', 
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    color: '#334155'
  },

  // SIDEBAR (Dark Theme)
  sidebar: {
    width: '280px',
    backgroundColor: '#0f172a', 
    display: 'flex',
    flexDirection: 'column',
    position: 'fixed',
    left: 0,
    top: 0,
    bottom: 0,
    zIndex: 50,
    color: '#94a3b8'
  },
  sidebarHeader: {
    padding: '2rem 1.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    borderBottom: '1px solid #1e293b'
  },
  logoCircle: {
    width: '32px',
    height: '32px',
    backgroundColor: '#3b82f6',
    borderRadius: '8px',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 'bold',
    fontSize: '0.9rem'
  },
  logoText: {
    color: '#f8fafc',
    fontSize: '0.75rem',
    fontWeight: '700',
    letterSpacing: '1px',
    lineHeight: '1.2'
  },
  sidebarNav: {
    padding: '1.5rem 1rem',
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem'
  },
  sidebarItem: {
    padding: '0.75rem 1rem',
    borderRadius: '6px',
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    transition: 'all 0.2s ease',
    fontSize: '0.9rem'
  },
  sidebarItemActive: {
    backgroundColor: '#1e293b',
    color: '#f8fafc'
  },
  sidebarLabelContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem'
  },
  activeIndicator: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    backgroundColor: '#3b82f6',
    boxShadow: '0 0 8px #3b82f6'
  },
  sidebarBadge: {
    fontSize: '0.65rem',
    backgroundColor: '#3b82f6',
    color: 'white',
    padding: '2px 8px',
    borderRadius: '12px',
    fontWeight: '600'
  },
  sidebarFooter: {
    padding: '1.5rem',
    borderTop: '1px solid #1e293b'
  },
  userProfile: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem'
  },
  userAvatar: {
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    backgroundColor: '#334155',
    color: '#cbd5e1',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '0.8rem',
    fontWeight: '600'
  },
  userInfo: {
    display: 'flex',
    flexDirection: 'column'
  },
  userName: { color: '#f8fafc', fontSize: '0.85rem', fontWeight: '500' },
  userRole: { fontSize: '0.7rem', color: '#64748b' },

  // PANEL (Slide-out)
  panel: {
    width: '400px', 
    backgroundColor: 'white',
    boxShadow: '-4px 0 24px rgba(0,0,0,0.08)',
    display: 'flex',
    flexDirection: 'column',
    position: 'fixed',
    left: '280px',
    top: 0,
    bottom: 0,
    zIndex: 40,
    borderRight: '1px solid #e2e8f0'
  },
  panelHeader: {
    padding: '1.5rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    borderBottom: '1px solid #f1f5f9'
  },
  panelTitle: { fontSize: '1.1rem', fontWeight: '700', color: '#1e293b', margin: 0 },
  panelSubtitle: { fontSize: '0.75rem', color: '#94a3b8', marginTop: '4px', display: 'block' },
  panelClose: {
    background: 'none', border: 'none', color: '#cbd5e1', fontSize: '1.2rem', cursor: 'pointer'
  },

  // SEARCH & TABS
  searchContainer: { padding: '1rem 1.5rem', borderBottom: '1px solid #f1f5f9' },
  searchInput: {
    width: '100%', boxSizing: 'border-box', padding: '0.75rem 1rem', fontSize: '0.9rem',
    border: '1px solid #e2e8f0', borderRadius: '8px', outline: 'none',
    backgroundColor: '#f8fafc', transition: 'all 0.2s',
    boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.02)'
  },
  tabContainer: {
    display: 'flex', padding: '0.5rem 1.5rem 0', gap: '1.5rem', borderBottom: '1px solid #e2e8f0'
  },
  tabButton: {
    padding: '0.75rem 0', fontSize: '0.85rem', fontWeight: '500', color: '#64748b',
    border: 'none', background: 'none', cursor: 'pointer', borderBottom: '2px solid transparent',
    transition: 'color 0.2s'
  },
  tabButtonActive: { color: '#3b82f6', borderBottom: '2px solid #3b82f6' },

  // CONTENT AREA
  panelContent: { flex: 1, overflowY: 'auto', padding: '1.5rem' },
  
  viewContainer: { display: 'flex', flexDirection: 'column', gap: '1.5rem' },
  sectionHeader: { 
    fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', 
    color: '#94a3b8', fontWeight: '600', marginBottom: '0.5rem' 
  },

  // KPI CARDS
  kpiGrid: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' },
  kpiCard: {
    padding: '1rem', backgroundColor: 'white', borderRadius: '8px',
    border: '1px solid #e2e8f0', boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
    textAlign: 'center'
  },
  kpiLabel: { fontSize: '0.7rem', color: '#64748b', marginBottom: '0.25rem' },
  kpiValue: { fontSize: '1.25rem', fontWeight: '700', lineHeight: '1' },
  kpiSub: { fontSize: '0.7rem', fontWeight: '400', opacity: 0.7 },

  // CHARTS
  chartSection: { padding: '1rem', backgroundColor: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' },
  chartTitle: { fontSize: '0.8rem', fontWeight: '600', marginBottom: '0.75rem', color: '#475569' },
  barTrack: { display: 'flex', height: '12px', borderRadius: '6px', overflow: 'hidden', backgroundColor: '#e2e8f0' },
  barFill: { height: '100%' },
  chartLegend: { display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: '#94a3b8', marginTop: '0.5rem' },

  // INSIGHTS
  insightCard: {
    padding: '1rem', backgroundColor: '#eff6ff', border: '1px solid #dbeafe', borderRadius: '8px'
  },
  insightHeader: { fontSize: '0.7rem', color: '#3b82f6', fontWeight: '700', marginBottom: '0.5rem', letterSpacing: '0.5px' },
  insightList: { margin: 0, paddingLeft: '1rem', fontSize: '0.85rem', color: '#334155', lineHeight: '1.5' },

  // MESSAGES
  messageStream: { display: 'flex', flexDirection: 'column', gap: '1rem' },
  msgBubble: {
    padding: '1rem', backgroundColor: 'white', borderRadius: '0 8px 8px 0',
    border: '1px solid #e2e8f0', borderLeftWidth: '4px',
    boxShadow: '0 1px 2px rgba(0,0,0,0.02)'
  },
  msgMeta: { display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' },
  msgAuthor: { fontSize: '0.8rem', fontWeight: '700', color: '#1e293b' },
  msgScore: { fontSize: '0.7rem', color: '#94a3b8', backgroundColor: '#f1f5f9', padding: '1px 6px', borderRadius: '4px' },
  msgText: { fontSize: '0.9rem', lineHeight: '1.5', color: '#475569', margin: 0 },
  msgTime: { fontSize: '0.7rem', color: '#cbd5e1', marginTop: '0.5rem', textAlign: 'right' },

  // SEARCH RESULTS
  queryBadge: { 
    display: 'inline-block', fontSize: '0.75rem', color: '#3b82f6', backgroundColor: '#eff6ff', 
    padding: '4px 8px', borderRadius: '4px', marginBottom: '1rem', border: '1px solid #dbeafe' 
  },
  resultsStack: { display: 'flex', flexDirection: 'column', gap: '0.5rem' },
  resultRow: { 
    display: 'flex', gap: '0.75rem', padding: '0.75rem', backgroundColor: 'white', 
    border: '1px solid #e2e8f0', borderRadius: '6px' 
  },
  resultRank: { fontSize: '0.8rem', fontWeight: '700', color: '#cbd5e1' },
  resultContent: { flex: 1 },
  resultText: { fontSize: '0.85rem', color: '#334155', marginBottom: '0.25rem', fontStyle: 'italic' },
  resultMatch: { color: '#10b981', fontWeight: '600' },
  resultMeta: { fontSize: '0.7rem', color: '#94a3b8' },

  // MAIN LAYOUT & HERO
  mainContent: { flex: 1, transition: 'margin-left 0.3s ease', backgroundColor: '#f1f5f9' },
  topNav: { 
    height: '60px', borderBottom: '1px solid #e2e8f0', backgroundColor: 'white', 
    display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 2rem' 
  },
  breadcrumb: { fontSize: '0.85rem', color: '#64748b' },
  statusBadge: { fontSize: '0.75rem', color: '#10b981', fontWeight: '600', backgroundColor: '#ecfdf5', padding: '4px 8px', borderRadius: '12px' },
  contentWrapper: { padding: '2rem 3rem', maxWidth: '1200px', margin: '0 auto' },
  heroSection: { marginBottom: '2.5rem' },
  pageTitle: { fontSize: '1.8rem', fontWeight: '800', color: '#0f172a', margin: '0 0 0.5rem 0', letterSpacing: '-0.02em' },
  pageSubtitle: { fontSize: '1rem', color: '#64748b' },
  
  mainCard: { 
    backgroundColor: 'white', borderRadius: '12px', 
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
    border: '1px solid #e2e8f0'
  },
  cardHeaderBorder: { padding: '1.5rem 2rem', borderBottom: '1px solid #f1f5f9' },
  cardTitle: { fontSize: '1.1rem', fontWeight: '600', color: '#1e293b', margin: 0 },
  cardBody: { padding: '2rem' },
  
  statsRow: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '2rem', marginTop: '2rem' },
  featureBox: { padding: '1.5rem', backgroundColor: '#f8fafc', borderRadius: '8px', textAlign: 'center', border: '1px solid #e2e8f0' },
  fbIcon: { fontSize: '1.5rem', marginBottom: '0.5rem', filter: 'grayscale(100%) opacity(0.7)' }, // Icone desaturate per stile formale
  fbValue: { fontSize: '1.8rem', fontWeight: '700', color: '#3b82f6', marginBottom: '0.25rem', lineHeight: '1.2' },
  fbTitle: { fontSize: '0.7rem', fontWeight: '600', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' },

  // UTILS
  errorBanner: { padding: '1rem', backgroundColor: '#fef2f2', border: '1px solid #fee2e2', borderRadius: '6px', color: '#991b1b', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' },
  errorDot: { width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#ef4444' },
  loadingContainer: { padding: '3rem', textAlign: 'center', color: '#64748b', fontSize: '0.85rem' },
  loaderLine: { width: '40px', height: '4px', backgroundColor: '#3b82f6', margin: '0 auto 1rem auto', borderRadius: '2px' }
}

export default App