import { useState, useEffect } from 'react'

const API_URL = 'http://localhost:8000'

// ============================================
// iOS MODERN WIDGET BOARD - CUSTOMIZABLE
// Ogni widget personalizzabile singolarmente
// ============================================

function App() {
  const [meetingData, setMeetingData] = useState(null)
  const [participants, setParticipants] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Configurazioni per ogni widget (stile iOS)
  const [widgetConfigs, setWidgetConfigs] = useState({
    messages: {
      participantFilter: null,
      color: '#FF3B30',
      showDetails: true
    },
    sentiment: {
      participantFilter: null,
      color: '#34C759',
      showDetails: true
    },
    toxicity: {
      participantFilter: null,
      color: '#FF9500',
      showDetails: true
    },
    sentimentDist: {
      participantFilter: null,
      color: '#007AFF',
      showLabels: true,
      animated: true
    },
    toxicityGauge: {
      participantFilter: null,
      color: '#5856D6',
      showDetails: true
    },
    messageStream: {
      participantFilter: null,
      color: '#FF2D55',
      limit: 8,
      showTimestamps: true
    }
  })

  // Menu aperto (quale widget sta mostrando settings)
  const [openSettings, setOpenSettings] = useState(null)

  useEffect(() => {
    loadInitialData()
  }, [])

  const loadInitialData = async () => {
    setLoading(true)
    try {
      const respPart = await fetch(`${API_URL}/participants`)
      const dataPart = await respPart.json()
      setParticipants(dataPart.participants)

      const response = await fetch(`${API_URL}/meeting/mtg001/analysis`)
      if (!response.ok) throw new Error(`Status ${response.status}`)
      const data = await response.json()
      setMeetingData(data)
    } catch (err) {
      setError('Unable to load meeting data')
    } finally {
      setLoading(false)
    }
  }

  // Update config per un widget specifico
  const updateWidgetConfig = (widgetId, updates) => {
    setWidgetConfigs(prev => ({
      ...prev,
      [widgetId]: { ...prev[widgetId], ...updates }
    }))
  }

  // Funzione per filtrare transcript in base al widget config
  const getFilteredTranscript = (widgetId) => {
    if (!meetingData) return []
    
    const config = widgetConfigs[widgetId]
    if (!config.participantFilter) {
      return meetingData.transcript
    }

    const participant = participants.find(p => p.id === config.participantFilter)
    if (!participant) return meetingData.transcript

    return meetingData.transcript.filter(entry => 
      entry.nickname === participant.name
    )
  }

  // Calcola stats da transcript filtrato
  const calculateStats = (transcript) => {
    if (!transcript || transcript.length === 0) {
      return {
        total_messages: 0,
        sentiment: {
          distribution: { positive: 0, neutral: 0, negative: 0 },
          average_score: 0,
          positive_ratio: 0
        },
        toxicity: {
          toxic_count: 0,
          toxic_ratio: 0,
          severity_distribution: { low: 0, medium: 0, high: 0 },
          average_toxicity_score: 0
        }
      }
    }

    const total = transcript.length
    let sentimentScoreSum = 0
    let toxicityScoreSum = 0

    const sentimentDist = { positive: 0, neutral: 0, negative: 0 }
    const severityDist = { low: 0, medium: 0, high: 0 }
    let toxicCount = 0

    transcript.forEach(entry => {
      // Sentiment
      const sentLabel = entry.sentiment.label
      if (sentimentDist[sentLabel] !== undefined) {
        sentimentDist[sentLabel]++
      }
      sentimentScoreSum += entry.sentiment.score

      // Toxicity - NUOVO FORMATO
      if (entry.toxicity.is_toxic) {
        toxicCount++
      }
      
      const severity = entry.toxicity.severity
      if (severityDist[severity] !== undefined) {
        severityDist[severity]++
      }
      
      toxicityScoreSum += entry.toxicity.toxicity_score
    })

    return {
      total_messages: total,
      sentiment: {
        distribution: sentimentDist,
        average_score: sentimentScoreSum / total,
        positive_ratio: sentimentDist.positive / total
      },
      toxicity: {
        toxic_count: toxicCount,
        toxic_ratio: toxicCount / total,
        severity_distribution: severityDist,
        average_toxicity_score: toxicityScoreSum / total
      }
    }
  }

  return (
    <div style={styles.appContainer}>
      {/* HEADER - iOS Style */}
      <div style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.headerLeft}>
            <div style={styles.logoCircle}>MI</div>
            <div>
              <h1 style={styles.title}>Meeting Intelligence</h1>
              <p style={styles.subtitle}>Session MTG-001 · Real-time Analytics</p>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div style={styles.errorBanner}>
          <span style={styles.errorIcon}>!</span>
          <span>{error}</span>
        </div>
      )}

      {loading && (
        <div style={styles.loadingContainer}>
          <div style={styles.spinner}></div>
          <p style={styles.loadingText}>Loading analytics...</p>
        </div>
      )}

      {!loading && meetingData && (
        <div style={styles.widgetGrid}>
          {/* KPI Widget - Messages */}
          <CustomizableWidget
            widgetId="messages"
            title="Messages"
            config={widgetConfigs.messages}
            participants={participants}
            onConfigChange={(updates) => updateWidgetConfig('messages', updates)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          >
            {(() => {
              const data = getFilteredTranscript('messages')
              return (
                <>
                  <div style={styles.kpiValue}>{data.length}</div>
                  <div style={styles.kpiLabel}>Total messages</div>
                </>
              )
            })()}
          </CustomizableWidget>

          {/* KPI Widget - Sentiment */}
          <CustomizableWidget
            widgetId="sentiment"
            title="Sentiment"
            config={widgetConfigs.sentiment}
            participants={participants}
            onConfigChange={(updates) => updateWidgetConfig('sentiment', updates)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          >
            {(() => {
              const data = getFilteredTranscript('sentiment')
              const stats = calculateStats(data)
              return (
                <>
                  <div style={styles.kpiValue}>
                    {(stats.sentiment.average_score * 100).toFixed(0)}%
                  </div>
                  <div style={styles.kpiLabel}>
                    {(stats.sentiment.positive_ratio * 100).toFixed(0)}% positive
                  </div>
                </>
              )
            })()}
          </CustomizableWidget>

          {/* KPI Widget - Toxicity */}
          <CustomizableWidget
            widgetId="toxicity"
            title="Toxicity"
            config={widgetConfigs.toxicity}
            participants={participants}
            onConfigChange={(updates) => updateWidgetConfig('toxicity', updates)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          >
            {(() => {
              const data = getFilteredTranscript('toxicity')
              const stats = calculateStats(data)
              return (
                <>
                  <div style={styles.kpiValue}>
                    {stats.toxicity.toxic_count}
                  </div>
                  <div style={styles.kpiLabel}>
                    {(stats.toxicity.toxic_ratio * 100).toFixed(0)}% toxic rate
                  </div>
                </>
              )
            })()}
          </CustomizableWidget>

          {/* Sentiment Distribution - Wide */}
          <CustomizableWidget
            widgetId="sentimentDist"
            title="Sentiment Distribution"
            config={widgetConfigs.sentimentDist}
            participants={participants}
            onConfigChange={(updates) => updateWidgetConfig('sentimentDist', updates)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
            wide
          >
            {(() => {
              const data = getFilteredTranscript('sentimentDist')
              const stats = calculateStats(data)
              return (
                <SentimentDistribution
                  data={stats.sentiment.distribution}
                  config={widgetConfigs.sentimentDist}
                />
              )
            })()}
          </CustomizableWidget>

          {/* Toxicity Gauge */}
          <CustomizableWidget
            widgetId="toxicityGauge"
            title="Toxicity Level"
            config={widgetConfigs.toxicityGauge}
            participants={participants}
            onConfigChange={(updates) => updateWidgetConfig('toxicityGauge', updates)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          >
            {(() => {
              const data = getFilteredTranscript('toxicityGauge')
              const stats = calculateStats(data)
              return (
                <ToxicityGauge
                  score={stats.toxicity.average_toxicity_score}
                  config={widgetConfigs.toxicityGauge}
                />
              )
            })()}
          </CustomizableWidget>

          {/* Message Stream - Wide */}
          <CustomizableWidget
            widgetId="messageStream"
            title="Message Stream"
            config={widgetConfigs.messageStream}
            participants={participants}
            onConfigChange={(updates) => updateWidgetConfig('messageStream', updates)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
            wide
          >
            {(() => {
              const data = getFilteredTranscript('messageStream')
              return (
                <MessageStream
                  messages={data.slice(0, widgetConfigs.messageStream.limit)}
                  config={widgetConfigs.messageStream}
                />
              )
            })()}
          </CustomizableWidget>
        </div>
      )}
    </div>
  )
}

// ============================================
// CUSTOMIZABLE WIDGET (iOS Style)
// ============================================

function CustomizableWidget({
  widgetId,
  title,
  children,
  config,
  participants,
  onConfigChange,
  openSettings,
  setOpenSettings,
  wide
}) {
  const isOpen = openSettings === widgetId

  const toggleSettings = () => {
    setOpenSettings(isOpen ? null : widgetId)
  }

  return (
    <div style={{ ...styles.iosWidget, ...(wide && styles.wideWidget) }}>
      <div style={styles.widgetHeader}>
        <span style={styles.widgetTitle}>{title}</span>
        <div style={styles.headerActions}>
          <div style={{ ...styles.widgetDot, backgroundColor: config.color }} />
          <button onClick={toggleSettings} style={styles.settingsButton}>
            {isOpen ? '✕' : '⋯'}
          </button>
        </div>
      </div>

      {isOpen && (
        <WidgetSettings
          config={config}
          participants={participants}
          onConfigChange={onConfigChange}
        />
      )}

      <div style={styles.widgetContent}>{children}</div>
    </div>
  )
}

// ============================================
// WIDGET SETTINGS PANEL (iOS Style)
// ============================================

function WidgetSettings({ config, participants, onConfigChange }) {
  return (
    <div style={styles.settingsPanel}>
      {/* Filtro Partecipante */}
      <div style={styles.settingRow}>
        <span style={styles.settingLabel}>Filter</span>
        <select
          value={config.participantFilter || ''}
          onChange={(e) =>
            onConfigChange({ participantFilter: e.target.value || null })
          }
          style={styles.settingSelect}
        >
          <option value="">All</option>
          {participants.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      {/* Opzioni specifiche */}
      {config.showDetails !== undefined && (
        <div style={styles.settingRow}>
          <span style={styles.settingLabel}>Show Details</span>
          <label style={styles.toggleSwitch}>
            <input
              type="checkbox"
              checked={config.showDetails}
              onChange={(e) =>
                onConfigChange({ showDetails: e.target.checked })
              }
              style={styles.toggleInput}
            />
            <span
              style={{
                ...styles.toggleSlider,
                backgroundColor: config.showDetails ? config.color : '#3a3a3c'
              }}
            />
          </label>
        </div>
      )}

      {config.showLabels !== undefined && (
        <div style={styles.settingRow}>
          <span style={styles.settingLabel}>Show Labels</span>
          <label style={styles.toggleSwitch}>
            <input
              type="checkbox"
              checked={config.showLabels}
              onChange={(e) =>
                onConfigChange({ showLabels: e.target.checked })
              }
              style={styles.toggleInput}
            />
            <span
              style={{
                ...styles.toggleSlider,
                backgroundColor: config.showLabels ? config.color : '#3a3a3c'
              }}
            />
          </label>
        </div>
      )}

      {config.animated !== undefined && (
        <div style={styles.settingRow}>
          <span style={styles.settingLabel}>Animated</span>
          <label style={styles.toggleSwitch}>
            <input
              type="checkbox"
              checked={config.animated}
              onChange={(e) =>
                onConfigChange({ animated: e.target.checked })
              }
              style={styles.toggleInput}
            />
            <span
              style={{
                ...styles.toggleSlider,
                backgroundColor: config.animated ? config.color : '#3a3a3c'
              }}
            />
          </label>
        </div>
      )}

      {config.showTimestamps !== undefined && (
        <div style={styles.settingRow}>
          <span style={styles.settingLabel}>Timestamps</span>
          <label style={styles.toggleSwitch}>
            <input
              type="checkbox"
              checked={config.showTimestamps}
              onChange={(e) =>
                onConfigChange({ showTimestamps: e.target.checked })
              }
              style={styles.toggleInput}
            />
            <span
              style={{
                ...styles.toggleSlider,
                backgroundColor: config.showTimestamps ? config.color : '#3a3a3c'
              }}
            />
          </label>
        </div>
      )}

      {config.limit !== undefined && (
        <div style={styles.settingRow}>
          <span style={styles.settingLabel}>Message Limit</span>
          <input
            type="number"
            min="3"
            max="20"
            value={config.limit}
            onChange={(e) =>
              onConfigChange({ limit: parseInt(e.target.value) })
            }
            style={styles.numberInput}
          />
        </div>
      )}
    </div>
  )
}

// ============================================
// SENTIMENT DISTRIBUTION CHART
// ============================================

function SentimentDistribution({ data, config }) {
  const total = (data.positive || 0) + (data.neutral || 0) + (data.negative || 0)
  
  if (total === 0) {
    return <div style={styles.emptyState}>No data</div>
  }

  const items = [
    { label: 'Positive', value: data.positive || 0, color: '#34C759' },
    { label: 'Neutral', value: data.neutral || 0, color: '#FFCC00' },
    { label: 'Negative', value: data.negative || 0, color: '#FF3B30' }
  ]

  return (
    <div style={styles.distributionContainer}>
      <div style={styles.barContainer}>
        {items.map((item) => {
          const percentage = (item.value / total) * 100
          return percentage > 0 ? (
            <div
              key={item.label}
              style={{
                width: `${percentage}%`,
                height: '100%',
                backgroundColor: item.color,
                transition: config.animated ? 'width 0.5s ease' : 'none'
              }}
            />
          ) : null
        })}
      </div>
      
      {config.showLabels && (
        <div style={styles.legendContainer}>
          {items.map((item) => (
            <div key={item.label} style={styles.legendItem}>
              <div style={{ ...styles.legendDot, backgroundColor: item.color }} />
              <span style={styles.legendText}>
                {item.label}: {item.value} ({((item.value / total) * 100).toFixed(0)}%)
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ============================================
// TOXICITY GAUGE
// ============================================

function ToxicityGauge({ score, config }) {
  const safeScore = score ?? 0
  const percentage = (safeScore * 100).toFixed(0)
  
  const getColor = () => {
    if (safeScore < 0.3) return '#34C759'
    if (safeScore < 0.6) return '#FFCC00'
    return '#FF3B30'
  }

  const getLabel = () => {
    if (safeScore < 0.3) return 'LOW'
    if (safeScore < 0.6) return 'MEDIUM'
    return 'HIGH'
  }

  return (
    <div style={styles.gaugeContainer}>
      <div style={{ ...styles.gaugeCircle, borderColor: getColor() }}>
        <div style={styles.gaugeInner}>
          <div style={{ ...styles.gaugeValue, color: getColor() }}>
            {percentage}%
          </div>
          <div style={styles.gaugeLabel}>{getLabel()}</div>
        </div>
      </div>
      {config.showDetails && (
        <div style={styles.gaugeDetails}>
          <div style={styles.detailItem}>
            <span style={styles.detailLabel}>Toxicity Score</span>
            <span style={styles.detailValue}>{safeScore.toFixed(3)}</span>
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================
// MESSAGE STREAM
// ============================================

function MessageStream({ messages, config }) {
  if (!messages || messages.length === 0) {
    return <div style={styles.emptyState}>No messages</div>
  }

  return (
    <div style={styles.messageStreamContainer}>
      {messages.map((msg, idx) => (
        <MessageBubble key={idx} message={msg} config={config} />
      ))}
    </div>
  )
}

function MessageBubble({ message, config }) {
  const getSentimentColor = (label) => {
    if (label === 'positive') return '#34C759'
    if (label === 'neutral') return '#FFCC00'
    return '#FF3B30'
  }

  const getToxicityBadge = (toxicity) => {
    if (!toxicity.is_toxic) return null
    
    // Colore basato su severity
    const colors = {
      low: '#FFCC00',
      medium: '#FF9500',
      high: '#FF3B30'
    }
    
    return {
      text: toxicity.severity.toUpperCase(),
      color: colors[toxicity.severity] || '#FF3B30'
    }
  }

  const badge = getToxicityBadge(message.toxicity)

  return (
    <div style={styles.messageBubble}>
      <div style={styles.bubbleHeader}>
        <span style={styles.bubbleAuthor}>{message.nickname}</span>
        <div style={styles.bubbleBadges}>
          <span
            style={{
              ...styles.sentimentBadge,
              backgroundColor: getSentimentColor(message.sentiment.label)
            }}
          >
            {(message.sentiment.score * 100).toFixed(0)}%
          </span>
          {badge && (
            <span
              style={{
                ...styles.toxicBadge,
                backgroundColor: badge.color
              }}
            >
              {badge.text}
            </span>
          )}
        </div>
      </div>
      <p style={styles.bubbleText}>{message.text}</p>
      {config.showTimestamps && (
        <span style={styles.bubbleTime}>{message.from}</span>
      )}
    </div>
  )
}

// ============================================
// STYLES - iOS DARK WIDGETS
// ============================================

const styles = {
  appContainer: {
    minHeight: '100vh',
    backgroundColor: '#1c1c1e',
    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif',
    color: '#fff'
  },

  // HEADER
  header: {
    position: 'sticky',
    top: 0,
    zIndex: 100,
    backgroundColor: '#2c2c2e',
    borderBottom: '1px solid #3a3a3c',
    padding: '1rem 0'
  },
  headerContent: {
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '0 2rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem'
  },
  logoCircle: {
    width: '50px',
    height: '50px',
    borderRadius: '12px',
    background: 'linear-gradient(135deg, #FF3B30 0%, #FF9500 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.2rem',
    fontWeight: '700',
    color: '#fff',
    boxShadow: '0 4px 12px rgba(255, 59, 48, 0.4)'
  },
  title: {
    margin: 0,
    fontSize: '1.4rem',
    fontWeight: '700',
    color: '#fff'
  },
  subtitle: {
    margin: 0,
    fontSize: '0.85rem',
    color: '#8e8e93',
    fontWeight: '500'
  },

  // ERROR & LOADING
  errorBanner: {
    padding: '1rem 2rem',
    margin: '1rem 2rem',
    backgroundColor: '#3a3a3c',
    borderRadius: '12px',
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    fontSize: '0.9rem',
    color: '#FF3B30'
  },
  errorIcon: {
    width: '24px',
    height: '24px',
    borderRadius: '12px',
    backgroundColor: '#FF3B30',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '700'
  },
  loadingContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '60vh',
    gap: '1.5rem'
  },
  spinner: {
    width: '50px',
    height: '50px',
    border: '4px solid #3a3a3c',
    borderTop: '4px solid #007AFF',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite'
  },
  loadingText: {
    fontSize: '0.95rem',
    fontWeight: '500',
    color: '#8e8e93'
  },

  // WIDGET GRID
  widgetGrid: {
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '2rem',
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '1.25rem'
  },

  // iOS WIDGET CARD
  iosWidget: {
    borderRadius: '20px',
    padding: '1.5rem',
    backgroundColor: '#2c2c2e',
    border: '1px solid #3a3a3c',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
    transition: 'transform 0.2s ease, box-shadow 0.2s ease'
  },
  wideWidget: {
    gridColumn: 'span 2'
  },
  widgetHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1.25rem'
  },
  widgetTitle: {
    fontSize: '1rem',
    fontWeight: '600',
    color: '#fff',
    letterSpacing: '0.02em'
  },
  headerActions: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem'
  },
  widgetDot: {
    width: '12px',
    height: '12px',
    borderRadius: '6px',
    boxShadow: '0 0 8px currentColor'
  },
  settingsButton: {
    width: '32px',
    height: '32px',
    borderRadius: '8px',
    border: 'none',
    backgroundColor: '#3a3a3c',
    color: '#fff',
    fontSize: '1.2rem',
    fontWeight: '600',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s ease'
  },
  widgetContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem'
  },

  // SETTINGS PANEL
  settingsPanel: {
    marginBottom: '1.25rem',
    padding: '1rem',
    borderRadius: '12px',
    backgroundColor: '#1c1c1e',
    border: '1px solid #3a3a3c',
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    animation: 'slideDown 0.3s ease'
  },
  settingRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '1rem'
  },
  settingLabel: {
    fontSize: '0.85rem',
    fontWeight: '500',
    color: '#8e8e93'
  },
  settingSelect: {
    padding: '0.5rem 0.75rem',
    fontSize: '0.85rem',
    borderRadius: '8px',
    border: '1px solid #3a3a3c',
    backgroundColor: '#2c2c2e',
    color: '#fff',
    outline: 'none',
    fontWeight: '500',
    cursor: 'pointer',
    minWidth: '120px'
  },
  numberInput: {
    padding: '0.5rem 0.75rem',
    fontSize: '0.85rem',
    borderRadius: '8px',
    border: '1px solid #3a3a3c',
    backgroundColor: '#2c2c2e',
    color: '#fff',
    outline: 'none',
    fontWeight: '500',
    width: '80px'
  },

  // COLOR PICKER
  colorPicker: {
    display: 'flex',
    gap: '0.5rem',
    flexWrap: 'wrap'
  },
  colorButton: {
    width: '32px',
    height: '32px',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s ease'
  },

  // TOGGLE SWITCH (iOS Style)
  toggleSwitch: {
    position: 'relative',
    display: 'inline-block',
    width: '50px',
    height: '28px',
    cursor: 'pointer'
  },
  toggleInput: {
    opacity: 0,
    width: 0,
    height: 0
  },
  toggleSlider: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderRadius: '14px',
    transition: 'background-color 0.3s ease',
    display: 'flex',
    alignItems: 'center',
    padding: '0 2px'
  },

  // KPI
  kpiValue: {
    fontSize: '3rem',
    fontWeight: '700',
    color: '#fff',
    textAlign: 'center',
    lineHeight: '1'
  },
  kpiLabel: {
    fontSize: '0.85rem',
    fontWeight: '500',
    color: '#8e8e93',
    textAlign: 'center'
  },

  // DISTRIBUTION
  distributionContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.25rem'
  },
  barContainer: {
    display: 'flex',
    height: '32px',
    borderRadius: '16px',
    overflow: 'hidden',
    boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.4)'
  },
  legendContainer: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '0.75rem'
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem'
  },
  legendDot: {
    width: '10px',
    height: '10px',
    borderRadius: '5px'
  },
  legendText: {
    fontSize: '0.8rem',
    fontWeight: '500',
    color: '#8e8e93'
  },

  // GAUGE
  gaugeContainer: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '1rem',
    padding: '1rem 0'
  },
  gaugeCircle: {
    width: '160px',
    height: '160px',
    borderRadius: '80px',
    border: '12px solid',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    boxShadow: '0 0 20px currentColor'
  },
  gaugeInner: {
    textAlign: 'center'
  },
  gaugeValue: {
    fontSize: '2.5rem',
    fontWeight: '700'
  },
  gaugeLabel: {
    fontSize: '0.75rem',
    fontWeight: '600',
    color: '#8e8e93',
    letterSpacing: '1px',
    marginTop: '0.25rem'
  },
  gaugeDetails: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem'
  },
  detailItem: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    backgroundColor: '#1c1c1e'
  },
  detailLabel: {
    fontSize: '0.8rem',
    fontWeight: '500',
    color: '#8e8e93'
  },
  detailValue: {
    fontSize: '0.9rem',
    fontWeight: '600',
    color: '#fff'
  },

  // MESSAGE STREAM
  messageStreamContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
    maxHeight: '400px',
    overflowY: 'auto'
  },
  messageBubble: {
    padding: '0.75rem',
    borderRadius: '12px',
    backgroundColor: '#1c1c1e',
    border: '1px solid #3a3a3c',
    transition: 'all 0.2s ease'
  },
  bubbleHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.5rem'
  },
  bubbleAuthor: {
    fontSize: '0.85rem',
    fontWeight: '600',
    color: '#fff'
  },
  bubbleBadges: {
    display: 'flex',
    gap: '0.5rem'
  },
  sentimentBadge: {
    fontSize: '0.7rem',
    fontWeight: '700',
    color: '#000',
    padding: '0.25rem 0.5rem',
    borderRadius: '6px'
  },
  toxicBadge: {
    fontSize: '0.7rem',
    fontWeight: '700',
    color: '#fff',
    padding: '0.25rem 0.5rem',
    borderRadius: '6px'
  },
  bubbleText: {
    margin: 0,
    fontSize: '0.85rem',
    lineHeight: '1.5',
    color: '#d1d1d6'
  },
  bubbleTime: {
    display: 'block',
    marginTop: '0.5rem',
    fontSize: '0.7rem',
    fontWeight: '500',
    color: '#636366'
  },

  // EMPTY STATE
  emptyState: {
    textAlign: 'center',
    padding: '2rem',
    color: '#8e8e93',
    fontSize: '0.9rem',
    fontWeight: '500'
  }
}

// CSS Animations & Hover Effects
const styleSheet = document.createElement('style')
styleSheet.textContent = `
@keyframes spin {
  to { transform: rotate(360deg); }
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Widget hover effects */
div[style*="iosWidget"]:hover {
  transform: translateY(-4px);
}

/* Settings button hover */
button:hover {
  transform: scale(1.05);
}

button:active {
  transform: scale(0.95);
}

/* Color button hover */
button[style*="colorButton"]:hover {
  transform: scale(1.15);
}

/* Message bubble hover */
div[style*="messageBubble"]:hover {
  background-color: #2c2c2e;
  border-color: #48484a;
}

/* Toggle switch animation */
input[type="checkbox"]:checked + span::after {
  content: '';
  position: absolute;
  width: 24px;
  height: 24px;
  background-color: white;
  border-radius: 12px;
  right: 2px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
}

input[type="checkbox"]:not(:checked) + span::after {
  content: '';
  position: absolute;
  width: 24px;
  height: 24px;
  background-color: white;
  border-radius: 12px;
  left: 2px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #1c1c1e;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: #3a3a3c;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #48484a;
}
`
document.head.appendChild(styleSheet)

export default App