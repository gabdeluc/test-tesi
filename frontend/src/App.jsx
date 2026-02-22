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
    timelineSentiment: {
      participantFilter: null,
      color: '#00C7BE',
      showGrid: true,
      showArea: true,
      metric: 'sentiment'
    },
    timelineToxicity: {
      participantFilter: null,
      color: '#FF6B6B',
      showGrid: true,
      showArea: true,
      metric: 'toxicity'
    },
    messageStream: {
      participantFilter: null,
      color: '#FF2D55',
      limit: 30,
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
      // Carica partecipanti
      const respPart = await fetch(`${API_URL}/participants`)
      const dataPart = await respPart.json()
      setParticipants(dataPart.participants)

      // Carica meeting
      const response = await fetch(`${API_URL}/meeting/mtg001/analysis`)
      if (!response.ok) throw new Error(`Status ${response.status}`)
      const data = await response.json()
      setMeetingData(data)
      setError(null)
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
              <p style={styles.subtitle}>MTG-001 · Real-time Analytics</p>
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

          {/* Timeline Sentiment Widget - Wide */}
          <CustomizableWidget
            widgetId="timelineSentiment"
            title="Sentiment Timeline"
            config={widgetConfigs.timelineSentiment}
            participants={participants}
            onConfigChange={(updates) => updateWidgetConfig('timelineSentiment', updates)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
            wide
          >
            {(() => {
              const data = getFilteredTranscript('timelineSentiment')
              return (
                <TimelineChart
                  messages={data}
                  config={widgetConfigs.timelineSentiment}
                />
              )
            })()}
          </CustomizableWidget>

          {/* Timeline Toxicity Widget - Wide */}
          <CustomizableWidget
            widgetId="timelineToxicity"
            title="Toxicity Timeline"
            config={widgetConfigs.timelineToxicity}
            participants={participants}
            onConfigChange={(updates) => updateWidgetConfig('timelineToxicity', updates)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
            wide
          >
            {(() => {
              const data = getFilteredTranscript('timelineToxicity')
              return (
                <TimelineChart
                  messages={data}
                  config={widgetConfigs.timelineToxicity}
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

      {config.metric !== undefined && (
        <div style={styles.settingRow}>
          <span style={styles.settingLabel}>Metric</span>
          <select
            value={config.metric}
            onChange={(e) => onConfigChange({ metric: e.target.value })}
            style={styles.settingSelect}
          >
            <option value="sentiment">Sentiment</option>
            <option value="toxicity">Toxicity (Inverted)</option>
          </select>
        </div>
      )}

      {config.showGrid !== undefined && (
        <div style={styles.settingRow}>
          <span style={styles.settingLabel}>Show Grid</span>
          <label style={styles.toggleSwitch}>
            <input
              type="checkbox"
              checked={config.showGrid}
              onChange={(e) =>
                onConfigChange({ showGrid: e.target.checked })
              }
              style={styles.toggleInput}
            />
            <span
              style={{
                ...styles.toggleSlider,
                backgroundColor: config.showGrid ? config.color : '#3a3a3c'
              }}
            />
          </label>
        </div>
      )}

      {config.showArea !== undefined && (
        <div style={styles.settingRow}>
          <span style={styles.settingLabel}>Show Area</span>
          <label style={styles.toggleSwitch}>
            <input
              type="checkbox"
              checked={config.showArea}
              onChange={(e) =>
                onConfigChange({ showArea: e.target.checked })
              }
              style={styles.toggleInput}
            />
            <span
              style={{
                ...styles.toggleSlider,
                backgroundColor: config.showArea ? config.color : '#3a3a3c'
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
            min="5"
            max="100"
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
// TIMELINE CHART
// ============================================

function TimelineChart({ messages, config }) {
  const [hoveredPoint, setHoveredPoint] = useState(null)
  
  if (!messages || messages.length === 0) {
    return <div style={styles.emptyState}>No data</div>
  }

  // Dimensioni chart
  const width = 800
  const height = 300
  const padding = { top: 20, right: 20, bottom: 50, left: 60 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom

  // Prepara dati per il grafico
  const dataPoints = messages.map((msg, idx) => {
    // SENTIMENT: usa score diretto (high = positive)
    // TOXICITY: usa score diretto (high = toxic)
    const score = config.metric === 'sentiment' 
      ? msg.sentiment.score 
      : msg.toxicity.toxicity_score  // NON invertire - raw score
    
    return {
      index: idx,
      score: score,
      rawScore: config.metric === 'sentiment' ? msg.sentiment.score : msg.toxicity.toxicity_score,
      timestamp: msg.from,
      message: msg.text,
      nickname: msg.nickname
    }
  })

  // Scale
  const xScale = (index) => padding.left + (index / (dataPoints.length - 1)) * chartWidth
  const yScale = (score) => padding.top + (1 - score) * chartHeight

  // Genera path per la linea
  const linePath = dataPoints.map((point, idx) => {
    const x = xScale(point.index)
    const y = yScale(point.score)
    return idx === 0 ? `M ${x} ${y}` : `L ${x} ${y}`
  }).join(' ')

  // Genera path per l'area sotto la linea
  const areaPath = config.showArea ? 
    `${linePath} L ${xScale(dataPoints.length - 1)} ${padding.top + chartHeight} L ${padding.left} ${padding.top + chartHeight} Z`
    : null

  // Colore basato su metric
  const lineColor = config.color || (config.metric === 'sentiment' ? '#00C7BE' : '#FF6B6B')
  
  // Media per riferimento
  const avgScore = dataPoints.reduce((sum, p) => sum + p.score, 0) / dataPoints.length

  // Label per tooltip
  const getMetricLabel = () => config.metric === 'sentiment' ? 'Sentiment' : 'Toxicity'
  const getMetricDescription = () => config.metric === 'sentiment' 
    ? 'Higher is better' 
    : 'Higher is worse (more toxic)'

  return (
    <div style={styles.timelineContainer}>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
        {/* Griglia di sfondo */}
        {config.showGrid && (
          <g>
            {[0, 0.25, 0.5, 0.75, 1].map((value) => (
              <line
                key={value}
                x1={padding.left}
                y1={yScale(value)}
                x2={padding.left + chartWidth}
                y2={yScale(value)}
                stroke="rgba(255, 255, 255, 0.05)"
                strokeWidth="1"
              />
            ))}
          </g>
        )}

        {/* Linea media */}
        <line
          x1={padding.left}
          y1={yScale(avgScore)}
          x2={padding.left + chartWidth}
          y2={yScale(avgScore)}
          stroke="rgba(255, 255, 255, 0.2)"
          strokeWidth="1"
          strokeDasharray="4 4"
        />
        <text
          x={padding.left + chartWidth + 5}
          y={yScale(avgScore)}
          fill="#8e8e93"
          fontSize="10"
          alignmentBaseline="middle"
        >
          avg
        </text>

        {/* Area sotto la linea */}
        {areaPath && (
          <path
            d={areaPath}
            fill={`url(#gradient-${config.metric})`}
            opacity="0.2"
          />
        )}

        {/* Gradient definition */}
        <defs>
          <linearGradient id={`gradient-${config.metric}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={lineColor} stopOpacity="0.8" />
            <stop offset="100%" stopColor={lineColor} stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Linea principale */}
        <path
          d={linePath}
          fill="none"
          stroke={lineColor}
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ filter: `drop-shadow(0 0 8px ${lineColor})` }}
        />

        {/* Punti interattivi */}
        {dataPoints.map((point, idx) => {
          const x = xScale(point.index)
          const y = yScale(point.score)
          const isHovered = hoveredPoint === idx
          
          return (
            <g key={idx}>
              <circle
                cx={x}
                cy={y}
                r={isHovered ? 6 : 4}
                fill={lineColor}
                stroke="#1c1c1e"
                strokeWidth="2"
                style={{ 
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  filter: isHovered ? `drop-shadow(0 0 8px ${lineColor})` : 'none'
                }}
                onMouseEnter={() => setHoveredPoint(idx)}
                onMouseLeave={() => setHoveredPoint(null)}
              />
            </g>
          )
        })}

        {/* Asse Y labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((value) => (
          <text
            key={value}
            x={padding.left - 10}
            y={yScale(value)}
            fill="#8e8e93"
            fontSize="11"
            textAnchor="end"
            alignmentBaseline="middle"
          >
            {(value * 100).toFixed(0)}%
          </text>
        ))}

        {/* Asse X labels (ogni 5 messaggi) */}
        {dataPoints
          .filter((_, idx) => idx % 5 === 0 || idx === dataPoints.length - 1)
          .map((point, idx) => {
            const x = xScale(point.index)
            return (
              <text
                key={idx}
                x={x}
                y={padding.top + chartHeight + 20}
                fill="#8e8e93"
                fontSize="10"
                textAnchor="middle"
              >
                {point.timestamp.split(':').slice(0, 2).join(':')}
              </text>
            )
          })}
      </svg>

      {/* Tooltip on hover */}
      {hoveredPoint !== null && (
        <div style={styles.tooltip}>
          <div style={styles.tooltipHeader}>
            <strong>{dataPoints[hoveredPoint].nickname}</strong>
            <span style={{ color: '#8e8e93', fontSize: '0.8rem' }}>
              {dataPoints[hoveredPoint].timestamp}
            </span>
          </div>
          <div style={styles.tooltipScore}>
            {getMetricLabel()}: <strong style={{ color: lineColor }}>
              {config.metric === 'sentiment' 
                ? `${(dataPoints[hoveredPoint].rawScore * 100).toFixed(0)}%`
                : `${(dataPoints[hoveredPoint].rawScore * 100).toFixed(0)}% toxic`
              }
            </strong>
          </div>
          <div style={styles.tooltipMessage}>
            "{dataPoints[hoveredPoint].message.substring(0, 60)}
            {dataPoints[hoveredPoint].message.length > 60 ? '...' : ''}"
          </div>
        </div>
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
    backgroundColor: 'rgba(28, 28, 30, 0.85)',
    backdropFilter: 'saturate(180%) blur(20px)',
    WebkitBackdropFilter: 'saturate(180%) blur(20px)',
    borderBottom: '0.5px solid rgba(255, 255, 255, 0.1)',
    padding: '1.25rem 0',
    boxShadow: '0 4px 16px rgba(0, 0, 0, 0.3)'
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
    gap: '1.25rem'
  },
  logoCircle: {
    width: '56px',
    height: '56px',
    borderRadius: '14px',
    background: 'linear-gradient(135deg, #FF3B30 0%, #FF9500 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.4rem',
    fontWeight: '700',
    color: '#fff',
    boxShadow: '0 8px 24px rgba(255, 59, 48, 0.5)',
    transition: 'transform 0.3s ease'
  },
  title: {
    margin: 0,
    fontSize: '1.5rem',
    fontWeight: '700',
    color: '#fff',
    letterSpacing: '-0.02em'
  },
  subtitle: {
    margin: 0,
    fontSize: '0.9rem',
    color: '#8e8e93',
    fontWeight: '500',
    letterSpacing: '-0.01em'
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
    padding: '2.5rem 2rem',
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '1.5rem'
  },

  // iOS WIDGET CARD
  iosWidget: {
    borderRadius: '24px',
    padding: '1.75rem',
    backgroundColor: 'rgba(44, 44, 46, 0.6)',
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    border: '0.5px solid rgba(255, 255, 255, 0.08)',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5), 0 2px 8px rgba(0, 0, 0, 0.3)',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
  },
  wideWidget: {
    gridColumn: 'span 2'
  },
  widgetHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1.5rem',
    paddingBottom: '0.75rem',
    borderBottom: '0.5px solid rgba(255, 255, 255, 0.06)'
  },
  widgetTitle: {
    fontSize: '1.05rem',
    fontWeight: '600',
    color: '#fff',
    letterSpacing: '-0.01em'
  },
  headerActions: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.875rem'
  },
  widgetDot: {
    width: '10px',
    height: '10px',
    borderRadius: '5px',
    boxShadow: '0 0 12px currentColor, 0 0 4px currentColor'
  },
  settingsButton: {
    width: '34px',
    height: '34px',
    borderRadius: '10px',
    border: 'none',
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    color: '#fff',
    fontSize: '1.3rem',
    fontWeight: '600',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)'
  },
  widgetContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.25rem'
  },

  // SETTINGS PANEL
  settingsPanel: {
    marginBottom: '1.5rem',
    padding: '1.25rem',
    borderRadius: '16px',
    backgroundColor: 'rgba(28, 28, 30, 0.8)',
    backdropFilter: 'blur(10px)',
    WebkitBackdropFilter: 'blur(10px)',
    border: '0.5px solid rgba(255, 255, 255, 0.08)',
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    animation: 'slideDown 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    boxShadow: '0 4px 16px rgba(0, 0, 0, 0.3)'
  },
  settingRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '1rem',
    padding: '0.5rem 0'
  },
  settingLabel: {
    fontSize: '0.9rem',
    fontWeight: '500',
    color: '#8e8e93',
    letterSpacing: '-0.01em'
  },
  settingSelect: {
    padding: '0.625rem 1rem',
    fontSize: '0.875rem',
    borderRadius: '10px',
    border: '0.5px solid rgba(255, 255, 255, 0.1)',
    backgroundColor: 'rgba(44, 44, 46, 0.6)',
    backdropFilter: 'blur(10px)',
    WebkitBackdropFilter: 'blur(10px)',
    color: '#fff',
    outline: 'none',
    fontWeight: '500',
    cursor: 'pointer',
    minWidth: '140px',
    transition: 'all 0.2s ease'
  },
  numberInput: {
    padding: '0.625rem 1rem',
    fontSize: '0.875rem',
    borderRadius: '10px',
    border: '0.5px solid rgba(255, 255, 255, 0.1)',
    backgroundColor: 'rgba(44, 44, 46, 0.6)',
    backdropFilter: 'blur(10px)',
    WebkitBackdropFilter: 'blur(10px)',
    color: '#fff',
    outline: 'none',
    fontWeight: '500',
    width: '90px',
    textAlign: 'center',
    transition: 'all 0.2s ease'
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
    fontSize: '3.5rem',
    fontWeight: '700',
    color: '#fff',
    textAlign: 'center',
    lineHeight: '1',
    letterSpacing: '-0.03em',
    textShadow: '0 2px 8px rgba(0, 0, 0, 0.3)'
  },
  kpiLabel: {
    fontSize: '0.9rem',
    fontWeight: '500',
    color: '#8e8e93',
    textAlign: 'center',
    letterSpacing: '-0.01em',
    marginTop: '0.5rem'
  },

  // DISTRIBUTION
  distributionContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem'
  },
  barContainer: {
    display: 'flex',
    height: '40px',
    borderRadius: '20px',
    overflow: 'hidden',
    boxShadow: 'inset 0 3px 8px rgba(0, 0, 0, 0.4)',
    border: '0.5px solid rgba(255, 255, 255, 0.06)'
  },
  legendContainer: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '1rem'
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.625rem',
    padding: '0.5rem',
    borderRadius: '10px',
    backgroundColor: 'rgba(28, 28, 30, 0.4)',
    backdropFilter: 'blur(5px)',
    WebkitBackdropFilter: 'blur(5px)',
    transition: 'all 0.2s ease'
  },
  legendDot: {
    width: '12px',
    height: '12px',
    borderRadius: '6px',
    boxShadow: '0 0 8px currentColor'
  },
  legendText: {
    fontSize: '0.85rem',
    fontWeight: '500',
    color: '#8e8e93',
    letterSpacing: '-0.01em'
  },

  // GAUGE
  gaugeContainer: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '1.25rem',
    padding: '1.5rem 0'
  },
  gaugeCircle: {
    width: '180px',
    height: '180px',
    borderRadius: '90px',
    border: '14px solid',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    boxShadow: '0 0 30px currentColor, inset 0 0 20px rgba(0, 0, 0, 0.3)',
    transition: 'all 0.3s ease'
  },
  gaugeInner: {
    textAlign: 'center'
  },
  gaugeValue: {
    fontSize: '2.75rem',
    fontWeight: '700',
    letterSpacing: '-0.02em',
    textShadow: '0 2px 8px rgba(0, 0, 0, 0.3)'
  },
  gaugeLabel: {
    fontSize: '0.8rem',
    fontWeight: '600',
    color: '#8e8e93',
    letterSpacing: '1.5px',
    marginTop: '0.375rem',
    textTransform: 'uppercase'
  },
  gaugeDetails: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.625rem'
  },
  detailItem: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '0.75rem 1.25rem',
    borderRadius: '12px',
    backgroundColor: 'rgba(28, 28, 30, 0.6)',
    backdropFilter: 'blur(10px)',
    WebkitBackdropFilter: 'blur(10px)',
    border: '0.5px solid rgba(255, 255, 255, 0.06)'
  },
  detailLabel: {
    fontSize: '0.85rem',
    fontWeight: '500',
    color: '#8e8e93',
    letterSpacing: '-0.01em'
  },
  detailValue: {
    fontSize: '0.95rem',
    fontWeight: '600',
    color: '#fff',
    letterSpacing: '-0.01em'
  },

  // MESSAGE STREAM
  messageStreamContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.875rem',
    maxHeight: '600px',
    overflowY: 'auto',
    paddingRight: '0.5rem'
  },
  messageBubble: {
    padding: '1rem 1.25rem',
    borderRadius: '16px',
    backgroundColor: 'rgba(28, 28, 30, 0.6)',
    backdropFilter: 'blur(10px)',
    WebkitBackdropFilter: 'blur(10px)',
    border: '0.5px solid rgba(255, 255, 255, 0.06)',
    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
  },
  bubbleHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.625rem'
  },
  bubbleAuthor: {
    fontSize: '0.9rem',
    fontWeight: '600',
    color: '#fff',
    letterSpacing: '-0.01em'
  },
  bubbleBadges: {
    display: 'flex',
    gap: '0.5rem'
  },
  sentimentBadge: {
    fontSize: '0.75rem',
    fontWeight: '700',
    color: '#000',
    padding: '0.3rem 0.6rem',
    borderRadius: '8px',
    letterSpacing: '0.02em'
  },
  toxicBadge: {
    fontSize: '0.7rem',
    fontWeight: '700',
    color: '#fff',
    padding: '0.3rem 0.6rem',
    borderRadius: '8px',
    letterSpacing: '0.5px',
    textTransform: 'uppercase'
  },
  bubbleText: {
    margin: 0,
    fontSize: '0.9rem',
    lineHeight: '1.6',
    color: '#d1d1d6',
    letterSpacing: '-0.01em'
  },
  bubbleTime: {
    display: 'block',
    marginTop: '0.625rem',
    fontSize: '0.75rem',
    fontWeight: '500',
    color: '#636366',
    letterSpacing: '0.02em'
  },

  // EMPTY STATE
  emptyState: {
    textAlign: 'center',
    padding: '2rem',
    color: '#8e8e93',
    fontSize: '0.9rem',
    fontWeight: '500'
  },

  // TIMELINE CHART
  timelineContainer: {
    position: 'relative',
    padding: '1rem 0'
  },
  tooltip: {
    position: 'absolute',
    top: '10px',
    right: '10px',
    backgroundColor: 'rgba(28, 28, 30, 0.95)',
    backdropFilter: 'blur(10px)',
    WebkitBackdropFilter: 'blur(10px)',
    border: '0.5px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '12px',
    padding: '1rem',
    minWidth: '220px',
    maxWidth: '300px',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
    zIndex: 10
  },
  tooltipHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.5rem',
    paddingBottom: '0.5rem',
    borderBottom: '0.5px solid rgba(255, 255, 255, 0.08)'
  },
  tooltipScore: {
    fontSize: '0.85rem',
    color: '#d1d1d6',
    marginBottom: '0.5rem'
  },
  tooltipMessage: {
    fontSize: '0.8rem',
    color: '#8e8e93',
    fontStyle: 'italic',
    lineHeight: '1.4'
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

/* Widget hover effects - Premium */
div[style*="iosWidget"]:hover {
  transform: translateY(-6px);
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.6), 0 4px 12px rgba(0, 0, 0, 0.4);
  border-color: rgba(255, 255, 255, 0.12);
}

/* Logo hover */
div[style*="logoCircle"]:hover {
  transform: scale(1.05) rotate(5deg);
}

/* Settings button hover */
button[style*="settingsButton"]:hover {
  background-color: rgba(255, 255, 255, 0.15);
  transform: scale(1.08);
}

button[style*="settingsButton"]:active {
  transform: scale(0.96);
  background-color: rgba(255, 255, 255, 0.12);
}

/* Message bubble hover */
div[style*="messageBubble"]:hover {
  background-color: rgba(44, 44, 46, 0.6);
  border-color: rgba(255, 255, 255, 0.1);
  transform: translateX(4px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
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
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

input[type="checkbox"]:not(:checked) + span::after {
  content: '';
  position: absolute;
  width: 24px;
  height: 24px;
  background-color: white;
  border-radius: 12px;
  left: 2px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Scrollbar styling - Premium */
::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-track {
  background: rgba(28, 28, 30, 0.4);
  border-radius: 5px;
  margin: 4px;
}

::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.15);
  border-radius: 5px;
  border: 2px solid transparent;
  background-clip: padding-box;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.25);
  background-clip: padding-box;
}

/* Select hover */
select[style*="settingSelect"]:hover {
  background-color: rgba(44, 44, 46, 0.8);
  border-color: rgba(255, 255, 255, 0.2);
}

/* Input hover */
input[type="number"]:hover {
  background-color: rgba(44, 44, 46, 0.8);
  border-color: rgba(255, 255, 255, 0.2);
}

input[type="number"]:focus {
  outline: none;
  border-color: rgba(255, 255, 255, 0.3);
  box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.05);
}

select:focus {
  outline: none;
  border-color: rgba(255, 255, 255, 0.3);
  box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.05);
}

/* Badge animations */
span[style*="sentimentBadge"] {
  transition: all 0.2s ease;
}

span[style*="sentimentBadge"]:hover {
  transform: scale(1.1);
  box-shadow: 0 2px 8px currentColor;
}

span[style*="toxicBadge"] {
  transition: all 0.2s ease;
}

span[style*="toxicBadge"]:hover {
  transform: scale(1.1);
  box-shadow: 0 2px 8px currentColor;
}

/* Legend item hover */
div[style*="legendItem"]:hover {
  background-color: rgba(44, 44, 46, 0.6);
  transform: translateX(4px);
}

/* Gauge hover */
div[style*="gaugeCircle"]:hover {
  transform: scale(1.05);
}
`
document.head.appendChild(styleSheet)

export default App