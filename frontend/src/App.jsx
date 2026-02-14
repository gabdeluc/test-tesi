import { useState, useEffect } from 'react'

const API_URL = 'http://localhost:8000'

// ============================================
// CORPORATE FORMAL WIDGET BOARD
// ============================================

function App() {
  const [meetingData, setMeetingData] = useState(null)
  const [participants, setParticipants] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Configurazione widget (con impostazioni avanzate)
  const [widgetConfigs, setWidgetConfigs] = useState({
    kpiMessages: { participant: null, showBorder: true },
    kpiSentiment: { participant: null, showBorder: true },
    kpiToxicity: { participant: null, showBorder: true },
    sentimentChart: { participant: null, showLabels: true, showLegend: true },
    toxicityGauge: { participant: null, showDetails: true },
    messageStream: { participant: null, limit: 10, showTimestamps: true, compact: false },
    alerts: { participant: null, threshold: 2.0, toxicityThreshold: 0.7 }
  })

  // Widget settings aperto
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

  const updateWidgetConfig = (widgetId, config) => {
    setWidgetConfigs(prev => ({
      ...prev,
      [widgetId]: { ...prev[widgetId], ...config }
    }))
  }

  const calculateStats = (filteredTranscript) => {
    if (!filteredTranscript || filteredTranscript.length === 0) {
      return {
        totalMessages: 0,
        avgSentiment: 0,
        positiveRatio: 0,
        toxicCount: 0,
        toxicRatio: 0,
        avgToxicity: 0
      }
    }

    const totalMessages = filteredTranscript.length
    let totalStars = 0
    let positiveCount = 0
    let toxicCount = 0
    let totalToxicity = 0

    filteredTranscript.forEach(entry => {
      totalStars += entry.sentiment.stars
      if (entry.sentiment.stars >= 3.5) positiveCount++
      if (entry.toxicity.is_toxic) toxicCount++
      totalToxicity += entry.toxicity.toxicity_score
    })

    return {
      totalMessages,
      avgSentiment: totalStars / totalMessages,
      positiveRatio: positiveCount / totalMessages,
      toxicCount,
      toxicRatio: toxicCount / totalMessages,
      avgToxicity: totalToxicity / totalMessages
    }
  }

  return (
    <div style={styles.appContainer}>
      {/* HEADER - Corporate Style */}
      <div style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.headerLeft}>
            <div style={styles.logoContainer}>
              <span style={styles.logoText}>MI</span>
            </div>
            <div>
              <h1 style={styles.title}>Meeting Intelligence Platform</h1>
              <span style={styles.subtitle}>Session MTG-001 • Real-time Analytics</span>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div style={styles.errorBanner}>
          <span style={styles.errorLabel}>Error:</span>
          <span>{error}</span>
        </div>
      )}

      {loading && (
        <div style={styles.loadingContainer}>
          <div style={styles.spinner}></div>
          <p style={styles.loadingText}>Loading analytics data...</p>
        </div>
      )}

      {!loading && meetingData && (
        <div style={styles.widgetGrid}>
          {/* KPI Cards */}
          <FormalKPIWidget
            widgetId="kpiMessages"
            title="Total Messages"
            config={widgetConfigs.kpiMessages}
            participants={participants}
            data={meetingData.transcript}
            onConfigChange={(config) => updateWidgetConfig('kpiMessages', config)}
            calculateValue={(data) => data.length}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          />

          <FormalKPIWidget
            widgetId="kpiSentiment"
            title="Average Sentiment"
            config={widgetConfigs.kpiSentiment}
            participants={participants}
            data={meetingData.transcript}
            onConfigChange={(config) => updateWidgetConfig('kpiSentiment', config)}
            calculateValue={(data) => {
              const stats = calculateStats(data)
              return stats.avgSentiment.toFixed(2)
            }}
            subtitle={(data) => {
              const stats = calculateStats(data)
              return `${(stats.positiveRatio * 100).toFixed(0)}% positive sentiment`
            }}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          />

          <FormalKPIWidget
            widgetId="kpiToxicity"
            title="Toxicity Incidents"
            config={widgetConfigs.kpiToxicity}
            participants={participants}
            data={meetingData.transcript}
            onConfigChange={(config) => updateWidgetConfig('kpiToxicity', config)}
            calculateValue={(data) => {
              const stats = calculateStats(data)
              return `${stats.toxicCount} of ${stats.totalMessages}`
            }}
            subtitle={(data) => {
              const stats = calculateStats(data)
              return `${(stats.toxicRatio * 100).toFixed(1)}% detection rate`
            }}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          />

          {/* Sentiment Chart Widget */}
          <FormalSentimentChart
            widgetId="sentimentChart"
            config={widgetConfigs.sentimentChart}
            participants={participants}
            data={meetingData.transcript}
            onConfigChange={(config) => updateWidgetConfig('sentimentChart', config)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          />

          {/* Toxicity Gauge Widget */}
          <FormalToxicityGauge
            widgetId="toxicityGauge"
            config={widgetConfigs.toxicityGauge}
            participants={participants}
            data={meetingData.transcript}
            calculateStats={calculateStats}
            onConfigChange={(config) => updateWidgetConfig('toxicityGauge', config)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          />

          {/* Message Stream Widget */}
          <FormalMessageStream
            widgetId="messageStream"
            config={widgetConfigs.messageStream}
            participants={participants}
            data={meetingData.transcript}
            onConfigChange={(config) => updateWidgetConfig('messageStream', config)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          />
        </div>
      )}
    </div>
  )
}

// ============================================
// FORMAL KPI WIDGET
// ============================================

function FormalKPIWidget({
  widgetId,
  title,
  config,
  participants,
  data,
  onConfigChange,
  calculateValue,
  subtitle,
  openSettings,
  setOpenSettings
}) {
  const filteredData = config.participant
    ? data.filter(entry => {
        const participant = participants.find(p => p.id === config.participant)
        return entry.nickname === participant?.name
      })
    : data

  const value = calculateValue(filteredData)
  const subtitleText = subtitle ? subtitle(filteredData) : null

  return (
    <div style={{
      ...styles.card,
      borderLeft: config.showBorder ? '3px solid #333333' : 'none'
    }}>
      <div style={styles.cardHeader}>
        <div style={styles.cardTitle}>
          <span style={styles.cardTitleText}>{title}</span>
        </div>
        <button
          onClick={() => setOpenSettings(openSettings === widgetId ? null : widgetId)}
          style={styles.settingsButton}
          title="Settings"
        >
          ⋮
        </button>
      </div>

      {openSettings === widgetId && (
        <FormalSettings
          config={config}
          participants={participants}
          onConfigChange={onConfigChange}
          options={[
            { type: 'select', key: 'participant', label: 'Filter by Participant' },
            { type: 'toggle', key: 'showBorder', label: 'Show Border Accent' }
          ]}
        />
      )}

      <div style={styles.kpiContent}>
        <div style={styles.kpiValueContainer}>
          <span style={styles.kpiValue}>{value}</span>
        </div>
        {subtitleText && (
          <div style={styles.kpiSubtitle}>{subtitleText}</div>
        )}
      </div>
    </div>
  )
}

// ============================================
// FORMAL SENTIMENT CHART
// ============================================

function FormalSentimentChart({
  widgetId,
  config,
  participants,
  data,
  onConfigChange,
  openSettings,
  setOpenSettings
}) {
  const filteredData = config.participant
    ? data.filter(entry => {
        const participant = participants.find(p => p.id === config.participant)
        return entry.nickname === participant?.name
      })
    : data

  const counts = {
    very_positive: 0,
    positive: 0,
    neutral: 0,
    negative: 0,
    very_negative: 0
  }

  filteredData.forEach(entry => {
    const sentiment = entry.sentiment.sentiment
    if (counts[sentiment] !== undefined) counts[sentiment]++
  })

  const total = filteredData.length
  const percentages = Object.entries(counts).map(([key, count]) => ({
    key,
    count,
    percentage: total > 0 ? (count / total) * 100 : 0
  }))

  // Scala di grigi (dal chiaro al scuro per rappresentare neg to pos)
  const grayScaleMap = {
    very_negative: '#1a1a1a',
    negative: '#4a4a4a',
    neutral: '#808080',
    positive: '#b0b0b0',
    very_positive: '#d0d0d0'
  }

  const labelMap = {
    very_positive: 'Very Positive',
    positive: 'Positive',
    neutral: 'Neutral',
    negative: 'Negative',
    very_negative: 'Very Negative'
  }

  return (
    <div style={{ ...styles.card, ...styles.wideCard }}>
      <div style={styles.cardHeader}>
        <div style={styles.cardTitle}>
          <span style={styles.cardTitleText}>Sentiment Distribution Analysis</span>
        </div>
        <button
          onClick={() => setOpenSettings(openSettings === widgetId ? null : widgetId)}
          style={styles.settingsButton}
          title="Settings"
        >
          ⋮
        </button>
      </div>

      {openSettings === widgetId && (
        <FormalSettings
          config={config}
          participants={participants}
          onConfigChange={onConfigChange}
          options={[
            { type: 'select', key: 'participant', label: 'Filter by Participant' },
            { type: 'toggle', key: 'showLabels', label: 'Display Labels' },
            { type: 'toggle', key: 'showLegend', label: 'Display Legend' }
          ]}
        />
      )}

      <div style={styles.chartContent}>
        {config.showLabels && (
          <div style={styles.chartLabels}>
            <span style={styles.chartLabelText}>Negative</span>
            <span style={styles.chartLabelText}>Neutral</span>
            <span style={styles.chartLabelText}>Positive</span>
          </div>
        )}

        <div style={styles.barContainer}>
          {percentages.map(({ key, percentage }) =>
            percentage > 0 ? (
              <div
                key={key}
                style={{
                  width: `${percentage}%`,
                  height: '100%',
                  backgroundColor: grayScaleMap[key],
                  position: 'relative'
                }}
                title={`${labelMap[key]}: ${percentage.toFixed(1)}%`}
              />
            ) : null
          )}
        </div>

        {config.showLegend && (
          <div style={styles.legend}>
            {percentages.map(({ key, count, percentage }) => (
              <div key={key} style={styles.legendRow}>
                <div
                  style={{
                    ...styles.legendSquare,
                    backgroundColor: grayScaleMap[key]
                  }}
                />
                <span style={styles.legendText}>
                  {labelMap[key]}: {count} ({percentage.toFixed(1)}%)
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================
// FORMAL TOXICITY GAUGE
// ============================================

function FormalToxicityGauge({
  widgetId,
  config,
  participants,
  data,
  calculateStats,
  onConfigChange,
  openSettings,
  setOpenSettings
}) {
  const filteredData = config.participant
    ? data.filter(entry => {
        const participant = participants.find(p => p.id === config.participant)
        return entry.nickname === participant?.name
      })
    : data

  const stats = calculateStats(filteredData)
  const percentage = stats.avgToxicity * 100

  const getRiskLevel = () => {
    if (stats.avgToxicity < 0.2) return 'LOW'
    if (stats.avgToxicity < 0.5) return 'MEDIUM'
    return 'HIGH'
  }

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <div style={styles.cardTitle}>
          <span style={styles.cardTitleText}>Toxicity Analysis</span>
        </div>
        <button
          onClick={() => setOpenSettings(openSettings === widgetId ? null : widgetId)}
          style={styles.settingsButton}
          title="Settings"
        >
          ⋮
        </button>
      </div>

      {openSettings === widgetId && (
        <FormalSettings
          config={config}
          participants={participants}
          onConfigChange={onConfigChange}
          options={[
            { type: 'select', key: 'participant', label: 'Filter by Participant' },
            { type: 'toggle', key: 'showDetails', label: 'Show Detailed Metrics' }
          ]}
        />
      )}

      {filteredData.length > 0 ? (
        <div style={styles.gaugeContent}>
          <div style={styles.gaugeContainer}>
            <svg width="180" height="180" viewBox="0 0 180 180">
              {/* Background circle */}
              <circle
                cx="90"
                cy="90"
                r="70"
                fill="none"
                stroke="#e5e5e5"
                strokeWidth="12"
              />
              {/* Progress circle */}
              <circle
                cx="90"
                cy="90"
                r="70"
                fill="none"
                stroke="#333333"
                strokeWidth="12"
                strokeDasharray={`${(percentage / 100) * 439.8} 439.8`}
                strokeDashoffset="0"
                transform="rotate(-90 90 90)"
                strokeLinecap="round"
              />
              {/* Center text */}
              <text
                x="90"
                y="85"
                textAnchor="middle"
                style={{
                  fontSize: '32px',
                  fontWeight: '600',
                  fill: '#1a1a1a',
                  fontFamily: 'monospace'
                }}
              >
                {percentage.toFixed(0)}%
              </text>
              <text
                x="90"
                y="105"
                textAnchor="middle"
                style={{
                  fontSize: '12px',
                  fontWeight: '600',
                  fill: '#666666',
                  letterSpacing: '1px'
                }}
              >
                {getRiskLevel()}
              </text>
            </svg>
          </div>

          {config.showDetails && (
            <div style={styles.detailsGrid}>
              <div style={styles.detailItem}>
                <span style={styles.detailLabel}>Average Score</span>
                <span style={styles.detailValue}>{stats.avgToxicity.toFixed(3)}</span>
              </div>
              <div style={styles.detailItem}>
                <span style={styles.detailLabel}>Toxic Messages</span>
                <span style={styles.detailValue}>
                  {stats.toxicCount} / {stats.totalMessages}
                </span>
              </div>
              <div style={styles.detailItem}>
                <span style={styles.detailLabel}>Detection Rate</span>
                <span style={styles.detailValue}>
                  {(stats.toxicRatio * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div style={styles.emptyState}>No data available for selected filter</div>
      )}
    </div>
  )
}

// ============================================
// FORMAL MESSAGE STREAM
// ============================================

function FormalMessageStream({
  widgetId,
  config,
  participants,
  data,
  onConfigChange,
  openSettings,
  setOpenSettings
}) {
  const filteredData = config.participant
    ? data.filter(entry => {
        const participant = participants.find(p => p.id === config.participant)
        return entry.nickname === participant?.name
      })
    : data

  const displayMessages = filteredData.slice(0, config.limit || 10)

  return (
    <div style={{ ...styles.card, ...styles.wideCard }}>
      <div style={styles.cardHeader}>
        <div style={styles.cardTitle}>
          <span style={styles.cardTitleText}>Message Stream</span>
          <span style={styles.cardBadge}>{displayMessages.length} messages</span>
        </div>
        <button
          onClick={() => setOpenSettings(openSettings === widgetId ? null : widgetId)}
          style={styles.settingsButton}
          title="Settings"
        >
          ⋮
        </button>
      </div>

      {openSettings === widgetId && (
        <FormalSettings
          config={config}
          participants={participants}
          onConfigChange={onConfigChange}
          options={[
            { type: 'select', key: 'participant', label: 'Filter by Participant' },
            { type: 'slider', key: 'limit', label: 'Message Limit', min: 5, max: 20 },
            { type: 'toggle', key: 'showTimestamps', label: 'Show Timestamps' },
            { type: 'toggle', key: 'compact', label: 'Compact View' }
          ]}
        />
      )}

      <div style={styles.messageList}>
        {displayMessages.map((msg, idx) => (
          <FormalMessageItem
            key={idx}
            message={msg}
            showTimestamp={config.showTimestamps}
            compact={config.compact}
          />
        ))}
      </div>
    </div>
  )
}

function FormalMessageItem({ message, showTimestamp, compact }) {
  // Calcola intensità grigio basato su sentiment (1=scuro, 5=chiaro)
  const sentimentGray = Math.round((message.sentiment.stars / 5) * 255)
  const sentimentColor = `rgb(${sentimentGray}, ${sentimentGray}, ${sentimentGray})`

  return (
    <div
      style={{
        ...styles.messageItem,
        ...(compact ? styles.messageItemCompact : {}),
        borderLeft: `3px solid ${sentimentColor}`
      }}
    >
      <div style={styles.messageHeader}>
        <span style={styles.messageAuthor}>{message.nickname}</span>
        <div style={styles.messageMetrics}>
          <span style={styles.metricBadge}>
            Sentiment: {message.sentiment.stars.toFixed(1)}
          </span>
          {message.toxicity.is_toxic && (
            <span style={{ ...styles.metricBadge, ...styles.metricBadgeToxic }}>
              Toxic
            </span>
          )}
        </div>
      </div>
      <p style={styles.messageText}>{message.text}</p>
      {showTimestamp && (
        <span style={styles.messageTime}>{message.from}</span>
      )}
    </div>
  )
}

// ============================================
// FORMAL SETTINGS PANEL
// ============================================

function FormalSettings({ config, participants, onConfigChange, options }) {
  return (
    <div style={styles.settingsPanel}>
      <div style={styles.settingsPanelHeader}>
        <span style={styles.settingsPanelTitle}>Widget Configuration</span>
      </div>
      {options.map((option, idx) => (
        <div key={idx} style={styles.settingRow}>
          <span style={styles.settingLabel}>{option.label}</span>

          {option.type === 'select' && (
            <select
              value={config[option.key] || ''}
              onChange={(e) =>
                onConfigChange({ [option.key]: e.target.value || null })
              }
              style={styles.settingSelect}
            >
              <option value="">All Participants</option>
              {participants.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          )}

          {option.type === 'toggle' && (
            <label style={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={config[option.key] || false}
                onChange={(e) =>
                  onConfigChange({ [option.key]: e.target.checked })
                }
                style={styles.checkbox}
              />
              <span style={styles.checkboxText}>
                {config[option.key] ? 'Enabled' : 'Disabled'}
              </span>
            </label>
          )}

          {option.type === 'slider' && (
            <div style={styles.sliderContainer}>
              <input
                type="range"
                min={option.min}
                max={option.max}
                value={config[option.key] || option.min}
                onChange={(e) =>
                  onConfigChange({ [option.key]: parseInt(e.target.value) })
                }
                style={styles.slider}
              />
              <span style={styles.sliderValue}>
                {config[option.key] || option.min}
              </span>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

// ============================================
// STYLES - FORMAL CORPORATE THEME
// ============================================

const styles = {
  appContainer: {
    minHeight: '100vh',
    backgroundColor: '#fafafa',
    fontFamily: '"Inter", "Helvetica Neue", Arial, sans-serif',
    color: '#1a1a1a'
  },

  // HEADER
  header: {
    backgroundColor: '#ffffff',
    borderBottom: '1px solid #e0e0e0',
    position: 'sticky',
    top: 0,
    zIndex: 100,
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)'
  },
  headerContent: {
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '1.25rem 2rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem'
  },
  logoContainer: {
    width: '48px',
    height: '48px',
    backgroundColor: '#1a1a1a',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    border: '2px solid #333333'
  },
  logoText: {
    color: '#ffffff',
    fontSize: '1.2rem',
    fontWeight: '700',
    letterSpacing: '1px',
    fontFamily: 'monospace'
  },
  title: {
    margin: 0,
    fontSize: '1.25rem',
    fontWeight: '600',
    color: '#1a1a1a',
    letterSpacing: '-0.01em'
  },
  subtitle: {
    fontSize: '0.8rem',
    color: '#666666',
    fontWeight: '400',
    letterSpacing: '0.02em'
  },

  // ERROR & LOADING
  errorBanner: {
    padding: '1rem 2rem',
    margin: '1rem 2rem',
    backgroundColor: '#f5f5f5',
    border: '1px solid #cccccc',
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    fontSize: '0.9rem',
    color: '#333333'
  },
  errorLabel: {
    fontWeight: '600',
    textTransform: 'uppercase',
    fontSize: '0.75rem',
    letterSpacing: '0.5px'
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
    width: '40px',
    height: '40px',
    border: '3px solid #e0e0e0',
    borderTop: '3px solid #333333',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite'
  },
  loadingText: {
    fontSize: '0.9rem',
    color: '#666666',
    fontWeight: '500',
    letterSpacing: '0.02em'
  },

  // WIDGET GRID
  widgetGrid: {
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '2rem',
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
    gap: '1.5rem'
  },

  // CARD
  card: {
    backgroundColor: '#ffffff',
    border: '1px solid #e0e0e0',
    padding: '1.5rem',
    transition: 'box-shadow 0.2s ease',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)'
  },
  wideCard: {
    gridColumn: 'span 2'
  },

  // CARD HEADER
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1.25rem',
    paddingBottom: '0.75rem',
    borderBottom: '1px solid #f0f0f0'
  },
  cardTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    flex: 1
  },
  cardTitleText: {
    fontSize: '0.95rem',
    fontWeight: '600',
    color: '#1a1a1a',
    letterSpacing: '0.01em',
    textTransform: 'uppercase',
    fontSize: '0.8rem'
  },
  cardBadge: {
    fontSize: '0.7rem',
    color: '#666666',
    backgroundColor: '#f5f5f5',
    padding: '0.25rem 0.6rem',
    fontWeight: '500',
    letterSpacing: '0.02em'
  },
  settingsButton: {
    width: '32px',
    height: '32px',
    backgroundColor: 'transparent',
    border: '1px solid #e0e0e0',
    fontSize: '1.2rem',
    color: '#666666',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    fontWeight: '700'
  },

  // KPI CONTENT
  kpiContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    padding: '1.5rem 0'
  },
  kpiValueContainer: {
    textAlign: 'center',
    padding: '2rem',
    backgroundColor: '#fafafa',
    border: '2px solid #e0e0e0'
  },
  kpiValue: {
    fontSize: '3rem',
    fontWeight: '300',
    color: '#1a1a1a',
    fontFamily: 'monospace',
    letterSpacing: '-0.02em'
  },
  kpiSubtitle: {
    fontSize: '0.85rem',
    color: '#666666',
    textAlign: 'center',
    fontWeight: '500',
    letterSpacing: '0.02em'
  },

  // CHART
  chartContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem'
  },
  chartLabels: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.75rem',
    color: '#666666',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    fontWeight: '600'
  },
  chartLabelText: {
    fontSize: '0.7rem'
  },
  barContainer: {
    display: 'flex',
    height: '32px',
    border: '1px solid #e0e0e0',
    overflow: 'hidden'
  },
  legend: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '0.75rem',
    marginTop: '0.5rem'
  },
  legendRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem'
  },
  legendSquare: {
    width: '14px',
    height: '14px',
    border: '1px solid #cccccc'
  },
  legendText: {
    fontSize: '0.8rem',
    color: '#666666',
    fontWeight: '500'
  },

  // GAUGE
  gaugeContent: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '2rem',
    padding: '1rem 0'
  },
  gaugeContainer: {
    display: 'flex',
    justifyContent: 'center'
  },
  detailsGrid: {
    width: '100%',
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '1rem'
  },
  detailItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
    padding: '1rem',
    backgroundColor: '#fafafa',
    border: '1px solid #e0e0e0',
    textAlign: 'center'
  },
  detailLabel: {
    fontSize: '0.7rem',
    color: '#666666',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    fontWeight: '600'
  },
  detailValue: {
    fontSize: '1.1rem',
    color: '#1a1a1a',
    fontWeight: '600',
    fontFamily: 'monospace'
  },

  // MESSAGES
  messageList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
    maxHeight: '500px',
    overflowY: 'auto'
  },
  messageItem: {
    padding: '1rem',
    backgroundColor: '#fafafa',
    border: '1px solid #e0e0e0',
    borderLeft: '3px solid #666666',
    transition: 'background-color 0.2s ease'
  },
  messageItemCompact: {
    padding: '0.75rem'
  },
  messageHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.5rem'
  },
  messageAuthor: {
    fontSize: '0.85rem',
    fontWeight: '600',
    color: '#1a1a1a',
    textTransform: 'uppercase',
    letterSpacing: '0.5px'
  },
  messageMetrics: {
    display: 'flex',
    gap: '0.5rem'
  },
  metricBadge: {
    fontSize: '0.7rem',
    color: '#666666',
    backgroundColor: '#ffffff',
    padding: '0.25rem 0.6rem',
    border: '1px solid #e0e0e0',
    fontWeight: '600',
    letterSpacing: '0.02em'
  },
  metricBadgeToxic: {
    color: '#1a1a1a',
    borderColor: '#333333',
    fontWeight: '700'
  },
  messageText: {
    margin: 0,
    fontSize: '0.9rem',
    lineHeight: '1.6',
    color: '#333333'
  },
  messageTime: {
    display: 'block',
    marginTop: '0.5rem',
    fontSize: '0.75rem',
    color: '#999999',
    fontFamily: 'monospace'
  },

  // SETTINGS PANEL
  settingsPanel: {
    marginTop: '1rem',
    padding: '1rem',
    backgroundColor: '#f5f5f5',
    border: '1px solid #e0e0e0',
    animation: 'slideDown 0.3s ease'
  },
  settingsPanelHeader: {
    marginBottom: '1rem',
    paddingBottom: '0.75rem',
    borderBottom: '1px solid #cccccc'
  },
  settingsPanelTitle: {
    fontSize: '0.75rem',
    fontWeight: '700',
    color: '#1a1a1a',
    textTransform: 'uppercase',
    letterSpacing: '1px'
  },
  settingRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '0.75rem 0',
    borderBottom: '1px solid #e0e0e0'
  },
  settingLabel: {
    fontSize: '0.85rem',
    color: '#333333',
    fontWeight: '500'
  },

  // FORM CONTROLS
  settingSelect: {
    padding: '0.5rem 0.75rem',
    fontSize: '0.85rem',
    backgroundColor: '#ffffff',
    border: '1px solid #cccccc',
    color: '#1a1a1a',
    fontWeight: '500',
    cursor: 'pointer',
    minWidth: '150px'
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    cursor: 'pointer'
  },
  checkbox: {
    width: '18px',
    height: '18px',
    cursor: 'pointer',
    accentColor: '#333333'
  },
  checkboxText: {
    fontSize: '0.8rem',
    color: '#666666',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.5px'
  },
  sliderContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem'
  },
  slider: {
    width: '120px',
    height: '2px',
    appearance: 'none',
    WebkitAppearance: 'none',
    background: '#cccccc',
    outline: 'none',
    cursor: 'pointer'
  },
  sliderValue: {
    fontSize: '0.85rem',
    fontWeight: '700',
    color: '#1a1a1a',
    minWidth: '30px',
    textAlign: 'right',
    fontFamily: 'monospace'
  },

  // EMPTY STATE
  emptyState: {
    textAlign: 'center',
    padding: '3rem 2rem',
    color: '#999999',
    fontSize: '0.9rem',
    fontWeight: '500',
    fontStyle: 'italic',
    backgroundColor: '#fafafa',
    border: '1px dashed #cccccc'
  }
}

// CSS Animation
const globalStyles = `
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

.card:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.settingsButton:hover {
  background-color: #f5f5f5;
  border-color: #cccccc;
}

.messageItem:hover {
  background-color: #f5f5f5;
}
`

export default App