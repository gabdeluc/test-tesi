import { useState, useEffect } from 'react'

const API_URL = 'http://localhost:8000'

// ============================================
// APPLE GRAYSCALE WIDGET BOARD
// Eleganza Apple + Scala di Grigi
// ============================================

function App() {
  const [meetingData, setMeetingData] = useState(null)
  const [participants, setParticipants] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Configurazione widget
  const [widgetConfigs, setWidgetConfigs] = useState({
    kpiMessages: { participant: null, style: 'minimal' },
    kpiSentiment: { participant: null, style: 'minimal' },
    kpiToxicity: { participant: null, style: 'minimal' },
    sentimentChart: { participant: null, showLabels: true, animated: true },
    toxicityGauge: { participant: null, showDetails: true },
    messageStream: { participant: null, limit: 10, showTimestamps: true },
    alerts: { participant: null, threshold: 2.0, toxicityThreshold: 0.7 }
  })

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
      {/* HEADER - Apple Style Glassmorphism */}
      <div style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.headerLeft}>
            <div style={styles.logoContainer}>
              <span style={styles.logoText}>MI</span>
            </div>
            <div>
              <h1 style={styles.title}>Meeting Intelligence</h1>
              <span style={styles.subtitle}>Session MTG-001 · Real-time Analytics</span>
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
          <div style={styles.appleSpinner}></div>
          <p style={styles.loadingText}>Loading analytics...</p>
        </div>
      )}

      {!loading && meetingData && (
        <div style={styles.widgetGrid}>
          {/* KPI Cards */}
          <AppleKPIWidget
            widgetId="kpiMessages"
            title="Messages"
            config={widgetConfigs.kpiMessages}
            participants={participants}
            data={meetingData.transcript}
            onConfigChange={(config) => updateWidgetConfig('kpiMessages', config)}
            calculateValue={(data) => data.length}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          />

          <AppleKPIWidget
            widgetId="kpiSentiment"
            title="Sentiment"
            config={widgetConfigs.kpiSentiment}
            participants={participants}
            data={meetingData.transcript}
            onConfigChange={(config) => updateWidgetConfig('kpiSentiment', config)}
            calculateValue={(data) => {
              const stats = calculateStats(data)
              return stats.avgSentiment.toFixed(1)
            }}
            subtitle={(data) => {
              const stats = calculateStats(data)
              return `${(stats.positiveRatio * 100).toFixed(0)}% positive`
            }}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          />

          <AppleKPIWidget
            widgetId="kpiToxicity"
            title="Toxicity"
            config={widgetConfigs.kpiToxicity}
            participants={participants}
            data={meetingData.transcript}
            onConfigChange={(config) => updateWidgetConfig('kpiToxicity', config)}
            calculateValue={(data) => {
              const stats = calculateStats(data)
              return `${stats.toxicCount}`
            }}
            subtitle={(data) => {
              const stats = calculateStats(data)
              return `${(stats.toxicRatio * 100).toFixed(0)}% detection rate`
            }}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          />

          {/* Sentiment Chart Widget */}
          <AppleSentimentChart
            widgetId="sentimentChart"
            config={widgetConfigs.sentimentChart}
            participants={participants}
            data={meetingData.transcript}
            onConfigChange={(config) => updateWidgetConfig('sentimentChart', config)}
            openSettings={openSettings}
            setOpenSettings={setOpenSettings}
          />

          {/* Toxicity Gauge Widget */}
          <AppleToxicityGauge
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
          <AppleMessageStream
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
// APPLE KPI WIDGET
// ============================================

function AppleKPIWidget({
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
    <div style={styles.appleCard}>
      <div style={styles.cardHeader}>
        <div style={styles.cardTitle}>
          <span style={styles.cardTitleText}>{title}</span>
        </div>
        <button
          onClick={() => setOpenSettings(openSettings === widgetId ? null : widgetId)}
          style={styles.settingsButton}
        >
          ···
        </button>
      </div>

      {openSettings === widgetId && (
        <AppleSettings
          config={config}
          participants={participants}
          onConfigChange={onConfigChange}
          options={[
            { type: 'select', key: 'participant', label: 'Filter' },
            { type: 'select', key: 'style', label: 'Style', selectOptions: [
              { value: 'minimal', label: 'Minimal' },
              { value: 'detailed', label: 'Detailed' }
            ]}
          ]}
        />
      )}

      <div style={styles.kpiContent}>
        <div style={styles.kpiCircle}>
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
// APPLE SENTIMENT CHART
// ============================================

function AppleSentimentChart({
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

  // Scala di grigi Apple-style (elegante)
  const grayScaleMap = {
    very_negative: '#2c2c2e',
    negative: '#48484a',
    neutral: '#8e8e93',
    positive: '#aeaeb2',
    very_positive: '#c7c7cc'
  }

  const labelMap = {
    very_positive: 'Very Positive',
    positive: 'Positive',
    neutral: 'Neutral',
    negative: 'Negative',
    very_negative: 'Very Negative'
  }

  return (
    <div style={{ ...styles.appleCard, ...styles.wideCard }}>
      <div style={styles.cardHeader}>
        <div style={styles.cardTitle}>
          <span style={styles.cardTitleText}>Sentiment Distribution</span>
        </div>
        <button
          onClick={() => setOpenSettings(openSettings === widgetId ? null : widgetId)}
          style={styles.settingsButton}
        >
          ···
        </button>
      </div>

      {openSettings === widgetId && (
        <AppleSettings
          config={config}
          participants={participants}
          onConfigChange={onConfigChange}
          options={[
            { type: 'select', key: 'participant', label: 'Filter' },
            { type: 'toggle', key: 'showLabels', label: 'Show Labels' },
            { type: 'toggle', key: 'animated', label: 'Animated' }
          ]}
        />
      )}

      <div style={styles.chartContent}>
        <div style={styles.appleBar}>
          {percentages.map(({ key, percentage }) =>
            percentage > 0 ? (
              <div
                key={key}
                style={{
                  width: `${percentage}%`,
                  height: '100%',
                  backgroundColor: grayScaleMap[key],
                  transition: config.animated ? 'all 0.5s ease' : 'none'
                }}
              />
            ) : null
          )}
        </div>

        {config.showLabels && (
          <div style={styles.appleLegend}>
            {percentages.map(({ key, count, percentage }) => (
              <div key={key} style={styles.legendRow}>
                <div
                  style={{
                    ...styles.legendDot,
                    backgroundColor: grayScaleMap[key]
                  }}
                />
                <span style={styles.legendText}>
                  {labelMap[key]}: {count} ({percentage.toFixed(0)}%)
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
// APPLE TOXICITY GAUGE
// ============================================

function AppleToxicityGauge({
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

  // Gradiente di grigio in base al rischio
  const getGradient = () => {
    if (stats.avgToxicity < 0.2) return 'linear-gradient(135deg, #c7c7cc 0%, #d1d1d6 100%)'
    if (stats.avgToxicity < 0.5) return 'linear-gradient(135deg, #8e8e93 0%, #aeaeb2 100%)'
    return 'linear-gradient(135deg, #3a3a3c 0%, #48484a 100%)'
  }

  return (
    <div style={styles.appleCard}>
      <div style={styles.cardHeader}>
        <div style={styles.cardTitle}>
          <span style={styles.cardTitleText}>Toxicity Level</span>
        </div>
        <button
          onClick={() => setOpenSettings(openSettings === widgetId ? null : widgetId)}
          style={styles.settingsButton}
        >
          ···
        </button>
      </div>

      {openSettings === widgetId && (
        <AppleSettings
          config={config}
          participants={participants}
          onConfigChange={onConfigChange}
          options={[
            { type: 'select', key: 'participant', label: 'Filter' },
            { type: 'toggle', key: 'showDetails', label: 'Show Details' }
          ]}
        />
      )}

      {filteredData.length > 0 ? (
        <div style={styles.gaugeContent}>
          <div
            style={{
              ...styles.appleGauge,
              background: getGradient()
            }}
          >
            <div style={styles.gaugeInner}>
              <span style={styles.gaugeValue}>{percentage.toFixed(0)}%</span>
              <span style={styles.gaugeLabel}>{getRiskLevel()}</span>
            </div>
          </div>

          {config.showDetails && (
            <div style={styles.gaugeDetails}>
              <div style={styles.detailRow}>
                <span style={styles.detailLabel}>Average</span>
                <span style={styles.detailValue}>{stats.avgToxicity.toFixed(3)}</span>
              </div>
              <div style={styles.detailRow}>
                <span style={styles.detailLabel}>Toxic Count</span>
                <span style={styles.detailValue}>
                  {stats.toxicCount} / {stats.totalMessages}
                </span>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div style={styles.emptyState}>No data available</div>
      )}
    </div>
  )
}

// ============================================
// APPLE MESSAGE STREAM
// ============================================

function AppleMessageStream({
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
    <div style={{ ...styles.appleCard, ...styles.wideCard }}>
      <div style={styles.cardHeader}>
        <div style={styles.cardTitle}>
          <span style={styles.cardTitleText}>Message Stream</span>
        </div>
        <button
          onClick={() => setOpenSettings(openSettings === widgetId ? null : widgetId)}
          style={styles.settingsButton}
        >
          ···
        </button>
      </div>

      {openSettings === widgetId && (
        <AppleSettings
          config={config}
          participants={participants}
          onConfigChange={onConfigChange}
          options={[
            { type: 'select', key: 'participant', label: 'Filter' },
            { type: 'slider', key: 'limit', label: 'Message Limit', min: 5, max: 20 },
            { type: 'toggle', key: 'showTimestamps', label: 'Show Timestamps' }
          ]}
        />
      )}

      <div style={styles.messageList}>
        {displayMessages.map((msg, idx) => (
          <AppleMessageBubble
            key={idx}
            message={msg}
            showTimestamp={config.showTimestamps}
          />
        ))}
      </div>
    </div>
  )
}

function AppleMessageBubble({ message, showTimestamp }) {
  // Scala di grigi in base al sentiment
  const sentimentGray = Math.round((message.sentiment.stars / 5) * 150) + 100
  const sentimentColor = `rgb(${sentimentGray}, ${sentimentGray}, ${sentimentGray})`

  return (
    <div style={styles.appleBubble}>
      <div style={styles.bubbleHeader}>
        <span style={styles.bubbleAuthor}>{message.nickname}</span>
        <div style={styles.bubbleScores}>
          <span
            style={{
              ...styles.bubbleBadge,
              backgroundColor: sentimentColor
            }}
          >
            {message.sentiment.stars.toFixed(1)}
          </span>
          {message.toxicity.is_toxic && (
            <span style={styles.bubbleBadgeToxic}>Toxic</span>
          )}
        </div>
      </div>
      <p style={styles.bubbleText}>{message.text}</p>
      {showTimestamp && (
        <span style={styles.bubbleTime}>{message.from}</span>
      )}
    </div>
  )
}

// ============================================
// APPLE SETTINGS PANEL
// ============================================

function AppleSettings({ config, participants, onConfigChange, options }) {
  return (
    <div style={styles.settingsPanel}>
      {options.map((option, idx) => (
        <div key={idx} style={styles.settingRow}>
          <span style={styles.settingLabel}>{option.label}</span>

          {option.type === 'select' && !option.selectOptions && (
            <select
              value={config[option.key] || ''}
              onChange={(e) =>
                onConfigChange({ [option.key]: e.target.value || null })
              }
              style={styles.appleSelect}
            >
              <option value="">All</option>
              {participants.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          )}

          {option.type === 'select' && option.selectOptions && (
            <select
              value={config[option.key] || option.selectOptions[0].value}
              onChange={(e) =>
                onConfigChange({ [option.key]: e.target.value })
              }
              style={styles.appleSelect}
            >
              {option.selectOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          )}

          {option.type === 'toggle' && (
            <label style={styles.toggleContainer}>
              <input
                type="checkbox"
                checked={config[option.key] || false}
                onChange={(e) =>
                  onConfigChange({ [option.key]: e.target.checked })
                }
                style={styles.toggleInput}
              />
              <span
                style={{
                  ...styles.toggleSlider,
                  backgroundColor: config[option.key] ? '#3a3a3c' : '#d1d1d6'
                }}
              >
                <span
                  style={{
                    ...styles.toggleThumb,
                    transform: config[option.key]
                      ? 'translateX(20px)'
                      : 'translateX(2px)'
                  }}
                />
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
                style={styles.appleSlider}
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
// STYLES - APPLE GRAYSCALE
// ============================================

const styles = {
  appContainer: {
    minHeight: '100vh',
    backgroundColor: '#f5f5f7',
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif',
    transition: 'background-color 0.3s ease'
  },

  // HEADER - Apple Glassmorphism
  header: {
    position: 'sticky',
    top: 0,
    zIndex: 100,
    backgroundColor: 'rgba(255, 255, 255, 0.72)',
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    borderBottom: '0.5px solid rgba(0, 0, 0, 0.1)',
    transition: 'all 0.3s ease'
  },
  headerContent: {
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '1rem 2rem',
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
    borderRadius: '12px',
    background: 'linear-gradient(135deg, #2c2c2e 0%, #48484a 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)'
  },
  logoText: {
    color: '#ffffff',
    fontSize: '1.2rem',
    fontWeight: '700',
    letterSpacing: '1px'
  },
  title: {
    margin: 0,
    fontSize: '1.25rem',
    fontWeight: '600',
    color: '#1d1d1f',
    letterSpacing: '-0.02em'
  },
  subtitle: {
    fontSize: '0.8rem',
    color: '#6e6e73',
    fontWeight: '400'
  },

  // ERROR & LOADING
  errorBanner: {
    padding: '1rem 2rem',
    margin: '1rem 2rem',
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    borderRadius: '12px',
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    fontSize: '0.9rem',
    color: '#1d1d1f'
  },
  errorIcon: {
    width: '24px',
    height: '24px',
    borderRadius: '12px',
    backgroundColor: '#2c2c2e',
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
  appleSpinner: {
    width: '50px',
    height: '50px',
    border: '4px solid #d1d1d6',
    borderTop: '4px solid #2c2c2e',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite'
  },
  loadingText: {
    fontSize: '0.95rem',
    fontWeight: '500',
    color: '#6e6e73'
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

  // APPLE CARD - Glassmorphism
  appleCard: {
    borderRadius: '20px',
    padding: '1.5rem',
    backgroundColor: 'rgba(255, 255, 255, 0.72)',
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    border: '0.5px solid rgba(0, 0, 0, 0.05)',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
    transition: 'transform 0.3s ease, box-shadow 0.3s ease'
  },
  wideCard: {
    gridColumn: 'span 2'
  },

  // CARD HEADER
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1.25rem'
  },
  cardTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem'
  },
  cardTitleText: {
    fontSize: '1.1rem',
    fontWeight: '600',
    color: '#1d1d1f',
    letterSpacing: '-0.01em'
  },
  settingsButton: {
    width: '32px',
    height: '32px',
    borderRadius: '8px',
    border: 'none',
    fontSize: '1.2rem',
    cursor: 'pointer',
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    color: '#6e6e73',
    fontWeight: '700',
    letterSpacing: '1px',
    transition: 'transform 0.2s ease, background-color 0.2s ease'
  },

  // KPI CONTENT
  kpiContent: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '1rem',
    padding: '1rem 0'
  },
  kpiCircle: {
    width: '120px',
    height: '120px',
    borderRadius: '60px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(135deg, #2c2c2e 0%, #48484a 100%)',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15)'
  },
  kpiValue: {
    fontSize: '2.5rem',
    fontWeight: '700',
    color: 'white',
    textShadow: '0 2px 4px rgba(0, 0, 0, 0.2)'
  },
  kpiSubtitle: {
    fontSize: '0.9rem',
    fontWeight: '500',
    color: '#6e6e73',
    textAlign: 'center'
  },

  // CHART
  chartContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem'
  },
  appleBar: {
    display: 'flex',
    height: '24px',
    borderRadius: '12px',
    overflow: 'hidden',
    boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.1)'
  },
  appleLegend: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '0.75rem'
  },
  legendRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem'
  },
  legendDot: {
    width: '12px',
    height: '12px',
    borderRadius: '6px'
  },
  legendText: {
    fontSize: '0.85rem',
    fontWeight: '500',
    color: '#6e6e73'
  },

  // GAUGE
  gaugeContent: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '1.5rem',
    padding: '1rem 0'
  },
  appleGauge: {
    width: '160px',
    height: '160px',
    borderRadius: '80px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15)'
  },
  gaugeInner: {
    width: '120px',
    height: '120px',
    borderRadius: '60px',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    backdropFilter: 'blur(20px)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.25rem'
  },
  gaugeValue: {
    fontSize: '2rem',
    fontWeight: '700',
    color: '#1d1d1f'
  },
  gaugeLabel: {
    fontSize: '0.7rem',
    fontWeight: '600',
    color: '#6e6e73',
    letterSpacing: '0.5px'
  },
  gaugeDetails: {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem'
  },
  detailRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '0.75rem',
    borderRadius: '10px',
    backgroundColor: 'rgba(0, 0, 0, 0.03)'
  },
  detailLabel: {
    fontSize: '0.85rem',
    fontWeight: '500',
    color: '#6e6e73'
  },
  detailValue: {
    fontSize: '0.95rem',
    fontWeight: '600',
    color: '#1d1d1f'
  },

  // MESSAGES
  messageList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
    maxHeight: '500px',
    overflowY: 'auto'
  },
  appleBubble: {
    padding: '1rem',
    borderRadius: '16px',
    backgroundColor: 'rgba(0, 0, 0, 0.03)',
    transition: 'transform 0.2s ease'
  },
  bubbleHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.5rem'
  },
  bubbleAuthor: {
    fontSize: '0.9rem',
    fontWeight: '600',
    color: '#1d1d1f'
  },
  bubbleScores: {
    display: 'flex',
    gap: '0.5rem'
  },
  bubbleBadge: {
    fontSize: '0.7rem',
    fontWeight: '600',
    padding: '0.25rem 0.6rem',
    borderRadius: '8px',
    color: '#1d1d1f'
  },
  bubbleBadgeToxic: {
    fontSize: '0.7rem',
    fontWeight: '600',
    color: '#1d1d1f',
    backgroundColor: 'rgba(0, 0, 0, 0.15)',
    padding: '0.25rem 0.6rem',
    borderRadius: '8px'
  },
  bubbleText: {
    margin: 0,
    fontSize: '0.9rem',
    lineHeight: '1.5',
    color: '#1d1d1f'
  },
  bubbleTime: {
    display: 'block',
    marginTop: '0.5rem',
    fontSize: '0.75rem',
    fontWeight: '500',
    color: '#86868b'
  },

  // SETTINGS PANEL
  settingsPanel: {
    marginTop: '1rem',
    padding: '1rem',
    borderRadius: '12px',
    backgroundColor: 'rgba(0, 0, 0, 0.03)',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
    animation: 'slideDown 0.3s ease'
  },
  settingRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '0.5rem 0'
  },
  settingLabel: {
    fontSize: '0.9rem',
    fontWeight: '500',
    color: '#1d1d1f'
  },

  // APPLE SELECT
  appleSelect: {
    padding: '0.5rem 0.75rem',
    fontSize: '0.85rem',
    borderRadius: '8px',
    border: '1px solid rgba(0, 0, 0, 0.1)',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    color: '#1d1d1f',
    outline: 'none',
    fontWeight: '500',
    cursor: 'pointer',
    minWidth: '120px'
  },

  // TOGGLE SWITCH - iOS Style Grayscale
  toggleContainer: {
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
    transition: 'background-color 0.3s ease'
  },
  toggleThumb: {
    position: 'absolute',
    top: '2px',
    width: '24px',
    height: '24px',
    backgroundColor: 'white',
    borderRadius: '12px',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
    transition: 'transform 0.3s ease'
  },

  // SLIDER
  sliderContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem'
  },
  appleSlider: {
    width: '120px',
    height: '4px',
    borderRadius: '2px',
    outline: 'none',
    appearance: 'none',
    WebkitAppearance: 'none',
    background: 'linear-gradient(to right, #2c2c2e 0%, #2c2c2e 50%, #d1d1d6 50%, #d1d1d6 100%)'
  },
  sliderValue: {
    fontSize: '0.85rem',
    fontWeight: '600',
    color: '#1d1d1f',
    minWidth: '30px',
    textAlign: 'right'
  },

  // EMPTY STATE
  emptyState: {
    textAlign: 'center',
    padding: '2rem',
    color: '#6e6e73',
    fontSize: '0.9rem',
    fontWeight: '500'
  }
}

// CSS Animations
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

.appleCard:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 48px rgba(0, 0, 0, 0.12);
}

.settingsButton:hover {
  transform: scale(1.1);
  background-color: rgba(0, 0, 0, 0.1);
}

.settingsButton:active {
  transform: scale(0.95);
}
`

export default App