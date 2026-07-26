import React, { useEffect, useState } from 'react';
import api from '../lib/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts';

interface Props {
  name: string;
  sessionId: string;
  onBack: () => void;
  onAnalyse: () => void;
}

interface BiographyEntry {
  fact_id: string;
  type: string;
  text: string;
  confidence: number;
  verified?: boolean;
  defied_by?: string | null;
}

interface EmotionPoint {
  timestamp: string;
  session_id: string;
  emotion: string;
}

// Map emotions to a numeric score for the chart
const EMOTION_SCORE: Record<string, number> = {
  joy: 9, excitement: 8, amusement: 8, pride: 8, admiration: 7,
  relief: 7, gratitude: 7, approval: 6, optimism: 6, love: 7,
  neutral: 5, realization: 5, curiosity: 5, surprise: 5,
  confusion: 4, embarrassment: 4, nervousness: 4,
  disappointment: 3, remorse: 3, sadness: 2, fear: 2,
  anger: 2, annoyance: 2, disgust: 2, grief: 1,
};

const EMOTION_COLOR: Record<string, string> = {
  joy: '#22c55e', excitement: '#22c55e', amusement: '#22c55e',
  relief: '#86efac', gratitude: '#86efac', optimism: '#86efac',
  neutral: '#94a3b8', realization: '#94a3b8', curiosity: '#94a3b8',
  sadness: '#818cf8', grief: '#6366f1', fear: '#f59e0b',
  anger: '#ef4444', annoyance: '#ef4444', disgust: '#ef4444',
  nervousness: '#f97316', confusion: '#f97316',
};

// Normalise emotion string — strips list brackets from stored values like "['approval']"
function cleanEmotion(raw: string): string {
  return raw.replace(/^\[['"]?/, '').replace(/['"]?\]$/, '').split(',')[0].trim().replace(/['"]/g, '');
}

function emotionScore(e: string): number {
  return EMOTION_SCORE[cleanEmotion(e).toLowerCase()] ?? 5;
}

function emotionColor(e: string): string {
  return EMOTION_COLOR[cleanEmotion(e).toLowerCase()] ?? '#94a3b8';
}

function confidenceBadge(c: number): string {
  if (c >= 0.8) return '🟢';
  if (c >= 0.5) return '🟡';
  return '🔴';
}

const CustomDot = (props: any) => {
  const { cx, cy, payload } = props;
  const color = emotionColor(payload.emotion);
  return <circle cx={cx} cy={cy} r={4} fill={color} stroke="#0f1115" strokeWidth={1} />;
};

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{ background: '#1e2430', border: '1px solid #333', padding: '8px 12px', borderRadius: 8, fontSize: 13 }}>
      <div style={{ color: emotionColor(d.emotion), fontWeight: 600 }}>{d.emotion}</div>
      <div style={{ color: '#9aa3b3', marginTop: 2 }}>{d.timestamp?.slice(0, 16).replace('T', ' ')}</div>
      <div style={{ color: '#555d6e', marginTop: 2 }}>session {d.session_id}</div>
    </div>
  );
};

export default function Dashboard({ name, sessionId, onBack, onAnalyse }: Props) {
  const userId = name.toLowerCase().trim();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [analysing, setAnalysing] = useState(false);

  const load = () => {
    setLoading(true);
    api.get(`/api/dashboard/${encodeURIComponent(userId)}`)
      .then(r => setData(r.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const triggerAnalysis = async () => {
    setAnalysing(true);
    try {
      await api.post(`/api/analyse/${sessionId}`);
      // Wait a moment for the background thread then reload
      setTimeout(() => { load(); setAnalysing(false); }, 4000);
    } catch {
      setAnalysing(false);
    }
  };

  if (loading) return <div className="dash-loading">Loading your profile…</div>;
  if (error)   return <div className="dash-loading" style={{ color: '#ef4444' }}>{error}</div>;

  const timeline: EmotionPoint[] = data?.emotion_timeline ?? [];
  const live: string[] = data?.live_emotions ?? [];
  const bio = data?.biography ?? {};
  const crux = bio.crux;

  // Build chart data — last 40 points
  const chartData = timeline.slice(-40).map((p: EmotionPoint, i: number) => ({
    i,
    score: emotionScore(p.emotion),
    emotion: p.emotion,
    timestamp: p.timestamp,
    session_id: p.session_id,
  }));

  // Current live session appended
  const liveChartData = live.slice(-10).map((e, i) => ({
    i: chartData.length + i,
    score: emotionScore(e),
    emotion: e,
    timestamp: 'now',
    session_id: 'live',
  }));

  const allChartData = [...chartData, ...liveChartData];
  const liveSplit = chartData.length;  // index where live starts

  return (
    <div className="dash-wrapper">
      {/* Header */}
      <div className="dash-header">
        <div>
          <h2 className="dash-title">Your Profile</h2>
          <span className="dash-sub">{data.session_count} session{data.session_count !== 1 ? 's' : ''} · {timeline.length} messages tracked</span>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button className="dash-btn-secondary" onClick={triggerAnalysis} disabled={analysing}>
            {analysing ? 'Analysing…' : '⚡ Run Analysis'}
          </button>
          <button className="dash-btn-secondary" onClick={onBack}>← Back to Chat</button>
        </div>
      </div>

      {/* Crux */}
      {crux && (
        <div className="dash-card">
          <div className="dash-card-label">Last Session Summary</div>
          <p className="dash-crux-text">{crux.text}</p>
        </div>
      )}

      {/* EQ / Emotion Chart */}
      <div className="dash-card">
        <div className="dash-card-label">Emotional Arc</div>
        {allChartData.length === 0 ? (
          <div className="dash-empty">No emotion data yet — start chatting</div>
        ) : (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={allChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2430" />
              <XAxis dataKey="i" hide />
              <YAxis domain={[1, 9]} ticks={[1, 3, 5, 7, 9]} tick={{ fill: '#555d6e', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine x={liveSplit} stroke="#6366f1" strokeDasharray="4 2" label={{ value: 'now', fill: '#6366f1', fontSize: 11 }} />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#6366f1"
                strokeWidth={2}
                dot={<CustomDot />}
                activeDot={{ r: 6 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
        {/* Y-axis legend */}
        <div className="dash-axis-legend">
          <span style={{ color: '#ef4444' }}>1 = distress</span>
          <span style={{ color: '#94a3b8' }}>5 = neutral</span>
          <span style={{ color: '#22c55e' }}>9 = joy</span>
        </div>
      </div>

      {/* Live emotions this session */}
      {live.length > 0 && (
        <div className="dash-card">
          <div className="dash-card-label">This Session — Emotion Trail</div>
          <div className="dash-emotion-trail">
            {live.map((e, i) => (
              <span key={i} className="dash-emotion-pill" style={{ background: emotionColor(e) + '22', color: emotionColor(e), borderColor: emotionColor(e) + '44' }}>
                {cleanEmotion(e)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Active phase */}
      {bio.phases?.length > 0 && (
        <div className="dash-card">
          <div className="dash-card-label">Emotional Phase</div>
          {bio.phases.map((p: BiographyEntry) => (
            <div key={p.fact_id} className="dash-phase">
              <span className="dash-phase-icon">🌀</span>
              <span>{p.text}</span>
              <span className="dash-conf">{confidenceBadge(p.confidence)} {(p.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}

      {/* Predictions */}
      {bio.predictions?.length > 0 && (
        <div className="dash-card">
          <div className="dash-card-label">Predictions</div>
          {bio.predictions.map((p: BiographyEntry) => (
            <div key={p.fact_id} className="dash-row">
              <span className="dash-type-badge pred">prediction</span>
              <span className="dash-entry-text">{p.text}</span>
              <span className="dash-conf">{confidenceBadge(p.confidence)} {(p.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}

      {/* Patterns */}
      {bio.patterns?.length > 0 && (
        <div className="dash-card">
          <div className="dash-card-label">Patterns</div>
          {bio.patterns.map((p: BiographyEntry) => (
            <div key={p.fact_id} className="dash-row">
              <span className="dash-type-badge pattern">pattern</span>
              <span className="dash-entry-text">{p.text}</span>
              <span className="dash-conf">{confidenceBadge(p.confidence)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Facts */}
      {bio.facts?.length > 0 && (
        <div className="dash-card">
          <div className="dash-card-label">Known Facts</div>
          {bio.facts.map((f: BiographyEntry) => (
            <div key={f.fact_id} className="dash-row">
              <span className="dash-type-badge fact">fact</span>
              <span className="dash-entry-text">{f.text}</span>
              <span className="dash-conf">{confidenceBadge(f.confidence)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!crux && bio.facts?.length === 0 && bio.patterns?.length === 0 && (
        <div className="dash-card dash-empty">
          <p>No analysis yet. Have a conversation then click <strong>⚡ Run Analysis</strong>.</p>
        </div>
      )}
    </div>
  );
}
