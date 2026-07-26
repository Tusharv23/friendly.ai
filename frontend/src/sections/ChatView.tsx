import React, { useEffect, useRef, useState } from 'react';
import api from '../lib/api';

interface Message { role: 'user' | 'assistant'; content: string; isPast?: boolean; isSeparator?: boolean; }

interface Props {
  name: string;
  personaKey: string;
  sessionId: string;
  onReset: () => void;
  onDashboard: () => void;
}

export default function ChatView({ name, personaKey, sessionId, onReset, onDashboard }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);

  // Load full history from all past sessions on mount
  useEffect(() => {
    const userId = name.toLowerCase().trim();
    api.get(`/api/full-history/${encodeURIComponent(userId)}`)
      .then(r => {
        const past: Message[] = (r.data.messages || []).map((m: any) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content,
          isPast: true,
        }));
        if (past.length > 0) setMessages(past);
      })
      .catch(() => {/* no history yet, that's fine */});
  }, [sessionId]);

  const send = async () => {
    const text = input.trim();
    if (!text) return;
    setInput('');
    // Insert a "— new session —" divider the first time the user sends in this session
    setMessages(m => {
      const hasNewMsg = m.some(msg => !msg.isPast && !msg.isSeparator);
      const separator: Message[] = (!hasNewMsg && m.length > 0)
        ? [{ role: 'user', content: '', isSeparator: true }]
        : [];
      return [...m, ...separator, { role: 'user', content: text }];
    });
    setLoading(true);
    try {
      const now = new Date();
      const r = await api.post('/api/message', {
        session_id: sessionId,
        message: text,
        client_hour: now.getHours(),
        client_minute: now.getMinutes(),
      });
      setMessages(m => [...m, { role: 'assistant', content: r.data.reply }]);
    } catch (e: any) {
      setMessages(m => [...m, { role: 'assistant', content: 'Error: ' + e.message }]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  return (
    <div className="chat-wrapper">
      <header>
        <div>
          <strong>{personaKey.split(',')[0]}</strong>
          <small className="sub">Chatting as {name}</small>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={onDashboard} title="Your profile">📊</button>
          <button onClick={onReset}>Change Persona</button>
        </div>
      </header>
      <div className="messages">
        {messages.map((m, i) => {
          if (m.isSeparator) {
            return (
              <div key={i} className="session-separator">
                <span>— new session —</span>
              </div>
            );
          }
          return (
            <div
              key={i}
              className={`msg ${m.role}${m.isPast ? ' past' : ''}`}
            >
              {m.content}
            </div>
          );
        })}
        {loading && <div className="msg assistant">Thinking…</div>}
        <div ref={endRef} />
      </div>
      <form className="input-row" onSubmit={e => { e.preventDefault(); send(); }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type a message"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>Send</button>
      </form>
    </div>
  );
}
