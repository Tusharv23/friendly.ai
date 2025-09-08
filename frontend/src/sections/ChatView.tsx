import React, { useEffect, useRef, useState } from 'react';
import api from '../lib/api';

interface Message { role: 'user' | 'assistant'; content: string; }

interface Props {
  name: string;
  personaKey: string;
  sessionId: string;
  onReset: () => void;
}

export default function ChatView({ name, personaKey, sessionId, onReset }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);

  const send = async () => {
    const text = input.trim();
    if (!text) return;
    setInput('');
    setMessages(m => [...m, { role: 'user', content: text }]);
    setLoading(true);
    try {
  const r = await api.post('/api/message', { session_id: sessionId, message: text });
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
        <button onClick={onReset}>Change Persona</button>
      </header>
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={"msg " + m.role}>{m.content}</div>
        ))}
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
