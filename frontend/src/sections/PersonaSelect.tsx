import React, { useEffect, useState } from 'react';
import api from '../lib/api';

interface Persona {
  key: string;
  short: string;
}

interface Props {
  name: string;
  onPersonaChosen: (personaKey: string, sessionId: string) => void;
}

export default function PersonaSelect({ name, onPersonaChosen }: Props) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [personas, setPersonas] = useState<Persona[]>([]);

  useEffect(() => {
    api.get('/api/personalities')
      .then(r => {
        console.log('Personalities response', r.data);
        const data = r.data;
        if (data && Array.isArray(data.personalities)) {
          setPersonas(data.personalities);
        } else {
          setError('Invalid personalities payload');
        }
      })
      .catch(e => setError(e.message || 'Request failed'))
      .finally(() => setLoading(false));
  }, []);

  const choose = async (key: string) => {
    try {
  const r = await api.post('/api/start', { name, persona_key: key });
      onPersonaChosen(key, r.data.session_id);
    } catch (e: any) {
      setError(e.message);
    }
  };

  if (loading) return <div className="centered">Loading personas…</div>;
  if (error) return <div className="centered error">{error}</div>;
  if (!personas) return <div className="centered error">No data</div>;

  return (
    <div className="persona-grid-wrapper">
      <h2>Hello {name}, pick someone to chat with:</h2>
      <div className="persona-grid">
        {personas.map(p => (
          <div key={p.key} className="persona-card" onClick={() => choose(p.key)}>
            <h3>{p.key.split(',')[0]}</h3>
            <p>{p.short}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
