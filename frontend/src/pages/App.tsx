import React, { useState } from 'react';
import NameEntry from '../sections/NameEntry';
import PersonaSelect from '../sections/PersonaSelect';
import ChatView from '../sections/ChatView';
import Dashboard from '../sections/Dashboard';

export default function App() {
  const [name, setName] = useState<string | null>(null);
  const [personaKey, setPersonaKey] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showDash, setShowDash] = useState(false);

  if (!name) {
    return <NameEntry onSubmit={setName} />;
  }
  if (!personaKey || !sessionId) {
    return <PersonaSelect name={name} onPersonaChosen={(k, s) => { setPersonaKey(k); setSessionId(s); }} />;
  }
  if (showDash) {
    return (
      <Dashboard
        name={name}
        sessionId={sessionId}
        onBack={() => setShowDash(false)}
        onAnalyse={() => {}}
      />
    );
  }
  return (
    <ChatView
      name={name}
      personaKey={personaKey}
      sessionId={sessionId}
      onReset={() => { setPersonaKey(null); setSessionId(null); }}
      onDashboard={() => setShowDash(true)}
    />
  );
}
