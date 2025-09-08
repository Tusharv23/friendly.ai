import React, { useState } from 'react';
import NameEntry from '../sections/NameEntry';
import PersonaSelect from '../sections/PersonaSelect';
import ChatView from '../sections/ChatView';

export default function App() {
  const [name, setName] = useState<string | null>(null);
  const [personaKey, setPersonaKey] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  if (!name) {
    return <NameEntry onSubmit={setName} />;
  }
  if (!personaKey) {
    return <PersonaSelect name={name} onPersonaChosen={(k, s) => { setPersonaKey(k); setSessionId(s); }} />;
  }
  return <ChatView name={name} personaKey={personaKey} sessionId={sessionId!} onReset={() => { setPersonaKey(null); setSessionId(null); }} />;
}
