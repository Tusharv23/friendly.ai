import React, { useState } from 'react';

interface Props {
  onSubmit: (name: string) => void;
}

export default function NameEntry({ onSubmit }: Props) {
  const [value, setValue] = useState('');
  return (
    <div className="centered">
      <h1>Welcome</h1>
      <p>Enter your name to begin.</p>
      <form onSubmit={e => { e.preventDefault(); if (value.trim()) onSubmit(value.trim()); }}>
        <input value={value} onChange={e => setValue(e.target.value)} placeholder="Your name" />
        <button type="submit" disabled={!value.trim()}>Continue</button>
      </form>
    </div>
  );
}
