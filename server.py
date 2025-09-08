import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Lock
from personalities import list_personalities, get_persona_summary, get_persona_text, DEFAULT_PERSONALITY_KEY
from therapy import Agent, bedrock_claude_llm

app = Flask(__name__)
CORS(app)

# In-memory session store
_sessions = {}
_lock = Lock()

@app.route('/api/personalities', methods=['GET'])
def personalities_endpoint():
    personas = []
    for key in list_personalities():
        personas.append({
            'key': key,
            'short': get_persona_summary(key),
        })
    return jsonify({'personalities': personas, 'default': DEFAULT_PERSONALITY_KEY})

@app.route('/api/start', methods=['POST'])
def start_session():
    data = request.get_json(force=True)
    user_name = data.get('name', 'User').strip() or 'User'
    persona_key = data.get('persona_key') or DEFAULT_PERSONALITY_KEY
    agent = Agent(bedrock_claude_llm, persona_key=persona_key)
    session_id = str(uuid.uuid4())
    with _lock:
        _sessions[session_id] = {
            'agent': agent,
            'user_name': user_name,
            'persona_key': persona_key,
            'messages': []  # store history for frontend echo
        }
    return jsonify({'session_id': session_id, 'persona_key': persona_key, 'user_name': user_name})

@app.route('/api/message', methods=['POST'])
def send_message():
    data = request.get_json(force=True)
    session_id = data.get('session_id')
    text = data.get('message', '').strip()
    if not session_id or session_id not in _sessions:
        return jsonify({'error': 'invalid_session'}), 400
    if not text:
        return jsonify({'error': 'empty_message'}), 400
    entry = _sessions[session_id]
    agent: Agent = entry['agent']
    reply = agent.reply(text)
    entry['messages'].append({'role': 'user', 'content': text})
    entry['messages'].append({'role': 'assistant', 'content': reply})
    return jsonify({'reply': reply})

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    if session_id not in _sessions:
        return jsonify({'error': 'invalid_session'}), 400
    return jsonify({'messages': _sessions[session_id]['messages']})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
