import uuid
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Lock
from personalities import list_personalities, get_persona_summary, DEFAULT_PERSONALITY_KEY
from therapy import Agent, bedrock_claude_llm, bedrock_claude_llm_long
from memory.memory_manager import MemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

_sessions: dict = {}
_lock = Lock()

_ANALYSIS_TRIGGER = 100   # also trigger every N turns within a session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flush_session(session_entry: dict) -> None:
    """
    Called when a session ends (new session started for same user).
    Saves persona state and triggers analysis on the ending session.
    """
    mm: MemoryManager   = session_entry["mm"]
    agent: Agent        = session_entry["agent"]
    persona_key: str    = session_entry["persona_key"]
    old_session_id: str = session_entry["session_id"]

    # Save persona disclosures to disk
    mm.save_persona_state(persona_key, agent.memory.persona_disclosures)

    # Run analysis on the session that just ended
    turn_count = mm.get_message_count(old_session_id)
    if turn_count >= 4:   # need at least 2 user turns to be worth analysing
        logger.info("Session end — triggering analysis for session %s", old_session_id[:8])
        mm.trigger_analysis(old_session_id)

    mm.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/api/personalities', methods=['GET'])
def personalities_endpoint():
    personas = [
        {'key': key, 'short': get_persona_summary(key)}
        for key in list_personalities()
    ]
    return jsonify({'personalities': personas, 'default': DEFAULT_PERSONALITY_KEY})


@app.route('/api/start', methods=['POST'])
def start_session():
    data        = request.get_json(force=True)
    user_name   = data.get('name', 'User').strip() or 'User'
    persona_key = data.get('persona_key') or DEFAULT_PERSONALITY_KEY
    client_hour = data.get('client_hour')
    client_tz   = data.get('client_tz', '')
    user_id     = user_name.lower().strip()

    mm = MemoryManager(user_id, llm_func=bedrock_claude_llm_long)

    # Flush any previous session for this user — triggers analysis on it
    with _lock:
        stale = [
            sid for sid, s in _sessions.items()
            if s['user_id'] == user_id
        ]
    for sid in stale:
        with _lock:
            old = _sessions.pop(sid, None)
        if old:
            _flush_session(old)

    # Load context for the new session
    last_session_id, past_turns = mm.get_past_session()
    prior_state = mm.load_persona_state(persona_key)

    agent = Agent(
        bedrock_claude_llm,
        persona_key=persona_key,
        past_turns=past_turns,
        client_hour=client_hour,
        client_tz=client_tz,
        prior_disclosures=prior_state["disclosures"],
        session_ended_at=prior_state["session_ended_at"],
    )
    session_id    = str(uuid.uuid4())
    agent.user_id = user_id

    with _lock:
        _sessions[session_id] = {
            'session_id':  session_id,
            'agent':       agent,
            'mm':          mm,
            'user_name':   user_name,
            'user_id':     user_id,
            'persona_key': persona_key,
            'turn_index':  0,
            'messages':    [],
        }

    return jsonify({
        'session_id':     session_id,
        'persona_key':    persona_key,
        'user_name':      user_name,
        'resumed':        last_session_id is not None,
        'past_turn_count': len(past_turns),
        'past_messages':  [
            {'role': t['role'], 'content': t['content']}
            for t in past_turns
        ],
    })


@app.route('/api/message', methods=['POST'])
def send_message():
    data       = request.get_json(force=True)
    session_id = data.get('session_id')
    text       = data.get('message', '').strip()

    if not session_id or session_id not in _sessions:
        return jsonify({'error': 'invalid_session'}), 400
    if not text:
        return jsonify({'error': 'empty_message'}), 400

    entry       = _sessions[session_id]
    agent:Agent = entry['agent']
    mm: MemoryManager = entry['mm']

    client_hour   = data.get('client_hour')
    client_minute = data.get('client_minute')

    reply = agent.reply(text, client_hour=client_hour, client_minute=client_minute)

    persona_key = entry['persona_key']
    turn_index  = entry['turn_index']

    mm.append_turn(session_id, turn_index,     'user',      text,  agent.last_emotion_labels, persona_key)
    mm.append_turn(session_id, turn_index + 1, 'assistant', reply, [],                        persona_key)
    entry['turn_index'] += 2

    # Save persona state after every message
    mm.save_persona_state(persona_key, agent.memory.persona_disclosures)

    # Mid-session analysis trigger every 100 turns
    if entry['turn_index'] % _ANALYSIS_TRIGGER == 0:
        logger.info("Mid-session analysis at turn %d", entry['turn_index'])
        mm.trigger_analysis(session_id)

    entry['messages'].append({'role': 'user',      'content': text})
    entry['messages'].append({'role': 'assistant', 'content': reply})
    return jsonify({'reply': reply})


@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    if session_id not in _sessions:
        return jsonify({'error': 'invalid_session'}), 400
    return jsonify({'messages': _sessions[session_id]['messages']})


@app.route('/api/full-history/<user_id>', methods=['GET'])
def get_full_history(user_id):
    user_id = user_id.strip().lower()
    if not user_id or '..' in user_id or '/' in user_id:
        return jsonify({'error': 'invalid_user_id'}), 400
    try:
        mm = MemoryManager(user_id)
        messages = mm.get_full_history()
        session_count = len(set(m['session_id'] for m in messages))
        mm.close()
        return jsonify({'messages': messages, 'session_count': session_count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyse/<session_id>', methods=['POST'])
def trigger_analysis(session_id):
    """Manually trigger analysis for an active session."""
    if session_id not in _sessions:
        return jsonify({'error': 'invalid_session'}), 400
    entry = _sessions[session_id]
    entry['mm'].trigger_analysis(session_id)
    return jsonify({'status': 'analysis_started', 'session_id': session_id})


@app.route('/api/dashboard/<user_id>', methods=['GET'])
def get_dashboard(user_id):
    user_id = user_id.strip().lower()
    if not user_id or '..' in user_id or '/' in user_id:
        return jsonify({'error': 'invalid_user_id'}), 400
    try:
        mm = MemoryManager(user_id)
        data = mm.get_dashboard_data()

        # Attach live emotions from the active session if any
        live_emotions = []
        with _lock:
            for sess in _sessions.values():
                if sess['user_id'] == user_id:
                    live_emotions = list(sess['agent'].memory.emotion_history.emotions)
                    break
        data['live_emotions'] = live_emotions

        mm.close()
        return jsonify(data)
    except Exception as e:
        logger.exception("Dashboard error for user=%r", user_id)
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
