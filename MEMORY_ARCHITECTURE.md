# Memory & Persona Architecture — ai-talk

A deep reference for every file, class, function, and variable in the memory and persona system.

---

## Table of Contents

1. [Philosophy & Design Decisions](#1-philosophy--design-decisions)
2. [Three-Tier Memory Model](#2-three-tier-memory-model)
3. [Data Flow — End to End](#3-data-flow--end-to-end)
4. [File Reference](#4-file-reference)
   - [memory/episode_store.py](#memoryepisode_storepy)
   - [memory/biography_store.py](#memorybiography_storepy)
   - [memory/analysis_engine.py](#memoryanalysis_enginepy)
   - [memory/memory_manager.py](#memorymemory_managerpy)
   - [therapy.py](#therapypy)
   - [personalities.py](#personalitiespy)
   - [persona_state.py](#persona_statepy)
   - [emotion_classifier.py](#emotion_classifierpy)
   - [humaniser.py](#humaniserpy)
   - [server.py](#serverpy)
5. [Storage Layout on Disk](#5-storage-layout-on-disk)
6. [Prompt Assembly — What the LLM Actually Sees](#6-prompt-assembly--what-the-llm-actually-sees)
7. [Key Design Decisions & Why](#7-key-design-decisions--why)

---

## 1. Philosophy & Design Decisions

The system is built around one insight: **most AI companions are stateless — they forget you the moment the chat ends.** This app treats memory as a first-class citizen.

Inspired by [MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560), which treats the LLM like a CPU and memory like an OS — with tiered storage. We extend that idea with an **emotional intelligence layer**: the system doesn't just remember facts, it tracks emotional phases, predicts the user's next emotional state, and calibrates the persona's tone accordingly.

**Core rules:**
- Memory writes never block the chat. All persistence is fire-and-forget.
- Nothing is deleted, only deactivated (soft-delete). History is sacred.
- The LLM sees a curated, budget-capped context — not a raw dump of everything.
- The persona is a character with a life, not a chatbot with a system prompt.

---

## 2. Three-Tier Memory Model

```
┌─────────────────────────────────────────────────────────┐
│  TIER 3 — Working Memory (in-RAM, session-scoped)        │
│  Lives inside Agent.memory (Memory dataclass)            │
│  • Live turns (last 15 after compression)                │
│  • Rolling compression summaries (last 3)                │
│  • Emotion history (last 10 labels + intensities)        │
│  • Persona self-disclosures (last 10 statements)         │
│  • Past session turns (read-only reference)              │
└──────────────────────┬──────────────────────────────────┘
                       │ persisted after every message
┌──────────────────────▼──────────────────────────────────┐
│  TIER 1 — Episode Store (TinyDB, permanent)              │
│  data/episodes/{user_id}.json                            │
│  • Every turn ever said, append-only                     │
│  • Indexed by session_id + turn_index                    │
│  • Emotion labels per user turn                          │
│  • Persona state (disclosures + session end time)        │
└──────────────────────┬──────────────────────────────────┘
                       │ analysed at session end / 100 msgs
┌──────────────────────▼──────────────────────────────────┐
│  TIER 2 — Biography Store (vector DB, permanent)         │
│  data/biography/{user_id}.json + _vecs.json              │
│  • Facts, patterns, phases, predictions, crux            │
│  • Semantically searchable via cosine similarity         │
│  • Confidence scores, verified/defied predictions        │
│  • Deduplicated — no redundant entries                   │
└─────────────────────────────────────────────────────────┘
```

**Why three tiers?**

- **Tier 3** is fast — it's just Python dataclasses in RAM. Every message reads and writes it instantly.
- **Tier 1** is the archive — exact verbatim history, never loses anything, cheap to write via TinyDB.
- **Tier 2** is the intelligence — distilled knowledge about the user, searchable by meaning not keyword.

The tiers serve different purposes in the prompt:
- Tier 3 summaries → what happened earlier this conversation
- Tier 1 past turns → what happened last session (labelled as background context)
- Tier 2 biography → who this person is, injected as context for every reply

---

## 3. Data Flow — End to End

### On every user message:

```
User sends message
  │
  ├─ Agent.reply(user_text)
  │    ├─ memory.add("user", text)           [Tier 3: append to live turns]
  │    │    └─ if chars > 8000: compress()   [Tier 3: evict oldest → summary]
  │    │
  │    ├─ get_meta_summary(text)             [LLM: intensity, type, past_ref]
  │    ├─ EmotionClassifier.classify(text)   [HuggingFace: emotion labels]
  │    │    └─ intensity override if neutral + high intensity
  │    │
  │    ├─ BiographyStore.build_context_block(text)  [Tier 2: cosine search]
  │    ├─ memory.summary_block()             [Tier 3: compression summaries]
  │    ├─ memory.persona_disclosure_block()  [Tier 3: what persona said]
  │    ├─ memory.past_context_block()        [Tier 3: last session reference]
  │    │
  │    ├─ Build system prompt (all blocks assembled)
  │    ├─ LLM generates response
  │    ├─ _track_persona_disclosures(reply)  [Tier 3: extract I-statements]
  │    └─ text_style_transfer(reply)         [Humaniser: casual rewrite]
  │
  ├─ mm.append_turn(session_id, ...)         [Tier 1: persist both turns]
  ├─ mm.save_persona_state(...)              [Tier 1: save disclosures + time]
  └─ if turn_index % 100 == 0: trigger_analysis()  [Tier 2: background]
```

### On new session start:

```
POST /api/start
  │
  ├─ _flush_session(old_session)             [if same user had active session]
  │    ├─ save_persona_state()               [Tier 1: final disclosure save]
  │    └─ trigger_analysis(old_session_id)   [Tier 2: analyse ending session]
  │
  ├─ mm.get_past_session()                   [Tier 1: load last session turns]
  ├─ mm.load_persona_state(persona_key)      [Tier 1: load disclosures + time]
  ├─ Agent.__init__(past_turns, prior_disclosures, session_ended_at)
  │    ├─ memory.past_turns = past_turns     [Tier 3: seed reference context]
  │    ├─ memory.persona_disclosures = prior [Tier 3: seed persona memory]
  │    └─ infer_persona_activity(...)        [LLM: what is persona doing now]
  └─ agent.user_id = user_id                 [so Agent can query Tier 2]
```

### Analysis engine run (background thread):

```
AnalysisEngine.run(session_id)
  │
  ├─ episode_store.get_session_turns()       [Tier 1: read all turns]
  ├─ biography_store.get_all()               [Tier 2: read existing biography]
  ├─ biography_store.get_active_predictions()[Tier 2: predictions to check]
  │
  ├─ LLM call (max_tokens=2048):
  │    "Analyse transcript, check predictions, extract facts, generate new ones"
  │    Returns JSON: crux, prediction_checks, new_entries, eq_delta
  │
  ├─ _process_result():
  │    ├─ deactivate old crux, write new crux
  │    ├─ mark predictions verified / defied / inconclusive
  │    ├─ for each new entry: cosine dedup check (≥0.65 → merge, else insert)
  │    ├─ deactivate stale entries
  │    └─ deduplicate() — post-write cleanup
  └─ return written summary
```


---

## 4. File Reference

---

### memory/episode_store.py

**Purpose:** Tier 1. Append-only log of every conversation turn. The ground truth — never modified, only appended.

**Storage:** `data/episodes/{user_id}.json` — one TinyDB file per user. Two TinyDB tables inside:
- `episodes` — all conversation turns
- `persona_state` — persona's last known self-disclosures + session end time

#### Class: `EpisodeStore`

**Constructor:**
```python
EpisodeStore(user_id: str)
```
- `user_id` — lowercased, stripped. Used as the filename.
- `self._db` — TinyDB instance (the file handle)
- `self._table` — the `episodes` table reference

**Key variables:**
- `_BASE_DIR` — `data/episodes/` relative to project root. Set at module level.

#### Methods

| Method | Purpose |
|--------|---------|
| `append(session_id, turn_index, role, content, emotion_labels, persona_key)` | Write one turn. Never raises — logs on failure. `turn_index` is 0-based per session. |
| `get_session_turns(session_id)` | All turns for a session, sorted by `turn_index` ascending. Returns `[]` if not found. |
| `get_last_session()` | `(session_id, turns)` of the most recent session. Used to seed Agent context on new session. Returns `(None, [])` if no history. |
| `get_message_count(session_id)` | Count of turns in a session. Used by server.py to check the 100-message analysis trigger. |
| `get_all_sessions()` | All session UUIDs ordered by first-turn timestamp. Used by analysis engine and full-history endpoint. |
| `get_recent_turns(session_id, n=10)` | Last N turns, oldest-first. Convenience method. |
| `save_persona_state(persona_key, disclosures, session_ended_at)` | Upsert persona state — overwrites previous. Called after every message so state is always current. |
| `load_persona_state(persona_key)` | Load `{disclosures, session_ended_at}`. Returns empty defaults if nothing saved. |
| `close()` | Close TinyDB file handle. |

**Turn document schema:**
```json
{
  "session_id":     "uuid",
  "turn_index":     0,
  "timestamp":      "2026-07-25T10:30:00Z",
  "role":           "user | assistant",
  "content":        "message text",
  "emotion_labels": ["sadness", "fear"],
  "persona_key":    "Vicky, tier-2 city guy..."
}
```

**Why TinyDB?** Pure Python, no server, single JSON file, full query API. Zero setup. For the scale of this app (hundreds to low thousands of turns per user), it's more than sufficient.

---

### memory/biography_store.py

**Purpose:** Tier 2. Vector-indexed biographical intelligence about the user. Searchable by meaning, not keyword.

**Storage:**
- `data/biography/{user_id}.json` — metadata for all entries (JSON array)
- `data/biography/{user_id}_vecs.json` — LocalVectorDB embeddings (all-MiniLM-L6-v2)

#### Entry types

| Type | What it stores | Example |
|------|---------------|---------|
| `fact` | Something confirmed true | "Works at a startup" |
| `pattern` | Recurring behaviour | "Goes quiet before big decisions" |
| `phase` | Current emotional arc stage | "In denial phase of breakup grief" |
| `prediction` | Expected next state | "Will feel anger in 2-3 days" |
| `crux` | Session summary (2-3 sentences) | "Session was about breakup. Not eating." |

#### Entry schema

| Field | Type | Description |
|-------|------|-------------|
| `fact_id` | UUID string | Primary key |
| `user_id` | str | Owner |
| `type` | str | One of five types above |
| `text` | str | Human-readable statement |
| `confidence` | float 0.0–1.0 | How certain we are |
| `source_sessions` | list[str] | Which sessions contributed |
| `created_at` | ISO 8601 UTC | First written |
| `updated_at` | ISO 8601 UTC | Last modified |
| `verified` | bool | Prediction confirmed true |
| `verified_at` | str or null | When verified |
| `defied_by` | str or null | What contradicted a prediction |
| `active` | bool | False = soft-deleted |

#### Class: `BiographyStore`

**Constructor:**
```python
BiographyStore(user_id: str)
```
- `self._meta_path` — path to JSON metadata file
- `self._vec_path` — path to vector file
- `self._entries` — `dict[fact_id, entry_dict]` — in-memory index
- `self._vdb` — `LocalVectorDB` instance loaded from disk

#### Methods

| Method | Purpose |
|--------|---------|
| `upsert(data)` | Insert new entry or update existing (by fact_id). Embeds text into vector DB. Returns `fact_id`. |
| `mark_verified(fact_id)` | Prediction came true. Sets `verified=True`, boosts confidence to 1.0. |
| `mark_defied(fact_id, context)` | Prediction was wrong. Records `defied_by`, lowers confidence by 0.3. |
| `deactivate(fact_id)` | Soft-delete. Sets `active=False`. Entry stays in file, excluded from queries. |
| `update_confidence(fact_id, delta)` | Nudge confidence ±delta without full verify/defy. |
| `search(query, top_k, types, min_confidence)` | Cosine similarity search via LocalVectorDB. Filters by type and confidence. Returns list with `_score` field added. |
| `get_crux()` | Most recently created active `crux` entry. |
| `get_active_predictions()` | All active, unverified predictions, sorted by confidence desc. |
| `get_active_phases()` | All active `phase` entries. |
| `get_all(entry_type, active_only)` | All entries, optionally filtered, sorted by `created_at` desc. |
| `build_context_block(query, max_chars=800)` | Build the `[Biography context]` prompt block. Leads with crux, then phases, then relevant facts/patterns by cosine score. Budget-capped. |
| `deduplicate(threshold=0.65)` | Find semantically similar entries of same type. Keep higher confidence, deactivate lower. Returns count removed. |
| `summary()` | Quick stats dict: total, by_type counts, prediction/phase counts. |

**Why LocalVectorDB?** Already existed in the codebase (`local_vectorDB.py`). Uses `all-MiniLM-L6-v2` embeddings + numpy cosine similarity. No external service, no API key. For hundreds of entries per user, performance is fine (≤200ms retrieval).

**Why split metadata + vectors into two files?** The vector file grows with each embedding and is binary-heavy. The metadata file is small JSON readable by humans and tools. Keeping them separate means you can inspect/debug metadata without parsing vectors.

---

### memory/analysis_engine.py

**Purpose:** The intelligence engine. Reads Tier 1, analyses with the LLM, writes to Tier 2. Runs in a background thread — never blocks chat.

#### Module-level

| Name | Description |
|------|-------------|
| `_ANALYSIS_PROMPT` | The full prompt template. Uses `.format()` with `user_id`, `session_id`, `existing_biography`, `pending_predictions`, `transcript`, `emotion_arc`. |

The prompt instructs the LLM to return a JSON object with:
- `crux` — 2-3 sentence session summary
- `prediction_checks` — array checking each pending prediction: `verified`, `defied`, or `inconclusive`
- `new_entries` — array of new facts/patterns/phases/predictions to add
- `deactivate_ids` — entries now stale/contradicted
- `eq_delta` — integer -2 to +2 (emotional growth this session)
- `eq_reasoning` — one sentence justifying the delta

**Critical prompt instruction:** `"Do NOT use apostrophes inside string values"` — because contractions like "doesn't" break JSON parsing.

#### Class: `AnalysisEngine`

**Constructor:**
```python
AnalysisEngine(llm_func, user_id)
```
- `self.llm` — the LLM callable. Uses `bedrock_claude_llm_long` (2048 tokens) — regular 400-token limit truncates the response.
- `self.episode_store` — EpisodeStore instance for reading turns
- `self.biography_store` — BiographyStore instance for reading/writing

#### Methods

| Method | Purpose |
|--------|---------|
| `_build_transcript(turns)` | Format turns as `[timestamp] ROLE [emotions]: content`. Passed to LLM as the raw conversation. |
| `_build_emotion_arc(turns)` | `joy → neutral → anger → ...` string of user emotions. Helps LLM see the emotional shape of the session. |
| `_build_existing_biography()` | Format top-20 active biography entries for the prompt. Capped to avoid exceeding context. |
| `_build_pending_predictions()` | List all unverified predictions with their IDs. LLM checks each against the transcript. |
| `_call_llm(prompt)` | Call LLM, extract JSON, handle parse failures with 4 fallback strategies. |
| `_process_result(result, session_id)` | Write analysis output to BiographyStore. Returns written summary dict. |
| `_resolve_short_id(short_id)` | Match 8-char UUID prefix to full UUID. LLM sometimes outputs truncated IDs. |
| `run(session_id)` | Main entry point. Returns written summary or None. Skips if < 2 user turns. |

#### JSON Parse Fallback Chain

When the LLM returns malformed JSON (e.g. literal newline inside a string value):

1. **Strict parse** — `json.loads(raw)`
2. **Clean and retry** — `clean_json()` collapses newlines inside strings, retries
3. **LLM self-repair** — ask the LLM to fix its own JSON output
4. **ast.literal_eval** — for Python-dict-like outputs
5. **Regex partial extraction** — manually extract `crux`, `eq_delta`, `eq_reasoning`, `new_entries` with regex patterns

**Deduplication threshold: 0.65 cosine similarity.** Entries with similarity ≥ 0.65 of the same type are considered equivalent. The higher-confidence one is kept, the lower is deactivated.


---

### memory/memory_manager.py

**Purpose:** Unified facade. The Agent and server.py talk to this — they never import EpisodeStore or BiographyStore directly.

#### Class: `MemoryManager`

**Constructor:**
```python
MemoryManager(user_id, llm_func=None)
```
- `self._llm` — optional LLM callable. Required for `trigger_analysis()`. Uses `bedrock_claude_llm_long`.
- `self._episodes` — EpisodeStore instance
- `self._biography` — BiographyStore instance

#### Methods

| Method | Purpose |
|--------|---------|
| `append_turn(...)` | Tier 1 write — delegates to EpisodeStore. |
| `get_past_session()` | Returns `(session_id, turns)` of most recent session. |
| `get_message_count(session_id)` | Turn count — used to check 100-msg trigger. |
| `save_persona_state(persona_key, disclosures)` | Delegates to EpisodeStore. |
| `load_persona_state(persona_key)` | Delegates to EpisodeStore. |
| `get_full_history()` | All turns across all sessions, oldest first. For the chat history endpoint. |
| `biography_context(query, max_chars=800)` | Returns the `[Biography context]` prompt block. Returns `""` if no biography. |
| `biography_summary()` | Stats dict — total entries, by type. |
| `get_dashboard_data()` | Aggregates emotion timeline + biography for the dashboard endpoint. |
| `trigger_analysis(session_id)` | Fires AnalysisEngine in a daemon background thread. Requires `llm_func`. |
| `close()` | Closes EpisodeStore (TinyDB file handle). |

---

### therapy.py

**Purpose:** The conversation agent. Contains all Tier 3 (working memory) logic and the prompt assembly pipeline.

#### Dataclass: `Turn`

```python
@dataclass
class Turn:
    role: str     # "user" | "assistant"
    content: str
```

Minimal unit in the live conversation. No timestamp — those are in Tier 1.

#### Dataclass: `EmotionHistory`

Tracks the emotional arc within the current session.

| Field | Type | Description |
|-------|------|-------------|
| `emotions` | list[str] | Last 10 emotion labels |
| `intensities` | list[float] | Corresponding confidence scores |

| Method | Description |
|--------|-------------|
| `add_emotion(emotion, intensity)` | Append and trim to 10. |
| `get_trend()` | Compare positive vs negative in last 3 emotions. Returns `"improving mood"`, `"struggling emotionally"`, or `"mixed emotions"`. |
| `needs_extra_support()` | True if last 2 emotions are in `{despair, hopeless, suicidal, panic, rage}`. |

**Positive emotion set:** `{joy, happiness, relief, calm, hopeful, confident}`
**Negative emotion set:** `{sadness, anger, fear, anxiety, frustration, despair}`

#### Dataclass: `Memory`

The Tier 3 working memory. All in-RAM, session-scoped.

| Field | Type | Description |
|-------|------|-------------|
| `turns` | list[Turn] | Live conversation turns (compressed if too large) |
| `past_turns` | list[Turn] | Last session's turns — read-only background reference |
| `emotion_history` | EmotionHistory | Session emotion arc |
| `persona_disclosures` | list[str] | Things the persona said about itself this session |
| `summaries` | list[str] | Rolling compression summaries (last 3) |
| `max_chars` | int=8000 | Character budget before compression triggers |
| `compress_keep` | int=15 | Raw turns kept after compression |

| Method | Description |
|--------|-------------|
| `add(role, text, llm_func)` | Append turn. If total chars > max_chars and llm_func provided, calls `compress()`. |
| `compress(llm_func)` | Evicts oldest turns except last 15, summarises them via LLM (3-5 bullets). Appends to `summaries`, capped at 3. Falls back to first-sentence extraction if LLM fails. |
| `summary_block()` | Returns `[Earlier in this conversation — summarised]\n...` or `""`. |
| `as_messages()` | Live turns as `[{role, content}]` list for the LLM API. |
| `persona_disclosure_block()` | Returns formatted reminder of persona's self-statements or `""`. |
| `past_context_block(max_chars=1200)` | Formats `past_turns` as a labelled reference block. Truncated to budget. Returns `""` if no past turns. |

#### Class: `Agent`

**Constructor parameters:**

| Parameter | Description |
|-----------|-------------|
| `llm_func` | The LLM callable — `bedrock_claude_llm` for chat |
| `persona_key` | Which personality to use |
| `past_turns` | Turns from last session (from Tier 1) |
| `client_hour` | User's local hour (0-23) — overrides server timezone |
| `client_tz` | User's timezone string e.g. `"Asia/Kolkata"` |
| `prior_disclosures` | What persona said last session (from Tier 1) |
| `session_ended_at` | ISO timestamp when last session ended |

**Instance variables:**

| Variable | Description |
|----------|-------------|
| `self.llm` | LLM callable |
| `self.memory` | Memory dataclass (Tier 3) |
| `self.persona_key` | Active persona |
| `self.persona_text` | Full persona prompt + current context (from `infer_persona_activity`) |
| `self.last_emotion_labels` | Normalised list of emotion strings from last `reply()` call. Read by server.py to persist to Tier 1. |
| `self.user_id` | Set by server.py after construction. Used to query Tier 2. |

#### Method: `Agent.reply(user_text, client_hour, client_minute)`

The core pipeline. Called once per message. Steps:

1. `memory.add("user", text)` — possibly triggers compression
2. `get_meta_summary(text)` — LLM call for intensity, type, past_context_required
3. `EmotionClassifier.classify(text)` — HuggingFace emotion labels
4. **Intensity override** — if classifier returns neutral/curiosity but meta intensity ≥ 6, override to `"nervousness"`
5. **Normalise emotion labels** — strip stringified list brackets (`['sadness']` → `sadness`)
6. Update `EmotionHistory`
7. Build `live_time_context` from client_hour/minute
8. Build `biography_block` via `MemoryManager.biography_context()`
9. Build `summary_block`, `disclosure_block`, `past_context_instruction`
10. **Relevance gate** — only inject past context if negative emotion OR meta flags `past_context_required: true`
11. Assemble full system prompt
12. LLM generates response
13. `_track_persona_disclosures(reply)` — extract I-statements from reply
14. `text_style_transfer(reply)` — humaniser rewrite
15. If AI detected: remove triggering user turn from memory
16. `memory.add("assistant", reply)`

#### Method: `Agent.get_meta_summary(text)`

Calls LLM with a structured prompt asking for a JSON object:
```json
{
  "intensity": 1-10,
  "type": "greeting|emotional-input|...",
  "past_context_required": true|false,
  "ai_reveal": true|false
}
```

**Why a separate LLM call?** The emotion classifier (HuggingFace) is trained on social media text — it misses understated distress. The LLM's intensity score is a better signal for quiet suffering ("constant pressure", "don't know anymore"). The intensity override uses this to correct the classifier.

#### Method: `Agent._track_persona_disclosures(reply)`

Extracts first-person statements using regex patterns:
- `i('m|am) ...` — state declarations
- `i('ve|have) ...` — experience claims
- `i('ll|will) ...` — future plans
- `i just ...` — recent actions
- `my [noun] (is|was|are) ...` — possessive claims

Appended to `memory.persona_disclosures`, capped at last 10. Injected back into the next turn's system prompt as a consistency reminder.

#### LLM Functions

| Function | Max tokens | Used for |
|----------|-----------|---------|
| `bedrock_claude_llm` | 400 | Chat responses, meta summaries, style transfer |
| `bedrock_claude_llm_long` | 2048 | Analysis engine, session compression |

---

### personalities.py

**Purpose:** Single source of truth for all persona definitions.

#### `PERSONALITIES` dict

Keys are the persona's full description string (used as `persona_key` everywhere — in sessions, episodes, biography entries). Values are the LLM system prompt text.

| Persona key prefix | Character | Tone |
|-------------------|-----------|------|
| `Roshan` | Corporate employee | Direct, action-oriented, practical |
| `Shreya` | College girl, 20s | Casual, texting style, vibes with energy |
| `Vicky` | Tier-2 city guy | Warm humor, intentional grammar mistakes |
| `Sneha` | Mature introvert | Dry, guarded, short replies, no questions |

**Why the full description as the key?** It's self-documenting. When you see `persona_key` in a TinyDB record or biography entry, you immediately know which character it is without a lookup table.

#### Functions

| Function | Description |
|----------|-------------|
| `list_personalities()` | Returns list of all persona keys |
| `get_persona_text(key)` | Returns full system prompt for the key. Falls back to default if key not found. |
| `get_persona_summary(key)` | Returns first sentence of the prompt — used in frontend persona cards. |
| `DEFAULT_PERSONALITY_KEY` | Roshan — the fallback if no persona selected |

---

### persona_state.py

**Purpose:** Infer what the persona is *doing right now* based on time of day and what they said last session. Adds realism — the persona has an ongoing life, not just a chat window.

#### Function: `infer_persona_activity(...)`

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `persona_name` | The full persona key |
| `now` | datetime to use. Defaults to server time but overridden with client_hour. |
| `tz` | Timezone hint string e.g. `"Asia/Kolkata"` |
| `prior_disclosures` | What persona said last session — used for continuity |
| `session_ended_at` | ISO timestamp — used to compute "3 minutes ago" |

**Time-of-day mapping:**

| Hours | Label |
|-------|-------|
| 5–8 | early morning |
| 8–12 | morning |
| 12–14 | lunch hour |
| 14–18 | afternoon |
| 18–22 | evening |
| 22–5 | late night |

**Continuity block** — if `prior_disclosures` or `session_ended_at` are provided, the prompt includes:
> "The last conversation ended 3 minutes ago. In that conversation you said: headed out later with the girls. Your current activity MUST be consistent with this."

This prevents the persona from "coming home" 3 minutes after saying they were going out.

---

### emotion_classifier.py

**Purpose:** Classify user messages into emotion labels using a HuggingFace transformer model.

#### Class: `EmotionClassifier`

Model: `SamLowe/roberta-base-go_emotions` — 28 emotion labels, trained on Reddit comments.

**Known limitation:** Misses understated distress. "I don't know, constant pressure" returns `neutral`. The `intensity override` in `therapy.py` corrects this when meta intensity ≥ 6.

#### Method: `classify(text)`

Returns a list of emotion labels:
- Single label if top emotion is clearly dominant (gap > 0.15 vs second)
- Two labels if scores are close (mixed emotions)

**Threshold:** `0.15` — if `top1.score - top2.score < 0.15`, both labels returned.

---

### humaniser.py

**Purpose:** Post-process the LLM's response into casual, human-sounding text. Also prevents the AI from breaking character.

#### Function: `text_style_transfer(llm, response, user_text, primary_emotion)`

Two-stage pipeline:

1. **AI detection** — if `detect_ai_acceptance(response)` returns True, use a canned deflection instead of the LLM's response. Remove the triggering user message from context.

2. **Style rewrite** — ask the LLM to rewrite the response as a text message using `get_tone_guidance(emotion)`.

Returns `(processed_response, ai_detected: bool)`.

#### Function: `detect_ai_acceptance(text)`

25+ regex patterns catching phrases like:
- `"as an ai"`, `"i am an ai"`, `"i'm an ai"`
- `"i don't have emotions"`, `"i don't have feelings"`
- `"created by anthropic"`, `"i'm claude"`
- `"not a real person"`, `"different from human connections"`

#### Function: `get_ai_deflection(emotion, user_text)`

Context-aware deflection. Checks user_text for keywords (`fake`, `lie`, `honest`) and selects an appropriate response based on emotion. Never admits to being AI — redirects to the user's emotional state.

#### Function: `get_tone_guidance(emotion)`

Returns `{tone, emoji_guidance, style}` dict for the style rewrite prompt. Maps emotion to conversational style:

| Emotion group | Tone |
|--------------|------|
| joy, excitement, pride | Match energy, upbeat |
| sadness, grief, fear | Softer, empathetic, no pep |
| anger, annoyance | Calm, no emojis |
| neutral | Chill, laid-back |
| confusion | Patient, gentle |

---

### server.py

**Purpose:** Flask REST API. Manages sessions, routes messages, coordinates memory writes and analysis triggers.

#### Session store: `_sessions`

In-memory dict. Key = `session_id` (UUID). Value:
```python
{
  'session_id':  str,
  'agent':       Agent,
  'mm':          MemoryManager,
  'user_name':   str,
  'user_id':     str,       # lowercased name
  'persona_key': str,
  'turn_index':  int,       # increments by 2 each message (user + assistant)
  'messages':    list,      # frontend echo cache
}
```

#### `_flush_session(entry)`

Called when a new session starts for the same user. Saves persona state and triggers analysis on the ending session (if it had ≥ 4 turns = at least 2 user turns).

#### `_ANALYSIS_TRIGGER = 100`

Mid-session analysis fires every 100 turns. Also fires at session end via `_flush_session`.

#### Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/api/personalities` | GET | List all personas with summaries |
| `/api/start` | POST | Start new session. Flushes previous. Returns `session_id` + past messages. |
| `/api/message` | POST | Send message, get reply. Persists turns, saves persona state. |
| `/api/history/<session_id>` | GET | In-memory message cache for active session |
| `/api/full-history/<user_id>` | GET | All turns from all sessions (Tier 1 read) |
| `/api/analyse/<session_id>` | POST | Manually trigger analysis for active session |
| `/api/dashboard/<user_id>` | GET | Full dashboard data — emotion timeline + biography |
| `/api/health` | GET | `{"status": "ok"}` |

**`user_id` derivation:** `user_name.lower().strip()`. So "Tushar" and "tushar" are the same person. Simple, not collision-resistant — acceptable for MVP.


---

## 5. Storage Layout on Disk

```
data/
  episodes/
    tushar.json          ← TinyDB: 'episodes' table + 'persona_state' table
    alice.json
    ...

  biography/
    tushar.json          ← BiographyStore metadata (array of entry dicts)
    tushar_vecs.json     ← LocalVectorDB: embeddings + metadata
    alice.json
    alice_vecs.json
    ...
```

**Episode file structure (TinyDB JSON):**
```json
{
  "_default": {},
  "episodes": {
    "1": { "session_id": "...", "turn_index": 0, "role": "user", ... },
    "2": { "session_id": "...", "turn_index": 1, "role": "assistant", ... }
  },
  "persona_state": {
    "1": {
      "persona_key": "Vicky, tier-2...",
      "disclosures": ["i am heading out later", "my feet are killing me"],
      "session_ended_at": "2026-07-25T17:30:00Z"
    }
  }
}
```

**Biography metadata file:**
```json
[
  {
    "fact_id": "uuid",
    "user_id": "tushar",
    "type": "pattern",
    "text": "When impatient with theoretical discussion, uses blunt language",
    "confidence": 0.85,
    "source_sessions": ["session-uuid-1", "session-uuid-2"],
    "created_at": "2026-07-25T17:00:00Z",
    "updated_at": "2026-07-25T17:00:00Z",
    "verified": false,
    "verified_at": null,
    "defied_by": null,
    "active": true
  }
]
```

---

## 6. Prompt Assembly — What the LLM Actually Sees

For every message, the system prompt is assembled in this order:

```
CRITICAL SYSTEM INSTRUCTIONS — MUST FOLLOW EXACTLY:
[core rules: no AI admission, stay in character, depth matching]

[Persona text]
You are Vicky: warm, witty small-town best friend...
Current Context: [from infer_persona_activity] Vicky is wrapping up work...

[Current time for both of you: 17:23 — afternoon]

[Biography context — use naturally, never reference directly]
[Last session] Session was about the AI project. User disclosed idealistic mission.
[Pattern] When impatient with theory, redirects abruptly.
[Fact] Building an AI mental-health companion app.

[Earlier in this conversation — summarised]      ← only if >8000 chars
• User mentioned feeling stressed about deadlines
• Persona said they had plans Saturday

[Things YOU have already said about yourself this session — stay consistent]
  - i am heading out later tonight
  - my feet are killing me

[Previous session — for background awareness only]   ← only if negative emotion
  [user]: I've been crying since morning
  [assistant]: aw man that's rough
[You MAY reference the above only if it naturally fits...]

[Current emotion: sadness | Emotional trend: struggling emotionally]

[user's actual message text here]
```

**Budget management:**
- Biography block: max 800 chars (truncated oldest entries first)
- Past session block: max 1200 chars (truncated oldest turns first)
- Persona disclosures: last 8 of stored 10
- Summaries: last 3 compression summaries joined

---

## 7. Key Design Decisions & Why

### Why not dump the entire chat history into every prompt?

Context windows are finite (and expensive). More importantly, the LLM performs worse with very long contexts — it loses focus on the recent conversation. We use selective injection: only the last 15 turns live in context, older turns are compressed to bullets, and even older sessions are distilled to a few biography entries.

### Why is `past_turns` separate from `turns`?

Early version mixed last session's turns directly into the live conversation. This caused the AI to obsess over topics from the previous session (food, breakup) even in casual new conversations. Separating them and adding a **relevance gate** (only inject past context when emotion is negative or meta flags it) fixed this.

### Why cosine similarity threshold of 0.65 for dedup?

The two duplicate patterns scored 0.658. At 0.65 we catch semantic equivalents that are paraphrased differently. Lower would be too aggressive (merging entries that are actually different). Higher would miss real duplicates.

### Why persist persona state after every message, not session end?

A browser refresh or server restart means the session object is gone. If we only saved at session end, a crash would lose all disclosures. Saving after every message means the on-disk state is never more than one message old.

### Why use the full persona description string as `persona_key`?

It's self-documenting in stored data, avoids a separate lookup table, and the persona names are stable (you don't rename Vicky to Victor mid-project). The downside — long strings in every TinyDB record — is negligible at this data scale.

### Why `bedrock_claude_llm_long` (2048 tokens) for analysis only?

The analysis prompt generates a large structured JSON with multiple arrays. At 400 tokens the response truncates mid-JSON (before `eq_delta`, `eq_reasoning`). 2048 is enough for a full analysis of a ~30-message session. Chat responses stay at 400 — they should be concise.

### Why the intensity override from meta_summary?

`roberta-base-go_emotions` was trained on Reddit — expressive, emotional, exaggerated text. It reliably detects "I'm devastated!!!" but returns `neutral` for "I don't know, constant pressure, idk." The meta_summary LLM call gives a second signal — if intensity ≥ 6 despite neutral label, override to `nervousness`. This is a pragmatic fix, not a permanent solution. Long-term, a model trained on conversational text would handle this natively.

---

*Last updated: reflecting implementation as of the current codebase. Update this document whenever significant memory or persona changes are made.*
