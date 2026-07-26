# Requirements Document

## Introduction

Phase 2 of ai-talk extends the companion from a stateless chat interface into a persistent, emotionally-aware companion that users return to daily. The four feature pillars are:

1. **Conversational Memory** — store and retrieve past conversations using the existing `local_vectorDB.py` infrastructure, so the companion can reference meaningful past moments ("Last time you said you were anxious about exams — how did it go?").
2. **Adaptive Empathy Layer** — extend the in-session `EmotionHistory` (already in `therapy.py`) to persist across sessions and drive deeper, rule-based response strategy shifts (e.g., sustained sadness triggers a nurturing tone mode).
3. **User Control Dashboard** — a frontend panel where users can inspect stored memories, delete individual entries, and view a plain-language summary of how the AI currently perceives them.
4. **Optional Voice Mode** — text-to-speech playback of companion replies using ElevenLabs or OpenAI TTS, with per-emotion voice tuning.

Phase 1 baseline: Flask + React/Vite, AWS Bedrock (Claude Opus 4), per-session in-memory `Agent`/`Memory` objects, `EmotionClassifier` (roberta-base-go_emotions), four personas (Roshan, Shreya, Vicky, Sneha), `humaniser.py` style-transfer pipeline, `local_vectorDB.py` with `all-MiniLM-L6-v2` embeddings (exists but unused in the main app).

---

## Glossary

- **Memory_Store**: The persistent vector database layer, backed by `LocalVectorDB` from `local_vectorDB.py`, that stores and retrieves conversation memories per user.
- **Memory_Entry**: A single storable unit — a text snippet (one or more turns) plus metadata: `user_id`, `persona_key`, `timestamp`, `emotion_labels`, `session_id`.
- **User_ID**: A stable identifier for a user, derived from the name they enter at session start and stored in browser `localStorage`.
- **Session**: A single uninterrupted conversation between a user and a persona, identified by `session_id` (UUID, already implemented in `server.py`).
- **Mood_Arc**: The sequence of dominant emotions recorded across multiple sessions for a user, used to infer cross-session emotional trends.
- **Mood_Profile**: A structured per-user object containing the `Mood_Arc`, dominant emotion counts, last-seen emotion, and a human-readable summary string.
- **Empathy_Mode**: A response-strategy state the `Agent` enters based on sustained emotional signals. Current modes: `default`, `nurturing`, `energising`, `grounding`.
- **Retrieval_Context**: The top-N semantically similar `Memory_Entry` items returned by the `Memory_Store` for a given user query, injected into the LLM system prompt.
- **Dashboard**: The frontend React page (new route `/dashboard`) where users manage their memories and view their `Mood_Profile`.
- **TTS_Service**: The external text-to-speech provider (ElevenLabs API or OpenAI TTS API) called after a companion reply is generated.
- **Voice_Profile**: Per-persona, per-emotion voice configuration (voice ID, speed, stability settings) used when constructing a `TTS_Service` request.
- **Agent**: The existing `Agent` class in `therapy.py` that owns an in-session `Memory` object and calls the LLM pipeline.
- **Emotion_Classifier**: The existing `EmotionClassifier` class in `emotion_classifier.py` (roberta-base-go_emotions) that classifies user text into emotion labels.
- **Humaniser**: The existing `text_style_transfer` function in `humaniser.py` that rewrites LLM output into casual, persona-consistent text.
- **Pretty_Printer**: A formatting utility that serialises a `Memory_Entry` or `Mood_Profile` object back to JSON and then parses it again (used for round-trip correctness verification).

---

## Requirements

### Requirement 1: Persistent User Identity

**User Story:** As a returning user, I want the app to remember who I am across browser sessions, so that my memories and mood history are linked to me consistently.

#### Acceptance Criteria

1. WHEN a user submits their name on the Name Entry screen, THE App SHALL validate that the name is between 1 and 50 characters (after trimming), generate a `user_id` by normalising the name to lowercase-trimmed form, and store it in browser `localStorage` under the key `aitalk_user_id`.
2. WHEN a user returns to the app and a non-empty `aitalk_user_id` exists in `localStorage`, THE App SHALL display a returning-user screen showing the stored name with two explicit actions: "That's me — continue" (proceeds to persona selection) and "Not me — change name" (clears `aitalk_user_id` and shows the full Name Entry flow).
3. WHEN the user selects "That's me — continue" on the returning-user screen, THE App SHALL proceed directly to persona selection using the stored `user_id` without re-entering the Name Entry flow.
4. IF the stored `aitalk_user_id` value is an empty string or missing, THEN THE App SHALL treat the user as new and show the full Name Entry flow.
5. THE App SHALL NOT transmit `user_id` values to any third-party service; the identifier is used only within the ai-talk backend.

---

### Requirement 2: Conversational Memory Storage

**User Story:** As a user, I want the companion to store meaningful moments from our conversations, so that it can recall and reference them in future sessions.

#### Acceptance Criteria

1. WHEN a new session is started via `POST /api/start` or a session-close endpoint is called, THE Memory_Store SHALL persist all conversation turns from the preceding session as `Memory_Entry` objects, each tagged with `user_id`, `persona_key`, `session_id`, `timestamp` (ISO 8601 UTC), and the detected `emotion_labels` for that turn (which may be an empty array if classification produced no result).
2. THE Memory_Store SHALL store `Memory_Entry` objects to a JSON file on disk under `data/memories/{user_id}.json`, creating the file and any missing parent directories if they do not exist.
3. THE Memory_Store SHALL generate a vector embedding for each `Memory_Entry` text using the `all-MiniLM-L6-v2` model already loaded in `local_vectorDB.py`.
4. WHEN the `Memory_Store` receives a `Memory_Entry` whose text, after trimming and lowercasing, is identical to an already-stored entry for the same `user_id`, THE Memory_Store SHALL skip the duplicate and not create a second entry.
5. THE Memory_Store SHALL complete a retrieval query over 500 stored `Memory_Entry` objects for a single user in ≤ 200 ms on development hardware.
6. WHEN the total count of `Memory_Entry` objects for a `user_id` exceeds 1000, THE Memory_Store SHALL evict the oldest 100 entries (by `timestamp`) to keep the store bounded.
7. WHEN a JSON file write fails (e.g., disk full, permission denied), THE Memory_Store SHALL retain the entries in memory for the current process lifetime, emit a warning-level log identifying the `user_id` and the I/O error, and not raise an unhandled exception to the caller.
8. WHEN the JSON file for a `user_id` is malformed or unreadable on load, THE Memory_Store SHALL initialise an empty in-memory store for that user, emit a warning-level log identifying the `user_id` and parse error, and not raise an unhandled exception to the caller.

---

### Requirement 3: Memory Retrieval and Injection

**User Story:** As a user, I want the companion to naturally reference things I've said in the past, so that conversations feel continuous and the AI feels alive.

#### Acceptance Criteria

1. WHEN the `Agent` builds a prompt for the LLM, THE Agent SHALL query the `Memory_Store` for the top-3 `Memory_Entry` items most semantically similar to the current user message, using cosine similarity with a minimum threshold of 0.45.
2. WHEN at least one `Memory_Entry` is returned by the retrieval query, THE Agent SHALL prepend a `[Past context]` block to the LLM system prompt containing the retrieved entries formatted as `"{timestamp_relative}: {memory_text}"`, where `timestamp_relative` is expressed as a human-readable relative duration rounded to the nearest whole unit (e.g., "3 days ago", "1 hour ago"), and entries SHALL be ordered from most recent to least recent by their stored timestamp.
3. WHEN no `Memory_Entry` meets the 0.45 similarity threshold, THE Agent SHALL not inject any past-context block and SHALL proceed with the standard prompt construction.
4. THE Agent SHALL limit the combined character length of all injected `Memory_Entry` texts to 600 characters; if the combined length exceeds 600 characters, THE Agent SHALL drop the lowest-similarity entries one at a time until the combined length is within the 600-character limit, retaining at least the single highest-similarity entry even if its text alone exceeds 600 characters.
5. WHEN the `Memory_Store` file for the `user_id` does not exist or is empty, THE Agent SHALL proceed with zero retrieved memories and SHALL not raise an error.
6. THE `Memory_Entry` data model SHALL guarantee round-trip serialisation fidelity, such that serialising any valid `Memory_Entry` to JSON and deserialising the result produces an object whose every defined field is equal in value and type to the corresponding field of the original entry.

---

### Requirement 4: Cross-Session Mood Arc Tracking

**User Story:** As a companion, I want to track how a user's emotional state evolves across multiple sessions, so that I can calibrate my tone and notice when someone has been persistently struggling.

#### Acceptance Criteria

1. WHEN the `Emotion_Classifier` produces a result for a user turn, THE Memory_Store SHALL append a record `{emotion: <dominant_label>, score: <confidence>, timestamp: <ISO 8601 UTC>}` to the user's `Mood_Profile` stored at `data/moods/{user_id}.json`, creating the file and any missing parent directories if they do not exist; if the file already contains more than 50 records, THE Memory_Store SHALL prune the oldest records on write until exactly 50 remain.
2. WHEN a `Mood_Profile` file is loaded and it contains more than 50 records (e.g., from a previous schema version), THE Memory_Store SHALL silently discard the oldest records until 50 remain before returning the profile.
3. THE Mood_Profile SHALL compute a `dominant_emotion` field as the emotion label with the highest frequency across the last 10 records; IF fewer than 10 records exist, THE Mood_Profile SHALL use all available records; IF two or more labels tie for highest frequency, THE Mood_Profile SHALL select the label whose most recent record has the latest timestamp; IF the profile has no records, `dominant_emotion` SHALL be `null`.
4. THE Mood_Profile SHALL expose a `cross_session_trend` string computed from the last 10 stored emotion labels (or all labels if fewer than 10 exist) using these enumerated rules: if the count of labels in `{"joy", "happiness", "relief", "calm", "hopeful", "confident"}` exceeds the count of labels in `{"sadness", "anger", "fear", "anxiety", "frustration", "despair"}`, the trend is `"improving mood"`; if the negative count exceeds the positive count, the trend is `"struggling emotionally"`; if counts are equal and at least 2 records exist, the trend is `"mixed emotions"`; if fewer than 2 records exist, the trend is `"establishing baseline"`.
5. WHEN a `Mood_Profile` is written to disk and then read back, THE loaded object SHALL produce `dominant_emotion` and `cross_session_trend` values that are equal in value and type to those computed from the in-memory object before the write.

---

### Requirement 5: Adaptive Empathy Mode

**User Story:** As a user who has been feeling down for several messages, I want the companion to notice and shift to a more nurturing tone automatically, so that it feels genuinely caring rather than scripted.

#### Acceptance Criteria

1. WHEN 3 or more consecutive user turns within a single session are classified with a negative emotion (`sadness`, `grief`, `fear`, `nervousness`, `disappointment`, `remorse`), THE Agent SHALL activate `nurturing` Empathy_Mode for that session.
2. WHILE the `Agent` is in `nurturing` Empathy_Mode, THE Agent SHALL append the instruction `"Prioritise emotional validation over advice. Use softer language and avoid unsolicited suggestions."` to the LLM system prompt.
3. WHEN 3 or more consecutive user turns within a single session are classified with a high-energy positive emotion (`joy`, `excitement`, `pride`, `amusement`), THE Agent SHALL activate `energising` Empathy_Mode.
4. WHILE the `Agent` is in `energising` Empathy_Mode, THE Agent SHALL append the instruction `"Match the user's energy. Be playful and celebratory."` to the LLM system prompt.
5. WHEN the `Mood_Profile` for a user shows `cross_session_trend` equal to `"struggling emotionally"` and the current session opens, THE Agent SHALL initialise in `nurturing` Empathy_Mode rather than `default` mode.
6. WHEN a session's consecutive negative-emotion run is broken by a turn classified with a neutral emotion (`neutral`, `realization`, `approval`, `curiosity`) or any positive emotion, THE Agent SHALL revert to `default` Empathy_Mode after exactly 1 such non-negative turn; if the very next turn is again negative, the consecutive-negative counter resets to 1.
7. IF within a single session a run of 3+ consecutive negative turns and a run of 3+ consecutive positive turns both occur (in either order), THE Agent SHALL apply the mode corresponding to the most recently completed run; the earlier run's mode is superseded.
8. THE Empathy_Mode transition function SHALL be deterministic and side-effect-free: given an identical sequence of emotion label inputs, the function SHALL always return the same mode regardless of how many times it is called.

---

### Requirement 6: Memory Management API

**User Story:** As a developer, I want REST endpoints to support the memory dashboard, so that the frontend can list, retrieve, and delete memories without direct file access.

#### Acceptance Criteria

1. THE Server SHALL expose `GET /api/memories/{user_id}` returning a JSON array of all `Memory_Entry` objects for the user, ordered by `timestamp` descending; IF no memories exist for the `user_id`, THE Server SHALL return HTTP 200 with an empty array `[]`.
2. THE Server SHALL expose `DELETE /api/memories/{user_id}/{memory_id}` which removes the `Memory_Entry` with the matching `memory_id` from the `Memory_Store` and returns HTTP 204 on success.
3. IF a `DELETE /api/memories/{user_id}/{memory_id}` request references a `memory_id` that does not exist for the given `user_id`, THEN THE Server SHALL return HTTP 404 with a JSON body describing the not-found error.
4. THE Server SHALL expose `GET /api/mood/{user_id}` returning the full `Mood_Profile` JSON for the user, including `dominant_emotion`, `cross_session_trend`, and the last 10 emotion records ordered by `timestamp` descending.
5. IF a `GET /api/mood/{user_id}` request is made for a `user_id` with no stored `Mood_Profile`, THEN THE Server SHALL return HTTP 200 with a default profile: `{"dominant_emotion": null, "cross_session_trend": "establishing baseline", "records": []}`.
6. THE Server SHALL expose `DELETE /api/memories/{user_id}` (bulk delete) which removes all `Memory_Entry` objects and the `Mood_Profile` for the user, returning HTTP 204 on success; IF no data exists for the `user_id`, THE Server SHALL still return HTTP 204 (idempotent).
7. WHEN any memory management endpoint receives a `user_id` that is an empty string, exceeds 128 characters, or contains path-traversal sequences (`..`, `/`, `\`), THE Server SHALL return HTTP 400 with a JSON body describing the invalid identifier error.
8. WHEN the `Memory_Store` is unavailable due to a file I/O error, THE Server SHALL return HTTP 503 with a JSON body describing the service unavailability, rather than propagating an unhandled exception.

---

### Requirement 7: User Control Dashboard

**User Story:** As a user, I want a dashboard where I can see what the AI remembers about me, understand how it perceives my emotional state, and delete memories I don't want it to keep.

#### Acceptance Criteria

1. THE App SHALL render a Dashboard page accessible via a settings icon in the `ChatView` header, navigating to `/dashboard` using client-side routing without a full page reload; IF no `aitalk_user_id` exists in `localStorage`, THE App SHALL redirect to the Name Entry screen instead of rendering the Dashboard.
2. WHEN the Dashboard loads, THE Dashboard SHALL fetch and display the user's `Mood_Profile` summary as a human-readable sentence (e.g., "Your companion has noticed you've mostly been feeling curious lately 🙂"); IF the `Mood_Profile` has no records, THE Dashboard SHALL display "Not enough data yet — keep chatting!" instead.
3. WHEN the Dashboard loads, THE Dashboard SHALL fetch and display all `Memory_Entry` objects as a scrollable list, showing for each entry: the memory text truncated to 120 characters with a trailing ellipsis if truncated, the relative timestamp expressed as "X minutes/hours/days ago", and the associated emotion label.
4. WHEN the user clicks the delete icon on a `Memory_Entry` row, THE Dashboard SHALL call `DELETE /api/memories/{user_id}/{memory_id}`, remove the entry from the displayed list on a 204 response, and show an inline error message directly below the affected row on failure without reloading the page or removing the entry from the list.
5. WHEN the user clicks "Clear All Memories", THE Dashboard SHALL display a confirmation modal before proceeding; IF the user confirms and THE Server returns HTTP 204, THEN THE Dashboard SHALL use client-side routing to redirect to the persona selection screen.
6. THE Dashboard SHALL display a loading skeleton while memory data is being fetched and SHALL display a non-blocking error banner at the top of the page if any fetch returns a non-2xx response or a network error, without crashing the page.
7. WHILE the Dashboard is open, THE App SHALL display a "Back to Chat" button that uses client-side routing to return to the `ChatView`, restoring the last active persona and the in-session message history that was visible before navigating to the Dashboard.

---

### Requirement 8: Mood Transparency Summary

**User Story:** As a user, I want the AI to occasionally surface a brief insight about my emotional patterns in the chat itself, so that I feel seen and understood over time.

#### Acceptance Criteria

1. WHEN a new session starts and the user's `Mood_Profile` contains records from at least 2 distinct prior `session_id` values, THE Agent SHALL include a single low-key acknowledgement in the first companion reply of the session if `cross_session_trend` is `"struggling emotionally"` (e.g., "You've had a rough few days — I noticed.").
2. IF a cross-session mood acknowledgement has already been included in the current session, THEN THE Agent SHALL not include any further mood acknowledgements for the remainder of that session.
3. WHEN the `cross_session_trend` is `"improving mood"` and the user's `Mood_Profile` contains records from at least 2 distinct prior `session_id` values, THE Agent SHALL include a brief positive acknowledgement in the first reply (e.g., "You seem to be doing better lately — good to see.").
4. THE acknowledgement text SHALL be passed through the existing `Humaniser` style-transfer pipeline, SHALL conform to the active persona's voice, and SHALL NOT contain any of the following clinical or meta terms: "emotion tracking", "mood data", "emotional analysis", "detected", "classified", "pattern", "algorithm", "model".
5. WHEN `cross_session_trend` is `"mixed emotions"` or `"establishing baseline"`, THE Agent SHALL not include any cross-session acknowledgement.

---

### Requirement 9: Optional Voice Mode — TTS Integration

**User Story:** As a user, I want to hear the companion's reply spoken aloud, so that the interaction feels more personal and immersive.

#### Acceptance Criteria

1. WHERE voice mode is enabled by the user via a toggle in the `ChatView` header, THE App SHALL send a `GET /api/tts` request after receiving each companion reply, passing `text`, `persona_key`, and `emotion` as query parameters.
2. THE TTS_Service SHALL call either the ElevenLabs API or the OpenAI TTS API based on the value of the `TTS_PROVIDER` environment variable (`elevenlabs` or `openai`); if the variable is absent, THE TTS_Service SHALL default to `openai`.
3. WHEN the `TTS_Service` receives a reply text, THE TTS_Service SHALL look up the `Voice_Profile` for the active `persona_key` and `emotion` combination; IF no exact emotion match exists in the profile, THE TTS_Service SHALL fall back to the `default` entry for that persona.
4. THE TTS_Service SHALL return an audio stream (MP3) as the HTTP response body with `Content-Type: audio/mpeg`.
5. WHEN the App receives the MP3 audio stream, THE App SHALL begin playback within ≤ 1 second of receiving the response using the browser's `<audio>` element without saving the file to disk.
6. IF the `TTS_Service` call fails (network error, API quota exceeded, invalid API key), THEN THE App SHALL silently fall back to text-only display, log the error to the browser console at the `error` level, and not show a modal, toast, or any other disruptive UI element to the user.
7. WHEN voice mode is toggled off by the user, THE App SHALL stop any currently playing audio within ≤ 500 ms, cancel any in-flight `GET /api/tts` request, and not make further `GET /api/tts` requests until the toggle is re-enabled.
8. THE `Voice_Profile` for `emotion` values in `["sadness", "grief", "fear", "nervousness"]` SHALL use a speech speed modifier of ≤ 0.90× relative to the persona's `default` speed.
9. THE `Voice_Profile` for `emotion` values in `["joy", "excitement", "amusement", "pride"]` SHALL use a speech speed modifier of ≥ 1.05× relative to the persona's `default` speed.

---

### Requirement 10: TTS Voice Profile Configuration

**User Story:** As a developer, I want each persona to have a configurable voice mapping, so that I can adjust voices without changing application code.

#### Acceptance Criteria

1. THE TTS_Service SHALL load voice profiles from a `config/voice_profiles.json` file at startup; WHEN the file is absent, THE TTS_Service SHALL use a built-in default profile for every persona defined in `personalities.py`, with `speed` = 1.0 and a provider-appropriate default `voice_id`.
2. THE `voice_profiles.json` file SHALL follow the schema: `{ "<persona_key>": { "default": { "voice_id": <non-empty string>, "speed": <float 0.5–2.0> }, "sad": { ... }, "happy": { ... } } }`; WHEN the file is present but violates this schema for a given persona entry, THE TTS_Service SHALL log a warning identifying the invalid entry and fall back to the built-in default for that persona only.
3. WHEN `voice_profiles.json` is edited and the server is restarted, THE TTS_Service SHALL load the updated profiles without requiring a code change.
4. FOR ALL valid `voice_profiles.json` files, parsing the file and then re-serialising to JSON and parsing again SHALL produce an object whose float values are equal within a tolerance of 1e-9 and whose string values are exactly equal to the original.

---

### Requirement 11: Anti-Regression — Phase 1 Pipeline Preservation

**User Story:** As a developer, I want Phase 2 additions to compose with Phase 1 features without breaking them, so that existing behaviour (emotion detection, style-transfer, anti-AI-reveal) is preserved.

#### Acceptance Criteria

1. WHEN memory retrieval or empathy-mode logic is added to the `Agent.reply()` method, THE Agent SHALL still invoke `EmotionClassifier.classify()` on every user turn and store the resulting emotion labels in the in-session `Memory` object, as it did in Phase 1.
2. WHEN memory retrieval or empathy-mode logic is added to the `Agent.reply()` method, THE Agent SHALL still pass the final LLM output through `Humaniser.text_style_transfer()` before returning the reply to the caller.
3. WHEN `Humaniser.detect_ai_acceptance()` returns `True`, THE Agent SHALL still remove the triggering user turn from `self.memory.turns` as it did in Phase 1; the same turn SHALL also be excluded from any `Memory_Store` persistence call for that session.
4. WHILE the `Memory_Store` is unavailable (file I/O error, corrupt JSON), THE Agent SHALL log a warning identifying the `user_id` and the error, continue the conversation using only the in-session `Memory` object, and return a reply to the user normally without surfacing the error in the chat.
5. THE Server SHALL continue to respond to `POST /api/message` within 4 seconds (p95) under single-user sequential load on Phase 1 development hardware, measured from request receipt to response sent, including memory retrieval and empathy-mode calculation overhead.

---

### Requirement 12: Memory Entry Parser and Serialiser

**User Story:** As a developer, I want a well-defined serialisation format for memory entries, so that data can be safely written, read, and migrated without corruption.

#### Acceptance Criteria

1. THE Memory_Store SHALL serialise every `Memory_Entry` to a valid JSON object containing exactly the fields in this order: `memory_id` (UUID v4 string), `user_id` (non-empty string), `persona_key` (non-empty string), `session_id` (non-empty string), `timestamp` (ISO 8601 UTC string ending in `Z`), `emotion_labels` (array of zero or more strings), `text` (string of 1–2000 characters).
2. WHEN the `Memory_Store` reads a JSON file containing `Memory_Entry` objects, THE Memory_Store SHALL validate that each object contains all required fields and that `timestamp` is a parseable ISO 8601 string; IF validation fails for an entry, THEN THE Memory_Store SHALL skip that entry, log a warning at the `warning` level identifying the entry by its `memory_id` field if present or by its array index if not, and continue loading the remaining entries.
3. THE Memory_Store SHALL expose a `pretty_print(entry: Memory_Entry) -> str` method that formats a `Memory_Entry` as a human-readable JSON string with 2-space indentation, with fields serialised in the order defined in criterion 1.
4. FOR ALL valid `Memory_Entry` objects `e`, `deserialise(serialise(e))` SHALL produce an object whose every field is equal in value and type to the corresponding field of `e`.
5. FOR ALL valid `Memory_Entry` JSON strings `s`, `serialise(deserialise(s))` SHALL produce a JSON string that when deserialised produces an object equal in value and type to `deserialise(s)`.
6. IF a `Memory_Entry` JSON object is missing one or more required fields, THEN THE Memory_Store SHALL raise a `ValueError` whose message lists all missing field names together, comma-separated, in a single error.
