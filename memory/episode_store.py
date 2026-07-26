"""
Tier 1 Memory — Episode Store
------------------------------
Append-only log of every conversation turn, stored per user in a TinyDB
JSON file.  Designed for fast writes (never blocks chat) and structured
reads (session replay, analysis engine input).

Storage layout:
    data/episodes/{user_id}.json   ← one TinyDB file per user

Turn schema (every document in the 'episodes' table):
    session_id     : str   — UUID of the session this turn belongs to
    turn_index     : int   — 0-based position within the session
    timestamp      : str   — ISO 8601 UTC  e.g. "2026-07-25T10:30:00Z"
    role           : str   — "user" | "assistant"
    content        : str   — the message text
    emotion_labels : list  — ["sadness", "fear"] or [] for assistant turns
    persona_key    : str   — active persona key for this session
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from tinydb import TinyDB, Query

logger = logging.getLogger(__name__)

# Path relative to project root
_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "episodes")


class EpisodeStore:
    """
    Per-user append-only episode log backed by TinyDB.

    Usage:
        store = EpisodeStore("tushar")
        store.append(session_id, turn_index, "user", "hey", ["neutral"], "Vicky...")
        turns = store.get_session_turns(session_id)
    """

    def __init__(self, user_id: str) -> None:
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        self.user_id = user_id.strip().lower()
        self._db = self._open_db()
        self._table = self._db.table("episodes")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_db(self) -> TinyDB:
        """Create the data directory if needed and open the TinyDB file."""
        os.makedirs(_BASE_DIR, exist_ok=True)
        path = os.path.join(_BASE_DIR, f"{self.user_id}.json")
        return TinyDB(path)

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(
        self,
        session_id: str,
        turn_index: int,
        role: str,
        content: str,
        emotion_labels: list[str],
        persona_key: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Persist one conversation turn.  Never raises — logs warnings on
        failure so a storage hiccup never crashes the chat.
        """
        if role not in ("user", "assistant"):
            logger.warning("EpisodeStore.append: unexpected role %r — storing anyway", role)

        document = {
            "session_id": session_id,
            "turn_index": turn_index,
            "timestamp": timestamp or self._now_utc(),
            "role": role,
            "content": content,
            "emotion_labels": emotion_labels if emotion_labels is not None else [],
            "persona_key": persona_key,
        }

        try:
            self._table.insert(document)
        except Exception as exc:
            logger.warning(
                "EpisodeStore: failed to write turn for user=%r session=%r turn=%d: %s",
                self.user_id, session_id, turn_index, exc,
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_session_turns(self, session_id: str) -> list[dict]:
        """
        Return all turns for a session, sorted by turn_index ascending.
        Returns an empty list if the session doesn't exist.
        """
        try:
            Episode = Query()
            results = self._table.search(Episode.session_id == session_id)
            return sorted(results, key=lambda t: t.get("turn_index", 0))
        except Exception as exc:
            logger.warning(
                "EpisodeStore: failed to read session %r for user=%r: %s",
                session_id, self.user_id, exc,
            )
            return []

    def get_last_session(self) -> tuple[Optional[str], list[dict]]:
        """
        Return (session_id, turns) of the most recent session.
        Useful at session start to restore context for the user.
        Returns (None, []) if no episodes exist yet.
        """
        try:
            all_turns = self._table.all()
            if not all_turns:
                return None, []

            # Find the session whose latest turn has the most recent timestamp
            sessions: dict[str, str] = {}  # session_id → latest timestamp
            for turn in all_turns:
                sid = turn.get("session_id")
                ts = turn.get("timestamp", "")
                if sid and ts > sessions.get(sid, ""):
                    sessions[sid] = ts

            latest_session_id = max(sessions, key=lambda s: sessions[s])
            turns = self.get_session_turns(latest_session_id)
            return latest_session_id, turns

        except Exception as exc:
            logger.warning(
                "EpisodeStore: failed to fetch last session for user=%r: %s",
                self.user_id, exc,
            )
            return None, []

    def get_message_count(self, session_id: str) -> int:
        """
        Return the number of turns stored for this session.
        Used by the analysis engine to check the 100-message trigger.
        """
        try:
            Episode = Query()
            return self._table.count(Episode.session_id == session_id)
        except Exception as exc:
            logger.warning(
                "EpisodeStore: failed to count messages for session=%r user=%r: %s",
                session_id, self.user_id, exc,
            )
            return 0

    def get_all_sessions(self) -> list[str]:
        """
        Return all distinct session_ids ordered by first-turn timestamp ascending.
        Useful for the analysis engine and biography builder.
        """
        try:
            all_turns = self._table.all()
            if not all_turns:
                return []

            # earliest timestamp seen per session
            first_seen: dict[str, str] = {}
            for turn in all_turns:
                sid = turn.get("session_id")
                ts = turn.get("timestamp", "")
                if sid and (sid not in first_seen or ts < first_seen[sid]):
                    first_seen[sid] = ts

            return sorted(first_seen.keys(), key=lambda s: first_seen[s])

        except Exception as exc:
            logger.warning(
                "EpisodeStore: failed to list sessions for user=%r: %s",
                self.user_id, exc,
            )
            return []

    def get_recent_turns(self, session_id: str, n: int = 10) -> list[dict]:
        """
        Return the last N turns from a session, oldest-first.
        Convenience method for injecting recent context into prompts.
        """
        turns = self.get_session_turns(session_id)
        return turns[-n:] if len(turns) > n else turns

    # ------------------------------------------------------------------
    # Persona state persistence
    # ------------------------------------------------------------------

    def save_persona_state(
        self,
        persona_key: str,
        disclosures: list[str],
        session_ended_at: Optional[str] = None,
    ) -> None:
        """
        Persist the persona's last known self-disclosures and the wall-clock
        time the session ended.  Overrides any previous state for this persona.
        Called by server.py when a new session starts (flushing the old one).
        """
        try:
            state_table = self._db.table("persona_state")
            PS = Query()
            record = {
                "persona_key": persona_key,
                "disclosures": disclosures[-10:],   # last 10 only
                "session_ended_at": session_ended_at or self._now_utc(),
            }
            if state_table.contains(PS.persona_key == persona_key):
                state_table.update(record, PS.persona_key == persona_key)
            else:
                state_table.insert(record)
        except Exception as exc:
            logger.warning(
                "EpisodeStore: failed to save persona state for %r user=%r: %s",
                persona_key, self.user_id, exc,
            )

    def load_persona_state(self, persona_key: str) -> dict:
        """
        Load the last persisted state for a persona.
        Returns {"disclosures": [], "session_ended_at": None} if nothing saved yet.
        """
        try:
            state_table = self._db.table("persona_state")
            PS = Query()
            result = state_table.get(PS.persona_key == persona_key)
            if result:
                return {
                    "disclosures": result.get("disclosures", []),
                    "session_ended_at": result.get("session_ended_at"),
                }
        except Exception as exc:
            logger.warning(
                "EpisodeStore: failed to load persona state for %r user=%r: %s",
                persona_key, self.user_id, exc,
            )
        return {"disclosures": [], "session_ended_at": None}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Explicitly close the TinyDB file handle."""
        try:
            self._db.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"EpisodeStore(user_id={self.user_id!r})"
