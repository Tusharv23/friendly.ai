"""
Tier 2 Memory — Biography Store
---------------------------------
Vector-indexed biographical facts about the user.  Written by the
AnalysisEngine at session end (or every 100 messages), retrieved at
prompt-build time for deep personalisation.

Storage layout:
    data/biography/{user_id}.json        ← BiographyEntry metadata (JSON)
    data/biography/{user_id}_vecs.json   ← LocalVectorDB embeddings

Entry types:
    fact        — confirmed truth about the user ("works at a startup")
    pattern     — recurring behaviour ("goes quiet before big decisions")
    prediction  — what we expect next, with confidence score
    phase       — current emotional arc stage ("bargaining phase of breakup")
    crux        — 2-3 sentence summary of the last session

Entry schema:
    fact_id         : str   — UUID v4
    user_id         : str
    type            : str   — one of the five types above
    text            : str   — human-readable statement
    confidence      : float — 0.0–1.0
    source_sessions : list  — session_ids that contributed to this entry
    created_at      : str   — ISO 8601 UTC
    updated_at      : str   — ISO 8601 UTC
    verified        : bool  — prediction confirmed true
    verified_at     : str | None
    defied_by       : str | None  — context that contradicted a prediction
    active          : bool  — False = soft-deleted / stale
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from local_vectorDB import LocalVectorDB

logger = logging.getLogger(__name__)

VALID_TYPES = {"fact", "pattern", "prediction", "phase", "crux"}
_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "biography")


class BiographyStore:
    """
    Per-user biographical fact store backed by LocalVectorDB for semantic
    retrieval and a JSON metadata file for structured access.

    Usage:
        store = BiographyStore("tushar")
        store.upsert({"type": "fact", "text": "Works at a startup", "confidence": 0.9})
        results = store.search("career stress", top_k=3)
    """

    def __init__(self, user_id: str) -> None:
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        self.user_id = user_id.strip().lower()
        os.makedirs(_BASE_DIR, exist_ok=True)
        self._meta_path = os.path.join(_BASE_DIR, f"{self.user_id}.json")
        self._vec_path  = os.path.join(_BASE_DIR, f"{self.user_id}_vecs.json")
        self._entries: dict[str, dict] = self._load_metadata()
        self._vdb = self._load_vectors()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _load_metadata(self) -> dict[str, dict]:
        if not os.path.exists(self._meta_path):
            return {}
        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Keyed by fact_id
            return {e["fact_id"]: e for e in data if "fact_id" in e}
        except Exception as exc:
            logger.warning("BiographyStore: failed to load metadata for %r: %s", self.user_id, exc)
            return {}

    def _save_metadata(self) -> None:
        try:
            with open(self._meta_path, "w", encoding="utf-8") as f:
                json.dump(list(self._entries.values()), f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("BiographyStore: failed to save metadata for %r: %s", self.user_id, exc)

    def _load_vectors(self) -> LocalVectorDB:
        vdb = LocalVectorDB()
        if os.path.exists(self._vec_path):
            try:
                vdb.load(self._vec_path)
            except Exception as exc:
                logger.warning("BiographyStore: failed to load vectors for %r: %s", self.user_id, exc)
        return vdb

    def _save_vectors(self) -> None:
        try:
            self._vdb.save(self._vec_path)
        except Exception as exc:
            logger.warning("BiographyStore: failed to save vectors for %r: %s", self.user_id, exc)

    def _build_entry(self, data: dict) -> dict:
        """Validate and fill defaults for a new entry."""
        entry_type = data.get("type", "fact")
        if entry_type not in VALID_TYPES:
            raise ValueError(f"Invalid entry type {entry_type!r}. Must be one of {VALID_TYPES}")
        text = data.get("text", "").strip()
        if not text:
            raise ValueError("BiographyEntry text must be a non-empty string")

        now = self._now_utc()
        return {
            "fact_id":         data.get("fact_id") or str(uuid.uuid4()),
            "user_id":         self.user_id,
            "type":            entry_type,
            "text":            text,
            "confidence":      max(0.0, min(1.0, float(data.get("confidence", 0.7)))),
            "source_sessions": data.get("source_sessions", []),
            "created_at":      data.get("created_at", now),
            "updated_at":      now,
            "verified":        bool(data.get("verified", False)),
            "verified_at":     data.get("verified_at"),
            "defied_by":       data.get("defied_by"),
            "active":          bool(data.get("active", True)),
        }

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, data: dict) -> str:
        """
        Add a new entry or update an existing one (matched by fact_id if
        provided, otherwise always inserts).  Returns the fact_id.

        Never raises — logs warnings on failure.
        """
        try:
            entry = self._build_entry(data)
            fact_id = entry["fact_id"]

            if fact_id in self._entries:
                # Update: preserve created_at, bump updated_at + confidence
                existing = self._entries[fact_id]
                existing.update({
                    "text":            entry["text"],
                    "confidence":      entry["confidence"],
                    "source_sessions": list(set(
                        existing.get("source_sessions", []) + entry["source_sessions"]
                    )),
                    "updated_at":      entry["updated_at"],
                    "active":          entry["active"],
                })
                # Re-embed with updated text
                self._vdb.add_texts([entry["text"]], [{"fact_id": fact_id}])
            else:
                # Insert
                self._entries[fact_id] = entry
                self._vdb.add_texts([entry["text"]], [{"fact_id": fact_id}])

            self._save_metadata()
            self._save_vectors()
            return fact_id

        except Exception as exc:
            logger.warning("BiographyStore.upsert failed for user=%r: %s", self.user_id, exc)
            return ""

    def mark_verified(self, fact_id: str) -> None:
        """Mark a prediction as confirmed true. Boosts confidence to 1.0."""
        entry = self._entries.get(fact_id)
        if not entry:
            logger.warning("BiographyStore.mark_verified: fact_id %r not found", fact_id)
            return
        entry.update({
            "verified":    True,
            "verified_at": self._now_utc(),
            "confidence":  1.0,
            "updated_at":  self._now_utc(),
        })
        self._save_metadata()

    def mark_defied(self, fact_id: str, context: str) -> None:
        """
        Mark a prediction as wrong.  Records what contradicted it,
        lowers confidence, but keeps the entry active for analysis.
        """
        entry = self._entries.get(fact_id)
        if not entry:
            logger.warning("BiographyStore.mark_defied: fact_id %r not found", fact_id)
            return
        entry.update({
            "defied_by":  context,
            "confidence": max(0.0, entry.get("confidence", 0.5) - 0.3),
            "updated_at": self._now_utc(),
        })
        self._save_metadata()

    def deactivate(self, fact_id: str) -> None:
        """Soft-delete a stale entry."""
        entry = self._entries.get(fact_id)
        if entry:
            entry["active"] = False
            entry["updated_at"] = self._now_utc()
            self._save_metadata()

    def update_confidence(self, fact_id: str, delta: float) -> None:
        """Nudge confidence up or down without fully verifying/defying."""
        entry = self._entries.get(fact_id)
        if entry:
            entry["confidence"] = max(0.0, min(1.0, entry.get("confidence", 0.5) + delta))
            entry["updated_at"] = self._now_utc()
            self._save_metadata()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        types: Optional[list[str]] = None,
        min_confidence: float = 0.3,
    ) -> list[dict]:
        """
        Return up to top_k active entries most semantically similar to query.
        Optionally filter by entry type(s) and minimum confidence.
        Results are sorted by similarity score descending.
        """
        try:
            raw = self._vdb.query(query, top_k=top_k * 3, threshold=0.3)
            results = []
            for r in raw:
                fact_id = r.get("metadata", {}).get("fact_id")
                if not fact_id:
                    continue
                entry = self._entries.get(fact_id)
                if not entry:
                    continue
                if not entry.get("active", True):
                    continue
                if entry.get("confidence", 0) < min_confidence:
                    continue
                if types and entry.get("type") not in types:
                    continue
                results.append({**entry, "_score": r["score"]})
                if len(results) >= top_k:
                    break
            return results
        except Exception as exc:
            logger.warning("BiographyStore.search failed for user=%r: %s", self.user_id, exc)
            return []

    def get_crux(self) -> Optional[dict]:
        """Return the most recently created active crux entry."""
        cruxes = [
            e for e in self._entries.values()
            if e.get("type") == "crux" and e.get("active", True)
        ]
        if not cruxes:
            return None
        return max(cruxes, key=lambda e: e.get("created_at", ""))

    def get_active_predictions(self) -> list[dict]:
        """Return all active, unverified predictions sorted by confidence desc."""
        preds = [
            e for e in self._entries.values()
            if e.get("type") == "prediction"
            and e.get("active", True)
            and not e.get("verified", False)
            and not e.get("defied_by")
        ]
        return sorted(preds, key=lambda e: e.get("confidence", 0), reverse=True)

    def get_active_phases(self) -> list[dict]:
        """Return all active emotional phase entries."""
        return [
            e for e in self._entries.values()
            if e.get("type") == "phase" and e.get("active", True)
        ]

    def get_all(self, entry_type: Optional[str] = None, active_only: bool = True) -> list[dict]:
        """
        Return all entries, optionally filtered by type and active status,
        sorted by created_at descending.
        """
        entries = list(self._entries.values())
        if active_only:
            entries = [e for e in entries if e.get("active", True)]
        if entry_type:
            entries = [e for e in entries if e.get("type") == entry_type]
        return sorted(entries, key=lambda e: e.get("created_at", ""), reverse=True)

    def get_by_id(self, fact_id: str) -> Optional[dict]:
        return self._entries.get(fact_id)

    # ------------------------------------------------------------------
    # Prompt helper
    # ------------------------------------------------------------------

    def build_context_block(self, query: str, max_chars: int = 800) -> str:
        """
        Build a compact [Biography context] block for LLM prompt injection.
        Pulls: latest crux + semantically relevant facts/patterns/phases.
        Truncates to max_chars.
        """
        lines = []

        # Always lead with the session crux
        crux = self.get_crux()
        if crux:
            lines.append(f"[Last session] {crux['text']}")

        # Active emotional phases
        phases = self.get_active_phases()
        if phases:
            phase_texts = "; ".join(p["text"] for p in phases[:2])
            lines.append(f"[Emotional phase] {phase_texts}")

        # Semantically relevant facts and patterns
        relevant = self.search(query, top_k=4, types=["fact", "pattern"], min_confidence=0.4)
        for r in relevant:
            lines.append(f"[{r['type'].capitalize()}] {r['text']}")

        # High-confidence pending predictions (so AI can watch for them)
        preds = self.get_active_predictions()
        for p in preds[:2]:
            if p["confidence"] >= 0.6:
                lines.append(f"[Prediction] {p['text']} (confidence: {p['confidence']:.0%})")

        if not lines:
            return ""

        block = "[Biography context — use naturally, never reference directly]\n"
        total = len(block)
        kept = []
        for line in lines:
            if total + len(line) + 1 > max_chars:
                break
            kept.append(line)
            total += len(line) + 1

        return block + "\n".join(kept) if kept else ""

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def deduplicate(self, threshold: float = 0.65) -> int:
        """
        Remove semantically duplicate entries of the same type.
        Keeps the higher-confidence entry, deactivates the lower.
        Returns number of entries deactivated.
        """
        deactivated_count = 0
        for entry_type in ["fact", "pattern", "phase", "prediction"]:
            entries = self.get_all(entry_type)
            deactivated: set = set()
            for e1 in entries:
                if e1["fact_id"] in deactivated:
                    continue
                results = self._vdb.query(e1["text"], top_k=10, threshold=0.0)
                for r in results:
                    fid = r.get("metadata", {}).get("fact_id", "")
                    if not fid or fid == e1["fact_id"] or fid in deactivated:
                        continue
                    other = self.get_by_id(fid)
                    if not other or other.get("type") != entry_type or not other.get("active", True):
                        continue
                    if r["score"] >= threshold:
                        drop_id = fid if e1["confidence"] >= other["confidence"] else e1["fact_id"]
                        self.deactivate(drop_id)
                        deactivated.add(drop_id)
                        deactivated_count += 1
        return deactivated_count

    def summary(self) -> dict:
        """Quick stats — useful for debugging and the dashboard."""
        active = [e for e in self._entries.values() if e.get("active", True)]
        by_type: dict[str, int] = {}
        for e in active:
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        return {
            "user_id":     self.user_id,
            "total":       len(active),
            "by_type":     by_type,
            "predictions": len(self.get_active_predictions()),
            "phases":      len(self.get_active_phases()),
        }

    def __repr__(self) -> str:
        return f"BiographyStore(user_id={self.user_id!r}, entries={len(self._entries)})"
