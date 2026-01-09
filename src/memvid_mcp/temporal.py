"""
Temporal knowledge graph for time-versioned facts.

Stores facts with validity periods (valid_from/valid_to) enabling:
- Point-in-time queries: "What was true at time X?"
- Automatic fact evolution: New facts close old ones
- Confidence decay over time
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class TemporalGraph:
    """
    Time-aware fact storage with validity windows.

    Facts are stored as (subject, predicate, object) triples with:
    - valid_from: When the fact became true
    - valid_to: When the fact stopped being true (None = still valid)
    - confidence: Certainty level (decays over time)
    """

    def __init__(self, data_path: Path):
        """
        Initialize temporal graph.

        Args:
            data_path: Path to JSON file for persistence
        """
        self._path = data_path
        self._facts: dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load facts from disk."""
        if self._path.exists() and self._path.stat().st_size > 0:
            with open(self._path) as f:
                self._facts = json.load(f)

    def _save(self):
        """Save facts to disk."""
        with open(self._path, "w") as f:
            json.dump(self._facts, f, indent=2, default=str)

    def _now_ts(self) -> int:
        """Current timestamp in milliseconds."""
        return int(datetime.now().timestamp() * 1000)

    def _parse_time(self, t: str | int | datetime | None) -> int:
        """Convert various time formats to milliseconds timestamp."""
        if t is None:
            return self._now_ts()
        if isinstance(t, int):
            return t
        if isinstance(t, datetime):
            return int(t.timestamp() * 1000)
        if isinstance(t, str):
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        return self._now_ts()

    def insert_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str | int | datetime = None,
        confidence: float = 1.0,
        metadata: dict = None,
    ) -> str:
        """
        Insert a new fact, automatically closing conflicting old facts.

        When inserting (Alice, works_at, Meta), any existing
        (Alice, works_at, *) facts are closed with valid_to set.

        Args:
            subject: Entity the fact is about
            predicate: Relationship type
            obj: Value/target of the relationship
            valid_from: When fact became true (default: now)
            confidence: Certainty level 0-1
            metadata: Additional data

        Returns:
            Fact ID
        """
        fact_id = str(uuid.uuid4())
        valid_from_ts = self._parse_time(valid_from)
        now = self._now_ts()

        # Close existing facts with same subject+predicate
        for fid, fact in self._facts.items():
            if (
                fact["subject"] == subject
                and fact["predicate"] == predicate
                and fact["valid_to"] is None
                and fact["valid_from"] < valid_from_ts
            ):
                # Close old fact 1ms before new one starts
                self._facts[fid]["valid_to"] = valid_from_ts - 1
                self._facts[fid]["last_updated"] = now

        # Insert new fact
        self._facts[fact_id] = {
            "id": fact_id,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "valid_from": valid_from_ts,
            "valid_to": None,  # Currently valid
            "confidence": confidence,
            "last_updated": now,
            "metadata": metadata or {},
        }

        self._save()
        return fact_id

    def query_at_time(
        self,
        subject: str = None,
        predicate: str = None,
        obj: str = None,
        at: str | int | datetime = None,
        min_confidence: float = 0.1,
    ) -> list[dict[str, Any]]:
        """
        Query facts valid at a specific point in time.

        Args:
            subject: Filter by subject (optional)
            predicate: Filter by predicate (optional)
            obj: Filter by object (optional)
            at: Point in time to query (default: now)
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching facts
        """
        ts = self._parse_time(at)
        results = []

        for fact in self._facts.values():
            # Check time validity: valid_from <= ts AND (valid_to IS NULL OR valid_to >= ts)
            if fact["valid_from"] > ts:
                continue
            if fact["valid_to"] is not None and fact["valid_to"] <= ts:
                continue

            # Check confidence
            if fact["confidence"] < min_confidence:
                continue

            # Apply filters
            if subject and fact["subject"] != subject:
                continue
            if predicate and fact["predicate"] != predicate:
                continue
            if obj and fact["object"] != obj:
                continue

            results.append(fact.copy())

        return results

    def get_timeline(self, subject: str, predicate: str = None) -> list[dict[str, Any]]:
        """
        Get chronological history of facts for a subject.

        Args:
            subject: Entity to get timeline for
            predicate: Optional predicate filter

        Returns:
            Facts sorted by valid_from ascending
        """
        facts = []
        for fact in self._facts.values():
            if fact["subject"] != subject:
                continue
            if predicate and fact["predicate"] != predicate:
                continue
            facts.append(fact.copy())

        facts.sort(key=lambda f: f["valid_from"])
        return facts

    def invalidate_fact(self, fact_id: str, valid_to: str | int | datetime = None):
        """
        Mark a fact as no longer valid.

        Args:
            fact_id: ID of fact to invalidate
            valid_to: When it stopped being true (default: now)
        """
        if fact_id not in self._facts:
            return

        ts = self._parse_time(valid_to)
        self._facts[fact_id]["valid_to"] = ts
        self._facts[fact_id]["last_updated"] = self._now_ts()
        self._save()

    def delete_fact(self, fact_id: str) -> bool:
        """
        Permanently delete a fact.

        Args:
            fact_id: ID of fact to delete

        Returns:
            True if deleted
        """
        if fact_id in self._facts:
            del self._facts[fact_id]
            self._save()
            return True
        return False

    def apply_confidence_decay(self, decay_rate: float = 0.01) -> int:
        """
        Apply time-based confidence decay to all active facts.

        Formula: confidence = max(0.1, confidence * (1 - decay_rate * days_since_creation))

        Args:
            decay_rate: Decay rate per day

        Returns:
            Number of facts updated
        """
        now = self._now_ts()
        one_day_ms = 86400000
        updated = 0

        for fact in self._facts.values():
            if fact["valid_to"] is not None:
                continue  # Skip closed facts
            if fact["confidence"] <= 0.1:
                continue  # Already at minimum

            days = (now - fact["valid_from"]) / one_day_ms
            decay_factor = 1 - decay_rate * days
            new_confidence = max(0.1, fact["confidence"] * decay_factor)

            if new_confidence != fact["confidence"]:
                fact["confidence"] = new_confidence
                fact["last_updated"] = now
                updated += 1

        if updated > 0:
            self._save()

        return updated

    def stats(self) -> dict[str, Any]:
        """Get statistics about the temporal graph."""
        total = len(self._facts)
        active = sum(1 for f in self._facts.values() if f["valid_to"] is None)
        subjects = len(set(f["subject"] for f in self._facts.values()))
        predicates = len(set(f["predicate"] for f in self._facts.values()))

        return {
            "total_facts": total,
            "active_facts": active,
            "closed_facts": total - active,
            "unique_subjects": subjects,
            "unique_predicates": predicates,
        }
