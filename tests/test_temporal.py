"""Tests for temporal knowledge graph."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from memvid_mcp.temporal import TemporalGraph


@pytest.fixture
def temp_graph():
    """Create a temporary graph for testing."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        graph = TemporalGraph(Path(f.name))
        yield graph


class TestInsertFact:
    """Test fact insertion."""

    def test_insert_basic_fact(self, temp_graph):
        """Should insert a basic fact."""
        fact_id = temp_graph.insert_fact("Alice", "works_at", "Google")
        assert fact_id is not None
        assert len(fact_id) > 0

    def test_insert_with_valid_from(self, temp_graph):
        """Should insert fact with specific valid_from."""
        fact_id = temp_graph.insert_fact(
            "Alice", "works_at", "Google", valid_from="2020-01-01"
        )
        assert fact_id is not None
        facts = temp_graph.query_at_time(subject="Alice")
        assert len(facts) == 1
        assert facts[0]["object"] == "Google"

    def test_auto_close_conflicting_fact(self, temp_graph):
        """New fact should close conflicting old fact."""
        # Insert first fact
        temp_graph.insert_fact("Alice", "works_at", "Google", valid_from="2020-01-01")
        # Insert conflicting fact
        temp_graph.insert_fact("Alice", "works_at", "Meta", valid_from="2024-01-01")

        # Query at 2022 should return Google
        facts_2022 = temp_graph.query_at_time(
            subject="Alice", predicate="works_at", at="2022-06-01"
        )
        assert len(facts_2022) == 1
        assert facts_2022[0]["object"] == "Google"

        # Query now should return Meta
        facts_now = temp_graph.query_at_time(subject="Alice", predicate="works_at")
        assert len(facts_now) == 1
        assert facts_now[0]["object"] == "Meta"


class TestQueryAtTime:
    """Test point-in-time queries."""

    def test_query_current_facts(self, temp_graph):
        """Should return currently valid facts."""
        temp_graph.insert_fact("Bob", "lives_in", "NYC")
        facts = temp_graph.query_at_time(subject="Bob")
        assert len(facts) == 1
        assert facts[0]["object"] == "NYC"

    def test_query_with_predicate_filter(self, temp_graph):
        """Should filter by predicate."""
        temp_graph.insert_fact("Bob", "lives_in", "NYC")
        temp_graph.insert_fact("Bob", "works_at", "Startup")

        facts = temp_graph.query_at_time(subject="Bob", predicate="lives_in")
        assert len(facts) == 1
        assert facts[0]["predicate"] == "lives_in"

    def test_query_excludes_closed_facts(self, temp_graph):
        """Should exclude facts that are no longer valid."""
        fact_id = temp_graph.insert_fact("Carol", "status", "active")
        temp_graph.invalidate_fact(fact_id)

        facts = temp_graph.query_at_time(subject="Carol")
        assert len(facts) == 0

    def test_query_respects_confidence_threshold(self, temp_graph):
        """Should filter by minimum confidence."""
        temp_graph.insert_fact("Dave", "skill", "Python", confidence=0.05)
        temp_graph.insert_fact("Dave", "skill", "Java", confidence=0.5)

        facts = temp_graph.query_at_time(subject="Dave", min_confidence=0.1)
        assert len(facts) == 1
        assert facts[0]["object"] == "Java"


class TestGetTimeline:
    """Test timeline retrieval."""

    def test_timeline_chronological_order(self, temp_graph):
        """Timeline should be in chronological order."""
        temp_graph.insert_fact("Eve", "role", "Junior", valid_from="2020-01-01")
        temp_graph.insert_fact("Eve", "role", "Senior", valid_from="2022-01-01")
        temp_graph.insert_fact("Eve", "role", "Lead", valid_from="2024-01-01")

        timeline = temp_graph.get_timeline("Eve", "role")
        assert len(timeline) == 3
        assert timeline[0]["object"] == "Junior"
        assert timeline[1]["object"] == "Senior"
        assert timeline[2]["object"] == "Lead"


class TestConfidenceDecay:
    """Test confidence decay mechanism."""

    def test_decay_reduces_confidence(self, temp_graph):
        """Decay should reduce confidence over time."""
        # Insert fact with old valid_from
        old_date = datetime.now() - timedelta(days=100)
        temp_graph.insert_fact(
            "Frank",
            "preference",
            "dark_mode",
            valid_from=old_date.isoformat(),
            confidence=1.0,
        )

        # Apply decay
        updated = temp_graph.apply_confidence_decay(decay_rate=0.01)
        assert updated >= 1

        # Check confidence decreased
        facts = temp_graph.query_at_time(subject="Frank", min_confidence=0.0)
        assert len(facts) == 1
        assert facts[0]["confidence"] < 1.0

    def test_decay_respects_minimum(self, temp_graph):
        """Confidence should not go below 0.1."""
        very_old = datetime.now() - timedelta(days=1000)
        temp_graph.insert_fact(
            "Grace",
            "old_fact",
            "value",
            valid_from=very_old.isoformat(),
            confidence=0.2,
        )

        # Apply aggressive decay multiple times
        for _ in range(10):
            temp_graph.apply_confidence_decay(decay_rate=0.1)

        facts = temp_graph.query_at_time(subject="Grace", min_confidence=0.0)
        assert len(facts) == 1
        assert facts[0]["confidence"] >= 0.1


class TestStats:
    """Test statistics."""

    def test_stats_counts(self, temp_graph):
        """Stats should count facts correctly."""
        temp_graph.insert_fact("A", "rel", "B")
        temp_graph.insert_fact("C", "rel", "D")
        fact_id = temp_graph.insert_fact("E", "rel", "F")
        temp_graph.invalidate_fact(fact_id)

        stats = temp_graph.stats()
        assert stats["total_facts"] == 3
        assert stats["active_facts"] == 2
        assert stats["closed_facts"] == 1
