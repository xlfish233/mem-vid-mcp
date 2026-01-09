"""Tests for waypoint association graph."""
import pytest
import tempfile
from pathlib import Path

from memvid_mcp.waypoint import WaypointGraph


@pytest.fixture
def temp_graph():
    """Create a temporary graph for testing."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        graph = WaypointGraph(Path(f.name))
        yield graph


class TestCreateWaypoint:
    """Test waypoint creation."""

    def test_create_basic_waypoint(self, temp_graph):
        """Should create a waypoint between two memories."""
        temp_graph.create_waypoint("mem1", "mem2", weight=0.8)
        neighbors = temp_graph.get_neighbors("mem1")
        assert len(neighbors) == 1
        assert neighbors[0]["id"] == "mem2"
        assert neighbors[0]["weight"] == 0.8

    def test_create_bidirectional_waypoint(self, temp_graph):
        """Should create bidirectional waypoint by default."""
        temp_graph.create_waypoint("mem1", "mem2", weight=0.7)

        # Check forward edge
        neighbors1 = temp_graph.get_neighbors("mem1")
        assert len(neighbors1) == 1
        assert neighbors1[0]["id"] == "mem2"

        # Check reverse edge
        neighbors2 = temp_graph.get_neighbors("mem2")
        assert len(neighbors2) == 1
        assert neighbors2[0]["id"] == "mem1"

    def test_create_unidirectional_waypoint(self, temp_graph):
        """Should create unidirectional waypoint when specified."""
        temp_graph.create_waypoint("mem1", "mem2", weight=0.6, bidirectional=False)

        neighbors1 = temp_graph.get_neighbors("mem1")
        assert len(neighbors1) == 1

        neighbors2 = temp_graph.get_neighbors("mem2")
        assert len(neighbors2) == 0

    def test_no_self_waypoint(self, temp_graph):
        """Should not create waypoint to self."""
        temp_graph.create_waypoint("mem1", "mem1", weight=0.9)
        neighbors = temp_graph.get_neighbors("mem1")
        assert len(neighbors) == 0


class TestGetNeighbors:
    """Test neighbor retrieval."""

    def test_neighbors_sorted_by_weight(self, temp_graph):
        """Neighbors should be sorted by weight descending."""
        temp_graph.create_waypoint("mem1", "mem2", weight=0.5, bidirectional=False)
        temp_graph.create_waypoint("mem1", "mem3", weight=0.9, bidirectional=False)
        temp_graph.create_waypoint("mem1", "mem4", weight=0.7, bidirectional=False)

        neighbors = temp_graph.get_neighbors("mem1")
        assert len(neighbors) == 3
        assert neighbors[0]["id"] == "mem3"  # Highest weight
        assert neighbors[1]["id"] == "mem4"
        assert neighbors[2]["id"] == "mem2"  # Lowest weight

    def test_empty_neighbors(self, temp_graph):
        """Should return empty list for unknown memory."""
        neighbors = temp_graph.get_neighbors("unknown")
        assert neighbors == []


class TestExpand:
    """Test graph expansion."""

    def test_expand_finds_connected_nodes(self, temp_graph):
        """Should find nodes connected via waypoints."""
        # Create chain: mem1 -> mem2 -> mem3
        temp_graph.create_waypoint("mem1", "mem2", weight=0.8)
        temp_graph.create_waypoint("mem2", "mem3", weight=0.8)

        expanded = temp_graph.expand(["mem1"], max_expansion=10)

        # Should find mem2 and mem3
        expanded_ids = [e["id"] for e in expanded]
        assert "mem2" in expanded_ids
        assert "mem3" in expanded_ids

    def test_expand_weight_decay(self, temp_graph):
        """Expanded nodes should have decayed weights."""
        temp_graph.create_waypoint("mem1", "mem2", weight=0.8)
        temp_graph.create_waypoint("mem2", "mem3", weight=0.8)

        expanded = temp_graph.expand(["mem1"], max_expansion=10)

        # mem2 weight = 1.0 * 0.8 * 0.8 = 0.64
        mem2 = next(e for e in expanded if e["id"] == "mem2")
        assert abs(mem2["weight"] - 0.64) < 0.01

        # mem3 weight = 0.64 * 0.8 * 0.8 = 0.4096
        mem3 = next(e for e in expanded if e["id"] == "mem3")
        assert abs(mem3["weight"] - 0.4096) < 0.01

    def test_expand_respects_limit(self, temp_graph):
        """Should respect max_expansion limit."""
        # Create many connections
        for i in range(20):
            temp_graph.create_waypoint("mem1", f"mem{i+2}", weight=0.9, bidirectional=False)

        expanded = temp_graph.expand(["mem1"], max_expansion=5)
        assert len(expanded) <= 5

    def test_expand_prunes_low_weight(self, temp_graph):
        """Should prune paths with low accumulated weight."""
        # Create chain with low weights
        temp_graph.create_waypoint("mem1", "mem2", weight=0.3)
        temp_graph.create_waypoint("mem2", "mem3", weight=0.3)

        expanded = temp_graph.expand(["mem1"], max_expansion=10, min_weight=0.1)

        # mem3 weight would be 1.0 * 0.3 * 0.8 * 0.3 * 0.8 = 0.0576 < 0.1
        expanded_ids = [e["id"] for e in expanded]
        assert "mem3" not in expanded_ids

    def test_expand_includes_path(self, temp_graph):
        """Expanded nodes should include traversal path."""
        temp_graph.create_waypoint("mem1", "mem2", weight=0.8)
        temp_graph.create_waypoint("mem2", "mem3", weight=0.8)

        expanded = temp_graph.expand(["mem1"], max_expansion=10)

        mem3 = next(e for e in expanded if e["id"] == "mem3")
        assert mem3["path"] == ["mem1", "mem2", "mem3"]


class TestReinforce:
    """Test waypoint reinforcement."""

    def test_reinforce_increases_weight(self, temp_graph):
        """Reinforcement should increase waypoint weight."""
        temp_graph.create_waypoint("mem1", "mem2", weight=0.5)

        initial_weight = temp_graph.get_neighbors("mem1")[0]["weight"]
        temp_graph.reinforce(["mem1", "mem2"])
        new_weight = temp_graph.get_neighbors("mem1")[0]["weight"]

        assert new_weight > initial_weight

    def test_reinforce_respects_max_weight(self, temp_graph):
        """Weight should not exceed maximum."""
        temp_graph.create_waypoint("mem1", "mem2", weight=0.95)

        # Reinforce many times
        for _ in range(20):
            temp_graph.reinforce(["mem1", "mem2"])

        weight = temp_graph.get_neighbors("mem1")[0]["weight"]
        assert weight <= WaypointGraph.MAX_WEIGHT


class TestRemoveMemory:
    """Test memory removal."""

    def test_remove_clears_all_edges(self, temp_graph):
        """Removing memory should clear all its edges."""
        temp_graph.create_waypoint("mem1", "mem2", weight=0.8)
        temp_graph.create_waypoint("mem1", "mem3", weight=0.7)
        temp_graph.create_waypoint("mem4", "mem1", weight=0.6)

        temp_graph.remove_memory("mem1")

        # No outgoing edges from mem1
        assert temp_graph.get_neighbors("mem1") == []

        # No incoming edges to mem1
        assert "mem1" not in [n["id"] for n in temp_graph.get_neighbors("mem4")]


class TestPruneWeakEdges:
    """Test weak edge pruning."""

    def test_prune_removes_weak_edges(self, temp_graph):
        """Should remove edges below threshold."""
        temp_graph.create_waypoint("mem1", "mem2", weight=0.8, bidirectional=False)
        temp_graph.create_waypoint("mem1", "mem3", weight=0.03, bidirectional=False)

        pruned = temp_graph.prune_weak_edges(min_weight=0.05)

        assert pruned == 1
        neighbors = temp_graph.get_neighbors("mem1")
        assert len(neighbors) == 1
        assert neighbors[0]["id"] == "mem2"


class TestStats:
    """Test statistics."""

    def test_stats_counts(self, temp_graph):
        """Stats should count correctly."""
        temp_graph.create_waypoint("mem1", "mem2", weight=0.8)
        temp_graph.create_waypoint("mem1", "mem3", weight=0.6, bidirectional=False)

        stats = temp_graph.stats()
        assert stats["total_nodes"] == 3  # mem1, mem2, mem3
        assert stats["total_edges"] == 3  # mem1->mem2, mem2->mem1, mem1->mem3
