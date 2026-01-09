"""Tests for memory decay mechanism."""

from datetime import datetime

from memvid_mcp.decay import (
    apply_decay_to_memories,
    calculate_decay,
    pick_tier,
    propagate_reinforcement,
    reinforce,
)


def make_memory(
    salience=0.5,
    last_seen_days_ago=0,
    coactivations=0,
    sector="semantic",
):
    """Helper to create test memory dict."""
    now_ts = int(datetime.now().timestamp() * 1000)
    last_seen = now_ts - (last_seen_days_ago * 86400000)
    return {
        "id": "test-mem",
        "salience": salience,
        "last_seen_at": last_seen,
        "updated_at": last_seen,
        "coactivations": coactivations,
        "primary_sector": sector,
    }


class TestPickTier:
    """Test tier classification."""

    def test_hot_tier_recent_high_salience(self):
        """Recent + high salience = hot."""
        mem = make_memory(salience=0.8, last_seen_days_ago=2)
        now_ts = int(datetime.now().timestamp() * 1000)
        assert pick_tier(mem, now_ts) == "hot"

    def test_hot_tier_recent_high_coactivations(self):
        """Recent + high coactivations = hot."""
        mem = make_memory(salience=0.5, last_seen_days_ago=2, coactivations=10)
        now_ts = int(datetime.now().timestamp() * 1000)
        assert pick_tier(mem, now_ts) == "hot"

    def test_warm_tier_recent_low_salience(self):
        """Recent but low salience = warm."""
        mem = make_memory(salience=0.3, last_seen_days_ago=2)
        now_ts = int(datetime.now().timestamp() * 1000)
        assert pick_tier(mem, now_ts) == "warm"

    def test_warm_tier_old_moderate_salience(self):
        """Old but moderate salience = warm."""
        mem = make_memory(salience=0.5, last_seen_days_ago=30)
        now_ts = int(datetime.now().timestamp() * 1000)
        assert pick_tier(mem, now_ts) == "warm"

    def test_cold_tier_old_low_salience(self):
        """Old + low salience = cold."""
        mem = make_memory(salience=0.2, last_seen_days_ago=30)
        now_ts = int(datetime.now().timestamp() * 1000)
        assert pick_tier(mem, now_ts) == "cold"


class TestCalculateDecay:
    """Test decay calculation."""

    def test_no_decay_for_recent_memory(self):
        """Recently accessed memory should have minimal decay."""
        mem = make_memory(salience=0.8, last_seen_days_ago=0)
        now_ts = int(datetime.now().timestamp() * 1000)
        new_salience = calculate_decay(mem, now_ts)
        assert abs(new_salience - 0.8) < 0.01

    def test_decay_increases_with_time(self):
        """Older memories should decay more."""
        mem1 = make_memory(salience=0.8, last_seen_days_ago=10)
        mem2 = make_memory(salience=0.8, last_seen_days_ago=30)
        now_ts = int(datetime.now().timestamp() * 1000)

        decay1 = calculate_decay(mem1, now_ts)
        decay2 = calculate_decay(mem2, now_ts)

        assert decay2 < decay1  # Older memory decays more

    def test_high_salience_resists_decay(self):
        """High salience memories should decay slower."""
        mem_high = make_memory(salience=0.9, last_seen_days_ago=30)
        mem_low = make_memory(salience=0.3, last_seen_days_ago=30)
        now_ts = int(datetime.now().timestamp() * 1000)

        decay_high = calculate_decay(mem_high, now_ts)
        decay_low = calculate_decay(mem_low, now_ts)

        # High salience should retain more of its value
        assert (decay_high / 0.9) > (decay_low / 0.3)

    def test_sector_specific_decay(self):
        """Sector-specific decay rate should be applied."""
        mem = make_memory(salience=0.8, last_seen_days_ago=30)
        now_ts = int(datetime.now().timestamp() * 1000)

        # Emotional decays faster
        decay_emotional = calculate_decay(mem, now_ts, sector_decay_lambda=0.02)
        # Reflective decays slower
        decay_reflective = calculate_decay(mem, now_ts, sector_decay_lambda=0.001)

        assert decay_emotional < decay_reflective


class TestApplyDecayToMemories:
    """Test batch decay application."""

    def test_updates_multiple_memories(self):
        """Should update all memories."""
        memories = {
            "mem1": make_memory(salience=0.8, last_seen_days_ago=30),
            "mem2": make_memory(salience=0.6, last_seen_days_ago=60),
        }

        updated = apply_decay_to_memories(memories)

        assert len(updated) == 2
        assert memories["mem1"]["salience"] < 0.8
        assert memories["mem2"]["salience"] < 0.6

    def test_uses_sector_decay_rates(self):
        """Should use sector-specific decay rates."""
        memories = {
            "emotional": make_memory(
                salience=0.8, last_seen_days_ago=30, sector="emotional"
            ),
            "reflective": make_memory(
                salience=0.8, last_seen_days_ago=30, sector="reflective"
            ),
        }

        sector_lambdas = {
            "emotional": 0.02,
            "reflective": 0.001,
        }

        apply_decay_to_memories(memories, sector_lambdas)

        # Emotional should decay more
        assert memories["emotional"]["salience"] < memories["reflective"]["salience"]


class TestReinforce:
    """Test reinforcement mechanism."""

    def test_reinforce_increases_salience(self):
        """Reinforcement should increase salience."""
        mem = make_memory(salience=0.5)
        new_salience = reinforce(mem)
        assert new_salience > 0.5

    def test_reinforce_diminishing_returns(self):
        """High salience should get smaller boost."""
        mem_low = make_memory(salience=0.3)
        mem_high = make_memory(salience=0.9)

        boost_low = reinforce(mem_low) - 0.3
        boost_high = reinforce(mem_high) - 0.9

        assert boost_low > boost_high

    def test_reinforce_respects_maximum(self):
        """Salience should not exceed 1.0."""
        mem = make_memory(salience=0.95)
        for _ in range(10):
            reinforce(mem)
        assert mem["salience"] <= 1.0

    def test_reinforce_updates_last_seen(self):
        """Reinforcement should update last_seen_at."""
        mem = make_memory(salience=0.5, last_seen_days_ago=10)
        old_last_seen = mem["last_seen_at"]
        reinforce(mem)
        assert mem["last_seen_at"] > old_last_seen

    def test_reinforce_increments_coactivations(self):
        """Reinforcement should increment coactivations."""
        mem = make_memory(salience=0.5, coactivations=5)
        reinforce(mem)
        assert mem["coactivations"] == 6


class TestPropagateReinforcement:
    """Test reinforcement propagation through waypoints."""

    def test_propagation_positive_only(self):
        """Should only propagate positive reinforcement."""
        boost = propagate_reinforcement(
            source_salience=0.3,  # Lower than target
            target_salience=0.8,
            waypoint_weight=0.9,
            time_diff_days=1,
        )
        assert boost == 0  # No negative propagation

    def test_propagation_decays_with_time(self):
        """Propagation should decay with time."""
        boost_recent = propagate_reinforcement(
            source_salience=0.9,
            target_salience=0.3,
            waypoint_weight=0.8,
            time_diff_days=1,
        )
        boost_old = propagate_reinforcement(
            source_salience=0.9,
            target_salience=0.3,
            waypoint_weight=0.8,
            time_diff_days=30,
        )
        assert boost_recent > boost_old

    def test_propagation_scales_with_weight(self):
        """Propagation should scale with waypoint weight."""
        boost_strong = propagate_reinforcement(
            source_salience=0.9,
            target_salience=0.3,
            waypoint_weight=0.9,
            time_diff_days=1,
        )
        boost_weak = propagate_reinforcement(
            source_salience=0.9,
            target_salience=0.3,
            waypoint_weight=0.3,
            time_diff_days=1,
        )
        assert boost_strong > boost_weak
