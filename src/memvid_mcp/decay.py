"""
Memory decay and reinforcement mechanism.

Implements salience-based memory management:
- Tiered decay: hot/warm/cold memories decay at different rates
- Exponential decay formula based on time and salience
- Reinforcement on retrieval to strengthen accessed memories
"""
import math
from datetime import datetime


# Decay rates by tier (per day)
DECAY_RATES = {
    "hot": 0.005,   # Recently accessed, high salience - slow decay
    "warm": 0.02,   # Moderate activity - medium decay
    "cold": 0.05,   # Inactive - fast decay
}

# Tier thresholds
HOT_RECENCY_DAYS = 6  # Days since last access to be "hot"
HOT_SALIENCE = 0.7    # Minimum salience to be "hot"
WARM_SALIENCE = 0.4   # Minimum salience to be "warm"

# Reinforcement settings
REINFORCE_BOOST = 0.15  # Salience boost on retrieval
MIN_SALIENCE = 0.0      # Minimum salience floor
MAX_SALIENCE = 1.0      # Maximum salience ceiling


def pick_tier(memory: dict, now_ts: int) -> str:
    """
    Classify memory into decay tier based on recency and salience.

    Args:
        memory: Memory dict with salience, last_seen_at
        now_ts: Current timestamp in milliseconds

    Returns:
        Tier name: "hot", "warm", or "cold"
    """
    last_seen = memory.get("last_seen_at") or memory.get("updated_at") or now_ts
    days_since = (now_ts - last_seen) / 86400000  # ms to days

    salience = memory.get("salience", 0.5)
    coactivations = memory.get("coactivations", 0)

    # Hot: recently accessed AND (high salience OR frequently accessed)
    is_recent = days_since < HOT_RECENCY_DAYS
    is_high_value = coactivations > 5 or salience > HOT_SALIENCE

    if is_recent and is_high_value:
        return "hot"

    # Warm: recent OR moderate salience
    if is_recent or salience > WARM_SALIENCE:
        return "warm"

    return "cold"


def calculate_decay(
    memory: dict,
    now_ts: int,
    sector_decay_lambda: float = None,
) -> float:
    """
    Calculate new salience after decay.

    Formula: salience *= exp(-lambda * (days / (salience + 0.1)))

    Args:
        memory: Memory dict
        now_ts: Current timestamp in milliseconds
        sector_decay_lambda: Optional sector-specific decay rate

    Returns:
        New salience value
    """
    salience = memory.get("salience", 0.5)
    last_seen = memory.get("last_seen_at") or memory.get("updated_at") or now_ts

    # Days since last access
    days = max(0, (now_ts - last_seen) / 86400000)

    # Get decay rate from tier or sector
    tier = pick_tier(memory, now_ts)
    decay_lambda = sector_decay_lambda or DECAY_RATES[tier]

    # Exponential decay with salience-based resistance
    # Higher salience = slower decay
    decay_factor = math.exp(-decay_lambda * (days / (salience + 0.1)))
    new_salience = max(MIN_SALIENCE, min(MAX_SALIENCE, salience * decay_factor))

    return new_salience


def apply_decay_to_memories(
    memories: dict[str, dict],
    sector_decay_lambdas: dict[str, float] = None,
) -> list[str]:
    """
    Apply decay to all memories.

    Args:
        memories: Dict of memory_id -> memory
        sector_decay_lambdas: Optional per-sector decay rates

    Returns:
        List of memory IDs that were updated
    """
    now_ts = int(datetime.now().timestamp() * 1000)
    updated = []

    for mem_id, memory in memories.items():
        old_salience = memory.get("salience", 0.5)

        # Get sector-specific decay rate if available
        sector = memory.get("primary_sector", "semantic")
        sector_lambda = None
        if sector_decay_lambdas:
            sector_lambda = sector_decay_lambdas.get(sector)

        new_salience = calculate_decay(memory, now_ts, sector_lambda)

        if abs(new_salience - old_salience) > 0.001:  # Significant change
            memory["salience"] = new_salience
            updated.append(mem_id)

    return updated


def reinforce(memory: dict, boost: float = None) -> float:
    """
    Reinforce a memory's salience on retrieval.

    Formula: salience += boost * (1 - salience)
    This gives diminishing returns as salience approaches 1.

    Args:
        memory: Memory dict to reinforce
        boost: Boost amount (default: REINFORCE_BOOST)

    Returns:
        New salience value
    """
    boost = boost if boost is not None else REINFORCE_BOOST
    salience = memory.get("salience", 0.5)

    # Diminishing returns formula
    new_salience = min(MAX_SALIENCE, salience + boost * (1 - salience))

    memory["salience"] = new_salience
    memory["last_seen_at"] = int(datetime.now().timestamp() * 1000)
    memory["coactivations"] = memory.get("coactivations", 0) + 1

    return new_salience


def propagate_reinforcement(
    source_salience: float,
    target_salience: float,
    waypoint_weight: float,
    time_diff_days: float,
) -> float:
    """
    Calculate reinforcement to propagate through waypoint.

    Args:
        source_salience: Salience of retrieved memory
        target_salience: Current salience of linked memory
        waypoint_weight: Weight of the connecting waypoint
        time_diff_days: Days since target was last accessed

    Returns:
        Salience boost to apply to target
    """
    # Decay factor based on time
    decay_factor = math.exp(-0.02 * time_diff_days)

    # Propagated boost = gamma * (source - target) * decay * weight
    gamma = 0.1  # Propagation strength
    boost = gamma * (source_salience - target_salience) * decay_factor * waypoint_weight

    return max(0, boost)  # Only positive reinforcement
