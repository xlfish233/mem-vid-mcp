"""
Sector classifier for memory categorization.

Classifies content into cognitive sectors based on pattern matching:
- episodic: Events, experiences, time-specific memories
- semantic: Facts, knowledge, definitions
- procedural: How-to, steps, instructions
- emotional: Feelings, emotions, attitudes
- reflective: Insights, realizations, meta-cognition

Each sector has different decay rates and search weights.
"""
import re
from typing import Any

# Sector configurations with weights, decay rates, and detection patterns
SECTORS: dict[str, dict[str, Any]] = {
    "episodic": {
        "weight": 1.2,
        "decay_lambda": 0.015,
        "patterns": [
            # Time markers
            r"\b(today|yesterday|tomorrow|last\s+week|next\s+week)\b",
            r"\b(remember\s+when|recall|that\s+time)\b",
            # Past tense verbs
            r"\b(went|saw|met|felt|heard|visited|attended)\b",
            # Specific times
            r"\b(at\s+\d{1,2}:\d{2}|on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
            # Event words
            r"\b(event|moment|experience|incident|happened|occurred)\b",
        ],
    },
    "semantic": {
        "weight": 1.0,
        "decay_lambda": 0.005,  # Slowest decay - facts persist
        "patterns": [
            # Definitions
            r"\b(is\s+a|represents|means|defined\s+as|refers\s+to)\b",
            # Concepts
            r"\b(concept|theory|principle|law|rule|definition)\b",
            # Facts
            r"\b(fact|statistic|data|evidence|information)\b",
            # Academic
            r"\b(history|science|geography|math|physics|chemistry)\b",
            # Knowledge verbs
            r"\b(know|understand|learn|study)\b",
        ],
    },
    "procedural": {
        "weight": 1.1,
        "decay_lambda": 0.008,
        "patterns": [
            # Instructions
            r"\b(how\s+to|step\s+by\s+step|guide|tutorial|instructions)\b",
            # Sequence
            r"\b(first|second|third|then|next|finally|lastly)\b",
            # Operations
            r"\b(install|run|execute|compile|build|deploy|configure)\b",
            # Interactions
            r"\b(click|press|type|enter|select|choose|drag)\b",
            # Technical
            r"\b(method|function|class|algorithm|procedure|process)\b",
        ],
    },
    "emotional": {
        "weight": 1.3,  # Highest weight - emotions are salient
        "decay_lambda": 0.02,  # Fastest decay
        "patterns": [
            # Feeling words
            r"\b(feel|feeling|emotions?|mood)\b",
            # Emotion adjectives
            r"\b(happy|sad|angry|excited|scared|anxious|nervous)\b",
            # Attitude verbs
            r"\b(love|hate|like|dislike|enjoy|prefer)\b",
            # Intensity
            r"\b(amazing|terrible|awesome|awful|wonderful|horrible)\b",
            # States
            r"\b(frustrated|confused|overwhelmed|relieved|grateful)\b",
            # Exclamations
            r"\b(wow|omg|yay|ugh|oh\s+no)\b",
            r"[!]{2,}",  # Multiple exclamation marks
        ],
    },
    "reflective": {
        "weight": 0.8,  # Lowest weight
        "decay_lambda": 0.001,  # Almost permanent - insights persist
        "patterns": [
            # Realizations
            r"\b(realize|realization|insight|epiphany|discovered)\b",
            # Thinking
            r"\b(think|thought|ponder|contemplate|reflect)\b",
            # Understanding
            r"\b(understand|grasp|comprehend|see\s+now)\b",
            # Patterns
            r"\b(pattern|trend|connection|link|relationship)\b",
            # Conclusions
            r"\b(lesson|moral|takeaway|conclusion|summary)\b",
            # Evaluation
            r"\b(feedback|review|analysis|evaluation|assessment)\b",
            # Growth
            r"\b(improve|grow|change|adapt|evolve)\b",
        ],
    },
}

# Cross-sector relationship matrix for search scoring
# When query sector != memory sector, apply this penalty
SECTOR_RELATIONSHIPS: dict[str, dict[str, float]] = {
    "semantic": {"procedural": 0.8, "episodic": 0.6, "reflective": 0.7, "emotional": 0.4},
    "procedural": {"semantic": 0.8, "episodic": 0.6, "reflective": 0.6, "emotional": 0.3},
    "episodic": {"reflective": 0.8, "semantic": 0.6, "procedural": 0.6, "emotional": 0.7},
    "reflective": {"episodic": 0.8, "semantic": 0.7, "procedural": 0.6, "emotional": 0.6},
    "emotional": {"episodic": 0.7, "reflective": 0.6, "semantic": 0.4, "procedural": 0.3},
}


def classify_content(content: str, metadata: dict = None) -> dict[str, Any]:
    """
    Classify content into cognitive sectors.

    Args:
        content: Text to classify
        metadata: Optional metadata with explicit sector override

    Returns:
        {
            "primary": str,      # Main sector
            "additional": list,  # Secondary sectors
            "confidence": float, # Classification confidence (0-1)
            "decay_lambda": float  # Decay rate for this content
        }
    """
    # Allow explicit sector override via metadata
    if metadata and metadata.get("sector") in SECTORS:
        sector = metadata["sector"]
        return {
            "primary": sector,
            "additional": [],
            "confidence": 1.0,
            "decay_lambda": SECTORS[sector]["decay_lambda"],
        }

    # Score each sector by pattern matches
    content_lower = content.lower()
    scores: dict[str, float] = {}

    for sector, config in SECTORS.items():
        score = 0.0
        for pattern in config["patterns"]:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                score += len(matches) * config["weight"]
        scores[sector] = score

    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary, primary_score = sorted_scores[0]

    # Find additional sectors above threshold
    threshold = max(1.0, primary_score * 0.3)
    additional = [s for s, sc in sorted_scores[1:] if sc > 0 and sc >= threshold]

    # Calculate confidence based on score separation
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
    if primary_score > 0:
        confidence = min(1.0, primary_score / (primary_score + second_score + 1))
    else:
        confidence = 0.2

    # Default to semantic if no patterns matched
    if primary_score == 0:
        primary = "semantic"

    return {
        "primary": primary,
        "additional": additional,
        "confidence": confidence,
        "decay_lambda": SECTORS[primary]["decay_lambda"],
    }


def get_sector_penalty(query_sector: str, memory_sector: str) -> float:
    """
    Get cross-sector penalty for search scoring.

    Args:
        query_sector: Sector of the search query
        memory_sector: Sector of the memory being scored

    Returns:
        Penalty multiplier (1.0 = no penalty, <1.0 = penalized)
    """
    if query_sector == memory_sector:
        return 1.0
    return SECTOR_RELATIONSHIPS.get(query_sector, {}).get(memory_sector, 0.3)


def get_decay_lambda(sector: str) -> float:
    """Get decay rate for a sector."""
    return SECTORS.get(sector, SECTORS["semantic"])["decay_lambda"]
