"""Tests for sector classifier."""

from memvid_mcp.classifier import (
    classify_content,
    get_decay_lambda,
    get_sector_penalty,
)


class TestClassifyContent:
    """Test content classification into sectors."""

    def test_episodic_time_markers(self):
        """Content with time markers should be episodic."""
        result = classify_content("Yesterday I went to the store")
        assert result["primary"] == "episodic"

    def test_episodic_remember_when(self):
        """Content with 'remember when' should be episodic."""
        result = classify_content("Remember when we visited Paris last summer?")
        assert result["primary"] == "episodic"

    def test_semantic_definition(self):
        """Content with definitions should be semantic."""
        result = classify_content("Python is a programming language")
        assert result["primary"] == "semantic"

    def test_semantic_facts(self):
        """Content with facts should be semantic."""
        result = classify_content("The capital of France is Paris")
        assert result["primary"] == "semantic"

    def test_procedural_how_to(self):
        """Content with how-to should be procedural."""
        result = classify_content("How to install Python: first download the installer")
        assert result["primary"] == "procedural"

    def test_procedural_steps(self):
        """Content with steps should be procedural."""
        result = classify_content(
            "First, click the button. Then, enter your name. Finally, submit."
        )
        assert result["primary"] == "procedural"

    def test_emotional_feelings(self):
        """Content with feelings should be emotional."""
        result = classify_content("I feel so happy today!")
        assert result["primary"] == "emotional"

    def test_emotional_exclamations(self):
        """Content with exclamations should be emotional."""
        result = classify_content("This is amazing!! I love it!!!")
        assert result["primary"] == "emotional"

    def test_reflective_realization(self):
        """Content with realizations should be reflective."""
        result = classify_content("I realized that the pattern connects everything")
        assert result["primary"] == "reflective"

    def test_reflective_insight(self):
        """Content with insights should be reflective."""
        result = classify_content("The key insight is that feedback improves growth")
        assert result["primary"] == "reflective"

    def test_default_to_semantic(self):
        """Content with no patterns should default to semantic."""
        result = classify_content("xyz abc 123")
        assert result["primary"] == "semantic"

    def test_explicit_sector_override(self):
        """Metadata sector should override classification."""
        result = classify_content(
            "I feel happy",  # Would be emotional
            metadata={"sector": "semantic"},
        )
        assert result["primary"] == "semantic"
        assert result["confidence"] == 1.0

    def test_additional_sectors(self):
        """Content matching multiple sectors should have additional sectors."""
        result = classify_content(
            "Yesterday I learned how to install Python step by step"
        )
        # Should match both episodic (yesterday) and procedural (how to, step by step)
        assert len(result["additional"]) > 0 or result["primary"] in [
            "episodic",
            "procedural",
        ]

    def test_confidence_score(self):
        """Classification should include confidence score."""
        result = classify_content("I feel extremely happy and excited!")
        assert 0 <= result["confidence"] <= 1

    def test_decay_lambda_included(self):
        """Classification should include decay lambda."""
        result = classify_content("Some content")
        assert "decay_lambda" in result
        assert result["decay_lambda"] > 0


class TestGetSectorPenalty:
    """Test cross-sector penalty calculation."""

    def test_same_sector_no_penalty(self):
        """Same sector should have no penalty."""
        assert get_sector_penalty("semantic", "semantic") == 1.0
        assert get_sector_penalty("emotional", "emotional") == 1.0

    def test_related_sectors_moderate_penalty(self):
        """Related sectors should have moderate penalty."""
        penalty = get_sector_penalty("semantic", "procedural")
        assert 0.5 < penalty < 1.0

    def test_unrelated_sectors_high_penalty(self):
        """Unrelated sectors should have high penalty."""
        penalty = get_sector_penalty("emotional", "procedural")
        assert penalty < 0.5

    def test_unknown_sector_default_penalty(self):
        """Unknown sector should use default penalty."""
        penalty = get_sector_penalty("unknown", "semantic")
        assert penalty == 0.3


class TestGetDecayLambda:
    """Test decay rate retrieval."""

    def test_emotional_fastest_decay(self):
        """Emotional should have fastest decay."""
        assert get_decay_lambda("emotional") == 0.02

    def test_reflective_slowest_decay(self):
        """Reflective should have slowest decay."""
        assert get_decay_lambda("reflective") == 0.001

    def test_semantic_slow_decay(self):
        """Semantic should have slow decay."""
        assert get_decay_lambda("semantic") == 0.005

    def test_unknown_defaults_to_semantic(self):
        """Unknown sector should default to semantic decay."""
        assert get_decay_lambda("unknown") == get_decay_lambda("semantic")
