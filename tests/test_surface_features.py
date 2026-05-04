"""Unit tests for surface_features (logprob extraction structure)."""

from __future__ import annotations

from rae.surface_features import SurfaceFeatures


def test_surface_features_dataclass_shape():
    """Feature dataclass has the expected 5 numeric fields + id."""
    sf = SurfaceFeatures(
        instance_id="test_001",
        draft_mean_logprob=-1.5,
        draft_last_logprob=-2.0,
        draft_min_logprob=-5.0,
        draft_perplexity=4.48,
        draft_len_tokens=40,
    )
    assert sf.instance_id == "test_001"
    assert sf.draft_len_tokens == 40
    # Perplexity = exp(-mean_logprob)
    import math

    assert abs(sf.draft_perplexity - math.exp(-sf.draft_mean_logprob)) < 0.01


def test_logprob_semantic_ordering():
    """A more confident draft should have higher mean logprob (less negative)."""
    confident = SurfaceFeatures("a", -0.3, -0.5, -1.0, 1.35, 20)
    hesitant = SurfaceFeatures("b", -3.0, -4.0, -8.0, 20.1, 20)
    assert confident.draft_mean_logprob > hesitant.draft_mean_logprob
    assert confident.draft_perplexity < hesitant.draft_perplexity
