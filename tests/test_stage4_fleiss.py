import math
import pytest
from pipeline.stages.stage4_scorer import fleiss_kappa, _discretize


def test_discretize_bins_to_three_categories():
    assert _discretize(0.0) == 0
    assert _discretize(0.24) == 0
    assert _discretize(0.25) == 1
    assert _discretize(0.5) == 1
    assert _discretize(0.74) == 1
    assert _discretize(0.75) == 2
    assert _discretize(1.0) == 2


def test_fleiss_kappa_perfect_agreement():
    # 3 items, 2 raters, all rate 1.0 -> all category 2
    ratings = [[1.0, 1.0]] * 3
    k = fleiss_kappa(ratings)
    # Perfect agreement → 1.0 (or close to 1.0; small N may give NaN if all-same → handled)
    assert k == 1.0


def test_fleiss_kappa_perfect_disagreement():
    # 4 items, 2 raters, raters always opposite (one 0.0, one 1.0)
    ratings = [[0.0, 1.0]] * 4
    k = fleiss_kappa(ratings)
    # Two categories used equally → expected agreement matches observed → kappa near 0 or negative
    assert k < 0.2


def test_fleiss_kappa_handles_three_categories():
    ratings = [
        [1.0, 1.0],   # both cat 2
        [0.5, 0.5],   # both cat 1
        [0.0, 0.0],   # both cat 0
        [1.0, 1.0],
    ]
    k = fleiss_kappa(ratings)
    assert k > 0.9


def test_fleiss_kappa_returns_nan_on_too_few_raters():
    ratings = [[0.5]]  # only 1 rater
    k = fleiss_kappa(ratings)
    assert math.isnan(k)


def test_fleiss_kappa_handles_empty():
    assert math.isnan(fleiss_kappa([]))
