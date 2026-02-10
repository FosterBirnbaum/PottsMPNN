import numpy as np

from pareto_utils import pareto_front_2d, pareto_prefilter_mask


def brute_pareto(stability_scores: np.ndarray, binding_scores: np.ndarray) -> np.ndarray:
    n = len(stability_scores)
    front = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                stability_scores[j] <= stability_scores[i]
                and binding_scores[j] <= binding_scores[i]
                and (
                    stability_scores[j] < stability_scores[i]
                    or binding_scores[j] < binding_scores[i]
                )
            ):
                front[i] = False
                break
    return front


def test_pareto_front_matches_bruteforce_randomized():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = 200
        s = rng.normal(size=n)
        b = rng.normal(size=n)
        assert np.array_equal(pareto_front_2d(s, b), brute_pareto(s, b))


def test_pareto_front_keeps_exact_duplicates():
    s = np.array([0.0, 0.0, 1.0])
    b = np.array([0.0, 0.0, 1.0])
    front = pareto_front_2d(s, b)
    assert np.array_equal(front, np.array([True, True, False]))


def test_pareto_prefilter_mask_bounds():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 3.0, 2.0, 1.0])
    mask = pareto_prefilter_mask(s, b, 25.0)
    assert np.array_equal(mask, np.array([True, False, False, True]))
