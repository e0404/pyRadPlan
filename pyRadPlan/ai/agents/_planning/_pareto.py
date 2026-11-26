"""Pareto front of plan blends: weight sampling and epsilon dominance.

Costs are arrays with one row per candidate and one column per axis; lower is better.
"""

import numpy as np

# Blend weights are drawn with this seed, so the same plans give the same front
SEED = 0
# A goal changing by less than this share of its limit over the plans is no axis
FLAT = 0.01
# Differences below this share of a goal's limit do not count (epsilon dominance), so the
# front holds blends that differ noticeably
EPS = 0.01


def sample_weights(n: int, n_samples: int) -> np.ndarray:
    """Blend weights over *n* plans, rounded to 1 %, without repeats.

    The first *n* rows are the plans themselves, then pairwise mixes, then random
    weights: sparse ones (Dirichlet alpha < 1) reach the edges of the simplex, uniform
    ones its inside.
    """
    rng = np.random.default_rng(SEED)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            for t in (0.25, 0.5, 0.75):
                w = np.zeros(n)
                w[i], w[j] = 1 - t, t
                pairs.append(w)
    k = max(n_samples - n - len(pairs), 0)
    random = [rng.dirichlet(np.ones(n), k - k // 2), rng.dirichlet(np.full(n, 0.3), k // 2)]
    weights = np.round(100 * np.vstack([np.eye(n), *pairs, *random])) / 100
    weights = weights[weights.sum(axis=1) > 0]
    _, first = np.unique(weights, axis=0, return_index=True)
    return weights[np.sort(first)]


def dominated(points: np.ndarray, by: np.ndarray) -> np.ndarray:
    """Whether each row of *points* is Pareto-dominated by a row of *by*."""
    return np.array(
        [np.any(np.all(by <= x, axis=1) & np.any(by < x, axis=1)) for x in points], dtype=bool
    )


def epsilon_front(cost: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Row indices of the epsilon-Pareto front of *cost* (*eps* per axis).

    The costs are binned on a grid of *eps*; each cell keeps the candidate nearest the
    best on every axis, then Pareto dominance is applied between the cells.
    """
    cells = np.floor(cost / eps)
    spread = np.ptp(cost, axis=0)
    span = np.where(spread > 0, spread, 1.0)
    by_closeness = np.argsort(((cost - cost.min(axis=0)) / span).sum(axis=1), kind="stable")
    _, first_in_cell = np.unique(cells[by_closeness], axis=0, return_index=True)
    reps = by_closeness[np.sort(first_in_cell)]
    return reps[~dominated(cells[reps], cells[reps])]


def spread_picks(front_cost: np.ndarray, top_n: int) -> tuple[np.ndarray, list[int]]:
    """Regret of every front member per axis and up to *top_n* spread-out members.

    The regret is the distance from the front's best on each axis as a share of the
    front's range. The first pick is the compromise (smallest worst regret, then
    smallest total); each further pick is farthest from the picks so far.
    """
    best, worst = front_cost.min(axis=0), front_cost.max(axis=0)
    regret = (front_cost - best) / np.where(worst > best, worst - best, 1.0)
    picked = [int(np.lexsort((regret.sum(axis=1), regret.max(axis=1)))[0])]
    while len(picked) < min(top_n, len(front_cost)):
        dist = np.linalg.norm(regret[:, None, :] - regret[picked][None], axis=2).min(axis=1)
        if dist.max() <= 0:
            break
        picked.append(int(np.argmax(dist)))
    return regret, picked
