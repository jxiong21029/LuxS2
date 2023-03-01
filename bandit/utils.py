"""
Helper library for shared types and utilites.
"""
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# multi-armed bandit arm
Arm = int | np.int64
# function that takes two arms and returns whether the first one won
Duel = Callable[[Arm, Arm], bool]
# bandit algorithm
BanditAlgorithm = Callable[[int, Duel, int], Arm]
# history of queries by a bandit algorithm
History = list[tuple[Arm, Arm]]
# preference matrix
Preferences = NDArray[np.float64]


def valid_preferences(p: Preferences) -> bool:
    """Validate that the given preference matrix is valid."""
    assert ((0 <= p) & (p <= 1)).all(), "entries must be valid probabilities"
    assert np.allclose(p + p.T, 1), "probabilities should be skew-symmetric"
    # this is technically redundant
    # assert np.allclose(
    #     np.diagonal(p), 0.5
    # ), "playing an arm against itself must tie"
    return True


def count_history(history: History) -> dict[Arm, int]:
    """Frequency of each arm in the history."""
    count = {}
    for pair in history:
        for arm in pair:
            count[arm] = count.get(arm, 0) + 1
    return count


def d_kl(p: float, q: float) -> float:
    """Kullback-Leibler (KL) divergence between Bernoulli distributions."""
    if p == 0:
        return -np.log1p(-q)
    elif p == 1:
        return -np.log(q)
    else:
        # fmt: off
        return (
            p  * (np.log(p) - np.log(q)) +
            (1 - p) * (np.log1p(-p) - np.log1p(-q))
        )
        # fmt: on


# Copeland functions (see [9, 4, 6])


def copeland_scores(p: Preferences, normalize: bool = False) -> np.ndarray:
    """
    Compute the (normalized) Copeland scores for each arm.

    The Copeland score for a given arm is the number of
    arms beaten by it (wins with probability > 1/2).
    """
    scores = np.sum(p > 0.5, axis=1)
    return scores / (p.shape[1] - 1) if normalize else scores


def copeland_winners(p: Preferences) -> np.ndarray:
    """
    Return the set of all Copeland winners.

    A Copeland winner is defined as an arm with the highest Copeland score.
    """
    scores = copeland_scores(p)
    score = np.max(scores)
    return np.arange(p.shape[0])[scores == score]


def copeland_winner(p: Preferences) -> Arm:
    """Return an arbitrary Copeland winner if there are multiple."""
    return np.argmax(copeland_scores(p))


def copeland_regret(p: Preferences, history: History) -> float:
    """Cumulative regret w.r.t. to Copeland winners."""
    scores = copeland_scores(p, normalize=True)
    return (
        len(history) * np.max(scores)
        - sum(scores[arm1] + scores[arm2] for arm1, arm2 in history) / 2
    )


def copeland_winner_wins(wins: np.ndarray) -> Arm:
    """Return an arbitrary Copeland winner from a number of wins matrix."""
    nums = wins + wins.T
    # define 0/0 := 1/2
    mask = nums == 0
    wins[mask] = 1
    nums[mask] = 2
    p = wins / nums
    np.fill_diagonal(p, 0.5)
    return copeland_winner(p)


# Condorcet functions (see [7, 3, 5])


def condorcet_winner(p: Preferences) -> Arm | None:
    """
    Return the (necessarily unique) Condorcet winner.

    A Condorcet winner is defined as an arm which beats all other arms.
    """
    scores = copeland_scores(p)
    arm = np.argmax(scores)
    return arm if scores[arm] == p.shape[0] - 1 else None


def condorcet_regret(p: Preferences, history: History) -> float:
    """Cumulative regret w.r.t. a Condorcet winner."""
    best = condorcet_winner(p)
    return (
        sum(p[best, arm1] + p[best, arm2] - 1 for arm1, arm2 in history) / 2
    )  # type: ignore
