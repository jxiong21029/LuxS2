import numpy as np

from ..utils import Arm, Preferences, condorcet_regret
from .common import Problem, is_condorcet_winner


class RankingProblem(Problem):
    """
    The ranking setup in https://github.com/jxiong21029/LuxS2/issues/1.

    Suppose we have agents with hidden "ratings" r_1, ..., r_n and player A
    wins a match against player B with probability 1/(1 + exp(r_B - r_A)).

    Note that this problem satisfies a total ordering on the arms,
    stochastic transitivity, and the stochastic triangle inequality.
    """

    def __init__(
        self,
        K: int,
        rating_max: float,
        rating_min: float = 0,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """Intialize the state of the problem."""
        self.K = K
        self.rng = rng

        self.ratings = rng.uniform(rating_min, rating_max, K)

    def duel(self, arm1: Arm, arm2: Arm) -> bool:
        """Whether arm1 beats arm2."""
        p = 1 / (1 + np.exp(self.ratings[arm2] - self.ratings[arm1]))
        return self.rng.random() < p

    def preference_matrix(self) -> Preferences:
        """Return the preference matrix for the problem."""
        return 1 / (1 + np.exp(self.ratings - self.ratings[:, np.newaxis]))

    regret_function = staticmethod(condorcet_regret)

    is_winner = is_condorcet_winner

    def shuffle(self) -> None:
        """Shuffle the internal state to prevent spurious correlations."""
        self.rng.shuffle(self.ratings)
