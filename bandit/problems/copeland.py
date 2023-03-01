import numpy as np

from ..utils import copeland_regret, copeland_winners
from .common import (
    Problem,
    duel_matrix,
    is_copeland_winner,
    preference_matrix_get,
    shuffle_matrix,
)
from .random import RandomProblem


class CopelandProblem(Problem):
    """A problem with a unique Copeland winner."""

    def __init__(
        self,
        K: int,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """Intialize the state of the problem."""
        self.K = K
        self.rng = rng

        # brute-force sample until satisfied
        get_matrix = lambda: (  # noqa: E731
            RandomProblem(K, rng).preference_matrix()
        )
        self.p = get_matrix()
        while len(copeland_winners(self.p)) > 1:
            self.p = get_matrix()

    duel = duel_matrix

    preference_matrix = preference_matrix_get

    regret_function = staticmethod(copeland_regret)

    is_winner = is_copeland_winner

    shuffle = shuffle_matrix
