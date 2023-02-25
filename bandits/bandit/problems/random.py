import numpy as np
from .common import Problem, duel_matrix, preference_matrix_get, shuffle_matrix
from .common import is_copeland_winner
from ..utils import copeland_regret
from .. import utils

class RandomProblem(Problem):
    """ A completely random problem. """

    def __init__(
        self,
        K: int,
        rng: np.random.Generator=np.random.default_rng(),
    ):
        """ Intialize the state of the problem. """
        self.K = K
        self.rng = rng

        self.p = np.zeros((K, K))
        lower = np.tril_indices(n=K, k=-1)
        self.p[lower] = rng.uniform(0, 1, size=lower[0].shape[0])
        self.p = self.p + (1 - self.p.T)
        self.p[lower] -= 1
        np.fill_diagonal(self.p, 0.5)

    duel = duel_matrix

    preference_matrix = preference_matrix_get

    regret_function = staticmethod(copeland_regret)

    is_winner = is_copeland_winner

    shuffle = shuffle_matrix
