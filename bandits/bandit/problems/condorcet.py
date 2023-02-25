import numpy as np
from .common import Problem, duel_matrix, preference_matrix_get, shuffle_matrix
from .common import is_condorcet_winner
from ..utils import condorcet_regret

class CondorcetProblem(Problem):
    """ A problem extended with a guaranteed Condorcet winner. """

    def __init__(
        self,
        base: Problem,
        rng: np.random.Generator=np.random.default_rng(),
    ):
        """ Intialize the state of the problem. """
        self.K = base.K + 1
        self.rng = rng

        p = base.preference_matrix()
        rest = rng.uniform(0.5, 1, (base.K, 1))
        self.p = np.block([
            [0.5, rest.T],
            [1 - rest, p],
        ]) # type: ignore
        # avoid Condorcet winner always placed first
        self.shuffle()

    duel = duel_matrix

    preference_matrix = preference_matrix_get

    regret_function = staticmethod(condorcet_regret)

    is_winner = is_condorcet_winner

    shuffle = shuffle_matrix

