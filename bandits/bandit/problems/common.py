"""
Common utilities between all problem environments.
"""
from abc import ABC, abstractmethod
import numpy as np
from ..utils import Arm, BanditAlgorithm, History, Preferences
from ..utils import copeland_winners, condorcet_winner

class Problem(ABC):
    """ An instance of a general dueling bandit problem. """

    K: int

    @abstractmethod
    def duel(self, arm1: Arm, arm2: Arm) -> bool:
        """ Whether arm1 beats arm2. """

    @abstractmethod
    def preference_matrix(self) -> Preferences:
        """ Return the preference matrix for the problem. """

    @staticmethod
    @abstractmethod
    def regret_function(p: Preferences, history: History) -> float:
        """ Cumulative regret for the given prefrence matrix and history. """

    def regret(self, history: History) -> float:
        """ Return the regret for a particular history. """
        return self.regret_function(self.preference_matrix(), history)

    @abstractmethod
    def is_winner(self, arm: Arm) -> bool:
        """ Whether the arm is a considered a winner. """

    @abstractmethod
    def shuffle(self) -> None:
        """ Shuffle the internal state to prevent spurious correlations. """

# common implementation patterns

def duel_matrix(self, arm1: Arm, arm2: Arm) -> bool:
    """ A duel for problems defined by explicit preference matrices. """
    return self.rng.random() < self.p[arm1, arm2]

def preference_matrix_get(self) -> Preferences:
    """ Return the preference matrix for the problem. """
    return self.p

def shuffle_matrix(self) -> None:
    """ Shuffle the internal state to prevent spurious correlations. """
    permute = self.rng.permutation(self.K)
    self.p = self.p[np.ix_(permute, permute)]

def is_copeland_winner(self, arm: Arm) -> bool:
    """ Whether the arm is a considered a Copeland winner. """
    return arm in copeland_winners(self.preference_matrix())

def is_condorcet_winner(self, arm: Arm) -> bool:
    """ Whether the arm is a considered a Condorcet winner. """
    return arm == condorcet_winner(self.preference_matrix())

# useful methods

def run_problem(
    problem: Problem,
    bandit_algorithm: BanditAlgorithm,
    T: int,
    *args,
    **kwargs,
) -> tuple[Arm, History]:
    """ Run the bandit algorithm on the problem. """
    history: History = []

    def duel(arm1: Arm, arm2: Arm) -> bool:
        """ Wrap the duel of the problem to track queries. """
        history.append((arm1, arm2))
        return problem.duel(arm1, arm2)

    result = bandit_algorithm(problem.K, duel, T, *args, **kwargs)
    assert len(history) == T, \
        f"algorithm used {len(history)} comparisions for time horizon {T}"
    return result, history

