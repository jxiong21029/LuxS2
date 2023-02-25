"""
Algorithms for the multi-armed bandit and dueling multi-armed bandit problems.

See the following references (not all of these are currently implemented):
[1] A. Agarwal, R. Ghuge, and V. Nagarajan, "An Asymptotically
    Optimal Batched Algorithm for the Dueling Bandit Problem."
    arXiv, Sep. 2022. Available: <https://arxiv.org/abs/2209.12108>
[2] A. Agarwal, R. Ghuge, and V. Nagarajan, "Batched Dueling Bandits."
    arXiv, Feb. 2022. Available: <https://arxiv.org/abs/2202.10660>
[3] J. Komiyama, J. Honda, H. Kashima, and H. Nakagawa, "Regret
    Lower Bound and Optimal Algorithm in Dueling Bandit Problem."
    arXiv, Jun. 2015. Available: <https://arxiv.org/abs/1506.02550>
[4] J. Komiyama, J. Honda, and H. Nakagawa, "Copeland Dueling Bandit Problem:
    Regret Lower Bound, Optimal Algorithm, and Computationally Efficient
    Algorithm." arXiv, May 2016. Available: <https://arxiv.org/abs/1605.01677>
[5] A. Saha and P. Gaillard, "Versatile Dueling Bandits:
    Best-of-both-World Analyses for Online Learning from Preferences."
    arXiv, Feb. 2022. Available: <https://arxiv.org/abs/2202.06694>
[6] H. Wu and X. Liu, "Double Thompson Sampling for Dueling Bandits."
    arXiv, Oct. 2016. Available: <https://arxiv.org/abs/1604.07101>
[7] Y. Yue and T. Joachims, "Beat the mean bandit," in
    *Proceedings of the 28th International Conference on International
    Conference on Machine Learning*, Jun. 2011, pp. 241–248.
[8] J. Zimmert and Y. Seldin, "Tsallis-INF: An Optimal
    Algorithm for Stochastic and Adversarial Bandits." arXiv,
    Mar. 2022. Available: <https://arxiv.org/abs/1807.07623>
[9] M. Zoghi, Z. Karnin, S. Whiteson, and M. de Rijke,
    "Copeland Dueling Bandits." arXiv, May 2015. doi:
    [10.48550/arXiv.1506.00312](https://doi.org/10.48550/arXiv.1506.00312).
"""
import numpy as np
from . import utils
from .utils import Arm, Duel

# TODO: dts+
# TODO: beat the mean
# TODO: copeland dueling bandits (scb)

def naive(K: int, duel: Duel, T: int) -> Arm:
    """ Plays every pair of arms against each other and picks the winner. """
    wins = np.zeros((K, K), dtype=np.int64)
    for _ in range(T):
        # pick the pair that have been played the least
        nums = wins + wins.T
        np.fill_diagonal(nums, 2*T)
        index = np.argmin(nums)
        arm1, arm2 = np.unravel_index(index, nums.shape)
        # play them against each other and update statistics
        winner, loser = (arm1, arm2) if duel(arm1, arm2) else (arm2, arm1)
        wins[winner, loser] += 1
    return utils.copeland_winner_wins(wins)

# Double Thompson Sampling ([6])

def d_ts(
    K: int, duel: Duel, T: int,
    alpha: float,
    rng: np.random.Generator=np.random.default_rng(),
) -> Arm:
    """ The Double Thompson Sampling (D-TS) algorithm 1 of [6]. """
    B = np.zeros((K, K))
    for t in range(1, T + 1):
        # phase 1: choose the first candidate
        total = B + B.T
        mask = total != 0
        U = np.zeros((K, K))
        U[mask] = B[mask]/total[mask] + np.sqrt(alpha*np.log(t)/total[mask])
        # x/0 := 1 for all x
        U[~mask] = 2
        L = np.zeros((K, K))
        L[mask] = B[mask]/total[mask] - np.sqrt(alpha*np.log(t)/total[mask])
        L[~mask] = 0
        # u_ii = l_ii = 1/2
        np.fill_diagonal(U, 0.5)
        np.fill_diagonal(L, 0.5)
        # upper bound of the normalized Copeland score
        C = utils.copeland_winners(U)
        # sample from beta distribution
        lower = np.tril_indices(n=K, k=-1)
        theta = np.zeros((K, K))
        theta[lower] = rng.beta(B + 1, B.T + 1)[lower]
        theta = theta + (1 - theta.T)
        theta[lower] -= 1
        np.fill_diagonal(theta, 0.5)
        # choose only from C to eliminate non-winner arms; break ties randomly
        rng.shuffle(C)
        arm1 = C[np.argmax(utils.copeland_scores(theta)[C])]
        # phase 2: choose the second candidate
        theta2 = rng.beta(B[:, arm1] + 1, B[arm1, :] + 1)
        theta2[arm1] = 0.5
        # choosing only from uncertain pairs
        C2 = np.arange(K)[L[:, arm1] <= 0.5]
        arm2 = C2[np.argmax(theta2[C2])]
        # compare the pair, observe the outcome, and update B
        winner, loser = (arm1, arm2) if duel(arm1, arm2) else (arm2, arm1)
        B[winner, loser] += 1
    return utils.copeland_winner_wins(B)

# Tsallis-INF methods ([8, 5])

def learning_rate(t: int, rv: bool) -> float:
    """ Return the learning rate (theorem 1 of [8]). """
    return (
        4*np.sqrt(1/t) if rv else
        2*np.sqrt(1/t)
    )

def omd_w(x: float, losses: np.ndarray, lr: float) -> tuple[np.ndarray, float]:
    """ Return the weights for a given normalizing constant. """
    w = 4/np.square(lr*(losses - x))
    return w, np.sum(w) # type: ignore

def omd_monotone(
    losses: np.ndarray,
    lr: float,
    eps: float=1e-12,
) -> tuple[float, np.ndarray]:
    """
    Binary search for the normalizing factor.

    If the sum of the weights is greater than 1, the normalizing
    factor can be arbitrarily negative, which lowers the sum. If
    the sum of the weights is less than 1, then as the normalizing
    factor gets closer to the minimum loss, the sum will become
    arbitrarily large. This guarantees both monotonicity and that
    a solution will be in the range (-infty, min(losses)).

    Binary search only converges linearly while Newton's method is able
    to converge quadratically. However, Newton's method can diverge.

    For example, when the losses start out all 0, if x works, then -x also
    works. The Newton step can be too big, resulting in a value that is greater
    than min(losses) = 0 which is not in the range (-infty, 0). This poses
    a problem once a loss is introduced and the sum becomes bigger than 1.
    If x is increased then the 0 losses decrease in weight but the positive
    loss increases in weight, becoming arbitrarily large as x approaches
    its value. If x is decreased, the 0 losses increase in weight, becoming
    arbitrarily large as x approaches 0. Thus in either direction the weights
    become arbitrarily large. The method needs to cross a singularity at 0
    to become negative and its inability to do so causes divergence.
    """
    right = np.min(losses)
    # search for left boundary by doubling
    left = -1
    w, wsum = omd_w(left, losses, lr)
    while wsum > 1:
        left *= 2
        w, wsum = omd_w(left, losses, lr)
    # binary search
    middle = left
    while not np.isclose(wsum, 1, rtol=eps):
        middle = (left + right)/2
        w, wsum = omd_w(middle, losses, lr)
        if wsum > 1:
            right = middle
        else:
            left = middle
    return middle, w

def loss_estimator(loss: float, w: float, lr: float, rv: bool):
    """
    Compute importance-weighted (IW) or reduced-variance (RV) loss estimators.

    These estimators are an unbiased estimate for the loss.
    """
    b = 1/2*(w >= lr**2)
    return (
        (loss - b)/w + b if rv else
        loss/w
    )

def omd_newton(
    x: float,
    losses: np.ndarray,
    lr: float,
    eps: float=1e-12,
) -> tuple[float, np.ndarray]:
    """ Newton's method for the weights, algorithm 2 of [8]. """
    w, wsum = omd_w(x, losses, lr)
    while not np.isclose(wsum, 1, rtol=eps):
        x -= (wsum - 1)/(lr*np.sum(np.sqrt(w**3)))
        w, newsum = omd_w(x, losses, lr)
        # not making progress, switch to safe binary search
        if abs(newsum - 1) > abs(wsum - 1):
            return omd_monotone(losses, lr)
        wsum = newsum
    return x, w

def vdb_ind(
    K: int, duel: Duel, T: int,
    rv: bool=True,
    rng: np.random.Generator=np.random.default_rng(),
) -> Arm:
    """ The Versatile-DB (VDB) algorithm 3 of [5]. """
    losses = [np.zeros(K) for _ in range(2)]
    x = [-np.sqrt(K)]*2
    for t in range(1, T + 1):
        lr = learning_rate(t, rv)
        w = [np.empty(0)]*2
        arms = [0]*2
        for i in range(2):
            # choose from distribution (update x for warm start)
            x[i], w[i] = omd_newton(x[i], losses[i], lr)
            # sample arms
            arms[i] = rng.choice(K, p=w[i])
        # observe outcome
        outcome = duel(arms[0], arms[1])
        outcomes = [1 - outcome, outcome]
        for i in range(2):
            # construct unbiased estimator by IW or RV sampling
            loss = loss_estimator(outcomes[i], w[i][arms[i]], lr, rv)
            # update losses
            losses[i][arms[i]] += loss
    return np.argmin(losses[0] + losses[1])

def vdb(
    K: int, duel: Duel, T: int,
    rv: bool=True,
    rng: np.random.Generator=np.random.default_rng(),
) -> Arm:
    """ The Versatile-DB (VDB) algorithm 3 of [5], remark 2. """
    losses = np.zeros(K)
    x = -np.sqrt(K)
    for t in range(1, T + 1):
        lr = learning_rate(t, rv)
        # choose from distribution (update x for warm start)
        x, w = omd_newton(x, losses, lr)
        # sample arms
        arms = [rng.choice(K, p=w) for _ in range(2)]
        # observe outcome
        outcome = duel(arms[0], arms[1])
        outcomes = [1 - outcome, outcome]
        for i in range(2):
            # construct unbiased estimator by IW or RV sampling
            loss = loss_estimator(outcomes[i], w[arms[i]], lr, rv)
            # update losses: ignore remark 2's suggestion to divide by 2
            losses[arms[i]] += loss
    return np.argmin(losses)
