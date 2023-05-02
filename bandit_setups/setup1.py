from functools import partial

import numpy as np

from bandit import algorithms, problems

rng = np.random.default_rng(1)

setup_name = "1"
K = 10
Ts = [*np.arange(1, 1000, 100)]
# Ts = [*np.arange(1, 1000, 100), 10_000, 20_000, 40_000, 80_000]
trials = int(5e1)

prob_list = {
    "trivial": problems.RankingProblem(K, 100, rng=rng),
    "easy": (
        problems.CondorcetProblem(
            problems.RandomProblem(K - 1, rng=rng), rng=rng
        )
    ),
    "medium": problems.CopelandProblem(K, rng=rng),
    "hard": problems.RandomProblem(K, rng=rng),
}
alg_list = {
    "naive": algorithms.naive,
    "beat-the-mean": partial(algorithms.btm_online, gamma=1, rng=rng),
    # "scb": partial(algorithms.scb, rng=rng),
    "dts": partial(algorithms.d_ts, alpha=0.51, plus=False, rng=rng),
    "dts+": partial(algorithms.d_ts, alpha=0.51, plus=True, rng=rng),
    "vdb-iw-independent": partial(algorithms.vdb_ind, rv=False, rng=rng),
    "vdb-iw-shared": partial(algorithms.vdb, rv=False, rng=rng),
    "vdb-rv-independent": partial(algorithms.vdb_ind, rv=True, rng=rng),
    "vdb-rv-shared": partial(algorithms.vdb, rv=True, rng=rng),
}
