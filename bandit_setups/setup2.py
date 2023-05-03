from functools import partial

import numpy as np

from bandit import algorithms, problems

rng = np.random.default_rng(1)

setup_name = "2"
K = 10
Ts = [*np.arange(1, 1000, 100), 10_000, 20_000]
trials = int(1e1)

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
    "dts+": partial(algorithms.d_ts, alpha=0.51, plus=True, rng=rng),
    "vdb-rv-shared": partial(algorithms.vdb, rv=True, rng=rng),
}
