from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bandit import algorithms, problems

# TODO: heatmap of picks and regret over time per problem, all algorithms

np.set_printoptions(precision=3, suppress=True)
sns.set_theme(context="paper", style="darkgrid")

rng = np.random.default_rng(1)

if __name__ == "__main__":
    K = 10
    Ts = [*np.arange(1, 1000, 100)]
    # Ts = [*np.arange(1, 1000, 100), 10_000, 20_000, 40_000, 80_000]
    trials = int(1e2)

    # K = 10
    # Ts = [100]
    # trials = 1

    prob_list = {
        "trivial": problems.RankingProblem(K, 100, rng=rng),
        # "easy": (
        #     problems.CondorcetProblem(
        #         problems.RandomProblem(K - 1, rng=rng),
        #     rng=rng)
        # ),
        # "medium": problems.RandomProblem(K, rng=rng),
        # "hard": problems.CopelandProblem(K, rng=rng),
    }
    alg_list = {
        "naive": algorithms.naive,
        # as [5] does in section 7
        "dts": partial(algorithms.d_ts, alpha=0.6, rng=rng),
        # "vdb-iw-independent": partial(algorithms.vdb_ind, rv=False, rng=rng),
        "vdb-iw-shared": partial(algorithms.vdb, rv=False, rng=rng),
        # "vdb-rv-independent": partial(algorithms.vdb_ind, rv=True, rng=rng),
        "vdb-rv-shared": partial(algorithms.vdb, rv=True, rng=rng),
    }
    data = {
        "problem": [],
        "algorithm": [],
        "time": [],
        "regret": [],
        "winner": [],
    }

    for problem_name, problem in prob_list.items():
        for algorithm_name, algorithm in alg_list.items():
            for T in Ts:
                for _ in range(trials):
                    problem.shuffle()
                    k, history = problems.run_problem(problem, algorithm, T)
                    data["problem"].append(problem_name)
                    data["algorithm"].append(algorithm_name)
                    data["time"].append(T)
                    data["regret"].append(problem.regret(history))
                    data["winner"].append(problem.is_winner(k))
    data = pd.DataFrame(data)
    # print(data)

    sns.relplot(
        data=data,
        x="time",
        y="regret",
        hue="algorithm",
        col="problem",
        col_wrap=2,
        kind="line",
    )
    plt.savefig("figures/regret.png")
    plt.clf()

    sns.relplot(
        data=data,
        x="time",
        y="winner",
        hue="algorithm",
        col="problem",
        col_wrap=2,
        kind="line",
    )
    plt.savefig("figures/winner.png")
