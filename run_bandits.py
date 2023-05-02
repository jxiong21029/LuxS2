import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bandit import problems
from bandit_setups.setup1 import Ts, alg_list, prob_list, setup_name, trials

# TODO: heatmap of picks and regret over time per problem, all algorithms

np.set_printoptions(precision=3, suppress=True)
sns.set_theme(context="paper", style="darkgrid")

if __name__ == "__main__":
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
    plt.savefig(f"figures/regret_{setup_name}.png")
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
    plt.savefig(f"figures/winner_{setup_name}.png")
