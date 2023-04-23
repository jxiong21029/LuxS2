import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import sklearn.gaussian_process.kernels as kernels
import zarr
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import Kernel
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from tqdm import tqdm

# from sklearn.utils.estimator_checks import check_estimator
# from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

np.random.seed(1)
rng = np.random.default_rng(1)

BOARD_SHAPE = (48, 48)

VERBOSE = True  # enable tqdm

# found by random search, mse was ...
BEST_COEFS = np.array([1, 1, -1])
BEST_SCALES = np.array([1, 1, 1])
BEST_SMOOTH = np.array([1 / 2, 1 / 2, 1 / 2])


class TrainBoard:
    """Board object for training purposes."""

    def __init__(self, board: np.ndarray) -> None:
        """Save slices of board as the right variables."""
        self.ice = board[0]
        self.ore = board[1]
        self.rubble = board[2]


class SampleIID:
    """Sample independently and identically (i.i.d.) from the distribution."""

    def __init__(self, rv, size: int = 1) -> None:
        """Take rv as either a scipy random variable or a list of options."""
        self.rv = rv
        self.size = size

    def rvs(
        self,
        random_state: np.random.Generator
        | np.random.RandomState
        | None = None,
    ):
        """Take size i.i.d. samples from the random variable."""
        random_state = rng if random_state is None else random_state
        return (
            random_state.choice(self.rv, size=self.size, replace=True)
            if isinstance(self.rv, list)
            else self.rv.rvs(size=self.size, random_state=random_state)
        )


class SetupEstimator(BaseEstimator):
    """Estimator for the setup phase factory placement."""

    def __init__(
        self,
        coefs: np.ndarray = BEST_COEFS,
        length_scales: np.ndarray = BEST_SCALES,
        smoothness: np.ndarray = BEST_SMOOTH,
    ) -> None:
        self.coefs = coefs
        self.length_scales = length_scales
        self.smoothness = smoothness

    def fit(self, X=None, y=None) -> "SetupEstimator":  # pyright: ignore
        # pre-compute kernel matrices
        locs = np.indices(BOARD_SHAPE).reshape((2, np.product(BOARD_SHAPE))).T
        self.__kernel_matrices = [  # pyright: ignore
            kernels.Matern(length_scale=length_scale, nu=smooth)(locs)
            for length_scale, smooth in zip(
                self.length_scales, self.smoothness
            )
        ]

        self.is_fitted_ = True
        return self

    def __predict(self, board) -> np.ndarray:
        """Predict the factory scores for a board."""
        names = [
            "ice",
            "ore",
            "rubble",
            # "factory_occupancy_map",
        ]
        scores: NDArray[np.double] = self.coefs @ np.array(
            [
                self.__kernel_matrices[i] @ getattr(board, name).flatten()
                for i, name in enumerate(names)
            ]
        )  # type: ignore
        diff = np.max(scores) - np.min(scores)
        return (scores - np.min(scores)) / (diff if diff != 0 else 1)

    def predict(self, boards, verbose: bool = VERBOSE):
        """Predict the factory scores for a board or multiple boards."""
        wrap = tqdm if verbose else lambda x: x
        return (
            np.array([self.__predict(board) for board in wrap(boards)])
            if isinstance(boards, list)
            else self.__predict(boards)
        )


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the system Ax = b for symmetric positive definite A."""
    return scipy.linalg.solve(A, b, assume_a="pos")


def gp_regr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kernel: Kernel,
) -> np.ndarray:
    """Estimate y_test with direct Gaussian process regression."""
    K_TT = kernel(x_train)
    K_TP = kernel(x_train, x_test)
    K_PT_TT = solve(K_TT, K_TP).T  # type: ignore

    mu_pred = K_PT_TT @ y_train
    return mu_pred


def kernel_regr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kernel: Kernel,
) -> np.ndarray:
    """Estimate y_test with kernel linear regression."""
    return kernel(x_test, x_train) @ y_train


if __name__ == "__main__":
    # assert check_estimator(SetupEstimator()), "invalid estimator"
    # number of hyperparameter settings to check
    ITERS = int(2e1)

    # data processing

    array = zarr.open("replays/replay_data_zarr/replay_data.zarr")
    time_log1p = array["obs_meta"][: array.attrs["length"]][:, 2]
    starting_mask = np.isclose(time_log1p, 0, atol=1e-6)  # type: ignore
    boards = [
        TrainBoard(np.array(array["obs_tiles"][i, :3]))
        for i in np.arange(array.attrs["length"])[starting_mask]
    ]
    factories = np.array(
        [
            array["obs_tiles"][i, 23:25]
            for i in np.arange(array.attrs["length"])[starting_mask]
        ]
    )
    total_factories = factories[:, 0] + factories[:, 1]
    filtered_factories = gaussian_filter(total_factories, sigma=(0, 2, 2))
    final_factories = filtered_factories.reshape(
        factories.shape[0], np.product(factories.shape[2:])
    )

    i = 0
    fig = plt.figure()
    plt.gray()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(total_factories[i].reshape(BOARD_SHAPE))
    plt.title("Factory placements")
    ax2.imshow(filtered_factories[i].reshape(BOARD_SHAPE))
    plt.title("Gaussian blurred")

    # plt.legend()
    plt.savefig("early_game/factory.png")

    # random search over hyperparameters

    features = 3
    param_grid = {
        "coefs": SampleIID(uniform(loc=-1, scale=2), size=features),
        "length_scales": SampleIID(uniform(loc=0, scale=20), size=features),
        "smoothness": SampleIID([1 / 2, 3 / 2, 5 / 2, np.inf], size=features),
    }
    random_search = RandomizedSearchCV(
        estimator=SetupEstimator(),
        param_distributions=param_grid,
        n_iter=ITERS,
        scoring="neg_mean_squared_error",
        cv=ShuffleSplit(
            test_size=len(boards) - 1, n_splits=1, random_state=0
        ),  # disable cross validation
        random_state=1,
    )
    search = random_search.fit(boards, final_factories)
    print(f"best parameters: {search.best_params_}")
    print(f"best mse: {-search.best_score_:.6f}")
