import numpy as np
import scipy.linalg
import sklearn.gaussian_process.kernels as kernels
import zarr
from luxai_s2.state import State
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import Kernel
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import torch

BEST_COEFS = np.array([1, 1, -1, -1])
BEST_SCALES = np.array([1, 1, 1, 1])
BEST_SMOOTH = np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2])


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

    def fit(self, X, y) -> "SetupEstimator":  # pyright: ignore
        self.is_fitted_ = True
        return self

    def predict(self, board) -> np.ndarray:
        """Predict the factory scores for a board."""
        names = ["ice", "ore", "rubble", "factory_occupancy_map"]
        board_shape = board.ice.shape
        locs = np.indices(board_shape).reshape((2, np.product(board_shape))).T
        # fmt: off
        return torch.softmax(np.array([
            sum(
                (
                    coef
                    * kernel_regr(
                        locs,
                        getattr(board, name).flatten(),
                        loc[np.newaxis, :],
                        kernels.Matern(length_scale, smooth),
                    )
                    for name, coef, length_scale, smooth in zip(
                        names,
                        self.coefs,
                        self.length_scales,
                        self.smoothness,
                    )
                ),
                start=np.array([0]),
            ).item()
            for loc in locs
        ]), dim=0)
        # fmt: on


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


def factory_heuristic(state: State, loc: np.ndarray) -> float:
    """Given a (i, j)-th location, find its placement score."""
    Matern = kernels.Matern
    features = [
        ("ice", 1, Matern(length_scale=1, nu=1 / 2)),
        ("ore", 1, Matern(length_scale=1, nu=1 / 2)),
        ("rubble", -0.5, Matern(length_scale=1, nu=1 / 2)),
        ("factory_occupancy_map", -1, Matern(length_scale=1, nu=1 / 2)),
    ]
    board = state.board
    board_shape = board.ice.shape
    locs = np.indices(board_shape).reshape((2, np.product(board_shape))).T
    point = loc[np.newaxis, :]
    return sum(
        (
            coef
            * kernel_regr(locs, getattr(board, name).flatten(), point, kernel)
            for name, coef, kernel in features
        ),
        start=np.array([0]),
    ).item()


if __name__ == "__main__":
    # assert check_estimator(SetupEstimator()), "invalid estimator"

    array = zarr.open("replays/replay_data_zarr/replay_data.zarr")
    time_log1p = array["obs_meta"][: array.attrs["length"]][:, 2]
    starting_mask = np.isclose(time_log1p, 0, atol=1e-6)
    print(starting_mask.sum(), array.attrs["length"])
    # print(array["obs_tiles"])
