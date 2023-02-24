import time

import numpy as np
import tqdm
from scipy.optimize import linprog
from scipy.sparse import coo_array

# x1 + x2 <= 1
# x1  + x2 == 1

t = 999
n = 200

directions = np.array(
    [
        [0, 0],  # stay
        [0, -1],  # up
        [1, 0],  # right
        [0, 1],  # down
        [-1, 0],  # left
    ]
)

times = []
for _ in tqdm.trange(t):
    pos = set()
    while len(pos) < n:
        pos.add(tuple(np.random.randint(48, size=2)))
    pos = np.array(list(pos))

    c = np.random.randn(5 * n)

    # pos[i]: (N, 2)

    start_time = time.perf_counter()
    # the starting positions which, after taking some move, end up at pos[i]
    start_pos = pos[:, None] - directions  # (N, 5, 2)
    flat_idx = 48 * start_pos[..., 0] + start_pos[..., 1]  # (N, 5)
    in_bounds = (
        (0 <= start_pos[..., 0])
        & (start_pos[..., 0] < 48)
        & (0 <= start_pos[..., 1])
        & (start_pos[..., 1] < 48)
    )  # (N, 5)

    a_lt = coo_array(
        (
            np.ones(in_bounds.sum()),
            (flat_idx[in_bounds].flatten(), np.arange(5 * n)[in_bounds.flatten()]),
        ),
        shape=(48 * 48, 5 * n)
    )
    b_lt = np.ones(48 * 48)

    i = np.arange(n).repeat(5)
    j = np.arange(5 * n)
    a_eq = coo_array(
        (
            np.ones(5 * n),
            (i, j)
        )
    )
    b_eq = np.ones(n)

    # print(c.shape)
    # print(a_lt.toarray().shape)
    # print(b_lt.shape)
    # print(a_eq.toarray().shape)
    # print(b_eq.shape)

    res = linprog(c, a_lt, b_lt, a_eq, b_eq, method="highs-ds")
    end_time = time.perf_counter() - start_time
    times.append(end_time)
    assert all(entry in (0, 1) for entry in np.unique(res.x))

print(np.percentile(times, 10) * 1000)
