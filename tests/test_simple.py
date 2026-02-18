import numpy as np
from src.eigen import solve_eigen


def test_small_grid_sorted():
    vals, _ = solve_eigen(N=5, potential="well", n_eigs=3)
    assert len(vals) == 3
    assert np.all(np.diff(vals) >= 0), "Eigenvalues are not sorted"


def test_neigs_limit():
    # neigs must be <= N^2
    try:
        solve_eigen(N=5, potential="well", n_eigs=26)
        assert False, "Expected ValueError for neigs > N^2"
    except ValueError:
        pass
