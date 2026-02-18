import argparse
import numpy as np
from scipy.linalg import eigh


def build_2d_hamiltonian(N: int = 20, potential: str = "well") -> np.ndarray:
    """
    Build a discretized 2D Hamiltonian on an N x N grid using a 5-point stencil.

    We approximate:  -∂^2/∂x^2 - ∂^2/∂y^2 + V(x,y)
    on a uniform grid. The resulting matrix is (N^2) x (N^2).

    Notes
    -----
    - This uses a simple finite-difference Laplacian with Dirichlet-like handling
      by simply not coupling outside the domain.
    - Units/scales are arbitrary for this assignment; we choose dx = 1/N.
    """
    if not isinstance(N, int) or N <= 1:
        raise ValueError("N must be an integer >= 2.")
    if potential not in {"well", "harmonic"}:
        raise ValueError("potential must be 'well' or 'harmonic'.")

    dx = 1.0 / float(N)
    inv_dx2 = 1.0 / (dx * dx)  # = N^2

    size = N * N
    H = np.zeros((size, size), dtype=np.float64)

    def idx(i: int, j: int) -> int:
        return i * N + j

    def V(i: int, j: int) -> float:
        if potential == "well":
            # "Infinite well" interior: V = 0 (boundary conditions not explicitly enforced here)
            return 0.0
        elif potential == "harmonic":
            # Centered 2D harmonic oscillator: V = k (x^2 + y^2)
            x = (i - (N - 1) / 2.0) * dx
            y = (j - (N - 1) / 2.0) * dx
            k = 4.0
            return k * (x * x + y * y)
        return 0.0

    # Discrete operator for (-d2/dx2 - d2/dy2):
    # diagonal: +4/dx^2, neighbors: -1/dx^2
    for i in range(N):
        for j in range(N):
            row = idx(i, j)

            H[row, row] = 4.0 * inv_dx2 + V(i, j)

            if i > 0:
                H[row, idx(i - 1, j)] = -inv_dx2
            if i < N - 1:
                H[row, idx(i + 1, j)] = -inv_dx2
            if j > 0:
                H[row, idx(i, j - 1)] = -inv_dx2
            if j < N - 1:
                H[row, idx(i, j + 1)] = -inv_dx2

    return H


def solve_eigen(N: int = 20, potential: str = "well", n_eigs: int | None = None):
    """
    Solve for eigenvalues/eigenvectors of the discretized Hamiltonian.

    Returns eigenvalues in ascending order.
    """
    if n_eigs is not None:
        if not isinstance(n_eigs, int) or n_eigs <= 0:
            raise ValueError("n_eigs must be a positive integer.")
        if n_eigs > N * N:
            raise ValueError("n_eigs must be <= N^2.")

    H = build_2d_hamiltonian(N, potential)

    vals, vecs = eigh(H)  # full spectrum (OK for moderate N)
    # eigh already returns sorted eigenvalues for symmetric/Hermitian matrices,
    # but we sort anyway to be safe across implementations:
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    if n_eigs is None:
        return vals, vecs
    return vals[:n_eigs], vecs[:, :n_eigs]


def parse_args():
    p = argparse.ArgumentParser(description="2D Hamiltonian eigenvalue solver on an NxN grid.")
    p.add_argument("--N", type=int, default=10, help="Grid size in each dimension (N>=2).")
    p.add_argument(
        "--potential",
        type=str,
        default="well",
        choices=["well", "harmonic"],
        help="Potential type.",
    )
    p.add_argument("--neigs", type=int, default=5, help="Number of lowest eigenvalues to print (1..N^2).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.N < 2:
        raise SystemExit("Error: N must be >= 2.")
    if args.neigs < 1 or args.neigs > args.N * args.N:
        raise SystemExit("Error: neigs must satisfy 1 <= neigs <= N^2.")

    vals, _ = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.neigs)
    print(f"N={args.N}, potential={args.potential}, neigs={args.neigs}")
    print("Lowest eigenvalues:")
    for k, v in enumerate(vals, start=1):
        print(f"{k:2d}: {v:.10f}")


if __name__ == "__main__":
    main()
