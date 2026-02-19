import argparse
import numpy as np
from scipy.linalg import eigh


def build_2d_hamiltonian(
    N: int = 20,
    potential: str = "well",
    bc_a: float = 0.0,
    bc_b: float = 0.0,
) -> np.ndarray:
    """
    Build a discretized 2D Hamiltonian on an N x N grid using a 5-point stencil.

    Operator:  -∂^2/∂x^2 - ∂^2/∂y^2 + V(x,y)

    Boundary exercise (Section 7.2):
    - Apply an additional Dirichlet-like boundary term at the edges:
        f(x,y) = a*x + b*y
      controlled by (bc_a, bc_b). This is added ONLY on boundary grid points.
    """
    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer >= 2.")
    if potential not in {"well", "harmonic"}:
        raise ValueError("potential must be 'well' or 'harmonic'.")

    dx = 1.0 / float(N)
    inv_dx2 = 1.0 / (dx * dx)

    size = N * N
    H = np.zeros((size, size), dtype=np.float64)

    def idx(i: int, j: int) -> int:
        return i * N + j

    def xy_from_ij(i: int, j: int):
        # Centered coordinates in [-~0.5, ~0.5] depending on N
        x = (i - (N - 1) / 2.0) * dx
        y = (j - (N - 1) / 2.0) * dx
        return x, y

    def boundary_term(i: int, j: int) -> float:
        # Apply only at edges (Dirichlet-like boundary modification)
        if i == 0 or i == N - 1 or j == 0 or j == N - 1:
            x, y = xy_from_ij(i, j)
            return bc_a * x + bc_b * y
        return 0.0

    def V(i: int, j: int) -> float:
        # Base potential
        if potential == "well":
            base = 0.0
        elif potential == "harmonic":
            x, y = xy_from_ij(i, j)
            k = 4.0
            base = k * (x * x + y * y)
        else:
            base = 0.0

        return base + boundary_term(i, j)

    # Discrete (-∇^2): diagonal +4/dx^2, neighbors -1/dx^2
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


def solve_eigen(
    N: int = 20,
    potential: str = "well",
    n_eigs: int | None = None,
    bc_a: float = 0.0,
    bc_b: float = 0.0,
):
    """
    Build the 2D Hamiltonian and solve eigenvalues/eigenvectors.

    Returns eigenvalues sorted ascending.
    """
    if n_eigs is not None:
        if not isinstance(n_eigs, int) or n_eigs <= 0:
            raise ValueError("n_eigs must be a positive integer.")
        if n_eigs > N * N:
            raise ValueError("n_eigs must be <= N^2.")

    H = build_2d_hamiltonian(N=N, potential=potential, bc_a=bc_a, bc_b=bc_b)

    vals, vecs = eigh(H)
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

    # Existing Section 5 output option
    p.add_argument("--out", type=str, default=None, help="Optional output text file to save eigenvalues.")

    # Section 7.1: save ground-state probability density |psi|^2
    p.add_argument(
        "--save-psi2",
        type=str,
        default=None,
        help="Optional .npy output file to save ground-state |psi(x,y)|^2 (shape N x N).",
    )

    # Section 7.2: generalized boundary conditions term f(x,y)=a x + b y (edges only)
    p.add_argument("--bc-a", type=float, default=0.0, help="Boundary term coefficient a in f=a*x + b*y (edges).")
    p.add_argument("--bc-b", type=float, default=0.0, help="Boundary term coefficient b in f=a*x + b*y (edges).")

    return p.parse_args()


def main():
    args = parse_args()

    if args.N < 2:
        raise SystemExit("Error: N must be >= 2.")
    if args.neigs < 1 or args.neigs > args.N * args.N:
        raise SystemExit("Error: neigs must satisfy 1 <= neigs <= N^2.")

    vals, vecs = solve_eigen(
        N=args.N,
        potential=args.potential,
        n_eigs=args.neigs,
        bc_a=args.bc_a,
        bc_b=args.bc_b,
    )

    # Save eigenvalues (Section 5)
    if args.out is not None:
        np.savetxt(args.out, vals)
        print(f"Saved eigenvalues to {args.out}")

    # Save ground-state |psi|^2 (Section 7.1)
    if args.save_psi2 is not None:
        psi0 = vecs[:, 0].reshape(args.N, args.N)
        rho0 = np.abs(psi0) ** 2
        np.save(args.save_psi2, rho0)
        print(f"Saved ground-state |psi|^2 to {args.save_psi2}")

    print(f"N={args.N}, potential={args.potential}, neigs={args.neigs}, bc_a={args.bc_a}, bc_b={args.bc_b}")
    print("Lowest eigenvalues:")
    for k, v in enumerate(vals, start=1):
        print(f"{k:2d}: {v:.10f}")


if __name__ == "__main__":
    main()
