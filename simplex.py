import numpy as np

# ============================================================
# Revised simplex-type algorithms for linear programming
#   - Revised primal simplex            (Lectures 05 / 06)
#   - Revised dual simplex              (Lecture 07)
#   - Two-phase method (single          (Lecture 07)
#     artificial variable)
#
# All problems are solved in standard form:
#       min { c^T x : A x = b, x >= 0 }
#
# For a MAX problem, negate the objective coefficients before
# solving; the reported minimum z then equals -(max), so the
# true maximum is -z.
# ============================================================


def update_basis_inverse(AB_inv, hl, r):
    """Product-form update of the basis inverse after pivoting on row r.

    AB_inv_new = E^-1 @ AB_inv  (Lecture 06, "ΑΝΑΝΕΩΣΗ ΤΗΣ ΒΑΣΗΣ").
    Only the elementary matrix E^-1 is built, avoiding a full re-inversion.
    """
    m = len(AB_inv)
    pivot = hl[r]
    E_inv = np.eye(m)
    E_inv[:, r] = -hl / pivot
    E_inv[r, r] = 1 / pivot
    return E_inv @ AB_inv


def simplex(c, A, b, B, N, AB_inv):
    """Revised primal simplex algorithm (Lecture 06).

    Starts from a primal-feasible basis (xB >= 0) and returns the optimal
    partition, or None if the problem is unbounded.
    """
    while True:
        # Step 0 quantities for the current basis [B, N]
        cB, cN, AN = c[B], c[N], A[:, N]
        xB = AB_inv @ b                 # basic variable values
        w_T = cB @ AB_inv               # dual variables   w^T = cB^T AB^-1
        sN = cN - w_T @ AN              # reduced costs    sN = cN^T - w^T AN

        # Step 1 - optimality test:  sN >= 0  =>  optimal
        if np.min(sN) >= 0:
            return B, N, xB, AB_inv

        # Step 2 - entering variable (Dantzig rule: most negative reduced cost)
        t = int(np.argmin(sN))
        l = N[t]
        hl = AB_inv @ A[:, l]           # pivot column  hl = AB^-1 A_.l

        # Unboundedness check: hl <= 0 means x_l can grow without bound
        if np.max(hl) <= 0:
            print("Problem is unbounded.")
            return None

        # Step 2 - leaving variable (minimum-ratio test over hl[i] > 0)
        ratios = np.full(xB.shape, np.inf)
        mask = hl > 0
        ratios[mask] = xB[mask] / hl[mask]
        r = int(np.argmin(ratios))
        k = B[r]

        # Step 3 - pivot: x_l enters, x_k leaves; refresh the basis inverse
        AB_inv = update_basis_inverse(AB_inv, hl, r)
        B[r], N[t] = l, k


def dual_simplex(c, A, b, B, N, AB_inv):
    """Revised dual simplex algorithm (Lecture 07).

    Starts from a dual-feasible basis (sN >= 0) and returns the optimal
    partition, or None if the primal problem is infeasible.
    """
    while True:
        # Step 0 quantities for the current basis [B, N]
        cB, cN, AN = c[B], c[N], A[:, N]
        xB = AB_inv @ b
        w_T = cB @ AB_inv
        sN = cN - w_T @ AN

        # Step 1 - optimality test:  xB >= 0  =>  optimal
        if np.min(xB) >= 0:
            return B, N, xB, AB_inv

        # Step 2 - leaving variable (most negative basic variable)
        r = int(np.argmin(xB))
        k = B[r]
        H_rN = AB_inv[r, :] @ AN        # pivot row  H_rN = AB^-1(r) AN

        # H_rN >= 0  =>  dual unbounded, primal infeasible
        if np.min(H_rN) >= 0:
            print("Primal problem is infeasible (dual is unbounded).")
            return None

        # Step 2 - entering variable (dual ratio test over H_rj < 0)
        ratios = np.full(sN.shape, np.inf)
        mask = H_rN < 0
        ratios[mask] = -sN[mask] / H_rN[mask]
        t = int(np.argmin(ratios))
        l = N[t]

        # Step 3 - pivot: x_l enters, x_k leaves; refresh the basis inverse
        hl = AB_inv @ A[:, l]
        AB_inv = update_basis_inverse(AB_inv, hl, r)
        B[r], N[t] = l, k


def two_phase_simplex(c, A, b, B, N, AB_inv):
    """Two-phase method with a single artificial variable (Lecture 07).

    Phase I solves a modified problem (T.G.P.) that drives the artificial
    variable to zero; Phase II solves the original problem (A.G.P.) from the
    feasible basis produced by Phase I.
    """
    m = len(B)
    n = A.shape[1]                      # number of structural variables
    art = n                            # index of the artificial variable x_{n+1}

    # --- Step 0: build the modified problem (T.G.P.) ---
    # Artificial column d = -A_B e, so that AB^-1 d = -e (an all-ones, all-negative
    # pivot column). The Phase I objective minimises only the artificial variable.
    AB = A[:, B]
    d = (-AB @ np.ones(m)).reshape(-1, 1)
    A = np.hstack([A, d])
    c = np.append(c, 0.0)
    f = np.zeros(n + 1)
    f[art] = 1.0

    # --- Step 1: pivot the artificial variable into the basis ---
    # Leaving variable: the most infeasible basic variable (smallest xB[i]).
    xB = AB_inv @ b
    r = int(np.argmin(xB))
    k = B[r]
    hl = AB_inv @ A[:, art]
    AB_inv = update_basis_inverse(AB_inv, hl, r)
    B[r] = art
    N = np.append(N, k)

    # --- Step 2: Phase I (solve the T.G.P. with the primal simplex) ---
    B, N, xB, AB_inv = simplex(f, A, b, B, N, AB_inv)

    # --- Transition to Phase II ---
    if art in B:
        idx = list(B).index(art)
        if xB[idx] > 1e-9:
            # x_{n+1} > 0 at the optimum: the original problem is infeasible.
            print("Problem is infeasible.")
            return None
        # x_{n+1} = 0 but still basic: pivot it out, any eligible variable enters.
        row = AB_inv[idx, :] @ A[:, N]
        t = next(j for j, h in enumerate(row) if abs(h) > 1e-9)
        l = N[t]
        hl = AB_inv @ A[:, l]
        AB_inv = update_basis_inverse(AB_inv, hl, idx)
        B[idx] = l
        N = np.delete(N, t)
    else:
        # x_{n+1} is already non-basic: simply drop it from N.
        N = N[N != art]

    # --- Step 3: Phase II (solve the original A.G.P. with the primal simplex) ---
    return simplex(c, A, b, B, N, AB_inv)


# ============================================================
# PROBLEM DATA  --  edit this block for your own problem
#
# Put the problem in standard form first:  min c^T x, A x = b, x >= 0.
# Add slack/surplus variables yourself so every constraint is an equality.
# For a MAX problem, negate the objective coefficients here (max = -z).
#
# Example:  max 3x1 + 2x2
#           s.t.  x1 +  x2 <= 4
#                2x1 +  x2 <= 5,   x1, x2 >= 0
# Adding slacks x3, x4 and negating the objective (max -> min):
#   min -3x1 - 2x2
#   s.t. x1 +  x2 + x3      = 4
#       2x1 +  x2      + x4 = 5,   x1..x4 >= 0
# ============================================================

c = np.array([-3.0, -2.0, 0.0, 0.0])          # objective coefficients
A = np.array([[1.0, 1.0, 1.0, 0.0],           # constraint matrix
              [2.0, 1.0, 0.0, 1.0]])
b = np.array([4.0, 5.0])                       # right-hand side

# Initial basis / non-basis index sets (0-indexed columns of A).
# Here the slack columns x3, x4 give an identity basis (AB = I).
B = [2, 3]                                     # basic variable indices
N = [0, 1]                                     # non-basic variable indices

# ----------------------------------------------------------
# Driver: pick the right algorithm from the initial partition
# ----------------------------------------------------------
AB_inv = np.linalg.inv(A[:, B])
xB = AB_inv @ b
sN = c[N] - (c[B] @ AB_inv) @ A[:, N]

if np.min(xB) >= 0 and np.min(sN) >= 0:
    print("Initial basis is already optimal.")
    result = (B, N, xB, AB_inv)
elif np.min(xB) >= 0:
    print("Using primal simplex.")
    result = simplex(c, A, b, B, N, AB_inv)
elif np.min(sN) >= 0:
    print("Using dual simplex.")
    result = dual_simplex(c, A, b, B, N, AB_inv)
else:
    print("Using two-phase method.")
    result = two_phase_simplex(c, A, b, B, N, AB_inv)

if result is not None:
    B, N, xB, AB_inv = result
    z = c[B] @ xB
    print("Optimal solution found.")
    print(f"Basis B = {list(B)}")
    print(f"xB = {xB}")
    print(f"Min objective z = {z}   (max = {-z})")
