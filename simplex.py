import numpy as np

# ============================================================
# REVISED PRIMAL SIMPLEX ALGORITHM
#
# Solves problems in standard form:
#   min { c^T x : Ax = b, x >= 0 }
#
# For MAX problems: negate c before running, then negate z at
# the end to recover the true maximum value.
# ============================================================

# ----------------------------------------------------------
# STEP 0 — INITIALIZATION
#
# Define the problem data and choose an initial basis B.
# B must index m linearly independent columns of A so that
# the basis matrix AB = A[:, B] is invertible (rank(AB) = m).
#
# This example encodes:   max 3x1 + x2
#                  s.t.   x1 +  x2 <= 2
#                         2x1 + x2 <= 4
#                         x1, x2 >= 0
#
# After adding slack variables x3 and x4 the standard form is:
#   min -3x1 - x2  (c negated to convert max -> min)
#   s.t. x1 + x2 + x3       = 2
#        2x1 + x2      + x4  = 4
#        x1, x2, x3, x4 >= 0
# ----------------------------------------------------------

prev_pivot = None
prev_k = None
prev_sl = None

c = np.array([-100, -10, -1, 0, 0, 0])   # cost vector (negated for max->min)

A = np.array([
    [1, 0, 0, 1, 0, 0],
    [20, 1, 0, 0, 1, 0],
    [200, 20, 1, 0, 0, 1],
])

b = np.array([1, 100,  10000])

# Initial basis: columns 2 and 3 (0-indexed) correspond to the
# slack variables x3 and x4. AB = I (identity), which is trivially
# invertible and gives the initial basic feasible solution xB = b >= 0.
B = [3, 4, 5]   # basis index set  (|B| = m)
N = [0, 1, 2]   # non-basis index set  (|N| = n - m)

# Compute the inverse of the basis matrix AB^{-1}
AB_inv = np.eye(3)

count = 0
# ----------------------------------------------------------
# MAIN LOOP — each iteration is one pivot
# ----------------------------------------------------------
while True:

    # Partition c and A according to the current basis B and non-basis N.
    # cB, cN  — cost sub-vectors for basic / non-basic variables
    # AB, AN  — sub-matrices of A for basic / non-basic columns
    cB = c[B]
    cN = c[N]
    AB = A[:, B]
    AN = A[:, N]

    # Current basic feasible solution: xB = AB^{-1} b
    # Non-basic variables are all zero: xN = 0
    xB = AB_inv @ b

    # Dual variables (shadow prices):  w^T = (cB)^T * AB^{-1}
    w_T = cB @ AB_inv

    # Dual slack variables (reduced costs) for non-basic columns:
    #   sN = (cN)^T - (cB)^T * AB^{-1} * AN
    sN = cN - w_T @ AN
    if prev_k is not None:
        sk_new = sN[t]
        print("Στοιχεια")
        print(f"Παλιο sL: {prev_sl}")
        print(f"Pivot element: {prev_pivot}")
        print(f"Νεο sk: {sk_new}")

    # ----------------------------------------------------------
    # STEP 1 — OPTIMALITY TEST
    #
    # Theorem: the current basic solution (xB, xN=0)
    # is optimal for the min problem if:
    #   xB >= 0  (feasibility, guaranteed by construction)  AND
    #   sN >= 0  (all reduced costs non-negative)
    #
    # When both hold the complementary slackness conditions xj*sj = 0
    # are satisfied, confirming optimality.
    # ----------------------------------------------------------
    if np.min(sN) >= 0:
        z_min = cB @ xB
        print("Optimal solution found.")
        print(f"Basic variable indices B = {B}")
        print(f"Basic variable values  xB = {xB}")
        print(f"Min objective value  z = {z_min}")
        print(f"Max objective value  z = {-z_min}  (negate because max->min)")
        print(f"Iterations: {count}")
        break

    # ----------------------------------------------------------
    # STEP 2 — CHOOSE ENTERING VARIABLE
    #
    # Dantzig's rule / Least-element rule:
    #   Choose the non-basic index l whose reduced cost s_l is the
    #   most negative (largest improvement per unit step):
    #       s_l = min { sj : sj < 0,  j in N }
    #
    # t  = position of l inside the N list
    # l  = actual variable index that will ENTER the basis
    # ----------------------------------------------------------
    t = int(np.argmin(sN))   # position in N of the most negative reduced cost
    l = N[t]                 # index of the entering variable x_l

    # Pivot column: hl = AB^{-1} * A_{.l}
    # hl[i] tells how much the i-th basic variable decreases when x_l increases by 1.
    hl = AB_inv @ A[:, l]

    # ----------------------------------------------------------
    # UNBOUNDEDNESS CHECK
    #
    # If every component of hl is <= 0, the objective can decrease
    # without bound along the direction of x_l — the LP is unbounded.
    # ----------------------------------------------------------
    if np.max(hl) <= 0:
        print("Problem is unbounded.")
        print(count)
        break

    # ----------------------------------------------------------
    # STEP 2 (cont.) — CHOOSE LEAVING VARIABLE
    #
    # Minimum-ratio test:
    #   Find the basic variable x_{B(r)} that hits zero first as x_l grows:
    #       x_{B(r)} / h_{rl} = min { xB[i] / hl[i] : hl[i] > 0,  i = 1..m }
    #
    # Only rows where hl[i] > 0 are eligible (dividing by <= 0 would
    # allow the variable to stay non-negative or go to infinity).
    #
    # r  = row index of the leaving variable inside B
    # k  = actual variable index that will LEAVE the basis
    # ----------------------------------------------------------
    ratios = np.full(xB.shape, np.inf)   # initialise all ratios to +inf
    mask = hl > 0                         # only consider positive pivot elements
    ratios[mask] = xB[mask] / hl[mask]

    r = int(np.argmin(ratios))   # row position of the leaving variable in B
    k = B[r]                     # index of the leaving variable x_k

    #Calculate inverse with E matrix to reduce time
    m = len(AB_inv)
    pivot = hl[r]

    E_inv = np.eye(m)
    E_inv[:, r] = -hl/ pivot

    E_inv[r, r] = 1 / pivot

    AB_inv = E_inv @ AB_inv

    # ----------------------------------------------------------
    # STEP 3 — PIVOT
    #
    # Swap the entering index l into position r of B and
    # the leaving index k into position t of N.
    # The next iteration will recompute AB^{-1} from scratch.
    # ----------------------------------------------------------
    prev_pivot = pivot
    prev_sl = sN[t]
    prev_k = k

    B[r] = l
    N[t] = k
    # Loop continues with the updated basis.
    count += 1



