# Revised Simplex Solver

A small, self-contained linear programming solver written in Python with NumPy.
It implements three revised simplex-type algorithms and automatically picks the
right one for a given problem.

## What it does

Solves linear programs in standard form:

```
min  c^T x
s.t. A x = b
     x >= 0
```

It includes:

- **Revised primal simplex** — for a primal-feasible starting basis (`xB >= 0`).
- **Revised dual simplex** — for a dual-feasible starting basis (`sN >= 0`).
- **Two-phase method** — for problems with no feasible starting basis, using a
  single artificial variable to find one.

The "revised" form keeps and updates the basis inverse (`AB^-1`) directly via a
product-form update instead of re-inverting the basis every iteration.

## How it chooses an algorithm

From the initial basis the solver inspects the basic variables `xB` and the
reduced costs `sN` and dispatches automatically:

| `xB >= 0` | `sN >= 0` | Algorithm used         |
|-----------|-----------|------------------------|
| yes       | yes       | already optimal        |
| yes       | no        | primal simplex         |
| no        | yes       | dual simplex           |
| no        | no        | two-phase method       |

## Requirements

- Python 3
- NumPy

```
pip install numpy
```

## Usage

Edit the **PROBLEM DATA** block at the bottom of `simplex.py` with your own
problem, then run:

```
python simplex.py
```

### Setting up a problem

1. Convert every constraint to an equality by adding slack/surplus variables.
2. Fill in:
   - `c` — objective coefficients (one entry per variable, slacks included).
   - `A` — constraint matrix.
   - `b` — right-hand side.
   - `B` — indices (0-based columns of `A`) of the starting basis.
   - `N` — indices of the remaining (non-basic) variables.
3. For a **maximization** problem, negate the objective coefficients. The solver
   reports the minimum `z`; the true maximum is `-z`.

### Example

`max 3x1 + 2x2` subject to `x1 + x2 <= 4`, `2x1 + x2 <= 5`, `x1, x2 >= 0`.

Add slacks `x3, x4` and negate the objective:

```python
c = np.array([-3.0, -2.0, 0.0, 0.0])
A = np.array([[1.0, 1.0, 1.0, 0.0],
              [2.0, 1.0, 0.0, 1.0]])
b = np.array([4.0, 5.0])
B = [2, 3]   # slack columns form the identity basis
N = [0, 1]
```

Output:

```
Using primal simplex.
Optimal solution found.
Basis B = [1, 0]
xB = [3. 1.]
Min objective z = -9.0   (max = 9.0)
```

## Output

On success the solver prints the optimal basis `B`, the corresponding basic
variable values `xB`, and the objective value. It reports clearly when a problem
is **unbounded** or **infeasible**.
