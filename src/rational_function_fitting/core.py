"""Core fitting routine for rational function approximation.

The problem is to find polynomials p and q such that

    p(x_i) - y_i * q(x_i) = 0

for all data points (x_i, y_i).  This is a linear system in the
coefficients of p and q.  We set the least significant coefficient of
q that is not forced by a normalisation condition to a fixed value,
solve the remaining square system, and return p and q.

The implementation uses fractions.Fraction everywhere so that the
result is exact for rational input data.  Floating point input is
converted exactly from its binary representation; if approximate
behaviour is desired, convert to Fraction before calling.
"""

from fractions import Fraction
from typing import Iterable, Sequence, Tuple


def _frac(value) -> Fraction:
    """Convert a value to Fraction exactly.

    Fractions pass through unchanged; integers and floats are converted
    by the standard Fraction constructor.  Floats are converted from
    their binary representation, which is deterministic but may surprise
    users expecting decimal rounding.
    """
    if isinstance(value, Fraction):
        return value
    return Fraction(value)


def _solve_linear_system(matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]) -> list[Fraction]:
    """Solve a square linear system exactly using Gaussian elimination.

    The matrix must be square and non-singular.  The returned vector is a
    list of Fractions.
    """
    n = len(matrix)
    if n == 0:
        return []

    # Work on a copy because elimination mutates the matrix.
    a = [list(row) for row in matrix]
    b = list(rhs)

    for col in range(n):
        # Find a pivot row with a non-zero entry in this column.
        pivot = None
        for row in range(col, n):
            if a[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            raise ValueError("singular system")

        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            b[col], b[pivot] = b[pivot], b[col]

        # Normalise pivot row.
        pivot_value = a[col][col]
        a[col] = [entry / pivot_value for entry in a[col]]
        b[col] = b[col] / pivot_value

        # Eliminate this column from all other rows.
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor == 0:
                continue
            a[row] = [
                a[row][j] - factor * a[col][j]
                for j in range(n)
            ]
            b[row] = b[row] - factor * b[col]

    return b


def fit_rational_function(
    x_values: Sequence[float | int | Fraction],
    y_values: Sequence[float | int | Fraction],
    numerator_degree: int,
    denominator_degree: int,
) -> Tuple[Tuple[Fraction, ...], Tuple[Fraction, ...]]:
    """Fit a rational function p(x)/q(x) to the given data.

    Args:
        x_values: x coordinates of the data points.
        y_values: corresponding y coordinates.
        numerator_degree: degree of p (non-negative integer).
        denominator_degree: degree of q (non-negative integer).

    Returns:
        A tuple (p_coeffs, q_coeffs) where p_coeffs[i] is the
        coefficient of x**i in p, and q_coeffs[i] is the coefficient
        of x**i in q.  Both tuples have length degree + 1.

    Raises:
        ValueError: if the inputs have different lengths, if either
            degree is negative, if there are not enough data points to
            determine the unknowns, or if the resulting system is
            singular.

    The normalisation is chosen so that the coefficient of the highest
    power of x in the denominator is 1.  This makes the representation
    unique for a given rational function, but it means that the fitted
    denominator may evaluate to zero at some x values.  No attempt is
    made to avoid poles in the data range.
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if numerator_degree < 0 or denominator_degree < 0:
        raise ValueError("degrees must be non-negative")

    n_points = len(x_values)
    n_unknowns = (numerator_degree + 1) + (denominator_degree + 1)
    # One unknown is fixed by normalisation, so we need at least
    # n_unknowns - 1 equations.  Extra data points produce an
    # overdetermined system; we use only the first n_unknowns - 1
    # points and ignore the rest.  This is a deliberate design choice:
    # the caller can filter or weight data before calling if a
    # least-squares fit is desired.
    required_points = n_unknowns - 1
    if n_points < required_points:
        raise ValueError(
            f"need at least {required_points} data points for "
            f"numerator_degree={numerator_degree}, "
            f"denominator_degree={denominator_degree}"
        )

    # Work with exact Fractions from the start.
    xs = [_frac(x) for x in x_values[:required_points]]
    ys = [_frac(y) for y in y_values[:required_points]]

    # Build the homogeneous linear system.
    # Unknowns are p_0, ..., p_n, q_0, ..., q_{m-1}; q_m is fixed to 1.
    # Equation for point (x, y):
    #   sum_{i=0}^n p_i x^i - y * sum_{i=0}^{m-1} q_i x^i = y * x^m
    n = numerator_degree
    m = denominator_degree
    matrix = []
    rhs = []
    for x, y in zip(xs, ys):
        row = []
        # p coefficients: 1, x, x^2, ..., x^n
        x_power = Fraction(1)
        for _ in range(n + 1):
            row.append(x_power)
            x_power *= x
        # q coefficients (excluding q_m): -y, -y*x, ..., -y*x^{m-1}
        yx_power = -y
        for _ in range(m):
            row.append(yx_power)
            yx_power *= x
        matrix.append(row)
        # Right-hand side: y * x^m
        rhs.append(y * (x ** m))

    solution = _solve_linear_system(matrix, rhs)

    # solution has n+1 numerator coefficients followed by m denominator
    # coefficients; append the fixed q_m = 1.
    p_coeffs = tuple(solution[: n + 1])
    q_coeffs = tuple(solution[n + 1 :]) + (Fraction(1),)

    return p_coeffs, q_coeffs
