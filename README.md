# Rational Function Fitting

Fits a rational function p(x)/q(x) to sample data using exact linear algebra over `fractions.Fraction`.

## Usage

```python
from rational_function_fitting import fit_rational_function

# Fit y = (x + 1) / (x - 1) to three points.
p, q = fit_rational_function(
    [0, 2, 3],
    [-1, 3, 2],
    numerator_degree=1,
    denominator_degree=1,
)
print(p)  # (Fraction(1, 1), Fraction(1, 1))
print(q)  # (Fraction(-1, 1), Fraction(1, 1))
```

The returned tuples contain the coefficients in increasing powers of x.  The denominator is normalised so that its leading coefficient is 1.

## Why this exists

Rational functions can represent curves that polynomials cannot, such as functions with vertical asymptotes.  Fitting one exactly to data is a linear problem when the equation `p(x_i) = y_i * q(x_i)` is used, but floating-point solvers introduce rounding errors that can change the result.  This library uses `fractions.Fraction` for every arithmetic operation, so the fit is exact for rational input data.  The trade-off is that performance is much lower than a floating-point implementation, and the input values are interpreted exactly: a Python `float` is converted from its binary representation, not from its decimal representation.

## Edge cases

The normalisation `q_lead = 1` means the denominator may vanish at some x values; no attempt is made to avoid poles in the data range.  If the data points are not enough to determine the unknowns, a `ValueError` is raised.  If more points are provided than needed, only the first `(numerator_degree + 1) + (denominator_degree + 1) - 1` points are used; the rest are ignored.  This is a deliberate choice to keep the API simple and exact.  If a least-squares fit is desired, pre-process the data before calling this function.
