"""Tests for rational_function_fitting.core."""

import unittest
from fractions import Fraction

from rational_function_fitting import fit_rational_function


class TestFitRationalFunction(unittest.TestCase):
    def test_constant_over_constant(self):
        # y = 3, so p=3, q=1.
        p, q = fit_rational_function(
            [0, 1, 2], [3, 3, 3], numerator_degree=0, denominator_degree=0
        )
        self.assertEqual(p, (Fraction(3),))
        self.assertEqual(q, (Fraction(1),))

    def test_linear_numerator_constant_denominator(self):
        # y = 2x + 1, exact with degree (1, 0).
        p, q = fit_rational_function(
            [0, 1, 2], [1, 3, 5], numerator_degree=1, denominator_degree=0
        )
        self.assertEqual(p, (Fraction(1), Fraction(2)))
        self.assertEqual(q, (Fraction(1),))

    def test_genuine_rational_function(self):
        # y = (x + 1) / (x - 1), degree (1, 1).
        # Normalisation forces q_1 = 1, so q = (-1, 1).
        p, q = fit_rational_function(
            [0, 2, 3], [Fraction(-1), Fraction(3), Fraction(2)],
            numerator_degree=1, denominator_degree=1,
        )
        self.assertEqual(p, (Fraction(1), Fraction(1)))
        self.assertEqual(q, (Fraction(-1), Fraction(1)))

    def test_higher_degrees(self):
        # y = (x^2 + 1) / (x + 2) = x - 2 + 5/(x+2), degree (2, 1).
        # The implementation normalises q_1 = 1, so the returned
        # representation may be a scalar multiple of the canonical one.
        p, q = fit_rational_function(
            [0, 1, -1, 3],
            [Fraction(1, 2), Fraction(2, 3), Fraction(-2), Fraction(2)],
            numerator_degree=2, denominator_degree=1,
        )
        # Check that p/q matches the data exactly.
        for x, y in zip(
            [0, 1, -1, 3],
            [Fraction(1, 2), Fraction(2, 3), Fraction(-2), Fraction(2)],
        ):
            p_val = sum(c * (Fraction(x) ** i) for i, c in enumerate(p))
            q_val = sum(c * (Fraction(x) ** i) for i, c in enumerate(q))
            self.assertEqual(p_val, y * q_val)

    def test_float_input_is_converted_exactly(self):
        # 0.5 is exactly representable, so this should give clean fractions.
        p, q = fit_rational_function(
            [0.0, 1.0], [0.5, 1.5], numerator_degree=1, denominator_degree=0
        )
        self.assertEqual(p, (Fraction(1, 2), Fraction(1)))
        self.assertEqual(q, (Fraction(1),))

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            fit_rational_function([1, 2], [1], 0, 0)

    def test_negative_degree_raises(self):
        with self.assertRaises(ValueError):
            fit_rational_function([1], [1], -1, 0)
        with self.assertRaises(ValueError):
            fit_rational_function([1], [1], 0, -1)

    def test_insufficient_points_raises(self):
        # Degree (1, 1) has 4 unknowns, normalisation fixes one, so 3
        # points are needed; 2 should fail.
        with self.assertRaises(ValueError):
            fit_rational_function([1, 2], [1, 2], 1, 1)

    def test_extra_points_are_ignored(self):
        # If the first three points determine the function, extra points
        # should not change the result.
        p1, q1 = fit_rational_function(
            [0, 1, 2], [1, 2, 3], numerator_degree=1, denominator_degree=0
        )
        p2, q2 = fit_rational_function(
            [0, 1, 2, 100], [1, 2, 3, 101], numerator_degree=1, denominator_degree=0
        )
        self.assertEqual(p1, p2)
        self.assertEqual(q1, q2)

    def test_singular_system_raises(self):
        # A singular square system arises from linearly dependent rows,
        # e.g. points with repeated x but inconsistent y values.
        with self.assertRaises(ValueError):
            fit_rational_function(
                [1, 1, 1], [1, 2, 3], numerator_degree=1, denominator_degree=1
            )

    def test_result_evaluates_correctly(self):
        # Verify that p(x_i)/q(x_i) equals y_i for a small set.
        xs = [0, 1, 2, 3]
        ys = [Fraction(2), Fraction(3), Fraction(8, 3), Fraction(7, 2)]
        p, q = fit_rational_function(xs, ys, 2, 1)
        for x, y in zip(xs, ys):
            p_val = sum(c * (Fraction(x) ** i) for i, c in enumerate(p))
            q_val = sum(c * (Fraction(x) ** i) for i, c in enumerate(q))
            self.assertEqual(p_val, y * q_val)

    def test_fraction_input_is_exact(self):
        # Repeating decimal 1/3 should remain exact.
        third = Fraction(1, 3)
        p, q = fit_rational_function(
            [third, 1], [third + 1, 2], numerator_degree=1, denominator_degree=0
        )
        self.assertEqual(p, (Fraction(1), Fraction(1)))
        self.assertEqual(q, (Fraction(1),))


if __name__ == "__main__":
    unittest.main()
