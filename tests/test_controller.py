"""Regression and behavior tests for the fuzzy controllers."""

import unittest

from main import (
    calculate_feedback,
    determine_feedback_label,
    get_recommendation,
    validate_scale_value,
)
from parameter_comparison import evaluate_models
from scenario_experiments import evaluate_scenarios


class BaselineControllerTests(unittest.TestCase):
    """Verify stable reference values and public helper behavior."""

    def test_report_reference_case(self) -> None:
        output = calculate_feedback(8.0, 7.0)
        self.assertAlmostEqual(output, 84.86, places=2)
        self.assertEqual(determine_feedback_label(output), "strong")

    def test_representative_outputs(self) -> None:
        expected = [13.17, 13.17, 50.00, 13.17, 50.00, 86.83, 50.00, 86.83, 86.83]
        actual = [row["feedback_intensity"] for row in evaluate_scenarios()]
        self.assertEqual(actual, expected)

    def test_controller_is_symmetric(self) -> None:
        self.assertAlmostEqual(
            calculate_feedback(3.5, 7.0),
            calculate_feedback(7.0, 3.5),
            places=9,
        )

    def test_representative_grid_is_monotonic(self) -> None:
        rows = evaluate_scenarios()
        matrix = [
            [rows[row * 3 + column]["feedback_intensity"] for column in range(3)]
            for row in range(3)
        ]
        for row in matrix:
            self.assertEqual(row, sorted(row))
        for column in range(3):
            values = [matrix[row][column] for row in range(3)]
            self.assertEqual(values, sorted(values))

    def test_invalid_input_is_rejected(self) -> None:
        for invalid_value in (-0.01, 10.01, float("inf"), float("nan")):
            with self.subTest(value=invalid_value), self.assertRaises(ValueError):
                validate_scale_value(invalid_value, "Test input")

    def test_unknown_recommendation_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_recommendation("unknown")


class ModifiedControllerTests(unittest.TestCase):
    """Verify the intended effect of the modified memberships."""

    def test_modified_outputs_are_not_lower(self) -> None:
        for row in evaluate_models():
            with self.subTest(scenario=row["scenario"]):
                self.assertGreaterEqual(
                    row["modified_output"],
                    row["baseline_output"],
                )

    def test_largest_reported_increase(self) -> None:
        largest_difference = max(row["difference"] for row in evaluate_models())
        self.assertAlmostEqual(largest_difference, 13.43, places=2)


if __name__ == "__main__":
    unittest.main()
