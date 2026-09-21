"""Evaluate the nine representative fuzzy-rule scenarios."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np

from main import RESULTS_DIRECTORY, calculate_feedback, determine_feedback_label

LEVEL_NAMES = ("Low", "Medium", "High")
LEVEL_VALUES = (1.0, 5.0, 9.0)


class ScenarioResult(TypedDict):
    """One representative controller evaluation."""

    scenario: str
    stress: float
    arousal: float
    feedback_intensity: float
    category: str


def evaluate_scenarios() -> list[ScenarioResult]:
    """Evaluate all nine low, medium, and high input combinations."""

    results: list[ScenarioResult] = []
    for stress_name, stress_value in zip(LEVEL_NAMES, LEVEL_VALUES):
        for arousal_name, arousal_value in zip(LEVEL_NAMES, LEVEL_VALUES):
            feedback_value = calculate_feedback(stress_value, arousal_value)
            results.append(
                {
                    "scenario": f"{stress_name} stress / {arousal_name} arousal",
                    "stress": stress_value,
                    "arousal": arousal_value,
                    "feedback_intensity": round(feedback_value, 2),
                    "category": determine_feedback_label(feedback_value).capitalize(),
                }
            )
    return results


def save_results_to_csv(results: list[ScenarioResult]) -> None:
    """Save scenario results to a CSV file."""

    output_path = RESULTS_DIRECTORY / "scenario_comparison.csv"
    fieldnames = (
        "scenario",
        "stress",
        "arousal",
        "feedback_intensity",
        "category",
    )
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV results saved to: results/{output_path.name}")


def create_comparison_heatmap(
    results: list[ScenarioResult], show_plot: bool = True
) -> None:
    """Create and save a heatmap of the nine scenarios."""

    intensity_matrix = np.array(
        [result["feedback_intensity"] for result in results]
    ).reshape(3, 3)
    category_matrix = np.array([result["category"] for result in results]).reshape(3, 3)

    figure, axis = plt.subplots(figsize=(9, 7))
    heatmap = axis.imshow(intensity_matrix, cmap="RdYlGn_r", vmin=0, vmax=100)
    axis.set_xticks(range(3), LEVEL_NAMES)
    axis.set_yticks(range(3), LEVEL_NAMES)
    axis.set_xlabel("Arousal Level")
    axis.set_ylabel("Stress Level")
    axis.set_title("Feedback Intensity for Nine Fuzzy-Rule Scenarios")

    for row in range(3):
        for column in range(3):
            intensity = intensity_matrix[row, column]
            category = category_matrix[row, column]
            text_color = "white" if intensity < 25 or intensity > 75 else "black"
            axis.text(
                column,
                row,
                f"{intensity:.2f}%\n{category}",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )

    colorbar = figure.colorbar(heatmap, ax=axis)
    colorbar.set_label("Feedback Intensity (%)")
    figure.tight_layout()

    output_path = RESULTS_DIRECTORY / "scenario_comparison.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to: results/{output_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def print_results(results: list[ScenarioResult]) -> None:
    """Print a formatted table of scenario results."""

    print("\n" + "=" * 78)
    print("NINE-SCENARIO FUZZY CONTROLLER COMPARISON")
    print("=" * 78)
    print(f"{'Scenario':<34}{'Stress':>9}{'Arousal':>10}{'Output':>12}{'Category':>12}")
    print("-" * 78)
    for result in results:
        print(
            f"{result['scenario']:<34}"
            f"{result['stress']:>9.1f}"
            f"{result['arousal']:>10.1f}"
            f"{result['feedback_intensity']:>11.2f}%"
            f"{result['category']:>12}"
        )
    print("=" * 78)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse display options."""

    parser = argparse.ArgumentParser(
        description="Evaluate the nine representative fuzzy-controller scenarios."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the heatmap without opening a graphical window",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the scenario experiment."""

    arguments = parse_arguments(argv)
    results = evaluate_scenarios()
    print_results(results)
    save_results_to_csv(results)
    create_comparison_heatmap(results, show_plot=not arguments.no_show)


if __name__ == "__main__":
    main()
