"""Compare baseline and more sensitive fuzzy membership parameters."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from main import (
    INPUT_UNIVERSE,
    OUTPUT_UNIVERSE,
    RESULTS_DIRECTORY,
    calculate_feedback,
    determine_feedback_label,
    stress,
)

MODIFIED_PARAMETERS = {
    "low": [0, 0, 1.5, 3.5],
    "medium": [1.5, 4.5, 7.5],
    "high": [5, 7, 10, 10],
}

SCENARIOS = [
    ("Lower transition", 3.0, 3.0),
    ("Central state", 5.0, 5.0),
    ("Rising both", 5.5, 5.5),
    ("Stress dominant", 6.0, 4.0),
    ("Arousal dominant", 4.0, 6.0),
    ("Elevated both", 6.0, 6.0),
    ("Upper transition", 7.0, 7.0),
    ("Previous example", 8.0, 7.0),
]


class ComparisonResult(TypedDict):
    """One baseline-versus-modified comparison result."""

    scenario: str
    stress: float
    arousal: float
    baseline_output: float
    baseline_category: str
    modified_output: float
    modified_category: str
    difference: float


def build_modified_controller() -> tuple[ctrl.ControlSystem, ctrl.Antecedent]:
    """Build a controller whose input memberships activate earlier."""

    modified_stress = ctrl.Antecedent(INPUT_UNIVERSE, "stress")
    modified_arousal = ctrl.Antecedent(INPUT_UNIVERSE, "arousal")
    modified_feedback = ctrl.Consequent(OUTPUT_UNIVERSE, "feedback_intensity")

    for variable in (modified_stress, modified_arousal):
        variable["low"] = fuzz.trapmf(variable.universe, MODIFIED_PARAMETERS["low"])
        variable["medium"] = fuzz.trimf(
            variable.universe, MODIFIED_PARAMETERS["medium"]
        )
        variable["high"] = fuzz.trapmf(variable.universe, MODIFIED_PARAMETERS["high"])

    modified_feedback["gentle"] = fuzz.trapmf(
        modified_feedback.universe, [0, 0, 15, 35]
    )
    modified_feedback["moderate"] = fuzz.trimf(modified_feedback.universe, [20, 50, 80])
    modified_feedback["strong"] = fuzz.trapmf(
        modified_feedback.universe, [65, 85, 100, 100]
    )

    rules = [
        ctrl.Rule(
            modified_stress["low"] & modified_arousal["low"],
            modified_feedback["gentle"],
        ),
        ctrl.Rule(
            modified_stress["low"] & modified_arousal["medium"],
            modified_feedback["gentle"],
        ),
        ctrl.Rule(
            modified_stress["low"] & modified_arousal["high"],
            modified_feedback["moderate"],
        ),
        ctrl.Rule(
            modified_stress["medium"] & modified_arousal["low"],
            modified_feedback["gentle"],
        ),
        ctrl.Rule(
            modified_stress["medium"] & modified_arousal["medium"],
            modified_feedback["moderate"],
        ),
        ctrl.Rule(
            modified_stress["medium"] & modified_arousal["high"],
            modified_feedback["strong"],
        ),
        ctrl.Rule(
            modified_stress["high"] & modified_arousal["low"],
            modified_feedback["moderate"],
        ),
        ctrl.Rule(
            modified_stress["high"] & modified_arousal["medium"],
            modified_feedback["strong"],
        ),
        ctrl.Rule(
            modified_stress["high"] & modified_arousal["high"],
            modified_feedback["strong"],
        ),
    ]
    return ctrl.ControlSystem(rules), modified_stress


MODIFIED_SYSTEM, MODIFIED_STRESS = build_modified_controller()


def calculate_modified(stress_value: float, arousal_value: float) -> float:
    """Calculate feedback intensity using the modified controller."""

    simulation = ctrl.ControlSystemSimulation(MODIFIED_SYSTEM)
    simulation.input["stress"] = stress_value
    simulation.input["arousal"] = arousal_value
    simulation.compute()
    return float(simulation.output["feedback_intensity"])


def evaluate_models() -> list[ComparisonResult]:
    """Evaluate the baseline and modified models on selected inputs."""

    results: list[ComparisonResult] = []
    for scenario, stress_value, arousal_value in SCENARIOS:
        baseline_output = calculate_feedback(stress_value, arousal_value)
        modified_output = calculate_modified(stress_value, arousal_value)
        difference = modified_output - baseline_output
        if abs(difference) < 0.005:
            difference = 0.0

        results.append(
            {
                "scenario": scenario,
                "stress": stress_value,
                "arousal": arousal_value,
                "baseline_output": round(baseline_output, 2),
                "baseline_category": determine_feedback_label(
                    baseline_output
                ).capitalize(),
                "modified_output": round(modified_output, 2),
                "modified_category": determine_feedback_label(
                    modified_output
                ).capitalize(),
                "difference": round(difference, 2),
            }
        )
    return results


def save_results(results: list[ComparisonResult]) -> None:
    """Save the comparison results as CSV."""

    if not results:
        raise ValueError("At least one comparison result is required.")

    output_path = RESULTS_DIRECTORY / "parameter_comparison.csv"
    fieldnames = tuple(results[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV results saved to: results/{output_path.name}")


def plot_membership_comparison(show_plot: bool = False) -> None:
    """Plot baseline and modified input membership functions."""

    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5), sharey=True)
    colors = {"low": "#2E8B57", "medium": "#F59E0B", "high": "#DC2626"}

    for axis, term in zip(axes, ("low", "medium", "high")):
        axis.plot(
            stress.universe,
            stress[term].mf,
            color=colors[term],
            linewidth=2.2,
            label="Baseline",
        )
        axis.plot(
            MODIFIED_STRESS.universe,
            MODIFIED_STRESS[term].mf,
            color=colors[term],
            linestyle="--",
            linewidth=2.2,
            label="Modified",
        )
        axis.set_title(f"{term.capitalize()} Membership")
        axis.set_xlabel("Input Value")
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[0].set_ylabel("Membership Degree")
    figure.suptitle(
        "Baseline vs Modified Input Membership Functions",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    output_path = RESULTS_DIRECTORY / "parameter_membership_comparison.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Membership comparison saved to: results/{output_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def plot_output_comparison(
    results: list[ComparisonResult], show_plot: bool = True
) -> None:
    """Compare baseline and modified outputs with grouped bars."""

    positions = np.arange(len(results))
    width = 0.36
    labels = [row["scenario"] for row in results]
    baseline_values = [row["baseline_output"] for row in results]
    modified_values = [row["modified_output"] for row in results]

    figure, axis = plt.subplots(figsize=(13, 7))
    baseline_bars = axis.bar(
        positions - width / 2,
        baseline_values,
        width,
        label="Baseline",
        color="#4C78A8",
    )
    modified_bars = axis.bar(
        positions + width / 2,
        modified_values,
        width,
        label="Modified",
        color="#E45756",
    )
    axis.bar_label(baseline_bars, fmt="%.2f", padding=3, fontsize=8)
    axis.bar_label(modified_bars, fmt="%.2f", padding=3, fontsize=8)
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("Feedback Intensity (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Effect of Membership-Function Parameter Modification")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()

    output_path = RESULTS_DIRECTORY / "parameter_output_comparison.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Output comparison saved to: results/{output_path.name}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def print_results(results: list[ComparisonResult]) -> None:
    """Print the baseline-versus-modified comparison table."""

    print("\n" + "=" * 88)
    print("BASELINE AND MODIFIED FUZZY CONTROLLER COMPARISON")
    print("=" * 88)
    print(
        f"{'Scenario':<22}{'Stress':>8}{'Arousal':>9}"
        f"{'Baseline':>12}{'Modified':>12}{'Difference':>13}"
    )
    print("-" * 88)
    for row in results:
        print(
            f"{row['scenario']:<22}"
            f"{row['stress']:>8.1f}"
            f"{row['arousal']:>9.1f}"
            f"{row['baseline_output']:>11.2f}%"
            f"{row['modified_output']:>11.2f}%"
            f"{row['difference']:>+12.2f}"
        )
    print("=" * 88)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse display options."""

    parser = argparse.ArgumentParser(
        description="Compare baseline and modified membership parameters."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save plots without opening graphical windows",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the parameter comparison experiment."""

    arguments = parse_arguments(argv)
    results = evaluate_models()
    print_results(results)
    save_results(results)
    plot_membership_comparison()
    plot_output_comparison(results, show_plot=not arguments.no_show)


if __name__ == "__main__":
    main()
