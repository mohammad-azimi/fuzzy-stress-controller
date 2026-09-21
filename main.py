"""Interactive Mamdani fuzzy controller for stress-regulation feedback."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
RESULTS_DIRECTORY.mkdir(exist_ok=True)

INPUT_UNIVERSE = np.arange(0, 10.1, 0.1)
OUTPUT_UNIVERSE = np.arange(0, 100.1, 1.0)

# Fuzzy antecedents and consequent.
stress = ctrl.Antecedent(INPUT_UNIVERSE, "stress")
arousal = ctrl.Antecedent(INPUT_UNIVERSE, "arousal")
feedback_intensity = ctrl.Consequent(OUTPUT_UNIVERSE, "feedback_intensity")

# Baseline input membership functions.
for variable in (stress, arousal):
    variable["low"] = fuzz.trapmf(variable.universe, [0, 0, 2, 4])
    variable["medium"] = fuzz.trimf(variable.universe, [2, 5, 8])
    variable["high"] = fuzz.trapmf(variable.universe, [6, 8, 10, 10])

# Output membership functions.
feedback_intensity["gentle"] = fuzz.trapmf(feedback_intensity.universe, [0, 0, 15, 35])
feedback_intensity["moderate"] = fuzz.trimf(feedback_intensity.universe, [20, 50, 80])
feedback_intensity["strong"] = fuzz.trapmf(
    feedback_intensity.universe, [65, 85, 100, 100]
)

# Complete 3 x 3 Mamdani rule base.
rules = [
    ctrl.Rule(stress["low"] & arousal["low"], feedback_intensity["gentle"]),
    ctrl.Rule(stress["low"] & arousal["medium"], feedback_intensity["gentle"]),
    ctrl.Rule(stress["low"] & arousal["high"], feedback_intensity["moderate"]),
    ctrl.Rule(stress["medium"] & arousal["low"], feedback_intensity["gentle"]),
    ctrl.Rule(stress["medium"] & arousal["medium"], feedback_intensity["moderate"]),
    ctrl.Rule(stress["medium"] & arousal["high"], feedback_intensity["strong"]),
    ctrl.Rule(stress["high"] & arousal["low"], feedback_intensity["moderate"]),
    ctrl.Rule(stress["high"] & arousal["medium"], feedback_intensity["strong"]),
    ctrl.Rule(stress["high"] & arousal["high"], feedback_intensity["strong"]),
]

control_system = ctrl.ControlSystem(rules)

TERM_COLORS = {
    "low": "#2E8B57",
    "medium": "#F59E0B",
    "high": "#DC2626",
    "gentle": "#2E8B57",
    "moderate": "#F59E0B",
    "strong": "#DC2626",
}


def validate_scale_value(value: float, variable_name: str) -> float:
    """Validate and return a finite value on the inclusive 0-to-10 scale."""

    if not np.isfinite(value):
        raise ValueError(f"{variable_name} must be a finite number.")
    if not 0 <= value <= 10:
        raise ValueError(f"{variable_name} must be between 0 and 10.")
    return float(value)


def read_input(prompt: str, variable_name: str) -> float:
    """Read a valid numeric value from the terminal."""

    while True:
        raw_value = input(prompt).strip().replace(",", ".")
        try:
            return validate_scale_value(float(raw_value), variable_name)
        except ValueError as error:
            print(f"Invalid input: {error}")


def calculate_feedback(stress_value: float, arousal_value: float) -> float:
    """Calculate crisp feedback intensity using Mamdani inference."""

    stress_value = validate_scale_value(stress_value, "Stress")
    arousal_value = validate_scale_value(arousal_value, "Arousal")

    simulation = ctrl.ControlSystemSimulation(control_system)
    simulation.input["stress"] = stress_value
    simulation.input["arousal"] = arousal_value
    simulation.compute()
    return float(simulation.output["feedback_intensity"])


def determine_feedback_label(feedback_value: float) -> str:
    """Return the output term with the greatest membership degree."""

    memberships = {
        term: fuzz.interp_membership(
            feedback_intensity.universe,
            feedback_intensity[term].mf,
            feedback_value,
        )
        for term in ("gentle", "moderate", "strong")
    }
    return max(memberships, key=memberships.get)


def get_recommendation(feedback_label: str) -> str:
    """Return an illustrative action for the selected feedback category."""

    recommendations = {
        "gentle": "Gentle breathing guidance",
        "moderate": "Breathing guidance with calming audio",
        "strong": "Combined calming feedback and a rest reminder",
    }
    try:
        return recommendations[feedback_label.lower()]
    except KeyError as error:
        valid = ", ".join(recommendations)
        raise ValueError(
            f"Unknown feedback label. Expected one of: {valid}."
        ) from error


def _display_path(path: Path) -> Path:
    """Return a short project-relative path when possible."""

    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def plot_fuzzy_variables(
    output_filename: str,
    figure_title: str,
    crisp_values: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    ),
    show_plot: bool = False,
) -> Path:
    """Plot all membership functions and optional crisp values."""

    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(11, 12))
    variable_data = [
        (stress, ("low", "medium", "high"), "Stress Level", "Stress Level"),
        (arousal, ("low", "medium", "high"), "Arousal Level", "Arousal Level"),
        (
            feedback_intensity,
            ("gentle", "moderate", "strong"),
            "Feedback Intensity",
            "Feedback Intensity (%)",
        ),
    ]

    for axis, data, crisp_value in zip(axes, variable_data, crisp_values):
        variable, terms, title, x_label = data
        for term in terms:
            axis.plot(
                variable.universe,
                variable[term].mf,
                label=term.capitalize(),
                color=TERM_COLORS[term],
                linewidth=2.2,
            )

        if crisp_value is not None:
            axis.axvline(
                crisp_value,
                color="#2563EB",
                linestyle="--",
                linewidth=2,
                label=f"Crisp value = {crisp_value:.2f}",
            )

        axis.set(title=title, xlabel=x_label, ylabel="Membership Degree")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    figure.suptitle(figure_title, fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = RESULTS_DIRECTORY / output_filename
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {_display_path(output_path)}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)

    return output_path


def print_result(
    stress_value: float,
    arousal_value: float,
    feedback_value: float,
    feedback_label: str,
    recommendation: str,
) -> None:
    """Print one simulation result in a presentation-friendly format."""

    print("\n" + "-" * 58)
    print("FUZZY INFERENCE RESULT")
    print("-" * 58)
    print(f"Stress input:       {stress_value:.2f} / 10")
    print(f"Arousal input:      {arousal_value:.2f} / 10")
    print(f"Feedback intensity: {feedback_value:.2f} %")
    print(f"Feedback category:  {feedback_label.capitalize()}")
    print(f"Suggested action:   {recommendation}")
    print("-" * 58)
    print("Educational simulation only - not a medical device.\n")


def _command_line_value(raw_value: str) -> float:
    """Parse and validate a command-line input value."""

    try:
        return validate_scale_value(float(raw_value.replace(",", ".")), "Input")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse optional non-interactive inputs and display settings."""

    parser = argparse.ArgumentParser(
        description="Run the fuzzy stress-regulation controller."
    )
    parser.add_argument(
        "--stress", type=_command_line_value, help="stress level (0-10)"
    )
    parser.add_argument(
        "--arousal", type=_command_line_value, help="arousal level (0-10)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save plots without opening graphical windows",
    )
    arguments = parser.parse_args(argv)

    if (arguments.stress is None) != (arguments.arousal is None):
        parser.error("--stress and --arousal must be supplied together")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:
    """Run one interactive or command-line fuzzy-control simulation."""

    arguments = parse_arguments(argv)

    print("=" * 58)
    print("FUZZY STRESS REGULATION CONTROLLER")
    print("=" * 58)
    print("Enter both input values on a scale from 0 to 10.\n")

    plot_fuzzy_variables(
        output_filename="membership_functions.png",
        figure_title="Fuzzy Stress Regulation Controller",
    )

    if arguments.stress is None:
        stress_value = read_input("Enter stress level: ", "Stress")
        arousal_value = read_input("Enter arousal level: ", "Arousal")
    else:
        stress_value = arguments.stress
        arousal_value = arguments.arousal

    feedback_value = calculate_feedback(stress_value, arousal_value)
    feedback_label = determine_feedback_label(feedback_value)
    recommendation = get_recommendation(feedback_label)
    print_result(
        stress_value,
        arousal_value,
        feedback_value,
        feedback_label,
        recommendation,
    )

    plot_fuzzy_variables(
        output_filename="simulation_result.png",
        figure_title="Fuzzy Controller Simulation Result",
        crisp_values=(stress_value, arousal_value, feedback_value),
        show_plot=not arguments.no_show,
    )


if __name__ == "__main__":
    main()
