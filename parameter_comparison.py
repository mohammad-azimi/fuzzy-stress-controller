import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from main import calculate_feedback as calculate_baseline
from main import feedback_intensity as output_variable
from main import stress as baseline_stress


# --------------------------------------------------
# 1. Output directory
# --------------------------------------------------

RESULTS_DIRECTORY = Path("results")
RESULTS_DIRECTORY.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Modified membership-function parameters
# --------------------------------------------------

MODIFIED_PARAMETERS = {
    "low": [0, 0, 1.5, 3.5],
    "medium": [1.5, 4.5, 7.5],
    "high": [5, 7, 10, 10],
}


# --------------------------------------------------
# 3. Comparison scenarios
# --------------------------------------------------

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


# --------------------------------------------------
# 4. Build the modified fuzzy controller
# --------------------------------------------------

def build_modified_controller():
    """Build a more sensitive fuzzy controller."""

    stress = ctrl.Antecedent(
        np.arange(0, 10.1, 0.1),
        "stress",
    )

    arousal = ctrl.Antecedent(
        np.arange(0, 10.1, 0.1),
        "arousal",
    )

    feedback = ctrl.Consequent(
        np.arange(0, 100.1, 1),
        "feedback_intensity",
    )

    for variable in (stress, arousal):
        variable["low"] = fuzz.trapmf(
            variable.universe,
            MODIFIED_PARAMETERS["low"],
        )

        variable["medium"] = fuzz.trimf(
            variable.universe,
            MODIFIED_PARAMETERS["medium"],
        )

        variable["high"] = fuzz.trapmf(
            variable.universe,
            MODIFIED_PARAMETERS["high"],
        )

    feedback["gentle"] = fuzz.trapmf(
        feedback.universe,
        [0, 0, 15, 35],
    )

    feedback["moderate"] = fuzz.trimf(
        feedback.universe,
        [20, 50, 80],
    )

    feedback["strong"] = fuzz.trapmf(
        feedback.universe,
        [65, 85, 100, 100],
    )

    rules = [
        ctrl.Rule(
            stress["low"] & arousal["low"],
            feedback["gentle"],
        ),
        ctrl.Rule(
            stress["low"] & arousal["medium"],
            feedback["gentle"],
        ),
        ctrl.Rule(
            stress["low"] & arousal["high"],
            feedback["moderate"],
        ),
        ctrl.Rule(
            stress["medium"] & arousal["low"],
            feedback["gentle"],
        ),
        ctrl.Rule(
            stress["medium"] & arousal["medium"],
            feedback["moderate"],
        ),
        ctrl.Rule(
            stress["medium"] & arousal["high"],
            feedback["strong"],
        ),
        ctrl.Rule(
            stress["high"] & arousal["low"],
            feedback["moderate"],
        ),
        ctrl.Rule(
            stress["high"] & arousal["medium"],
            feedback["strong"],
        ),
        ctrl.Rule(
            stress["high"] & arousal["high"],
            feedback["strong"],
        ),
    ]

    control_system = ctrl.ControlSystem(rules)

    return control_system, stress


MODIFIED_SYSTEM, MODIFIED_STRESS = (
    build_modified_controller()
)


# --------------------------------------------------
# 5. Calculate the modified output
# --------------------------------------------------

def calculate_modified(
    stress_value: float,
    arousal_value: float,
) -> float:
    """Calculate feedback using the modified model."""

    simulation = ctrl.ControlSystemSimulation(
        MODIFIED_SYSTEM
    )

    simulation.input["stress"] = stress_value
    simulation.input["arousal"] = arousal_value

    simulation.compute()

    return float(
        simulation.output["feedback_intensity"]
    )


# --------------------------------------------------
# 6. Determine the output category
# --------------------------------------------------

def determine_category(
    output_value: float,
) -> str:
    """Determine the dominant output category."""

    memberships = {
        term: fuzz.interp_membership(
            output_variable.universe,
            output_variable[term].mf,
            output_value,
        )
        for term in (
            "gentle",
            "moderate",
            "strong",
        )
    }

    return max(
        memberships,
        key=memberships.get,
    ).capitalize()


# --------------------------------------------------
# 7. Evaluate baseline and modified models
# --------------------------------------------------

def evaluate_models():
    """Compare the two fuzzy controllers."""

    results = []

    for (
        scenario,
        stress_value,
        arousal_value,
    ) in SCENARIOS:
        baseline_output = calculate_baseline(
            stress_value,
            arousal_value,
        )

        modified_output = calculate_modified(
            stress_value,
            arousal_value,
        )

        difference = (
            modified_output - baseline_output
        )

        if abs(difference) < 0.005:
            difference = 0.0

        results.append(
            {
                "scenario": scenario,
                "stress": stress_value,
                "arousal": arousal_value,
                "baseline_output": round(
                    baseline_output,
                    2,
                ),
                "baseline_category": (
                    determine_category(
                        baseline_output
                    )
                ),
                "modified_output": round(
                    modified_output,
                    2,
                ),
                "modified_category": (
                    determine_category(
                        modified_output
                    )
                ),
                "difference": round(
                    difference,
                    2,
                ),
            }
        )

    return results


# --------------------------------------------------
# 8. Save results as CSV
# --------------------------------------------------

def save_results(results) -> None:
    """Save comparison results as CSV."""

    output_path = (
        RESULTS_DIRECTORY
        / "parameter_comparison.csv"
    )

    fieldnames = tuple(
        results[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"CSV results saved to: {output_path}"
    )


# --------------------------------------------------
# 9. Compare membership functions
# --------------------------------------------------

def plot_membership_comparison() -> None:
    """Compare baseline and modified memberships."""

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 4.5),
        sharey=True,
    )

    colors = {
        "low": "green",
        "medium": "orange",
        "high": "red",
    }

    for axis, term in zip(
        axes,
        ("low", "medium", "high"),
    ):
        axis.plot(
            baseline_stress.universe,
            baseline_stress[term].mf,
            color=colors[term],
            linewidth=2,
            label="Baseline",
        )

        axis.plot(
            MODIFIED_STRESS.universe,
            MODIFIED_STRESS[term].mf,
            color=colors[term],
            linestyle="--",
            linewidth=2,
            label="Modified",
        )

        axis.set_title(
            f"{term.capitalize()} Membership"
        )

        axis.set_xlabel("Input Value")
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.3)
        axis.legend()

    axes[0].set_ylabel(
        "Membership Degree"
    )

    figure.suptitle(
        "Baseline vs Modified Input "
        "Membership Functions",
        fontsize=15,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.93)
    )

    output_path = (
        RESULTS_DIRECTORY
        / "parameter_membership_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Membership comparison saved to: "
        f"{output_path}"
    )


# --------------------------------------------------
# 10. Compare model outputs
# --------------------------------------------------

def plot_output_comparison(
    results,
) -> None:
    """Compare outputs using grouped bars."""

    positions = np.arange(
        len(results)
    )

    width = 0.36

    labels = [
        str(row["scenario"])
        for row in results
    ]

    baseline_values = [
        float(row["baseline_output"])
        for row in results
    ]

    modified_values = [
        float(row["modified_output"])
        for row in results
    ]

    figure, axis = plt.subplots(
        figsize=(13, 7)
    )

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

    axis.bar_label(
        baseline_bars,
        fmt="%.2f",
        padding=3,
        fontsize=8,
    )

    axis.bar_label(
        modified_bars,
        fmt="%.2f",
        padding=3,
        fontsize=8,
    )

    axis.set_xticks(
        positions,
        labels,
        rotation=25,
        ha="right",
    )

    axis.set_ylabel(
        "Feedback Intensity (%)"
    )

    axis.set_ylim(0, 100)

    axis.set_title(
        "Effect of Membership-Function "
        "Parameter Modification"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.legend()
    figure.tight_layout()

    output_path = (
        RESULTS_DIRECTORY
        / "parameter_output_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Output comparison saved to: "
        f"{output_path}"
    )


# --------------------------------------------------
# 11. Print results in the terminal
# --------------------------------------------------

def print_results(results) -> None:
    """Print the comparison table."""

    print("\n" + "=" * 88)

    print(
        "BASELINE AND MODIFIED "
        "FUZZY CONTROLLER COMPARISON"
    )

    print("=" * 88)

    print(
        f"{'Scenario':<22}"
        f"{'Stress':>8}"
        f"{'Arousal':>9}"
        f"{'Baseline':>12}"
        f"{'Modified':>12}"
        f"{'Difference':>13}"
    )

    print("-" * 88)

    for row in results:
        print(
            f"{str(row['scenario']):<22}"
            f"{float(row['stress']):>8.1f}"
            f"{float(row['arousal']):>9.1f}"
            f"{float(row['baseline_output']):>11.2f}%"
            f"{float(row['modified_output']):>11.2f}%"
            f"{float(row['difference']):>+12.2f}"
        )

    print("=" * 88)


# --------------------------------------------------
# 12. Start the experiment
# --------------------------------------------------

def main() -> None:
    """Run the parameter comparison."""

    results = evaluate_models()

    print_results(results)
    save_results(results)
    plot_membership_comparison()
    plot_output_comparison(results)

    plt.show()


if __name__ == "__main__":
    main()