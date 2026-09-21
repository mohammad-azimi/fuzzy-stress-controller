from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# --------------------------------------------------
# 1. Project output directory
# --------------------------------------------------

RESULTS_DIRECTORY = Path("results")
RESULTS_DIRECTORY.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Fuzzy input and output variables
# --------------------------------------------------

stress = ctrl.Antecedent(
    np.arange(0, 10.1, 0.1),
    "stress",
)

arousal = ctrl.Antecedent(
    np.arange(0, 10.1, 0.1),
    "arousal",
)

feedback_intensity = ctrl.Consequent(
    np.arange(0, 100.1, 1),
    "feedback_intensity",
)


# --------------------------------------------------
# 3. Stress membership functions
# --------------------------------------------------

stress["low"] = fuzz.trapmf(
    stress.universe,
    [0, 0, 2, 4],
)

stress["medium"] = fuzz.trimf(
    stress.universe,
    [2, 5, 8],
)

stress["high"] = fuzz.trapmf(
    stress.universe,
    [6, 8, 10, 10],
)


# --------------------------------------------------
# 4. Arousal membership functions
# --------------------------------------------------

arousal["low"] = fuzz.trapmf(
    arousal.universe,
    [0, 0, 2, 4],
)

arousal["medium"] = fuzz.trimf(
    arousal.universe,
    [2, 5, 8],
)

arousal["high"] = fuzz.trapmf(
    arousal.universe,
    [6, 8, 10, 10],
)


# --------------------------------------------------
# 5. Feedback membership functions
# --------------------------------------------------

feedback_intensity["gentle"] = fuzz.trapmf(
    feedback_intensity.universe,
    [0, 0, 15, 35],
)

feedback_intensity["moderate"] = fuzz.trimf(
    feedback_intensity.universe,
    [20, 50, 80],
)

feedback_intensity["strong"] = fuzz.trapmf(
    feedback_intensity.universe,
    [65, 85, 100, 100],
)


# --------------------------------------------------
# 6. Fuzzy rule base
# --------------------------------------------------

rules = [
    ctrl.Rule(
        stress["low"] & arousal["low"],
        feedback_intensity["gentle"],
    ),
    ctrl.Rule(
        stress["low"] & arousal["medium"],
        feedback_intensity["gentle"],
    ),
    ctrl.Rule(
        stress["low"] & arousal["high"],
        feedback_intensity["moderate"],
    ),
    ctrl.Rule(
        stress["medium"] & arousal["low"],
        feedback_intensity["gentle"],
    ),
    ctrl.Rule(
        stress["medium"] & arousal["medium"],
        feedback_intensity["moderate"],
    ),
    ctrl.Rule(
        stress["medium"] & arousal["high"],
        feedback_intensity["strong"],
    ),
    ctrl.Rule(
        stress["high"] & arousal["low"],
        feedback_intensity["moderate"],
    ),
    ctrl.Rule(
        stress["high"] & arousal["medium"],
        feedback_intensity["strong"],
    ),
    ctrl.Rule(
        stress["high"] & arousal["high"],
        feedback_intensity["strong"],
    ),
]


# --------------------------------------------------
# 7. Mamdani fuzzy control system
# --------------------------------------------------

control_system = ctrl.ControlSystem(rules)


# --------------------------------------------------
# 8. Read and validate user input
# --------------------------------------------------

def read_input(prompt: str) -> float:
    """Read a numeric input between 0 and 10."""

    while True:
        raw_value = input(prompt).strip().replace(",", ".")

        try:
            value = float(raw_value)
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
            continue

        if 0 <= value <= 10:
            return value

        print("The value must be between 0 and 10.")


# --------------------------------------------------
# 9. Run fuzzy inference
# --------------------------------------------------

def calculate_feedback(
    stress_value: float,
    arousal_value: float,
) -> float:
    """Calculate feedback intensity using Mamdani inference."""

    simulation = ctrl.ControlSystemSimulation(control_system)

    simulation.input["stress"] = stress_value
    simulation.input["arousal"] = arousal_value

    simulation.compute()

    return float(
        simulation.output["feedback_intensity"]
    )


# --------------------------------------------------
# 10. Determine the output category
# --------------------------------------------------

def determine_feedback_label(
    feedback_value: float,
) -> str:
    """Find the output term with the highest membership."""

    memberships = {
        term: fuzz.interp_membership(
            feedback_intensity.universe,
            feedback_intensity[term].mf,
            feedback_value,
        )
        for term in ("gentle", "moderate", "strong")
    }

    return max(
        memberships,
        key=memberships.get,
    )


# --------------------------------------------------
# 11. Select a feedback action
# --------------------------------------------------

def get_recommendation(
    feedback_label: str,
) -> str:
    """Return an illustrative feedback action."""

    recommendations = {
        "gentle": "Gentle breathing guidance",
        "moderate": (
            "Breathing guidance with calming audio"
        ),
        "strong": (
            "Combined calming feedback and a rest reminder"
        ),
    }

    return recommendations[feedback_label]


# --------------------------------------------------
# 12. Plot fuzzy variables and simulation result
# --------------------------------------------------

def plot_fuzzy_variables(
    output_filename: str,
    figure_title: str,
    crisp_values: tuple[
        float | None,
        float | None,
        float | None,
    ] = (
        None,
        None,
        None,
    ),
    show_plot: bool = False,
) -> None:
    """Plot membership functions with optional crisp values."""

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 12),
    )

    variable_data = [
        (
            stress,
            ("low", "medium", "high"),
            "Stress Level",
            "Stress Level",
        ),
        (
            arousal,
            ("low", "medium", "high"),
            "Arousal Level",
            "Arousal Level",
        ),
        (
            feedback_intensity,
            ("gentle", "moderate", "strong"),
            "Feedback Intensity",
            "Feedback Intensity (%)",
        ),
    ]

    colors = (
        "green",
        "orange",
        "red",
    )

    for axis, data, crisp_value in zip(
        axes,
        variable_data,
        crisp_values,
    ):
        variable, terms, title, x_label = data

        for term, color in zip(terms, colors):
            axis.plot(
                variable.universe,
                variable[term].mf,
                label=term.capitalize(),
                color=color,
                linewidth=2,
            )

        if crisp_value is not None:
            axis.axvline(
                crisp_value,
                color="blue",
                linestyle="--",
                linewidth=2,
                label=(
                    f"Crisp value = {crisp_value:.2f}"
                ),
            )

        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Membership Degree")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle(
        figure_title,
        fontsize=16,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.97)
    )

    output_path = (
        RESULTS_DIRECTORY / output_filename
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Plot saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


# --------------------------------------------------
# 13. Main program
# --------------------------------------------------

def main() -> None:
    """Run one interactive fuzzy-control simulation."""

    print("=" * 58)
    print("FUZZY STRESS REGULATION CONTROLLER")
    print("=" * 58)
    print(
        "Enter both input values "
        "on a scale from 0 to 10.\n"
    )

    plot_fuzzy_variables(
        output_filename="membership_functions.png",
        figure_title=(
            "Fuzzy Stress Regulation Controller"
        ),
    )

    stress_value = read_input(
        "Enter stress level: "
    )

    arousal_value = read_input(
        "Enter arousal level: "
    )

    feedback_value = calculate_feedback(
        stress_value,
        arousal_value,
    )

    feedback_label = determine_feedback_label(
        feedback_value
    )

    recommendation = get_recommendation(
        feedback_label
    )

    print("\n" + "-" * 58)
    print("FUZZY INFERENCE RESULT")
    print("-" * 58)
    print(
        f"Stress input:       "
        f"{stress_value:.2f} / 10"
    )
    print(
        f"Arousal input:      "
        f"{arousal_value:.2f} / 10"
    )
    print(
        f"Feedback intensity: "
        f"{feedback_value:.2f} %"
    )
    print(
        f"Feedback category:  "
        f"{feedback_label.capitalize()}"
    )
    print(
        f"Suggested action:   "
        f"{recommendation}"
    )
    print("-" * 58)
    print(
        "Educational simulation only - "
        "not a medical device.\n"
    )

    plot_fuzzy_variables(
        output_filename="simulation_result.png",
        figure_title=(
            "Fuzzy Controller Simulation Result"
        ),
        crisp_values=(
            stress_value,
            arousal_value,
            feedback_value,
        ),
        show_plot=True,
    )


if __name__ == "__main__":
    main()