import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main import calculate_feedback
from main import determine_feedback_label


# --------------------------------------------------
# 1. Results directory
# --------------------------------------------------

RESULTS_DIRECTORY = Path("results")
RESULTS_DIRECTORY.mkdir(exist_ok=True)


# Representative values for low, medium, and high
LEVEL_NAMES = (
    "Low",
    "Medium",
    "High",
)

LEVEL_VALUES = (
    1.0,
    5.0,
    9.0,
)


# --------------------------------------------------
# 2. Evaluate all nine rule scenarios
# --------------------------------------------------

def evaluate_scenarios() -> list[
    dict[str, str | float]
]:
    """Evaluate all nine representative scenarios."""

    results = []

    for stress_name, stress_value in zip(
        LEVEL_NAMES,
        LEVEL_VALUES,
    ):
        for arousal_name, arousal_value in zip(
            LEVEL_NAMES,
            LEVEL_VALUES,
        ):
            feedback_value = calculate_feedback(
                stress_value,
                arousal_value,
            )

            category = determine_feedback_label(
                feedback_value
            ).capitalize()

            results.append(
                {
                    "scenario": (
                        f"{stress_name} stress / "
                        f"{arousal_name} arousal"
                    ),
                    "stress": stress_value,
                    "arousal": arousal_value,
                    "feedback_intensity": round(
                        feedback_value,
                        2,
                    ),
                    "category": category,
                }
            )

    return results


# --------------------------------------------------
# 3. Save results to CSV
# --------------------------------------------------

def save_results_to_csv(
    results: list[dict[str, str | float]],
) -> None:
    """Save scenario results to a CSV file."""

    output_path = (
        RESULTS_DIRECTORY
        / "scenario_comparison.csv"
    )

    fieldnames = (
        "scenario",
        "stress",
        "arousal",
        "feedback_intensity",
        "category",
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
# 4. Create comparison heatmap
# --------------------------------------------------

def create_comparison_heatmap(
    results: list[dict[str, str | float]],
) -> None:
    """Create a heatmap of the nine scenarios."""

    intensity_matrix = np.array(
        [
            float(result["feedback_intensity"])
            for result in results
        ]
    ).reshape(3, 3)

    category_matrix = np.array(
        [
            str(result["category"])
            for result in results
        ]
    ).reshape(3, 3)

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    heatmap = axis.imshow(
        intensity_matrix,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=100,
    )

    axis.set_xticks(
        range(3),
        LEVEL_NAMES,
    )

    axis.set_yticks(
        range(3),
        LEVEL_NAMES,
    )

    axis.set_xlabel("Arousal Level")
    axis.set_ylabel("Stress Level")

    axis.set_title(
        "Feedback Intensity for Nine "
        "Fuzzy-Rule Scenarios"
    )

    for row in range(3):
        for column in range(3):
            intensity = intensity_matrix[
                row,
                column,
            ]

            category = category_matrix[
                row,
                column,
            ]

            if intensity < 25 or intensity > 75:
                text_color = "white"
            else:
                text_color = "black"

            axis.text(
                column,
                row,
                f"{intensity:.2f}%\n{category}",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )

    colorbar = figure.colorbar(
        heatmap,
        ax=axis,
    )

    colorbar.set_label(
        "Feedback Intensity (%)"
    )

    figure.tight_layout()

    output_path = (
        RESULTS_DIRECTORY
        / "scenario_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Comparison plot saved to: "
        f"{output_path}"
    )

    plt.show()


# --------------------------------------------------
# 5. Print results in the terminal
# --------------------------------------------------

def print_results(
    results: list[dict[str, str | float]],
) -> None:
    """Print a formatted results table."""

    print("\n" + "=" * 78)
    print(
        "NINE-SCENARIO FUZZY "
        "CONTROLLER COMPARISON"
    )
    print("=" * 78)

    print(
        f"{'Scenario':<34}"
        f"{'Stress':>9}"
        f"{'Arousal':>10}"
        f"{'Output':>12}"
        f"{'Category':>12}"
    )

    print("-" * 78)

    for result in results:
        print(
            f"{str(result['scenario']):<34}"
            f"{float(result['stress']):>9.1f}"
            f"{float(result['arousal']):>10.1f}"
            f"{float(result['feedback_intensity']):>11.2f}%"
            f"{str(result['category']):>12}"
        )

    print("=" * 78)


# --------------------------------------------------
# 6. Start the experiment
# --------------------------------------------------

def main() -> None:
    """Run all nine comparison scenarios."""

    results = evaluate_scenarios()

    print_results(results)
    save_results_to_csv(results)
    create_comparison_heatmap(results)


if __name__ == "__main__":
    main()