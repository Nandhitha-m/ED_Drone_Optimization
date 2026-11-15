"""Compare RBF and quadratic surrogates using leave-one-out cross-validation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from quadratic_res_surf import (
    QuadraticResponseSurface,
    leave_one_out_predictions as quadratic_loocv_predictions,
)
from rbf_surrogate import (
    PredictiveModel,
    RBFSurrogate,
    leave_one_out_predictions as rbf_loocv_predictions,
    load_dataset,
    save_surrogate_plot,
    write_surrogate_grid,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ComparisonResult:
    """LOOCV predictions and aggregate metrics for one surrogate method."""

    name: str
    predictions: FloatArray
    model: PredictiveModel
    rmse: float
    mae: float
    r_squared: float


def evaluate(
    name: str,
    model: PredictiveModel,
    predictions: FloatArray,
    actual: FloatArray,
) -> ComparisonResult:
    """Calculate comparable LOOCV metrics for a fitted surrogate."""
    residuals = predictions - actual
    total_sum_squares = float(np.sum(np.square(actual - np.mean(actual))))
    r_squared = (
        float(1.0 - np.sum(np.square(residuals)) / total_sum_squares)
        if total_sum_squares > 0.0
        else float("nan")
    )
    return ComparisonResult(
        name=name,
        predictions=predictions,
        model=model,
        rmse=float(np.sqrt(np.mean(np.square(residuals)))),
        mae=float(np.mean(np.abs(residuals))),
        r_squared=r_squared,
    )


def compare_surrogates(
    x: FloatArray, actual: FloatArray
) -> tuple[tuple[ComparisonResult, ...], ComparisonResult]:
    """Run LOOCV for both methods and return the lowest-RMSE full-data fit."""
    rbf_predictions = rbf_loocv_predictions(x, actual)
    quadratic_predictions = quadratic_loocv_predictions(x, actual)
    results = (
        evaluate(
            "RBF thin-plate spline",
            RBFSurrogate.fit(x, actual),
            rbf_predictions,
            actual,
        ),
        evaluate(
            "Quadratic response surface",
            QuadraticResponseSurface.fit(x, actual),
            quadratic_predictions,
            actual,
        ),
    )
    return results, min(results, key=lambda result: result.rmse)


def write_comparison_table(
    path: Path,
    header: Sequence[str],
    x: FloatArray,
    actual: FloatArray,
    results: Sequence[ComparisonResult],
) -> None:
    """Write side-by-side per-row LOOCV predictions and residuals to CSV."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "row",
                *header[:2],
                "actual",
                "rbf_loo_predicted",
                "rbf_error",
                "quadratic_loo_predicted",
                "quadratic_error",
            ]
        )
        for index, values in enumerate(
            zip(x, actual, results[0].predictions, results[1].predictions),
            start=1,
        ):
            inputs, observed, rbf_prediction, quadratic_prediction = values
            writer.writerow(
                [
                    index,
                    *inputs,
                    observed,
                    rbf_prediction,
                    rbf_prediction - observed,
                    quadratic_prediction,
                    quadratic_prediction - observed,
                ]
            )


def print_comparison_table(
    header: Sequence[str],
    x: FloatArray,
    actual: FloatArray,
    results: Sequence[ComparisonResult],
    best: ComparisonResult,
) -> None:
    """Print per-row predictions and a compact summary of both methods."""
    columns = [
        ("Row", 4),
        (header[0], 10),
        (header[1], 10),
        ("Actual", 10),
        ("RBF pred", 11),
        ("RBF error", 11),
        ("Quadratic pred", 15),
        ("Quadratic error", 15),
    ]
    print("\nLOOCV comparison (prediction error = predicted - actual)")
    print("  ".join(f"{name:>{width}}" for name, width in columns))
    print("  ".join("-" * width for _, width in columns))
    for index, (inputs, observed, rbf_prediction, quadratic_prediction) in enumerate(
        zip(x, actual, results[0].predictions, results[1].predictions),
        start=1,
    ):
        values = (
            f"{index:>{columns[0][1]}}",
            f"{inputs[0]:>{columns[1][1]}.5g}",
            f"{inputs[1]:>{columns[2][1]}.5g}",
            f"{observed:>{columns[3][1]}.6f}",
            f"{rbf_prediction:>{columns[4][1]}.6f}",
            f"{rbf_prediction - observed:>{columns[5][1]}.6f}",
            f"{quadratic_prediction:>{columns[6][1]}.6f}",
            f"{quadratic_prediction - observed:>{columns[7][1]}.6f}",
        )
        print("  ".join(values))

    print("\nLOOCV metric summary (lower RMSE/MAE is better)")
    summary_columns = (("Method", 26), ("RMSE", 12), ("MAE", 12), ("R-squared", 12))
    print("  ".join(f"{name:>{width}}" for name, width in summary_columns))
    print("  ".join("-" * width for _, width in summary_columns))
    for result in results:
        print(
            f"{result.name:>{summary_columns[0][1]}}  "
            f"{result.rmse:>{summary_columns[1][1]}.6f}  "
            f"{result.mae:>{summary_columns[2][1]}.6f}  "
            f"{result.r_squared:>{summary_columns[3][1]}.6f}"
        )
    print(f"\nBest surrogate by lowest LOOCV RMSE: {best.name}")


def print_best_equation(result: ComparisonResult, header: Sequence[str]) -> None:
    """Print the fitted equation for the selected surrogate."""
    model = result.model
    print("\nBest surrogate equation (inputs standardized before evaluation):")

    if isinstance(model, QuadraticResponseSurface):
        first, second = "z_" + header[0], "z_" + header[1]
        c0, c1, c2, c11, c12, c22 = model.coefficients
        print(
            f"  {header[2]} = {c0:.10g} {c1:+.10g}*{first} "
            f"{c2:+.10g}*{second} {c11:+.10g}*{first}^2 "
            f"{c12:+.10g}*{first}*{second} {c22:+.10g}*{second}^2"
        )
    elif isinstance(model, RBFSurrogate):
        interpolator = model.interpolator
        coefficients = getattr(interpolator, "_coeffs", None)
        if coefficients is None:
            raise RuntimeError(
                "The installed SciPy RBFInterpolator does not expose the "
                "coefficients needed to print its fitted equation."
            )
        coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        centers = (model.x_train - model.x_mean) / model.x_scale
        center_count = len(centers)
        if coefficients.size != center_count + 3:
            raise RuntimeError(
                "Unexpected RBF coefficient layout; cannot print the fitted equation."
            )
        shift = getattr(interpolator, "_shift", None)
        scale = getattr(interpolator, "_scale", None)
        if shift is None or scale is None:
            raise RuntimeError(
                "The installed SciPy RBFInterpolator does not expose the "
                "coordinate scaling needed to print its fitted equation."
            )
        shift = np.asarray(shift, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        if shift.shape != (2,) or scale.shape != (2,) or np.any(scale == 0.0):
            raise RuntimeError(
                "Unexpected RBF coordinate scaling; cannot print the fitted equation."
            )
        first, second = "z_" + header[0], "z_" + header[1]
        print(
            f"  z_{header[0]} = ({header[0]} - {model.x_mean[0]:.10g})"
            f" / {model.x_scale[0]:.10g}"
        )
        print(
            f"  z_{header[1]} = ({header[1]} - {model.x_mean[1]:.10g})"
            f" / {model.x_scale[1]:.10g}"
        )
        print(
            f"  u_{header[0]} = (z_{header[0]} - {shift[0]:.10g})"
            f" / {scale[0]:.10g}"
        )
        print(
            f"  u_{header[1]} = (z_{header[1]} - {shift[1]:.10g})"
            f" / {scale[1]:.10g}"
        )
        print(
            f"  {header[2]} = {coefficients[center_count]:.10g}"
            f" {coefficients[center_count + 1]:+.10g}*u_{header[0]}"
            f" {coefficients[center_count + 2]:+.10g}*u_{header[1]}"
        )
        print("    + sum_i weight_i * phi(r_i), for i = 1..n")
        print("  phi(r) = r^2 * ln(r) for r > 0; phi(0) = 0")
        print(
            f"  r_i = sqrt(({first} - center_i_1)^2 "
            f"+ ({second} - center_i_2)^2)"
        )
        print("  RBF weights and centers in standardized z coordinates:")
        for index, (weight, center) in enumerate(
            zip(coefficients[:center_count], centers),
            start=1,
        ):
            print(
                f"    i={index:02d}: weight={weight:+.10g}, "
                f"center=({center[0]:.10g}, {center[1]:.10g})"
            )
    else:
        raise TypeError(f"Unsupported surrogate model type: {type(model).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RBF and quadratic surrogates with LOOCV, then plot "
            "the model with the lower LOOCV RMSE."
        )
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=Path("dataset_with_infill_2.csv"),
        help="CSV file with two design-variable columns and one response "
        "(default: dataset_with_infill_2.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_outputs2"),
        help="Directory for comparison tables, grid data, and the best-model plot "
        "(default: comparison_outputs2)",
    )
    args = parser.parse_args()

    header, x, actual = load_dataset(args.csv_file)
    results, best = compare_surrogates(x, actual)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "loocv_comparison.csv"
    grid_path = args.output_dir / "best_surrogate_grid.csv"
    plot_path = args.output_dir / "best_surrogate.png"
    write_comparison_table(comparison_path, header, x, actual, results)
    grid_first, grid_second, grid_predictions, uncertainty = write_surrogate_grid(
        grid_path,
        header,
        best.model,
        x,
        best.predictions - actual,
    )
    save_surrogate_plot(
        plot_path,
        header,
        x,
        grid_first,
        grid_second,
        grid_predictions,
        uncertainty,
        model_name=best.name,
    )

    print_comparison_table(header, x, actual, results, best)
    print_best_equation(best, header)
    print(f"Combined LOOCV table: {comparison_path}")
    print(f"Best-model response and uncertainty plot: {plot_path}")
    print(f"Best-model grid predictions: {grid_path}")


if __name__ == "__main__":
    main()