"""Fit a quadratic response surface and evaluate it with leave-one-out CV."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rbf_surrogate import (
    load_dataset,
    print_cv_report,
    save_parity_plot,
    save_surrogate_plot,
    write_cv_table,
    write_surrogate_grid,
)


FloatArray = NDArray[np.float64]


def quadratic_features(x: FloatArray) -> FloatArray:
    """Build constant, linear, squared, and interaction terms for two inputs."""
    first = x[:, 0]
    second = x[:, 1]
    return np.column_stack(
        (np.ones(len(x)), first, second, first**2, first * second, second**2)
    )


@dataclass
class QuadraticResponseSurface:
    """Least-squares quadratic response surface in standardized input space."""

    x_mean: FloatArray
    x_scale: FloatArray
    coefficients: FloatArray
    x_train: FloatArray
    y_train: FloatArray

    @classmethod
    def fit(cls, x: FloatArray, y: FloatArray) -> QuadraticResponseSurface:
        """Fit a full quadratic model with six coefficients."""
        inputs = np.asarray(x, dtype=np.float64)
        response = np.asarray(y, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError("Design data must have exactly two columns.")
        if response.ndim != 1 or response.shape[0] != inputs.shape[0]:
            raise ValueError("Response data must be one-dimensional and match X.")
        if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(response)):
            raise ValueError("Training data must contain only finite numbers.")
        if len(response) < 6:
            raise ValueError(
                "At least six training rows are required for a full quadratic model."
            )

        x_mean = np.mean(inputs, axis=0)
        x_scale = np.std(inputs, axis=0)
        x_scale[x_scale == 0.0] = 1.0
        design_matrix = quadratic_features((inputs - x_mean) / x_scale)
        coefficients, _, rank, _ = np.linalg.lstsq(
            design_matrix, response, rcond=None
        )
        if rank < design_matrix.shape[1]:
            raise ValueError(
                "The quadratic design matrix is rank-deficient; "
                "the input points do not identify all six coefficients."
            )
        return cls(x_mean, x_scale, coefficients, inputs.copy(), response.copy())

    def predict(self, x: FloatArray) -> FloatArray:
        """Predict responses for one or more two-variable design points."""
        points = np.asarray(x, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Prediction input must contain two design variables.")
        if not np.all(np.isfinite(points)):
            raise ValueError("Prediction input must contain only finite numbers.")
        return quadratic_features((points - self.x_mean) / self.x_scale) @ self.coefficients

    def save(self, path: Path) -> None:
        """Save training data needed to reproduce this response surface."""
        with path.open("wb") as model_file:
            np.savez_compressed(
                model_file,
                x_train=self.x_train,
                y_train=self.y_train,
            )

    @classmethod
    def load(cls, path: Path) -> QuadraticResponseSurface:
        """Load a saved model without unpickling executable data."""
        with np.load(path, allow_pickle=False) as model_file:
            return cls.fit(model_file["x_train"], model_file["y_train"])


def leave_one_out_predictions(x: FloatArray, y: FloatArray) -> FloatArray:
    """Return one prediction per row, each made without that row in training."""
    predictions = np.empty_like(y)
    for index in range(len(y)):
        train_mask = np.arange(len(y)) != index
        try:
            model = QuadraticResponseSurface.fit(x[train_mask], y[train_mask])
            predictions[index] = model.predict(x[index])[0]
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(
                f"Could not fit leave-one-out fold {index + 1}: {exc}"
            ) from exc
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a quadratic response surface and run leave-one-out CV."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=Path("dataset_with_infill_2.csv"),
        help="CSV file with two design-variable columns followed by one response "
        "(default: dataset_with_infill_2.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quadratic_outputs2"),
        help="Directory for LOOCV results, plots, and the fitted model "
        "(default: quadratic_outputs2)",
    )
    args = parser.parse_args()

    header, x, y = load_dataset(args.csv_file)
    predictions = leave_one_out_predictions(x, y)
    print_cv_report(header, y, predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors = predictions - y
    cv_path = args.output_dir / "loocv_errors.csv"
    parity_path = args.output_dir / "loocv_parity.png"
    surface_path = args.output_dir / "surrogate_with_uncertainty.png"
    grid_path = args.output_dir / "surrogate_grid.csv"
    model_path = args.output_dir / "quadratic_response_surface.npz"

    write_cv_table(cv_path, header, x, y, predictions)
    save_parity_plot(parity_path, header, y, predictions)
    model = QuadraticResponseSurface.fit(x, y)
    grid_first, grid_second, grid_predictions, uncertainty = write_surrogate_grid(
        grid_path, header, model, x, errors
    )
    save_surrogate_plot(
        surface_path,
        header,
        x,
        grid_first,
        grid_second,
        grid_predictions,
        uncertainty,
        model_name="Quadratic response surface",
    )
    model.save(model_path)

    coefficient_path = args.output_dir / "quadratic_coefficients.csv"
    with coefficient_path.open("w", newline="", encoding="utf-8") as coefficient_file:
        writer = csv.writer(coefficient_file)
        writer.writerow(["term", "coefficient_in_standardized_input_space"])
        writer.writerows(
            zip(
                ["intercept", "z1", "z2", "z1_squared", "z1_z2", "z2_squared"],
                model.coefficients,
            )
        )

    print(f"\nFinal quadratic response surface fitted using all {len(y)} data rows.")
    print(f"LOOCV error table: {cv_path}")
    print(f"LOOCV parity plot: {parity_path}")
    print(f"Response and uncertainty plot: {surface_path}")
    print(f"Response surface grid predictions: {grid_path}")
    print(f"Standardized polynomial coefficients: {coefficient_path}")
    print(f"Reusable fitted model: {model_path}")


if __name__ == "__main__":
    main()