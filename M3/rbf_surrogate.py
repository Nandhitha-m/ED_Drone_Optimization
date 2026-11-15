"""Fit an RBF surrogate and evaluate it with leave-one-out cross-validation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RBFInterpolator


FloatArray = NDArray[np.float64]


class PredictiveModel(Protocol):
    def predict(self, x: FloatArray) -> FloatArray: ...


@dataclass
class RBFSurrogate:
    """A thin-plate-spline RBF model with standardized design variables."""

    interpolator: RBFInterpolator
    x_mean: FloatArray
    x_scale: FloatArray
    x_train: FloatArray
    y_train: FloatArray

    @classmethod
    def fit(cls, x: FloatArray, y: FloatArray) -> RBFSurrogate:
        """Fit the surrogate to two-column design data and one response."""
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("Design data must have exactly two columns.")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("Response data must be one-dimensional and match X.")
        if x.shape[0] < 3:
            raise ValueError("At least three training rows are required.")

        x_mean = np.mean(x, axis=0)
        x_scale = np.std(x, axis=0)
        x_scale[x_scale == 0.0] = 1.0
        x_scaled = (x - x_mean) / x_scale
        interpolator = RBFInterpolator(
            x_scaled,
            y,
            kernel="thin_plate_spline",
            degree=1,
        )
        return cls(interpolator, x_mean, x_scale, x.copy(), y.copy())

    def predict(self, x: FloatArray) -> FloatArray:
        """Predict responses for one or more two-variable design points."""
        points = np.asarray(x, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Prediction input must contain two design variables.")
        if not np.all(np.isfinite(points)):
            raise ValueError("Prediction input must contain only finite numbers.")
        return np.asarray(
            self.interpolator((points - self.x_mean) / self.x_scale),
            dtype=np.float64,
        )

    def save(self, path: Path) -> None:
        """Save the training data needed to reproduce this fitted RBF model."""
        with path.open("wb") as model_file:
            np.savez_compressed(
                model_file,
                x_train=self.x_train,
                y_train=self.y_train,
            )

    @classmethod
    def load(cls, path: Path) -> RBFSurrogate:
        """Load a saved RBF model without unpickling executable data."""
        with np.load(path, allow_pickle=False) as model_file:
            return cls.fit(model_file["x_train"], model_file["y_train"])


def load_dataset(path: Path) -> tuple[list[str], FloatArray, FloatArray]:
    """Read a CSV with a header and exactly two inputs followed by one response."""
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty.") from exc

        if len(header) != 3 or any(not name.strip() for name in header):
            raise ValueError("CSV must have three non-empty column headers.")

        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 3:
                raise ValueError(
                    f"Expected three values on line {line_number}, got {len(row)}."
                )
            try:
                values = [float(value) for value in row]
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value on line {line_number}: {row}"
                ) from exc
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite value on line {line_number}.")
            rows.append(values)

    if len(rows) < 4:
        raise ValueError("At least four data rows are required for leave-one-out CV.")

    data = np.asarray(rows, dtype=np.float64)
    return header, data[:, :2], data[:, 2]


def leave_one_out_predictions(x: FloatArray, y: FloatArray) -> FloatArray:
    """Return one prediction per row, each made without that row in training."""
    predictions = np.empty_like(y)
    for index in range(len(y)):
        train_mask = np.arange(len(y)) != index
        try:
            model = RBFSurrogate.fit(x[train_mask], y[train_mask])
            predictions[index] = model.predict(x[index])[0]
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(
                f"Could not fit leave-one-out fold {index + 1}: {exc}"
            ) from exc
    return predictions


def estimate_cv_uncertainty(
    query: FloatArray, x: FloatArray, cv_errors: FloatArray
) -> FloatArray:
    """Estimate local prediction error from distance-weighted LOOCV residuals."""
    x_mean = np.mean(x, axis=0)
    x_scale = np.std(x, axis=0)
    x_scale[x_scale == 0.0] = 1.0
    scaled_x = (x - x_mean) / x_scale
    scaled_query = (query - x_mean) / x_scale
    distances = np.linalg.norm(
        scaled_query[:, np.newaxis, :] - scaled_x[np.newaxis, :, :], axis=2
    )
    training_distances = np.linalg.norm(
        scaled_x[:, np.newaxis, :] - scaled_x[np.newaxis, :, :], axis=2
    )
    training_distances[training_distances == 0.0] = np.inf
    bandwidth = float(np.median(np.min(training_distances, axis=1)))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("Cannot estimate uncertainty: design points are duplicated.")

    weights = np.exp(-0.5 * np.square(distances / bandwidth))
    weight_sums = np.sum(weights, axis=1)
    if np.any(weight_sums == 0.0):
        raise RuntimeError("Could not estimate uncertainty for one or more points.")
    variance = (weights @ np.square(cv_errors)) / weight_sums
    return np.sqrt(variance)


def write_cv_table(
    path: Path,
    header: Sequence[str],
    x: FloatArray,
    y: FloatArray,
    predictions: FloatArray,
) -> None:
    """Write per-row LOOCV predictions and errors to CSV."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["row", *header[:2], "actual", "loo_predicted", "error", "absolute_error"]
        )
        for index, (inputs, actual, predicted) in enumerate(
            zip(x, y, predictions), start=1
        ):
            error = float(predicted - actual)
            writer.writerow(
                [
                    index,
                    *inputs,
                    actual,
                    predicted,
                    error,
                    abs(error),
                ]
            )


def write_surrogate_grid(
    path: Path,
    header: Sequence[str],
    model: PredictiveModel,
    x: FloatArray,
    cv_errors: FloatArray,
    grid_size: int = 100,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Evaluate the fitted surrogate and empirical uncertainty over the input range."""
    first = np.linspace(float(np.min(x[:, 0])), float(np.max(x[:, 0])), grid_size)
    second = np.linspace(float(np.min(x[:, 1])), float(np.max(x[:, 1])), grid_size)
    grid_first, grid_second = np.meshgrid(first, second)
    grid_points = np.column_stack((grid_first.ravel(), grid_second.ravel()))
    predictions = model.predict(grid_points)
    uncertainty = estimate_cv_uncertainty(grid_points, x, cv_errors)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([header[0], header[1], "predicted_response", "estimated_cv_rmse"])
        writer.writerows(
            zip(
                grid_points[:, 0],
                grid_points[:, 1],
                predictions,
                uncertainty,
            )
        )
    return grid_first, grid_second, predictions.reshape(grid_first.shape), uncertainty.reshape(
        grid_first.shape
    )


def save_parity_plot(
    path: Path, header: Sequence[str], y: FloatArray, predictions: FloatArray
) -> None:
    """Plot leave-one-out predictions against measured responses."""
    rmse = float(np.sqrt(np.mean(np.square(predictions - y))))
    lower = float(min(np.min(y), np.min(predictions)))
    upper = float(max(np.max(y), np.max(predictions)))
    fig, axis = plt.subplots(figsize=(6.5, 6))
    axis.scatter(y, predictions, color="tab:blue", edgecolor="black", alpha=0.85)
    axis.plot([lower, upper], [lower, upper], "k--", label="Perfect prediction")
    axis.fill_between(
        [lower, upper],
        [lower - rmse, upper - rmse],
        [lower + rmse, upper + rmse],
        color="tab:blue",
        alpha=0.12,
        label="± LOOCV RMSE reference band",
    )
    axis.set(
        xlabel=f"Actual {header[2]}",
        ylabel=f"LOOCV predicted {header[2]}",
        title="LOOCV parity plot",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_surrogate_plot(
    path: Path,
    header: Sequence[str],
    x: FloatArray,
    grid_first: FloatArray,
    grid_second: FloatArray,
    predictions: FloatArray,
    uncertainty: FloatArray,
    model_name: str = "RBF",
) -> None:
    """Plot the response surface and the LOOCV-calibrated uncertainty estimate."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    response_plot = axes[0].contourf(
        grid_first, grid_second, predictions, levels=20, cmap="viridis"
    )
    axes[0].scatter(
        x[:, 0], x[:, 1], c="white", edgecolor="black", s=35, label="Data"
    )
    axes[0].set_title(f"{model_name}: {header[2]}")
    axes[0].set_xlabel(header[0])
    axes[0].set_ylabel(header[1])
    axes[0].legend()
    fig.colorbar(response_plot, ax=axes[0], label=f"Predicted {header[2]}")

    uncertainty_plot = axes[1].contourf(
        grid_first, grid_second, uncertainty, levels=20, cmap="magma"
    )
    axes[1].scatter(
        x[:, 0], x[:, 1], c="white", edgecolor="black", s=35, label="Data"
    )
    axes[1].set_title("Estimated local LOOCV error (RMSE)")
    axes[1].set_xlabel(header[0])
    axes[1].set_ylabel(header[1])
    axes[1].legend()
    fig.colorbar(uncertainty_plot, ax=axes[1], label="Estimated response error")
    fig.suptitle("Surrogate response and empirical uncertainty")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_cv_report(
    header: Sequence[str], y: FloatArray, predictions: FloatArray
) -> None:
    """Print per-row LOO predictions and aggregate regression metrics."""
    residuals = predictions - y
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))
    total_sum_squares = float(np.sum(np.square(y - np.mean(y))))
    r_squared = (
        float(1.0 - np.sum(np.square(residuals)) / total_sum_squares)
        if total_sum_squares > 0.0
        else float("nan")
    )

    print(f"Leave-one-out cross-validation ({len(y)} folds)")
    print(f"{'Row':>4}  {'Actual':>12}  {'Predicted':>12}  {'Residual':>12}")
    for index, (actual, predicted, residual) in enumerate(
        zip(y, predictions, residuals), start=1
    ):
        print(f"{index:4d}  {actual:12.6g}  {predicted:12.6g}  {residual:12.6g}")
    print(f"\nInputs: {header[0]}, {header[1]} (standardized before fitting)")
    print(f"Response: {header[2]}")
    print(f"RMSE: {rmse:.6g}")
    print(f"MAE:  {mae:.6g}")
    print(f"R^2:  {r_squared:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a thin-plate-spline RBF surrogate and run leave-one-out CV."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=Path("dataset_with_infill_2.csv"),
        help="CSV file with a header and two input columns followed by one response "
        "(default: dataset_with_infill_2.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rbf_outputs2"),
        help="Directory for LOOCV results, plots, and the fitted model "
        "(default: rbf_outputs2)",
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
    model_path = args.output_dir / "rbf_surrogate_model.npz"

    write_cv_table(cv_path, header, x, y, predictions)
    save_parity_plot(parity_path, header, y, predictions)
    model = RBFSurrogate.fit(x, y)
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
    )
    model.save(model_path)

    print(f"\nFinal surrogate fitted using all {len(y)} data rows.")
    print(f"LOOCV error table: {cv_path}")
    print(f"LOOCV parity plot: {parity_path}")
    print(f"Surrogate and uncertainty plot: {surface_path}")
    print(f"Surrogate grid predictions: {grid_path}")
    print(f"Reusable fitted model: {model_path}")


if __name__ == "__main__":
    main()