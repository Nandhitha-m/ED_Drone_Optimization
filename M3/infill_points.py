"""Select a diverse batch of surrogate-guided infill points."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from compare import ComparisonResult, compare_surrogates
from rbf_surrogate import estimate_cv_uncertainty, load_dataset


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InfillBatch:
    """Selected points and grid data used to rank them."""

    points: FloatArray
    predictions: FloatArray
    uncertainties: FloatArray
    conditional_ucb: FloatArray
    bandwidth: float
    grid_first: FloatArray
    grid_second: FloatArray
    initial_ucb: FloatArray


def _standardize(points: FloatArray, mean: FloatArray, scale: FloatArray) -> FloatArray:
    """Scale design coordinates using the observed input-data statistics."""
    return (points - mean) / scale


def _pairwise_distances(first: FloatArray, second: FloatArray) -> FloatArray:
    """Return Euclidean distances between two sets of standardized points."""
    return np.linalg.norm(first[:, np.newaxis, :] - second[np.newaxis, :, :], axis=2)


def select_parallel_infill(
    x: FloatArray,
    actual: FloatArray,
    result: ComparisonResult,
    n_points: int = 5,
    grid_size: int = 120,
    beta: float = 2.0,
) -> InfillBatch:
    """Greedily select a diverse max-UCB batch with conditional uncertainty."""
    if n_points < 1:
        raise ValueError("The number of infill points must be at least one.")
    if grid_size < 2:
        raise ValueError("The candidate grid size must be at least two.")
    if beta <= 0.0:
        raise ValueError("The UCB exploration weight must be positive.")
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Infill selection requires exactly two design variables.")
    if actual.ndim != 1 or actual.shape[0] != x.shape[0]:
        raise ValueError("Response data must be one-dimensional and match X.")

    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale == 0.0] = 1.0
    standardized_x = _standardize(x, mean, scale)

    first_axis = np.linspace(float(np.min(x[:, 0])), float(np.max(x[:, 0])), grid_size)
    second_axis = np.linspace(
        float(np.min(x[:, 1])), float(np.max(x[:, 1])), grid_size
    )
    grid_first, grid_second = np.meshgrid(first_axis, second_axis)
    candidates = np.column_stack((grid_first.ravel(), grid_second.ravel()))
    standardized_candidates = _standardize(candidates, mean, scale)

    design_distances = _pairwise_distances(standardized_x, standardized_x)
    np.fill_diagonal(design_distances, np.inf)
    nearest_design_distances = np.min(design_distances, axis=1)
    bandwidth = float(np.median(nearest_design_distances))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("Cannot select infill points when design points are duplicated.")

    nearest_observed_distance = np.min(
        _pairwise_distances(standardized_candidates, standardized_x), axis=1
    )
    exclusion_radius = 0.1 * bandwidth
    eligible = nearest_observed_distance > exclusion_radius
    if int(np.count_nonzero(eligible)) < n_points:
        raise ValueError(
            "Not enough eligible grid points remain after excluding candidates "
            "too close to existing data; increase --grid-size or reduce --n."
        )

    residuals = result.predictions - actual
    predictions = result.model.predict(candidates)
    empirical_uncertainty = estimate_cv_uncertainty(candidates, x, residuals)
    initial_ucb = predictions + beta * empirical_uncertainty
    initial_ucb_grid = initial_ucb.reshape(grid_first.shape)

    selected_indices: list[int] = []
    selected_scores: list[float] = []
    selected_uncertainties: list[float] = []
    candidate_scores = np.full(len(candidates), -np.inf, dtype=np.float64)
    candidate_scores[eligible] = initial_ucb[eligible]

    for _ in range(n_points):
        if selected_indices:
            selected_standardized = standardized_candidates[selected_indices]
            correlation_to_selected = np.exp(
                -0.5
                * np.square(
                    _pairwise_distances(
                        standardized_candidates, selected_standardized
                    )
                    / bandwidth
                )
            )
            selected_correlation = np.exp(
                -0.5
                * np.square(
                    _pairwise_distances(
                        selected_standardized, selected_standardized
                    )
                    / bandwidth
                )
            )
            selected_correlation.flat[:: len(selected_indices) + 1] += 1e-10
            conditional_projection = np.linalg.solve(
                selected_correlation, correlation_to_selected.T
            ).T
            conditional_variance_fraction = np.clip(
                1.0 - np.sum(correlation_to_selected * conditional_projection, axis=1),
                0.0,
                1.0,
            )
            conditional_uncertainty = (
                empirical_uncertainty * np.sqrt(conditional_variance_fraction)
            )
            candidate_scores[eligible] = (
                predictions[eligible] + beta * conditional_uncertainty[eligible]
            )
            candidate_scores[selected_indices] = -np.inf
        else:
            conditional_uncertainty = empirical_uncertainty

        selected_index = int(np.argmax(candidate_scores))
        if not np.isfinite(candidate_scores[selected_index]):
            raise RuntimeError(
                "Could not select another finite-scoring infill candidate."
            )
        selected_indices.append(selected_index)
        selected_scores.append(float(candidate_scores[selected_index]))
        selected_uncertainties.append(
            float(conditional_uncertainty[selected_index])
        )

    selected = np.asarray(selected_indices, dtype=np.int64)
    return InfillBatch(
        points=candidates[selected],
        predictions=predictions[selected],
        uncertainties=np.asarray(selected_uncertainties, dtype=np.float64),
        conditional_ucb=np.asarray(selected_scores, dtype=np.float64),
        bandwidth=bandwidth,
        grid_first=grid_first,
        grid_second=grid_second,
        initial_ucb=initial_ucb_grid,
    )


def write_infill_table(
    path: Path,
    header: Sequence[str],
    batch: InfillBatch,
    result: ComparisonResult,
) -> None:
    """Save selected locations and their surrogate acquisition scores."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "batch_order",
                header[0],
                header[1],
                f"predicted_{header[2]}",
                "conditional_loocv_rmse_proxy",
                "conditional_ucb_score",
                "selected_surrogate",
            ]
        )
        for index, (point, prediction, uncertainty, score) in enumerate(
            zip(
                batch.points,
                batch.predictions,
                batch.uncertainties,
                batch.conditional_ucb,
            ),
            start=1,
        ):
            writer.writerow(
                [
                    index,
                    point[0],
                    point[1],
                    prediction,
                    uncertainty,
                    score,
                    result.name,
                ]
            )


def save_infill_plot(
    path: Path,
    header: Sequence[str],
    x: FloatArray,
    actual: FloatArray,
    batch: InfillBatch,
    result: ComparisonResult,
) -> None:
    """Plot the selected response surface and the UCB acquisition map."""
    grid_points = np.column_stack(
        (batch.grid_first.ravel(), batch.grid_second.ravel())
    )
    response_grid = result.model.predict(grid_points).reshape(batch.grid_first.shape)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    response_plot = axes[0].contourf(
        batch.grid_first,
        batch.grid_second,
        response_grid,
        levels=24,
        cmap="viridis",
    )
    axes[0].scatter(
        x[:, 0],
        x[:, 1],
        c=actual,
        cmap="viridis",
        edgecolor="black",
        s=42,
        label="Existing observations",
    )
    axes[0].scatter(
        batch.points[:, 0],
        batch.points[:, 1],
        marker="*",
        c="red",
        edgecolor="white",
        linewidth=0.8,
        s=180,
        label="Parallel infill batch",
    )
    for index, point in enumerate(batch.points, start=1):
        axes[0].annotate(
            str(index),
            (point[0], point[1]),
            xytext=(5, 5),
            textcoords="offset points",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
    axes[0].set_title(f"Selected model response: {result.name}")
    axes[0].set_xlabel(header[0])
    axes[0].set_ylabel(header[1])
    axes[0].legend(loc="best")
    figure.colorbar(response_plot, ax=axes[0], label=f"Predicted {header[2]}")

    acquisition_plot = axes[1].contourf(
        batch.grid_first,
        batch.grid_second,
        batch.initial_ucb,
        levels=24,
        cmap="magma",
    )
    axes[1].scatter(
        x[:, 0], x[:, 1], marker="x", c="cyan", s=42, label="Existing observations"
    )
    axes[1].scatter(
        batch.points[:, 0],
        batch.points[:, 1],
        marker="*",
        c="lime",
        edgecolor="black",
        linewidth=0.6,
        s=180,
        label="Parallel infill batch",
    )
    for index, point in enumerate(batch.points, start=1):
        axes[1].annotate(
            str(index),
            (point[0], point[1]),
            xytext=(5, 5),
            textcoords="offset points",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
    axes[1].set_title("Initial upper-confidence-bound acquisition")
    axes[1].set_xlabel(header[0])
    axes[1].set_ylabel(header[1])
    axes[1].legend(loc="best")
    figure.colorbar(
        acquisition_plot,
        ax=axes[1],
        label="Predicted response + beta * error proxy",
    )

    figure.suptitle(
        f"Maximization infill batch (n={len(batch.points)}, beta=2)"
        f" | LOOCV RMSE={result.rmse:.5g}"
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select parallel infill points for maximizing the best LOOCV "
            "surrogate from compare.py."
        )
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=Path("dataset_with_infill_1.csv"),
        help="Dataset used by compare.py (default: dataset_with_infill_1.csv)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of infill points to select in one batch (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infill_outputs1"),
        help="Directory for the infill table and plot (default: infill_outputs1)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=120,
        help="Number of candidate coordinates per input axis (default: 120)",
    )
    args = parser.parse_args()

    header, x, actual = load_dataset(args.csv_file)
    _, best = compare_surrogates(x, actual)
    batch = select_parallel_infill(
        x,
        actual,
        best,
        n_points=args.n,
        grid_size=args.grid_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "infill_points.csv"
    plot_path = args.output_dir / "infill_points.png"
    write_infill_table(table_path, header, batch, best)
    save_infill_plot(plot_path, header, x, actual, batch, best)

    print("Parallel infill selection for maximization")
    print(f"Best LOOCV surrogate: {best.name} (RMSE={best.rmse:.6g})")
    print("Acquisition: conditional upper confidence bound (UCB)")
    print(
        "Reason: uncertainty is an empirical LOOCV error proxy rather than a "
        "calibrated posterior, so UCB avoids EI's Gaussian-posterior assumption."
    )
    print(
        "Batch selection: greedy conditional UCB; each selected point reduces "
        "nearby candidate uncertainty before the next point is chosen."
    )
    print(
        "Domain: observed input ranges; existing observations within 0.1 "
        "standardized nearest-neighbor spacings are excluded."
    )
    print(
        f"\n{'Batch':>5}  {header[0]:>12}  {header[1]:>12}  "
        f"{'Predicted ' + header[2]:>16}  {'Cond. error':>12}  {'Cond. UCB':>12}"
    )
    for index, (point, prediction, uncertainty, score) in enumerate(
        zip(
            batch.points,
            batch.predictions,
            batch.uncertainties,
            batch.conditional_ucb,
        ),
        start=1,
    ):
        print(
            f"{index:5d}  {point[0]:12.6g}  {point[1]:12.6g}  "
            f"{prediction:16.6g}  {uncertainty:12.6g}  {score:12.6g}"
        )
    print(f"\nInfill table: {table_path}")
    print(f"Infill plot: {plot_path}")


if __name__ == "__main__":
    main()