"""Find the maximum of the best LOOCV-selected surrogate model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

from compare import compare_surrogates
from rbf_surrogate import PredictiveModel, load_dataset


FloatArray = NDArray[np.float64]


def save_optimization_plot(
    path: Path,
    header: list[str],
    x: FloatArray,
    actual: FloatArray,
    model: PredictiveModel,
    optimum: FloatArray,
    maximum: float,
    grid_size: int = 200,
) -> None:
    """Plot the fitted response surface, observations, and predicted maximum."""
    first = np.linspace(float(np.min(x[:, 0])), float(np.max(x[:, 0])), grid_size)
    second = np.linspace(float(np.min(x[:, 1])), float(np.max(x[:, 1])), grid_size)
    grid_first, grid_second = np.meshgrid(first, second)
    grid_points = np.column_stack((grid_first.ravel(), grid_second.ravel()))
    response = model.predict(grid_points).reshape(grid_first.shape)

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    surface = axis.contourf(
        grid_first, grid_second, response, levels=30, cmap="viridis"
    )
    observations = axis.scatter(
        x[:, 0],
        x[:, 1],
        c=actual,
        cmap="viridis",
        edgecolor="black",
        s=42,
        label="Observed data",
    )
    axis.scatter(
        [optimum[0]],
        [optimum[1]],
        marker="*",
        c="red",
        edgecolor="white",
        linewidth=0.9,
        s=240,
        label=f"Predicted maximum: {maximum:.6g}",
        zorder=3,
    )
    axis.set(
        xlabel=header[0],
        ylabel=header[1],
        title=f"Best surrogate response surface: {header[2]}",
    )
    axis.legend(loc="best")
    figure.colorbar(surface, ax=axis, label=f"Predicted {header[2]}")
    figure.colorbar(observations, ax=axis, label=f"Observed {header[2]}")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best surrogate by LOOCV RMSE, then maximize its "
            "predicted response within the observed input ranges."
        )
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=Path("dataset_with_infill_2.csv"),
        help="Dataset used for comparison (default: dataset_with_infill_2.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optimization_outputs"),
        help="Directory for the optimization plot (default: optimization_outputs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the global optimizer (default: 42)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum differential-evolution iterations (default: 1000)",
    )
    args = parser.parse_args()

    if args.maxiter < 1:
        parser.error("--maxiter must be at least one.")

    header, x, actual = load_dataset(args.csv_file)
    _, best = compare_surrogates(x, actual)
    bounds = [
        (float(np.min(x[:, index])), float(np.max(x[:, index])))
        for index in range(x.shape[1])
    ]

    optimization = differential_evolution(
        lambda point: -float(best.model.predict(np.asarray(point))[0]),
        bounds=bounds,
        seed=args.seed,
        maxiter=args.maxiter,
        tol=1e-9,
        polish=True,
    )
    if not optimization.success:
        raise RuntimeError(
            "Surrogate optimization did not converge: "
            f"{optimization.message}"
        )

    optimum = np.asarray(optimization.x, dtype=np.float64)
    maximum = float(best.model.predict(optimum)[0])
    observed_best_index = int(np.argmax(actual))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "best_surrogate_maximum.png"
    save_optimization_plot(
        plot_path,
        header,
        x,
        actual,
        best.model,
        optimum,
        maximum,
    )

    print("Surrogate maximization")
    print(f"Selected model: {best.name}")
    print(f"LOOCV RMSE: {best.rmse:.8g}")
    print("Search bounds: observed minimum to maximum for each input")
    print(f"Maximum predicted {header[2]}: {maximum:.10g}")
    print("Inputs at the predicted maximum:")
    for name, value in zip(header[:2], optimum):
        print(f"  {name} = {value:.10g}")
    print(
        f"Highest observed {header[2]}: {actual[observed_best_index]:.10g} "
        f"at {header[0]}={x[observed_best_index, 0]:.10g}, "
        f"{header[1]}={x[observed_best_index, 1]:.10g}"
    )
    print(
        "The surrogate maximum is a model prediction; its response value "
        "requires evaluation to confirm experimentally."
    )
    print(f"Optimization plot: {plot_path}")


if __name__ == "__main__":
    main()
