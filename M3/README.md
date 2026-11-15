# RBF surrogate

This project fits a radial basis function (RBF) surrogate to a CSV dataset and
reports leave-one-out cross-validation (LOOCV) predictions and metrics.

## Dataset format

The CSV must have a header and exactly three columns, in this order:

1. First design variable
2. Second design variable
3. Response/result

All data rows must contain finite numeric values. The included `dataset.csv`
uses the columns `D_m`, `T_N`, and `FM`.

## Setup

From the project directory, create and activate a virtual environment, then
install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Use the included dataset:

```powershell
python rbf_surrogate.py
```

Or specify a CSV path:

```powershell
python rbf_surrogate.py path\to\your_data.csv
```

Each fold fits a new model using all other rows. The script prints held-out
predictions and aggregate RMSE, MAE, and R-squared. It also creates an
`rbf_outputs` directory containing:

- `loocv_errors.csv`: actual, leave-one-out predicted, and signed/absolute
  errors for every row.
- `loocv_parity.png`: parity plot with a ± LOOCV RMSE reference band.
- `surrogate_with_uncertainty.png`: response contour and estimated local
  prediction error over the design-variable range.
- `surrogate_grid.csv`: predictions and estimated errors on a 100 × 100 grid.
- `rbf_surrogate_model.npz`: the training data needed to reproduce the fitted
  model using all dataset rows.

Use `--output-dir` to choose another output folder:

```powershell
python rbf_surrogate.py dataset.csv --output-dir results
```

The uncertainty shown is an empirical local RMSE estimate: absolute LOOCV
residuals are combined by distance-weighted RMS in standardized input space.
It is a data-based error indicator, not a formal confidence interval.

## Use the fitted model in Python

```python
from pathlib import Path

from rbf_surrogate import RBFSurrogate, load_dataset

_, x, y = load_dataset(Path("dataset.csv"))
model = RBFSurrogate.fit(x, y)
predictions = model.predict([[0.5, 20.0], [0.6, 25.0]])
print(predictions)
```

The saved fitted model can also be loaded for reuse:

```python
from pathlib import Path

from rbf_surrogate import RBFSurrogate

model = RBFSurrogate.load(Path("rbf_outputs/rbf_surrogate_model.npz"))
print(model.predict([[0.5, 20.0]]))
```

The implementation uses SciPy's thin-plate-spline `RBFInterpolator` with
degree-1 polynomial terms. Both design variables are standardized using the
training data before fitting, which keeps their different units and ranges
from dominating the radial distances.

## Quadratic response surface

To run the same LOOCV, error table, parity plot, response-and-uncertainty plot,
grid export, and reusable-model workflow with a full quadratic response
surface instead of RBF:

```powershell
python quadratic_res_surf.py
```

The script also accepts a CSV path and `--output-dir`:

```powershell
python quadratic_res_surf.py path\to\your_data.csv --output-dir quadratic_outputs
```

By default, outputs are saved in `quadratic_outputs`. The six-term model
contains an intercept, two linear terms, both squared terms, and their
interaction. `quadratic_coefficients.csv` lists the coefficients in
standardized-input coordinates. The model standardizes inputs before fitting
and requires at least six non-collinear design points. To load the saved model, use
`QuadraticResponseSurface.load(Path("quadratic_outputs/quadratic_response_surface.npz"))`
from `quadratic_res_surf`.

## Compare the surrogate methods

Run both methods on the same dataset, compare their per-row LOOCV predictions
and aggregate metrics, and plot the model with the lower LOOCV RMSE:

```powershell
python compare.py
```

The comparison accepts a dataset path and output directory:

```powershell
python compare.py path\to\your_data.csv --output-dir comparison_outputs
```

The output directory contains `loocv_comparison.csv` with both models'
predictions and signed errors side by side, `best_surrogate.png` with the
selected model's response surface and estimated local LOOCV error, and
`best_surrogate_grid.csv` with the grid values used to make the plot. The
winner is selected by the smaller LOOCV RMSE. The console output also prints
the selected model's fitted equation, including the RBF weights and centers
when the RBF model wins.

## Select parallel infill points

After comparing the models, select a batch of 10 candidate design points to
maximize the response with the winning surrogate:

```powershell
python infill_points.py
```

You can change the dataset, batch size, candidate-grid resolution, or output
directory:

```powershell
python infill_points.py path\to\your_data.csv --n 10 --grid-size 120 --output-dir infill_outputs
```

The script selects the best model using the same LOOCV-RMSE comparison as
`compare.py`. It uses a **conditional upper confidence bound (UCB)** for this
maximization: predicted response plus twice the empirical local LOOCV error
estimate. The LOOCV error is a data-based proxy, not a calibrated posterior
standard deviation, so UCB is more transparent here than expected improvement,
which assumes a probabilistic predictive distribution. The batch is selected
greedily in parallel: after each choice, a distance-based conditional
uncertainty reduction discourages nearby redundant points. Candidates are
restricted to the observed input ranges and kept away from existing samples.
The output directory contains `infill_points.csv` and `infill_points.png`.

## Maximize the best surrogate

Find the maximum predicted response from the model with the lowest LOOCV RMSE:

```powershell
python maximize_surrogate.py
```

By default, this uses `dataset_with_infill_2.csv`, matching the comparison
script's current default, and searches within the observed minimum and maximum
of each input variable. Pass a dataset path or output directory as needed:

```powershell
python maximize_surrogate.py path\to\your_data.csv --output-dir optimization_outputs
```

The script uses SciPy's differential evolution with local polishing to search
for the maximum of the selected model's prediction. It prints the selected
model, the maximizing inputs, the predicted response, and the highest observed
response for context. The predicted optimum is a surrogate estimate and should
be validated with a real evaluation. A response-surface plot is saved as
`best_surrogate_maximum.png`.
