# Degradation-mode analysis

Loss of sodium inventory (LSI) and loss of active anode and cathode material
(LAMNE, LAMPE) from the discharge qOCV of every RPT of all 81 cells. All
settings of the article are the defaults in `dma.py` (class `Settings`).

```bash
python dma.py --input <dataset>/degradation_mode_analysis --output results
```

## Output

- `fit_results.csv`: per cell and RPT the fitted electrode parameters, the
  degradation modes `lsi`, `lamne` and `lampe`, and the qOCV and dV/dSOC
  errors of the voltage rebuilt from them (`traj_qocv_rmse_v`, `traj_dv_rmse_v`)
- `condition_means.csv`: mean and standard deviation of the modes and errors
  per condition up to 9,000 EFCs

## Relation to Rehm et al.

The analysis follows the method of the open-source framework of Rehm et al.
(https://github.com/tum-ees/DegradationModeAnalysis, MIT License), implemented
here in Python and extended. Please also cite M. Rehm et al., J. Power Sources
(2026) and M. Rehm et al., EES Batteries 2, 665–682 (2026).

| Part | Origin |
| --- | --- |
| Electrode model with scaling and offset per electrode | Rehm et al. |
| Electrode inhomogeneity model | Rehm et al. |
| Objective from the qOCV and dV/dSOC residuals | Rehm et al. |
| Resampling and LOWESS smoothing of the qOCV | Rehm et al. |
| Several fits per RPT | Rehm et al. |
| Additional fits started from the solution of the previous RPT | this work |
| Selection by dV/dSOC peak distances and continuity with the previous RPT | this work |
| Monotone, smoothed trajectories over all RPTs of a cell | this work |
