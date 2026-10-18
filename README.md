# Mechanical stress as the root cause for low-voltage degradation in commercial sodium-ion batteries

Analysis code for the paper. The data are available at https://doi.org/10.5281/zenodo.23455637.

| Folder | Content |
| --- | --- |
| `degradation_mode_analysis/` | loss of sodium inventory and of active anode and cathode material |
| `dilatometry/` | electrode thickness change and cumulative thickness change |
| `operando_ct/` | radial expansion of the jelly roll from the operando CT cross-sections |

## 1. System requirements

- Python ≥ 3.10, numpy ≥ 2.0 and pandas ≥ 2.1.
- Tested on Windows 11 with Python 3.14.5, numpy 2.5.2, scipy 1.18.0,
  pandas 3.0.5, Pillow 12.3.0 and statsmodels 0.14.6.

## 2. Installation guide

```bash
pip install numpy scipy pandas Pillow statsmodels
```

## 3. Demo and instructions for use

The README of each folder lists the commands that reproduce the results of the
article from the published dataset and the outputs they write.
