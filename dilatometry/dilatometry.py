"""Electrochemical dilatometry: thickness change against voltage and cumulative
thickness change of every aged cell.

    python dilatometry.py --data <dataset> --output results

Writes thickness_change.csv and cumulative_thickness_change.csv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import glob
import os
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, make_smoothing_spline
from scipy.optimize import curve_fit


# =============================================================================
# 1. Dilatometer measurements
#    Anode against sodium (anode.txt) and the reassembled full stack of anode,
#    separator and cathode (full_stack.txt).
# =============================================================================

MEASUREMENTS = {"anode": ("anode.txt", "V1R"), "full_stack": ("full_stack.txt", "V12")}
NUMERIC_COLUMNS = ("V1R", "V12", "|Qcd|", "Dilation", "cycle", "t", "I12")
FULL_STACK_DISCHARGE = "c"
# Charge in mAh still stored in the anode at the end of the full-stack discharge,
# per dilatometry cycle; it shifts the anode charge axis onto the full-stack one.
ANODE_OFFSET_MAH = {1: 0.1853842973155856, 3: 0.1461102058253929}
# Dilatometry cycle whose discharge is used (cycle 0 is the initial rest).
THICKNESS_CHANGE_CYCLE = 1  # thickness_change.csv
CUMULATIVE_CYCLE = 3        # cumulative_thickness_change.csv
SMOOTHING_P = 0.95
GRID_POINTS = 1000


def _drift(t, amplitude, tau, offset):
    return amplitude * np.exp(-t / tau) + offset


def load_measurements(folder: Path) -> dict[str, pd.DataFrame]:
    """Both measurements with the drift-corrected thickness change dil_corr.

    The initial rest (cycle 0) holds no electrochemistry, so A exp(-t/tau) + C
    fitted to it is the drift of the sensor; A exp(-t/tau) is subtracted.
    """
    measurements = {}
    for name, (file, _) in MEASUREMENTS.items():
        frame = pd.read_csv(folder / file, sep="\t", header=0, skiprows=[1], low_memory=False)
        frame.columns = frame.columns.str.strip()
        frame["c/d"] = frame["c/d"].astype(str).str.strip()
        for column in NUMERIC_COLUMNS:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        rest = frame[frame["cycle"] == 0].dropna(subset=["t"])
        t0 = rest["t"].min()
        t_rest, y_rest = (rest["t"] - t0).to_numpy(), rest["Dilation"].to_numpy()
        finite = np.isfinite(t_rest) & np.isfinite(y_rest)
        t_rest, y_rest = t_rest[finite], y_rest[finite]
        (amplitude, tau, _), _ = curve_fit(
            _drift, t_rest, y_rest,
            p0=[y_rest[0] - y_rest[-1], t_rest.max() / 3, y_rest[-1]], maxfev=10000)
        frame["dil_corr"] = frame["Dilation"] - amplitude * np.exp(-(frame["t"] - t0).to_numpy() / tau)
        measurements[name] = frame
    return measurements


def branch(measurements, name: str, direction: str, cycle: int, drop_rest: bool) -> pd.DataFrame:
    """One half cycle sorted by charge; with drop_rest, rows without current are removed."""
    frame = measurements[name]
    selected = frame[(frame["c/d"] == direction) & (frame["cycle"] == cycle)]
    selected = selected.dropna(subset=["|Qcd|", MEASUREMENTS[name][1]])
    # Only where the current is logged; anode.txt logs 0 throughout.
    if drop_rest and len(selected) and (selected["I12"].abs() > 1e-6).mean() > 0.5:
        selected = selected[selected["I12"].abs() > 1e-6]
    return selected.sort_values("|Qcd|")


@dataclass(frozen=True)
class ReferenceCurves:
    """Thickness change against full-stack voltage, zero at the lowest voltage."""

    voltage: np.ndarray
    full_stack: np.ndarray
    anode: np.ndarray

    @property
    def cathode(self) -> np.ndarray:
        return self.full_stack - self.anode


def build_reference(measurements, cycle: int, drop_rest: bool) -> ReferenceCurves:
    """Full stack and anode on the full-stack voltage axis of one discharge.

    The anode thickness change at full-stack charge Q is that of the half cell
    at Q + ANODE_OFFSET_MAH; the cathode is the difference to the full stack.
    """
    full = branch(measurements, "full_stack", FULL_STACK_DISCHARGE, cycle, drop_rest)
    q_full, v_full = full["|Qcd|"].to_numpy(), -full["V12"].to_numpy()
    thickness = (full["dil_corr"] - full["dil_corr"].dropna().iloc[0]).to_numpy()
    order = np.argsort(v_full)
    voltage = v_full[order]
    full_stack = thickness[order] - thickness[order][0]

    anode = branch(measurements, "anode", "c", cycle, drop_rest)
    anode_thickness = (anode["dil_corr"] - anode["dil_corr"].dropna().iloc[0]).to_numpy()
    voltage_to_q = interp1d(voltage, q_full[order], bounds_error=False,
                            fill_value=(q_full[order][0], q_full[order][-1]))
    q_to_anode = interp1d(anode["|Qcd|"].to_numpy(), anode_thickness, bounds_error=False,
                          fill_value=(anode_thickness[0], anode_thickness[-1]))
    anode_on_voltage = q_to_anode(voltage_to_q(voltage) + ANODE_OFFSET_MAH[cycle])
    return ReferenceCurves(voltage, full_stack, anode_on_voltage - anode_on_voltage[0])


def smoothing_spline(voltage, thickness, p: float = SMOOTHING_P):
    """Cubic smoothing spline as MATLAB's csaps(x, y, p), i.e. lam = (1 - p) / p.

    The spline needs strictly increasing voltages; of samples logged at the
    same voltage only the first is kept.
    """
    finite = np.isfinite(voltage) & np.isfinite(thickness)
    order = np.argsort(voltage[finite], kind="stable")
    v, d = voltage[finite][order], thickness[finite][order]
    v, first = np.unique(v, return_index=True)
    return make_smoothing_spline(v, d[first], lam=(1.0 - p) / p)


def thickness_change(measurements) -> pd.DataFrame:
    """Smoothed thickness change and its derivative dL/dV for the discharge."""
    curves = build_reference(measurements, THICKNESS_CHANGE_CYCLE, drop_rest=True)
    grid = np.linspace(curves.voltage.min(), curves.voltage.max(), GRID_POINTS)
    table = {"voltage_V": grid}
    for name, values in (("full_stack", curves.full_stack), ("anode", curves.anode),
                         ("cathode", curves.cathode)):
        spline = smoothing_spline(curves.voltage, values)
        curve = spline(grid)
        table[f"{name}_um"] = curve - curve[0]
        table[f"d{name}_dV_um_per_V"] = spline.derivative()(grid)
    return pd.DataFrame(table)


# =============================================================================
# 2. Cycling and RPT data
#    One zip per file, cell<ID>_RPT<n>.zip or cell<ID>_Cyc<n>.zip; the
#    condition of every cell is listed in cells.csv.
# =============================================================================

FILE_NAME = re.compile(r"cell(\d{3})_(RPT|Cyc)\d+\.zip$")
COLUMNS = ["time", "state", "cycle", "voltage_V", "current_A"]


def parse_name(path) -> tuple[str, str]:
    """Cell and kind of file, RPT or Cyc."""
    return FILE_NAME.search(Path(path).name).groups()


def read(path) -> dict[str, np.ndarray]:
    """cycle, state (CHA, DCH or PAU), voltage, current and time in s."""
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(member) as handle:
            data = pd.read_csv(handle, usecols=COLUMNS, dtype={"state": str})
    seconds = pd.to_datetime(data["time"], format="ISO8601")
    return {
        "cycle": data["cycle"].to_numpy(),
        # A trailing * marks the same direction in a different sub-step.
        "state": np.char.rstrip(data["state"].to_numpy(dtype=str), "*"),
        "voltage": data["voltage_V"].to_numpy(dtype=float),
        "current": data["current_A"].to_numpy(dtype=float),
        "time": (seconds - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(),
    }


# =============================================================================
# 3. Cumulative thickness change
#    For every charge and discharge half cycle of the cycling data:
#    OCV = V - R(V, sgn I) I with the 10 s pulse resistance at 2C; the OCV is
#    mapped onto the thickness change of the pristine electrodes and the
#    absolute changes between successive samples are summed, leaving out the
#    first SKIP_AFTER_REVERSAL samples after each current reversal. The charge
#    throughput counts every sample; the RPT files add throughput only.
# =============================================================================

NOMINAL_CAPACITY_AH = 1.2
SKIP_AFTER_REVERSAL = 60

_REFERENCE = None   # set in every worker
_RESISTANCE = None


def _interpolator(x, y):
    return interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))


def _init_worker(curves: ReferenceCurves, resistance_table: pd.DataFrame):
    global _REFERENCE, _RESISTANCE
    _REFERENCE = (_interpolator(curves.voltage, curves.full_stack),
                  _interpolator(curves.voltage, curves.anode),
                  (float(curves.voltage[0]), float(curves.voltage[-1])))
    _RESISTANCE = {}
    for direction, table in resistance_table.groupby("direction"):
        table = table.dropna().sort_values("voltage_V")
        _RESISTANCE[direction] = _interpolator(table["voltage_V"].to_numpy(float),
                                               table["resistance_mOhm"].to_numpy(float) / 1000.0)


def traversed_thickness(voltage, current) -> tuple[float, float, float]:
    """Summed absolute thickness change of full stack, anode and cathode."""
    resistance = np.where(current < 0, _RESISTANCE["discharge"](voltage),
                          _RESISTANCE["charge"](voltage))
    ocv = voltage - current * resistance
    ocv = ocv[np.isfinite(ocv)]
    full_stack, anode, (lower, upper) = _REFERENCE
    ocv = np.clip(ocv, lower, upper)
    full = np.asarray(full_stack(ocv), dtype=float)
    an = np.asarray(anode(ocv), dtype=float)
    return tuple(float(np.nansum(np.abs(np.diff(x, prepend=x[0])))) for x in (full, an, full - an))


def analyse_file(path: str, is_rpt: bool) -> np.ndarray:
    """Charge throughput in Ah and thickness change of full stack, anode and cathode."""
    data = read(path)
    cycle, state, time = data["cycle"], data["state"], data["time"]
    voltage, current = data["voltage"], data["current"]
    by_cycle = np.argsort(cycle, kind="stable")
    _, starts = np.unique(cycle[by_cycle], return_index=True)
    bounds = np.append(starts, len(by_cycle))
    total = np.zeros(4)
    for number in range(len(starts)):
        in_cycle = by_cycle[bounds[number]:bounds[number + 1]]
        for direction in ("CHA", "DCH"):
            index = in_cycle[state[in_cycle] == direction]
            valid = np.isfinite(voltage[index]) & np.isfinite(current[index]) & np.isfinite(time[index])
            index = index[valid]
            if len(index) < 2:
                continue
            index = index[np.argsort(time[index])]
            v, i, t = voltage[index], current[index], time[index]
            # Throughput over every contiguous run of samples.
            runs = np.split(np.arange(len(index)), np.flatnonzero(np.diff(index) > 1) + 1)
            total[0] += sum(np.trapezoid(np.abs(i[run]), t[run]) / 3600.0
                            for run in runs if len(run) >= 2)
            if not is_rpt and len(v) > SKIP_AFTER_REVERSAL + 1:
                total[1:] += traversed_thickness(v[SKIP_AFTER_REVERSAL:], i[SKIP_AFTER_REVERSAL:])
    return total


def cumulative_thickness_change(data_dir: Path, curves: ReferenceCurves,
                                resistance_table: pd.DataFrame, jobs: int) -> pd.DataFrame:
    cells = pd.read_csv(data_dir / "cells.csv", dtype={"cell": str})
    files = pd.DataFrame([(*parse_name(path), path)
                          for path in sorted(glob.glob(os.path.join(data_dir, "*.zip")))],
                         columns=["cell", "kind", "path"])
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                             initargs=(curves, resistance_table)) as pool:
        totals = np.asarray(list(pool.map(analyse_file, files["path"], files["kind"] == "RPT",
                                          chunksize=4)))
    columns = ["efc", "full_stack_um", "anode_um", "cathode_um"]
    per_file = pd.DataFrame(totals, columns=columns)
    per_file["efc"] /= 2 * NOMINAL_CAPACITY_AH
    per_file["cell"] = files["cell"]
    per_cell = cells.merge(per_file.groupby("cell")[columns].sum().reset_index(), on="cell")
    per_cell["full_stack_um_per_efc"] = per_cell["full_stack_um"] / per_cell["efc"]
    return per_cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="root folder of the dataset")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cycling = args.data / "cycling_and_rpt"
    measurements = load_measurements(args.data / "dilatometry")
    thickness_change(measurements).to_csv(args.output / "thickness_change.csv", index=False)
    cumulative_thickness_change(
        cycling, build_reference(measurements, CUMULATIVE_CYCLE, drop_rest=False),
        pd.read_csv(args.data / "dilatometry" / "pulse_resistance.csv"), args.jobs,
    ).to_csv(args.output / "cumulative_thickness_change.csv", index=False)


if __name__ == "__main__":
    main()
