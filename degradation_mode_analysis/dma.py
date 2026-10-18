"""Degradation-mode analysis: LSI, LAMNE and LAMPE from the discharge qOCV of every RPT.

    python dma.py --input <dataset>/degradation_mode_analysis --output results

Writes fit_results.csv (one row per cell and RPT) and condition_means.csv
(mean and standard deviation per condition up to 9,000 EFCs).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
import os
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, differential_evolution, minimize
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess


# =============================================================================
# Settings of the article
# =============================================================================

@dataclass(frozen=True)
class Settings:
    input_dir: Path
    reference_csv: Path
    jobs: int = 1
    seed: int = 20260415

    # qOCV preprocessing
    n_soc: int = 1000
    lowess_frame: int = 30
    min_curve_points: int = 100
    min_voltage_span_v: float = 1.5
    capacity_agreement_fraction: float = 0.01

    # Objective
    fit_voltage_v: tuple[float, float] = (2.0, 3.8)
    qocv_weight: float = 1.0
    dv_weight: float = 50.0
    rmse_limit_v: float = 0.008           # fits above are marginal ...
    marginal_rmse_limit_v: float = 0.050  # ... and above this unsupported

    # Electrode inhomogeneity
    max_inhomogeneity: float = 0.3
    max_inhomogeneity_delta: float = 0.1  # per RPT

    # Candidate search per RPT
    candidate_dv_weights: tuple[float, ...] = (50.0, 0.0, 10.0, 1.0)
    first_starts: int = 1
    first_escalated_starts: int = 2
    later_starts: int = 2
    later_escalated_starts: int = 4
    max_global_attempts: int = 20
    max_accepted_candidates: int = 15
    min_feasible_candidates: int = 3
    warm_start_local_attempts: int = 4
    warm_start_radii: tuple[float, ...] = (0.025, 0.012, 0.025, 0.012, 0.05, 0.05)
    de_maxiter: int = 45
    de_popsize: int = 7
    local_maxiter: int = 250

    # Candidate selection
    peak_tracking_window_ah: float = 0.055
    peak_distance_scale_ah: float = 0.004
    peak_distance_weight: float = 100.0
    continuity_weight: float = 100.0
    continuity_scales: tuple[float, ...] = (0.06, 0.03, 0.06, 0.03, 0.05, 0.05)

    # Trajectories
    roughness_weight: float = 1.0
    mode_scale: float = 0.02
    max_mode: float = 0.60
    marginal_scale_multiplier: float = 2.0
    marginal_max_scale_multiplier: float = 5.0
    lampe_negative_tolerance: float = 0.02
    lampe_negative_penalty_weight: float = 75.0
    lampe_negative_bound: float = 0.10
    lamne_negative_bound: float = 0.015
    lsi_negative_tolerance: float = 0.01
    lsi_negative_penalty_weight: float = 5.0
    lsi_negative_bound: float = 0.05


SUPPORTED = ("accepted", "marginal")
MODES = ("lamne", "lampe", "lsi")


# =============================================================================
# 1. Data
# =============================================================================

@dataclass(frozen=True)
class RawCurve:
    cell: str
    condition: str
    rpt: int
    efc: float
    qocv_capacity_ah: float
    charge_ah: np.ndarray          # starts at the empty cell
    voltage: np.ndarray
    peak_locations_ah: dict        # dV/dSOC maxima HP1.. and minima LP1..
    peak_distances_ah: dict        # Diff_<A><B>; NaN where not reliable


@lru_cache(maxsize=2)
def _rpt_table(path: str) -> pd.DataFrame:
    return pd.read_csv(Path(path) / "rpt.csv", dtype={"cell": str, "condition": str},
                       float_precision="round_trip")


@lru_cache(maxsize=2)
def _cell_curves(path: str, cell: str) -> dict:
    frame = pd.read_csv(Path(path) / "qocv_discharge" / f"cell{cell}.zip",
                        float_precision="round_trip")
    return {int(rpt): (group["charge_Ah"].to_numpy(float), group["voltage_V"].to_numpy(float))
            for rpt, group in frame.groupby("rpt", sort=False)}


def read_rpt_table(path: Path) -> pd.DataFrame:
    frame = _rpt_table(str(Path(path).resolve()))
    table = frame[["cell", "condition", "rpt", "efc", "capacity_Ah"]].copy()
    return table.sort_values(["cell", "rpt"]).reset_index(drop=True)


def read_curve(path: Path, cell: str, rpt: int) -> RawCurve:
    path = str(Path(path).resolve())
    frame = _rpt_table(path)
    row = frame[(frame["cell"] == cell) & (frame["rpt"] == rpt)].iloc[0]
    capacity, voltage = _cell_curves(path, cell)[rpt]
    if np.any(np.isfinite(capacity)):
        # Charge removed during discharge, mirrored to start at the empty cell.
        capacity = float(np.nanmax(capacity)) - capacity
        order = np.argsort(capacity, kind="stable")
        capacity, voltage = capacity[order], voltage[order]
    peaks = {name[:-3]: float(row[name]) for name in frame.columns
             if name.endswith("_Ah") and name not in ("capacity_Ah", "qocv_capacity_Ah")}
    return RawCurve(cell, str(row["condition"]), rpt, float(row["efc"]),
                    float(row["qocv_capacity_Ah"]), capacity, voltage,
                    {k: v for k, v in peaks.items() if not k.startswith("Diff_")},
                    {k: v for k, v in peaks.items() if k.startswith("Diff_")})


# =============================================================================
# 2. qOCV preprocessing: SOC grid and LOWESS smoothing
# =============================================================================

@dataclass(frozen=True)
class Curve:
    soc: np.ndarray
    voltage: np.ndarray
    dv_dsoc: np.ndarray
    capacity_ah: float


class CurveError(ValueError):
    pass


def lowess_frame(y: np.ndarray, x: np.ndarray, frame: int) -> np.ndarray:
    """LOWESS over a fixed number of points, as MATLAB's smooth(..., 'lowess')."""
    if y.size < 3 or frame <= 1:
        return y.copy()
    frac = min(1.0, max(3, int(frame)) / y.size)
    return np.asarray(lowess(y, x, frac=frac, it=0, is_sorted=True, return_sorted=False),
                      dtype=float)


def check_curve(charge_ah: np.ndarray, voltage: np.ndarray, s: Settings):
    """Finite, strictly increasing charge axis and a sufficient voltage span."""
    finite = np.isfinite(charge_ah) & np.isfinite(voltage)
    q, voltage = charge_ah[finite], voltage[finite]
    if q.size < s.min_curve_points:
        raise CurveError("too_few_points")
    order = np.argsort(q, kind="mergesort")
    q, voltage = q[order], voltage[order]
    q, start, counts = np.unique(q, return_index=True, return_counts=True)
    voltage = voltage[start] if np.all(counts == 1) else np.add.reduceat(voltage, start) / counts
    if q.size < s.min_curve_points:
        raise CurveError("too_few_points")
    if q[-1] <= q[0]:
        raise CurveError("non_increasing_capacity")
    if float(np.ptp(voltage)) < s.min_voltage_span_v:
        raise CurveError("voltage_span_too_small")
    return q, voltage


def preprocess(raw: RawCurve, s: Settings) -> Curve:
    q, voltage = check_curve(raw.charge_ah, raw.voltage, s)
    q = q - q[0]
    span = float(q[-1])
    # The reported qOCV capacity is used when it agrees with the measured span.
    capacity = span
    if np.isfinite(raw.qocv_capacity_ah) and raw.qocv_capacity_ah > 0:
        if abs(raw.qocv_capacity_ah - span) / span <= s.capacity_agreement_fraction:
            capacity = raw.qocv_capacity_ah
    soc = np.linspace(0.0, 1.0, s.n_soc)
    smooth = lowess_frame(np.interp(soc, q / span, voltage), soc, s.lowess_frame)
    dv = lowess_frame(np.gradient(smooth, soc, edge_order=2), soc, s.lowess_frame)
    return Curve(soc, smooth, dv, capacity)


# =============================================================================
# 3. Electrode model
#    Cell voltage = cathode - anode potential, each mapped onto the cell SOC by a
#    scaling alpha and an offset beta. Inhomogeneity averages an electrode
#    potential over a Gaussian distribution of local states of charge.
#    params = [alpha_ca, beta_ca, alpha_an, beta_an, inhom_an, inhom_ca].
# =============================================================================

INHOMOGENEITY_SUPPORT = np.linspace(0.5, 1.5, 61)


def _inhomogeneity_weights(spread: float) -> np.ndarray:
    z = (INHOMOGENEITY_SUPPORT - 1.0) / spread
    weights = np.exp(-0.5 * z * z)
    return weights / weights.sum()


def _inhomogeneity_basis(soc: np.ndarray, potential: np.ndarray) -> np.ndarray:
    # The spread grows linearly from zero at SOC 0 to its full value at SOC 1.
    query = soc[:, None] + soc[:, None] * (INHOMOGENEITY_SUPPORT - 1.0)[None, :]
    return np.interp(query.ravel(), soc, potential).reshape(query.shape)


@dataclass(frozen=True)
class Electrodes:
    soc: np.ndarray
    cathode_v: np.ndarray
    anode_v: np.ndarray
    cathode_dv: np.ndarray
    anode_dv: np.ndarray
    cathode_basis: np.ndarray
    anode_basis: np.ndarray

    @classmethod
    def from_csv(cls, path: Path, s: Settings) -> "Electrodes":
        data = pd.read_csv(path)
        soc = data["soc"].to_numpy(float)
        cathode = lowess_frame(data["cathode_potential_v"].to_numpy(float), soc, s.lowess_frame)
        anode = lowess_frame(data["anode_potential_v"].to_numpy(float), soc, s.lowess_frame)
        return cls(soc, cathode, anode,
                   lowess_frame(np.gradient(cathode, soc, edge_order=2), soc, s.lowess_frame),
                   lowess_frame(np.gradient(anode, soc, edge_order=2), soc, s.lowess_frame),
                   _inhomogeneity_basis(soc, cathode), _inhomogeneity_basis(soc, anode))

    def smeared(self, cathode_spread: float, anode_spread: float) -> "Electrodes":
        if cathode_spread <= 0.0 and anode_spread <= 0.0:
            return self
        cathode, anode = self.cathode_v, self.anode_v
        if cathode_spread > 0.0:
            cathode = self.cathode_basis @ _inhomogeneity_weights(cathode_spread)
        if anode_spread > 0.0:
            anode = self.anode_basis @ _inhomogeneity_weights(anode_spread)
        return replace(self, cathode_v=cathode, anode_v=anode,
                       cathode_dv=np.gradient(cathode, self.soc, edge_order=2),
                       anode_dv=np.gradient(anode, self.soc, edge_order=2))


def _with_inhomogeneity(params: np.ndarray, electrodes: Electrodes):
    if params.size > 4:
        electrodes = electrodes.smeared(float(params[5]), float(params[4]))
    return params[:4], electrodes


def _coordinates(soc, params):
    alpha_ca, beta_ca, alpha_an, beta_an = params[:4]
    return (soc - beta_ca) / alpha_ca, (soc - beta_an) / alpha_an


def domain_covers(params: np.ndarray, roi: tuple[float, float]) -> bool:
    ca, an = _coordinates(np.asarray(roi, dtype=float), params)
    return bool(np.all((ca >= 0.0) & (ca <= 1.0) & (an >= 0.0) & (an <= 1.0)))


def cell_voltage(soc, params, electrodes: Electrodes) -> np.ndarray:
    params, electrodes = _with_inhomogeneity(np.asarray(params, dtype=float), electrodes)
    ca_soc, an_soc = _coordinates(soc, params)
    ca = np.interp(ca_soc, electrodes.soc, electrodes.cathode_v, left=np.nan, right=np.nan)
    an = np.interp(an_soc, electrodes.soc, electrodes.anode_v, left=np.nan, right=np.nan)
    return ca - an


def cell_dv(soc, params, electrodes: Electrodes) -> np.ndarray:
    params, electrodes = _with_inhomogeneity(np.asarray(params, dtype=float), electrodes)
    ca_soc, an_soc = _coordinates(soc, params)
    ca = np.interp(ca_soc, electrodes.soc, electrodes.cathode_dv, left=np.nan, right=np.nan)
    an = np.interp(an_soc, electrodes.soc, electrodes.anode_dv, left=np.nan, right=np.nan)
    return ca / params[0] - an / params[2]


def capacities(params, capacity_ah: float) -> tuple[float, float, float]:
    """Anode capacity, cathode capacity and sodium inventory in Ah."""
    alpha_ca, beta_ca, alpha_an, beta_an = np.asarray(params, dtype=float)[:4]
    return (float(alpha_an * capacity_ah), float(alpha_ca * capacity_ah),
            float((alpha_ca + beta_ca - beta_an) * capacity_ah))


def degradation_modes(params, capacity_ah: float, baseline) -> tuple[float, float, float]:
    """LAMNE, LAMPE and LSI relative to the baseline capacities."""
    c_an, c_ca, c_inv = capacities(params, capacity_ah)
    return 1.0 - c_an / baseline[0], 1.0 - c_ca / baseline[1], 1.0 - c_inv / baseline[2]


def params_from_modes(modes, capacity_ah: float, baseline, beta_an: float) -> np.ndarray:
    c_an, c_ca, c_inv = (b * (1.0 - m) for b, m in zip(baseline, modes))
    alpha_an, alpha_ca = c_an / capacity_ah, c_ca / capacity_ah
    return np.array([alpha_ca, beta_an + c_inv / capacity_ah - alpha_ca, alpha_an, beta_an])


# =============================================================================
# 4. Fit of one RPT
#    Several fits per RPT, also started from the solution of the previous RPT;
#    selection by the objective, dV/dSOC peak distances and continuity.
# =============================================================================

PARAMETER_BOUNDS = ((0.50, 1.60), (-0.50, 0.20), (0.50, 1.60), (-0.50, 0.20))
PARAMETER_WIDTH = np.concatenate([np.ptp(np.asarray(PARAMETER_BOUNDS), axis=1), [0.3, 0.3]])


@dataclass(frozen=True)
class Candidate:
    params: np.ndarray
    objective: float
    qocv_sse: float
    dv_sse: float
    qocv_rmse_v: float
    dv_rmse_v: float
    domain_feasible: bool
    rmse_accepted: bool
    search_stage: str
    peak_score: float = float("nan")
    selection_score: float = float("nan")
    likelihood: float = 0.0


def fit_mask(curve: Curve, s: Settings) -> np.ndarray:
    low, high = s.fit_voltage_v
    return np.isfinite(curve.voltage) & (curve.voltage >= low) & (curve.voltage <= high)


def fit_domain(curve: Curve, s: Settings) -> tuple[float, float]:
    mask = fit_mask(curve, s)
    if not np.any(mask):
        return 0.0, 1.0
    return float(np.min(curve.soc[mask])), float(np.max(curve.soc[mask]))


def domain_constraint(roi: tuple[float, float]) -> LinearConstraint:
    """beta <= SOC_min and alpha + beta >= SOC_max for both electrodes."""
    lo, hi = roi
    matrix = np.asarray([[0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    return LinearConstraint(matrix, [-lo, hi, -lo, hi], np.full(4, np.inf))


def residual_sums(params, curve: Curve, electrodes: Electrodes, s: Settings):
    mask = fit_mask(curve, s)
    if np.count_nonzero(mask) < 3:
        return float("inf"), float("inf")
    voltage = cell_voltage(curve.soc, params, electrodes)
    dv = cell_dv(curve.soc, params, electrodes)
    if not (np.all(np.isfinite(voltage[mask])) and np.all(np.isfinite(dv[mask]))):
        return float("inf"), float("inf")
    qocv_residual = voltage[mask] - curve.voltage[mask]
    dv_residual = dv[mask] - curve.dv_dsoc[mask]
    return float(qocv_residual @ qocv_residual), float(dv_residual @ dv_residual)


def _search_objective(curve, electrodes, s: Settings, dv_weight: float):
    constraint = domain_constraint(fit_domain(curve, s))

    def objective(params):
        margins = constraint.A @ np.asarray(params, dtype=float) - constraint.lb
        if np.any(margins < 0.0):
            violation = np.minimum(margins, 0.0)
            return float(1.0e8 + 1.0e8 * (violation @ violation))
        qocv_sse, dv_sse = residual_sums(params, curve, electrodes, s)
        if not np.isfinite(qocv_sse + dv_sse):
            return 1.0e8
        return s.qocv_weight * qocv_sse + float(dv_weight) * dv_sse

    return objective


def _candidate(params, curve, electrodes, s: Settings, stage: str) -> Candidate:
    params = np.asarray(params, dtype=float)
    qocv_sse, dv_sse = residual_sums(params, curve, electrodes, s)
    points = np.count_nonzero(fit_mask(curve, s))
    finite = np.isfinite(qocv_sse + dv_sse)
    qocv_rmse = float(np.sqrt(qocv_sse / points)) if finite else float("inf")
    dv_rmse = float(np.sqrt(dv_sse / points)) if finite else float("inf")
    domain_ok = domain_covers(params, fit_domain(curve, s))
    return Candidate(params, s.qocv_weight * qocv_sse + s.dv_weight * dv_sse, qocv_sse, dv_sse,
                     qocv_rmse, dv_rmse, domain_ok,
                     bool(domain_ok and qocv_rmse <= s.rmse_limit_v * (1.0 + 1e-8)), stage)


def parameter_bounds(s: Settings, previous_inhomogeneity) -> tuple:
    """Inhomogeneity may grow by at most max_inhomogeneity_delta per RPT."""
    bounds = list(PARAMETER_BOUNDS)
    for earlier in previous_inhomogeneity:
        upper = min(s.max_inhomogeneity, earlier + s.max_inhomogeneity_delta)
        bounds.append((0.0, max(0.0, upper)))
    return tuple((float(lo), float(hi)) for lo, hi in bounds)


def _interior_start(params, roi, bounds) -> np.ndarray:
    output = np.asarray(params, dtype=float).copy()
    lo, hi = roi
    for alpha, beta in ((0, 1), (2, 3)):
        output[beta] = min(output[beta], lo - 2e-5)
        output[alpha] = max(output[alpha], hi - output[beta] + 2e-5)
    return np.clip(output, [b[0] for b in bounds], [b[1] for b in bounds])


def _global_candidates(curve, electrodes, s: Settings, bounds, minimum, maximum, seed_offset):
    constraints = (domain_constraint(fit_domain(curve, s)),)
    candidates: list[Candidate] = []
    for attempt in range(max(1, min(maximum, s.max_global_attempts))):
        weight = s.candidate_dv_weights[attempt % len(s.candidate_dv_weights)]
        objective = _search_objective(curve, electrodes, s, weight)
        result = differential_evolution(
            objective, bounds, constraints=constraints, seed=s.seed + seed_offset + attempt,
            maxiter=s.de_maxiter, popsize=s.de_popsize, tol=1e-8, atol=1e-10, polish=False,
            updating="immediate", workers=1)
        polished = minimize(objective, result.x, method="SLSQP", bounds=bounds,
                            constraints=constraints,
                            options={"maxiter": s.local_maxiter, "ftol": 1e-12})
        candidates.append(_candidate(polished.x, curve, electrodes, s, "global"))
        feasible = sum(c.rmse_accepted for c in candidates)
        if (attempt + 1 >= minimum and feasible >= s.min_feasible_candidates) \
                or feasible >= s.max_accepted_candidates:
            break
    return candidates


def _local_candidates(warm, curve, electrodes, s: Settings, bounds, seed_offset):
    """Starts close to the solution of the previous RPT."""
    lower = np.asarray([b[0] for b in bounds])
    upper = np.asarray([b[1] for b in bounds])
    center = np.clip(warm, lower, upper)
    radius = np.asarray(s.warm_start_radii)
    local_bounds = tuple((float(lo), float(hi)) for lo, hi in
                         zip(np.maximum(lower, center - radius), np.minimum(upper, center + radius)))
    roi = fit_domain(curve, s)
    start = _interior_start(warm, roi, local_bounds)
    rng = np.random.default_rng(s.seed + seed_offset - 17)
    starts = [start] + [_interior_start(start + rng.normal(0.0, np.asarray(s.continuity_scales) * 0.35),
                                        roi, local_bounds)
                        for _ in range(s.warm_start_local_attempts - 1)]
    constraints = (domain_constraint(roi),)
    candidates = []
    for attempt, x0 in enumerate(starts):
        weight = s.candidate_dv_weights[attempt % len(s.candidate_dv_weights)]
        result = minimize(_search_objective(curve, electrodes, s, weight), x0, method="SLSQP",
                          bounds=local_bounds, constraints=constraints,
                          options={"maxiter": s.local_maxiter, "ftol": 1e-12})
        candidates.append(_candidate(result.x, curve, electrodes, s, "local"))
    return candidates


def _distance(left: Candidate, right: Candidate) -> float:
    return float(np.linalg.norm((left.params - right.params) / PARAMETER_WIDTH))


def _pareto(candidates: list[Candidate]) -> list[Candidate]:
    """Accepted, distinct candidates that are not dominated in (qOCV, dV/dSOC) error."""
    feasible = sorted((c for c in candidates if c.rmse_accepted
                       and np.isfinite(c.qocv_sse) and np.isfinite(c.dv_sse)),
                      key=lambda c: (c.qocv_sse, c.dv_sse, c.objective))
    distinct: list[Candidate] = []
    for c in feasible:
        if not any(_distance(c, other) < 1e-4 for other in distinct):
            distinct.append(c)
    pareto = [c for i, c in enumerate(distinct)
              if not any(o.qocv_sse <= c.qocv_sse and o.dv_sse <= c.dv_sse
                         and (o.qocv_sse < c.qocv_sse or o.dv_sse < c.dv_sse)
                         for j, o in enumerate(distinct) if j != i)]
    return sorted(pareto, key=lambda c: c.objective)


def _select(candidates: list[Candidate], s: Settings, warm) -> tuple[Candidate, list[Candidate]]:
    """Score = normalized objective + peak-distance term + continuity term."""
    pool = _pareto(candidates)
    if warm is not None:
        for c in candidates:
            if c.search_stage == "local" and c.rmse_accepted \
                    and not any(_distance(c, other) < 1e-4 for other in pool):
                pool.append(c)
    pool = pool or [c for c in candidates if c.domain_feasible and np.isfinite(c.objective)]
    if not pool:
        raise RuntimeError("no_finite_solution")
    objectives = np.asarray([c.objective for c in pool])
    objective_scale = max(
        float(np.subtract(*np.nanpercentile(objectives, [75, 25]))) if len(pool) > 1 else 0.0,
        abs(float(np.nanmin(objectives))) * 0.02, 1e-10)
    base = float(np.nanmin(objectives))
    scores = []
    for c in pool:
        peak = c.peak_score if np.isfinite(c.peak_score) else 0.0
        continuity = 0.0
        if warm is not None:
            delta = np.abs((c.params - warm) / np.asarray(s.continuity_scales))
            continuity = float(np.mean(np.where(delta <= 1.0, 0.5 * delta**2, delta - 0.5)))
        scores.append((c.objective - base) / objective_scale + s.peak_distance_weight * peak
                      + s.continuity_weight * continuity)
    scores = np.asarray(scores)
    likelihood = np.exp(-0.5 * (scores - np.min(scores)))
    likelihood /= max(float(np.sum(likelihood)), 1e-12)
    scored = [replace(c, selection_score=float(score), likelihood=float(weight))
              for c, score, weight in zip(pool, scores, likelihood)]
    accepted = [c for c in scored if c.rmse_accepted]
    return min(accepted or scored, key=lambda c: c.selection_score), scored


def fit_rpt(curve, raw, electrodes, s: Settings, rpt, warm, previous_inhomogeneity,
            peak_context, seed_offset):
    bounds = parameter_bounds(s, previous_inhomogeneity)
    minimum = s.first_starts if rpt == 1 else s.later_starts
    maximum = s.first_escalated_starts if rpt == 1 else s.later_escalated_starts
    candidates = []
    if warm is not None:
        candidates.extend(_local_candidates(warm, curve, electrodes, s, bounds, seed_offset))
        if sum(c.rmse_accepted for c in candidates) >= s.min_feasible_candidates:
            minimum, maximum = min(2, minimum), min(2, maximum)
    candidates.extend(_global_candidates(curve, electrodes, s, bounds, minimum, maximum,
                                         seed_offset))
    if peak_context is not None:
        aged = peak_locations_in_window(raw.peak_locations_ah, raw.charge_ah, raw.voltage,
                                        s.fit_voltage_v)
        candidates = [replace(c, peak_score=peak_distance_score(
            peak_context, aged, raw.peak_distances_ah,
            modeled_peak_locations(c.params, curve, electrodes, aged, s), s))
            for c in candidates]
    best, scored = _select(candidates, s, warm)
    caps = capacities(best.params, curve.capacity_ah)
    reasons = []
    if not best.rmse_accepted:
        reasons.append("qocv_rmse_above_limit")
    if not best.domain_feasible:
        reasons.append("electrode_domain_not_covered")
    if any(not np.isfinite(value) or value <= 0 for value in caps):
        reasons.append("nonpositive_capacity")
    return best, scored, caps, reasons


def classify(reasons: list[str], qocv_rmse: float, s: Settings) -> str:
    if not reasons:
        return "accepted"
    if reasons == ["qocv_rmse_above_limit"] and np.isfinite(qocv_rmse) \
            and qocv_rmse <= s.marginal_rmse_limit_v * (1.0 + 1e-8):
        return "marginal"
    return "unreliable"


# =============================================================================
# 5. dV/dSOC peak distances
#    The change of the measured distances between labelled dV/dSOC peaks since
#    the baseline RPT is compared with the change predicted by a candidate.
# =============================================================================

FEATURE_RE = re.compile(r"^Diff_(HP\d+)(HP\d+|LP\d+)$")


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _labels(name: str):
    match = FEATURE_RE.match(name)
    return (match.group(1), match.group(2)) if match else None


def matched_peak_distances(reference_locations, reference_distances,
                           aged_locations, aged_distances) -> dict:
    """Distances available at both RPTs; NaN marks a distance judged unreliable."""
    matched = {}
    for name, reference_distance in reference_distances.items():
        labels = _labels(name)
        aged_distance = aged_distances.get(name, np.nan)
        if labels is None or not (_finite(reference_distance) and _finite(aged_distance)):
            continue
        if not all(_finite(reference_locations.get(label, np.nan))
                   and _finite(aged_locations.get(label, np.nan)) for label in labels):
            continue
        matched[name] = (float(reference_distance), float(aged_distance))
    return matched


def peak_locations_in_window(locations: Mapping, charge_ah, voltage, window) -> dict:
    """Peak positions in Ah; NaN for peaks outside the fitted voltage window."""
    finite = np.isfinite(charge_ah) & np.isfinite(voltage)
    q, v = charge_ah[finite], voltage[finite]
    if len(q) < 2:
        return {label: float("nan") for label in locations}
    order = np.argsort(q)
    q, v = q[order] - q[order][0], v[order]
    output = {}
    for label, location in locations.items():
        if not _finite(location) or location < q[0] or location > q[-1]:
            output[label] = float("nan")
            continue
        inside = window[0] <= float(np.interp(location, q, v)) <= window[1]
        output[label] = float(location) if inside else float("nan")
    return output


def modeled_peak_locations(params, curve: Curve, electrodes: Electrodes,
                           observed: Mapping, s: Settings) -> dict:
    """Extremum of the modelled dV/dQ nearest to each measured peak."""
    q_ah = curve.soc * curve.capacity_ah
    dva = cell_dv(curve.soc, params, electrodes) / curve.capacity_ah
    spacing = max(float(np.nanmedian(np.diff(q_ah))), 1e-8)
    distance = max(2, int(round(0.012 / spacing)))
    maxima = q_ah[find_peaks(dva, prominence=0.005, distance=distance)[0]]
    minima = q_ah[find_peaks(-dva, prominence=0.005, distance=distance)[0]]
    output = {}
    for label, location in observed.items():
        if not _finite(location):
            continue
        extrema = maxima if label.startswith("HP") else minima
        if extrema.size == 0:
            continue
        nearest = float(extrema[np.argmin(np.abs(extrema - location))])
        if abs(nearest - location) <= s.peak_tracking_window_ah:
            output[label] = nearest
    return output


def _peak_distance(locations: Mapping, name: str) -> float:
    labels = _labels(name)
    if labels is None or any(label not in locations for label in labels):
        return float("nan")
    return abs(float(locations[labels[1]]) - float(locations[labels[0]]))


def huber(residual) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    absolute = np.abs(residual)
    return np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)


def peak_distance_score(context: dict, aged_locations, aged_distances,
                        modeled_aged_locations, s: Settings) -> float:
    """Mean Huber loss between modelled and measured change of the peak distances."""
    observed = matched_peak_distances(context["locations"], context["distances"],
                                      aged_locations, aged_distances)
    residuals = []
    for name, (reference, aged) in observed.items():
        modeled_reference = _peak_distance(context["modeled_locations"], name)
        modeled_aged = _peak_distance(modeled_aged_locations, name)
        if _finite(modeled_reference) and _finite(modeled_aged):
            residuals.append((modeled_aged - modeled_reference) - (aged - reference))
    standardized = np.asarray(residuals, dtype=float) / max(s.peak_distance_scale_ah, 1e-9)
    return float(np.mean(huber(standardized))) if standardized.size else float("nan")


# =============================================================================
# 6. Trajectories
#    The per-RPT fits of a cell are combined into degradation-mode trajectories
#    that start at zero at the baseline RPT, increase monotonically and are
#    penalized for roughness. Each RPT enters with the likelihood-weighted mean
#    and spread of its candidates.
# =============================================================================

def _roughness(modes: np.ndarray, efc: np.ndarray) -> float:
    if len(modes) < 3:
        return 0.0
    slopes = np.diff(modes, axis=0) / (np.maximum(np.diff(efc), 1.0) / 1000.0)[:, None]
    return float(np.mean(np.diff(slopes, axis=0) ** 2))


def fit_trajectory(observed, scale, efc, s: Settings, fix_reference: bool, floor):
    """Monotone, smooth trajectory of the three modes over one segment of RPTs."""
    scale = np.maximum(scale, 1e-4)
    n = len(observed)

    def objective(flat):
        modes = flat.reshape(observed.shape)
        value = float(np.mean(((modes - observed) / scale) ** 2))
        value += s.roughness_weight * _roughness(modes, efc) / max(s.mode_scale**2, 1e-12)
        value += s.lampe_negative_penalty_weight * float(
            np.mean((np.minimum(modes[:, 1], 0.0) / s.lampe_negative_tolerance) ** 2))
        value += s.lsi_negative_penalty_weight * float(
            np.mean((np.minimum(modes[:, 2], 0.0) / s.lsi_negative_tolerance) ** 2))
        return value / optimization_scale

    def gradient(flat):
        modes = flat.reshape(observed.shape)
        result = 2.0 * (modes - observed) / (scale**2 * observed.size)
        result[:, 1] += (2.0 * s.lampe_negative_penalty_weight * np.minimum(modes[:, 1], 0.0)
                         / (s.lampe_negative_tolerance**2 * n))
        result[:, 2] += (2.0 * s.lsi_negative_penalty_weight * np.minimum(modes[:, 2], 0.0)
                         / (s.lsi_negative_tolerance**2 * n))
        if n >= 3 and s.roughness_weight > 0:
            delta = np.maximum(np.diff(efc), 1.0) / 1000.0
            slopes = np.diff(modes, axis=0) / delta[:, None]
            curvature = np.diff(slopes, axis=0)
            curvature_gradient = (2.0 * s.roughness_weight * curvature
                                  / (max(s.mode_scale**2, 1e-12) * curvature.size))
            slope_gradient = np.zeros_like(slopes)
            slope_gradient[:-1] -= curvature_gradient
            slope_gradient[1:] += curvature_gradient
            roughness_gradient = np.zeros_like(modes)
            roughness_gradient[:-1] -= slope_gradient / delta[:, None]
            roughness_gradient[1:] += slope_gradient / delta[:, None]
            result += roughness_gradient
        return result.reshape(-1) / optimization_scale

    bounds = []
    for index in range(n):
        lamne_lower = 0.0 if fix_reference and index == 0 else -s.lamne_negative_bound
        bounds.extend(((lamne_lower, s.max_mode), (-s.lampe_negative_bound, s.max_mode),
                       (-s.lsi_negative_bound, s.max_mode)))
    start = observed.copy()
    start[:, 0] = np.maximum(start[:, 0], -s.lamne_negative_bound)
    start[:, 1] = np.maximum(start[:, 1], -s.lampe_negative_bound)
    start[:, 2] = np.maximum(start[:, 2], -s.lsi_negative_bound)
    if fix_reference:
        bounds[:3] = [(0.0, 0.0)] * 3
        start[0] = 0.0
    elif floor is not None:
        # A lost mode stays lost across a gap of unsupported RPTs.
        bounds[:3] = [(max(lo, v), max(hi, v)) for (lo, hi), v in zip(bounds[:3], floor)]
        start[0] = np.maximum(start[0], floor)
    for index in range(1, n):
        start[index] = np.maximum(start[index], start[index - 1])
    constraints = ()
    if n > 1:
        # Monotonicity: every mode is non-decreasing from one RPT to the next.
        rows = []
        for index in range(1, n):
            for mode in range(3):
                row = np.zeros(3 * n)
                row[3 * index + mode], row[3 * (index - 1) + mode] = 1.0, -1.0
                rows.append(row)
        constraints = (LinearConstraint(np.vstack(rows), np.full(len(rows), -0.0),
                                        np.full(len(rows), np.inf)),)
    start = start.reshape(-1)
    # The objective is scaled so that its largest gradient at the start is one.
    optimization_scale = 1.0
    optimization_scale = max(1.0, float(np.max(np.abs(gradient(start)))))
    result = minimize(objective, start, jac=gradient, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 1800, "ftol": 1e-9})
    return (result.x.reshape(observed.shape), True) if result.success else (None, False)


def _observation(row: dict, candidates: list, s: Settings):
    """Likelihood-weighted mean and spread of the candidates of one RPT."""
    values = np.asarray([c[1] for c in candidates if c[0] == row["rpt"]], dtype=float)
    weights = np.asarray([c[2] for c in candidates if c[0] == row["rpt"]], dtype=float)
    if len(values):
        keep = np.all(np.isfinite(values), axis=1) & np.isfinite(weights)
        values, weights = values[keep], weights[keep]
    if len(values):
        weights = weights / max(float(np.sum(weights)), 1e-12)
        mean = np.average(values, axis=0, weights=weights)
        spread = np.sqrt(np.average((values - mean) ** 2, axis=0, weights=weights))
        scale = np.maximum(spread, s.mode_scale / 4.0)
    else:
        mean = np.asarray([row[f"raw_{m}"] for m in MODES], dtype=float)
        scale = np.full(3, s.mode_scale)
    if row["fit_status"] == "marginal":
        # The uncertainty grows from 2x at the 8 mV limit to 5x at 50 mV.
        width = max(s.marginal_rmse_limit_v - s.rmse_limit_v, 1e-12)
        position = float(np.clip((row["qocv_rmse_v"] - s.rmse_limit_v) / width, 0.0, 1.0))
        scale = scale * float(s.marginal_scale_multiplier + position
                              * (s.marginal_max_scale_multiplier - s.marginal_scale_multiplier))
    return mean, scale


def add_trajectories(rows: list[dict], curves: dict, candidates: list, s: Settings) -> None:
    ordered = sorted(rows, key=lambda r: r["rpt"])
    baseline_rpt = next((r["rpt"] for r in ordered if r["fit_status"] in SUPPORTED), None)
    supported = np.asarray([i for i, r in enumerate(ordered)
                            if r["rpt"] in curves and r["fit_status"] in SUPPORTED
                            and all(np.isfinite(r[f"raw_{m}"]) for m in MODES)], dtype=int)
    if baseline_rpt is None or not len(supported):
        return
    floor = None
    for segment in np.split(supported, np.flatnonzero(np.diff(supported) > 1) + 1):
        segment_rows = [ordered[i] for i in segment]
        observed, scale = map(np.asarray, zip(*(_observation(r, candidates, s)
                                                for r in segment_rows)))
        efc = np.asarray([r["efc"] for r in segment_rows], dtype=float)
        latent, success = fit_trajectory(observed, scale, efc, s,
                                         segment_rows[0]["rpt"] == baseline_rpt, floor)
        if not success:
            continue
        floor = latent[-1].copy()
        for row, modes in zip(segment_rows, latent):
            row.update(dict(zip(MODES, map(float, modes))))


def add_trajectory_errors(rows: list[dict], curves: dict, electrodes: Electrodes,
                          s: Settings) -> None:
    """qOCV and dV/dSOC RMSE of the voltage rebuilt from the trajectories.

    The trajectories fix the balancing parameters; the inhomogeneity of each
    electrode is taken from the fit of the same RPT.
    """
    baseline = next((r for r in sorted(rows, key=lambda r: r["rpt"])
                     if r["fit_status"] in SUPPORTED), None)
    if baseline is None:
        return
    baseline_caps = capacities([baseline[p] for p in ("alpha_ca", "beta_ca", "alpha_an",
                                                      "beta_an")], baseline["fit_capacity_ah"])
    for row in rows:
        modes = [row.get(m, np.nan) for m in MODES]
        if row["rpt"] not in curves or not np.all(np.isfinite(modes + [row["beta_an"]])):
            continue
        curve = curves[row["rpt"]]
        params = np.concatenate([
            params_from_modes(modes, row["fit_capacity_ah"], baseline_caps, row["beta_an"]),
            [row["inhom_an"], row["inhom_ca"]]])
        mask = fit_mask(curve, s)
        for name, model, measured in (("traj_qocv_rmse_v", cell_voltage, curve.voltage),
                                      ("traj_dv_rmse_v", cell_dv, curve.dv_dsoc)):
            residual = (model(curve.soc, params, electrodes) - measured)[mask]
            residual = residual[np.isfinite(residual)]
            if residual.size:
                row[name] = float(np.sqrt(np.mean(residual**2)))


# =============================================================================
# 7. Analysis of all cells
# =============================================================================

def fit_cell(args) -> list[dict]:
    cell, rpts, s = args
    electrodes = Electrodes.from_csv(s.reference_csv, s)
    rows, curves, candidates = [], {}, []
    baseline = baseline_params = baseline_curve = baseline_raw = None
    warm, previous_inhomogeneity = None, (0.0, 0.0)
    for rpt in sorted(rpts):
        raw = read_curve(s.input_dir, cell, rpt)
        row = {"cell": cell, "condition": raw.condition, "rpt": rpt, "efc": raw.efc}
        rows.append(row)
        try:
            curve = preprocess(raw, s)
        except CurveError as error:
            row.update(fit_status="skipped", reason=str(error))
            continue
        context = None
        if baseline_raw is not None:
            locations = peak_locations_in_window(baseline_raw.peak_locations_ah,
                                                 baseline_raw.charge_ah, baseline_raw.voltage,
                                                 s.fit_voltage_v)
            context = {"locations": locations, "distances": baseline_raw.peak_distances_ah,
                       "modeled_locations": modeled_peak_locations(
                           baseline_params, baseline_curve, electrodes, locations, s)}
        try:
            best, scored, caps, reasons = fit_rpt(
                curve, raw, electrodes, s, rpt, warm, previous_inhomogeneity,
                context, seed_offset=int(cell) * 10000 + rpt * 100)
        except Exception as error:  # an optimizer failure loses this RPT only
            row.update(fit_status="failed", reason=f"{type(error).__name__}: {error}")
            continue
        curves[rpt] = curve
        status = classify(reasons, best.qocv_rmse_v, s)
        if baseline is None and status not in SUPPORTED:
            reasons, status = reasons + ["no_baseline_yet"], "unreliable"
        modes = (degradation_modes(best.params, curve.capacity_ah, baseline)
                 if baseline is not None else (0.0, 0.0, 0.0))
        row.update(
            fit_status=status, reason=";".join(reasons), fit_capacity_ah=curve.capacity_ah,
            **dict(zip(("alpha_ca", "beta_ca", "alpha_an", "beta_an", "inhom_an", "inhom_ca"),
                       map(float, best.params))),
            qocv_rmse_v=best.qocv_rmse_v, dv_rmse_v=best.dv_rmse_v,
            **{f"raw_{m}": float(v) for m, v in zip(MODES, modes)})
        for c in scored:
            candidate_modes = (degradation_modes(c.params, curve.capacity_ah, baseline)
                               if baseline is not None else (0.0, 0.0, 0.0))
            candidates.append((rpt, candidate_modes, c.likelihood))
        if status in SUPPORTED:
            if baseline is None:
                baseline, baseline_params = caps, best.params
                baseline_curve, baseline_raw = curve, raw
            previous_inhomogeneity = (float(best.params[4]), float(best.params[5]))
            warm = best.params
    add_trajectories(rows, curves, candidates, s)
    add_trajectory_errors(rows, curves, electrodes, s)
    print(f"cell {cell} done", flush=True)
    return rows


def run(s: Settings, output_dir: Path) -> pd.DataFrame:
    table = read_rpt_table(s.input_dir)
    jobs = [(cell, group["rpt"].astype(int).tolist(), s)
            for cell, group in table.groupby("cell", sort=True)]
    with ProcessPoolExecutor(max_workers=s.jobs) as pool:
        rows = [row for cell_rows in pool.map(fit_cell, jobs) for row in cell_rows]
    fits = pd.DataFrame(rows).sort_values(["cell", "rpt"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fits.to_csv(output_dir / "fit_results.csv", index=False)
    return fits


# =============================================================================
# 8. Condition means
# =============================================================================

def condition_means(fits: pd.DataFrame, max_efc: float = 9000.0, points: int = 100):
    """Mean and standard deviation over the cells of each condition on an EFC grid.

    Each cell is interpolated between consecutive supported RPTs; a grid point
    counts where at least three cells contribute.
    """
    out = []
    for condition, group in fits.groupby("condition"):
        grid = np.linspace(group["efc"].min(), min(group["efc"].max(), max_efc), points)
        for column in (*MODES, "traj_qocv_rmse_v", "traj_dv_rmse_v"):
            matrix = []
            for _, cell in group.sort_values("efc").groupby("cell"):
                x, y = cell["efc"].to_numpy(float), cell[column].to_numpy(float)
                good = np.isfinite(x) & np.isfinite(y) & cell["fit_status"].isin(SUPPORTED).to_numpy()
                values = np.full(points, np.nan)
                indices = np.flatnonzero(good)
                for segment in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
                    if segment.size >= 2:
                        inside = (grid >= x[segment].min()) & (grid <= x[segment].max())
                        values[inside] = np.interp(grid[inside], x[segment], y[segment])
                    elif segment.size == 1:
                        values[np.argmin(abs(grid - x[segment[0]]))] = y[segment[0]]
                matrix.append(values)
            matrix = np.asarray(matrix)
            count = np.sum(np.isfinite(matrix), axis=0)
            with warnings.catch_warnings():  # grid points without any cell
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.where(count >= 3, np.nanmean(matrix, axis=0), np.nan)
                sd = np.where(count >= 3, np.nanstd(matrix, axis=0, ddof=1), np.nan)
            out.append(pd.DataFrame({"condition": condition, "quantity": column, "efc": grid,
                                     "mean": mean, "sd": sd, "cells": count}))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True,
                        help="folder degradation_mode_analysis of the dataset")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    settings = Settings(input_dir=args.input,
                        reference_csv=args.input / "electrode_ocp_reference.csv", jobs=args.jobs)
    fits = run(settings, args.output)
    condition_means(fits).to_csv(args.output / "condition_means.csv", index=False)


if __name__ == "__main__":
    main()
