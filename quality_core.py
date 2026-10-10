"""quality_core -- pure quality-engineering calculations for the dashboard.

The statistics behind the Manufacturing Quality Analytics Dashboard used to live
inline in ``app.py``, interleaved with Streamlit widgets.  That made them
impossible to unit-test, duplicated the tool-wear model in two places, and meant
every number on screen was only verifiable by eyeballing a chart.

This module lifts the calculations into small, deterministic, side-effect-free
functions.  The statistical helpers use the Python standard library only; the
single synthetic-dataset factory is the one function that reaches for NumPy
(imported lazily) so the rest of the module stays usable anywhere.

Formulas follow the standard quality-engineering definitions:

* Defect rate / first-pass yield      -- simple count ratios.
* Pareto analysis                     -- descending counts + cumulative share.
* SPC control limits                  -- mean +/- k * sigma (k defaults to 3).
* Cp / Cpk and Pp / Ppk               -- short-term (sample) and long-term
  (overall) process capability/performance indices.
* Cpm                                  -- Taguchi target-capability index, which
  penalises a process for being off-target rather than only off-centre.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "FEATURE_NAMES",
    "CAPABLE_THRESHOLD",
    "MARGINAL_THRESHOLD",
    "DEMO_DEFECT_TYPES",
    "DEMO_DEFECT_COUNTS",
    "DEMO_TOTAL_INSPECTED",
    "DEMO_BORE_MEASUREMENTS",
    "DEFAULT_NOMINAL_BORE",
    "DEFAULT_LSL",
    "DEFAULT_USL",
    "DEFAULT_TARGET",
    "RECOMMENDED_TOOL_LIFE",
    "defect_rate",
    "first_pass_yield",
    "pareto_analysis",
    "spc_control_limits",
    "capability_indices",
    "performance_indices",
    "target_capability_index",
    "capability_summary",
    "capability_status",
    "resolve_lot_limits",
    "simulate_bore_diameter",
    "tool_life_percent",
    "tool_life_status",
    "dimensional_status",
    "ml_quality_assessment",
    "ml_decision",
    "generate_synthetic_dataset",
]

# --------------------------------------------------------------------------- #
# Demo dataset constants (shared by the dashboard and the test-suite so both
# sides agree on exactly one set of "known" numbers).
# --------------------------------------------------------------------------- #

FEATURE_NAMES: List[str] = [
    "Tool Cycles",
    "Spindle Speed",
    "Feed Rate",
    "Vibration",
    "Temperature",
]

CAPABLE_THRESHOLD = 1.33
MARGINAL_THRESHOLD = 1.00

DEMO_DEFECT_TYPES: List[str] = [
    "Bore Diameter",
    "Surface Finish",
    "Runout",
    "Burr",
    "Scratch",
]
DEMO_DEFECT_COUNTS: List[int] = [16, 5, 3, 2, 2]
DEMO_TOTAL_INSPECTED = 240

DEMO_BORE_MEASUREMENTS: List[float] = [
    25.01, 24.98, 25.03, 25.00, 24.97,
    25.02, 25.04, 24.99, 25.01, 24.96,
    25.00, 25.03, 24.98, 25.02, 25.01,
    24.99, 25.04, 25.00, 24.97, 25.02,
]

DEFAULT_NOMINAL_BORE = 25.000
DEFAULT_LSL = 24.900
DEFAULT_USL = 25.100
DEFAULT_TARGET = 25.000
RECOMMENDED_TOOL_LIFE = 800


# --------------------------------------------------------------------------- #
# Simple count ratios
# --------------------------------------------------------------------------- #

def defect_rate(defects: float, inspected: float) -> float:
    """Return the defect rate as a percentage.

    Raises ``ValueError`` for a non-positive inspection count instead of letting
    a bare ``ZeroDivisionError`` escape from deep inside the chart code.
    """
    if inspected <= 0:
        raise ValueError("inspected count must be positive")
    if defects < 0:
        raise ValueError("defect count cannot be negative")
    return (defects / inspected) * 100.0


def first_pass_yield(defects: float, inspected: float) -> float:
    """Return the first-pass yield as a percentage."""
    if inspected <= 0:
        raise ValueError("inspected count must be positive")
    if defects < 0:
        raise ValueError("defect count cannot be negative")
    return ((inspected - defects) / inspected) * 100.0


# --------------------------------------------------------------------------- #
# Pareto analysis
# --------------------------------------------------------------------------- #

def pareto_analysis(
    labels: Sequence[str], counts: Sequence[float]
) -> Dict[str, Any]:
    """Sort defect categories descending and add a cumulative share column.

    Returns a dict with:

    ``rows``            -- list of ``{"label", "count", "cumulative_pct"}``
                           ordered from largest to smallest count.  Ties keep
                           their original order (deterministic, unlike the
                           previous ``DataFrame.sort_values`` call).
    ``total``           -- sum of all counts.
    ``vital_few``       -- labels making up the classic ~80%% of defects.
    """
    if len(labels) != len(counts):
        raise ValueError("labels and counts must be the same length")
    if not labels:
        raise ValueError("at least one category is required")
    if any(c < 0 for c in counts):
        raise ValueError("defect counts cannot be negative")

    total = float(sum(counts))
    # Stable sort: primary key descending count, tie-break on original position.
    order = sorted(range(len(counts)), key=lambda i: (-counts[i], i))

    rows: List[Dict[str, Any]] = []
    cumulative = 0.0
    vital_few: List[str] = []
    reached_80 = False
    for i in order:
        cumulative += float(counts[i])
        pct = (cumulative / total * 100.0) if total else 0.0
        rows.append(
            {"label": labels[i], "count": counts[i], "cumulative_pct": pct}
        )
        if not reached_80:
            vital_few.append(labels[i])
            if pct >= 80.0:
                reached_80 = True

    return {"rows": rows, "total": total, "vital_few": vital_few}


# --------------------------------------------------------------------------- #
# Statistical process control
# --------------------------------------------------------------------------- #

def spc_control_limits(
    measurements: Sequence[float],
    sigma_multiplier: float = 3.0,
    ddof: int = 1,
) -> Dict[str, Any]:
    """Return X-bar control limits for a series of individual measurements.

    ``ddof=1`` (the default) matches the sample standard deviation the dashboard
    has always used.  ``out_of_control`` lists the indices whose measurement
    falls outside the computed limits so the caller can highlight them -- the
    original chart drew the limits but never flagged a violation.
    """
    values = [float(v) for v in measurements]
    if len(values) < 2:
        raise ValueError("at least two measurements are required")
    if sigma_multiplier <= 0:
        raise ValueError("sigma_multiplier must be positive")

    mean = statistics.fmean(values)
    std = statistics.stdev(values) if ddof == 1 else statistics.pstdev(values)
    ucl = mean + sigma_multiplier * std
    lcl = mean - sigma_multiplier * std
    out_of_control = [i for i, v in enumerate(values) if v > ucl or v < lcl]

    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ucl": ucl,
        "lcl": lcl,
        "out_of_control": out_of_control,
        "in_control": not out_of_control,
    }


# --------------------------------------------------------------------------- #
# Process capability / performance
# --------------------------------------------------------------------------- #

def capability_indices(mean: float, std: float, lsl: float, usl: float) -> Dict[str, float]:
    """Return Cp / Cpu / Cpl / Cpk for a stable process.

    ``Cp`` measures potential capability (spread only); ``Cpk`` measures actual
    capability (spread and centring).  A zero standard deviation or an inverted
    specification window raise ``ValueError`` rather than dividing by zero.
    """
    if usl <= lsl:
        raise ValueError("USL must be greater than LSL")
    if std <= 0:
        raise ValueError("standard deviation must be positive")
    cp = (usl - lsl) / (6.0 * std)
    cpu = (usl - mean) / (3.0 * std)
    cpl = (mean - lsl) / (3.0 * std)
    return {"cp": cp, "cpu": cpu, "cpl": cpl, "cpk": min(cpu, cpl)}


def performance_indices(
    mean: float, std_overall: float, lsl: float, usl: float
) -> Dict[str, float]:
    """Return Pp / Ppu / Ppl / Ppk using the long-term (overall) sigma.

    Identical arithmetic to :func:`capability_indices` but named for the
    performance family; callers supply the overall standard deviation.
    """
    perf = capability_indices(mean, std_overall, lsl, usl)
    return {
        "pp": perf["cp"],
        "ppu": perf["cpu"],
        "ppl": perf["cpl"],
        "ppk": perf["cpk"],
    }


def target_capability_index(
    mean: float, std: float, lsl: float, usl: float, target: float
) -> float:
    """Return the Taguchi capability index ``Cpm``.

    ``Cpm = (USL - LSL) / (6 * sqrt(sigma^2 + (mean - target)^2))``.  Unlike Cpk
    it penalises the process for drifting away from the target value, which is
    what makes the dashboard's "Target Diameter" input meaningful.
    """
    if usl <= lsl:
        raise ValueError("USL must be greater than LSL")
    if std < 0:
        raise ValueError("standard deviation cannot be negative")
    spread = math.sqrt(std * std + (mean - target) ** 2)
    if spread <= 0:
        return math.inf
    return (usl - lsl) / (6.0 * spread)


def capability_summary(
    measurements: Sequence[float],
    lsl: float,
    usl: float,
    target: Optional[float] = None,
    sigma_multiplier: float = 3.0,
) -> Dict[str, Any]:
    """Compute every capability metric the dashboard displays from raw data.

    Combines the short-term indices (Cp/Cpk from the sample sigma, ``ddof=1``)
    with the long-term performance indices (Pp/Ppk from the overall sigma,
    ``ddof=0``) and, when a ``target`` is supplied, the target-aware ``Cpm``.
    """
    values = [float(v) for v in measurements]
    if len(values) < 2:
        raise ValueError("at least two measurements are required")

    mean = statistics.fmean(values)
    std_sample = statistics.stdev(values)
    std_overall = statistics.pstdev(values)

    cap = capability_indices(mean, std_sample, lsl, usl)
    perf = performance_indices(mean, std_overall, lsl, usl)

    summary: Dict[str, Any] = {
        "mean": mean,
        "std_sample": std_sample,
        "std_overall": std_overall,
        "cp": cap["cp"],
        "cpu": cap["cpu"],
        "cpl": cap["cpl"],
        "cpk": cap["cpk"],
        "pp": perf["pp"],
        "ppk": perf["ppk"],
        "status": capability_status(cap["cpk"]),
        "limits": spc_control_limits(values, sigma_multiplier),
    }
    if target is not None:
        summary["cpm"] = target_capability_index(mean, std_sample, lsl, usl, target)
        summary["target"] = target
    return summary


def capability_status(cpk: float) -> str:
    """Classify a Cpk value using the conventional 1.33 / 1.00 decision bands."""
    if cpk >= CAPABLE_THRESHOLD:
        return "Capable"
    if cpk >= MARGINAL_THRESHOLD:
        return "Marginal"
    return "Not Capable"


def resolve_lot_limits(target: float, tolerance: float) -> Dict[str, float]:
    """Return symmetric specification limits for a target and total tolerance.

    Handy for the "what-if" controls: a 25.000 mm target with a 0.200 mm
    tolerance window becomes LSL 24.900 / USL 25.100.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    half = tolerance / 2.0
    return {"lsl": target - half, "usl": target + half}


# --------------------------------------------------------------------------- #
# Tool-wear model
# --------------------------------------------------------------------------- #

def simulate_bore_diameter(
    tool_cycles: float,
    nominal: float = DEFAULT_NOMINAL_BORE,
    wear_start: float = 600.0,
    moderate_limit: float = 800.0,
    drift_first_stage: float = 0.08,
    drift_second_stage: float = 0.07,
) -> float:
    """Piecewise bore-diameter drift as a function of cutting-tool usage.

    * below ``wear_start`` the bore sits exactly on nominal (no measurable wear),
    * between ``wear_start`` and ``moderate_limit`` it drifts by
      ``drift_first_stage`` over that span,
    * past ``moderate_limit`` it keeps drifting at ``drift_second_stage`` per
      ``(moderate_limit - wear_start)`` cycles.

    The model is continuous at both breakpoints.  Previously this exact branch
    lived twice in ``app.py`` (once for the slider, once for the chart loop),
    which is exactly the kind of duplication that lets the two copies drift
    apart.
    """
    if moderate_limit <= wear_start:
        raise ValueError("moderate_limit must exceed wear_start")
    span = moderate_limit - wear_start
    if tool_cycles < wear_start:
        return float(nominal)
    if tool_cycles < moderate_limit:
        return float(nominal + ((tool_cycles - wear_start) / span) * drift_first_stage)
    base = nominal + drift_first_stage
    return float(base + ((tool_cycles - moderate_limit) / span) * drift_second_stage)


def tool_life_percent(cycles: float, recommended: float = RECOMMENDED_TOOL_LIFE) -> float:
    """Return tool life consumed as a percentage of the recommended interval."""
    if recommended <= 0:
        raise ValueError("recommended tool life must be positive")
    return (cycles / recommended) * 100.0


def tool_life_status(
    cycles: float, recommended: float = RECOMMENDED_TOOL_LIFE
) -> Dict[str, str]:
    """Classify tool usage into normal / monitor / replace bands."""
    pct = tool_life_percent(cycles, recommended)
    if pct < 75:
        return {"level": "ok", "label": "Normal \u2014 Continue Monitoring"}
    if pct < 100:
        return {"level": "warn", "label": "Monitor \u2014 Approaching Preventive Replacement"}
    return {
        "level": "alert",
        "label": "Replacement Recommended \u2014 Tool-Life Limit Reached",
    }


def dimensional_status(value: float, lsl: float, usl: float) -> str:
    """Return ``"within"`` or ``"out"`` for a bore diameter against spec limits."""
    return "within" if lsl <= value <= usl else "out"


# --------------------------------------------------------------------------- #
# ML decision support
# --------------------------------------------------------------------------- #

def ml_quality_assessment(
    predicted: float,
    lsl: float = DEFAULT_LSL,
    usl: float = DEFAULT_USL,
    near_upper: float = 25.080,
) -> str:
    """Classify a model-predicted bore diameter into a risk band.

    One of ``out_high``, ``out_low``, ``near_upper`` or ``ok``.
    """
    if predicted > usl:
        return "out_high"
    if predicted < lsl:
        return "out_low"
    if predicted >= near_upper:
        return "near_upper"
    return "ok"


def ml_decision(
    tool_cycles: float,
    predicted: float,
    lsl: float = DEFAULT_LSL,
    usl: float = DEFAULT_USL,
    recommended: float = RECOMMENDED_TOOL_LIFE,
    near_upper: float = 25.080,
    monitor_threshold: float = 75.0,
) -> str:
    """Return the recommended action code for the decision-support panel.

    Ordered by severity: an out-of-spec prediction always wins, then a spent
    tool, then an approach to the upper limit, then the monitoring band.
    """
    if predicted > usl or predicted < lsl:
        return "stop_and_inspect"
    if tool_cycles >= recommended:
        return "replace_tool"
    if predicted >= near_upper:
        return "prepare_tool_change"
    if tool_life_percent(tool_cycles, recommended) >= monitor_threshold:
        return "increase_monitoring"
    return "continue_production"


# --------------------------------------------------------------------------- #
# Synthetic ML dataset
# --------------------------------------------------------------------------- #

def generate_synthetic_dataset(
    sample_size: int = 500, seed: int = 42
) -> Dict[str, Any]:
    """Build the reproducible synthetic CNC dataset used to train the model.

    Returns a dict with one NumPy array per :data:`FEATURE_NAMES` entry plus
    ``"Bore Diameter"``.  The generator uses a private ``RandomState`` rather
    than calling ``numpy.random.seed`` -- the previous code reseeded NumPy's
    *global* generator as a side effect of rendering the page, which silently
    changed any randomness elsewhere in the process.  Seeding a local generator
    reproduces the exact same numbers while leaving global state untouched.

    NumPy is imported here (and only here) so the rest of the module needs no
    third-party dependencies.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    import numpy as np  # local import: keeps the statistical helpers dependency-free

    rng = np.random.RandomState(seed)
    tool_cycles = rng.randint(0, 1001, sample_size)
    spindle_speed = rng.randint(1800, 3201, sample_size)
    feed_rate = rng.uniform(80, 180, sample_size)
    vibration = rng.uniform(0.5, 4.0, sample_size)
    temperature = rng.uniform(20, 45, sample_size)

    tool_wear_effect = np.where(
        tool_cycles < 600, 0, (tool_cycles - 600) * 0.00035
    )
    bore_diameter = (
        25.000
        + tool_wear_effect
        + (vibration - 2.0) * 0.008
        + (temperature - 30.0) * 0.001
        + (feed_rate - 130.0) * 0.00015
        + rng.normal(0, 0.008, sample_size)
    )

    return {
        "Tool Cycles": tool_cycles,
        "Spindle Speed": spindle_speed,
        "Feed Rate": feed_rate,
        "Vibration": vibration,
        "Temperature": temperature,
        "Bore Diameter": bore_diameter,
    }
