"""Unit tests for quality_core -- the pure, dependency-light calculation module
extracted from app.py.  These mirror the exact numbers the dashboard renders so
a change anywhere is caught by the test-suite, not by a surprised maintainer."""

import math

import pytest

from quality_core import (
    CAPABLE_THRESHOLD,
    DEMO_BORE_MEASUREMENTS,
    DEMO_DEFECT_COUNTS,
    DEMO_DEFECT_TYPES,
    DEMO_TOTAL_INSPECTED,
    DEFAULT_LSL,
    DEFAULT_TARGET,
    DEFAULT_USL,
    FEATURE_NAMES,
    MARGINAL_THRESHOLD,
    RECOMMENDED_TOOL_LIFE,
    capability_indices,
    capability_status,
    capability_summary,
    defect_rate,
    dimensional_status,
    first_pass_yield,
    generate_synthetic_dataset,
    ml_decision,
    ml_quality_assessment,
    pareto_analysis,
    performance_indices,
    resolve_lot_limits,
    simulate_bore_diameter,
    spc_control_limits,
    target_capability_index,
    tool_life_percent,
    tool_life_status,
)


# --------------------------------------------------------------------------- #
# defect_rate / first_pass_yield
# --------------------------------------------------------------------------- #

class TestDefectMetrics:
    def test_demo_defect_rate(self):
        assert defect_rate(28, DEMO_TOTAL_INSPECTED) == pytest.approx(11.6667, abs=1e-3)

    def test_demo_first_pass_yield(self):
        assert first_pass_yield(28, DEMO_TOTAL_INSPECTED) == pytest.approx(88.3333, abs=1e-3)

    def test_rate_and_yield_complement(self):
        e = DEMO_TOTAL_INSPECTED
        d = 28
        assert defect_rate(d, e) + first_pass_yield(d, e) == pytest.approx(100.0)

    def test_zero_defects(self):
        assert defect_rate(0, 100) == 0.0
        assert first_pass_yield(0, 100) == 100.0

    def test_all_defective(self):
        assert defect_rate(100, 100) == 100.0
        assert first_pass_yield(100, 100) == 0.0

    @pytest.mark.parametrize("inspected", [0, -5])
    def test_non_positive_inspected_raises(self, inspected):
        with pytest.raises(ValueError):
            defect_rate(1, inspected)
        with pytest.raises(ValueError):
            first_pass_yield(1, inspected)

    @pytest.mark.parametrize("defects", [-1, -10])
    def test_negative_defects_raise(self, defects):
        with pytest.raises(ValueError):
            defect_rate(defects, 100)
        with pytest.raises(ValueError):
            first_pass_yield(defects, 100)


# --------------------------------------------------------------------------- #
# Pareto
# --------------------------------------------------------------------------- #

class TestPareto:
    def test_demo_pareto_rows_ordered(self):
        p = pareto_analysis(DEMO_DEFECT_TYPES, DEMO_DEFECT_COUNTS)
        labels = [r["label"] for r in p["rows"]]
        assert labels == ["Bore Diameter", "Surface Finish", "Runout", "Burr", "Scratch"]
        counts = [r["count"] for r in p["rows"]]
        assert counts == sorted(counts, reverse=True)

    def test_demo_pareto_cumulative(self):
        p = pareto_analysis(DEMO_DEFECT_TYPES, DEMO_DEFECT_COUNTS)
        assert [round(r["cumulative_pct"], 1) for r in p["rows"]] == [
            57.1, 75.0, 85.7, 92.9, 100.0,
        ]

    def test_demo_pareto_total_and_vital_few(self):
        p = pareto_analysis(DEMO_DEFECT_TYPES, DEMO_DEFECT_COUNTS)
        assert p["total"] == 28.0
        assert p["vital_few"] == ["Bore Diameter", "Surface Finish", "Runout"]

    def test_last_row_always_100(self):
        p = pareto_analysis(["a", "b", "c"], [1, 2, 3])
        assert p["rows"][-1]["cumulative_pct"] == pytest.approx(100.0)

    def test_single_category(self):
        p = pareto_analysis(["only"], [7])
        assert p["total"] == 7.0
        assert p["vital_few"] == ["only"]
        assert p["rows"][0]["cumulative_pct"] == pytest.approx(100.0)

    def test_tie_keeps_original_order(self):
        p = pareto_analysis(["x", "y", "z"], [5, 5, 5])
        assert [r["label"] for r in p["rows"]] == ["x", "y", "z"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            pareto_analysis(["a", "b"], [1])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pareto_analysis([], [])

    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            pareto_analysis(["a"], [-1])

    def test_80_pct_cut_includes_itself(self):
        p = pareto_analysis(["a", "b"], [90, 10])
        assert p["vital_few"] == ["a"]
        # "a" alone reaches 90% >= 80, so it is the vital few
        assert p["rows"][0]["cumulative_pct"] == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# SPC
# --------------------------------------------------------------------------- #

class TestSpc:
    def test_demo_in_control(self):
        s = spc_control_limits(DEMO_BORE_MEASUREMENTS)
        assert s["n"] == 20
        assert s["in_control"] is True
        assert s["out_of_control"] == []
        assert s["mean"] == pytest.approx(25.0035, abs=1e-4)
        assert s["std"] == pytest.approx(0.02368, abs=1e-4)
        assert s["ucl"] == pytest.approx(25.07454, abs=1e-4)
        assert s["lcl"] == pytest.approx(24.93246, abs=1e-4)

    def test_out_of_control_flagged(self):
        s = spc_control_limits([5.0] * 19 + [2000.0])
        assert s["out_of_control"] == [19]
        assert s["in_control"] is False

    def test_ddof_changes_limits(self):
        a = spc_control_limits([1, 2, 3, 4], ddof=0)
        b = spc_control_limits([1, 2, 3, 4], ddof=1)
        assert a["std"] != b["std"]

    @pytest.mark.parametrize("values", [[], [1.0]])
    def test_too_few_measurements_raises(self, values):
        with pytest.raises(ValueError):
            spc_control_limits(values)

    @pytest.mark.parametrize("mult", [0, -1])
    def test_non_positive_sigma_raises(self, mult):
        with pytest.raises(ValueError):
            spc_control_limits([1, 2, 3], sigma_multiplier=mult)

    def test_constant_measurements_have_zero_uci_band(self):
        s = spc_control_limits([5.0, 5.0, 5.0])
        assert s["std"] == pytest.approx(0.0)
        assert s["ucl"] == pytest.approx(5.0)
        assert s["lcl"] == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# Capability / performance / target indices
# --------------------------------------------------------------------------- #

class TestCapability:
    def test_symmetric_centered_cp_equals_cpk(self):
        c = capability_indices(mean=10.0, std=1.0, lsl=7.0, usl=13.0)
        assert c["cp"] == pytest.approx(1.0)
        assert c["cpk"] == pytest.approx(1.0)

    def test_cp_standard_formula(self):
        c = capability_indices(mean=0.0, std=0.5, lsl=-1.5, usl=1.5)
        # (usl-lsl)/(6*std) = 3.0/(3.0) = 1.0
        assert c["cp"] == pytest.approx(1.0)

    def test_cpk_penalizes_off_centre(self):
        c = capability_indices(mean=12.0, std=1.0, lsl=7.0, usl=13.0)
        cpu = (13 - 12) / 3.0
        cpl = (12 - 7) / 3.0
        assert c["cpk"] == pytest.approx(min(cpu, cpl))

    def test_inverted_spec_raises(self):
        with pytest.raises(ValueError):
            capability_indices(mean=0, std=1, lsl=5, usl=5)

    def test_zero_std_raises(self):
        with pytest.raises(ValueError):
            capability_indices(mean=0, std=0, lsl=-1, usl=1)


class TestPerformance:
    def test_pp_standard_formula(self):
        p = performance_indices(mean=0.0, std_overall=0.5, lsl=-1.5, usl=1.5)
        assert p["pp"] == pytest.approx(1.0)

    def test_ppk_off_centre(self):
        p = performance_indices(mean=1.0, std_overall=0.5, lsl=0.0, usl=2.0)
        assert p["ppu"] == pytest.approx((2 - 1) / 1.5)
        assert p["ppl"] == pytest.approx((1 - 0) / 1.5)
        assert p["ppk"] == pytest.approx(min(p["ppu"], p["ppl"]))


class TestTargetCapability:
    def test_cpm_on_target_reduces_to_cp(self):
        cpm = target_capability_index(mean=10.0, std=1.0, lsl=7.0, usl=13.0, target=10.0)
        cp = capability_indices(mean=10.0, std=1.0, lsl=7.0, usl=13.0)["cp"]
        assert cpm == pytest.approx(cp)

    def test_cpm_penalizes_off_target(self):
        on = target_capability_index(mean=10.0, std=1.0, lsl=7.0, usl=13.0, target=10.0)
        off = target_capability_index(mean=11.0, std=1.0, lsl=7.0, usl=13.0, target=10.0)
        assert off < on

    def test_identical_std_and_pure_shift(self):
        # Without target term, mean 10 vs 11 with same std give same cpk; Cpm differs.
        a = target_capability_index(10.0, 1.0, 7.0, 13.0, 10.0)
        b = target_capability_index(11.0, 1.0, 7.0, 13.0, 10.0)
        assert a != b

    def test_inverted_spec_raises(self):
        with pytest.raises(ValueError):
            target_capability_index(0, 1, 5, 5, 0)

    def test_zero_spread_infinite(self):
        cpm = target_capability_index(10.0, 0.0, 7.0, 13.0, 10.0)
        assert cpm == math.inf


class TestCapabilityStatus:
    @pytest.mark.parametrize(
        "cpk,expected",
        [
            (2.0, "Capable"),
            (CAPABLE_THRESHOLD, "Capable"),
            (1.10, "Marginal"),
            (MARGINAL_THRESHOLD, "Marginal"),
            (0.5, "Not Capable"),
            (0.99, "Not Capable"),
        ],
    )
    def test_bands(self, cpk, expected):
        assert capability_status(cpk) == expected


class TestCapabilitySummary:
    def test_demo_summary_values(self):
        c = capability_summary(DEMO_BORE_MEASUREMENTS, DEFAULT_LSL, DEFAULT_USL, target=DEFAULT_TARGET)
        assert c["cp"] == pytest.approx(1.4076, abs=1e-3)
        assert c["cpk"] == pytest.approx(1.3583, abs=1e-3)
        assert c["pp"] == pytest.approx(1.4442, abs=1e-3)
        assert c["ppk"] == pytest.approx(1.3936, abs=1e-3)
        assert c["cpm"] == pytest.approx(1.3925, abs=1e-3)
        assert c["status"] == "Capable"

    def test_summary_includes_limits(self):
        c = capability_summary(DEMO_BORE_MEASUREMENTS, DEFAULT_LSL, DEFAULT_USL)
        assert "limits" in c
        assert c["limits"]["mean"] == c["mean"]

    def test_summary_without_target_omits_cpm(self):
        c = capability_summary(DEMO_BORE_MEASUREMENTS, DEFAULT_LSL, DEFAULT_USL)
        assert "cpm" not in c

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            capability_summary([25.0], DEFAULT_LSL, DEFAULT_USL)


class TestResolveLotLimits:
    def test_symmetric_limits(self):
        lim = resolve_lot_limits(target=25.0, tolerance=0.2)
        assert lim["lsl"] == pytest.approx(24.9)
        assert lim["usl"] == pytest.approx(25.1)

    def test_non_positive_tolerance_raises(self):
        with pytest.raises(ValueError):
            resolve_lot_limits(25.0, 0)


# --------------------------------------------------------------------------- #
# Tool-wear model
# --------------------------------------------------------------------------- #

class TestSimulateBoreDiameter:
    @pytest.mark.parametrize("cycles", [0, 100, 599])
    def test_below_wear_start_is_nominal(self, cycles):
        assert simulate_bore_diameter(cycles) == pytest.approx(25.0)

    def test_at_wear_start_is_nominal(self):
        assert simulate_bore_diameter(600) == pytest.approx(25.0)

    def test_between_wear_start_and_moderate(self):
        # At the midpoint (700) drift is half of 0.08 via first-stage slope.
        assert simulate_bore_diameter(700) == pytest.approx(25.04, abs=1e-9)

    @pytest.mark.parametrize(
        "cycles,expected",
        [(600, 25.0), (799, 25.0796), (800, 25.08), (1000, 25.15)],
    )
    def test_breakpoints_and_tail(self, cycles, expected):
        assert simulate_bore_diameter(cycles) == pytest.approx(expected, abs=1e-9)

    def test_continuity_at_both_breakpoints(self):
        left = simulate_bore_diameter(600 - 1e-9)
        at = simulate_bore_diameter(600)
        assert abs(left - at) < 1e-9
        left2 = simulate_bore_diameter(800 - 1e-9)
        at2 = simulate_bore_diameter(800)
        assert abs(left2 - at2) < 1e-9

    def test_custom_limits(self):
        v = simulate_bore_diameter(100, nominal=20.0, wear_start=50, moderate_limit=100,
                                   drift_first_stage=0.1, drift_second_stage=0.05)
        # 100 == moderate_limit -> base = nominal + 0.1
        assert v == pytest.approx(20.1, abs=1e-9)

    def test_inverted_limits_raise(self):
        with pytest.raises(ValueError):
            simulate_bore_diameter(100, wear_start=100, moderate_limit=50)


class TestToolLife:
    def test_percent(self):
        assert tool_life_percent(400) == pytest.approx(50.0)
        assert tool_life_percent(RECOMMENDED_TOOL_LIFE) == pytest.approx(100.0)
        assert tool_life_percent(0) == pytest.approx(0.0)

    def test_percent_non_positive_recommended_raises(self):
        with pytest.raises(ValueError):
            tool_life_percent(400, recommended=0)

    @pytest.mark.parametrize(
        "cycles,level",
        # Bands are on tool-life percent (cycles / 800 * 100): <75 ok,
        # 75-99 warn, >=100 alert -> cycles 0-599 / 600-799 / 800+.
        [(0, "ok"), (400, "ok"), (599, "ok"), (600, "warn"),
         (799, "warn"), (800, "alert"), (1000, "alert")],
    )
    def test_bands(self, cycles, level):
        assert tool_life_status(cycles)["level"] == level

    def test_labels_present(self):
        assert tool_life_status(0)["label"]
        assert tool_life_status(90)["label"]
        assert tool_life_status(800)["label"]


class TestDimensionalStatus:
    @pytest.mark.parametrize(
        "value,expected",
        [(25.05, "within"), (24.90, "within"), (25.10, "within"),
         (24.89, "out"), (25.11, "out")],
    )
    def test_bounds(self, value, expected):
        assert dimensional_status(value, DEFAULT_LSL, DEFAULT_USL) == expected


# --------------------------------------------------------------------------- #
# ML assessment / decision
# --------------------------------------------------------------------------- #

class TestMlQualityAssessment:
    @pytest.mark.parametrize(
        "value,expected",
        [(25.15, "out_high"), (24.80, "out_low"), (25.09, "near_upper"),
         (25.08, "near_upper"), (25.02, "ok"), (24.95, "ok")],
    )
    def test_bands(self, value, expected):
        assert ml_quality_assessment(value) == expected


class TestMlDecision:
    def test_out_of_spec_always_wins(self):
        # even a completely fresh tool must stop if the prediction is out of spec
        assert ml_decision(0, 25.20) == "stop_and_inspect"
        assert ml_decision(0, 24.70) == "stop_and_inspect"

    def test_spent_tool_replaces(self):
        assert ml_decision(800, 25.01) == "replace_tool"

    def test_approach_upper_prepares_change(self):
        assert ml_decision(400, 25.09) == "prepare_tool_change"

    def test_monitoring_band(self):
        # 600/800 = 75% -> increase monitoring
        assert ml_decision(600, 25.04) == "increase_monitoring"

    def test_normal_continues(self):
        assert ml_decision(400, 25.01) == "continue_production"

    def test_priority_order_out_of_spec_beats_spent_tool(self):
        assert ml_decision(800, 25.30) == "stop_and_inspect"


# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #

class TestSyntheticDataset:
    def test_shape_and_features(self):
        d = generate_synthetic_dataset(sample_size=500, seed=42)
        for name in FEATURE_NAMES + ["Bore Diameter"]:
            assert name in d
            assert len(d[name]) == 500

    def test_reproducible_with_same_seed(self):
        a = generate_synthetic_dataset(seed=42)
        b = generate_synthetic_dataset(seed=42)
        assert (a["Bore Diameter"] == b["Bore Diameter"]).all()

    def test_different_seed_changes_data(self):
        a = generate_synthetic_dataset(seed=42)
        b = generate_synthetic_dataset(seed=99)
        assert not (a["Bore Diameter"] == b["Bore Diameter"]).all()

    def test_feature_ranges(self):
        d = generate_synthetic_dataset(seed=42)
        assert d["Tool Cycles"].min() >= 0
        assert d["Tool Cycles"].max() <= 1000
        assert d["Spindle Speed"].min() >= 1800
        assert d["Spindle Speed"].max() <= 3200

    def test_bore_mean_close_to_nominal(self):
        d = generate_synthetic_dataset(seed=42)
        import numpy as np
        assert float(np.mean(d["Bore Diameter"])) == pytest.approx(25.0326, abs=1e-3)

    def test_wear_effect_is_monotonic_in_cycles(self):
        d = generate_synthetic_dataset(seed=42)
        import numpy as np
        high = d["Bore Diameter"][d["Tool Cycles"] >= 800]
        low = d["Bore Diameter"][d["Tool Cycles"] < 600]
        assert float(np.mean(high)) > float(np.mean(low))

    def test_spindle_speed_zero_effect(self):
        # Spindle speed is a filler feature: bore diameter does not depend on it.
        # A linear fit coefficient should be ~0 (far weaker than Tool Cycles).
        d = generate_synthetic_dataset(seed=42)
        from numpy import corrcoef
        c = corrcoef(d["Spindle Speed"], d["Bore Diameter"])[0, 1]
        assert abs(float(c)) < 0.1

    def test_non_positive_sample_size_raises(self):
        with pytest.raises(ValueError):
            generate_synthetic_dataset(sample_size=0)


def test_numpy_available():
    # Dataset tests require numpy; the statistical helpers above do not.
    import numpy  # noqa: F401  (pytest will fail loudly here if it is missing)