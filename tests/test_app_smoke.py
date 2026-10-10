"""End-to-end smoke test: run the actual Streamlit dashboard headlessly and make
sure the refactor of app.py onto quality_core still produces a working page with
the expected metrics and status messages.

Uses Streamlit's official testing harness (AppTest), which executes app.py
exactly like a real render.  Requires the full dependency stack (streamlit,
plotly, scikit-learn, pandas) -- skipped when it is not installed.
"""

import pathlib

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = pathlib.Path(__file__).resolve().parent.parent / "app.py"


def _run():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, f"dashboard raised: {at.exception}"
    return at


class TestDashboardSmoke:
    def test_runs_without_exception(self):
        assert _run()

    def test_key_metrics_present(self):
        at = _run()
        labels = [m.label for m in at.metric]
        for want in [
            "Units Inspected",
            "Nonconforming",
            "First Pass Yield",
            "Process Mean",
            "Std. Deviation",
            "LSL",
            "USL",
            "Cp",
            "Cpk",
            "Pp (long-term)",
            "Ppk (long-term)",
            "Cpm (target-aware)",
            "Tool Life Used",
            "Predicted Bore Diameter",
            "ML Predicted Bore Diameter",
            "Mean Absolute Error",
        ]:
            assert want in labels, f"missing metric {want}; have {labels}"

    def test_capability_assessment_status(self):
        at = _run()
        texts = [getattr(e, "value", "") for e in at.success]
        assert any("Demonstration Status: Capable" in t for t in texts), texts

    def test_control_chart_in_control(self):
        at = _run()
        ok = [getattr(e, "value", "") for e in at.success]
        assert any("within the 3-sigma limits" in t for t in ok), ok

    def test_defect_rate_value(self):
        at = _run()
        labels = {m.label: m.value for m in at.metric}
        assert labels["Defect Rate"] == "11.7%"
        assert labels["First Pass Yield"] == "88.3%"

    def test_ml_decision_action_rendered(self):
        at = _run()
        # Default simulator state (400 cycles, mid-scale inputs) should give the
        # "Continue Production" decision.
        ok = [getattr(e, "value", "") for e in at.success]
        assert any("ACTION: Continue Production" in t for t in ok), ok

    def test_charts_render(self):
        at = _run()
        # The dashboard renders several Plotly figures (Pareto, SPC, trend,
        # tool-wear, feature-importance and validation charts).
        charts = list(at.get("plotly_chart"))
        assert len(charts) >= 5, f"expected chart figures, found {len(charts)}"