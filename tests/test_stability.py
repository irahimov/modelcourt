import duckdb
import numpy as np
import pandas as pd
import pytest

from modelcourt.stability import (
    DEV_NAME,
    NEW_NAME,
    numeric_edges,
    psi_categorical,
    psi_categorical_sql,
    psi_numeric,
    psi_numeric_sql,
    stability_findings,
)


def _con(dev, new):
    con = duckdb.connect()
    con.register(DEV_NAME, dev)
    con.register(NEW_NAME, new)
    return con


def test_numeric_edges_sorted_unique_and_bounded():
    rng = np.random.default_rng(0)
    edges = numeric_edges(pd.Series(rng.normal(size=1000)), n_bins=10)
    assert len(edges) <= 9
    assert np.all(np.diff(edges) > 0)


def test_identical_samples_give_zero_psi():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=500))
    assert psi_numeric(x, x, numeric_edges(x)) == pytest.approx(0.0, abs=1e-12)


def test_shifted_sample_is_detected():
    rng = np.random.default_rng(2)
    dev = pd.Series(rng.normal(0, 1, 3000))
    new = pd.Series(rng.normal(0.8, 1, 3000))
    assert psi_numeric(dev, new, numeric_edges(dev)) > 0.25


def test_numeric_psi_pandas_equals_sql():
    rng = np.random.default_rng(3)
    dev = pd.DataFrame({"v": rng.gamma(2, 1000, 800)})
    new = pd.DataFrame({"v": rng.gamma(2.5, 1000, 400)})
    edges = numeric_edges(dev["v"])
    p_pd = psi_numeric(dev["v"], new["v"], edges)
    p_sql = psi_numeric_sql(_con(dev, new), "v", edges)
    assert p_sql == pytest.approx(p_pd, abs=1e-9)


def test_categorical_psi_pandas_equals_sql():
    rng = np.random.default_rng(4)
    dev = pd.DataFrame({"c": rng.choice(["a", "b", "c"], 600, p=[0.5, 0.3, 0.2])})
    new = pd.DataFrame({"c": rng.choice(["a", "b", "c"], 300, p=[0.2, 0.3, 0.5])})
    p_pd = psi_categorical(dev["c"], new["c"])
    assert psi_categorical_sql(_con(dev, new), "c") == pytest.approx(p_pd, abs=1e-9)


def test_unseen_category_gives_finite_psi_in_both_versions():
    dev = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 50})
    new = pd.DataFrame({"c": ["a"] * 40 + ["b"] * 40 + ["z"] * 20})
    v = psi_categorical(dev["c"], new["c"])
    assert np.isfinite(v) and v > 0
    assert psi_categorical_sql(_con(dev, new), "c") == pytest.approx(v, abs=1e-9)


def test_empty_bin_is_handled():
    dev = pd.Series(np.arange(100, dtype=float))
    new = pd.Series(np.full(50, 5.0))  # all new mass in one bin
    assert np.isfinite(psi_numeric(dev, new, numeric_edges(dev)))


def test_control_detection_drives_pass_fail():
    base = dict(score_psi=0.05, csi={"x": 0.01}, consistency_diff=0.0, dataset="t")
    ok = {r["test_id"]: r for r in stability_findings(control_psi=0.40, **base)}
    bad = {r["test_id"]: r for r in stability_findings(control_psi=0.10, **base)}
    assert ok["M4.control"]["decision"] == "pass"
    assert bad["M4.control"]["decision"] == "fail"
    assert bad["M4.control"]["severity"] == "critical"


def test_consistency_mismatch_is_critical():
    rows = {r["test_id"]: r for r in stability_findings(
        score_psi=0.05, csi={}, consistency_diff=1e-6, control_psi=0.4, dataset="t")}
    assert rows["M4.consistency"]["decision"] == "fail"
    assert rows["M4.consistency"]["severity"] == "critical"


def test_csi_count_rule():
    csi = {"a": 0.2, "b": 0.3, "c": 0.4}  # three features not pass
    rows = {r["test_id"]: r for r in stability_findings(
        score_psi=0.05, csi=csi, consistency_diff=0.0, control_psi=0.4, dataset="t")}
    assert rows["M4.csi"]["value"] == 3
    assert rows["M4.csi"]["decision"] == "fail"
