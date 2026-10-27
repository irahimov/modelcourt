import pandas as pd

from modelcourt.report import decide, not_applicable_row


def _f(decision, severity="none"):
    return {"decision": decision, "severity": severity}


def test_all_pass_is_pass():
    assert decide(pd.DataFrame([_f("pass")] * 5)) == "pass"


def test_critical_fail_rejects_even_with_passes():
    df = pd.DataFrame([_f("pass"), _f("fail", "critical"), _f("pass")])
    assert decide(df) == "rejected"


def test_high_fail_or_two_conditionals_is_conditional():
    assert decide(pd.DataFrame([_f("pass"), _f("fail", "high")])) == "conditional"
    assert decide(pd.DataFrame([_f("conditional", "low"), _f("conditional", "low")])) == "conditional"


def test_single_conditional_is_still_pass():
    assert decide(pd.DataFrame([_f("pass"), _f("conditional", "low")])) == "pass"


def test_not_applicable_is_not_counted():
    df = pd.DataFrame([_f("pass"), not_applicable_row("M5.backtest", "t", "no dates")])
    assert decide(df) == "pass"
