import numpy as np
import pandas as pd

from systematic_alpha.pipeline import build_common_sample, build_evaluation_sample


def test_common_sample_filters_security_rows_not_dates_only():
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
            "Ticker": list("ABC") * 2,
            "a": [1.0, 2.0, np.nan, 1.0, 2.0, 3.0],
            "b": [1.0, 2.0, 3.0, 1.0, np.nan, 3.0],
            "forward_return_1d": [0.01] * 6,
        }
    )
    result = build_common_sample(data, ("a", "b"), minimum_cross_section_size=2)
    first = result.loc[result["Date"].eq(pd.Timestamp("2024-01-01")), "Ticker"]
    second = result.loc[result["Date"].eq(pd.Timestamp("2024-01-02")), "Ticker"]
    assert set(first) == {"A", "B"}
    assert set(second) == {"A", "C"}
    assert result[["a", "b"]].notna().all().all()


def test_common_sample_counts_only_eligible_securities():
    data = pd.DataFrame(
        {
            "Date": pd.Timestamp("2024-01-01"),
            "Ticker": list("ABCDE"),
            "a": 1.0,
            "b": 1.0,
            "forward_return_1d": 0.01,
            "eligible": [True, True, False, False, False],
        }
    )
    result = build_common_sample(data, ("a", "b"), minimum_cross_section_size=3)
    assert result.empty


def test_future_label_availability_does_not_change_selection_sample():
    data = pd.DataFrame(
        {
            "Date": pd.Timestamp("2024-01-01"),
            "Ticker": list("ABCD"),
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "forward_return_1d": [0.01, np.nan, 0.02, np.nan],
            "eligible": True,
        }
    )

    selected = build_common_sample(data, ("a", "b"), minimum_cross_section_size=4)

    assert set(selected["Ticker"]) == set("ABCD")


def test_evaluation_sample_filters_labels_without_affecting_selection():
    data = pd.DataFrame(
        {
            "Date": pd.Timestamp("2024-01-01"),
            "Ticker": list("ABCD"),
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "forward_return_1d": [0.01, np.nan, 0.02, np.nan],
            "eligible": True,
        }
    )
    selected = build_common_sample(data, ("a", "b"), minimum_cross_section_size=4)

    evaluated = build_evaluation_sample(selected, minimum_cross_section_size=2)

    assert set(selected["Ticker"]) == set("ABCD")
    assert set(evaluated["Ticker"]) == {"A", "C"}
