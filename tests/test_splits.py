import pandas as pd

from systematic_alpha.config import Period
from systematic_alpha.validation import select_period


def test_period_boundaries_are_inclusive_and_isolated():
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01"]
            )
        }
    )
    selected = select_period(data, Period("2022-01-01", "2022-12-31"))
    assert selected["Date"].dt.year.tolist() == [2022, 2022]


def test_label_containment_purges_cross_boundary_returns():
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2021-12-29", "2021-12-30", "2021-12-31"]),
            "entry_date_1d": pd.to_datetime(
                ["2021-12-30", "2021-12-31", "2022-01-03"]
            ),
            "exit_date_1d": pd.to_datetime(
                ["2021-12-31", "2022-01-03", "2022-01-04"]
            ),
        }
    )
    selected = select_period(
        data,
        Period("2021-01-01", "2021-12-31"),
        require_label_within=True,
    )
    assert selected["Date"].tolist() == [pd.Timestamp("2021-12-29")]
