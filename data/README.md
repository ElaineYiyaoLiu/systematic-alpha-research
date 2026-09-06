# Data contract

Raw vendor data is intentionally not committed. The reproducible pipeline expects
`data/processed/market_data.csv` with these columns:

`Date,Ticker,open_adj,high_adj,low_adj,close_adj,volume_adj`

Use `systematic_alpha.data.adjust_ohlcv` to convert raw yfinance-style
columns. The adjustment factor is `Adj Close / Close`; OHLC fields are multiplied
by this factor and volume is divided by it.

The published experiment used current S&P 500 constituents, so it retains
survivorship bias. This limitation is not repaired by price adjustment. A serious
production implementation should replace it with point-in-time constituent membership.

For point-in-time evaluation, provide a membership CSV with `Ticker`, `StartDate`
and `EndDate` columns, set `data.universe_mode` to `point_in_time`, and set
`data.membership_path` or pass `--membership`. Dates are inclusive and a missing
`EndDate` means membership remains active. Delisting returns must still be supplied
in the market data; the pipeline never invents a zero return for a missing holding.
