"""Command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import adjust_ohlcv, download_current_sp500, mark_point_in_time_eligibility
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Systematic alpha research")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the frozen research pipeline")
    run.add_argument("--config", default="configs/research.yaml")
    run.add_argument("--data", default="data/processed/market_data.csv")
    run.add_argument("--output", default="results/reproduced")
    run.add_argument(
        "--membership",
        help="CSV with Ticker, StartDate and EndDate point-in-time intervals",
    )
    run.add_argument("--download-if-missing", action="store_true")
    prepare = subparsers.add_parser("prepare-data", help="adjust a raw vendor CSV")
    prepare.add_argument("--raw", default="data/raw/market_data.csv")
    prepare.add_argument("--output", default="data/processed/market_data.csv")
    download = subparsers.add_parser(
        "download", help="download the disclosed current-S&P-500 universe"
    )
    download.add_argument("--config", default="configs/research.yaml")
    download.add_argument("--raw", default="data/raw/market_data.csv")
    download.add_argument("--output", default="data/processed/market_data.csv")
    clean = subparsers.add_parser("clean-results", help="remove reproduced outputs")
    clean.add_argument("--output", default="results/reproduced")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "clean-results":
        output = Path(args.output).resolve()
        expected = (Path.cwd() / "results" / "reproduced").resolve()
        if output != expected:
            raise ValueError("Refusing to clean anything except results/reproduced")
        if output.exists():
            shutil.rmtree(output)
        return 0
    if args.command == "prepare-data":
        raw = pd.read_csv(args.raw, parse_dates=["Date"])
        processed = adjust_ohlcv(raw)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(destination, index=False)
        return 0
    if args.command == "download":
        raw_config = __import__("yaml").safe_load(
            Path(args.config).read_text(encoding="utf-8")
        )
        download_current_sp500(
            raw_config["data"]["start_date"],
            raw_config["data"]["end_date"],
            args.raw,
            args.output,
        )
        return 0
    data_path = Path(args.data)
    if not data_path.exists():
        if args.download_if_missing:
            raw_config = __import__("yaml").safe_load(
                Path(args.config).read_text(encoding="utf-8")
            )
            download_current_sp500(
                raw_config["data"]["start_date"],
                raw_config["data"]["end_date"],
                "data/raw/market_data.csv",
                data_path,
            )
        else:
            raise FileNotFoundError(
                f"Processed data not found at {data_path}. See data/README.md."
            )
    config = load_config(args.config)
    data = pd.read_csv(data_path, parse_dates=["Date"])
    membership_path = args.membership or config.membership_path
    membership_sha256 = None
    if membership_path:
        membership = pd.read_csv(membership_path)
        canonical_membership = membership.sort_values(
            ["Ticker", "StartDate", "EndDate"], na_position="last"
        ).to_csv(index=False)
        membership_sha256 = hashlib.sha256(canonical_membership.encode()).hexdigest()
        data = mark_point_in_time_eligibility(data, membership)
    elif config.universe_mode == "point_in_time":
        raise ValueError(
            "point_in_time universe_mode requires --membership or data.membership_path"
        )
    summary = run_pipeline(data, config, args.output, membership_sha256)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
