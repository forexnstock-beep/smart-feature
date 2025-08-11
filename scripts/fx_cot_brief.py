#!/usr/bin/env python3
"""
Summarize COT for major FX from Historical FX Future & Options Combined.txt.

Targets (CME futures only): EUR, GBP, JPY, CHF, AUD, NZD, CAD.
USD/DXY is intentionally excluded per user request.

Outputs:
- data/clean/fx_cot_summary.csv with percentiles, z-scores, 13w changes
- Prints a plain-English brief per instrument for the most recent date

Usage:
  python scripts/fx_cot_brief.py --input "Historical FX Future & Options Combined.txt"
"""
import argparse
import os
import re
import sys
from typing import Dict, List
import pandas as pd
import numpy as np

TARGETS = {
    "EURO FX": r"\bEURO FX\b",
    "BRITISH POUND": r"\bBRITISH POUND\b",
    "JAPANESE YEN": r"\bJAPANESE YEN\b",
    "SWISS FRANC": r"\bSWISS FRANC\b",
    "AUSTRALIAN DOLLAR": r"\bAUSTRALIAN DOLLAR\b",
    "NZ DOLLAR": r"\bNZ DOLLAR\b",
    "CANADIAN DOLLAR": r"\bCANADIAN DOLLAR\b",
}

NAME_COL = "Market_and_Exchange_Names"
DATE_COL = "Report_Date_as_YYYY-MM-DD"
OI_COL = "Open_Interest_All"

# Disaggregated (financial futures) column patterns
DISAGG = dict(
    dealer_long=re.compile(r"^Dealer.*Long.*All$", re.I),
    dealer_short=re.compile(r"^Dealer.*Short.*All$", re.I),
    asset_long=re.compile(r"^(Asset|Asset[_\s]*Mgr).*Long.*All$", re.I),
    asset_short=re.compile(r"^(Asset|Asset[_\s]*Mgr).*Short.*All$", re.I),
    lev_long=re.compile(r"^(Lev|Leverage|Lev[_\s]*Money).*Long.*All$", re.I),
    lev_short=re.compile(r"^(Lev|Leverage|Lev[_\s]*Money).*Short.*All$", re.I),
)

# Legacy fallback
LEGACY = dict(
    noncomm_long=re.compile(r"^Non.?commercial.*Long.*All$", re.I),
    noncomm_short=re.compile(r"^Non.?commercial.*Short.*All$", re.I),
)


def find_first_col(cols: List[str], pat: re.Pattern) -> str:
    for c in cols:
        if pat.search(c):
            return c
    return ""


def detect_schema(cols: List[str]) -> Dict[str, str]:
    # Prefer disaggregated; fallback to legacy
    mapping = {}
    for k, pat in DISAGG.items():
        mapping[k] = find_first_col(cols, pat)
    mapping["schema"] = "disaggregated" if all(mapping[k] for k in ["lev_long", "lev_short", "asset_long", "asset_short"]) else "unknown"
    if mapping["schema"] == "disaggregated":
        return mapping

    mapping = {}
    for k, pat in LEGACY.items():
        mapping[k] = find_first_col(cols, pat)
    mapping["schema"] = "legacy" if all(mapping[k] for k in ["noncomm_long", "noncomm_short"]) else "unknown"
    return mapping


def cot_index(series: pd.Series, lookback: int = 156) -> pd.Series:
    roll_min = series.rolling(lookback, min_periods=max(8, lookback // 6)).min()
    roll_max = series.rolling(lookback, min_periods=max(8, lookback // 6)).max()
    return (series - roll_min) / (roll_max - roll_min).replace(0, np.nan) * 100.0


def pct_rank(series: pd.Series, lookback: int = 156) -> pd.Series:
    def ranker(x):
        r = pd.Series(x).rank(pct=True).iloc[-1] * 100
        return r
    return series.rolling(lookback, min_periods=max(8, lookback // 6)).apply(ranker, raw=False)


def zscore(series: pd.Series, lookback: int = 156) -> pd.Series:
    roll_mean = series.rolling(lookback, min_periods=max(8, lookback // 6)).mean()
    roll_std = series.rolling(lookback, min_periods=max(8, lookback // 6)).std(ddof=0)
    return (series - roll_mean) / roll_std.replace(0, np.nan)


def summarize_one(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    out = df[[DATE_COL, OI_COL]].copy()
    if mapping["schema"] == "disaggregated":
        long_short = lambda lo, sh: df[lo].fillna(0).astype(float) - df[sh].fillna(0).astype(float)
        out["net_lev"] = long_short(mapping["lev_long"], mapping["lev_short"])  # Leveraged Funds
        out["net_asset"] = long_short(mapping["asset_long"], mapping["asset_short"])  # Asset Managers
        out["net_lev_pct_oi"] = 100.0 * out["net_lev"] / out[OI_COL].replace(0, np.nan)
        out["net_asset_pct_oi"] = 100.0 * out["net_asset"] / out[OI_COL].replace(0, np.nan)
        for col in ["net_lev", "net_asset"]:
            out[f"{col}_pctile_3y"] = pct_rank(out[col])
            out[f"{col}_z_3y"] = zscore(out[col])
            out[f"{col}_idx_3y"] = cot_index(out[col])
            out[f"{col}_chg_13w"] = out[col].diff(13)
    elif mapping["schema"] == "legacy":
        net_noncomm = df[mapping["noncomm_long"]].fillna(0).astype(float) - df[mapping["noncomm_short"]].fillna(0).astype(float)
        out["net_noncomm"] = net_noncomm
        out["net_noncomm_pct_oi"] = 100.0 * out["net_noncomm"] / out[OI_COL].replace(0, np.nan)
        for col in ["net_noncomm"]:
            out[f"{col}_pctile_3y"] = pct_rank(out[col])
            out[f"{col}_z_3y"] = zscore(out[col])
            out[f"{col}_idx_3y"] = cot_index(out[col])
            out[f"{col}_chg_13w"] = out[col].diff(13)
    else:
        raise ValueError("Unknown schema: cannot summarize")
    return out


def print_brief(label: str, sdf: pd.DataFrame):
    last = sdf.dropna(subset=[DATE_COL]).iloc[-1]
    date = last[DATE_COL]
    if "net_lev" in sdf.columns:
        net = float(last["net_lev"])
        pct = float(last["net_lev_pct_oi"])
        pctile = float(last["net_lev_pctile_3y"])
        z = float(last["net_lev_z_3y"])
        chg13 = float(last["net_lev_chg_13w"])
        tone = "extreme long" if pctile >= 80 else "extreme short" if pctile <= 20 else "balanced"
        expectation = "mean-reversion risk" if (pctile >= 85 or pctile <= 15) else "trend-continuation risk"
        print(f"[{label}] as of {date.date()}: LevFunds net={net:,.0f} ({pct:+.2f}% OI), 3y pct={pctile:.0f}, z={z:+.2f}, 13w Δ={chg13:+,.0f}.")
        print("  Read: Leveraged Funds positioning is {} ; {} next 4–8 weeks (watch catalysts).".format(tone, expectation))
    elif "net_noncomm" in sdf.columns:
        net = float(last["net_noncomm"])
        pct = float(last["net_noncomm_pct_oi"])
        pctile = float(last["net_noncomm_pctile_3y"])
        z = float(last["net_noncomm_z_3y"])
        chg13 = float(last["net_noncomm_chg_13w"])
        tone = "extreme long" if pctile >= 80 else "extreme short" if pctile <= 20 else "balanced"
        expectation = "mean-reversion risk" if (pctile >= 85 or pctile <= 15) else "trend-continuation risk"
        print(f"[{label}] as of {date.date()}: NonComm net={net:,.0f} ({pct:+.2f}% OI), 3y pct={pctile:.0f}, z={z:+.2f}, 13w Δ={chg13:+,.0f}.")
        print(f"  Read: Spec positioning is {tone}; {expectation} next 4–8 weeks.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Historical FX Future & Options Combined.txt")
    ap.add_argument("--outdir", default="data/clean", help="Where to write fx_cot_summary.csv")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(
        args.input,
        encoding="utf-8-sig",
        quotechar='"',
        skip_blank_lines=True,
        low_memory=False
    )

    # BOM/header repair if needed
    if NAME_COL not in df.columns:
        cols = list(df.columns)
        if cols:
            cols[0] = cols[0].lstrip("\ufeff")
            df.columns = cols

    # Parse date
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    cols = list(df.columns)
    mapping = detect_schema(cols)
    if mapping.get("schema") == "unknown":
        print(f"Could not detect COT schema from columns. Example cols: {cols[:12]}", file=sys.stderr)
        sys.exit(1)

    summaries = []
    for label, pat in TARGETS.items():
        mask = df[NAME_COL].astype(str).str.upper().str.contains(pat, regex=True)
        sub = df.loc[mask].sort_values(DATE_COL).copy()
        if sub.empty:
            continue
        ssum = summarize_one(sub, mapping)
        ssum.insert(0, "label", label)
        summaries.append(ssum)

    if not summaries:
        print("No target instruments found in file.", file=sys.stderr)
        sys.exit(2)

    out = pd.concat(summaries, ignore_index=True)
    out = out.sort_values(["label", DATE_COL])
    csv_path = os.path.join(args.outdir, "fx_cot_summary.csv")
    out.to_csv(csv_path, index=False)
    print(f"Wrote summary: {csv_path}")

    # Print one-paragraph briefs for the latest date per label
    for label in out["label"].unique():
        sdf = out[out["label"] == label].dropna(subset=[DATE_COL]).sort_values(DATE_COL)
        if len(sdf) >= 8:
            print_brief(label, sdf)

if __name__ == "__main__":
    main()