#!/usr/bin/env python3
"""
Extract GOLD and SILVER rows from a large COT combined text file (that also contains wheat, etc.).
Saves two CSVs: gold_cot.csv and silver_cot.csv with the original columns preserved.

Usage:
  python scripts/extract_gold_silver.py --input path/to/Gold_and_Silver.txt --outdir data/clean

Notes:
- Reads with utf-8-sig to strip BOM automatically if present.
- Matches common COMEX naming variants.
"""
import argparse
import os
import re
import sys
import pandas as pd

GOLD_PATTERNS = [
    r"\bGOLD\b",
    r"\bGOLD\s*-\s*COMMODITY EXCHANGE\b",
    r"\bGOLD\s*-\s*COMEX\b",
]
SILVER_PATTERNS = [
    r"\bSILVER\b",
    r"\bSILVER\s*-\s*COMMODITY EXCHANGE\b",
    r"\bSILVER\s*-\s*COMEX\b",
]

NAME_COL_DEFAULT = "Market_and_Exchange_Names"

def any_match(name: str, patterns) -> bool:
    n = (name or "").upper()
    return any(re.search(p, n) for p in patterns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the combined Gold/Silver txt (CSV-like)")
    ap.add_argument("--outdir", required=True, help="Directory to write gold_cot.csv and silver_cot.csv")
    ap.add_argument("--name-col", default=NAME_COL_DEFAULT,
                    help="Column that carries the market name; default matches COT files")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    try:
        df = pd.read_csv(
            args.input,
            encoding="utf-8-sig",  # strips BOM if present
            quotechar='"',
            skip_blank_lines=True,
            low_memory=False
        )
    except Exception as e:
        print(f"Failed to read {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    # Repair BOM in first header if necessary
    if args.name_col not in df.columns:
        cols = list(df.columns)
        if cols:
            cols[0] = cols[0].lstrip("\ufeff")
            df.columns = cols
    if args.name_col not in df.columns:
        print(f"Column {args.name_col} not found in file. Columns: {list(df.columns)[:12]}...", file=sys.stderr)
        sys.exit(2)

    name_series = df[args.name_col].astype(str)

    gold_mask = name_series.apply(lambda x: any_match(x, GOLD_PATTERNS))
    silver_mask = name_series.apply(lambda x: any_match(x, SILVER_PATTERNS))

    gold_df = df.loc[gold_mask].copy()
    silver_df = df.loc[silver_mask].copy()

    gold_out = os.path.join(args.outdir, "gold_cot.csv")
    silver_out = os.path.join(args.outdir, "silver_cot.csv")
    gold_df.to_csv(gold_out, index=False)
    silver_df.to_csv(silver_out, index=False)

    print(f"Wrote {len(gold_df):,} gold rows -> {gold_out}")
    print(f"Wrote {len(silver_df):,} silver rows -> {silver_out}")

if __name__ == "__main__":
    main()