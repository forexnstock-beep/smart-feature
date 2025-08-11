#!/usr/bin/env python3
"""
Generate a weekly COT HTML report for FX majors and precious metals.

Inputs (from your workflow outputs):
- data/clean/fx_cot_summary.csv
- data/clean/gold_cot.csv
- data/clean/silver_cot.csv

Assumed columns (case-insensitive; the script will try to auto-detect):
- For FX rows (one row per symbol per week):
  symbol: str (e.g., EUR, GBP, JPY, CHF, AUD, NZD, CAD)
  report_date: date or str
  lf_net: leveraged funds net (futures), optional
  am_net: asset managers net (futures), optional
  lf_combined_net: leveraged funds net (futures+options), optional
  am_combined_net: asset managers net (futures+options), optional
  oi: open interest (contracts), optional
  close: spot or futures closing price, optional

- For metals csvs (gold_cot.csv, silver_cot.csv):
  report_date, lf_net / lf_combined_net, am_net / am_combined_net, oi, close (if available)

Outputs:
- docs/cot_report.html (colorful table + tips)
- docs/charts/{SYMBOL}_lf.png, {SYMBOL}_am.png (time series with 15/85 percentile bands)

Crowding logic (configurable below):
- lookback_weeks: 156 (≈3y)
- heavy crowd if percentile >= 0.85 (long crowd) or <= 0.15 (short crowd)
- confirm crowd with OI percentile >= 0.70 when oi is available
- alignment: LF and AM are same sign; divergence: opposite signs
"""
from __future__ import annotations

import os
import io
import math
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- Configuration -----------------

LOOKBACK_WEEKS = int(os.getenv("COT_LOOKBACK_WEEKS", 156))  # ~3 years
HEAVY_CROWD_PCTL = float(os.getenv("COT_HEAVY_CROWD_PCTL", 0.85))  # >= 85% long crowd, <= 15% short crowd
HEAVY_CROWD_PCTL_LOW = 1.0 - HEAVY_CROWD_PCTL
OI_CONFIRM_PCTL = float(os.getenv("COT_OI_CONFIRM_PCTL", 0.70))
USE_COMBINED_IF_AVAILABLE = True  # prefer futures+options combined nets when present
OUTPUT_DIR = Path("docs")
CHART_DIR = OUTPUT_DIR / "charts"
OUTPUT_HTML = OUTPUT_DIR / "cot_report.html"

FX_FILE = Path("data/clean/fx_cot_summary.csv")
GOLD_FILE = Path("data/clean/gold_cot.csv")
SILVER_FILE = Path("data/clean/silver_cot.csv")

FX_SYMBOLS_ORDER = ["EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]

# ----------------- Helpers -----------------

def _col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _ensure_date(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_datetime(df[col], errors="coerce", utc=True).dt.date
    return s

def _percentile_rank(arr: np.ndarray, value: float) -> float:
    # Percentile rank within array (inclusive of equals)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return np.nan
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    less_equal = np.sum(arr <= value)
    return less_equal / len(arr)

def _zscore(arr: np.ndarray, value: float) -> float:
    if len(arr) < 2:
        return np.nan
    mu = np.nanmean(arr)
    sd = np.nanstd(arr, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return (value - mu) / sd

def _last_window(series: pd.Series, lookback: int) -> np.ndarray:
    vals = series.dropna().values
    if len(vals) == 0:
        return np.array([])
    return vals[-lookback:]

def _sign(x: float) -> int:
    if pd.isna(x):
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)

def _fmt_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x*100:.0f}%"

def _fmt_num(x: float) -> str:
    if pd.isna(x):
        return ""
    # Thousands with k suffix
    return f"{x/1000:.1f}k"

def _spark_chart(
    dates: List[pd.Timestamp],
    values: List[float],
    p15: float,
    p85: float,
    title: str,
    outfile: Path,
) -> None:
    plt.figure(figsize=(8, 2.4), dpi=150)
    plt.plot(dates, values, color="#1f77b4", linewidth=1.4, label="Net")
    plt.axhline(p85, color="#2ca02c", linestyle="--", linewidth=1.0, label="85th pct")
    plt.axhline(p15, color="#d62728", linestyle="--", linewidth=1.0, label="15th pct")
    plt.title(title, fontsize=10)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()

@dataclass
class InstrumentMetrics:
    symbol: str
    date: str
    group: str  # LF or AM
    net: float
    pctl: float
    z: float
    oi_pctl: float
    crowd: str  # "heavy_long", "heavy_short", "normal"
    alignment: str  # "aligned", "divergent", "unknown"

@dataclass
class InstrumentSummary:
    symbol: str
    date: str
    lf_pctl: float
    am_pctl: float
    lf_z: float
    am_z: float
    lf_net: float
    am_net: float
    oi_pctl: float
    crowd_tag: str
    bias: str
    note: str
    charts: Dict[str, str]  # {"lf": path, "am": path}

# ----------------- Core scoring & rules -----------------

def score_series(
    df: pd.DataFrame,
    symbol: str,
    which: str,
    use_combined: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[InstrumentMetrics], Optional[str]]:
    """
    Return window dataframe, metrics for last point, and chart path.
    """
    # Column inference
    date_col = _col(df, ["report_date", "report_date_as_yyyy-mm-dd", "date"])
    if not date_col:
        return None, None, None

    # Handle different data formats for FX vs metals
    sdf = df.copy()
    sdf[date_col] = _ensure_date(sdf, date_col)
    sdf = sdf.sort_values(date_col)

    # Filter symbol if present  
    sym_col = _col(sdf, ["symbol", "code", "ticker", "label", "market_and_exchange_names"])
    if sym_col:
        # Handle different symbol formats for FX vs metals
        if "AUSTRALIAN DOLLAR" in symbol.upper() or symbol.upper() == "AUD":
            symbol_match = "AUSTRALIAN DOLLAR"
        elif "CANADIAN DOLLAR" in symbol.upper() or symbol.upper() == "CAD":
            symbol_match = "CANADIAN DOLLAR"  
        elif "NZ DOLLAR" in symbol.upper() or symbol.upper() == "NZD":
            symbol_match = "NZ DOLLAR"
        elif "SWISS FRANC" in symbol.upper() or symbol.upper() == "CHF":
            symbol_match = "SWISS FRANC"
        elif "EURO" in symbol.upper() or symbol.upper() == "EUR":
            symbol_match = "EURO FX"
        elif "BRITISH POUND" in symbol.upper() or symbol.upper() == "GBP":
            symbol_match = "BRITISH POUND"
        elif "JAPANESE YEN" in symbol.upper() or symbol.upper() == "JPY":
            symbol_match = "JAPANESE YEN"
        elif "GOLD" in symbol.upper():
            symbol_match = "GOLD"
        elif "SILVER" in symbol.upper():
            symbol_match = "SILVER"
        else:
            symbol_match = symbol.upper()
            
        # Filter for matching symbols
        mask = sdf[sym_col].astype(str).str.upper().str.contains(symbol_match, na=False)
        sdf = sdf[mask].copy()

    if len(sdf) == 0:
        return None, None, None

    # Now determine the net position based on data format and which group
    net_col = None
    
    # Check if this is FX summary format (has net_lev, net_asset columns)
    if 'net_lev' in sdf.columns and 'net_asset' in sdf.columns:
        if which == "LF":
            net_col = 'net_lev'
        else:  # AM
            net_col = 'net_asset'
    else:
        # This is raw CFTC disaggregated format - calculate net positions
        if which == "LF":
            # Leveraged funds = Money Managers in disaggregated format
            long_col = _col(sdf, ['m_money_positions_long_all', 'm_money_long_all'])
            short_col = _col(sdf, ['m_money_positions_short_all', 'm_money_short_all'])
            
            if long_col and short_col and long_col in sdf.columns and short_col in sdf.columns:
                sdf['lf_net_calc'] = sdf[long_col].astype(float) - sdf[short_col].astype(float)
                net_col = 'lf_net_calc'
        else:  # AM
            # Asset Managers = Swap Dealers in metals disaggregated format, or use Other Reportables
            long_col = _col(sdf, ['swap_positions_long_all', 'other_rept_positions_long_all'])
            short_col = _col(sdf, ['swap__positions_short_all', 'swap_positions_short_all', 'other_rept_positions_short_all'])
            
            if long_col and short_col and long_col in sdf.columns and short_col in sdf.columns:
                sdf['am_net_calc'] = sdf[long_col].astype(float) - sdf[short_col].astype(float)
                net_col = 'am_net_calc'

    if net_col is None or net_col not in sdf.columns:
        return None, None, None

    oi_col = _col(sdf, ["oi", "open_interest", "open_interest_all"])

    # Window
    net = sdf[net_col].astype(float)
    dates = pd.to_datetime(sdf[date_col])
    window_net = _last_window(net, LOOKBACK_WEEKS)
    window_oi = _last_window(sdf[oi_col].astype(float), LOOKBACK_WEEKS) if oi_col else np.array([])
    
    if len(window_net) == 0:
        return None, None, None
        
    pctl = _percentile_rank(window_net, window_net[-1]) if len(window_net) else np.nan
    z = _zscore(window_net, window_net[-1]) if len(window_net) else np.nan
    oi_pctl = _percentile_rank(window_oi, window_oi[-1]) if len(window_oi) else np.nan

    crowd = "normal"
    if not pd.isna(pctl):
        if pctl >= HEAVY_CROWD_PCTL and (pd.isna(oi_pctl) or oi_pctl >= OI_CONFIRM_PCTL):
            crowd = "heavy_long"
        elif pctl <= HEAVY_CROWD_PCTL_LOW and (pd.isna(oi_pctl) or oi_pctl >= OI_CONFIRM_PCTL):
            crowd = "heavy_short"

    last_date = str(sdf[date_col].iloc[-1])
    last_net = float(net.iloc[-1])

    # Chart values: plot the level lines at historical 15/85th percentiles
    if len(window_net) >= 10:
        p15 = np.nanpercentile(window_net, HEAVY_CROWD_PCTL_LOW * 100)
        p85 = np.nanpercentile(window_net, HEAVY_CROWD_PCTL * 100)
    else:
        p15 = np.nan
        p85 = np.nan

    # Compute full series for chart
    cdates = pd.to_datetime(sdf[date_col])
    cvalues = sdf[net_col].astype(float).values
    chart_file = CHART_DIR / f"{symbol}_{which.lower()}.png"
    try:
        _spark_chart(
            dates=list(cdates),
            values=list(cvalues),
            p15=p15 if not pd.isna(p15) else 0.0,
            p85=p85 if not pd.isna(p85) else 0.0,
            title=f"{symbol} {which} net ({'combined' if use_combined else 'futures'})",
            outfile=chart_file,
        )
        chart_path = str(chart_file).replace("\\", "/")
    except Exception as e:
        print(f"Error creating chart for {symbol} {which}: {e}")
        chart_path = ""

    metrics = InstrumentMetrics(
        symbol=symbol,
        date=last_date,
        group=which,
        net=last_net,
        pctl=float(pctl) if not pd.isna(pctl) else np.nan,
        z=float(z) if not pd.isna(z) else np.nan,
        oi_pctl=float(oi_pctl) if not pd.isna(oi_pctl) else np.nan,
        crowd=crowd,
        alignment="unknown",
    )
    return sdf, metrics, chart_path

    # Window
    net = sdf[net_col].astype(float)
    dates = pd.to_datetime(sdf[date_col])
    window_net = _last_window(net, LOOKBACK_WEEKS)
    window_oi = _last_window(sdf[oi_col].astype(float), LOOKBACK_WEEKS) if oi_col else np.array([])
    pctl = _percentile_rank(window_net, window_net[-1]) if len(window_net) else np.nan
    z = _zscore(window_net, window_net[-1]) if len(window_net) else np.nan
    oi_pctl = _percentile_rank(window_oi, window_oi[-1]) if len(window_oi) else np.nan

    crowd = "normal"
    if not pd.isna(pctl):
        if pctl >= HEAVY_CROWD_PCTL and (pd.isna(oi_pctl) or oi_pctl >= OI_CONFIRM_PCTL):
            crowd = "heavy_long"
        elif pctl <= HEAVY_CROWD_PCTL_LOW and (pd.isna(oi_pctl) or oi_pctl >= OI_CONFIRM_PCTL):
            crowd = "heavy_short"

    last_date = str(sdf[date_col].iloc[-1])
    last_net = float(net.iloc[-1])

    # Chart values: plot the level lines at historical 15/85th percentiles
    if len(window_net) >= 10:
        p15 = np.nanpercentile(window_net, HEAVY_CROWD_PCTL_LOW * 100)
        p85 = np.nanpercentile(window_net, HEAVY_CROWD_PCTL * 100)
    else:
        p15 = np.nan
        p85 = np.nan

    # Compute full series for chart
    cdates = pd.to_datetime(sdf[date_col])
    cvalues = sdf[net_col].astype(float).values
    chart_file = CHART_DIR / f"{symbol}_{which.lower()}.png"
    try:
        _spark_chart(
            dates=list(cdates),
            values=list(cvalues),
            p15=p15 if not pd.isna(p15) else 0.0,
            p85=p85 if not pd.isna(p85) else 0.0,
            title=f"{symbol} {which} net ({'combined' if use_combined else 'futures'})",
            outfile=chart_file,
        )
        chart_path = str(chart_file).replace("\\", "/")
    except Exception:
        chart_path = ""

    metrics = InstrumentMetrics(
        symbol=symbol,
        date=last_date,
        group=which,
        net=last_net,
        pctl=float(pctl) if not pd.isna(pctl) else np.nan,
        z=float(z) if not pd.isna(z) else np.nan,
        oi_pctl=float(oi_pctl) if not pd.isna(oi_pctl) else np.nan,
        crowd=crowd,
        alignment="unknown",
    )
    return sdf, metrics, chart_path

def decide_bias(
    symbol: str,
    lf: InstrumentMetrics | None,
    am: InstrumentMetrics | None,
) -> Tuple[str, str, str]:
    """
    Return (crowd_tag, bias, note).
    """
    if lf is None or am is None:
        return "unknown", "neutral", "Insufficient data to score both LF and AM."

    # Alignment / divergence
    lf_sign = _sign(lf.net)
    am_sign = _sign(am.net)
    if lf_sign == 0 or am_sign == 0:
        alignment = "unknown"
    elif lf_sign == am_sign:
        alignment = "aligned"
    else:
        alignment = "divergent"
    lf.alignment = am.alignment = alignment

    # Heavy crowd tags
    def tag_for(m: InstrumentMetrics) -> Optional[str]:
        if m.crowd == "heavy_long":
            return f"{m.group}:crowded_long"
        if m.crowd == "heavy_short":
            return f"{m.group}:crowded_short"
        return None

    tags = [t for t in [tag_for(lf), tag_for(am)] if t]
    crowd_tag = ",".join(tags) if tags else "normal"

    # Rule-of-thumb bias
    if alignment == "aligned" and lf_sign > 0:
        if "crowded_long" in crowd_tag:
            bias = "buy-dips-cautious"
            note = "Aligned long and crowded; buy pullbacks into support with confirmation; avoid chasing highs."
        else:
            bias = "buy-dips"
            note = "Aligned long; buy dips while structure holds; watch USD/real yields."
    elif alignment == "aligned" and lf_sign < 0:
        if "crowded_short" in crowd_tag:
            bias = "sell-rallies-cautious"
            note = "Aligned short and crowded; sell rallies into resistance; manage squeeze risk."
        else:
            bias = "sell-rallies"
            note = "Aligned short; fade bounces while lower-highs persist."
    elif alignment == "divergent":
        if am_sign > 0 and lf_sign < 0:
            bias = "range-to-upside-risk"
            note = "Divergent: AM long vs LF short; upside squeeze risk if price reclaims MAs; wait for higher-low."
        elif am_sign < 0 and lf_sign > 0:
            bias = "range-to-downside-risk"
            note = "Divergent: AM short vs LF long; rallies can fail near resistance; wait for breakdown to confirm."
        else:
            bias = "neutral"
            note = "Divergent but weak signals; let price action lead."
    else:
        bias = "neutral"
        note = "Signals mixed; follow price."

    return crowd_tag, bias, note

def build_fx_report(fx_df: pd.DataFrame) -> List[InstrumentSummary]:
    results: List[InstrumentSummary] = []
    for symbol in FX_SYMBOLS_ORDER:
        # Score LF and AM
        _, lf_m, lf_chart = score_series(fx_df, symbol, "LF", use_combined=USE_COMBINED_IF_AVAILABLE)
        _, am_m, am_chart = score_series(fx_df, symbol, "AM", use_combined=USE_COMBINED_IF_AVAILABLE)
        if lf_m is None and am_m is None:
            continue
        crowd_tag, bias, note = decide_bias(symbol, lf_m, am_m)

        # Use OI percentile if available from either
        oi_pctl = np.nan
        if lf_m and not pd.isna(lf_m.oi_pctl):
            oi_pctl = lf_m.oi_pctl
        if am_m and not pd.isna(am_m.oi_pctl):
            oi_pctl = np.nanmean([oi_pctl, am_m.oi_pctl]) if not pd.isna(oi_pctl) else am_m.oi_pctl

        results.append(
            InstrumentSummary(
                symbol=symbol,
                date=lf_m.date if lf_m else (am_m.date if am_m else ""),
                lf_pctl=lf_m.pctl if lf_m else np.nan,
                am_pctl=am_m.pctl if am_m else np.nan,
                lf_z=lf_m.z if lf_m else np.nan,
                am_z=am_m.z if am_m else np.nan,
                lf_net=lf_m.net if lf_m else np.nan,
                am_net=am_m.net if am_m else np.nan,
                oi_pctl=oi_pctl,
                crowd_tag=crowd_tag,
                bias=bias,
                note=note,
                charts={"lf": lf_chart or "", "am": am_chart or ""},
            )
        )
    return results

def build_metal_report(df: pd.DataFrame, symbol: str) -> Optional[InstrumentSummary]:
    _, lf_m, lf_chart = score_series(df, symbol, "LF", use_combined=USE_COMBINED_IF_AVAILABLE)
    _, am_m, am_chart = score_series(df, symbol, "AM", use_combined=USE_COMBINED_IF_AVAILABLE)
    if lf_m is None and am_m is None:
        return None
    crowd_tag, bias, note = decide_bias(symbol, lf_m, am_m)
    oi_pctl = np.nan
    if lf_m and not pd.isna(lf_m.oi_pctl):
        oi_pctl = lf_m.oi_pctl
    if am_m and not pd.isna(am_m.oi_pctl):
        oi_pctl = np.nanmean([oi_pctl, am_m.oi_pctl]) if not pd.isna(oi_pctl) else am_m.oi_pctl
    return InstrumentSummary(
        symbol=symbol,
        date=lf_m.date if lf_m else (am_m.date if am_m else ""),
        lf_pctl=lf_m.pctl if lf_m else np.nan,
        am_pctl=am_m.pctl if am_m else np.nan,
        lf_z=lf_m.z if lf_m else np.nan,
        am_z=am_m.z if am_m else np.nan,
        lf_net=lf_m.net if lf_m else np.nan,
        am_net=am_m.net if am_m else np.nan,
        oi_pctl=oi_pctl,
        crowd_tag=crowd_tag,
        bias=bias,
        note=note,
        charts={"lf": lf_chart or "", "am": am_chart or ""},
    )

# ----------------- HTML rendering -----------------

CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }
small { color: #555; }
table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: center; font-size: 13px; }
th { background: #f7f7f7; }
.badge { padding: 2px 6px; border-radius: 4px; color: #fff; font-weight: 600; font-size: 12px; }
.bull { background: #2e7d32; }
.bear { background: #c62828; }
.warn { background: #ef6c00; }
.neu  { background: #546e7a; }
.pchip { background: #1976d2; }
.fade { color: #777; }
.cell-good { background: #e8f5e9; }
.cell-bad  { background: #ffebee; }
.cell-warn { background: #fff3e0; }
.subtle { font-size: 12px; color: #666; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }
.card { border: 1px solid #ddd; border-radius: 6px; padding: 8px; }
.card h4 { margin: 4px 0 8px 0; }
figure { margin: 0; }
figcaption { font-size: 12px; color: #555; }
"""

def badge_for(summary: InstrumentSummary) -> str:
    if summary.bias.startswith("buy-dips"):
        return '<span class="badge bull">Buy dips</span>'
    if summary.bias.startswith("sell-rallies"):
        return '<span class="badge bear">Sell rallies</span>'
    if summary.bias.startswith("range-to-upside"):
        return '<span class="badge bull">Divergent: Upside risk</span>'
    if summary.bias.startswith("range-to-downside"):
        return '<span class="badge bear">Divergent: Downside risk</span>'
    return '<span class="badge neu">Neutral</span>'

def crowd_chip(tag: str) -> str:
    if "crowded_long" in tag and "crowded_short" in tag:
        return '<span class="badge warn">Crowded both</span>'
    if "crowded_long" in tag:
        return '<span class="badge warn">Crowded long</span>'
    if "crowded_short" in tag:
        return '<span class="badge warn">Crowded short</span>'
    return '<span class="badge neu">Normal</span>'

def cell_class_from_pctl(p: float, inverse: bool = False) -> str:
    if pd.isna(p):
        return ""
    if not inverse:
        if p >= HEAVY_CROWD_PCTL: return "cell-good"
        if p <= HEAVY_CROWD_PCTL_LOW: return "cell-bad"
        if abs(p - 0.5) <= 0.1: return ""
        return "cell-warn"
    else:
        # For risk cells (OI crowding) high is caution
        if p >= OI_CONFIRM_PCTL: return "cell-warn"
        return ""

def render_table(title: str, rows: List[InstrumentSummary]) -> str:
    head = """
    <table>
      <thead>
        <tr>
          <th>Symbol</th>
          <th>Bias</th>
          <th>LF pct</th>
          <th>AM pct</th>
          <th>LF z</th>
          <th>AM z</th>
          <th>LF net</th>
          <th>AM net</th>
          <th>OI pct</th>
          <th>Crowd</th>
          <th>Note</th>
        </tr>
      </thead>
      <tbody>
    """
    body = []
    for r in rows:
        body.append(f"""
        <tr>
          <td><b>{r.symbol}</b><br><small class=\"fade\">{r.date}</small></td>
          <td>{badge_for(r)}</td>
          <td class=\"{cell_class_from_pctl(r.lf_pctl)}\">{_fmt_pct(r.lf_pctl)}</td>
          <td class=\"{cell_class_from_pctl(r.am_pctl)}\">{_fmt_pct(r.am_pctl)}</td>
          <td>{'' if pd.isna(r.lf_z) else f'{r.lf_z:+.1f}'}</td>
          <td>{'' if pd.isna(r.am_z) else f'{r.am_z:+.1f}'}</td>
          <td>{_fmt_num(r.lf_net)}</td>
          <td>{_fmt_num(r.am_net)}</td>
          <td class=\"{cell_class_from_pctl(r.oi_pctl, inverse=True)}\">{_fmt_pct(r.oi_pctl)}</td>
          <td>{crowd_chip(r.crowd_tag)}</td>
          <td style=\"text-align:left\">{r.note}</td>
        </tr>
        """)
    tail = """
      </tbody>
    </table>
    """
    return f"<h3>{title}</h3>\n" + head + "\n".join(body) + tail

def render_charts_grid(rows: List[InstrumentSummary]) -> str:
    cards = []
    for r in rows:
        card = f"""
        <div class=\"card\">
          <h4>{r.symbol} positioning</h4>
          <div class=\"grid\">
            <figure>
              <img src=\"{r.charts.get('lf','')}\" alt=\"{r.symbol} LF chart\" width=\"100%\" />
              <figcaption>LF net with 15/85th percentile lines</figcaption>
            </figure>
            <figure>
              <img src=\"{r.charts.get('am','')}\" alt=\"{r.symbol} AM chart\" width=\"100%\" />
              <figcaption>AM net with 15/85th percentile lines</figcaption>
            </figure>
          </div>
        </div>
        """
        cards.append(card)
    return "\n".join(cards)

def render_suggestions(fx: List[InstrumentSummary], metals: List[InstrumentSummary]) -> str:
    bullets = []
    def line(r: InstrumentSummary) -> str:
        if r.bias.startswith("buy-dips"):
            extra = "Prefer pullbacks into support; avoid chasing breakouts; invalidate on lower-low."
        elif r.bias.startswith("sell-rallies"):
            extra = "Prefer fades into resistance; beware squeeze risk; invalidate on higher-high."
        elif r.bias.startswith("range-to-upside"):
            extra = "Divergent; upside squeeze risk—wait for reclaim and higher-low."
        elif r.bias.startswith("range-to-downside"):
            extra = "Divergent; downside risk—wait for breakdown and lower-high."
        else:
            extra = "Neutral; let price action lead."
        crowd_hint = ""
        if "crowded" in r.crowd_tag:
            crowd_hint = " Crowded: take profits faster; expect mean reversion."
        return f"<li><b>{r.symbol}</b>: {badge_for(r)} — {extra}{crowd_hint}</li>"

    for r in fx + metals:
        bullets.append(line(r))
    return "<ul>" + "\n".join(bullets) + "</ul>"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    fx_df = pd.read_csv(FX_FILE) if FX_FILE.exists() else pd.DataFrame()
    gold_df = pd.read_csv(GOLD_FILE) if GOLD_FILE.exists() else pd.DataFrame()
    silver_df = pd.read_csv(SILVER_FILE) if SILVER_FILE.exists() else pd.DataFrame()

    fx_rows = build_fx_report(fx_df) if not fx_df.empty else []
    metals_rows: List[InstrumentSummary] = []
    if not gold_df.empty:
        g = build_metal_report(gold_df, "GOLD")
        if g: metals_rows.append(g)
    if not silver_df.empty:
        s = build_metal_report(silver_df, "SILVER")
        if s: metals_rows.append(s)

    # HTML body
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>COT Weekly Trader Brief</title>
<style>{CSS}</style>
</head>
<body>
  <h2>COT Weekly Trader Brief</h2>
  <p class="subtle">Generated: {timestamp} • Lookback: {LOOKBACK_WEEKS}w • Heavy crowd: ≥{int(HEAVY_CROWD_PCTL*100)}% or ≤{int(HEAVY_CROWD_PCTL_LOW*100)}%, OI confirm ≥{int(OI_CONFIRM_PCTL*100)}%</p>

  {render_table("FX (LF vs AM, futures+options where available)", fx_rows)}
  <div class="grid">
    {render_charts_grid(fx_rows)}
  </div>

  {render_table("Metals", metals_rows)}
  <div class="grid">
    {render_charts_grid(metals_rows)}
  </div>

  <h3>How to read this</h3>
  <ul>
    <li>LF = leveraged funds (faster money), AM = asset managers (real money, slower).</li>
    <li>Percentiles/z-scores use a {LOOKBACK_WEEKS} week window.</li>
    <li>Heavy crowd if percentile ≥{int(HEAVY_CROWD_PCTL*100)}% (crowded long) or ≤{int(HEAVY_CROWD_PCTL_LOW*100)}% (crowded short), confirmed by high OI percentile when available.</li>
    <li>Aligned = LF and AM same direction; Divergent = opposite directions.</li>
  </ul>

  <h3>Actionable suggestions</h3>
  {render_suggestions(fx_rows, metals_rows)}

  <p class="subtle">This report is informational; pair with price action (VWAP/MA structure) and macro (DXY, real yields).</p>
</body>
</html>
"""
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote report to {OUTPUT_HTML}")

if __name__ == "__main__":
    main()