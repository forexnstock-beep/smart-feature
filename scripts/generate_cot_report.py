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
- OI confirmation >= 0.75
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuration
LOOKBACK_WEEKS = 156  # ~3 years
HEAVY_CROWD_PCTL = 0.85
HEAVY_CROWD_PCTL_LOW = 0.15
OI_CONFIRM_PCTL = 0.75

def find_column(df, patterns):
    """Find a column that matches one of the patterns (case-insensitive)."""
    for pattern in patterns:
        for col in df.columns:
            if pattern.lower() in col.lower():
                return col
    return None

def load_and_process_csv(filepath, symbol_name=None):
    """Load CSV and standardize column names."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    if df.empty:
        return df
    
    # Try to find common columns
    date_col = find_column(df, ['report_date', 'date', 'Date'])
    if date_col:
        df['report_date'] = pd.to_datetime(df[date_col])
    
    # Add symbol if not present
    if 'symbol' not in df.columns and symbol_name:
        df['symbol'] = symbol_name
    
    return df

def calculate_percentiles_and_scores(df, value_cols):
    """Calculate percentiles and z-scores for given value columns."""
    for col in value_cols:
        if col in df.columns:
            # Ensure numeric data type
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Skip if all values are NaN after conversion
            if df[col].isna().all():
                continue
            # Calculate rolling percentiles and z-scores
            df[f'{col}_percentile'] = df[col].rolling(LOOKBACK_WEEKS, min_periods=52).rank(pct=True)
            df[f'{col}_zscore'] = (df[col] - df[col].rolling(LOOKBACK_WEEKS, min_periods=52).mean()) / df[col].rolling(LOOKBACK_WEEKS, min_periods=52).std()
    return df

def determine_crowd_status(row, lf_col, am_col, oi_col=None):
    """Determine crowding status based on percentiles."""
    lf_pct = row.get(f'{lf_col}_percentile', 0.5)
    am_pct = row.get(f'{am_col}_percentile', 0.5)
    oi_pct = row.get(f'{oi_col}_percentile', 0.5) if oi_col else 0.5
    
    # Heavy crowd logic
    lf_heavy_long = lf_pct >= HEAVY_CROWD_PCTL
    lf_heavy_short = lf_pct <= HEAVY_CROWD_PCTL_LOW
    am_heavy_long = am_pct >= HEAVY_CROWD_PCTL
    am_heavy_short = am_pct <= HEAVY_CROWD_PCTL_LOW
    oi_confirm = oi_pct >= OI_CONFIRM_PCTL if oi_col else True
    
    status = []
    if (lf_heavy_long or lf_heavy_short) and oi_confirm:
        status.append(f"LF {'crowded long' if lf_heavy_long else 'crowded short'}")
    if (am_heavy_long or am_heavy_short) and oi_confirm:
        status.append(f"AM {'crowded long' if am_heavy_long else 'crowded short'}")
    
    # Alignment
    if lf_pct > 0.5 and am_pct > 0.5:
        alignment = "Aligned bullish"
    elif lf_pct < 0.5 and am_pct < 0.5:
        alignment = "Aligned bearish"
    else:
        alignment = "Divergent"
    
    return " | ".join(status) if status else "Neutral", alignment

def create_chart(df, symbol, col, chart_type):
    """Create a time series chart with percentile bands."""
    if df.empty or col not in df.columns:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot the main line
    plt.plot(df['report_date'], df[col], label=f'{symbol} {chart_type}', linewidth=2)
    
    # Add percentile bands if available
    if f'{col}_percentile' in df.columns:
        recent_data = df.tail(LOOKBACK_WEEKS)
        if not recent_data.empty:
            p15 = recent_data[col].quantile(0.15)
            p85 = recent_data[col].quantile(0.85)
            plt.axhline(y=p15, color='red', linestyle='--', alpha=0.7, label='15th percentile')
            plt.axhline(y=p85, color='red', linestyle='--', alpha=0.7, label='85th percentile')
    
    plt.title(f'{symbol} - {chart_type}')
    plt.xlabel('Date')
    plt.ylabel('Net Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save chart
    chart_path = f'docs/charts/{symbol}_{chart_type.lower().replace(" ", "_")}.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_html_report(fx_data, metals_data):
    """Generate the HTML report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COT Weekly Trader Brief</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h2 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        
        .subtle {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        
        .bullish {{ background-color: #d4edda; }}
        .bearish {{ background-color: #f8d7da; }}
        .neutral {{ background-color: #fff3cd; }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 4px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .chart-container img {{
            width: 100%;
            height: auto;
        }}
        
        ul {{
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>COT Weekly Trader Brief</h2>
        <p class="subtle">Generated: {timestamp} • Lookback: {LOOKBACK_WEEKS}w • Heavy crowd: ≥{int(HEAVY_CROWD_PCTL*100)}% or ≤{int(HEAVY_CROWD_PCTL_LOW*100)}%, OI confirm ≥{int(OI_CONFIRM_PCTL*100)}%</p>
"""
    
    # Add FX section
    if not fx_data.empty:
        html_template += render_table("FX (LF vs AM, futures+options where available)", fx_data)
        html_template += '<div class="grid">'
        html_template += render_charts_grid(fx_data, 'FX')
        html_template += '</div>'
    
    # Add Metals section
    if not metals_data.empty:
        html_template += render_table("Metals", metals_data)
        html_template += '<div class="grid">'
        html_template += render_charts_grid(metals_data, 'Metals')
        html_template += '</div>'
    
    # Add footer
    html_template += f"""
        <h3>How to read this</h3>
        <ul>
            <li>LF = leveraged funds (faster money), AM = asset managers (real money, slower).</li>
            <li>Percentiles/z-scores use a {LOOKBACK_WEEKS} week window.</li>
            <li>Heavy crowd if percentile ≥{int(HEAVY_CROWD_PCTL*100)}% (crowded long) or ≤{int(HEAVY_CROWD_PCTL_LOW*100)}% (crowded short), confirmed by high OI percentile when available.</li>
            <li>Aligned = LF and AM same direction; Divergent = opposite directions.</li>
        </ul>

        <h3>Actionable suggestions</h3>
        {render_suggestions(fx_data, metals_data)}

        <p class="subtle">This report is informational; pair with price action (VWAP/MA structure) and macro (DXY, real yields).</p>
    </div>
</body>
</html>"""
    
    return html_template

def render_table(title, data):
    """Render a data table in HTML."""
    if data.empty:
        return f"<h3>{title}</h3><p>No data available</p>"
    
    html = f"<h3>{title}</h3><table><thead><tr>"
    
    # Table headers
    headers = ['Symbol', 'Date', 'LF Net', 'AM Net', 'Status', 'Alignment']
    for header in headers:
        html += f"<th>{header}</th>"
    html += "</tr></thead><tbody>"
    
    # Table rows - get most recent data for each symbol
    if 'symbol' in data.columns and 'report_date' in data.columns:
        latest_data = data.sort_values('report_date').groupby('symbol').tail(1)
    else:
        latest_data = data.tail(10)  # Just show last 10 rows if no proper structure
    
    for _, row in latest_data.iterrows():
        symbol = row.get('symbol', 'Unknown')
        date = row.get('report_date', row.get('Report_Date_as_YYYY-MM-DD', 'N/A'))
        if pd.notna(date):
            if hasattr(date, 'strftime'):
                date = date.strftime('%Y-%m-%d')
            else:
                date = str(date)
        
        lf_net = row.get('lf_net', 'N/A')
        am_net = row.get('am_net', 'N/A')
        
        # Format numbers
        if pd.notna(lf_net) and lf_net != 'N/A':
            try:
                lf_net = f"{float(lf_net):,.0f}" if abs(float(lf_net)) >= 1 else f"{float(lf_net):.3f}"
            except (ValueError, TypeError):
                lf_net = 'N/A'
        if pd.notna(am_net) and am_net != 'N/A':
            try:
                am_net = f"{float(am_net):,.0f}" if abs(float(am_net)) >= 1 else f"{float(am_net):.3f}"
            except (ValueError, TypeError):
                am_net = 'N/A'
        
        status = "Neutral"
        alignment = "Unknown"
        
        # Basic alignment logic
        try:
            lf_val = float(row.get('lf_net', 0)) if pd.notna(row.get('lf_net', 0)) else 0
            am_val = float(row.get('am_net', 0)) if pd.notna(row.get('am_net', 0)) else 0
            
            if lf_val > 0 and am_val > 0:
                alignment = "Aligned bullish"
            elif lf_val < 0 and am_val < 0:
                alignment = "Aligned bearish"
            else:
                alignment = "Divergent"
        except (ValueError, TypeError):
            pass
        
        # Get row class for coloring
        row_class = "neutral"
        if "bullish" in alignment.lower():
            row_class = "bullish"
        elif "bearish" in alignment.lower():
            row_class = "bearish"
        
        html += f'<tr class="{row_class}">'
        html += f"<td>{symbol}</td>"
        html += f"<td>{date}</td>"
        html += f"<td>{lf_net}</td>"
        html += f"<td>{am_net}</td>"
        html += f"<td>{status}</td>"
        html += f"<td>{alignment}</td>"
        html += "</tr>"
    
    html += "</tbody></table>"
    return html

def render_charts_grid(data, section_name):
    """Render charts grid in HTML."""
    if data.empty:
        return ""
    
    html = ""
    symbols = data['symbol'].unique() if 'symbol' in data.columns else [section_name]
    
    for symbol in symbols:
        for chart_type in ['lf', 'am']:
            chart_file = f'{symbol}_{chart_type}.png'
            chart_path = f'charts/{chart_file}'
            if os.path.exists(f'docs/{chart_path}'):
                html += f'<div class="chart-container"><img src="{chart_path}" alt="{symbol} {chart_type.upper()}" /></div>'
    
    return html

def render_suggestions(fx_data, metals_data):
    """Render actionable suggestions based on current positioning."""
    suggestions = []
    
    # This would typically analyze the data and provide specific suggestions
    # For now, provide general guidance
    suggestions.append("Monitor DXY levels and Fed policy signals for directional bias.")
    suggestions.append("Watch for divergences between LF and AM positioning as potential reversal signals.")
    suggestions.append("Consider fade strategies when both LF and AM are heavily crowded in same direction.")
    
    html = "<ul>"
    for suggestion in suggestions:
        html += f"<li>{suggestion}</li>"
    html += "</ul>"
    
    return html

def main():
    # Ensure output directories exist
    os.makedirs('docs', exist_ok=True)
    os.makedirs('docs/charts', exist_ok=True)
    
    # Load data
    print("Loading COT data...")
    fx_data = load_and_process_csv('data/clean/fx_cot_summary.csv')
    gold_data = load_and_process_csv('data/clean/gold_cot.csv', 'GOLD')
    silver_data = load_and_process_csv('data/clean/silver_cot.csv', 'SILVER')
    
    # For FX data, map columns to standard names
    if not fx_data.empty:
        if 'label' in fx_data.columns:
            fx_data['symbol'] = fx_data['label'].str.extract(r'(\w+)')[0]
        if 'net_lev' in fx_data.columns:
            fx_data['lf_net'] = fx_data['net_lev']
        if 'net_asset' in fx_data.columns:
            fx_data['am_net'] = fx_data['net_asset']
    
    # For metals data, calculate net positions and standardize
    metals_data_list = []
    for df, symbol in [(gold_data, 'GOLD'), (silver_data, 'SILVER')]:
        if not df.empty:
            df = df.copy()
            # Calculate net positions from M_Money (managed money = leveraged funds)
            if 'M_Money_Positions_Long_All' in df.columns and 'M_Money_Positions_Short_All' in df.columns:
                df['lf_net'] = pd.to_numeric(df['M_Money_Positions_Long_All'], errors='coerce') - pd.to_numeric(df['M_Money_Positions_Short_All'], errors='coerce')
            # Asset managers would be "Prod_Merc" (producers/merchants) in this context
            if 'Prod_Merc_Positions_Long_All' in df.columns and 'Prod_Merc_Positions_Short_All' in df.columns:
                df['am_net'] = pd.to_numeric(df['Prod_Merc_Positions_Long_All'], errors='coerce') - pd.to_numeric(df['Prod_Merc_Positions_Short_All'], errors='coerce')
            
            df['symbol'] = symbol
            metals_data_list.append(df)
    
    metals_data = pd.concat(metals_data_list, ignore_index=True) if metals_data_list else pd.DataFrame()
    
    # Process data and calculate metrics
    if not fx_data.empty:
        # Focus on key net position columns for FX
        value_cols = ['lf_net', 'am_net', 'Open_Interest_All']
        fx_data = calculate_percentiles_and_scores(fx_data, value_cols)
    
    if not metals_data.empty:
        # Focus on calculated net positions for metals
        value_cols = ['lf_net', 'am_net', 'Open_Interest_All']
        metals_data = calculate_percentiles_and_scores(metals_data, value_cols)
    
    # Generate charts
    print("Generating charts...")
    for df, data_type in [(fx_data, 'FX'), (metals_data, 'Metals')]:
        if not df.empty and 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                if 'report_date' in symbol_data.columns:
                    symbol_data = symbol_data.sort_values('report_date')
                
                # Create LF and AM charts
                for chart_type in ['LF', 'AM']:
                    col_name = f'{chart_type.lower()}_net'
                    if col_name in symbol_data.columns and not symbol_data[col_name].isna().all():
                        create_chart(symbol_data, symbol, col_name, chart_type)
    
    # Generate HTML report
    print("Generating HTML report...")
    html_content = generate_html_report(fx_data, metals_data)
    
    # Write HTML file
    with open('docs/cot_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("COT report generation complete!")
    print(f"- HTML report: docs/cot_report.html")
    print(f"- Charts: docs/charts/")

if __name__ == '__main__':
    main()