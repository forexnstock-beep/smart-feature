#!/usr/bin/env python3
"""
Generate weekly COT "Trader Brief" HTML report.

This script:
1. Reads FX COT data from data/clean/fx_cot_summary.csv
2. Reads metals COT data from data/clean/gold_cot.csv and data/clean/silver_cot.csv
3. Calculates positioning metrics and flags "heavy crowd" zones (>=85% long or <=15% short)
4. Generates HTML report with visualizations
5. Saves to docs/index.html for GitHub Pages

Usage:
  python scripts/generate_cot_report.py [--outdir docs]
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def load_fx_data(data_dir: str) -> pd.DataFrame:
    """Load FX COT summary data."""
    fx_path = os.path.join(data_dir, "fx_cot_summary.csv")
    if not os.path.exists(fx_path):
        print(f"Warning: FX COT data not found at {fx_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(fx_path)
    df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'])
    return df


def process_metals_data(data_dir: str) -> pd.DataFrame:
    """Load and process metals COT data (Gold and Silver)."""
    metals_data = []
    
    for metal in ['gold', 'silver']:
        metal_path = os.path.join(data_dir, f"{metal}_cot.csv")
        if not os.path.exists(metal_path):
            print(f"Warning: {metal.title()} COT data not found at {metal_path}")
            continue
            
        df = pd.read_csv(metal_path)
        if df.empty:
            continue
            
        df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'])
        
        # Calculate net positions for managed money (similar to leveraged funds in FX)
        if 'M_Money_Positions_Long_All' in df.columns and 'M_Money_Positions_Short_All' in df.columns:
            df['net_mm'] = df['M_Money_Positions_Long_All'].fillna(0) - df['M_Money_Positions_Short_All'].fillna(0)
            df['net_mm_pct_oi'] = 100.0 * df['net_mm'] / df['Open_Interest_All'].replace(0, np.nan)
        else:
            continue
            
        # Calculate 3-year percentiles and z-scores (156 weeks)
        df = df.sort_values('Report_Date_as_YYYY-MM-DD')
        df['net_mm_pctile_3y'] = df['net_mm'].rolling(156, min_periods=26).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        
        # Z-score calculation
        roll_mean = df['net_mm'].rolling(156, min_periods=26).mean()
        roll_std = df['net_mm'].rolling(156, min_periods=26).std(ddof=0)
        df['net_mm_z_3y'] = (df['net_mm'] - roll_mean) / roll_std.replace(0, np.nan)
        
        # 13-week change
        df['net_mm_chg_13w'] = df['net_mm'].diff(13)
        
        df['label'] = metal.upper()
        metals_data.append(df[['label', 'Report_Date_as_YYYY-MM-DD', 'Open_Interest_All', 
                             'net_mm', 'net_mm_pct_oi', 'net_mm_pctile_3y', 'net_mm_z_3y', 'net_mm_chg_13w']])
    
    if metals_data:
        return pd.concat(metals_data, ignore_index=True)
    return pd.DataFrame()


def identify_heavy_crowd_zones(df: pd.DataFrame, pctile_col: str) -> pd.DataFrame:
    """Flag heavy crowd zones where positioning is >=85% or <=15%."""
    df = df.copy()
    df['heavy_crowd'] = (df[pctile_col] >= 85) | (df[pctile_col] <= 15)
    df['crowd_direction'] = np.where(df[pctile_col] >= 85, 'EXTREME_LONG', 
                                   np.where(df[pctile_col] <= 15, 'EXTREME_SHORT', 'BALANCED'))
    return df


def create_positioning_chart(df: pd.DataFrame, title: str, filename: str, outdir: str) -> str:
    """Create a positioning chart and return the filename."""
    if df.empty:
        return ""
        
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top plot: Net positioning
    for label in df['label'].unique():
        subset = df[df['label'] == label].sort_values('Report_Date_as_YYYY-MM-DD')
        if 'net_lev' in subset.columns:
            ax1.plot(subset['Report_Date_as_YYYY-MM-DD'], subset['net_lev'], 
                    label=label, linewidth=2)
        elif 'net_mm' in subset.columns:
            ax1.plot(subset['Report_Date_as_YYYY-MM-DD'], subset['net_mm'], 
                    label=label, linewidth=2)
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Net Position (Contracts)')
    ax1.set_title(f'{title} - Net Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Percentiles with heavy crowd zones
    for label in df['label'].unique():
        subset = df[df['label'] == label].sort_values('Report_Date_as_YYYY-MM-DD')
        pctile_col = 'net_lev_pctile_3y' if 'net_lev_pctile_3y' in subset.columns else 'net_mm_pctile_3y'
        
        if pctile_col in subset.columns:
            ax2.plot(subset['Report_Date_as_YYYY-MM-DD'], subset[pctile_col], 
                    label=label, linewidth=2)
    
    # Add heavy crowd zone highlights
    ax2.axhspan(85, 100, alpha=0.2, color='red', label='Extreme Long Zone (>=85%)')
    ax2.axhspan(0, 15, alpha=0.2, color='blue', label='Extreme Short Zone (<=15%)')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50th Percentile')
    
    ax2.set_ylabel('3-Year Percentile (%)')
    ax2.set_xlabel('Date')
    ax2.set_title(f'{title} - Positioning Percentiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    chart_path = os.path.join(outdir, filename)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def generate_summary_text(df: pd.DataFrame, asset_type: str) -> List[str]:
    """Generate summary text for the latest data."""
    summaries = []
    
    for label in df['label'].unique():
        subset = df[df['label'] == label].dropna().sort_values('Report_Date_as_YYYY-MM-DD')
        if subset.empty:
            continue
            
        latest = subset.iloc[-1]
        date = latest['Report_Date_as_YYYY-MM-DD'].strftime('%Y-%m-%d')
        
        if asset_type == 'FX':
            net = latest['net_lev']
            pct_oi = latest['net_lev_pct_oi']
            pctile = latest['net_lev_pctile_3y']
            z_score = latest['net_lev_z_3y']
            chg_13w = latest['net_lev_chg_13w']
            trader_type = "Leveraged Funds"
        else:  # Metals
            net = latest['net_mm']
            pct_oi = latest['net_mm_pct_oi']
            pctile = latest['net_mm_pctile_3y']
            z_score = latest['net_mm_z_3y']
            chg_13w = latest['net_mm_chg_13w']
            trader_type = "Managed Money"
        
        if pd.isna(pctile):
            continue
            
        # Determine crowd status and implications
        if pctile >= 85:
            crowd_status = "🔴 EXTREME LONG"
            implication = "HIGH mean-reversion risk"
        elif pctile <= 15:
            crowd_status = "🔵 EXTREME SHORT"
            implication = "HIGH mean-reversion risk"
        elif pctile >= 70:
            crowd_status = "🟠 Elevated Long"
            implication = "MODERATE mean-reversion risk"
        elif pctile <= 30:
            crowd_status = "🟠 Elevated Short"
            implication = "MODERATE mean-reversion risk"
        else:
            crowd_status = "🟢 BALANCED"
            implication = "trend-continuation likely"
        
        summary = f"""
        <div class="asset-summary">
            <h3>{label}</h3>
            <p><strong>As of {date}</strong></p>
            <p><strong>{trader_type} Net:</strong> {net:,.0f} contracts ({pct_oi:+.1f}% of OI)</p>
            <p><strong>3-Year Percentile:</strong> {pctile:.0f}% | <strong>Z-Score:</strong> {z_score:+.2f}</p>
            <p><strong>13-Week Change:</strong> {chg_13w:+,.0f} contracts</p>
            <p><strong>Crowd Status:</strong> {crowd_status}</p>
            <p><strong>Implication:</strong> {implication} over next 4-8 weeks</p>
        </div>
        """
        summaries.append(summary)
    
    return summaries


def generate_html_report(fx_data: pd.DataFrame, metals_data: pd.DataFrame, 
                        fx_chart: str, metals_chart: str, outdir: str) -> str:
    """Generate the complete HTML report."""
    
    report_date = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    
    fx_summaries = generate_summary_text(fx_data, 'FX') if not fx_data.empty else []
    metals_summaries = generate_summary_text(metals_data, 'Metals') if not metals_data.empty else []
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>COT Trader Brief</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f5f5f5; 
                color: #333;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{ 
                color: #2c3e50; 
                text-align: center; 
                border-bottom: 3px solid #3498db; 
                padding-bottom: 10px;
            }}
            h2 {{ 
                color: #34495e; 
                margin-top: 40px; 
                border-left: 4px solid #3498db; 
                padding-left: 15px;
            }}
            h3 {{ 
                color: #2980b9; 
                margin-bottom: 10px;
            }}
            .asset-summary {{ 
                background: #f8f9fa; 
                border: 1px solid #e9ecef; 
                border-radius: 8px; 
                padding: 20px; 
                margin: 20px 0;
                border-left: 4px solid #3498db;
            }}
            .chart-container {{ 
                text-align: center; 
                margin: 30px 0; 
                padding: 20px;
                background: #fafafa;
                border-radius: 8px;
            }}
            .chart-container img {{ 
                max-width: 100%; 
                height: auto; 
                border-radius: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .methodology {{ 
                background: #e8f4fd; 
                padding: 20px; 
                border-radius: 8px; 
                margin: 30px 0;
                border-left: 4px solid #2196F3;
            }}
            .timestamp {{ 
                text-align: center; 
                color: #666; 
                font-style: italic; 
                margin-bottom: 30px;
            }}
            .legend {{
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 15px;
                margin: 20px 0;
            }}
            .legend h4 {{
                margin-top: 0;
                color: #856404;
            }}
            .warning {{ 
                background: #fff3cd; 
                color: #856404; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 15px 0;
                border: 1px solid #ffeaa7;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 COT Trader Brief</h1>
            <div class="timestamp">Generated on {report_date}</div>
            
            <div class="methodology">
                <h4>📋 Methodology</h4>
                <p><strong>Data Source:</strong> CFTC Commitments of Traders reports (Futures + Options Combined where available)</p>
                <p><strong>Lookback Period:</strong> 3 years (156 weeks) for percentiles and z-scores</p>
                <p><strong>Heavy Crowd Zones:</strong> ≥85th percentile (Extreme Long) or ≤15th percentile (Extreme Short)</p>
                <p><strong>FX Focus:</strong> Leveraged Funds net positioning | <strong>Metals Focus:</strong> Managed Money net positioning</p>
            </div>
            
            <div class="legend">
                <h4>🚦 Crowd Status Legend</h4>
                <p><strong>🔴 EXTREME LONG/SHORT (≥85% or ≤15%):</strong> High mean-reversion risk</p>
                <p><strong>🟠 Elevated (70-84% or 16-30%):</strong> Moderate mean-reversion risk</p>
                <p><strong>🟢 BALANCED (31-69%):</strong> Trend-continuation likely</p>
            </div>
    """
    
    # Add FX section
    if fx_summaries:
        html_content += "<h2>💱 Foreign Exchange</h2>\n"
        for summary in fx_summaries:
            html_content += summary
        
        if fx_chart:
            html_content += f"""
            <div class="chart-container">
                <h3>FX Positioning Charts</h3>
                <img src="{fx_chart}" alt="FX Positioning Chart">
            </div>
            """
    else:
        html_content += """
        <h2>💱 Foreign Exchange</h2>
        <div class="warning">⚠️ FX COT data not available</div>
        """
    
    # Add Metals section
    if metals_summaries:
        html_content += "<h2>🥇 Precious Metals</h2>\n"
        for summary in metals_summaries:
            html_content += summary
            
        if metals_chart:
            html_content += f"""
            <div class="chart-container">
                <h3>Metals Positioning Charts</h3>
                <img src="{metals_chart}" alt="Metals Positioning Chart">
            </div>
            """
    else:
        html_content += """
        <h2>🥇 Precious Metals</h2>
        <div class="warning">⚠️ Metals COT data not available</div>
        """
    
    html_content += """
            <div class="methodology">
                <h4>⚠️ Disclaimer</h4>
                <p>This analysis is for educational purposes only and should not be considered investment advice. 
                COT data reflects positioning as of the report date and can change rapidly. Past performance 
                does not guarantee future results. Always consult with a qualified financial advisor before 
                making investment decisions.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content


def main():
    parser = argparse.ArgumentParser(description='Generate COT Trader Brief HTML report')
    parser.add_argument('--data-dir', default='data/clean', 
                       help='Directory containing COT CSV files')
    parser.add_argument('--outdir', default='docs', 
                       help='Output directory for HTML report and charts')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load data
    print("Loading FX COT data...")
    fx_data = load_fx_data(args.data_dir)
    
    print("Loading and processing metals COT data...")
    metals_data = process_metals_data(args.data_dir)
    
    if fx_data.empty and metals_data.empty:
        print("Error: No COT data found. Please run the data extraction scripts first.")
        sys.exit(1)
    
    # Apply heavy crowd analysis
    if not fx_data.empty:
        fx_data = identify_heavy_crowd_zones(fx_data, 'net_lev_pctile_3y')
    
    if not metals_data.empty:
        metals_data = identify_heavy_crowd_zones(metals_data, 'net_mm_pctile_3y')
    
    # Create charts
    fx_chart = ""
    if not fx_data.empty:
        # Filter to last 2 years for charts
        cutoff_date = datetime.now() - timedelta(days=730)
        fx_recent = fx_data[fx_data['Report_Date_as_YYYY-MM-DD'] >= cutoff_date]
        fx_chart = create_positioning_chart(fx_recent, "Foreign Exchange", "fx_positioning.png", args.outdir)
        print(f"Created FX chart: {fx_chart}")
    
    metals_chart = ""
    if not metals_data.empty:
        # Filter to last 2 years for charts
        cutoff_date = datetime.now() - timedelta(days=730)
        metals_recent = metals_data[metals_data['Report_Date_as_YYYY-MM-DD'] >= cutoff_date]
        metals_chart = create_positioning_chart(metals_recent, "Precious Metals", "metals_positioning.png", args.outdir)
        print(f"Created metals chart: {metals_chart}")
    
    # Generate HTML report
    print("Generating HTML report...")
    html_content = generate_html_report(fx_data, metals_data, fx_chart, metals_chart, args.outdir)
    
    # Save HTML report
    report_path = os.path.join(args.outdir, 'index.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ COT Trader Brief generated: {report_path}")
    
    # Print summary of heavy crowd zones
    if not fx_data.empty:
        fx_latest = fx_data.groupby('label').tail(1)
        heavy_fx = fx_latest[fx_latest['heavy_crowd'] == True]
        if not heavy_fx.empty:
            print(f"\n🚨 FX Heavy Crowd Alerts:")
            for _, row in heavy_fx.iterrows():
                print(f"  {row['label']}: {row['crowd_direction']} ({row['net_lev_pctile_3y']:.0f}%)")
    
    if not metals_data.empty:
        metals_latest = metals_data.groupby('label').tail(1)
        heavy_metals = metals_latest[metals_latest['heavy_crowd'] == True]
        if not heavy_metals.empty:
            print(f"\n🚨 Metals Heavy Crowd Alerts:")
            for _, row in heavy_metals.iterrows():
                print(f"  {row['label']}: {row['crowd_direction']} ({row['net_mm_pctile_3y']:.0f}%)")


if __name__ == '__main__':
    main()