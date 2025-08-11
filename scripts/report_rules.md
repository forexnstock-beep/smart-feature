# COT Trader Brief - Report Rules and Methodology

## Overview

The COT (Commitments of Traders) Trader Brief is an automated weekly report that analyzes positioning data from the CFTC to identify potential market turning points and crowd behavior patterns.

## Data Sources

### Foreign Exchange (FX)
- **Source**: CFTC Disaggregated COT Reports (Futures + Options Combined)
- **Instruments**: EUR, GBP, JPY, CHF, AUD, NZD, CAD
- **Focus**: Leveraged Funds net positioning
- **File**: `Historical FX Future & Options Combined.txt`

### Precious Metals
- **Source**: CFTC Disaggregated COT Reports (Futures + Options Combined)
- **Instruments**: Gold, Silver (COMEX)
- **Focus**: Managed Money net positioning
- **File**: `Gold and Silver.txt`

## Calculation Methodology

### Net Positioning
- **FX**: Net Leveraged Funds = Long Positions - Short Positions
- **Metals**: Net Managed Money = Long Positions - Short Positions
- Expressed both in absolute contracts and as % of Open Interest

### Statistical Measures
- **Lookback Period**: 3 years (156 weeks)
- **Percentile Ranking**: Position of current net value within 3-year distribution
- **Z-Score**: Standard deviations from 3-year mean
- **13-Week Change**: Change in net positioning over last 13 weeks (quarterly)

### Heavy Crowd Zone Rules

#### Extreme Zones (High Mean-Reversion Risk)
- **Extreme Long**: ≥85th percentile
- **Extreme Short**: ≤15th percentile
- **Implication**: High probability of mean reversion over 4-8 week horizon

#### Elevated Zones (Moderate Mean-Reversion Risk)
- **Elevated Long**: 70th-84th percentile
- **Elevated Short**: 16th-30th percentile
- **Implication**: Moderate mean-reversion risk, monitor for catalysts

#### Balanced Zone (Trend-Continuation Likely)
- **Range**: 31st-69th percentile
- **Implication**: Positioning not extreme, trend continuation more likely

## Report Components

### 1. Asset Summaries
Each instrument shows:
- Current net positioning (contracts and % of OI)
- 3-year percentile ranking
- Z-score (standard deviations from mean)
- 13-week change in positioning
- Crowd status classification
- Tactical implication

### 2. Visual Charts
- **Top Panel**: Historical net positioning (2-year view)
- **Bottom Panel**: Percentile rankings with heavy crowd zones highlighted
- **Color Coding**: Red (extreme long zones), Blue (extreme short zones)

### 3. Alert System
- 🔴 **EXTREME LONG/SHORT**: ≥85% or ≤15% percentile
- 🟠 **Elevated**: 70-84% or 16-30% percentile  
- 🟢 **BALANCED**: 31-69% percentile

## Interpretation Guidelines

### Mean-Reversion Signals
When positioning reaches extreme levels (≥85% or ≤15%):
- **High Probability**: Contrarian positioning becomes favorable
- **Timeline**: Reversals typically occur within 4-8 weeks
- **Catalysts**: Look for fundamental or technical triggers
- **Risk Management**: Consider reducing exposure in direction of crowd

### Trend-Continuation Signals
When positioning remains balanced (31-69%):
- **Lower Risk**: Current trends may continue
- **Momentum**: Lack of extreme positioning supports trend persistence
- **Entry Opportunities**: May favor trend-following strategies

### Transition Zones
When positioning moves from extreme to balanced:
- **De-risking**: Extreme crowd positions being unwound
- **Volatility**: Potential for increased price volatility
- **Opportunity**: May present tactical entry points

## Automation Schedule

- **Frequency**: Weekly on Tuesdays at 6:00 AM UTC
- **Data Refresh**: Attempts to refresh underlying COT data files
- **Fallback**: Uses existing CSV files if raw data unavailable
- **Output**: HTML report published to GitHub Pages + downloadable artifacts

## Technical Notes

### Data Processing Pipeline
1. Extract raw COT data from text files
2. Calculate net positioning metrics
3. Compute rolling statistics (percentiles, z-scores)
4. Flag heavy crowd zones
5. Generate visualizations
6. Compile HTML report

### Quality Controls
- Minimum data requirements (26 weeks for calculations)
- Handling of missing/invalid data points
- Date validation and sorting
- Error logging and graceful degradation

## Limitations and Disclaimers

### Data Limitations
- COT data has a 3-day reporting lag
- Positioning can change rapidly between reports
- Does not capture intraday or retail positioning
- Subject to revisions and corrections

### Analytical Limitations
- Historical patterns may not repeat
- External factors can override positioning signals
- Extreme positioning can persist longer than expected
- Correlations between instruments may break down

### Usage Guidelines
- **Educational Purpose**: Analysis is for educational purposes only
- **Not Investment Advice**: Should not be used as sole basis for trading decisions
- **Risk Management**: Always use proper position sizing and stop losses
- **Professional Advice**: Consult qualified financial advisors for investment decisions

## Version History

- **v1.0**: Initial implementation with FX and metals coverage
- Automated weekly generation and GitHub Pages publishing
- Heavy crowd zone detection and visual alerts