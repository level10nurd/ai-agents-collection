"""
Visualization Tools for CFO Agent

Creates executive-ready charts and dashboards for financial reports.
Returns base64-encoded PNG images suitable for embedding in reports or saving to files.
"""

import base64
import io
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

from ..models.cash_forecast import CashForecast
from ..models.sales_forecast import SalesForecast
from ..models.unit_economics import UnitEconomicsAnalysis


def create_cash_forecast_chart(
    forecast: CashForecast,
    return_path: bool = False,
    output_path: Optional[str] = None
) -> str:
    """
    Create a line chart showing 13-week cash forecast with scenarios.

    Features:
    - Three scenario lines (base, optimistic, pessimistic)
    - Red zone highlighting for weeks < $100K
    - Professional styling for executive reports
    - Grid lines and clear labels

    Args:
        forecast: CashForecast model with scenarios and runway metrics
        return_path: If True, save to file and return path instead of base64
        output_path: File path to save chart (if return_path=True)

    Returns:
        Base64 encoded PNG string (if return_path=False) or file path (if return_path=True)

    Example:
        ```python
        chart_b64 = create_cash_forecast_chart(forecast)
        # Use in HTML: <img src="data:image/png;base64,{chart_b64}" />
        ```
    """
    # Create figure with appropriate size for executive reports
    fig, ax = plt.subplots(figsize=(12, 7))

    # Extract week numbers and dates from base scenario
    weeks = [flow.week_number for flow in forecast.base_scenario.weekly_flows]
    dates = [flow.week_start_date for flow in forecast.base_scenario.weekly_flows]

    # Extract ending cash for each scenario
    base_cash = [flow.ending_cash for flow in forecast.base_scenario.weekly_flows]
    optimistic_cash = [flow.ending_cash for flow in forecast.optimistic_scenario.weekly_flows]
    pessimistic_cash = [flow.ending_cash for flow in forecast.pessimistic_scenario.weekly_flows]

    # Plot scenario lines
    ax.plot(weeks, base_cash, label='Base Case', color='#2E86AB', linewidth=2.5, marker='o')
    ax.plot(weeks, optimistic_cash, label='Optimistic (+30%)', color='#06A77D', 
            linewidth=2, linestyle='--', marker='^', alpha=0.8)
    ax.plot(weeks, pessimistic_cash, label='Pessimistic (-30%)', color='#D62828', 
            linewidth=2, linestyle='--', marker='v', alpha=0.8)

    # Add red zone for weeks < $100K
    danger_threshold = 100000
    ax.axhline(y=danger_threshold, color='red', linestyle=':', linewidth=1.5, 
               alpha=0.6, label='Danger Zone ($100K)')
    
    # Shade the danger zone area
    y_min = min(min(base_cash), min(pessimistic_cash)) * 0.95  # 5% below minimum
    ax.fill_between(weeks, y_min, danger_threshold, color='red', alpha=0.1)

    # Formatting
    ax.set_xlabel('Week Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cash Balance ($)', fontsize=12, fontweight='bold')
    ax.set_title('13-Week Cash Forecast - Scenario Analysis', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend with better positioning
    ax.legend(loc='best', framealpha=0.95, fontsize=10)
    
    # Add current runway annotation
    runway_text = (f"Current Runway: {forecast.runway_metrics.runway_months:.1f} months\n"
                   f"Weekly Burn: ${forecast.runway_metrics.average_weekly_burn:,.0f}")
    ax.text(0.02, 0.98, runway_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Tight layout for better appearance
    plt.tight_layout()

    # Return base64 or save to file
    if return_path:
        save_path = output_path or f"cash_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64


def create_sales_forecast_chart(
    forecast: SalesForecast,
    return_path: bool = False,
    output_path: Optional[str] = None
) -> str:
    """
    Create a line chart showing sales forecast with uncertainty bands.

    Features:
    - Forecast line (yhat) with uncertainty bands (yhat_lower, yhat_upper)
    - Highlighted Nov-Dec shopping season
    - Monthly aggregation for clarity
    - Professional styling for executive reports

    Args:
        forecast: SalesForecast model with Prophet predictions
        return_path: If True, save to file and return path instead of base64
        output_path: File path to save chart (if return_path=True)

    Returns:
        Base64 encoded PNG string (if return_path=False) or file path (if return_path=True)

    Example:
        ```python
        chart_b64 = create_sales_forecast_chart(forecast)
        ```
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Extract data
    dates = [period.ds for period in forecast.forecast_periods]
    yhat = [period.yhat for period in forecast.forecast_periods]
    yhat_lower = [period.yhat_lower for period in forecast.forecast_periods]
    yhat_upper = [period.yhat_upper for period in forecast.forecast_periods]

    # Plot forecast line
    ax.plot(dates, yhat, label='Forecast', color='#2E86AB', linewidth=2.5)

    # Add uncertainty band
    ax.fill_between(dates, yhat_lower, yhat_upper, 
                     color='#2E86AB', alpha=0.2, label='Uncertainty Band')

    # Highlight Nov-Dec shopping season
    shopping_periods = forecast.get_shopping_season_periods()
    if shopping_periods:
        shopping_dates = [p.ds for p in shopping_periods]
        shopping_yhat = [p.yhat for p in shopping_periods]
        
        # Add markers for shopping season
        ax.scatter(shopping_dates, shopping_yhat, color='#D62828', 
                   s=100, zorder=5, label='Shopping Season (Nov-Dec)', marker='*')
        
        # Add shaded regions for each shopping season period
        for period in shopping_periods:
            # Create a small shaded area around each shopping season point
            ax.axvspan(period.ds, period.ds, alpha=0.15, color='#FFD700')

    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'{forecast.forecast_horizon_months}-Month Sales Forecast', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45, ha='right')
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', framealpha=0.95, fontsize=10)
    
    # Add model metrics annotation
    if forecast.model_metadata.mape is not None:
        metrics_text = (f"Model Accuracy (MAPE): {forecast.model_metadata.mape:.1f}%\n"
                        f"Shopping Season %: {forecast.get_shopping_season_revenue_percentage():.1f}%")
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Tight layout
    plt.tight_layout()

    # Return base64 or save to file
    if return_path:
        save_path = output_path or f"sales_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64


def create_unit_economics_dashboard(
    analysis: UnitEconomicsAnalysis,
    return_path: bool = False,
    output_path: Optional[str] = None
) -> str:
    """
    Create a dashboard with bar charts and traffic light indicators.

    Features:
    - Bar charts for CAC, LTV, LTV:CAC ratio
    - Traffic light indicators (🟢 green, 🟡 yellow, 🔴 red) for benchmark compliance
    - Benchmark reference lines
    - Clean dashboard layout

    Args:
        analysis: UnitEconomicsAnalysis model with calculated metrics
        return_path: If True, save to file and return path instead of base64
        output_path: File path to save chart (if return_path=True)

    Returns:
        Base64 encoded PNG string (if return_path=False) or file path (if return_path=True)

    Example:
        ```python
        dashboard_b64 = create_unit_economics_dashboard(analysis)
        ```
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Color scheme for traffic lights
    def get_status_color(is_violation: bool) -> str:
        return '#D62828' if is_violation else '#06A77D'  # Red or Green

    # 1. CAC Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    cac_color = get_status_color(False)  # CAC itself doesn't have a direct benchmark
    ax1.bar(['CAC'], [analysis.cac], color=cac_color, width=0.5)
    ax1.set_ylabel('Cost ($)', fontweight='bold')
    ax1.set_title('Customer Acquisition Cost (CAC)', fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. LTV Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    ltv_color = get_status_color(False)  # LTV itself doesn't have a direct benchmark
    # Handle infinite LTV (zero churn)
    ltv_display = min(analysis.ltv, analysis.cac * 10) if analysis.ltv != float('inf') else analysis.cac * 10
    ax2.bar(['LTV'], [ltv_display], color=ltv_color, width=0.5)
    ax2.set_ylabel('Value ($)', fontweight='bold')
    ax2.set_title('Customer Lifetime Value (LTV)', fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax2.grid(True, alpha=0.3, axis='y')
    if analysis.ltv == float('inf'):
        ax2.text(0, ltv_display * 0.5, 'Infinite\n(0% churn)', 
                ha='center', va='center', fontsize=12, fontweight='bold')

    # 3. LTV:CAC Ratio with Benchmark
    ax3 = fig.add_subplot(gs[1, 0])
    ratio_color = get_status_color(analysis.ltv_cac_below_benchmark)
    ratio_display = min(analysis.ltv_cac_ratio, 15) if analysis.ltv_cac_ratio != float('inf') else 15
    ax3.bar(['LTV:CAC Ratio'], [ratio_display], color=ratio_color, width=0.5)
    ax3.axhline(y=3.0, color='orange', linestyle='--', linewidth=2, label='Benchmark (3.0)')
    ax3.set_ylabel('Ratio', fontweight='bold')
    ax3.set_title('LTV:CAC Ratio (Target ≥ 3.0)', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add traffic light indicator
    status_symbol = '🔴' if analysis.ltv_cac_below_benchmark else '🟢'
    ax3.text(0, ratio_display * 0.9, status_symbol, ha='center', fontsize=40)

    # 4. CAC Payback Period with Benchmark
    ax4 = fig.add_subplot(gs[1, 1])
    if analysis.cac_payback_months is not None:
        payback_color = get_status_color(analysis.cac_payback_above_benchmark)
        ax4.bar(['CAC Payback'], [analysis.cac_payback_months], color=payback_color, width=0.5)
        ax4.axhline(y=12.0, color='orange', linestyle='--', linewidth=2, label='Benchmark (12 mo)')
        ax4.set_ylabel('Months', fontweight='bold')
        ax4.set_title('CAC Payback Period (Target ≤ 12 mo)', fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add traffic light indicator
        status_symbol = '🔴' if analysis.cac_payback_above_benchmark else '🟢'
        ax4.text(0, analysis.cac_payback_months * 0.9, status_symbol, ha='center', fontsize=40)
    else:
        ax4.text(0.5, 0.5, 'CAC Payback\nNot Available\n(missing monthly revenue)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('CAC Payback Period', fontweight='bold')
        ax4.axis('off')

    # 5. Churn Rate with Benchmark
    ax5 = fig.add_subplot(gs[2, 0])
    churn_color = get_status_color(analysis.churn_above_benchmark)
    monthly_churn_pct = analysis.monthly_churn_rate * 100
    ax5.bar(['Monthly Churn'], [monthly_churn_pct], color=churn_color, width=0.5)
    ax5.axhline(y=8.0, color='orange', linestyle='--', linewidth=2, label='Benchmark (8%)')
    ax5.set_ylabel('Percentage (%)', fontweight='bold')
    ax5.set_title('Monthly Churn Rate (Target ≤ 8%)', fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add traffic light indicator
    status_symbol = '🔴' if analysis.churn_above_benchmark else '🟢'
    ax5.text(0, monthly_churn_pct * 0.9, status_symbol, ha='center', fontsize=40)

    # 6. Gross Margin with Benchmark
    ax6 = fig.add_subplot(gs[2, 1])
    margin_color = get_status_color(analysis.gross_margin_below_benchmark)
    margin_pct = analysis.gross_margin * 100
    ax6.bar(['Gross Margin'], [margin_pct], color=margin_color, width=0.5)
    ax6.axhline(y=60.0, color='orange', linestyle='--', linewidth=2, label='Benchmark (60%)')
    ax6.set_ylabel('Percentage (%)', fontweight='bold')
    ax6.set_title('Gross Margin (Target ≥ 60%)', fontweight='bold')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add traffic light indicator
    status_symbol = '🔴' if analysis.gross_margin_below_benchmark else '🟢'
    ax6.text(0, margin_pct * 0.9, status_symbol, ha='center', fontsize=40)

    # Overall title
    period_label = analysis.period_label or 'Current Period'
    fig.suptitle(f'Unit Economics Dashboard - {period_label}', 
                 fontsize=16, fontweight='bold', y=0.98)

    # Return base64 or save to file
    if return_path:
        save_path = output_path or f"unit_economics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64
