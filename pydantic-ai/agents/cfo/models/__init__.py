"""
CFO Pydantic Models

Structured data models for CFO analyses:
- UnitEconomicsAnalysis: CAC, LTV, ratios, benchmark flags
- SalesForecast: Prophet forecast with uncertainty intervals
- CashForecast: 13-week cash flow with scenarios
- InventoryAnalysis: Stock levels, reorder points
- FinancialModel: 3-statement financial projections
- ExecutiveReport: Executive summary format
- TechnicalReport: Detailed methodology and calculations
- Ledger Models: Chart of Accounts, General Ledger, Trial Balance
"""

from .unit_economics import UnitEconomicsAnalysis
from .sales_forecast import SalesForecast, ForecastPeriod, ModelMetadata
from .cash_forecast import (
    CashForecast,
    CashScenario,
    WeeklyCashFlow,
    RunwayMetrics,
)
from .inventory import (
    InventoryAnalysis,
    SKUInventory,
    FulfillmentMetrics,
    Channel,
)
from .financial_model import (
    FinancialModel,
    Scenario,
    IncomeStatementPeriod,
    CashFlowPeriod,
    BalanceSheetPeriod,
    SensitivityAnalysis,
    Frequency,
)
from .reports import (
    ExecutiveReport,
    TechnicalReport,
    KeyMetric,
)
from .ledger import (
    ChartOfAccounts,
    ChartOfAccountsEntry,
    GeneralLedger,
    GeneralLedgerEntry,
    TrialBalance,
    TrialBalanceEntry,
    AccountType,
    NormalBalance,
)

__all__ = [
    # Unit Economics
    "UnitEconomicsAnalysis",
    # Sales Forecast
    "SalesForecast",
    "ForecastPeriod",
    "ModelMetadata",
    # Cash Forecast
    "CashForecast",
    "CashScenario",
    "WeeklyCashFlow",
    "RunwayMetrics",
    # Inventory
    "InventoryAnalysis",
    "SKUInventory",
    "FulfillmentMetrics",
    "Channel",
    # Financial Model
    "FinancialModel",
    "Scenario",
    "IncomeStatementPeriod",
    "CashFlowPeriod",
    "BalanceSheetPeriod",
    "SensitivityAnalysis",
    "Frequency",
    # Reports
    "ExecutiveReport",
    "TechnicalReport",
    "KeyMetric",
    # Ledger
    "ChartOfAccounts",
    "ChartOfAccountsEntry",
    "GeneralLedger",
    "GeneralLedgerEntry",
    "TrialBalance",
    "TrialBalanceEntry",
    "AccountType",
    "NormalBalance",
]
