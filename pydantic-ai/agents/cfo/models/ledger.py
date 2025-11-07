"""
Ledger Models

Pydantic models for Chart of Accounts, General Ledger entries, and Trial Balance.
These models support CSV import of accounting data for financial analysis.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date
from enum import Enum
from decimal import Decimal


class AccountType(str, Enum):
    """Standard account types in a chart of accounts."""
    ASSET = "Asset"
    LIABILITY = "Liability"
    EQUITY = "Equity"
    REVENUE = "Revenue"
    EXPENSE = "Expense"
    COGS = "COGS"  # Cost of Goods Sold


class NormalBalance(str, Enum):
    """Normal balance for an account (debit or credit)."""
    DEBIT = "Debit"
    CREDIT = "Credit"


class ChartOfAccountsEntry(BaseModel):
    """
    Single entry in the Chart of Accounts.
    
    Maps an account code to its properties and categorization.
    """
    
    account_code: str = Field(
        ...,
        description="Unique account identifier (e.g., '1000', '4100')"
    )
    account_name: str = Field(
        ...,
        description="Human-readable account name"
    )
    account_type: AccountType = Field(
        ...,
        description="Type of account (Asset, Liability, Equity, Revenue, Expense, COGS)"
    )
    account_subtype: Optional[str] = Field(
        None,
        description="Subtype for more granular classification (e.g., 'Bank', 'AccountsReceivable')"
    )
    parent_account: Optional[str] = Field(
        None,
        description="Parent account code for hierarchical rollups"
    )
    is_active: bool = Field(
        True,
        description="Whether the account is currently in use"
    )
    normal_balance: NormalBalance = Field(
        ...,
        description="Normal balance type (Debit or Credit)"
    )
    description: Optional[str] = Field(
        None,
        description="Additional notes about the account"
    )
    
    @field_validator('account_code')
    @classmethod
    def validate_account_code(cls, v: str) -> str:
        """Ensure account code is not empty and stripped of whitespace."""
        code = v.strip()
        if not code:
            raise ValueError("Account code cannot be empty")
        return code
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "account_code": "4100",
                "account_name": "Product Sales",
                "account_type": "Revenue",
                "account_subtype": "ProductSales",
                "parent_account": "4000",
                "is_active": True,
                "normal_balance": "Credit",
                "description": "Revenue from product sales"
            }
        }


class GeneralLedgerEntry(BaseModel):
    """
    Single line in the General Ledger.
    
    Represents one side of a double-entry journal entry.
    """
    
    transaction_id: str = Field(
        ...,
        description="Unique identifier for the journal entry (multiple lines share same ID)"
    )
    transaction_date: date = Field(
        ...,
        description="Date of the transaction"
    )
    account_code: str = Field(
        ...,
        description="Account code (must exist in Chart of Accounts)"
    )
    account_name: str = Field(
        ...,
        description="Account name (for reference)"
    )
    debit: Decimal = Field(
        default=Decimal("0.00"),
        ge=0,
        description="Debit amount (0 if credit entry)"
    )
    credit: Decimal = Field(
        default=Decimal("0.00"),
        ge=0,
        description="Credit amount (0 if debit entry)"
    )
    description: str = Field(
        ...,
        description="Transaction description"
    )
    vendor: Optional[str] = Field(
        None,
        description="Vendor/customer name"
    )
    memo: Optional[str] = Field(
        None,
        description="Additional notes or reference"
    )
    department: Optional[str] = Field(
        None,
        description="Department or class for tracking"
    )
    location: Optional[str] = Field(
        None,
        description="Location or branch"
    )
    
    @field_validator('debit', 'credit')
    @classmethod
    def validate_amounts(cls, v: Decimal) -> Decimal:
        """Ensure amounts have at most 2 decimal places."""
        if v.as_tuple().exponent < -2:
            raise ValueError("Amounts must have at most 2 decimal places")
        return v.quantize(Decimal("0.01"))
    
    def validate_entry(self) -> None:
        """Validate that entry has either debit or credit (not both or neither)."""
        if self.debit > 0 and self.credit > 0:
            raise ValueError(f"Entry {self.transaction_id} has both debit and credit")
        if self.debit == 0 and self.credit == 0:
            raise ValueError(f"Entry {self.transaction_id} has neither debit nor credit")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "transaction_id": "TX001",
                "transaction_date": "2024-01-05",
                "account_code": "6100",
                "account_name": "Marketing",
                "debit": "500.00",
                "credit": "0.00",
                "description": "Meta Ads - January",
                "vendor": "Meta Platforms",
                "memo": "Campaign XYZ",
                "department": "Marketing",
                "location": "Online"
            }
        }


class TrialBalanceEntry(BaseModel):
    """
    Single entry in the Trial Balance.
    
    Shows the balance of each account at a specific date.
    """
    
    account_code: str = Field(
        ...,
        description="Account code"
    )
    account_name: str = Field(
        ...,
        description="Account name"
    )
    debit_balance: Decimal = Field(
        default=Decimal("0.00"),
        ge=0,
        description="Debit balance"
    )
    credit_balance: Decimal = Field(
        default=Decimal("0.00"),
        ge=0,
        description="Credit balance"
    )
    as_of_date: date = Field(
        ...,
        description="Balance sheet date"
    )
    
    @field_validator('debit_balance', 'credit_balance')
    @classmethod
    def validate_amounts(cls, v: Decimal) -> Decimal:
        """Ensure amounts have at most 2 decimal places."""
        if v.as_tuple().exponent < -2:
            raise ValueError("Amounts must have at most 2 decimal places")
        return v.quantize(Decimal("0.01"))
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "account_code": "1100",
                "account_name": "Cash",
                "debit_balance": "50000.00",
                "credit_balance": "0.00",
                "as_of_date": "2024-12-31"
            }
        }


class ChartOfAccounts(BaseModel):
    """
    Complete Chart of Accounts for a company.
    """
    
    company_id: Optional[str] = Field(
        None,
        description="Company identifier"
    )
    entries: list[ChartOfAccountsEntry] = Field(
        ...,
        min_length=1,
        description="All accounts in the chart"
    )
    created_at: date = Field(
        ...,
        description="Date the COA was imported/created"
    )
    
    def get_account(self, account_code: str) -> Optional[ChartOfAccountsEntry]:
        """Get an account by its code."""
        for entry in self.entries:
            if entry.account_code == account_code:
                return entry
        return None
    
    def get_accounts_by_type(self, account_type: AccountType) -> list[ChartOfAccountsEntry]:
        """Get all accounts of a specific type."""
        return [entry for entry in self.entries if entry.account_type == account_type]
    
    def validate_hierarchy(self) -> list[str]:
        """Validate that all parent accounts exist. Returns list of errors."""
        errors = []
        account_codes = {entry.account_code for entry in self.entries}
        
        for entry in self.entries:
            if entry.parent_account and entry.parent_account not in account_codes:
                errors.append(
                    f"Account {entry.account_code} references non-existent parent {entry.parent_account}"
                )
        
        return errors


class GeneralLedger(BaseModel):
    """
    Complete General Ledger for a company.
    """
    
    company_id: Optional[str] = Field(
        None,
        description="Company identifier"
    )
    entries: list[GeneralLedgerEntry] = Field(
        ...,
        min_length=1,
        description="All journal entry lines"
    )
    period_start: date = Field(
        ...,
        description="Start date of ledger period"
    )
    period_end: date = Field(
        ...,
        description="End date of ledger period"
    )
    
    def validate_entries(self) -> list[str]:
        """Validate all entries. Returns list of errors."""
        errors = []
        
        for entry in self.entries:
            try:
                entry.validate_entry()
            except ValueError as e:
                errors.append(str(e))
        
        return errors
    
    def validate_double_entry(self) -> dict[str, Decimal]:
        """
        Validate double-entry bookkeeping for each transaction.
        
        Returns dict of transaction_id -> net imbalance.
        Balanced transactions have 0 imbalance.
        """
        from collections import defaultdict
        
        balances: dict[str, Decimal] = defaultdict(Decimal)
        
        for entry in self.entries:
            # Debits are positive, credits are negative
            net = entry.debit - entry.credit
            balances[entry.transaction_id] += net
        
        return dict(balances)
    
    def get_entries_by_account(self, account_code: str) -> list[GeneralLedgerEntry]:
        """Get all entries for a specific account."""
        return [entry for entry in self.entries if entry.account_code == account_code]
    
    def get_entries_by_date_range(
        self, 
        start_date: date, 
        end_date: date
    ) -> list[GeneralLedgerEntry]:
        """Get all entries within a date range."""
        return [
            entry for entry in self.entries 
            if start_date <= entry.transaction_date <= end_date
        ]


class TrialBalance(BaseModel):
    """
    Complete Trial Balance for a company at a specific date.
    """
    
    company_id: Optional[str] = Field(
        None,
        description="Company identifier"
    )
    entries: list[TrialBalanceEntry] = Field(
        ...,
        min_length=1,
        description="All account balances"
    )
    as_of_date: date = Field(
        ...,
        description="Balance date"
    )
    
    def is_balanced(self, tolerance: Decimal = Decimal("1.00")) -> bool:
        """
        Check if trial balance is balanced (total debits = total credits).
        
        Args:
            tolerance: Maximum acceptable difference (default $1.00 for rounding)
        
        Returns:
            True if balanced within tolerance
        """
        total_debits = sum(entry.debit_balance for entry in self.entries)
        total_credits = sum(entry.credit_balance for entry in self.entries)
        
        imbalance = abs(total_debits - total_credits)
        return imbalance <= tolerance
    
    def get_imbalance(self) -> Decimal:
        """Get the current imbalance (debits - credits)."""
        total_debits = sum(entry.debit_balance for entry in self.entries)
        total_credits = sum(entry.credit_balance for entry in self.entries)
        return total_debits - total_credits
    
    def get_total_debits(self) -> Decimal:
        """Get total of all debit balances."""
        return sum(entry.debit_balance for entry in self.entries)
    
    def get_total_credits(self) -> Decimal:
        """Get total of all credit balances."""
        return sum(entry.credit_balance for entry in self.entries)
