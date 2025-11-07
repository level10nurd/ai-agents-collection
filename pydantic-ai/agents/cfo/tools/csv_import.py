"""
CSV Import Tool

Parses financial data from CSV files (Chart of Accounts, General Ledger, Trial Balance).
Validates data structure and returns standardized Pydantic models.
"""

import csv
import pandas as pd
from pathlib import Path
from typing import Union
from datetime import datetime, date
from decimal import Decimal, InvalidOperation

from ..models.ledger import (
    ChartOfAccounts,
    ChartOfAccountsEntry,
    GeneralLedger,
    GeneralLedgerEntry,
    TrialBalance,
    TrialBalanceEntry,
    AccountType,
    NormalBalance,
)


class CSVImportError(Exception):
    """Raised when CSV import fails validation."""
    pass


def parse_chart_of_accounts(
    file_path: Union[str, Path],
    company_id: str | None = None
) -> ChartOfAccounts:
    """
    Parse Chart of Accounts from CSV file.
    
    Expected CSV columns:
    - account_code (required): Unique account identifier
    - account_name (required): Account name
    - account_type (required): Asset, Liability, Equity, Revenue, Expense, or COGS
    - account_subtype (optional): More specific categorization
    - parent_account (optional): Parent account code for hierarchy
    - is_active (optional): true/false, defaults to true
    - normal_balance (required): Debit or Credit
    - description (optional): Additional notes
    
    Args:
        file_path: Path to CSV file
        company_id: Optional company identifier
    
    Returns:
        ChartOfAccounts object with validated entries
    
    Raises:
        CSVImportError: If file is invalid or data fails validation
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise CSVImportError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise CSVImportError(f"Failed to read CSV: {e}")
    
    # Validate required columns
    required_columns = ['account_code', 'account_name', 'account_type', 'normal_balance']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise CSVImportError(f"Missing required columns: {missing_columns}")
    
    # Parse entries
    entries = []
    errors = []
    
    for idx, row in df.iterrows():
        try:
            # Handle is_active conversion
            is_active = True
            if 'is_active' in row and pd.notna(row['is_active']):
                is_active_str = str(row['is_active']).strip().lower()
                is_active = is_active_str in ('true', '1', 'yes', 't', 'y')
            
            # Handle parent_account conversion (pandas may convert to float)
            parent_account = None
            if 'parent_account' in row and pd.notna(row['parent_account']):
                parent_str = str(row['parent_account']).strip()
                # Remove .0 if pandas converted to float
                if parent_str.endswith('.0'):
                    parent_str = parent_str[:-2]
                parent_account = parent_str if parent_str else None
            
            entry = ChartOfAccountsEntry(
                account_code=str(row['account_code']).strip(),
                account_name=str(row['account_name']).strip(),
                account_type=AccountType(str(row['account_type']).strip()),
                account_subtype=str(row['account_subtype']).strip() if 'account_subtype' in row and pd.notna(row['account_subtype']) else None,
                parent_account=parent_account,
                is_active=is_active,
                normal_balance=NormalBalance(str(row['normal_balance']).strip()),
                description=str(row['description']).strip() if 'description' in row and pd.notna(row['description']) else None,
            )
            entries.append(entry)
        except Exception as e:
            errors.append(f"Row {idx + 2}: {e}")  # +2 for header and 0-indexing
    
    if errors:
        raise CSVImportError(f"Validation errors:\n" + "\n".join(errors))
    
    if not entries:
        raise CSVImportError("No valid entries found in CSV")
    
    # Create ChartOfAccounts
    coa = ChartOfAccounts(
        company_id=company_id,
        entries=entries,
        created_at=date.today()
    )
    
    # Validate hierarchy
    hierarchy_errors = coa.validate_hierarchy()
    if hierarchy_errors:
        raise CSVImportError(f"Hierarchy validation errors:\n" + "\n".join(hierarchy_errors))
    
    return coa


def parse_general_ledger(
    file_path: Union[str, Path],
    period_start: date | None = None,
    period_end: date | None = None,
    company_id: str | None = None
) -> GeneralLedger:
    """
    Parse General Ledger from CSV file.
    
    Expected CSV columns:
    - transaction_id (required): Unique identifier for journal entry
    - date or transaction_date (required): Transaction date (YYYY-MM-DD)
    - account_code (required): Account code
    - account_name (required): Account name
    - debit (required): Debit amount (0 if credit entry)
    - credit (required): Credit amount (0 if debit entry)
    - description (required): Transaction description
    - vendor (optional): Vendor/customer name
    - memo (optional): Additional notes
    - department or class (optional): Department/class
    - location (optional): Location
    
    Args:
        file_path: Path to CSV file
        period_start: Start date of ledger period (inferred if not provided)
        period_end: End date of ledger period (inferred if not provided)
        company_id: Optional company identifier
    
    Returns:
        GeneralLedger object with validated entries
    
    Raises:
        CSVImportError: If file is invalid or data fails validation
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise CSVImportError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise CSVImportError(f"Failed to read CSV: {e}")
    
    # Normalize column names (handle variations)
    df.columns = df.columns.str.strip().str.lower()
    
    # Map column name variations
    if 'date' in df.columns and 'transaction_date' not in df.columns:
        df['transaction_date'] = df['date']
    if 'class' in df.columns and 'department' not in df.columns:
        df['department'] = df['class']
    
    # Validate required columns
    required_columns = ['transaction_id', 'transaction_date', 'account_code', 
                        'account_name', 'debit', 'credit', 'description']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise CSVImportError(f"Missing required columns: {missing_columns}")
    
    # Parse entries
    entries = []
    errors = []
    dates = []
    
    for idx, row in df.iterrows():
        try:
            # Parse date
            try:
                trans_date = pd.to_datetime(row['transaction_date']).date()
                dates.append(trans_date)
            except Exception as e:
                raise ValueError(f"Invalid date format: {row['transaction_date']}")
            
            # Parse amounts
            try:
                debit = Decimal(str(row['debit']).replace('$', '').replace(',', '').strip() or "0")
                credit = Decimal(str(row['credit']).replace('$', '').replace(',', '').strip() or "0")
            except (InvalidOperation, ValueError) as e:
                raise ValueError(f"Invalid amount: {e}")
            
            entry = GeneralLedgerEntry(
                transaction_id=str(row['transaction_id']).strip(),
                transaction_date=trans_date,
                account_code=str(row['account_code']).strip(),
                account_name=str(row['account_name']).strip(),
                debit=debit,
                credit=credit,
                description=str(row['description']).strip(),
                vendor=str(row['vendor']).strip() if 'vendor' in row and pd.notna(row['vendor']) else None,
                memo=str(row['memo']).strip() if 'memo' in row and pd.notna(row['memo']) else None,
                department=str(row['department']).strip() if 'department' in row and pd.notna(row['department']) else None,
                location=str(row['location']).strip() if 'location' in row and pd.notna(row['location']) else None,
            )
            entries.append(entry)
        except Exception as e:
            errors.append(f"Row {idx + 2}: {e}")
    
    if errors:
        raise CSVImportError(f"Validation errors:\n" + "\n".join(errors))
    
    if not entries:
        raise CSVImportError("No valid entries found in CSV")
    
    # Infer period if not provided
    if not period_start:
        period_start = min(dates)
    if not period_end:
        period_end = max(dates)
    
    # Create GeneralLedger
    gl = GeneralLedger(
        company_id=company_id,
        entries=entries,
        period_start=period_start,
        period_end=period_end
    )
    
    # Validate entries
    entry_errors = gl.validate_entries()
    if entry_errors:
        raise CSVImportError(f"Entry validation errors:\n" + "\n".join(entry_errors))
    
    # Validate double-entry bookkeeping
    imbalances = gl.validate_double_entry()
    unbalanced = {
        tid: balance for tid, balance in imbalances.items() 
        if abs(balance) > Decimal("0.01")  # Allow 1 cent rounding error
    }
    
    if unbalanced:
        error_msg = "Unbalanced transactions:\n"
        for tid, balance in list(unbalanced.items())[:10]:  # Show first 10
            error_msg += f"  {tid}: ${balance}\n"
        if len(unbalanced) > 10:
            error_msg += f"  ... and {len(unbalanced) - 10} more"
        raise CSVImportError(error_msg)
    
    return gl


def parse_trial_balance(
    file_path: Union[str, Path],
    as_of_date: date | None = None,
    company_id: str | None = None
) -> TrialBalance:
    """
    Parse Trial Balance from CSV file.
    
    Expected CSV columns:
    - account_code (required): Account code
    - account_name (required): Account name
    - debit_balance (required): Debit balance (0 if credit balance)
    - credit_balance (required): Credit balance (0 if debit balance)
    - as_of_date (optional): Balance date (can be provided as argument)
    
    Args:
        file_path: Path to CSV file
        as_of_date: Balance date (uses CSV column if not provided)
        company_id: Optional company identifier
    
    Returns:
        TrialBalance object with validated entries
    
    Raises:
        CSVImportError: If file is invalid or data fails validation
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise CSVImportError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise CSVImportError(f"Failed to read CSV: {e}")
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Validate required columns
    required_columns = ['account_code', 'account_name', 'debit_balance', 'credit_balance']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise CSVImportError(f"Missing required columns: {missing_columns}")
    
    # Determine as_of_date
    if not as_of_date:
        if 'as_of_date' in df.columns:
            # Use first non-null as_of_date from CSV
            date_values = df['as_of_date'].dropna()
            if len(date_values) > 0:
                as_of_date = pd.to_datetime(date_values.iloc[0]).date()
            else:
                as_of_date = date.today()
        else:
            as_of_date = date.today()
    
    # Parse entries
    entries = []
    errors = []
    
    for idx, row in df.iterrows():
        try:
            # Parse balances
            try:
                debit_balance = Decimal(str(row['debit_balance']).replace('$', '').replace(',', '').strip() or "0")
                credit_balance = Decimal(str(row['credit_balance']).replace('$', '').replace(',', '').strip() or "0")
            except (InvalidOperation, ValueError) as e:
                raise ValueError(f"Invalid balance amount: {e}")
            
            # Parse as_of_date from row if present
            entry_date = as_of_date
            if 'as_of_date' in row and pd.notna(row['as_of_date']):
                try:
                    entry_date = pd.to_datetime(row['as_of_date']).date()
                except Exception:
                    pass  # Use default as_of_date
            
            entry = TrialBalanceEntry(
                account_code=str(row['account_code']).strip(),
                account_name=str(row['account_name']).strip(),
                debit_balance=debit_balance,
                credit_balance=credit_balance,
                as_of_date=entry_date
            )
            entries.append(entry)
        except Exception as e:
            errors.append(f"Row {idx + 2}: {e}")
    
    if errors:
        raise CSVImportError(f"Validation errors:\n" + "\n".join(errors))
    
    if not entries:
        raise CSVImportError("No valid entries found in CSV")
    
    # Create TrialBalance
    tb = TrialBalance(
        company_id=company_id,
        entries=entries,
        as_of_date=as_of_date
    )
    
    # Check if balanced
    if not tb.is_balanced():
        imbalance = tb.get_imbalance()
        total_debits = tb.get_total_debits()
        total_credits = tb.get_total_credits()
        raise CSVImportError(
            f"Trial Balance is not balanced:\n"
            f"  Total Debits:  ${total_debits:,.2f}\n"
            f"  Total Credits: ${total_credits:,.2f}\n"
            f"  Imbalance:     ${imbalance:,.2f}"
        )
    
    return tb


def validate_gl_against_coa(
    general_ledger: GeneralLedger,
    chart_of_accounts: ChartOfAccounts
) -> list[str]:
    """
    Validate that all GL account codes exist in the Chart of Accounts.
    
    Args:
        general_ledger: GeneralLedger to validate
        chart_of_accounts: ChartOfAccounts to check against
    
    Returns:
        List of validation errors (empty if all valid)
    """
    coa_codes = {entry.account_code for entry in chart_of_accounts.entries}
    errors = []
    
    for entry in general_ledger.entries:
        if entry.account_code not in coa_codes:
            errors.append(
                f"Transaction {entry.transaction_id} references unknown account: {entry.account_code}"
            )
    
    return errors


def validate_tb_against_coa(
    trial_balance: TrialBalance,
    chart_of_accounts: ChartOfAccounts
) -> list[str]:
    """
    Validate that all TB account codes exist in the Chart of Accounts.
    
    Args:
        trial_balance: TrialBalance to validate
        chart_of_accounts: ChartOfAccounts to check against
    
    Returns:
        List of validation errors (empty if all valid)
    """
    coa_codes = {entry.account_code for entry in chart_of_accounts.entries}
    errors = []
    
    for entry in trial_balance.entries:
        if entry.account_code not in coa_codes:
            errors.append(
                f"Account code {entry.account_code} not found in Chart of Accounts"
            )
    
    return errors
