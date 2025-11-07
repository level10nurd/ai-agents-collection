"""
Test CSV Import Tool

Tests for parsing Chart of Accounts, General Ledger, and Trial Balance CSV files.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import date
from decimal import Decimal
from agents.cfo.tools.csv_import import (
    parse_chart_of_accounts,
    parse_general_ledger,
    parse_trial_balance,
    validate_gl_against_coa,
    validate_tb_against_coa,
    CSVImportError,
)
from agents.cfo.models.ledger import AccountType, NormalBalance


# Sample CSV data
SAMPLE_COA_CSV = """account_code,account_name,account_type,account_subtype,parent_account,is_active,normal_balance,description
1000,Assets,Asset,,,true,Debit,All assets
1100,Cash,Asset,Bank,1000,true,Debit,Cash and cash equivalents
1200,Accounts Receivable,Asset,AccountsReceivable,1000,true,Debit,Customer invoices
2000,Liabilities,Liability,,,true,Credit,All liabilities
2100,Accounts Payable,Liability,AccountsPayable,2000,true,Credit,Vendor bills
3000,Equity,Equity,,,true,Credit,Shareholder equity
4000,Revenue,Revenue,,,true,Credit,All revenue
4100,Product Sales,Revenue,ProductSales,4000,true,Credit,Product revenue
5000,Cost of Goods Sold,COGS,,,true,Debit,COGS
6000,Operating Expenses,Expense,,,true,Debit,Operating expenses
6100,Marketing,Expense,AdvertisingExpense,6000,true,Debit,Marketing and advertising
6200,Salaries,Expense,Payroll,6000,true,Debit,Employee salaries
"""

SAMPLE_GL_CSV = """transaction_id,date,account_code,account_name,debit,credit,description,vendor,memo,department,location
TX001,2024-01-05,6100,Marketing,500.00,0.00,Meta Ads - January,Meta Platforms,Campaign XYZ,Marketing,Online
TX001,2024-01-05,1100,Cash,0.00,500.00,Meta Ads - January,Meta Platforms,Campaign XYZ,Marketing,Online
TX002,2024-01-10,1200,Accounts Receivable,2500.00,0.00,Invoice #1001,ACME Corp,Wine chiller order,Sales,
TX002,2024-01-10,4100,Product Sales,0.00,2500.00,Invoice #1001,ACME Corp,Wine chiller order,Sales,
TX003,2024-01-15,1100,Cash,1000.00,0.00,Payment received,Customer ABC,Invoice #999,Sales,
TX003,2024-01-15,1200,Accounts Receivable,0.00,1000.00,Payment received,Customer ABC,Invoice #999,Sales,
"""

SAMPLE_TB_CSV = """account_code,account_name,debit_balance,credit_balance,as_of_date
1100,Cash,50000.00,0.00,2024-12-31
1200,Accounts Receivable,25000.00,0.00,2024-12-31
2100,Accounts Payable,0.00,15000.00,2024-12-31
4100,Product Sales,0.00,500000.00,2024-12-31
5000,Cost of Goods Sold,250000.00,0.00,2024-12-31
6100,Marketing,75000.00,0.00,2024-12-31
6200,Salaries,115000.00,0.00,2024-12-31
"""


class TestParseChartOfAccounts:
    """Tests for parse_chart_of_accounts function."""
    
    def test_parse_valid_coa(self, tmp_path):
        """Test parsing a valid Chart of Accounts CSV."""
        csv_file = tmp_path / "coa.csv"
        csv_file.write_text(SAMPLE_COA_CSV)
        
        coa = parse_chart_of_accounts(csv_file, company_id="TEST001")
        
        assert coa.company_id == "TEST001"
        assert len(coa.entries) == 12
        assert coa.created_at == date.today()
        
        # Check specific entry
        cash_account = coa.get_account("1100")
        assert cash_account is not None
        assert cash_account.account_name == "Cash"
        assert cash_account.account_type == AccountType.ASSET
        assert cash_account.normal_balance == NormalBalance.DEBIT
        assert cash_account.parent_account == "1000"
    
    def test_get_accounts_by_type(self, tmp_path):
        """Test filtering accounts by type."""
        csv_file = tmp_path / "coa.csv"
        csv_file.write_text(SAMPLE_COA_CSV)
        
        coa = parse_chart_of_accounts(csv_file)
        
        revenue_accounts = coa.get_accounts_by_type(AccountType.REVENUE)
        assert len(revenue_accounts) == 2
        assert all(acc.account_type == AccountType.REVENUE for acc in revenue_accounts)
    
    def test_missing_required_columns(self, tmp_path):
        """Test error when required columns are missing."""
        csv_file = tmp_path / "coa_bad.csv"
        csv_file.write_text("account_code,account_name\n1000,Assets\n")
        
        with pytest.raises(CSVImportError, match="Missing required columns"):
            parse_chart_of_accounts(csv_file)
    
    def test_invalid_account_type(self, tmp_path):
        """Test error with invalid account type."""
        csv_file = tmp_path / "coa_bad.csv"
        csv_file.write_text(
            "account_code,account_name,account_type,normal_balance\n"
            "1000,Assets,InvalidType,Debit\n"
        )
        
        with pytest.raises(CSVImportError, match="Validation errors"):
            parse_chart_of_accounts(csv_file)
    
    def test_invalid_hierarchy(self, tmp_path):
        """Test error when parent account doesn't exist."""
        csv_file = tmp_path / "coa_bad.csv"
        csv_file.write_text(
            "account_code,account_name,account_type,parent_account,normal_balance\n"
            "1100,Cash,Asset,9999,Debit\n"
        )
        
        with pytest.raises(CSVImportError, match="Hierarchy validation"):
            parse_chart_of_accounts(csv_file)
    
    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(CSVImportError, match="File not found"):
            parse_chart_of_accounts("/nonexistent/file.csv")


class TestParseGeneralLedger:
    """Tests for parse_general_ledger function."""
    
    def test_parse_valid_gl(self, tmp_path):
        """Test parsing a valid General Ledger CSV."""
        csv_file = tmp_path / "gl.csv"
        csv_file.write_text(SAMPLE_GL_CSV)
        
        gl = parse_general_ledger(
            csv_file,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 1, 31),
            company_id="TEST001"
        )
        
        assert gl.company_id == "TEST001"
        assert len(gl.entries) == 6  # 3 transactions × 2 entries each
        assert gl.period_start == date(2024, 1, 1)
        assert gl.period_end == date(2024, 1, 31)
    
    def test_double_entry_validation(self, tmp_path):
        """Test that all transactions balance."""
        csv_file = tmp_path / "gl.csv"
        csv_file.write_text(SAMPLE_GL_CSV)
        
        gl = parse_general_ledger(csv_file)
        
        imbalances = gl.validate_double_entry()
        
        # All transactions should balance (within 1 cent)
        for tid, balance in imbalances.items():
            assert abs(balance) <= Decimal("0.01"), f"Transaction {tid} is unbalanced: {balance}"
    
    def test_unbalanced_transaction(self, tmp_path):
        """Test error when transaction doesn't balance."""
        csv_file = tmp_path / "gl_bad.csv"
        csv_file.write_text(
            "transaction_id,date,account_code,account_name,debit,credit,description\n"
            "TX001,2024-01-05,6100,Marketing,500.00,0.00,Test\n"
            "TX001,2024-01-05,1100,Cash,0.00,400.00,Test\n"  # Doesn't balance!
        )
        
        with pytest.raises(CSVImportError, match="Unbalanced transactions"):
            parse_general_ledger(csv_file)
    
    def test_get_entries_by_account(self, tmp_path):
        """Test filtering entries by account code."""
        csv_file = tmp_path / "gl.csv"
        csv_file.write_text(SAMPLE_GL_CSV)
        
        gl = parse_general_ledger(csv_file)
        
        cash_entries = gl.get_entries_by_account("1100")
        assert len(cash_entries) == 2
        assert all(entry.account_code == "1100" for entry in cash_entries)
    
    def test_get_entries_by_date_range(self, tmp_path):
        """Test filtering entries by date range."""
        csv_file = tmp_path / "gl.csv"
        csv_file.write_text(SAMPLE_GL_CSV)
        
        gl = parse_general_ledger(csv_file)
        
        jan_5_to_10 = gl.get_entries_by_date_range(
            date(2024, 1, 5),
            date(2024, 1, 10)
        )
        assert len(jan_5_to_10) == 4  # TX001 and TX002
    
    def test_missing_required_columns(self, tmp_path):
        """Test error when required columns are missing."""
        csv_file = tmp_path / "gl_bad.csv"
        csv_file.write_text("transaction_id,date,account_code\nTX001,2024-01-05,1100\n")
        
        with pytest.raises(CSVImportError, match="Missing required columns"):
            parse_general_ledger(csv_file)
    
    def test_invalid_date_format(self, tmp_path):
        """Test error with invalid date."""
        csv_file = tmp_path / "gl_bad.csv"
        csv_file.write_text(
            "transaction_id,date,account_code,account_name,debit,credit,description\n"
            "TX001,not-a-date,1100,Cash,100.00,0.00,Test\n"
        )
        
        with pytest.raises(CSVImportError, match="Invalid date format"):
            parse_general_ledger(csv_file)
    
    def test_invalid_amount(self, tmp_path):
        """Test error with invalid amount."""
        csv_file = tmp_path / "gl_bad.csv"
        csv_file.write_text(
            "transaction_id,date,account_code,account_name,debit,credit,description\n"
            "TX001,2024-01-05,1100,Cash,not-a-number,0.00,Test\n"
        )
        
        with pytest.raises(CSVImportError, match="Invalid amount"):
            parse_general_ledger(csv_file)
    
    def test_amount_formatting(self, tmp_path):
        """Test that amounts with $ and commas are parsed correctly."""
        csv_file = tmp_path / "gl.csv"
        csv_file.write_text(
            "transaction_id,date,account_code,account_name,debit,credit,description\n"
            'TX001,2024-01-05,6100,Marketing,"$1,500.00",0.00,Test\n'
            'TX001,2024-01-05,1100,Cash,0.00,"$1,500.00",Test\n'
        )
        
        gl = parse_general_ledger(csv_file)
        
        assert gl.entries[0].debit == Decimal("1500.00")
        assert gl.entries[1].credit == Decimal("1500.00")


class TestParseTrialBalance:
    """Tests for parse_trial_balance function."""
    
    def test_parse_valid_tb(self, tmp_path):
        """Test parsing a valid Trial Balance CSV."""
        csv_file = tmp_path / "tb.csv"
        csv_file.write_text(SAMPLE_TB_CSV)
        
        tb = parse_trial_balance(
            csv_file,
            as_of_date=date(2024, 12, 31),
            company_id="TEST001"
        )
        
        assert tb.company_id == "TEST001"
        assert len(tb.entries) == 7
        assert tb.as_of_date == date(2024, 12, 31)
    
    def test_trial_balance_balanced(self, tmp_path):
        """Test that trial balance debits equal credits."""
        csv_file = tmp_path / "tb.csv"
        csv_file.write_text(SAMPLE_TB_CSV)
        
        tb = parse_trial_balance(csv_file)
        
        assert tb.is_balanced()
        assert abs(tb.get_imbalance()) <= Decimal("1.00")
    
    def test_trial_balance_totals(self, tmp_path):
        """Test trial balance total calculations."""
        csv_file = tmp_path / "tb.csv"
        csv_file.write_text(SAMPLE_TB_CSV)
        
        tb = parse_trial_balance(csv_file)
        
        total_debits = tb.get_total_debits()
        total_credits = tb.get_total_credits()
        
        assert total_debits == Decimal("515000.00")  # 50000 + 25000 + 250000 + 75000 + 115000
        assert total_credits == Decimal("515000.00")  # 15000 + 500000
    
    def test_unbalanced_trial_balance(self, tmp_path):
        """Test error when trial balance doesn't balance."""
        csv_file = tmp_path / "tb_bad.csv"
        csv_file.write_text(
            "account_code,account_name,debit_balance,credit_balance,as_of_date\n"
            "1100,Cash,50000.00,0.00,2024-12-31\n"
            "4100,Revenue,0.00,40000.00,2024-12-31\n"  # Doesn't balance!
        )
        
        with pytest.raises(CSVImportError, match="Trial Balance is not balanced"):
            parse_trial_balance(csv_file)
    
    def test_missing_required_columns(self, tmp_path):
        """Test error when required columns are missing."""
        csv_file = tmp_path / "tb_bad.csv"
        csv_file.write_text("account_code,account_name\n1100,Cash\n")
        
        with pytest.raises(CSVImportError, match="Missing required columns"):
            parse_trial_balance(csv_file)
    
    def test_as_of_date_inference(self, tmp_path):
        """Test that as_of_date is inferred from CSV if not provided."""
        csv_file = tmp_path / "tb.csv"
        csv_file.write_text(SAMPLE_TB_CSV)
        
        tb = parse_trial_balance(csv_file)
        
        # Should use date from CSV
        assert tb.as_of_date == date(2024, 12, 31)


class TestCrossValidation:
    """Tests for cross-validation between COA, GL, and TB."""
    
    def test_validate_gl_against_coa(self, tmp_path):
        """Test that GL entries reference valid COA accounts."""
        coa_file = tmp_path / "coa.csv"
        coa_file.write_text(SAMPLE_COA_CSV)
        
        gl_file = tmp_path / "gl.csv"
        gl_file.write_text(SAMPLE_GL_CSV)
        
        coa = parse_chart_of_accounts(coa_file)
        gl = parse_general_ledger(gl_file)
        
        errors = validate_gl_against_coa(gl, coa)
        
        assert len(errors) == 0  # All accounts should be valid
    
    def test_validate_gl_invalid_account(self, tmp_path):
        """Test error when GL references non-existent account."""
        coa_file = tmp_path / "coa.csv"
        coa_file.write_text(SAMPLE_COA_CSV)
        
        gl_file = tmp_path / "gl.csv"
        gl_file.write_text(
            "transaction_id,date,account_code,account_name,debit,credit,description\n"
            "TX001,2024-01-05,9999,Invalid Account,500.00,0.00,Test\n"
            "TX001,2024-01-05,1100,Cash,0.00,500.00,Test\n"
        )
        
        coa = parse_chart_of_accounts(coa_file)
        gl = parse_general_ledger(gl_file)
        
        errors = validate_gl_against_coa(gl, coa)
        
        assert len(errors) == 1
        assert "9999" in errors[0]
    
    def test_validate_tb_against_coa(self, tmp_path):
        """Test that TB entries reference valid COA accounts."""
        coa_file = tmp_path / "coa.csv"
        coa_file.write_text(SAMPLE_COA_CSV)
        
        tb_file = tmp_path / "tb.csv"
        tb_file.write_text(SAMPLE_TB_CSV)
        
        coa = parse_chart_of_accounts(coa_file)
        tb = parse_trial_balance(tb_file)
        
        errors = validate_tb_against_coa(tb, coa)
        
        assert len(errors) == 0  # All accounts should be valid


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_csv(self, tmp_path):
        """Test error with empty CSV file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("account_code,account_name,account_type,normal_balance\n")
        
        with pytest.raises(CSVImportError, match="No valid entries"):
            parse_chart_of_accounts(csv_file)
    
    def test_csv_with_blank_lines(self, tmp_path):
        """Test that blank lines are handled gracefully."""
        csv_file = tmp_path / "coa.csv"
        csv_file.write_text(
            "account_code,account_name,account_type,normal_balance\n"
            "1100,Cash,Asset,Debit\n"
            "\n"
            "4100,Revenue,Revenue,Credit\n"
        )
        
        coa = parse_chart_of_accounts(csv_file)
        
        assert len(coa.entries) == 2
    
    def test_unicode_in_descriptions(self, tmp_path):
        """Test that Unicode characters are handled correctly."""
        csv_file = tmp_path / "gl.csv"
        csv_file.write_text(
            "transaction_id,date,account_code,account_name,debit,credit,description\n"
            "TX001,2024-01-05,6100,Marketing,500.00,0.00,Café ☕ promotion\n"
            "TX001,2024-01-05,1100,Cash,0.00,500.00,Café ☕ promotion\n",
            encoding='utf-8'
        )
        
        gl = parse_general_ledger(csv_file)
        
        assert "Café" in gl.entries[0].description
        assert "☕" in gl.entries[0].description
    
    def test_very_large_amounts(self, tmp_path):
        """Test handling of very large monetary amounts."""
        csv_file = tmp_path / "gl.csv"
        csv_file.write_text(
            "transaction_id,date,account_code,account_name,debit,credit,description\n"
            "TX001,2024-01-05,1100,Cash,99999999.99,0.00,Test\n"
            "TX001,2024-01-05,4100,Revenue,0.00,99999999.99,Test\n"
        )
        
        gl = parse_general_ledger(csv_file)
        
        assert gl.entries[0].debit == Decimal("99999999.99")
