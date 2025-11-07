"""
Unit Tests for Supabase Integration Tools

Tests all CRUD operations with mocked Supabase client.
"""

import pytest
from datetime import date, datetime
from unittest.mock import Mock, MagicMock, patch

from agents.cfo.tools.supabase import (
    get_supabase_client,
    save_analysis,
    get_historical_sales,
    save_forecast
)


class TestGetSupabaseClient:
    """Tests for get_supabase_client function"""

    @patch('agents.cfo.tools.supabase.create_client')
    def test_creates_client_with_url_and_key(self, mock_create_client):
        """Should create Supabase client with provided URL and service key"""
        # Arrange
        test_url = "https://test-project.supabase.co"
        test_key = "test-service-key"
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        # Act
        result = get_supabase_client(test_url, test_key)

        # Assert
        mock_create_client.assert_called_once_with(test_url, test_key)
        assert result == mock_client


class TestSaveAnalysis:
    """Tests for save_analysis function"""

    def test_saves_analysis_successfully(self):
        """Should insert analysis data and return inserted record"""
        # Arrange
        mock_client = Mock()
        mock_table = Mock()
        mock_insert = Mock()
        mock_execute = Mock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_execute.data = [{
            "id": "analysis-123",
            "company_id": "company-456",
            "analysis_type": "cash_forecast",
            "analysis_data": {"revenue": 150000},
            "created_at": "2024-01-15T12:00:00"
        }]
        mock_insert.execute.return_value = mock_execute

        company_id = "company-456"
        analysis_type = "cash_forecast"
        analysis_data = {
            "forecast_period": "Q1 2024",
            "revenue": 150000,
            "expenses": 100000
        }

        # Act
        result = save_analysis(mock_client, company_id, analysis_type, analysis_data)

        # Assert
        mock_client.table.assert_called_once_with('cfo_analyses')
        assert mock_table.insert.called
        assert result["id"] == "analysis-123"
        assert result["company_id"] == company_id
        assert result["analysis_type"] == analysis_type

    def test_raises_exception_when_insert_fails(self):
        """Should raise exception if insert returns no data"""
        # Arrange
        mock_client = Mock()
        mock_table = Mock()
        mock_insert = Mock()
        mock_execute = Mock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_execute.data = None  # Simulate failure
        mock_insert.execute.return_value = mock_execute

        # Act & Assert
        with pytest.raises(Exception, match="Failed to save analysis"):
            save_analysis(
                mock_client,
                "company-123",
                "test_analysis",
                {"data": "test"}
            )

    def test_includes_created_at_timestamp(self):
        """Should include created_at timestamp in inserted data"""
        # Arrange
        mock_client = Mock()
        mock_table = Mock()
        mock_insert = Mock()
        mock_execute = Mock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_execute.data = [{"id": "123", "created_at": "2024-01-15T12:00:00+00:00"}]
        mock_insert.execute.return_value = mock_execute

        # Act
        save_analysis(
            mock_client,
            "company-123",
            "test",
            {"key": "value"}
        )

        # Assert
        insert_call_args = mock_table.insert.call_args[0][0]
        assert "created_at" in insert_call_args
        # Verify it's a string timestamp
        assert isinstance(insert_call_args["created_at"], str)


class TestGetHistoricalSales:
    """Tests for get_historical_sales function"""

    def test_queries_sales_data_with_filters(self):
        """Should query sales_data table with company_id and date filters"""
        # Arrange
        mock_client = Mock()
        mock_table = Mock()
        mock_select = Mock()
        mock_eq = Mock()
        mock_gte = Mock()
        mock_lte = Mock()
        mock_order = Mock()
        mock_execute = Mock()

        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.eq.return_value = mock_eq
        mock_eq.gte.return_value = mock_gte
        mock_gte.lte.return_value = mock_lte
        mock_lte.order.return_value = mock_order
        
        mock_execute.data = [
            {"date": "2024-01-01", "revenue": 50000, "units_sold": 100},
            {"date": "2024-01-02", "revenue": 52000, "units_sold": 105}
        ]
        mock_order.execute.return_value = mock_execute

        company_id = "company-123"
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        # Act
        result = get_historical_sales(mock_client, company_id, start_date, end_date)

        # Assert
        mock_client.table.assert_called_once_with('sales_data')
        mock_table.select.assert_called_once_with('date, revenue, units_sold')
        mock_select.eq.assert_called_once_with('company_id', company_id)
        mock_eq.gte.assert_called_once_with('date', '2024-01-01')
        mock_gte.lte.assert_called_once_with('date', '2024-01-31')
        mock_lte.order.assert_called_once_with('date', desc=False)
        assert len(result) == 2
        assert result[0]["revenue"] == 50000

    def test_returns_empty_list_when_no_data(self):
        """Should return empty list when no sales data found"""
        # Arrange
        mock_client = Mock()
        mock_table = Mock()
        mock_select = Mock()
        mock_eq = Mock()
        mock_gte = Mock()
        mock_lte = Mock()
        mock_order = Mock()
        mock_execute = Mock()

        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.eq.return_value = mock_eq
        mock_eq.gte.return_value = mock_gte
        mock_gte.lte.return_value = mock_lte
        mock_lte.order.return_value = mock_order
        mock_execute.data = []
        mock_order.execute.return_value = mock_execute

        # Act
        result = get_historical_sales(
            mock_client,
            "company-123",
            date(2024, 1, 1),
            date(2024, 1, 31)
        )

        # Assert
        assert result == []


class TestSaveForecast:
    """Tests for save_forecast function"""

    def test_saves_forecast_successfully(self):
        """Should insert forecast data and return inserted record"""
        # Arrange
        mock_client = Mock()
        mock_table = Mock()
        mock_insert = Mock()
        mock_execute = Mock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_execute.data = [{
            "id": "forecast-789",
            "company_id": "company-456",
            "forecast_type": "sales",
            "forecast_data": {"predictions": []},
            "created_at": "2024-01-15T12:00:00"
        }]
        mock_insert.execute.return_value = mock_execute

        company_id = "company-456"
        forecast_type = "sales"
        forecast_data = {
            "model": "prophet",
            "predictions": [
                {"date": "2024-04-01", "yhat": 50000}
            ]
        }

        # Act
        result = save_forecast(mock_client, company_id, forecast_type, forecast_data)

        # Assert
        mock_client.table.assert_called_once_with('forecasts')
        assert mock_table.insert.called
        assert result["id"] == "forecast-789"
        assert result["company_id"] == company_id
        assert result["forecast_type"] == forecast_type

    def test_raises_exception_when_insert_fails(self):
        """Should raise exception if insert returns no data"""
        # Arrange
        mock_client = Mock()
        mock_table = Mock()
        mock_insert = Mock()
        mock_execute = Mock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_execute.data = None  # Simulate failure
        mock_insert.execute.return_value = mock_execute

        # Act & Assert
        with pytest.raises(Exception, match="Failed to save forecast"):
            save_forecast(
                mock_client,
                "company-123",
                "test_forecast",
                {"data": "test"}
            )

    def test_stores_prophet_forecast_data(self):
        """Should store complete Prophet forecast with predictions and metrics"""
        # Arrange
        mock_client = Mock()
        mock_table = Mock()
        mock_insert = Mock()
        mock_execute = Mock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_execute.data = [{
            "id": "forecast-123",
            "forecast_data": {
                "model": "prophet",
                "predictions": [
                    {"date": "2024-04-01", "yhat": 50000, "yhat_lower": 45000, "yhat_upper": 55000}
                ],
                "metrics": {"mae": 2500, "mape": 0.05}
            }
        }]
        mock_insert.execute.return_value = mock_execute

        forecast_data = {
            "model": "prophet",
            "predictions": [
                {"date": "2024-04-01", "yhat": 50000, "yhat_lower": 45000, "yhat_upper": 55000}
            ],
            "metrics": {"mae": 2500, "mape": 0.05}
        }

        # Act
        result = save_forecast(mock_client, "company-123", "sales", forecast_data)

        # Assert
        assert result["forecast_data"]["model"] == "prophet"
        assert len(result["forecast_data"]["predictions"]) == 1
        assert "metrics" in result["forecast_data"]


class TestIntegration:
    """Integration tests with mock Supabase client"""

    def test_full_workflow_create_and_retrieve(self):
        """Should be able to save analysis and retrieve historical data"""
        # Arrange
        mock_client = Mock()
        
        # Mock save_analysis
        mock_table_analyses = Mock()
        mock_insert_analyses = Mock()
        mock_execute_analyses = Mock()
        mock_execute_analyses.data = [{
            "id": "analysis-123",
            "company_id": "company-456",
            "analysis_type": "cash_forecast",
            "analysis_data": {"revenue": 150000}
        }]
        mock_insert_analyses.execute.return_value = mock_execute_analyses
        mock_table_analyses.insert.return_value = mock_insert_analyses
        
        # Mock get_historical_sales
        mock_table_sales = Mock()
        mock_select = Mock()
        mock_eq = Mock()
        mock_gte = Mock()
        mock_lte = Mock()
        mock_order = Mock()
        mock_execute_sales = Mock()
        mock_execute_sales.data = [
            {"date": "2024-01-01", "revenue": 50000, "units_sold": 100}
        ]
        mock_order.execute.return_value = mock_execute_sales
        mock_lte.order.return_value = mock_order
        mock_gte.lte.return_value = mock_lte
        mock_eq.gte.return_value = mock_gte
        mock_select.eq.return_value = mock_eq
        mock_table_sales.select.return_value = mock_select
        
        # Setup table routing
        def table_router(name):
            if name == 'cfo_analyses':
                return mock_table_analyses
            elif name == 'sales_data':
                return mock_table_sales
            return Mock()
        
        mock_client.table.side_effect = table_router

        # Act
        analysis_result = save_analysis(
            mock_client,
            "company-456",
            "cash_forecast",
            {"revenue": 150000}
        )
        
        sales_result = get_historical_sales(
            mock_client,
            "company-456",
            date(2024, 1, 1),
            date(2024, 1, 31)
        )

        # Assert
        assert analysis_result["id"] == "analysis-123"
        assert len(sales_result) == 1
        assert sales_result[0]["revenue"] == 50000
