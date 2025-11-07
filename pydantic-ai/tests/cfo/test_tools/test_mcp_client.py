"""
Unit tests for MCP client with Supabase backend.

Tests cover:
- Successful context storage and retrieval
- Authentication errors
- Connection errors
- Timeout handling
- Server errors
- Graceful degradation workflows
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Dict, Any

from agents.cfo.tools.mcp_client import (
    store_analysis_context,
    retrieve_related_analyses,
    save_forecast_for_comparison,
    check_connection,
    MCPClientError,
    MCPConnectionError,
    MCPAuthenticationError,
)


# ============================================
# Test Fixtures
# ============================================


@pytest.fixture
def supabase_url() -> str:
    """Test Supabase URL."""
    return "https://test-project.supabase.co"


@pytest.fixture
def service_key() -> str:
    """Test service key (mock JWT)."""
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.key"


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client with common methods."""
    mock_client = MagicMock()
    
    # Setup table() method chain
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    
    # Setup common query methods
    mock_table.upsert.return_value = mock_table
    mock_table.select.return_value = mock_table
    mock_table.filter.return_value = mock_table
    mock_table.order.return_value = mock_table
    mock_table.limit.return_value = mock_table
    
    # Setup execute() to return mock response
    mock_response = MagicMock()
    mock_response.data = []
    mock_table.execute.return_value = mock_response
    
    return mock_client


@pytest.fixture
def sample_context_data() -> Dict[str, Any]:
    """Sample analysis context data."""
    return {
        "analysis_type": "revenue_forecast",
        "date_range": {
            "start": "2024-01-01",
            "end": "2024-12-31"
        },
        "key_findings": [
            "Seasonal spike in Q4",
            "Growth rate: 15%"
        ],
        "data_sources": ["shopify", "quickbooks"],
        "metadata": {
            "analyst": "ai_cfo",
            "confidence": 0.85
        }
    }


@pytest.fixture
def sample_forecast_data() -> Dict[str, Any]:
    """Sample forecast data."""
    return {
        "period": "2024-Q2",
        "predictions": {
            "revenue": 1500000,
            "costs": 950000,
            "profit": 550000
        },
        "confidence_intervals": {
            "revenue": {"lower": 1300000, "upper": 1700000}
        }
    }


# ============================================
# Test: store_analysis_context
# ============================================


@pytest.mark.asyncio
async def test_store_analysis_context_success(
    supabase_url,
    service_key,
    mock_supabase_client,
    sample_context_data
):
    """Test successful context storage."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.data = [{
        "analysis_id": "test_analysis_001",
        "context_data": sample_context_data,
        "created_at": datetime.now(timezone.utc).isoformat()
    }]
    mock_supabase_client.table().upsert().execute.return_value = mock_response
    
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        return_value=mock_supabase_client
    ):
        result = await store_analysis_context(
            supabase_url=supabase_url,
            service_key=service_key,
            analysis_id="test_analysis_001",
            context_data=sample_context_data
        )
    
    assert result["status"] == "success"
    assert result["analysis_id"] == "test_analysis_001"
    assert "record" in result
    assert result["record"]["context_data"] == sample_context_data


@pytest.mark.asyncio
async def test_store_analysis_context_authentication_error(
    supabase_url,
    service_key,
    sample_context_data
):
    """Test authentication error during context storage."""
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        side_effect=MCPAuthenticationError("Invalid API key")
    ):
        with pytest.raises(MCPAuthenticationError):
            await store_analysis_context(
                supabase_url=supabase_url,
                service_key="invalid_key",
                analysis_id="test_analysis_001",
                context_data=sample_context_data
            )


@pytest.mark.asyncio
async def test_store_analysis_context_connection_error(
    supabase_url,
    service_key,
    sample_context_data
):
    """Test connection error during context storage."""
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        side_effect=MCPConnectionError("Connection refused")
    ):
        with pytest.raises(MCPConnectionError):
            await store_analysis_context(
                supabase_url=supabase_url,
                service_key=service_key,
                analysis_id="test_analysis_001",
                context_data=sample_context_data
            )


@pytest.mark.asyncio
async def test_store_analysis_context_storage_error(
    supabase_url,
    service_key,
    mock_supabase_client,
    sample_context_data
):
    """Test storage operation error."""
    # Make execute() raise an exception
    mock_supabase_client.table().upsert().execute.side_effect = Exception(
        "Database error"
    )
    
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        return_value=mock_supabase_client
    ):
        with pytest.raises(MCPClientError):
            await store_analysis_context(
                supabase_url=supabase_url,
                service_key=service_key,
                analysis_id="test_analysis_001",
                context_data=sample_context_data
            )


# ============================================
# Test: retrieve_related_analyses
# ============================================


@pytest.mark.asyncio
async def test_retrieve_related_analyses_success(
    supabase_url,
    service_key,
    mock_supabase_client
):
    """Test successful retrieval of related analyses."""
    # Setup mock response with multiple analyses
    mock_response = MagicMock()
    mock_response.data = [
        {
            "analysis_id": "revenue_2024_q1",
            "context_data": {
                "analysis_type": "revenue_forecast",
                "key_findings": ["Revenue growth", "Seasonal patterns"]
            },
            "created_at": "2024-01-15T10:00:00"
        },
        {
            "analysis_id": "revenue_2023_q4",
            "context_data": {
                "analysis_type": "revenue_forecast",
                "key_findings": ["Holiday sales spike"]
            },
            "created_at": "2023-10-15T10:00:00"
        }
    ]
    mock_supabase_client.table().select().order().limit().execute.return_value = mock_response
    
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        return_value=mock_supabase_client
    ):
        result = await retrieve_related_analyses(
            supabase_url=supabase_url,
            service_key=service_key,
            query="revenue forecasts with seasonal patterns",
            limit=5
        )
    
    assert result["status"] == "success"
    assert result["total_results"] == 2
    assert len(result["analyses"]) == 2
    assert all("relevance_score" in a for a in result["analyses"])
    assert all("analysis_id" in a for a in result["analyses"])


@pytest.mark.asyncio
async def test_retrieve_related_analyses_with_filters(
    supabase_url,
    service_key,
    mock_supabase_client
):
    """Test retrieval with filters applied."""
    mock_response = MagicMock()
    mock_response.data = [{
        "analysis_id": "revenue_2024_q1",
        "context_data": {"analysis_type": "revenue_forecast"},
        "created_at": "2024-01-15T10:00:00"
    }]
    
    # Setup filter chain
    mock_table = mock_supabase_client.table()
    mock_table.select().filter().order().limit().execute.return_value = mock_response
    
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        return_value=mock_supabase_client
    ):
        result = await retrieve_related_analyses(
            supabase_url=supabase_url,
            service_key=service_key,
            query="revenue forecasts",
            filters={"analysis_type": "revenue_forecast"},
            limit=5
        )
    
    assert result["status"] == "success"
    assert result["total_results"] >= 0


@pytest.mark.asyncio
async def test_retrieve_related_analyses_empty_results(
    supabase_url,
    service_key,
    mock_supabase_client
):
    """Test retrieval when no matches found."""
    mock_response = MagicMock()
    mock_response.data = []
    mock_supabase_client.table().select().order().limit().execute.return_value = mock_response
    
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        return_value=mock_supabase_client
    ):
        result = await retrieve_related_analyses(
            supabase_url=supabase_url,
            service_key=service_key,
            query="nonexistent analysis type",
            limit=5
        )
    
    assert result["status"] == "success"
    assert result["total_results"] == 0
    assert result["analyses"] == []


@pytest.mark.asyncio
async def test_retrieve_related_analyses_connection_error(
    supabase_url,
    service_key
):
    """Test retrieval with connection error."""
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        side_effect=MCPConnectionError("Connection refused")
    ):
        with pytest.raises(MCPConnectionError):
            await retrieve_related_analyses(
                supabase_url=supabase_url,
                service_key=service_key,
                query="test query"
            )


# ============================================
# Test: save_forecast_for_comparison
# ============================================


@pytest.mark.asyncio
async def test_save_forecast_success(
    supabase_url,
    service_key,
    mock_supabase_client,
    sample_forecast_data
):
    """Test successful forecast save."""
    mock_response = MagicMock()
    mock_response.data = [{
        "forecast_id": "revenue_2024_q2",
        "forecast_data": sample_forecast_data,
        "created_at": datetime.now(timezone.utc).isoformat()
    }]
    mock_supabase_client.table().upsert().execute.return_value = mock_response
    
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        return_value=mock_supabase_client
    ):
        result = await save_forecast_for_comparison(
            supabase_url=supabase_url,
            service_key=service_key,
            forecast_id="revenue_2024_q2",
            forecast_data=sample_forecast_data,
            forecast_metadata={"model": "prophet"}
        )
    
    assert result["status"] == "success"
    assert result["forecast_id"] == "revenue_2024_q2"
    assert "record" in result


@pytest.mark.asyncio
async def test_save_forecast_without_metadata(
    supabase_url,
    service_key,
    mock_supabase_client,
    sample_forecast_data
):
    """Test forecast save without metadata."""
    mock_response = MagicMock()
    mock_response.data = [{
        "forecast_id": "revenue_2024_q2",
        "forecast_data": sample_forecast_data,
        "forecast_metadata": {},
        "created_at": datetime.now(timezone.utc).isoformat()
    }]
    mock_supabase_client.table().upsert().execute.return_value = mock_response
    
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        return_value=mock_supabase_client
    ):
        result = await save_forecast_for_comparison(
            supabase_url=supabase_url,
            service_key=service_key,
            forecast_id="revenue_2024_q2",
            forecast_data=sample_forecast_data
        )
    
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_save_forecast_storage_error(
    supabase_url,
    service_key,
    mock_supabase_client,
    sample_forecast_data
):
    """Test forecast save with storage error."""
    mock_supabase_client.table().upsert().execute.side_effect = Exception(
        "Database error"
    )
    
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        return_value=mock_supabase_client
    ):
        with pytest.raises(MCPClientError):
            await save_forecast_for_comparison(
                supabase_url=supabase_url,
                service_key=service_key,
                forecast_id="revenue_2024_q2",
                forecast_data=sample_forecast_data
            )


# ============================================
# Test: check_connection
# ============================================


@pytest.mark.asyncio
async def test_check_connection_success(
    supabase_url,
    service_key,
    mock_supabase_client
):
    """Test successful connection check."""
    mock_response = MagicMock()
    mock_response.data = []
    mock_supabase_client.table().select().limit().execute.return_value = mock_response
    
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        return_value=mock_supabase_client
    ):
        result = await check_connection(
            supabase_url=supabase_url,
            service_key=service_key
        )
    
    assert result["status"] == "connected"
    assert result["supabase_url"] == supabase_url


@pytest.mark.asyncio
async def test_check_connection_auth_error(
    supabase_url,
    service_key
):
    """Test connection check with authentication error."""
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        side_effect=MCPAuthenticationError("Invalid key")
    ):
        result = await check_connection(
            supabase_url=supabase_url,
            service_key="invalid_key"
        )
    
    assert result["status"] == "error"
    assert result["error_type"] == "authentication"


@pytest.mark.asyncio
async def test_check_connection_connection_error(
    supabase_url,
    service_key
):
    """Test connection check with connection error."""
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        side_effect=MCPConnectionError("Connection refused")
    ):
        result = await check_connection(
            supabase_url=supabase_url,
            service_key=service_key
        )
    
    assert result["status"] == "error"
    assert result["error_type"] == "connection"


# ============================================
# Integration Tests
# ============================================


@pytest.mark.asyncio
async def test_graceful_degradation_workflow(
    supabase_url,
    service_key,
    sample_context_data
):
    """Test that system degrades gracefully when Supabase is unavailable."""
    # Simulate Supabase being unavailable
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        side_effect=MCPConnectionError("Service unavailable")
    ):
        # Check connection should return error status
        conn_result = await check_connection(supabase_url, service_key)
        assert conn_result["status"] == "error"
        
        # Operations should raise appropriate exceptions
        with pytest.raises(MCPConnectionError):
            await store_analysis_context(
                supabase_url=supabase_url,
                service_key=service_key,
                analysis_id="test_001",
                context_data=sample_context_data
            )


@pytest.mark.asyncio
async def test_end_to_end_context_workflow(
    supabase_url,
    service_key,
    mock_supabase_client,
    sample_context_data
):
    """Test complete workflow: store context, then retrieve it."""
    # Setup mock for storage
    store_response = MagicMock()
    store_response.data = [{
        "analysis_id": "test_analysis_001",
        "context_data": sample_context_data,
        "created_at": datetime.now(timezone.utc).isoformat()
    }]
    
    # Setup mock for retrieval
    retrieve_response = MagicMock()
    retrieve_response.data = [store_response.data[0]]
    
    with patch(
        "agents.cfo.tools.mcp_client._create_supabase_client",
        return_value=mock_supabase_client
    ):
        # First, store context
        mock_supabase_client.table().upsert().execute.return_value = store_response
        store_result = await store_analysis_context(
            supabase_url=supabase_url,
            service_key=service_key,
            analysis_id="test_analysis_001",
            context_data=sample_context_data
        )
        assert store_result["status"] == "success"
        
        # Then, retrieve it
        mock_supabase_client.table().select().order().limit().execute.return_value = retrieve_response
        retrieve_result = await retrieve_related_analyses(
            supabase_url=supabase_url,
            service_key=service_key,
            query="revenue forecast"
        )
        assert retrieve_result["status"] == "success"
        assert len(retrieve_result["analyses"]) > 0
