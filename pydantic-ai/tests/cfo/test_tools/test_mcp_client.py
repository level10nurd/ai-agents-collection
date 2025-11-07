"""
Unit tests for MCP client integration.

Tests cover:
- Successful context storage and retrieval
- Authentication errors
- Connection errors (server unavailable)
- Timeout handling
- Graceful degradation
- Forecast storage

All tests use mocked HTTP client (httpx) to avoid requiring actual MCP server.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import httpx

from agents.cfo.tools.mcp_client import (
    store_analysis_context,
    retrieve_related_analyses,
    save_forecast_for_comparison,
    MCPClientError,
    MCPConnectionError,
    MCPAuthenticationError
)


# Test fixtures
@pytest.fixture
def mcp_url():
    """MCP server base URL for testing."""
    return "http://localhost:8051/mcp"


@pytest.fixture
def api_key():
    """Test API key."""
    return "test_api_key_12345"


@pytest.fixture
def sample_analysis_context():
    """Sample analysis context data."""
    return {
        "analysis_type": "cash_forecast",
        "timestamp": "2024-01-15T10:00:00Z",
        "parameters": {
            "forecast_period": "6_months",
            "confidence_level": 0.95
        },
        "insights": [
            "Cash balance projected to be positive for next 6 months",
            "Peak cash requirement in Q2 due to inventory buildup"
        ],
        "data_sources": ["quickbooks", "shopify"],
        "recommendations": [
            "Maintain current cash reserves",
            "Consider line of credit for Q2 inventory purchase"
        ]
    }


@pytest.fixture
def sample_forecast_data():
    """Sample forecast data for storage."""
    return {
        "forecast_id": "forecast_2024_01_15",
        "forecast_type": "sales",
        "model_used": "prophet",
        "parameters": {
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.05
        },
        "predictions": [
            {"date": "2024-02-01", "value": 50000, "lower": 45000, "upper": 55000},
            {"date": "2024-03-01", "value": 52000, "lower": 47000, "upper": 57000}
        ],
        "confidence_intervals": {
            "level": 0.95
        },
        "created_at": "2024-01-15T10:00:00Z"
    }


# Tests for store_analysis_context
@pytest.mark.asyncio
async def test_store_analysis_context_success(mcp_url, api_key, sample_analysis_context):
    """Test successful context storage."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "analysis_123",
        "status": "stored"
    }
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        result = await store_analysis_context(
            mcp_url=mcp_url,
            api_key=api_key,
            analysis_id="analysis_123",
            context_data=sample_analysis_context
        )
        
        assert result["success"] is True
        assert result["stored_id"] == "analysis_123"
        assert "successfully" in result["message"].lower()
        
        # Verify API call was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args.kwargs["headers"]["Authorization"] == f"Bearer {api_key}"


@pytest.mark.asyncio
async def test_store_analysis_context_missing_fields(mcp_url, api_key):
    """Test context storage with missing required fields."""
    incomplete_context = {
        "analysis_type": "cash_forecast"
        # Missing timestamp
    }
    
    result = await store_analysis_context(
        mcp_url=mcp_url,
        api_key=api_key,
        analysis_id="analysis_123",
        context_data=incomplete_context
    )
    
    assert result["success"] is False
    assert "missing" in result["message"].lower()
    assert "timestamp" in result["message"].lower()


@pytest.mark.asyncio
async def test_store_analysis_context_authentication_error(mcp_url, api_key, sample_analysis_context):
    """Test authentication error handling."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Invalid API key"
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        with pytest.raises(MCPAuthenticationError):
            await store_analysis_context(
                mcp_url=mcp_url,
                api_key=api_key,
                analysis_id="analysis_123",
                context_data=sample_analysis_context
            )


@pytest.mark.asyncio
async def test_store_analysis_context_connection_error(mcp_url, api_key, sample_analysis_context):
    """Test graceful handling when MCP server is unavailable."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client_class.return_value = mock_client
        
        result = await store_analysis_context(
            mcp_url=mcp_url,
            api_key=api_key,
            analysis_id="analysis_123",
            context_data=sample_analysis_context
        )
        
        # Should return unsuccessful but not crash
        assert result["success"] is False
        assert "unavailable" in result["message"].lower()


@pytest.mark.asyncio
async def test_store_analysis_context_timeout(mcp_url, api_key, sample_analysis_context):
    """Test timeout handling."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))
        mock_client_class.return_value = mock_client
        
        result = await store_analysis_context(
            mcp_url=mcp_url,
            api_key=api_key,
            analysis_id="analysis_123",
            context_data=sample_analysis_context
        )
        
        # Should return unsuccessful but not crash
        assert result["success"] is False
        assert "timeout" in result["message"].lower()


@pytest.mark.asyncio
async def test_store_analysis_context_server_error(mcp_url, api_key, sample_analysis_context):
    """Test handling of server errors (5xx)."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        with pytest.raises(MCPClientError):
            await store_analysis_context(
                mcp_url=mcp_url,
                api_key=api_key,
                analysis_id="analysis_123",
                context_data=sample_analysis_context
            )


# Tests for retrieve_related_analyses
@pytest.mark.asyncio
async def test_retrieve_related_analyses_success(mcp_url, api_key):
    """Test successful retrieval of related analyses."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "analysis_id": "analysis_100",
                "analysis_type": "cash_forecast",
                "timestamp": "2024-01-01T10:00:00Z",
                "relevance_score": 0.95,
                "summary": "Similar cash forecast from January",
                "context": {}
            },
            {
                "analysis_id": "analysis_101",
                "analysis_type": "unit_economics",
                "timestamp": "2024-01-05T14:00:00Z",
                "relevance_score": 0.82,
                "summary": "Related unit economics analysis",
                "context": {}
            }
        ]
    }
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        result = await retrieve_related_analyses(
            mcp_url=mcp_url,
            api_key=api_key,
            query="cash forecast for Q1"
        )
        
        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["analysis_id"] == "analysis_100"


@pytest.mark.asyncio
async def test_retrieve_related_analyses_with_filters(mcp_url, api_key):
    """Test retrieval with analysis type filters."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        result = await retrieve_related_analyses(
            mcp_url=mcp_url,
            api_key=api_key,
            query="forecast analysis",
            analysis_types=["cash_forecast", "sales_forecast"]
        )
        
        assert result["success"] is True
        
        # Verify filters were applied
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert "filters" in payload
        assert "analysis_type" in payload["filters"]


@pytest.mark.asyncio
async def test_retrieve_related_analyses_no_results(mcp_url, api_key):
    """Test retrieval when no related analyses found."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        result = await retrieve_related_analyses(
            mcp_url=mcp_url,
            api_key=api_key,
            query="nonexistent analysis"
        )
        
        assert result["success"] is True
        assert result["count"] == 0
        assert len(result["results"]) == 0


@pytest.mark.asyncio
async def test_retrieve_related_analyses_connection_error(mcp_url, api_key):
    """Test graceful handling when MCP server is unavailable during retrieval."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client_class.return_value = mock_client
        
        result = await retrieve_related_analyses(
            mcp_url=mcp_url,
            api_key=api_key,
            query="test query"
        )
        
        # Should return empty results but not crash
        assert result["success"] is False
        assert result["count"] == 0
        assert len(result["results"]) == 0
        assert "unavailable" in result["message"].lower()


@pytest.mark.asyncio
async def test_retrieve_related_analyses_authentication_error(mcp_url, api_key):
    """Test authentication error during retrieval."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Invalid API key"
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        with pytest.raises(MCPAuthenticationError):
            await retrieve_related_analyses(
                mcp_url=mcp_url,
                api_key=api_key,
                query="test query"
            )


# Tests for save_forecast_for_comparison
@pytest.mark.asyncio
async def test_save_forecast_success(mcp_url, api_key, sample_forecast_data):
    """Test successful forecast storage."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "forecast_id": "forecast_2024_01_15",
        "status": "stored"
    }
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        result = await save_forecast_for_comparison(
            mcp_url=mcp_url,
            api_key=api_key,
            forecast_data=sample_forecast_data
        )
        
        assert result["success"] is True
        assert result["forecast_id"] == "forecast_2024_01_15"
        assert "successfully" in result["message"].lower()


@pytest.mark.asyncio
async def test_save_forecast_missing_fields(mcp_url, api_key):
    """Test forecast storage with missing required fields."""
    incomplete_forecast = {
        "forecast_id": "forecast_123",
        "forecast_type": "sales"
        # Missing predictions
    }
    
    result = await save_forecast_for_comparison(
        mcp_url=mcp_url,
        api_key=api_key,
        forecast_data=incomplete_forecast
    )
    
    assert result["success"] is False
    assert "missing" in result["message"].lower()
    assert "predictions" in result["message"].lower()


@pytest.mark.asyncio
async def test_save_forecast_connection_error(mcp_url, api_key, sample_forecast_data):
    """Test graceful handling when MCP server is unavailable during forecast save."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client_class.return_value = mock_client
        
        result = await save_forecast_for_comparison(
            mcp_url=mcp_url,
            api_key=api_key,
            forecast_data=sample_forecast_data
        )
        
        # Should return unsuccessful but not crash
        assert result["success"] is False
        assert "unavailable" in result["message"].lower()


@pytest.mark.asyncio
async def test_save_forecast_authentication_error(mcp_url, api_key, sample_forecast_data):
    """Test authentication error during forecast save."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Invalid API key"
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        with pytest.raises(MCPAuthenticationError):
            await save_forecast_for_comparison(
                mcp_url=mcp_url,
                api_key=api_key,
                forecast_data=sample_forecast_data
            )


# Integration test scenarios
@pytest.mark.asyncio
async def test_graceful_degradation_workflow(mcp_url, api_key, sample_analysis_context):
    """
    Test that the system can continue operating even when MCP is unavailable.
    This is critical for ensuring the CFO agent doesn't fail when MCP server is down.
    """
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client_class.return_value = mock_client
        
        # Store context - should fail gracefully
        store_result = await store_analysis_context(
            mcp_url=mcp_url,
            api_key=api_key,
            analysis_id="analysis_123",
            context_data=sample_analysis_context
        )
        assert store_result["success"] is False
        
        # Retrieve analyses - should fail gracefully
        retrieve_result = await retrieve_related_analyses(
            mcp_url=mcp_url,
            api_key=api_key,
            query="test query"
        )
        assert retrieve_result["success"] is False
        assert retrieve_result["count"] == 0
        
        # In both cases, the system continues operating without MCP


@pytest.mark.asyncio
async def test_url_normalization(api_key, sample_analysis_context):
    """Test that URLs with trailing slashes are handled correctly."""
    urls_to_test = [
        "http://localhost:8051/mcp",
        "http://localhost:8051/mcp/",
        "http://localhost:8051/mcp///"
    ]
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "test"}
    
    for url in urls_to_test:
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await store_analysis_context(
                mcp_url=url,
                api_key=api_key,
                analysis_id="test",
                context_data=sample_analysis_context
            )
            
            assert result["success"] is True
            
            # Verify the endpoint URL was normalized (no double slashes)
            call_args = mock_client.post.call_args
            endpoint = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
            assert "//" not in endpoint.replace("http://", "")
