"""
MCP (Model Context Protocol) Client Integration

Provides context storage and retrieval for cross-analysis information coordination.
Enables the CFO coordinator to maintain context across multiple analyses using
the MCP knowledge graph and RAG capabilities.

Features:
- Store analysis context in MCP knowledge graph
- Retrieve related past analyses via RAG search
- Save forecasts for historical accuracy tracking
- Graceful degradation when MCP server unavailable

Note: MCP integration is optional - all functions handle server unavailability gracefully.
"""

import httpx
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    """Base exception for MCP client errors."""
    pass


class MCPConnectionError(MCPClientError):
    """Raised when MCP server is unavailable."""
    pass


class MCPAuthenticationError(MCPClientError):
    """Raised when authentication with MCP server fails."""
    pass


async def store_analysis_context(
    mcp_url: str,
    api_key: str,
    analysis_id: str,
    context_data: Dict[str, Any],
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Store analysis context in MCP knowledge graph for future retrieval.
    
    Enables cross-analysis information coordination by storing:
    - Analysis metadata (type, date, parameters)
    - Key insights and findings
    - Data sources used
    - Recommendations made
    
    Args:
        mcp_url: Base URL of the MCP server (e.g., "http://localhost:8051/mcp")
        api_key: API key for MCP authentication
        analysis_id: Unique identifier for this analysis
        context_data: Dictionary containing analysis context:
            - analysis_type (str): Type of analysis (e.g., "cash_forecast", "unit_economics")
            - timestamp (str): ISO format timestamp
            - parameters (dict): Input parameters used
            - insights (list): Key findings and insights
            - data_sources (list): APIs/sources used
            - recommendations (list): Actions recommended
        timeout: Request timeout in seconds (default: 10.0)
    
    Returns:
        Dict containing:
            - success (bool): Whether storage succeeded
            - stored_id (str): ID of stored context (if successful)
            - message (str): Status message
    
    Raises:
        MCPConnectionError: If MCP server is unreachable
        MCPAuthenticationError: If API key is invalid
        MCPClientError: For other MCP-related errors
        
    Note:
        Returns gracefully with success=False if MCP unavailable rather than crashing.
        This allows the system to function without MCP coordination.
    """
    endpoint = f"{mcp_url.rstrip('/')}/knowledge/store"
    
    # Validate required context fields
    required_fields = ["analysis_type", "timestamp"]
    missing_fields = [field for field in required_fields if field not in context_data]
    if missing_fields:
        return {
            "success": False,
            "stored_id": None,
            "message": f"Missing required context fields: {', '.join(missing_fields)}"
        }
    
    # Prepare payload with metadata
    payload = {
        "id": analysis_id,
        "context": context_data,
        "stored_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "version": "1.0",
            "source": "cfo_agent"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers=headers
            )
            
            # Handle authentication errors
            if response.status_code == 401:
                logger.error("MCP authentication failed - invalid API key")
                raise MCPAuthenticationError("Invalid MCP API key")
            
            # Handle client errors (4xx) - raise exception
            if 400 <= response.status_code < 500:
                error_msg = f"MCP client error: {response.status_code}"
                logger.error(f"{error_msg} - {response.text}")
                raise MCPClientError(error_msg)
            
            # Handle server errors (5xx) - raise exception
            if response.status_code >= 500:
                error_msg = f"MCP server error: {response.status_code}"
                logger.error(f"{error_msg} - {response.text}")
                raise MCPClientError(error_msg)
            
            # Parse successful response
            result = response.json()
            logger.info(f"Successfully stored analysis context: {analysis_id}")
            
            return {
                "success": True,
                "stored_id": result.get("id", analysis_id),
                "message": "Context stored successfully"
            }
            
    except httpx.ConnectError as e:
        logger.warning(f"MCP server unavailable: {e}")
        # Graceful degradation - return unsuccessful but don't crash
        return {
            "success": False,
            "stored_id": None,
            "message": "MCP server unavailable - analysis will proceed without context storage"
        }
        
    except httpx.TimeoutException as e:
        logger.warning(f"MCP server timeout: {e}")
        return {
            "success": False,
            "stored_id": None,
            "message": "MCP server timeout - analysis will proceed without context storage"
        }
        
    except (MCPAuthenticationError, MCPClientError):
        # Re-raise MCP-specific errors (authentication, client/server errors)
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error storing context: {e}")
        # Graceful degradation for unexpected errors
        return {
            "success": False,
            "stored_id": None,
            "message": f"Error storing context: {str(e)}"
        }


async def retrieve_related_analyses(
    mcp_url: str,
    api_key: str,
    query: str,
    limit: int = 5,
    analysis_types: Optional[List[str]] = None,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Retrieve related past analyses using RAG search.
    
    Searches the MCP knowledge graph for analyses related to the current query.
    Enables the coordinator to leverage insights from past analyses when making
    recommendations.
    
    Args:
        mcp_url: Base URL of the MCP server
        api_key: API key for MCP authentication
        query: Natural language search query describing what to find
        limit: Maximum number of results to return (default: 5)
        analysis_types: Optional list of analysis types to filter by
            (e.g., ["cash_forecast", "unit_economics"])
        timeout: Request timeout in seconds (default: 10.0)
    
    Returns:
        Dict containing:
            - success (bool): Whether retrieval succeeded
            - results (list): List of related analyses, each containing:
                - analysis_id (str): ID of the analysis
                - analysis_type (str): Type of analysis
                - timestamp (str): When analysis was performed
                - relevance_score (float): Similarity score (0-1)
                - summary (str): Brief summary of findings
                - context (dict): Full context data
            - count (int): Number of results found
            - message (str): Status message
    
    Raises:
        MCPConnectionError: If MCP server is unreachable
        MCPAuthenticationError: If API key is invalid
        MCPClientError: For other MCP-related errors
        
    Note:
        Returns empty results with success=False if MCP unavailable rather than crashing.
    """
    endpoint = f"{mcp_url.rstrip('/')}/rag/search"
    
    # Prepare search parameters
    params = {
        "query": query,
        "limit": limit,
    }
    
    if analysis_types:
        params["filters"] = {
            "analysis_type": {"$in": analysis_types}
        }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                json=params,
                headers=headers
            )
            
            # Handle authentication errors
            if response.status_code == 401:
                logger.error("MCP authentication failed - invalid API key")
                raise MCPAuthenticationError("Invalid MCP API key")
            
            # Handle client errors (4xx) - raise exception
            if 400 <= response.status_code < 500:
                error_msg = f"MCP client error: {response.status_code}"
                logger.error(f"{error_msg} - {response.text}")
                raise MCPClientError(error_msg)
            
            # Handle server errors (5xx) - raise exception
            if response.status_code >= 500:
                error_msg = f"MCP server error: {response.status_code}"
                logger.error(f"{error_msg} - {response.text}")
                raise MCPClientError(error_msg)
            
            # Parse successful response
            result = response.json()
            results = result.get("results", [])
            
            logger.info(f"Retrieved {len(results)} related analyses for query: {query}")
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "message": f"Found {len(results)} related analyses"
            }
            
    except httpx.ConnectError as e:
        logger.warning(f"MCP server unavailable: {e}")
        # Graceful degradation - return empty results
        return {
            "success": False,
            "results": [],
            "count": 0,
            "message": "MCP server unavailable - no historical context available"
        }
        
    except httpx.TimeoutException as e:
        logger.warning(f"MCP server timeout: {e}")
        return {
            "success": False,
            "results": [],
            "count": 0,
            "message": "MCP server timeout - no historical context available"
        }
        
    except (MCPAuthenticationError, MCPClientError):
        # Re-raise MCP-specific errors (authentication, client/server errors)
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error retrieving analyses: {e}")
        # Graceful degradation for unexpected errors
        return {
            "success": False,
            "results": [],
            "count": 0,
            "message": f"Error retrieving analyses: {str(e)}"
        }


async def save_forecast_for_comparison(
    mcp_url: str,
    api_key: str,
    forecast_data: Dict[str, Any],
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Save forecast in MCP for historical accuracy tracking.
    
    Stores forecasts with their predictions and parameters to enable:
    - Historical accuracy analysis
    - Model performance tracking
    - Forecast vs. actual comparisons
    - Continuous improvement of forecasting models
    
    Args:
        mcp_url: Base URL of the MCP server
        api_key: API key for MCP authentication
        forecast_data: Dictionary containing:
            - forecast_id (str): Unique forecast identifier
            - forecast_type (str): Type of forecast (e.g., "sales", "cash_flow")
            - model_used (str): Model/algorithm used
            - parameters (dict): Model parameters
            - predictions (list): Forecasted values with dates
            - confidence_intervals (dict): Upper/lower bounds if available
            - created_at (str): ISO timestamp
        timeout: Request timeout in seconds (default: 10.0)
    
    Returns:
        Dict containing:
            - success (bool): Whether save succeeded
            - forecast_id (str): ID of saved forecast
            - message (str): Status message
    
    Raises:
        MCPConnectionError: If MCP server is unreachable
        MCPAuthenticationError: If API key is invalid
        MCPClientError: For other MCP-related errors
        
    Note:
        Returns gracefully with success=False if MCP unavailable.
    """
    endpoint = f"{mcp_url.rstrip('/')}/knowledge/forecasts"
    
    # Validate required forecast fields
    required_fields = ["forecast_id", "forecast_type", "predictions"]
    missing_fields = [field for field in required_fields if field not in forecast_data]
    if missing_fields:
        return {
            "success": False,
            "forecast_id": None,
            "message": f"Missing required forecast fields: {', '.join(missing_fields)}"
        }
    
    # Add metadata
    payload = {
        **forecast_data,
        "stored_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "version": "1.0",
            "source": "cfo_agent"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers=headers
            )
            
            # Handle authentication errors
            if response.status_code == 401:
                logger.error("MCP authentication failed - invalid API key")
                raise MCPAuthenticationError("Invalid MCP API key")
            
            # Handle client errors (4xx) - raise exception
            if 400 <= response.status_code < 500:
                error_msg = f"MCP client error: {response.status_code}"
                logger.error(f"{error_msg} - {response.text}")
                raise MCPClientError(error_msg)
            
            # Handle server errors (5xx) - raise exception
            if response.status_code >= 500:
                error_msg = f"MCP server error: {response.status_code}"
                logger.error(f"{error_msg} - {response.text}")
                raise MCPClientError(error_msg)
            
            # Parse successful response
            result = response.json()
            forecast_id = result.get("forecast_id", forecast_data["forecast_id"])
            
            logger.info(f"Successfully stored forecast: {forecast_id}")
            
            return {
                "success": True,
                "forecast_id": forecast_id,
                "message": "Forecast stored successfully"
            }
            
    except httpx.ConnectError as e:
        logger.warning(f"MCP server unavailable: {e}")
        # Graceful degradation
        return {
            "success": False,
            "forecast_id": None,
            "message": "MCP server unavailable - forecast not stored for comparison"
        }
        
    except httpx.TimeoutException as e:
        logger.warning(f"MCP server timeout: {e}")
        return {
            "success": False,
            "forecast_id": None,
            "message": "MCP server timeout - forecast not stored for comparison"
        }
        
    except (MCPAuthenticationError, MCPClientError):
        # Re-raise MCP-specific errors (authentication, client/server errors)
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error storing forecast: {e}")
        # Graceful degradation for unexpected errors
        return {
            "success": False,
            "forecast_id": None,
            "message": f"Error storing forecast: {str(e)}"
        }
