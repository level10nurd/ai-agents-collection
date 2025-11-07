"""
MCP Client for storing and retrieving analysis context using Supabase.

This module provides functions for:
1. Storing analysis context in Supabase for cross-analysis coordination
2. RAG (Retrieval-Augmented Generation) search for related past analyses
3. Saving forecasts for historical accuracy tracking

The Supabase database serves as the MCP (Model Context Protocol) knowledge graph,
enabling sophisticated context management and retrieval across multiple analyses.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

logger = logging.getLogger(__name__)


# ============================================
# Custom Exceptions
# ============================================


class MCPClientError(Exception):
    """Base exception for MCP client errors."""
    pass


class MCPConnectionError(MCPClientError):
    """Exception raised when connection to Supabase fails."""
    pass


class MCPAuthenticationError(MCPClientError):
    """Exception raised when authentication with Supabase fails."""
    pass


# ============================================
# Supabase Client Initialization
# ============================================


def _create_supabase_client(
    supabase_url: str,
    service_key: str,
    timeout: int = 30
) -> Client:
    """
    Create and configure a Supabase client instance.
    
    Args:
        supabase_url: Supabase project URL
        service_key: Supabase service role key (for admin access)
        timeout: Request timeout in seconds
        
    Returns:
        Configured Supabase client
        
    Raises:
        MCPAuthenticationError: If authentication fails
        MCPConnectionError: If connection fails
    """
    try:
        options = ClientOptions(
            auto_refresh_token=False,
            persist_session=False,
        )
        
        client = create_client(
            supabase_url=supabase_url,
            supabase_key=service_key,
            options=options
        )
        
        return client
        
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        if "auth" in str(e).lower() or "key" in str(e).lower():
            raise MCPAuthenticationError(f"Authentication failed: {e}")
        raise MCPConnectionError(f"Connection failed: {e}")


# ============================================
# Context Storage Functions
# ============================================


async def store_analysis_context(
    supabase_url: str,
    service_key: str,
    analysis_id: str,
    context_data: Dict[str, Any],
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Store analysis context in Supabase knowledge graph.
    
    This function enables cross-analysis information retrieval by storing
    structured context data that can be searched and retrieved later.
    
    Args:
        supabase_url: Supabase project URL
        service_key: Supabase service role key
        analysis_id: Unique identifier for this analysis
        context_data: Structured context data to store (must be JSON-serializable)
        timeout: Request timeout in seconds
        
    Returns:
        Dict with operation status and stored record metadata
        
    Raises:
        MCPClientError: If storage operation fails
        MCPConnectionError: If Supabase is unreachable
        MCPAuthenticationError: If authentication fails
        
    Example:
        >>> context = {
        ...     "analysis_type": "revenue_forecast",
        ...     "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
        ...     "key_findings": ["Seasonal spike in Q4", "Growth rate: 15%"],
        ...     "data_sources": ["shopify", "quickbooks"],
        ...     "metadata": {"analyst": "ai_cfo", "confidence": 0.85}
        ... }
        >>> result = await store_analysis_context(
        ...     supabase_url="https://xxx.supabase.co",
        ...     service_key="eyJ...",
        ...     analysis_id="revenue_2024_q1",
        ...     context_data=context
        ... )
        >>> print(result["status"])
        "success"
    """
    try:
        client = _create_supabase_client(supabase_url, service_key, timeout)
        
        # Prepare record for insertion
        record = {
            "analysis_id": analysis_id,
            "context_data": context_data,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Insert or update (upsert) the context
        response = client.table("analysis_contexts").upsert(
            record,
            on_conflict="analysis_id"
        ).execute()
        
        logger.info(f"Successfully stored context for analysis: {analysis_id}")
        
        return {
            "status": "success",
            "analysis_id": analysis_id,
            "record": response.data[0] if response.data else record,
            "message": f"Context stored for analysis {analysis_id}"
        }
        
    except (MCPConnectionError, MCPAuthenticationError):
        raise
    except Exception as e:
        logger.error(f"Failed to store analysis context: {e}")
        raise MCPClientError(f"Storage operation failed: {e}")


# ============================================
# RAG Retrieval Functions
# ============================================


async def retrieve_related_analyses(
    supabase_url: str,
    service_key: str,
    query: str,
    limit: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Search for related past analyses using RAG (Retrieval-Augmented Generation).
    
    This function performs semantic search across stored analysis contexts to find
    relevant historical analyses that can inform the current analysis.
    
    Args:
        supabase_url: Supabase project URL
        service_key: Supabase service role key
        query: Natural language search query
        limit: Maximum number of results to return
        filters: Optional filters to apply (e.g., {"analysis_type": "forecast"})
        timeout: Request timeout in seconds
        
    Returns:
        Dict containing matching analyses and their relevance scores
        
    Raises:
        MCPClientError: If retrieval operation fails
        MCPConnectionError: If Supabase is unreachable
        MCPAuthenticationError: If authentication fails
        
    Example:
        >>> results = await retrieve_related_analyses(
        ...     supabase_url="https://xxx.supabase.co",
        ...     service_key="eyJ...",
        ...     query="revenue forecasts with seasonal patterns",
        ...     limit=5,
        ...     filters={"analysis_type": "revenue_forecast"}
        ... )
        >>> for analysis in results["analyses"]:
        ...     print(f"{analysis['analysis_id']}: {analysis['relevance_score']}")
    """
    try:
        client = _create_supabase_client(supabase_url, service_key, timeout)
        
        # Build base query
        query_builder = client.table("analysis_contexts").select("*")
        
        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                # Filter on JSONB context_data field
                query_builder = query_builder.filter(
                    f"context_data->{key}", "eq", value
                )
        
        # Execute query
        response = query_builder.order(
            "created_at", desc=True
        ).limit(limit).execute()
        
        analyses = response.data if response.data else []
        
        # TODO: Implement semantic search using Supabase vector embeddings
        # For now, use simple keyword matching and recency as relevance proxy
        results = []
        query_terms = set(query.lower().split())
        
        for analysis in analyses:
            # Calculate simple relevance score based on keyword overlap
            context_text = str(analysis.get("context_data", {})).lower()
            context_terms = set(context_text.split())
            overlap = len(query_terms & context_terms)
            relevance_score = overlap / len(query_terms) if query_terms else 0
            
            results.append({
                "analysis_id": analysis["analysis_id"],
                "context_data": analysis["context_data"],
                "created_at": analysis["created_at"],
                "relevance_score": round(relevance_score, 3),
            })
        
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        logger.info(
            f"Retrieved {len(results)} related analyses for query: '{query}'"
        )
        
        return {
            "status": "success",
            "query": query,
            "total_results": len(results),
            "analyses": results,
            "message": f"Found {len(results)} related analyses"
        }
        
    except (MCPConnectionError, MCPAuthenticationError):
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve related analyses: {e}")
        raise MCPClientError(f"Retrieval operation failed: {e}")


# ============================================
# Forecast Comparison Functions
# ============================================


async def save_forecast_for_comparison(
    supabase_url: str,
    service_key: str,
    forecast_id: str,
    forecast_data: Dict[str, Any],
    forecast_metadata: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Save forecast data for historical accuracy tracking and comparison.
    
    This enables tracking forecast accuracy over time by storing predictions
    alongside their metadata for later comparison with actual results.
    
    Args:
        supabase_url: Supabase project URL
        service_key: Supabase service role key
        forecast_id: Unique identifier for this forecast
        forecast_data: The forecast predictions (time series data, projections, etc.)
        forecast_metadata: Optional metadata (model used, parameters, assumptions)
        timeout: Request timeout in seconds
        
    Returns:
        Dict with operation status and saved forecast metadata
        
    Raises:
        MCPClientError: If save operation fails
        MCPConnectionError: If Supabase is unreachable
        MCPAuthenticationError: If authentication fails
        
    Example:
        >>> forecast = {
        ...     "period": "2024-Q2",
        ...     "predictions": {
        ...         "revenue": 1500000,
        ...         "costs": 950000,
        ...         "profit": 550000
        ...     },
        ...     "confidence_intervals": {
        ...         "revenue": {"lower": 1300000, "upper": 1700000}
        ...     }
        ... }
        >>> metadata = {
        ...     "model": "prophet",
        ...     "training_period": "2022-01-01 to 2024-03-31",
        ...     "features": ["seasonality", "trends", "holidays"]
        ... }
        >>> result = await save_forecast_for_comparison(
        ...     supabase_url="https://xxx.supabase.co",
        ...     service_key="eyJ...",
        ...     forecast_id="revenue_2024_q2",
        ...     forecast_data=forecast,
        ...     forecast_metadata=metadata
        ... )
    """
    try:
        client = _create_supabase_client(supabase_url, service_key, timeout)
        
        # Prepare forecast record
        record = {
            "forecast_id": forecast_id,
            "forecast_data": forecast_data,
            "forecast_metadata": forecast_metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Insert or update (upsert) the forecast
        response = client.table("forecasts").upsert(
            record,
            on_conflict="forecast_id"
        ).execute()
        
        logger.info(f"Successfully saved forecast: {forecast_id}")
        
        return {
            "status": "success",
            "forecast_id": forecast_id,
            "record": response.data[0] if response.data else record,
            "message": f"Forecast saved: {forecast_id}"
        }
        
    except (MCPConnectionError, MCPAuthenticationError):
        raise
    except Exception as e:
        logger.error(f"Failed to save forecast: {e}")
        raise MCPClientError(f"Forecast save operation failed: {e}")


# ============================================
# Health Check Functions
# ============================================


async def check_connection(
    supabase_url: str,
    service_key: str,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Check if connection to Supabase is working.
    
    Args:
        supabase_url: Supabase project URL
        service_key: Supabase service role key
        timeout: Request timeout in seconds
        
    Returns:
        Dict with connection status and details
    """
    try:
        client = _create_supabase_client(supabase_url, service_key, timeout)
        
        # Try a simple query to verify connection
        client.table("analysis_contexts").select("analysis_id").limit(1).execute()
        
        return {
            "status": "connected",
            "supabase_url": supabase_url,
            "message": "Connection to Supabase successful"
        }
        
    except MCPAuthenticationError as e:
        return {
            "status": "error",
            "error_type": "authentication",
            "message": str(e)
        }
    except MCPConnectionError as e:
        return {
            "status": "error",
            "error_type": "connection",
            "message": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": "unknown",
            "message": str(e)
        }
