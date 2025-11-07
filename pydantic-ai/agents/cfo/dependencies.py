"""
CFO Agent Dependencies

This module defines the dependency injection structures for the CFO agent system.
Dependencies manage API credentials and configuration for both coordinator and specialist agents.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CFOCoordinatorDependencies:
    """
    Dependencies for the CFO coordinator agent.

    This dataclass holds all API credentials and configuration needed for the
    coordinator agent to orchestrate specialist agents and access external services.

    Attributes:
        qb_api_key: QuickBooks API credentials for accounting data access
        shopify_api_key: Shopify API credentials for e-commerce data
        amazon_api_key: Amazon Seller API credentials for marketplace data
        infoplus_api_key: InfoPlus API credentials for inventory/warehouse data
        supabase_url: Supabase project URL for database access
        supabase_key: Supabase API key for authentication
        mcp_api_key: MCP (Model Context Protocol) API credentials
        session_id: Optional session identifier for tracking and debugging
    """
    qb_api_key: str
    shopify_api_key: str
    amazon_api_key: str
    infoplus_api_key: str
    supabase_url: str
    supabase_key: str
    mcp_api_key: str
    session_id: Optional[str] = None


@dataclass
class SpecialistDependencies:
    """
    Dependencies for specialist agents.

    This dataclass holds all API credentials needed for specialist agents.
    It contains the same credentials as CFOCoordinatorDependencies but without
    the session_id, as specialists operate under the coordinator's session context.

    Attributes:
        qb_api_key: QuickBooks API credentials for accounting data access
        shopify_api_key: Shopify API credentials for e-commerce data
        amazon_api_key: Amazon Seller API credentials for marketplace data
        infoplus_api_key: InfoPlus API credentials for inventory/warehouse data
        supabase_url: Supabase project URL for database access
        supabase_key: Supabase API key for authentication
        mcp_api_key: MCP (Model Context Protocol) API credentials
    """
    qb_api_key: str
    shopify_api_key: str
    amazon_api_key: str
    infoplus_api_key: str
    supabase_url: str
    supabase_key: str
    mcp_api_key: str
