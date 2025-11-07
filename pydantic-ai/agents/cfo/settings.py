"""
Configuration management for AI CFO system using pydantic-settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """AI CFO application settings with environment variable support."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # ============================================
    # LLM Configuration
    # ============================================

    # OpenRouter (preferred provider)
    openrouter_api_key: Optional[str] = Field(default=None)
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1"
    )

    # Alternative LLM providers
    openai_api_key: Optional[str] = Field(default=None)
    openai_org_id: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)

    # Model selection
    # TODO: Implement programmatic model selection using tools/openrouter/all_models.sh
    default_coordinator_model: str = Field(
        default="anthropic/claude-sonnet-4"
    )
    default_specialist_model: str = Field(
        default="openai/gpt-4o"
    )

    # ============================================
    # QuickBooks Configuration
    # ============================================
    # Note: Using CSV uploads instead of API access per user requirements

    quickbooks_client_id: Optional[str] = Field(default=None)
    quickbooks_client_secret: Optional[str] = Field(default=None)
    quickbooks_company_id: Optional[str] = Field(default=None)
    quickbooks_access_token: Optional[str] = Field(default=None)
    quickbooks_refresh_token: Optional[str] = Field(default=None)

    # ============================================
    # Shopify Admin API
    # ============================================

    shopify_api_key: str = Field(...)
    shopify_api_secret: str = Field(...)
    shopify_access_token: str = Field(...)
    shopify_shop_name: str = Field(default="vochill")
    shopify_api_version: str = Field(default="2024-01")

    # ============================================
    # Amazon Seller Central SP-API
    # ============================================
    # Note: Placeholders for future implementation per user requirements

    amazon_lwa_client_id: Optional[str] = Field(default=None)
    amazon_lwa_client_secret: Optional[str] = Field(default=None)
    amazon_refresh_token: Optional[str] = Field(default=None)
    amazon_marketplace_id: Optional[str] = Field(default=None)
    amazon_seller_id: Optional[str] = Field(default=None)

    # ============================================
    # InfoPlus WMS API
    # ============================================

    infoplus_api_key: str = Field(...)
    infoplus_account_no: str = Field(...)

    # ============================================
    # Supabase Configuration
    # ============================================
    # Note: Using Supabase as MCP server per user requirements

    supabase_url: str = Field(...)
    supabase_service_key: str = Field(...)

    # ============================================
    # MCP Server Configuration
    # ============================================
    # Note: Using Supabase as MCP server

    mcp_server_url: Optional[str] = Field(default=None)
    mcp_api_key: Optional[str] = Field(default=None)

    # ============================================
    # Application Configuration
    # ============================================

    env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    company_id: str = Field(default="vochill")

    # ============================================
    # Field Validators
    # ============================================

    @field_validator("shopify_api_key", "shopify_api_secret", "shopify_access_token")
    @classmethod
    def validate_shopify_credentials(cls, v):
        """Ensure Shopify credentials are not empty."""
        if not v or v.strip() == "":
            raise ValueError("Shopify credential cannot be empty")
        return v.strip()

    @field_validator("shopify_access_token")
    @classmethod
    def validate_shopify_token_format(cls, v):
        """Validate Shopify access token format."""
        if v and not v.startswith("shpat_"):
            raise ValueError("Shopify access token must start with 'shpat_'")
        return v

    @field_validator("infoplus_api_key", "infoplus_account_no")
    @classmethod
    def validate_infoplus_credentials(cls, v):
        """Ensure InfoPlus credentials are not empty."""
        if not v or v.strip() == "":
            raise ValueError("InfoPlus credential cannot be empty")
        return v.strip()

    @field_validator("supabase_url")
    @classmethod
    def validate_supabase_url(cls, v):
        """Validate Supabase URL format."""
        if not v or v.strip() == "":
            raise ValueError("Supabase URL cannot be empty")
        if not v.startswith("https://"):
            raise ValueError("Supabase URL must start with 'https://'")
        if not v.endswith(".supabase.co"):
            raise ValueError("Supabase URL must end with '.supabase.co'")
        return v.strip()

    @field_validator("supabase_service_key")
    @classmethod
    def validate_supabase_key(cls, v):
        """Validate Supabase service key format."""
        if not v or v.strip() == "":
            raise ValueError("Supabase service key cannot be empty")
        if not v.startswith("eyJ"):
            raise ValueError("Supabase service key appears to be invalid (should be a JWT)")
        return v.strip()

    @field_validator("openrouter_api_key", "openai_api_key", "anthropic_api_key")
    @classmethod
    def validate_llm_api_keys(cls, v):
        """Validate LLM API keys if provided."""
        if v and v.strip() == "":
            raise ValueError("LLM API key cannot be empty string")
        return v.strip() if v else None

    @field_validator("openrouter_api_key")
    @classmethod
    def validate_openrouter_key_format(cls, v):
        """Validate OpenRouter API key format."""
        if v and not v.startswith("sk-or-v1-"):
            raise ValueError("OpenRouter API key must start with 'sk-or-v1-'")
        return v

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key_format(cls, v):
        """Validate OpenAI API key format."""
        if v and not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v

    @field_validator("anthropic_api_key")
    @classmethod
    def validate_anthropic_key_format(cls, v):
        """Validate Anthropic API key format."""
        if v and not v.startswith("sk-ant-"):
            raise ValueError("Anthropic API key must start with 'sk-ant-'")
        return v


# Global settings instance
try:
    settings = Settings()
except Exception:
    # For testing, create settings with dummy values
    os.environ.setdefault("SHOPIFY_API_KEY", "test_key")
    os.environ.setdefault("SHOPIFY_API_SECRET", "test_secret")
    os.environ.setdefault("SHOPIFY_ACCESS_TOKEN", "shpat_test")
    os.environ.setdefault("INFOPLUS_API_KEY", "test_key")
    os.environ.setdefault("INFOPLUS_ACCOUNT_NO", "test_account")
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_SERVICE_KEY", "eyJtest")
    settings = Settings()
