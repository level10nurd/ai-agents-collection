"""
Shopify Admin API Integration for CFO Agent.

This module provides functions to fetch orders and customer data from Shopify Admin API,
with proper pagination handling and rate limit management.

Reference: PRPs/04_api_integrations/financial_api_integration_research.md
"""

import asyncio
import time
from datetime import datetime
from typing import Any

import httpx


class ShopifyAPIError(Exception):
    """Base exception for Shopify API errors."""

    pass


class ShopifyRateLimitError(ShopifyAPIError):
    """Raised when Shopify rate limit is exceeded."""

    pass


async def fetch_orders(
    api_key: str,
    shop_name: str,
    start_date: str,
    end_date: str,
    limit: int = 250,
) -> list[dict[str, Any]]:
    """
    Fetch orders from Shopify Admin API with pagination and rate limiting.

    Args:
        api_key: Shopify API access token
        shop_name: Shopify shop name (e.g., 'myshop' for myshop.myshopify.com)
        start_date: Start date for orders (ISO 8601 format: YYYY-MM-DD)
        end_date: End date for orders (ISO 8601 format: YYYY-MM-DD)
        limit: Number of orders per page (max 250)

    Returns:
        List of order dictionaries with keys:
        - order_id: Shopify order ID
        - total_price: Total order price
        - created_at: Order creation timestamp
        - line_items: List of line items in the order

    Raises:
        ShopifyAPIError: If API request fails
        ShopifyRateLimitError: If rate limit is exceeded after retries

    Example:
        >>> orders = await fetch_orders(
        ...     api_key="shpat_xxxxx",
        ...     shop_name="myshop",
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31"
        ... )
        >>> print(len(orders))
        150
    """
    base_url = f"https://{shop_name}.myshopify.com/admin/api/2024-01"
    endpoint = f"{base_url}/orders.json"

    headers = {
        "X-Shopify-Access-Token": api_key,
        "Content-Type": "application/json",
    }

    all_orders = []
    next_page_url = None
    page_count = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            # Determine URL and params
            if next_page_url:
                url = next_page_url
                params = None
            else:
                url = endpoint
                params = {
                    "limit": limit,
                    "status": "any",
                    "created_at_min": f"{start_date}T00:00:00-00:00",
                    "created_at_max": f"{end_date}T23:59:59-00:00",
                }

            # Make request with retry logic
            response = await _make_request_with_retry(
                client, "GET", url, headers=headers, params=params
            )

            # Parse response
            data = response.json()
            orders = data.get("orders", [])

            # Transform to standardized format
            for order in orders:
                all_orders.append(
                    {
                        "order_id": order.get("id"),
                        "total_price": float(order.get("total_price", 0)),
                        "created_at": order.get("created_at"),
                        "line_items": [
                            {
                                "id": item.get("id"),
                                "product_id": item.get("product_id"),
                                "quantity": item.get("quantity"),
                                "price": float(item.get("price", 0)),
                                "name": item.get("name"),
                            }
                            for item in order.get("line_items", [])
                        ],
                    }
                )

            page_count += 1

            # Check for next page using Link header (cursor-based pagination)
            link_header = response.headers.get("Link")
            if link_header:
                next_page_url = _parse_link_header(link_header)
                if not next_page_url:
                    break
            else:
                # No Link header means no more pages
                break

            # Rate limit safety: small delay between pages
            await asyncio.sleep(0.5)

    return all_orders


async def fetch_customers(
    api_key: str, shop_name: str, limit: int = 250
) -> list[dict[str, Any]]:
    """
    Fetch customers from Shopify Admin API with pagination.

    Args:
        api_key: Shopify API access token
        shop_name: Shopify shop name (e.g., 'myshop' for myshop.myshopify.com)
        limit: Number of customers per page (max 250)

    Returns:
        List of customer dictionaries with keys:
        - customer_id: Shopify customer ID
        - orders_count: Total number of orders
        - total_spent: Total amount spent by customer

    Raises:
        ShopifyAPIError: If API request fails
        ShopifyRateLimitError: If rate limit is exceeded after retries

    Example:
        >>> customers = await fetch_customers(
        ...     api_key="shpat_xxxxx",
        ...     shop_name="myshop"
        ... )
        >>> print(len(customers))
        500
    """
    base_url = f"https://{shop_name}.myshopify.com/admin/api/2024-01"
    endpoint = f"{base_url}/customers.json"

    headers = {
        "X-Shopify-Access-Token": api_key,
        "Content-Type": "application/json",
    }

    all_customers = []
    next_page_url = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            # Determine URL and params
            if next_page_url:
                url = next_page_url
                params = None
            else:
                url = endpoint
                params = {"limit": limit}

            # Make request with retry logic
            response = await _make_request_with_retry(
                client, "GET", url, headers=headers, params=params
            )

            # Parse response
            data = response.json()
            customers = data.get("customers", [])

            # Transform to standardized format
            for customer in customers:
                all_customers.append(
                    {
                        "customer_id": customer.get("id"),
                        "orders_count": customer.get("orders_count", 0),
                        "total_spent": float(customer.get("total_spent", 0)),
                        "email": customer.get("email"),
                        "first_name": customer.get("first_name"),
                        "last_name": customer.get("last_name"),
                    }
                )

            # Check for next page using Link header
            link_header = response.headers.get("Link")
            if link_header:
                next_page_url = _parse_link_header(link_header)
                if not next_page_url:
                    break
            else:
                break

            # Rate limit safety: small delay between pages
            await asyncio.sleep(0.5)

    return all_customers


def calculate_customer_metrics(
    orders: list[dict[str, Any]], customers: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Calculate customer acquisition cost (CAC) related metrics.

    Args:
        orders: List of orders from fetch_orders()
        customers: List of customers from fetch_customers()

    Returns:
        Dictionary containing:
        - average_order_value: Average value per order
        - repeat_purchase_rate: Percentage of customers with multiple orders
        - total_orders: Total number of orders
        - total_revenue: Total revenue from all orders
        - unique_customers: Number of unique customers
        - orders_per_customer: Average orders per customer

    Example:
        >>> orders = await fetch_orders(...)
        >>> customers = await fetch_customers(...)
        >>> metrics = calculate_customer_metrics(orders, customers)
        >>> print(f"AOV: ${metrics['average_order_value']:.2f}")
        AOV: $156.78
    """
    # Calculate order-based metrics
    total_orders = len(orders)
    total_revenue = sum(order["total_price"] for order in orders)

    average_order_value = total_revenue / total_orders if total_orders > 0 else 0.0

    # Calculate customer-based metrics
    unique_customers = len(customers)
    customers_with_multiple_orders = sum(
        1 for customer in customers if customer["orders_count"] > 1
    )

    repeat_purchase_rate = (
        (customers_with_multiple_orders / unique_customers * 100)
        if unique_customers > 0
        else 0.0
    )

    orders_per_customer = (
        total_orders / unique_customers if unique_customers > 0 else 0.0
    )

    # Calculate additional useful metrics
    customer_lifetime_values = [
        customer["total_spent"] for customer in customers if customer["total_spent"] > 0
    ]
    average_customer_lifetime_value = (
        sum(customer_lifetime_values) / len(customer_lifetime_values)
        if customer_lifetime_values
        else 0.0
    )

    return {
        "average_order_value": round(average_order_value, 2),
        "repeat_purchase_rate": round(repeat_purchase_rate, 2),
        "total_orders": total_orders,
        "total_revenue": round(total_revenue, 2),
        "unique_customers": unique_customers,
        "orders_per_customer": round(orders_per_customer, 2),
        "average_customer_lifetime_value": round(average_customer_lifetime_value, 2),
    }


async def _make_request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> httpx.Response:
    """
    Make HTTP request with exponential backoff retry on rate limits.

    Implements Shopify's leaky bucket rate limiting strategy:
    - Handles 429 status code
    - Respects Retry-After header
    - Implements exponential backoff

    Args:
        client: httpx AsyncClient instance
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        headers: Request headers
        params: Query parameters
        max_retries: Maximum number of retry attempts

    Returns:
        httpx.Response object

    Raises:
        ShopifyRateLimitError: If max retries exceeded
        ShopifyAPIError: For other API errors
    """
    retry_count = 0

    while retry_count <= max_retries:
        try:
            response = await client.request(
                method=method, url=url, headers=headers, params=params
            )

            # Handle rate limiting (429)
            if response.status_code == 429:
                if retry_count >= max_retries:
                    raise ShopifyRateLimitError(
                        f"Rate limit exceeded after {max_retries} retries"
                    )

                # Get retry delay from Retry-After header (in seconds)
                retry_after = int(response.headers.get("Retry-After", 2))
                wait_time = min(retry_after, 60)  # Cap at 60 seconds

                # Log rate limit hit (in production, use proper logging)
                print(
                    f"Rate limit hit (attempt {retry_count + 1}/{max_retries}). "
                    f"Waiting {wait_time} seconds..."
                )

                await asyncio.sleep(wait_time)
                retry_count += 1
                continue

            # Handle other HTTP errors
            response.raise_for_status()

            return response

        except httpx.HTTPStatusError as e:
            # Non-429 HTTP errors
            raise ShopifyAPIError(
                f"HTTP error {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            # Network/connection errors
            if retry_count >= max_retries:
                raise ShopifyAPIError(f"Request failed: {str(e)}") from e

            # Exponential backoff for connection errors
            wait_time = min(2 ** retry_count, 32)
            print(
                f"Connection error (attempt {retry_count + 1}/{max_retries}). "
                f"Retrying in {wait_time} seconds..."
            )
            await asyncio.sleep(wait_time)
            retry_count += 1

    raise ShopifyRateLimitError("Max retries exceeded")


def _parse_link_header(link_header: str) -> str | None:
    """
    Parse Link header to extract next page URL.

    Shopify uses Link headers for cursor-based pagination:
    Link: <https://shop.myshopify.com/admin/api/2024-01/orders.json?page_info=xyz>; rel="next"

    Args:
        link_header: Link header string from response

    Returns:
        Next page URL if exists, None otherwise
    """
    if not link_header:
        return None

    # Split multiple links
    links = link_header.split(",")

    for link in links:
        parts = link.strip().split(";")
        if len(parts) != 2:
            continue

        url_part = parts[0].strip()
        rel_part = parts[1].strip()

        # Check if this is the 'next' link
        if 'rel="next"' in rel_part or "rel='next'" in rel_part:
            # Extract URL from angle brackets
            if url_part.startswith("<") and url_part.endswith(">"):
                return url_part[1:-1]

    return None
