# Shopify Integration for CFO Agent

## Overview

This module provides Shopify Admin API integration for fetching orders and customer data, with proper pagination handling and rate limit management using the leaky bucket algorithm.

## Features

- **`fetch_orders()`** - Fetch orders with date filters, pagination, and rate limiting
- **`fetch_customers()`** - Fetch customers with pagination
- **`calculate_customer_metrics()`** - Calculate CAC-related metrics from orders and customers data

## Installation

The required dependencies are already included in `requirements.txt`:
- `httpx>=0.24.0` - Async HTTP client

## Usage

### Basic Example

```python
import asyncio
from agents.cfo.tools.shopify import (
    fetch_orders,
    fetch_customers,
    calculate_customer_metrics
)

async def analyze_shopify_data():
    # Fetch orders for January 2024
    orders = await fetch_orders(
        api_key="shpat_xxxxx",
        shop_name="myshop",
        start_date="2024-01-01",
        end_date="2024-01-31",
        limit=250
    )
    
    # Fetch customers
    customers = await fetch_customers(
        api_key="shpat_xxxxx",
        shop_name="myshop",
        limit=250
    )
    
    # Calculate metrics
    metrics = calculate_customer_metrics(orders, customers)
    
    print(f"Average Order Value: ${metrics['average_order_value']:.2f}")
    print(f"Repeat Purchase Rate: {metrics['repeat_purchase_rate']:.1f}%")
    print(f"Total Revenue: ${metrics['total_revenue']:,.2f}")

asyncio.run(analyze_shopify_data())
```

## API Reference

### `fetch_orders(api_key, shop_name, start_date, end_date, limit=250)`

Fetch orders from Shopify Admin API with pagination and rate limiting.

**Parameters:**
- `api_key` (str): Shopify API access token (e.g., "shpat_xxxxx")
- `shop_name` (str): Shopify shop name without .myshopify.com (e.g., "myshop")
- `start_date` (str): Start date in ISO 8601 format (YYYY-MM-DD)
- `end_date` (str): End date in ISO 8601 format (YYYY-MM-DD)
- `limit` (int): Orders per page, max 250 (default: 250)

**Returns:**
List of dictionaries with:
- `order_id`: Shopify order ID
- `total_price`: Total order price (float)
- `created_at`: Order creation timestamp
- `line_items`: List of line items with id, product_id, quantity, price, name

**Rate Limiting:**
- Automatically handles 429 responses with exponential backoff
- Respects `Retry-After` header
- Max 3 retry attempts

### `fetch_customers(api_key, shop_name, limit=250)`

Fetch customers from Shopify Admin API with pagination.

**Parameters:**
- `api_key` (str): Shopify API access token
- `shop_name` (str): Shopify shop name without .myshopify.com
- `limit` (int): Customers per page, max 250 (default: 250)

**Returns:**
List of dictionaries with:
- `customer_id`: Shopify customer ID
- `orders_count`: Total number of orders
- `total_spent`: Total amount spent (float)
- `email`: Customer email
- `first_name`: First name
- `last_name`: Last name

### `calculate_customer_metrics(orders, customers)`

Calculate customer acquisition cost (CAC) related metrics.

**Parameters:**
- `orders`: List of orders from `fetch_orders()`
- `customers`: List of customers from `fetch_customers()`

**Returns:**
Dictionary with:
- `average_order_value`: Average value per order
- `repeat_purchase_rate`: Percentage of customers with multiple orders
- `total_orders`: Total number of orders
- `total_revenue`: Total revenue from all orders
- `unique_customers`: Number of unique customers
- `orders_per_customer`: Average orders per customer
- `average_customer_lifetime_value`: Average lifetime value per customer

## Error Handling

The module includes custom exceptions:

- `ShopifyAPIError` - Base exception for all API errors
- `ShopifyRateLimitError` - Raised when rate limits are exceeded after retries

Example error handling:

```python
from agents.cfo.tools.shopify import (
    fetch_orders,
    ShopifyAPIError,
    ShopifyRateLimitError
)

try:
    orders = await fetch_orders(...)
except ShopifyRateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except ShopifyAPIError as e:
    print(f"API error: {e}")
```

## Rate Limits

Shopify enforces the following rate limits:
- **REST API**: 40 requests per app per store per minute (2 requests/second)
- **Leaky Bucket Algorithm**: Points refill gradually
- **Headers**: Monitor `X-Shopify-Shop-Api-Call-Limit` header

The implementation automatically:
1. Handles 429 status codes
2. Respects `Retry-After` headers
3. Implements exponential backoff
4. Adds small delays between pagination requests (0.5s)

## Pagination

The implementation uses Shopify's cursor-based pagination via Link headers:
- Automatically follows `rel="next"` links
- No manual page tracking required
- Handles both old (page-based) and new (cursor-based) pagination

## Testing

Run the test suite:

```bash
cd /workspace/pydantic-ai
python3 -m pytest tests/cfo/test_tools/test_shopify.py -v
```

All tests use mocked `httpx` clients - no real API calls are made.

Test coverage includes:
- ✓ Successful data fetching
- ✓ Pagination handling
- ✓ Rate limit retry logic
- ✓ HTTP error handling
- ✓ Empty response handling
- ✓ Metrics calculation edge cases

## Example Script

See `examples/scripts/shopify_example.py` for a complete working example.

Run with:

```bash
export SHOPIFY_API_KEY='shpat_xxxxx'
export SHOPIFY_SHOP_NAME='myshop'
python3 examples/scripts/shopify_example.py
```

## Reference Documentation

For detailed Shopify API documentation, see:
- `PRPs/04_api_integrations/financial_api_integration_research.md`
- Official Shopify API: https://shopify.dev/docs/api/admin-rest

## Notes

- Uses Shopify Admin API version 2024-01
- All monetary values returned as floats
- Dates use ISO 8601 format
- Timeout set to 30 seconds per request
- Async implementation using `httpx.AsyncClient`
