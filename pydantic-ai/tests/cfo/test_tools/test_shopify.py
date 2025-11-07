"""
Unit tests for Shopify integration module.

Tests fetch_orders, fetch_customers, and calculate_customer_metrics functions
with mocked httpx client.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from agents.cfo.tools.shopify import (
    fetch_orders,
    fetch_customers,
    calculate_customer_metrics,
    ShopifyAPIError,
    ShopifyRateLimitError,
    _parse_link_header,
)


# Test Data Fixtures
@pytest.fixture
def mock_orders_response():
    """Mock Shopify orders API response."""
    return {
        "orders": [
            {
                "id": 123456789,
                "total_price": "150.00",
                "created_at": "2024-01-15T10:30:00-05:00",
                "line_items": [
                    {
                        "id": 1,
                        "product_id": 111,
                        "quantity": 2,
                        "price": "50.00",
                        "name": "Product A",
                    },
                    {
                        "id": 2,
                        "product_id": 222,
                        "quantity": 1,
                        "price": "50.00",
                        "name": "Product B",
                    },
                ],
            },
            {
                "id": 123456790,
                "total_price": "250.50",
                "created_at": "2024-01-16T14:20:00-05:00",
                "line_items": [
                    {
                        "id": 3,
                        "product_id": 333,
                        "quantity": 1,
                        "price": "250.50",
                        "name": "Product C",
                    }
                ],
            },
        ]
    }


@pytest.fixture
def mock_customers_response():
    """Mock Shopify customers API response."""
    return {
        "customers": [
            {
                "id": 111,
                "orders_count": 3,
                "total_spent": "450.00",
                "email": "customer1@example.com",
                "first_name": "John",
                "last_name": "Doe",
            },
            {
                "id": 222,
                "orders_count": 1,
                "total_spent": "150.00",
                "email": "customer2@example.com",
                "first_name": "Jane",
                "last_name": "Smith",
            },
            {
                "id": 333,
                "orders_count": 5,
                "total_spent": "1200.00",
                "email": "customer3@example.com",
                "first_name": "Bob",
                "last_name": "Johnson",
            },
        ]
    }


class TestFetchOrders:
    """Test suite for fetch_orders function."""

    @pytest.mark.asyncio
    async def test_fetch_orders_success(self, mock_orders_response):
        """Test successful order fetching."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_orders_response
        mock_response.headers = {}  # No pagination

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            orders = await fetch_orders(
                api_key="test_key",
                shop_name="testshop",
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

            assert len(orders) == 2
            assert orders[0]["order_id"] == 123456789
            assert orders[0]["total_price"] == 150.0
            assert orders[0]["created_at"] == "2024-01-15T10:30:00-05:00"
            assert len(orders[0]["line_items"]) == 2
            assert orders[1]["order_id"] == 123456790
            assert orders[1]["total_price"] == 250.5

    @pytest.mark.asyncio
    async def test_fetch_orders_with_pagination(self, mock_orders_response):
        """Test order fetching with pagination using Link header."""
        # First page response
        mock_response_page1 = MagicMock()
        mock_response_page1.status_code = 200
        mock_response_page1.json.return_value = mock_orders_response
        mock_response_page1.headers = {
            "Link": '<https://testshop.myshopify.com/admin/api/2024-01/orders.json?page_info=abc123>; rel="next"'
        }

        # Second page response (last page)
        mock_response_page2 = MagicMock()
        mock_response_page2.status_code = 200
        mock_response_page2.json.return_value = {
            "orders": [
                {
                    "id": 999,
                    "total_price": "100.00",
                    "created_at": "2024-01-20T10:00:00-05:00",
                    "line_items": [],
                }
            ]
        }
        mock_response_page2.headers = {}  # No next page

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(
                side_effect=[mock_response_page1, mock_response_page2]
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            orders = await fetch_orders(
                api_key="test_key",
                shop_name="testshop",
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

            # Should have orders from both pages
            assert len(orders) == 3
            assert orders[2]["order_id"] == 999

    @pytest.mark.asyncio
    async def test_fetch_orders_rate_limit_retry(self, mock_orders_response):
        """Test rate limit handling with retry."""
        # First request hits rate limit
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}

        # Second request succeeds
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = mock_orders_response
        mock_response_200.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(
                side_effect=[mock_response_429, mock_response_200]
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Should succeed after retry
            orders = await fetch_orders(
                api_key="test_key",
                shop_name="testshop",
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

            assert len(orders) == 2
            assert mock_client.request.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_orders_rate_limit_max_retries_exceeded(self):
        """Test rate limit error when max retries exceeded."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response_429)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ShopifyRateLimitError) as excinfo:
                await fetch_orders(
                    api_key="test_key",
                    shop_name="testshop",
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                )

            assert "Rate limit exceeded" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_fetch_orders_http_error(self):
        """Test handling of HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ShopifyAPIError) as excinfo:
                await fetch_orders(
                    api_key="invalid_key",
                    shop_name="testshop",
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                )

            assert "HTTP error 401" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_fetch_orders_empty_response(self):
        """Test handling of empty orders list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"orders": []}
        mock_response.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            orders = await fetch_orders(
                api_key="test_key",
                shop_name="testshop",
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

            assert len(orders) == 0


class TestFetchCustomers:
    """Test suite for fetch_customers function."""

    @pytest.mark.asyncio
    async def test_fetch_customers_success(self, mock_customers_response):
        """Test successful customer fetching."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_customers_response
        mock_response.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            customers = await fetch_customers(
                api_key="test_key", shop_name="testshop"
            )

            assert len(customers) == 3
            assert customers[0]["customer_id"] == 111
            assert customers[0]["orders_count"] == 3
            assert customers[0]["total_spent"] == 450.0
            assert customers[0]["email"] == "customer1@example.com"

    @pytest.mark.asyncio
    async def test_fetch_customers_with_pagination(self, mock_customers_response):
        """Test customer fetching with pagination."""
        # First page
        mock_response_page1 = MagicMock()
        mock_response_page1.status_code = 200
        mock_response_page1.json.return_value = mock_customers_response
        mock_response_page1.headers = {
            "Link": '<https://testshop.myshopify.com/admin/api/2024-01/customers.json?page_info=xyz>; rel="next"'
        }

        # Second page
        mock_response_page2 = MagicMock()
        mock_response_page2.status_code = 200
        mock_response_page2.json.return_value = {
            "customers": [
                {
                    "id": 444,
                    "orders_count": 2,
                    "total_spent": "300.00",
                    "email": "customer4@example.com",
                    "first_name": "Alice",
                    "last_name": "Williams",
                }
            ]
        }
        mock_response_page2.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(
                side_effect=[mock_response_page1, mock_response_page2]
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            customers = await fetch_customers(
                api_key="test_key", shop_name="testshop"
            )

            assert len(customers) == 4
            assert customers[3]["customer_id"] == 444

    @pytest.mark.asyncio
    async def test_fetch_customers_rate_limit_retry(self, mock_customers_response):
        """Test rate limit handling for customers endpoint."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = mock_customers_response
        mock_response_200.headers = {}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(
                side_effect=[mock_response_429, mock_response_200]
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            customers = await fetch_customers(
                api_key="test_key", shop_name="testshop"
            )

            assert len(customers) == 3


class TestCalculateCustomerMetrics:
    """Test suite for calculate_customer_metrics function."""

    def test_calculate_metrics_success(self):
        """Test successful metrics calculation."""
        orders = [
            {"order_id": 1, "total_price": 100.0, "created_at": "2024-01-01", "line_items": []},
            {"order_id": 2, "total_price": 200.0, "created_at": "2024-01-02", "line_items": []},
            {"order_id": 3, "total_price": 150.0, "created_at": "2024-01-03", "line_items": []},
        ]

        customers = [
            {"customer_id": 1, "orders_count": 2, "total_spent": 300.0},
            {"customer_id": 2, "orders_count": 1, "total_spent": 150.0},
        ]

        metrics = calculate_customer_metrics(orders, customers)

        assert metrics["total_orders"] == 3
        assert metrics["total_revenue"] == 450.0
        assert metrics["average_order_value"] == 150.0
        assert metrics["unique_customers"] == 2
        assert metrics["orders_per_customer"] == 1.5
        assert metrics["repeat_purchase_rate"] == 50.0  # 1 out of 2 customers
        assert metrics["average_customer_lifetime_value"] == 225.0

    def test_calculate_metrics_no_repeat_customers(self):
        """Test metrics when all customers have single order."""
        orders = [
            {"order_id": 1, "total_price": 100.0, "created_at": "2024-01-01", "line_items": []},
            {"order_id": 2, "total_price": 200.0, "created_at": "2024-01-02", "line_items": []},
        ]

        customers = [
            {"customer_id": 1, "orders_count": 1, "total_spent": 100.0},
            {"customer_id": 2, "orders_count": 1, "total_spent": 200.0},
        ]

        metrics = calculate_customer_metrics(orders, customers)

        assert metrics["repeat_purchase_rate"] == 0.0

    def test_calculate_metrics_all_repeat_customers(self):
        """Test metrics when all customers have multiple orders."""
        orders = [
            {"order_id": 1, "total_price": 100.0, "created_at": "2024-01-01", "line_items": []},
            {"order_id": 2, "total_price": 200.0, "created_at": "2024-01-02", "line_items": []},
            {"order_id": 3, "total_price": 150.0, "created_at": "2024-01-03", "line_items": []},
        ]

        customers = [
            {"customer_id": 1, "orders_count": 2, "total_spent": 300.0},
            {"customer_id": 2, "orders_count": 3, "total_spent": 450.0},
        ]

        metrics = calculate_customer_metrics(orders, customers)

        assert metrics["repeat_purchase_rate"] == 100.0

    def test_calculate_metrics_empty_data(self):
        """Test metrics with empty orders and customers."""
        orders = []
        customers = []

        metrics = calculate_customer_metrics(orders, customers)

        assert metrics["total_orders"] == 0
        assert metrics["total_revenue"] == 0.0
        assert metrics["average_order_value"] == 0.0
        assert metrics["unique_customers"] == 0
        assert metrics["orders_per_customer"] == 0.0
        assert metrics["repeat_purchase_rate"] == 0.0
        assert metrics["average_customer_lifetime_value"] == 0.0

    def test_calculate_metrics_rounding(self):
        """Test that metrics are properly rounded to 2 decimal places."""
        orders = [
            {"order_id": 1, "total_price": 33.333, "created_at": "2024-01-01", "line_items": []},
            {"order_id": 2, "total_price": 66.667, "created_at": "2024-01-02", "line_items": []},
        ]

        customers = [
            {"customer_id": 1, "orders_count": 1, "total_spent": 33.333},
            {"customer_id": 2, "orders_count": 1, "total_spent": 66.667},
            {"customer_id": 3, "orders_count": 2, "total_spent": 100.0},
        ]

        metrics = calculate_customer_metrics(orders, customers)

        # Check all values are rounded to 2 decimals
        assert isinstance(metrics["average_order_value"], (int, float))
        assert metrics["average_order_value"] == 50.0
        assert metrics["total_revenue"] == 100.0


class TestParseLinkHeader:
    """Test suite for _parse_link_header helper function."""

    def test_parse_link_header_with_next(self):
        """Test parsing Link header with next relation."""
        link_header = '<https://testshop.myshopify.com/admin/api/2024-01/orders.json?page_info=abc123>; rel="next"'
        next_url = _parse_link_header(link_header)

        assert next_url == "https://testshop.myshopify.com/admin/api/2024-01/orders.json?page_info=abc123"

    def test_parse_link_header_multiple_links(self):
        """Test parsing Link header with multiple relations."""
        link_header = (
            '<https://testshop.myshopify.com/admin/api/2024-01/orders.json?page_info=prev>; rel="previous", '
            '<https://testshop.myshopify.com/admin/api/2024-01/orders.json?page_info=next>; rel="next"'
        )
        next_url = _parse_link_header(link_header)

        assert next_url == "https://testshop.myshopify.com/admin/api/2024-01/orders.json?page_info=next"

    def test_parse_link_header_no_next(self):
        """Test parsing Link header without next relation."""
        link_header = '<https://testshop.myshopify.com/admin/api/2024-01/orders.json?page_info=prev>; rel="previous"'
        next_url = _parse_link_header(link_header)

        assert next_url is None

    def test_parse_link_header_empty(self):
        """Test parsing empty Link header."""
        next_url = _parse_link_header("")

        assert next_url is None

    def test_parse_link_header_none(self):
        """Test parsing None Link header."""
        next_url = _parse_link_header(None)

        assert next_url is None

    def test_parse_link_header_malformed(self):
        """Test parsing malformed Link header."""
        link_header = "not a valid link header"
        next_url = _parse_link_header(link_header)

        assert next_url is None
