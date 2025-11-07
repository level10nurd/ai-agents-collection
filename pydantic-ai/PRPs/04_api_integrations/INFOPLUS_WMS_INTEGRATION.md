# InfoPlus WMS Integration Documentation

## Overview

InfoPlus is a Warehouse Management System (WMS) API that provides inventory, fulfillment, and shipping data. This document outlines the integration approach for the AI CFO Agent system.

## Critical Discovery: Inventory Data Model

**Important Note**: InfoPlus does NOT have a simple "inventory" table. Inventory is **continuously calculated** from multiple sources:

### Inventory Calculation Sources

1. **Inventory Snapshots** (`inventorySnapshots`)
   - Daily snapshots of inventory levels
   - Stored once per day
   - Provides historical inventory levels

2. **Inventory Movements** (`inventoryMovements`)
   - Real-time inventory changes
   - Can be used to calculate current inventory
   - More granular than snapshots

3. **Items** (`items`)
   - Master product/SKU catalog
   - Product definitions and metadata

4. **Recipes & Kits** (`recipes`, `kits`)
   - Bill of materials (BOM) for assembled products
   - Component relationships
   - Required for accurate inventory calculations

5. **Inventory Receipts** (`inventoryReceipts`)
   - Incoming inventory records
   - Purchase order receipts
   - Stock additions

## API Documentation

### Base URL
```
https://api.infopluscommerce.com/v3.0
```

### Authentication
- **Method**: API Key Authentication
- **Header**: `API-Key: your_api_key_here`
- **Account Number**: Required in most API calls as a filter parameter

### Rate Limits
- **Standard**: 500 requests per minute per API key
- **Burst**: 100 requests per second
- **Error Response**: HTTP 429 "Too Many Requests"
- **Retry-After Header**: Provided in 429 responses

## Key Endpoints for CFO Use

### 1. Inventory Snapshots
```
GET /inventorySnapshot
```

**Purpose**: Daily inventory levels snapshot

**Query Parameters**:
- `filter`: Filter by lobId, warehouseId, itemNoId, date
- `page`: Page number (default: 1)
- `limit`: Records per page (max: 500)
- `sort`: Sort field and direction

**Example**:
```python
GET /inventorySnapshot?filter=lobId eq {account_no} and snapshotDate ge 2024-01-01&limit=250
```

**Response Fields**:
- `id`: Snapshot ID
- `lobId`: Line of business ID (account number)
- `warehouseId`: Warehouse identifier
- `itemNoId`: Item/SKU ID
- `snapshotDate`: Date of snapshot
- `quantity`: Available quantity
- `allocatedQuantity`: Reserved quantity
- `onHandQuantity`: Total on-hand

### 2. Items (Products/SKUs)
```
GET /item
```

**Purpose**: Master product catalog

**Query Parameters**:
- `filter`: Filter by lobId, SKU, status
- `page`: Page number
- `limit`: Records per page (max: 500)

**Example**:
```python
GET /item?filter=lobId eq {account_no}&limit=250
```

**Response Fields**:
- `id`: Item ID
- `lobId`: Line of business ID
- `sku`: Stock keeping unit
- `itemDescription`: Product description
- `unitsPerWrap`: Pack size
- `majorGroupId`: Product category
- `status`: Active/Inactive status
- `unitCost`: Cost per unit

### 3. Orders (Fulfillment Status)
```
GET /order
```

**Purpose**: Order fulfillment data

**Query Parameters**:
- `filter`: Filter by lobId, orderDate, status
- `page`: Page number
- `limit`: Records per page (max: 500)

**Example**:
```python
GET /order?filter=lobId eq {account_no} and orderDate ge 2024-01-01 and orderDate le 2024-01-31&limit=250
```

**Response Fields**:
- `id`: Order ID
- `orderNo`: Order number
- `lobId`: Line of business ID
- `orderDate`: Order date
- `status`: Order status (Pending, Processing, Shipped, etc.)
- `customerOrderNo`: Customer reference
- `numberOfLines`: Line item count
- `shipDate`: Ship date
- `deliveryDate`: Delivery date

### 4. Fulfillment Plans
```
GET /fulfillmentPlan
```

**Purpose**: Detailed fulfillment execution data

**Query Parameters**:
- `filter`: Filter by lobId, warehouseId, createDate
- `page`: Page number
- `limit`: Records per page

**Example**:
```python
GET /fulfillmentPlan?filter=lobId eq {account_no} and createDate ge 2024-01-01&limit=250
```

**Response Fields**:
- `id`: Fulfillment plan ID
- `orderId`: Associated order ID
- `warehouseId`: Fulfilling warehouse
- `status`: Execution status
- `pickScanScheme`: Picking methodology
- `completeDate`: Completion timestamp

### 5. Shipments (Shipping Metrics)
```
GET /loadContent
```

**Purpose**: Shipping and carrier data

**Query Parameters**:
- `filter`: Filter by lobId, shipDate
- `page`: Page number
- `limit`: Records per page

**Example**:
```python
GET /loadContent?filter=lobId eq {account_no} and shipDate ge 2024-01-01&limit=250
```

**Response Fields**:
- `id`: Shipment ID
- `lobId`: Line of business ID
- `orderId`: Related order
- `shipDate`: Ship date
- `carrierServiceId`: Carrier and service level
- `trackingNo`: Tracking number
- `freightCost`: Shipping cost
- `weight`: Package weight
- `deliveryDate`: Delivery date

### 6. Inventory Receipts
```
GET /inventoryReceipt
```

**Purpose**: Incoming inventory tracking

**Query Parameters**:
- `filter`: Filter by lobId, receivedDate
- `page`: Page number
- `limit`: Records per page

**Example**:
```python
GET /inventoryReceipt?filter=lobId eq {account_no} and receivedDate ge 2024-01-01&limit=250
```

**Response Fields**:
- `id`: Receipt ID
- `lobId`: Line of business ID
- `receivedDate`: Date received
- `warehouseId`: Receiving warehouse
- `vendorId`: Vendor identifier
- `poNo`: Purchase order number
- `receivedQuantity`: Quantity received

## InfoPlus SDK

### Installation
```bash
pip install infoplus-python
```

### Python SDK Usage

**Authentication**:
```python
from infoplus import Configuration, ApiClient
from infoplus.api import ItemApi, InventorySnapshotApi, OrderApi

# Configure API client
configuration = Configuration()
configuration.api_key['API-Key'] = 'your_api_key'
configuration.host = 'https://api.infopluscommerce.com/v3.0'

api_client = ApiClient(configuration)
```

**Example: Fetch Items**:
```python
item_api = ItemApi(api_client)

# Get items for account
filter_query = f"lobId eq {account_no}"
items = item_api.get_item_by_filter(filter=filter_query, limit=250)

for item in items:
    print(f"SKU: {item.sku}, Description: {item.item_description}")
```

**Example: Fetch Inventory Snapshots**:
```python
inventory_api = InventorySnapshotApi(api_client)

# Get snapshots for date range
filter_query = f"lobId eq {account_no} and snapshotDate ge 2024-01-01"
snapshots = inventory_api.get_inventory_snapshot_by_filter(
    filter=filter_query,
    limit=250
)

for snapshot in snapshots:
    print(f"Item: {snapshot.item_no_id}, Qty: {snapshot.quantity}")
```

## Data Models for CFO Agent

### Inventory Level Model
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class InventoryLevel(BaseModel):
    """Standardized inventory level from InfoPlus."""
    sku: str
    item_description: str
    warehouse_id: str
    quantity_available: int
    quantity_allocated: int
    quantity_on_hand: int
    snapshot_date: datetime
    unit_cost: Optional[float] = None
    total_value: Optional[float] = None
    
    @classmethod
    def from_infoplus_snapshot(cls, snapshot, item_data):
        """Transform InfoPlus snapshot to standard format."""
        return cls(
            sku=item_data.sku,
            item_description=item_data.item_description,
            warehouse_id=snapshot.warehouse_id,
            quantity_available=snapshot.quantity,
            quantity_allocated=snapshot.allocated_quantity,
            quantity_on_hand=snapshot.on_hand_quantity,
            snapshot_date=snapshot.snapshot_date,
            unit_cost=item_data.unit_cost,
            total_value=snapshot.on_hand_quantity * (item_data.unit_cost or 0)
        )
```

### Fulfillment Status Model
```python
class FulfillmentStatus(BaseModel):
    """Standardized fulfillment status from InfoPlus."""
    order_id: str
    order_number: str
    order_date: datetime
    status: str
    customer_reference: Optional[str] = None
    line_item_count: int
    ship_date: Optional[datetime] = None
    delivery_date: Optional[datetime] = None
    warehouse_id: str
    
    @classmethod
    def from_infoplus_order(cls, order):
        """Transform InfoPlus order to standard format."""
        return cls(
            order_id=str(order.id),
            order_number=order.order_no,
            order_date=order.order_date,
            status=order.status,
            customer_reference=order.customer_order_no,
            line_item_count=order.number_of_lines,
            ship_date=order.ship_date,
            delivery_date=order.delivery_date,
            warehouse_id=str(order.warehouse_id)
        )
```

### Shipping Metrics Model
```python
class ShippingMetrics(BaseModel):
    """Standardized shipping metrics from InfoPlus."""
    shipment_id: str
    order_id: str
    ship_date: datetime
    carrier_service: str
    tracking_number: Optional[str] = None
    freight_cost: float
    weight: Optional[float] = None
    delivery_date: Optional[datetime] = None
    
    @classmethod
    def from_infoplus_load_content(cls, load_content):
        """Transform InfoPlus load content to standard format."""
        return cls(
            shipment_id=str(load_content.id),
            order_id=str(load_content.order_id),
            ship_date=load_content.ship_date,
            carrier_service=load_content.carrier_service_id,
            tracking_number=load_content.tracking_no,
            freight_cost=load_content.freight_cost or 0.0,
            weight=load_content.weight,
            delivery_date=load_content.delivery_date
        )
```

## Required Functions for Task 2.4

### 1. `fetch_inventory_levels(api_key, account_no)`
**Strategy**: Fetch latest inventory snapshot and join with item data

**Steps**:
1. Get most recent snapshot date
2. Fetch inventory snapshots for that date
3. Fetch item master data for SKU details
4. Join snapshot data with item data
5. Calculate total inventory value
6. Return standardized `InventoryLevel` objects

### 2. `fetch_fulfillment_status(api_key, account_no, start_date, end_date)`
**Strategy**: Fetch orders and fulfillment plans for date range

**Steps**:
1. Query orders by date range
2. Fetch associated fulfillment plans
3. Transform to standardized format
4. Calculate fulfillment metrics:
   - Orders shipped vs pending
   - Average fulfillment time
   - On-time delivery rate

### 3. `fetch_shipping_metrics(api_key, account_no, start_date, end_date)`
**Strategy**: Fetch shipping/load content data

**Steps**:
1. Query load content (shipments) by date range
2. Calculate shipping metrics:
   - Total freight cost
   - Average shipping cost per order
   - Carrier performance
   - On-time delivery percentage
3. Return standardized `ShippingMetrics` objects

## Rate Limit Handling

**Strategy**: Same pattern as Shopify integration

```python
async def _make_infoplus_request_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict,
    headers: dict,
    max_retries: int = 3
) -> httpx.Response:
    """Make InfoPlus API request with retry logic."""
    retry_count = 0
    
    while retry_count <= max_retries:
        response = await client.get(endpoint, params=params, headers=headers)
        
        if response.status_code == 429:
            if retry_count >= max_retries:
                raise InfoPlusRateLimitError("Rate limit exceeded")
            
            retry_after = int(response.headers.get("Retry-After", 2))
            await asyncio.sleep(retry_after)
            retry_count += 1
            continue
        
        response.raise_for_status()
        return response
    
    raise InfoPlusRateLimitError("Max retries exceeded")
```

## Pagination Handling

InfoPlus uses **offset-based pagination**:

```python
async def fetch_all_pages(api_endpoint, filter_query, limit=250):
    """Fetch all pages of data from InfoPlus API."""
    all_records = []
    page = 1
    
    while True:
        params = {
            "filter": filter_query,
            "limit": limit,
            "page": page
        }
        
        response = await _make_infoplus_request_with_retry(
            client, api_endpoint, params, headers
        )
        
        data = response.json()
        records = data.get("results", [])
        
        if not records:
            break
        
        all_records.extend(records)
        
        # Check if more pages exist
        if len(records) < limit:
            break
        
        page += 1
        await asyncio.sleep(0.5)  # Rate limit safety
    
    return all_records
```

## MCP Server Integration

Based on the PR comment, an **MCP (Model Context Protocol) server** is being added for InfoPlus.

### MCP Server Benefits
1. **Centralized API access**: Single point for InfoPlus queries
2. **Caching**: Reduce API calls with intelligent caching
3. **Resource abstraction**: Expose InfoPlus data as MCP resources
4. **Query optimization**: Complex joins handled server-side

### Using MCP Client for InfoPlus

```python
from agents.cfo.tools.mcp_client import fetch_mcp_resource

# Fetch inventory data via MCP server
inventory_data = await fetch_mcp_resource(
    server="infoplus",
    uri="infoplus://inventory_levels",
    params={
        "account_no": account_no,
        "date": "2024-01-31"
    }
)

# Fetch fulfillment status via MCP server
fulfillment_data = await fetch_mcp_resource(
    server="infoplus",
    uri="infoplus://fulfillment_status",
    params={
        "account_no": account_no,
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    }
)
```

### MCP Server Responsibilities
1. Handle InfoPlus API authentication
2. Manage rate limits and retries
3. Cache inventory snapshots (daily refresh)
4. Join inventory snapshots with item data
5. Calculate derived metrics
6. Expose standardized resources

## Implementation Approach

### Option A: Direct API Integration (Current Task)
**File**: `agents/cfo/tools/infoplus.py`

- Implement using `httpx` (like Shopify integration)
- Handle authentication, pagination, rate limits
- Transform to standardized Pydantic models
- Unit tests with mocked responses

**Pros**:
- Complete control over implementation
- Follows existing pattern (Shopify)
- No external dependencies

**Cons**:
- More complex inventory calculations
- Must handle joins client-side
- Duplicate code if MCP server exists

### Option B: MCP Server Integration (Future)
**File**: Use existing `mcp_client.py`

- Leverage MCP server for InfoPlus access
- Simpler client code
- Server handles complexity

**Pros**:
- Simplified client implementation
- Better caching and performance
- Centralized business logic

**Cons**:
- Depends on MCP server availability
- Less control over API details

### Recommended: Hybrid Approach

1. **Phase 1** (Task 2.4): Implement direct API integration
   - Complete `infoplus.py` with all required functions
   - Full unit test coverage
   - Satisfies immediate PR requirements

2. **Phase 2** (Future): Add MCP server integration
   - Create MCP server adapter
   - Refactor to use MCP when available
   - Fall back to direct API if MCP unavailable

## Testing Strategy

### Unit Tests Required
1. **Inventory Levels**:
   - Test snapshot fetching with pagination
   - Test item data joining
   - Test inventory value calculations
   - Test empty inventory scenarios

2. **Fulfillment Status**:
   - Test order fetching by date range
   - Test status transformations
   - Test fulfillment metrics calculations

3. **Shipping Metrics**:
   - Test load content fetching
   - Test freight cost aggregation
   - Test carrier performance metrics

4. **Error Handling**:
   - Test rate limit retry logic
   - Test authentication errors
   - Test network errors
   - Test empty responses

### Test Data Fixtures
```python
@pytest.fixture
def mock_inventory_snapshot_response():
    return {
        "results": [
            {
                "id": 1,
                "lobId": 12345,
                "warehouseId": "WH01",
                "itemNoId": 100,
                "snapshotDate": "2024-01-31T00:00:00Z",
                "quantity": 50,
                "allocatedQuantity": 10,
                "onHandQuantity": 60
            }
        ]
    }

@pytest.fixture
def mock_item_response():
    return {
        "results": [
            {
                "id": 100,
                "sku": "SKU-001",
                "itemDescription": "Product A",
                "unitCost": 10.50,
                "status": "Active"
            }
        ]
    }
```

## Security Best Practices

1. **API Key Storage**: Use environment variables
2. **Account Number**: Validate format before API calls
3. **Input Validation**: Validate date ranges and parameters
4. **Error Messages**: Don't expose API keys in logs
5. **HTTPS Only**: Always use secure connections

## Critical Gotchas

1. **No Direct Inventory Table**: Must calculate from snapshots/movements
2. **Daily Snapshots**: Inventory data has daily granularity
3. **Item Data Join Required**: SKU details not in snapshot data
4. **Kits & Recipes**: Complex products require BOM calculations
5. **Multiple Warehouses**: Filter by warehouse if needed
6. **Date Filters**: Use proper date format (ISO 8601)
7. **LobId Required**: All queries must filter by account number
8. **Pagination**: Max 500 records per page
9. **Rate Limits**: 500/minute, monitor closely
10. **SDK Availability**: Check InfoPlus SDK vs REST API approach

## References

- InfoPlus API Documentation: https://docs.infopluscommerce.com/
- InfoPlus Python SDK: https://github.com/infopluscommerce/infoplus-python-client
- REST API Reference: https://api.infopluscommerce.com/docs/v3.0/
- Authentication Guide: https://docs.infopluscommerce.com/authentication/

## Next Steps

1. Research InfoPlus Python SDK availability and maturity
2. Verify MCP server upload status and capabilities
3. Implement direct API integration in `infoplus.py`
4. Create comprehensive unit tests
5. Document MCP server integration path for Phase 2
