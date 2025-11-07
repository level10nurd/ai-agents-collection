# InfoPlus WMS Integration Summary

## Executive Summary

InfoPlus WMS integration presents unique challenges due to its **distributed inventory model** where inventory is continuously calculated rather than stored in a single table. This summary provides the implementation strategy for Task 2.4.

## Key Discovery

**InfoPlus does NOT have a simple "inventory" table.** Inventory must be calculated from:

1. **Inventory Snapshots** (daily snapshots)
2. **Inventory Movements** (real-time changes)
3. **Items** (product catalog)
4. **Recipes/Kits** (assembled products)
5. **Inventory Receipts** (incoming stock)

This significantly impacts our implementation approach.

## Implementation Path for Task 2.4

### Phase 1: Direct API Integration (Current PR)

**Goal**: Implement `agents/cfo/tools/infoplus.py` with three required functions

#### Required Functions

1. **`fetch_inventory_levels(api_key, account_no)`**
   - Strategy: Use daily inventory snapshots + item master data
   - Join snapshots with item catalog for complete picture
   - Calculate total inventory value
   - Return: List of standardized `InventoryLevel` objects

2. **`fetch_fulfillment_status(api_key, account_no, start_date, end_date)`**
   - Strategy: Query orders + fulfillment plans by date range
   - Calculate fulfillment metrics (shipped vs pending)
   - Return: List of standardized `FulfillmentStatus` objects

3. **`fetch_shipping_metrics(api_key, account_no, start_date, end_date)`**
   - Strategy: Query load content (shipments) by date range
   - Aggregate freight costs and carrier performance
   - Return: List of standardized `ShippingMetrics` objects

#### Technical Approach

**Pattern**: Follow Shopify integration as reference

- **HTTP Client**: `httpx.AsyncClient` (async/await pattern)
- **Authentication**: API-Key header
- **Rate Limiting**: Exponential backoff with retry logic (500/min limit)
- **Pagination**: Offset-based pagination (max 500 records/page)
- **Error Handling**: Custom exceptions (`InfoPlusAPIError`, `InfoPlusRateLimitError`)
- **Data Models**: Pydantic models for transformation
- **Testing**: Comprehensive unit tests with mocked responses

#### File Structure

```
agents/cfo/tools/
├── infoplus.py              # Main implementation
└── __init__.py              # Export functions

tests/cfo/test_tools/
└── test_infoplus.py         # Unit tests
```

### Phase 2: MCP Server Integration (Future Enhancement)

**Status**: MCP server being uploaded (per PR comment)

**Benefits**:
- Centralized API access and authentication
- Intelligent caching (reduce API calls)
- Complex joins handled server-side
- Resource abstraction layer

**Integration**: Once MCP server is available, can refactor to use `mcp_client.py` with fallback to direct API

## Implementation Timeline

### Week 1: Core Implementation

**Day 1-2**: Setup and Inventory Levels
- [ ] Create `infoplus.py` with base client setup
- [ ] Implement authentication and rate limiting
- [ ] Implement `fetch_inventory_levels()`
- [ ] Create Pydantic data models
- [ ] Basic unit tests for inventory

**Day 3-4**: Fulfillment Status
- [ ] Implement `fetch_fulfillment_status()`
- [ ] Add fulfillment metrics calculations
- [ ] Unit tests for fulfillment

**Day 5**: Shipping Metrics
- [ ] Implement `fetch_shipping_metrics()`
- [ ] Add shipping cost aggregations
- [ ] Unit tests for shipping

### Week 2: Polish and Documentation

**Day 6-7**: Testing and Error Handling
- [ ] Comprehensive unit test coverage
- [ ] Edge cases and error scenarios
- [ ] Rate limit retry testing
- [ ] Pagination testing

**Day 8**: Documentation
- [ ] Update docstrings with examples
- [ ] Create usage guide
- [ ] Document known limitations
- [ ] Integration patterns

## Critical Decisions

### Decision 1: Inventory Data Source

**Options**:
1. Use daily inventory snapshots (simpler)
2. Calculate from inventory movements (real-time)
3. Hybrid approach

**Recommendation**: Start with **daily snapshots** (Option 1)

**Rationale**:
- Snapshots are pre-calculated (faster)
- Good enough for CFO analysis (daily granularity)
- Easier to implement and test
- Can enhance later with real-time movements

### Decision 2: Item Data Joining

**Challenge**: Inventory snapshots don't include SKU details

**Solution**: Two-step fetch process
1. Fetch inventory snapshots
2. Fetch item master data
3. Join client-side by `itemNoId`

**Alternative**: Create MCP server endpoint that handles join server-side

### Decision 3: Fulfillment Metrics

**What to Calculate**:
- Total orders by status
- Average fulfillment time
- On-time delivery rate
- Orders shipped vs pending ratio

**Complexity**: Requires joining orders + fulfillment plans

**Approach**: 
- Fetch orders in date range
- Optionally fetch fulfillment plans for details
- Calculate metrics from order status

### Decision 4: Shipping Cost Aggregation

**Data Source**: Load Content (shipments)

**Metrics**:
- Total freight cost
- Average cost per shipment
- Cost by carrier
- Weight and delivery performance

## Data Models

### Core Pydantic Models

```python
class InventoryLevel(BaseModel):
    """Standardized inventory level."""
    sku: str
    item_description: str
    warehouse_id: str
    quantity_available: int
    quantity_allocated: int
    quantity_on_hand: int
    snapshot_date: datetime
    unit_cost: Optional[float] = None
    total_value: Optional[float] = None

class FulfillmentStatus(BaseModel):
    """Standardized fulfillment status."""
    order_id: str
    order_number: str
    order_date: datetime
    status: str
    customer_reference: Optional[str] = None
    line_item_count: int
    ship_date: Optional[datetime] = None
    delivery_date: Optional[datetime] = None
    warehouse_id: str

class ShippingMetrics(BaseModel):
    """Standardized shipping metrics."""
    shipment_id: str
    order_id: str
    ship_date: datetime
    carrier_service: str
    tracking_number: Optional[str] = None
    freight_cost: float
    weight: Optional[float] = None
    delivery_date: Optional[datetime] = None
```

## API Endpoints Reference

### Inventory Snapshots
```
GET /inventorySnapshot
Filter: lobId eq {account_no} and snapshotDate ge {date}
```

### Items
```
GET /item
Filter: lobId eq {account_no}
```

### Orders
```
GET /order
Filter: lobId eq {account_no} and orderDate ge {start} and orderDate le {end}
```

### Load Content (Shipments)
```
GET /loadContent
Filter: lobId eq {account_no} and shipDate ge {start} and shipDate le {end}
```

## Rate Limiting Strategy

**InfoPlus Limits**:
- 500 requests/minute
- 100 requests/second (burst)

**Strategy**:
1. Implement exponential backoff on 429 errors
2. Respect `Retry-After` header
3. Add small delays between pagination requests (0.5s)
4. Max 3 retries before failing

**Implementation**:
```python
async def _make_request_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict,
    headers: dict,
    max_retries: int = 3
) -> httpx.Response:
    # Retry logic with exponential backoff
    # Same pattern as Shopify integration
```

## Error Handling

### Custom Exceptions

```python
class InfoPlusAPIError(Exception):
    """Base exception for InfoPlus API errors."""
    pass

class InfoPlusRateLimitError(InfoPlusAPIError):
    """Raised when rate limit exceeded."""
    pass

class InfoPlusAuthenticationError(InfoPlusAPIError):
    """Raised when authentication fails."""
    pass
```

### Error Scenarios

1. **Authentication Failure** (401): Invalid API key
2. **Rate Limit** (429): Too many requests
3. **Not Found** (404): Invalid account number or resource
4. **Server Error** (500): InfoPlus API issue
5. **Network Error**: Connection timeout/failure

## Testing Strategy

### Unit Test Coverage

**Required Test Cases**:

1. **Inventory Levels**:
   - ✓ Successful fetch with single page
   - ✓ Pagination across multiple pages
   - ✓ Join with item data
   - ✓ Empty inventory
   - ✓ Rate limit retry
   - ✓ API errors

2. **Fulfillment Status**:
   - ✓ Fetch orders by date range
   - ✓ Various order statuses
   - ✓ Pagination
   - ✓ Empty results
   - ✓ Date filtering

3. **Shipping Metrics**:
   - ✓ Fetch shipments by date range
   - ✓ Freight cost aggregation
   - ✓ Carrier grouping
   - ✓ Empty shipments
   - ✓ Missing freight cost handling

4. **Error Handling**:
   - ✓ Rate limit with retry
   - ✓ Max retries exceeded
   - ✓ Authentication error
   - ✓ Network error
   - ✓ Malformed response

### Test Structure

```python
# tests/cfo/test_tools/test_infoplus.py

class TestFetchInventoryLevels:
    @pytest.mark.asyncio
    async def test_fetch_inventory_success(self, mock_snapshot_response):
        # Test implementation
        pass

class TestFetchFulfillmentStatus:
    @pytest.mark.asyncio
    async def test_fetch_fulfillment_success(self, mock_order_response):
        # Test implementation
        pass

class TestFetchShippingMetrics:
    @pytest.mark.asyncio
    async def test_fetch_shipping_success(self, mock_shipment_response):
        # Test implementation
        pass
```

## Success Criteria (from PR)

- [x] Documentation complete (`INFOPLUS_WMS_INTEGRATION.md`, `INFOPLUS_INTEGRATION_SUMMARY.md`)
- [ ] `fetch_inventory_levels(api_key, account_no)` implemented
- [ ] `fetch_fulfillment_status(api_key, account_no, start_date, end_date)` implemented
- [ ] `fetch_shipping_metrics(api_key, account_no, start_date, end_date)` implemented
- [ ] Unit tests passing
- [ ] Inventory and fulfillment data retrieval working

## Known Limitations

1. **Daily Inventory Granularity**: Using snapshots (not real-time)
2. **Client-Side Joins**: Item data joined in Python (not DB)
3. **No Kit/Recipe Support**: Initial version doesn't handle assembled products
4. **Single Warehouse**: Doesn't aggregate across warehouses (yet)
5. **Basic Metrics**: Advanced analytics deferred to future phases

## Future Enhancements

### Phase 2: MCP Server Integration
- Use MCP server for InfoPlus queries
- Server-side joins and caching
- Resource abstraction

### Phase 3: Advanced Features
- Real-time inventory from movements
- Kit/recipe support (BOM calculations)
- Multi-warehouse aggregation
- Advanced fulfillment metrics
- Carrier performance analytics

### Phase 4: Optimization
- Caching layer (Redis)
- Batch queries for efficiency
- Historical trend analysis
- Predictive inventory modeling

## MCP Server Integration Notes

**Current Status**: MCP server being uploaded (per PR comment)

**When MCP Server is Ready**:

1. **Inventory Resource**:
   ```
   URI: infoplus://inventory_levels
   Params: account_no, date
   Returns: Pre-joined inventory with item data
   ```

2. **Fulfillment Resource**:
   ```
   URI: infoplus://fulfillment_status
   Params: account_no, start_date, end_date
   Returns: Order fulfillment data with metrics
   ```

3. **Shipping Resource**:
   ```
   URI: infoplus://shipping_metrics
   Params: account_no, start_date, end_date
   Returns: Aggregated shipping data
   ```

**Integration Pattern**:
```python
from agents.cfo.tools.mcp_client import fetch_mcp_resource

# Use MCP server if available, fallback to direct API
try:
    inventory = await fetch_mcp_resource(
        server="infoplus",
        uri="infoplus://inventory_levels",
        params={"account_no": account_no}
    )
except MCPClientError:
    # Fallback to direct API
    inventory = await fetch_inventory_levels_direct(api_key, account_no)
```

## Dependencies

### Required Packages
```txt
httpx>=0.24.0         # Async HTTP client
pydantic>=2.0.0       # Data validation
pytest>=7.4.0         # Testing
pytest-asyncio>=0.21.0 # Async test support
```

### Optional (for MCP)
```txt
supabase>=2.0.0       # MCP server uses Supabase
```

## Code Review Checklist

Before submitting PR:

- [ ] All three functions implemented
- [ ] Comprehensive docstrings with examples
- [ ] Type hints on all functions
- [ ] Error handling with custom exceptions
- [ ] Rate limiting with retry logic
- [ ] Pagination handling
- [ ] Unit tests for all functions
- [ ] Unit tests for error scenarios
- [ ] Test coverage > 80%
- [ ] No hardcoded credentials
- [ ] Follows Shopify integration pattern
- [ ] Pydantic models documented
- [ ] README/docstring usage examples

## Questions for Further Research

1. **InfoPlus SDK**: Is `infoplus-python` SDK mature enough to use vs raw REST API?
2. **MCP Server**: What tables/endpoints are exposed? How to query?
3. **Kits/Recipes**: Should we support in v1 or defer?
4. **Real-time Inventory**: Is snapshot granularity sufficient for CFO use cases?
5. **Multi-warehouse**: Should inventory aggregate across warehouses?
6. **Historical Data**: How far back should date ranges support?

## References

- Main Documentation: `INFOPLUS_WMS_INTEGRATION.md`
- Shopify Integration: `agents/cfo/tools/shopify.py` (pattern reference)
- Test Reference: `tests/cfo/test_tools/test_shopify.py`
- MCP Client: `agents/cfo/tools/mcp_client.py`
- PR Discussion: GitHub PR Task 2.4

## Next Action Items

1. ✅ **Documentation**: Create integration docs
2. ⏳ **Implementation**: Start coding `infoplus.py`
3. ⏳ **Testing**: Write unit tests
4. ⏳ **Validation**: Test with real API (if credentials available)
5. ⏳ **MCP**: Research MCP server status and capabilities

---

## Implementation Plan Summary

**Approach**: Direct API integration following Shopify pattern

**Complexity**: Medium-High (due to inventory join requirements)

**Timeline**: 1-2 weeks (implementation + testing)

**Risk**: Inventory calculation complexity, API rate limits

**Mitigation**: Use snapshots (pre-calculated), implement robust retry logic

**Success**: Three functions working, tests passing, data retrievable
