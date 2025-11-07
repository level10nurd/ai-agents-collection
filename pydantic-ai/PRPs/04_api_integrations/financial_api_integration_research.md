# Financial API Integration Research

## IMPLEMENTATION DECISION: CSV Import Approach

**Date**: 2025-11-07  
**Decision**: Postpone QuickBooks Online API integration in favor of CSV import for General Ledger and Chart of Accounts data.

**Rationale**:
- Faster MVP delivery without OAuth 2.0 complexity
- Avoids API rate limiting constraints during development
- Allows focus on core AI CFO agent capabilities
- General Ledger provides single source of truth for all financial statements
- Easier to test with sample data

**CSV Files Required**:
1. **Chart of Accounts (COA)** - Maps account codes to categories
2. **General Ledger (GL)** - All transactional data at granular level
3. **Trial Balance (TB)** - Optional validation that debits = credits

**Future**: QuickBooks API integration can be added later for automated data sync.

---

## 1. QuickBooks Online API (Future Enhancement)

### Official Documentation
- **Official Docs**: https://developer.intuit.com/app/developer/qbo/docs/develop
- **API Reference**: https://developer.intuit.com/app/developer/qbo/docs/api/accounting/all-entities/profitandloss
- **OAuth 2.0 Guide**: https://developer.intuit.com/app/developer/qbo/docs/develop/authentication-and-authorization/faq

**Note**: API integration postponed in favor of CSV import approach. See decision above.

### Python SDK
- **Library Name**: `python-quickbooks`
- **PyPI**: https://pypi.org/project/python-quickbooks/
- **GitHub**: https://github.com/ej2/python-quickbooks
- **Official Sample App**: https://github.com/IntuitDeveloper/SampleApp-QuickBooksV3API-Python

### Authentication Pattern
- **Method**: OAuth 2.0
- **Flow**:
  1. Set up application in Intuit Developer Dashboard (App Management section)
  2. Obtain Client ID and Client Secret from "Keys & OAuth" tab
  3. Use `Oauth2SessionManager` from python-quickbooks library
  4. Access tokens expire after 6 months - implement automatic refresh logic
- **Library Support**: Uses `intuit-oauth` library for OAuth handling

**Example OAuth Setup:**
```python
import shopify
from quickbooks import QuickBooks
from intuitlib.client import AuthClient

# Setup AuthClient
auth_client = AuthClient(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    redirect_uri='YOUR_REDIRECT_URI',
    environment='sandbox'  # or 'production'
)

# Create authorization URL
auth_url = auth_client.get_authorization_url(['com.intuit.quickbooks.accounting'])

# After user authorizes, exchange code for tokens
auth_client.get_bearer_token(auth_code)

# Create QuickBooks client
client = QuickBooks(
    auth_client=auth_client,
    refresh_token=refresh_token,
    company_id=company_id
)
```

### Key Endpoints for CFO Use

**Financial Reports:**
- Profit & Loss: `/v3/company/{companyId}/reports/ProfitAndLoss`
- Balance Sheet: `/v3/company/{companyId}/reports/BalanceSheet`
- Cash Flow: `/v3/company/{companyId}/reports/CashFlow`
- General Ledger: Available through Report Service

**Transactional Data:**
- Invoices: `/v3/company/{companyId}/invoice`
- Expenses/Bills: `/v3/company/{companyId}/purchase`
- Customers: `/v3/company/{companyId}/customer`
- Accounts: `/v3/company/{companyId}/account`
- Vendors: `/v3/company/{companyId}/vendor`

**Query Operations:**
- Query endpoint: `/v3/company/{companyId}/query`
- Example: `SELECT * FROM Invoice WHERE TxnDate > '2024-01-01'`

### Example Code

**1. Getting Profit & Loss Report:**
```python
from quickbooks.objects.reports import ProfitAndLoss

# Using python-quickbooks
report = client.get_report(
    'ProfitAndLoss',
    'summarize_column_by=Month&start_date=2024-01-01&end_date=2024-12-31'
)
```

**2. Fetching Invoices:**
```python
from quickbooks.objects.invoice import Invoice

# Query all invoices
invoices = Invoice.query("SELECT * FROM Invoice", qb=client)

# Get specific invoice
invoice = Invoice.get(invoice_id, qb=client)

# Filter by date
recent_invoices = Invoice.query(
    "SELECT * FROM Invoice WHERE TxnDate > '2024-01-01'",
    qb=client
)
```

**3. Creating Invoices:**
```python
from quickbooks.objects import Customer, Invoice
from quickbooks.objects.detailline import SalesItemLine, SalesItemLineDetail

# Create customer
customer = Customer()
customer.DisplayName = "Test Customer"
customer.save(qb=client)

# Create invoice
invoice = Invoice()
invoice.CustomerRef = customer.to_ref()

line = SalesItemLine()
line.Amount = 100.0
line.SalesItemLineDetail = SalesItemLineDetail()
line.SalesItemLineDetail.ItemRef = item.to_ref()

invoice.Line.append(line)
invoice.save(qb=client)
```

**4. Getting Accounts:**
```python
from quickbooks.objects.account import Account

# Get all accounts
accounts = Account.all(qb=client)

# Filter accounts
asset_accounts = Account.query(
    "SELECT * FROM Account WHERE AccountType = 'Bank'",
    qb=client
)
```

### Rate Limits
- **Standard**: 500 requests per minute per company
- **Concurrent Limit**: 10 concurrent requests maximum
- **Batch Operations**: 40 requests per minute
- **Resource-Intensive Endpoints**: 200 requests per minute
- **Error Response**: HTTP 429 "Too Many Requests"

### Rate Limit Best Practices
1. **Batch Operations**: Group related operations (40/min limit) vs individual calls
2. **Exponential Backoff**: Implement retry logic with exponential backoff on 429 errors
3. **Request Spacing**: Add delays between requests to stay under 500/min threshold
4. **Concurrent Control**: Respect 10 concurrent request ceiling
5. **Monitor Headers**: Track rate limit status in API response headers

### Critical Gotchas
1. **Token Expiration**: Access tokens expire after 6 months - must implement refresh logic
2. **Environment Differences**: Always test in Sandbox before production
3. **Company ID Required**: Every API call requires company ID in URL
4. **Batch Limitations**: Batch operations limited to 40 requests/minute (vs 500 for regular)
5. **Query Syntax**: Uses SQL-like query syntax but with limitations
6. **Date Formats**: Must use ISO 8601 format (YYYY-MM-DD)
7. **Error Handling**: QuickbooksException contains error code, message, and detail
8. **Change Data Capture**: Use CDC for efficiently tracking entity changes
9. **Credentials Security**: Store Client ID/Secret in environment variables or secure vault
10. **Rate Limit Headers**: Monitor X-RateLimit headers to prevent throttling

### Code Examples to Reference
- **Integration Tests**: https://github.com/ej2/python-quickbooks/tree/master/tests/integration
- **Report Generation**: https://github.com/simonv3/quickbooks-python/blob/master/report.py
- **Flask Sample App**: https://github.com/IntuitDeveloper/SampleApp-QuickBooksV3API-Python
- **Custom PDF Reports**: https://github.com/Gradient-s-p/custom-quickbooks-report
- **Invoice Examples**: https://github.com/ej2/python-quickbooks/blob/master/tests/integration/test_invoice.py

### Recommended for PRPs/ai_docs/
**POSTPONED** - QuickBooks API integration has been postponed in favor of CSV import approach.

**Current Implementation**: CSV Import for General Ledger and Chart of Accounts
- See `agents/cfo/tools/csv_import.py` for implementation
- See `agents/cfo/models/ledger.py` for data models
- See `tests/cfo/test_tools/test_csv_import.py` for tests

**Future Enhancement**: QuickBooks API integration can be added later for:
1. Automated data synchronization
2. Real-time financial data access
3. Direct integration without manual CSV export

**Documentation to Save for Future**:
- OAuth 2.0 authentication flow patterns
- Financial report endpoint specifications
- Rate limiting strategies and implementation
- Error handling patterns (QuickbooksException)
- Query syntax and filtering examples

---

## 2. Shopify API

### Official Documentation
- **Official Docs**: https://shopify.dev/docs/api
- **Admin REST API**: https://shopify.dev/docs/api/admin-rest
- **Admin GraphQL API**: https://shopify.dev/docs/api/admin-graphql
- **Rate Limits**: https://shopify.dev/docs/api/usage/limits

### Python SDK
- **Library Name**: `ShopifyAPI` (shopify_python_api)
- **PyPI**: https://pypi.org/project/ShopifyAPI/
- **GitHub**: https://github.com/Shopify/shopify_python_api
- **Documentation**: https://shopify.github.io/shopify_python_api/

### Authentication Pattern
- **Primary Method**: OAuth 2.0 (for public apps)
- **Alternative**: API Keys/Access Tokens (for private apps and custom apps)

**OAuth 2.0 Flow for Public Apps:**
```python
import shopify

# 1. Setup with your app credentials
shopify.Session.setup(api_key=API_KEY, secret=API_SECRET)

# 2. Create permission URL for merchant authorization
session = shopify.Session(shop_url, api_version)
permission_url = session.create_permission_url(
    scope=['read_orders', 'read_products', 'read_customers'],
    redirect_uri='https://your-app.com/callback'
)

# 3. After merchant accepts, exchange code for permanent token
session = shopify.Session(shop_url, api_version)
access_token = session.request_token(params)

# 4. Activate session for API calls
shopify.ShopifyResource.activate_session(session)
```

**Private App Authentication:**
```python
import shopify

# Direct authentication with private app credentials
session = shopify.Session(shop_url, api_version, private_app_password)
shopify.ShopifyResource.activate_session(session)
```

### Key Endpoints for CFO Use

**Orders & Revenue:**
- Orders: `/admin/api/2024-01/orders.json`
- Order Analytics: GraphQL Analytics API
- Transactions: `/admin/api/2024-01/orders/{order_id}/transactions.json`

**Products & Inventory:**
- Products: `/admin/api/2024-01/products.json`
- Inventory Levels: `/admin/api/2024-01/inventory_levels.json`
- Variants: Part of Products endpoint

**Customer Data:**
- Customers: `/admin/api/2024-01/customers.json`
- Customer Analytics: GraphQL Analytics API

**Financial Analytics:**
- Shop Analytics: GraphQL `/admin/api/2024-01/graphql.json`
- Reports: Available through Analytics API

**Webhooks for Real-Time Data:**
- Orders Created: `orders/create`
- Orders Updated: `orders/updated`
- Products Updated: `products/update`
- Customers Created: `customers/create`

### Example Code

**1. Fetching Orders with Pagination:**
```python
import shopify

# Activate session
shopify.ShopifyResource.activate_session(session)

# Get orders with pagination
page = 1
orders = shopify.Order.find(limit=250, page=page)

while len(orders) > 0:
    for order in orders:
        print(f"Order #{order.order_number}: ${order.total_price}")
        print(f"Customer: {order.customer.email}")
        print(f"Items: {len(order.line_items)}")

    page += 1
    orders = shopify.Order.find(limit=250, page=page)
```

**2. Getting Products:**
```python
# Get all products
products = shopify.Product.find()

# Get specific product
product = shopify.Product.find(product_id)

# Filter products by date
products = shopify.Product.find(
    created_at_min='2024-01-01T00:00:00-00:00'
)
```

**3. Creating Webhooks:**
```python
import shopify

# Create webhook for order creation
webhook = shopify.Webhook()
webhook.topic = "orders/create"
webhook.address = "https://your-app-url.com/webhook/order-created"
webhook.format = "json"
webhook.save()

# List all webhooks
webhooks = shopify.Webhook.find()

# Delete webhook
webhook.destroy()
```

**4. Rate Limit Handling:**
```python
import shopify
import time

def fetch_with_retry(resource_class, **kwargs):
    """Fetch resource with automatic retry on rate limit."""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            return resource_class.find(**kwargs)
        except shopify.ShopifyResource.ClientError as e:
            if e.response.code == 429:
                # Rate limit hit
                retry_after = int(e.response.headers.get('Retry-After', 2))
                print(f"Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                retry_count += 1
            else:
                raise
        except shopify.ShopifyResource.ConnectionError as e:
            print(f"Connection error: {e}")
            time.sleep(2)
            retry_count += 1

    raise Exception("Max retries exceeded")

# Usage
products = fetch_with_retry(shopify.Product, limit=250)
```

**5. Customer Analytics:**
```python
# Get customer with order data
customers = shopify.Customer.find()

for customer in customers:
    print(f"Customer: {customer.email}")
    print(f"Total spent: ${customer.total_spent}")
    print(f"Orders count: {customer.orders_count}")
```

### Rate Limits
- **REST API**: 40 requests per app per store per minute (2 requests/second)
- **GraphQL API**: Calculated point system (1000 points per 60 seconds)
- **Leaky Bucket Algorithm**: Points refill gradually
- **Error Response**: HTTP 429 status code
- **Headers**: Monitor `X-Shopify-Shop-Api-Call-Limit` header

### Rate Limit Best Practices
1. **Monitor Headers**: Check `X-Shopify-Shop-Api-Call-Limit` in responses
2. **Implement Backoff**: Exponential backoff on 429 errors with `Retry-After` header
3. **Optimize Requests**: Only fetch required data, use GraphQL for complex queries
4. **Use Webhooks**: Subscribe to webhooks for real-time updates instead of polling
5. **Caching**: Cache frequently accessed data that doesn't change often
6. **Request Batching**: Use GraphQL bulk operations when possible
7. **Space Requests**: Distribute requests evenly over time

### Webhook Patterns for Real-Time Data
```python
from flask import Flask, request
import hmac
import hashlib

app = Flask(__name__)

@app.route('/webhook/order-created', methods=['POST'])
def order_created_webhook():
    # Verify webhook authenticity
    hmac_header = request.headers.get('X-Shopify-Hmac-Sha256')
    body = request.get_data()

    calculated_hmac = hmac.new(
        SHOPIFY_SECRET.encode('utf-8'),
        body,
        hashlib.sha256
    ).hexdigest()

    if hmac_header == calculated_hmac:
        order_data = request.get_json()
        # Process order data
        process_new_order(order_data)
        return '', 200
    else:
        return 'Unauthorized', 401
```

### Critical Gotchas
1. **API Versioning**: API versions are dated (e.g., 2024-01) and expire after 12 months
2. **Rate Limit Calculation**: Complex for GraphQL (point-based vs simple count)
3. **Webhook Verification**: Must verify HMAC signature for security
4. **Session Management**: Must activate session before making API calls
5. **Pagination Changes**: API uses cursor-based pagination (not page numbers) for newer endpoints
6. **Private App Deprecation**: Private apps deprecated in favor of custom apps
7. **Scope Permissions**: Must request specific OAuth scopes upfront
8. **API Call Limit Header**: Format is "current/limit" (e.g., "32/40")
9. **Multiple Stores**: Each store requires separate authentication
10. **Testing**: Use development stores for testing (free from Partners Dashboard)

### Code Examples to Reference
- **Official Examples**: https://github.com/Shopify/shopify_python_api (REST and GraphQL examples)
- **API Cheatsheet**: https://github.com/troyshu/shopify-python-api-cheatsheet
- **Integration Examples**: https://github.com/ziplokk1/python-shopify-api/blob/master/example.py
- **FastAPI Integration**: https://github.com/samyxdev/fastapi-supabase-template (demonstrates auth patterns)

### Recommended for PRPs/ai_docs/
**YES** - Highly recommended for the following reasons:
1. Essential for ecommerce revenue tracking and financial analytics
2. Official Python SDK well-maintained by Shopify
3. Comprehensive API covering orders, customers, products, analytics
4. Webhook support enables real-time financial data updates
5. Strong authentication patterns (OAuth 2.0) with good security
6. Critical for AI CFO system handling ecommerce businesses

**Documentation to Save:**
- OAuth 2.0 flow and scope management
- Webhook setup and HMAC verification patterns
- Rate limit handling strategies (with retry logic)
- Pagination patterns (both legacy and cursor-based)
- GraphQL query examples for analytics
- Order and financial data extraction patterns

---

## 3. Supabase Python Client

### Official Documentation
- **Official Docs**: https://supabase.com/docs/reference/python/introduction
- **Getting Started**: https://supabase.com/docs/reference/python/start
- **Client Library Features**: https://supabase.com/features/client-library-python
- **Auth Documentation**: https://supabase.com/docs/reference/python/auth-signup

### Python SDK
- **Library Name**: `supabase` (supabase-py)
- **PyPI**: https://pypi.org/project/supabase/
- **GitHub**: https://github.com/supabase/supabase-py
- **Auth Library**: https://github.com/supabase/auth-py

### Authentication Pattern
- **Method**: JWT-based authentication
- **Supported Methods**:
  - Email/password
  - Magic links
  - OAuth providers (Google, GitHub, etc.)
  - Multi-Factor Authentication (TOTP)
  - Phone/SMS authentication

**Basic Authentication Setup:**
```python
from supabase import create_client, Client
import os

# Initialize client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Sign up new user
auth_response = supabase.auth.sign_up({
    "email": "user@example.com",
    "password": "secure_password"
})

# Sign in existing user
auth_response = supabase.auth.sign_in_with_password({
    "email": "user@example.com",
    "password": "secure_password"
})

# Get current user
user = supabase.auth.get_user()

# Sign out
supabase.auth.sign_out()
```

**OAuth Authentication:**
```python
# Sign in with OAuth provider
auth_response = supabase.auth.sign_in_with_oauth({
    "provider": "google",
    "options": {
        "redirect_to": "https://your-app.com/auth/callback"
    }
})
```

**Magic Link Authentication:**
```python
# Send magic link
auth_response = supabase.auth.sign_in_with_otp({
    "email": "user@example.com",
    "options": {
        "email_redirect_to": "https://your-app.com/welcome"
    }
})
```

### Connection Patterns

**Basic Client Setup:**
```python
from supabase import create_client, Client

# Create client (entry point to all Supabase functionality)
supabase: Client = create_client(
    supabase_url="https://your-project.supabase.co",
    supabase_key="your-anon-key"
)
```

**Environment Variables (Recommended):**
```python
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)
```

### Query Builders

**Select Queries:**
```python
# Select all rows
response = supabase.table('financial_records').select('*').execute()

# Select specific columns
response = supabase.table('invoices').select('id, amount, status').execute()

# Filter queries
response = supabase.table('transactions') \
    .select('*') \
    .eq('status', 'completed') \
    .gte('amount', 1000) \
    .execute()

# Order and limit
response = supabase.table('expenses') \
    .select('*') \
    .order('created_at', desc=True) \
    .limit(10) \
    .execute()

# Complex filters
response = supabase.table('customers') \
    .select('*') \
    .or_('status.eq.active,status.eq.pending') \
    .execute()
```

**Insert Operations:**
```python
# Insert single record
data = {
    "company_name": "Acme Corp",
    "amount": 5000.00,
    "status": "pending"
}
response = supabase.table('invoices').insert(data).execute()

# Insert multiple records
data = [
    {"name": "Item 1", "price": 100},
    {"name": "Item 2", "price": 200}
]
response = supabase.table('products').insert(data).execute()
```

**Update Operations:**
```python
# Update records
response = supabase.table('invoices') \
    .update({"status": "paid"}) \
    .eq('id', invoice_id) \
    .execute()

# Update multiple records
response = supabase.table('expenses') \
    .update({"approved": True}) \
    .in_('category', ['travel', 'meals']) \
    .execute()
```

**Delete Operations:**
```python
# Delete records
response = supabase.table('temp_records') \
    .delete() \
    .eq('id', record_id) \
    .execute()

# Delete with filters
response = supabase.table('old_data') \
    .delete() \
    .lt('created_at', '2023-01-01') \
    .execute()
```

**Relationships (Foreign Keys):**
```python
# Join tables using foreign key relationships
response = supabase.table('orders') \
    .select('*, customers(name, email)') \
    .execute()

# Multiple levels
response = supabase.table('line_items') \
    .select('*, orders(*, customers(name))') \
    .execute()
```

### Real-Time Subscriptions

**Important Note**: As of 2025, native real-time subscriptions in Python SDK have limitations compared to JavaScript SDK. You can work around this using the Realtime server directly through websockets.

**Realtime Features Available:**
```python
from supabase import create_client
from realtime.connection import Socket

# The realtime-py library supports:
# 1. Broadcast: Send ephemeral messages between clients
# 2. Presence: Track and synchronize shared state (CRDTs)
# 3. Postgres CDC: Listen to database changes

# Note: Direct Python subscription support is limited
# Consider using webhooks or polling for production use cases
```

**Alternative: Database Webhooks**
For production, use Supabase Database Webhooks to trigger external endpoints when data changes.

### RPC (Remote Procedure Calls)

**Calling Postgres Functions:**
```python
# Call a custom database function
response = supabase.rpc('calculate_monthly_revenue', {
    'start_date': '2024-01-01',
    'end_date': '2024-01-31'
}).execute()

# Example Postgres function:
# CREATE FUNCTION calculate_monthly_revenue(start_date date, end_date date)
# RETURNS numeric AS $$
#   SELECT SUM(amount) FROM invoices
#   WHERE created_at BETWEEN start_date AND end_date
#   AND status = 'paid';
# $$ LANGUAGE sql;
```

### Storage Operations

**File Upload/Download:**
```python
# Upload file
with open('invoice.pdf', 'rb') as f:
    response = supabase.storage.from_('documents').upload(
        'invoices/invoice_001.pdf',
        f
    )

# Download file
response = supabase.storage.from_('documents').download('invoices/invoice_001.pdf')

# Get public URL
url = supabase.storage.from_('documents').get_public_url('invoices/invoice_001.pdf')

# List files
files = supabase.storage.from_('documents').list('invoices/')
```

### Example Code

**1. Complete CRUD Example:**
```python
from supabase import create_client
import os

# Initialize
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Create
new_record = {
    "customer_name": "John Doe",
    "amount": 1500.00,
    "status": "pending",
    "due_date": "2024-12-31"
}
create_response = supabase.table('invoices').insert(new_record).execute()
invoice_id = create_response.data[0]['id']

# Read
read_response = supabase.table('invoices').select('*').eq('id', invoice_id).execute()
invoice = read_response.data[0]

# Update
update_response = supabase.table('invoices') \
    .update({"status": "paid"}) \
    .eq('id', invoice_id) \
    .execute()

# Delete
delete_response = supabase.table('invoices').delete().eq('id', invoice_id).execute()
```

**2. Authentication with FastAPI:**
```python
from fastapi import FastAPI, Depends, HTTPException
from supabase import create_client, Client
import os

app = FastAPI()

def get_supabase() -> Client:
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY")
    )

@app.post("/auth/signup")
async def signup(email: str, password: str, supabase: Client = Depends(get_supabase)):
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        return {"user": response.user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
async def login(email: str, password: str, supabase: Client = Depends(get_supabase)):
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return {
            "access_token": response.session.access_token,
            "user": response.user
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/protected")
async def protected_route(supabase: Client = Depends(get_supabase)):
    user = supabase.auth.get_user()
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"user": user}
```

**3. Financial Data Aggregation:**
```python
from datetime import datetime, timedelta
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_monthly_revenue(year: int, month: int):
    """Get total revenue for a specific month."""
    start_date = f"{year}-{month:02d}-01"

    # Calculate last day of month
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month + 1:02d}-01"

    # Query invoices
    response = supabase.table('invoices') \
        .select('amount') \
        .eq('status', 'paid') \
        .gte('paid_date', start_date) \
        .lt('paid_date', end_date) \
        .execute()

    total = sum(record['amount'] for record in response.data)
    return total

def get_customer_lifetime_value(customer_id: str):
    """Calculate total revenue from a customer."""
    response = supabase.table('invoices') \
        .select('amount') \
        .eq('customer_id', customer_id) \
        .eq('status', 'paid') \
        .execute()

    return sum(record['amount'] for record in response.data)

def get_outstanding_invoices():
    """Get all unpaid invoices."""
    response = supabase.table('invoices') \
        .select('*, customers(name, email)') \
        .in_('status', ['pending', 'overdue']) \
        .order('due_date', desc=False) \
        .execute()

    return response.data
```

**4. Row Level Security (RLS) with Auth:**
```python
# Supabase uses RLS policies in Postgres
# Example policy (run in Supabase SQL editor):
"""
-- Users can only see their own financial records
CREATE POLICY "Users can view own records"
ON financial_records
FOR SELECT
USING (auth.uid() = user_id);

-- Users can only insert their own records
CREATE POLICY "Users can insert own records"
ON financial_records
FOR INSERT
WITH CHECK (auth.uid() = user_id);
"""

# In Python, RLS is automatically enforced when using auth:
# User must be authenticated
user = supabase.auth.get_user()

# This query automatically filters by user_id due to RLS
response = supabase.table('financial_records').select('*').execute()
# Only returns records where user_id = authenticated user's ID
```

### Rate Limits
- **No explicit rate limits** on Supabase client operations
- **Database connection limits**: Based on your plan (e.g., Free tier: 60 concurrent connections)
- **API Gateway limits**: Based on Supabase plan
- **Self-hosted**: No limits (you control infrastructure)

### Critical Gotchas
1. **Realtime Limitations**: Python SDK realtime subscriptions less mature than JavaScript
2. **RLS Required**: Enable Row Level Security for production security
3. **Connection Pooling**: Consider connection limits on your plan
4. **Service Role Key**: Never expose service_role key in client-side code (use anon key)
5. **JWT Expiration**: Access tokens expire, implement refresh logic
6. **Query Filters**: Use proper escaping for user input in filters
7. **Async Not Default**: Python SDK is synchronous (use async libs if needed)
8. **Error Handling**: Check response status, supabase doesn't raise exceptions by default
9. **Foreign Key Syntax**: Join syntax different from SQL (uses nested object notation)
10. **Case Sensitivity**: Table and column names are case-sensitive in Postgres

### Code Examples to Reference
- **Official Repository**: https://github.com/supabase/supabase-py
- **CRUD Tutorials**: https://github.com/AlvinKimata/supabase-tutorials
- **Supabase with Python**: https://github.com/kunal356/supabase-tuts
- **FastAPI Integration**: https://github.com/samyxdev/fastapi-supabase-template
- **Authentication Examples**: https://github.com/ruvnet/supabase-authentication
- **FastAPI Discussion**: https://github.com/orgs/supabase/discussions/33811

### Recommended for PRPs/ai_docs/
**YES** - Highly recommended for the following reasons:
1. Modern, developer-friendly Postgres database with full Python support
2. Built-in authentication with JWT, essential for multi-tenant CFO system
3. Real-time capabilities for live financial dashboard updates
4. Row Level Security (RLS) enables secure multi-tenant architecture
5. File storage for documents (invoices, receipts, reports)
6. Free tier sufficient for development and small deployments
7. Easy migration path (just Postgres under the hood)
8. Excellent for storing processed financial data and user information

**Documentation to Save:**
- Authentication patterns (email, OAuth, magic links)
- Query builder patterns for complex financial queries
- Row Level Security policy examples for multi-tenant setup
- FastAPI integration patterns
- File storage for financial documents
- RPC patterns for custom financial calculations in Postgres

---

## 4. Integration Architecture Recommendations

### Overall System Architecture

**Recommended Pattern: Microservices with Event-Driven Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway (FastAPI)                    │
│              Authentication & Request Routing                │
└──────┬──────────────────┬───────────────────┬───────────────┘
       │                  │                   │
       ▼                  ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌─────────────────┐
│  QuickBooks  │   │   Shopify    │   │   Supabase      │
│  Integration │   │ Integration  │   │   Service       │
│   Service    │   │   Service    │   │ (Primary DB)    │
└──────┬───────┘   └──────┬───────┘   └────────┬────────┘
       │                  │                     │
       │                  │                     │
       ▼                  ▼                     ▼
┌──────────────────────────────────────────────────────────┐
│              Event Bus / Message Queue                    │
│              (Redis Pub/Sub or RabbitMQ)                 │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│              AI CFO Analysis Engine                       │
│         (PydanticAI Agent with Financial Tools)          │
└──────────────────────────────────────────────────────────┘
```

### Service Organization

**1. QuickBooks Integration Service**
```
services/
├── quickbooks/
│   ├── __init__.py
│   ├── client.py          # OAuth & API client wrapper
│   ├── sync.py            # Data synchronization logic
│   ├── models.py          # Pydantic models for QB data
│   ├── transformers.py    # Transform QB data to standard format
│   └── webhooks.py        # QB webhook handlers (if available)
```

**Key Responsibilities:**
- OAuth 2.0 authentication and token refresh
- Fetch financial reports (P&L, Balance Sheet, Cash Flow)
- Sync invoices, expenses, accounts, customers
- Transform QuickBooks data into standardized format
- Handle rate limiting (500/min, 10 concurrent)
- Store in Supabase for AI analysis

**2. Shopify Integration Service**
```
services/
├── shopify/
│   ├── __init__.py
│   ├── client.py          # OAuth & API client wrapper
│   ├── sync.py            # Data synchronization logic
│   ├── models.py          # Pydantic models for Shopify data
│   ├── transformers.py    # Transform Shopify data to standard format
│   └── webhooks.py        # Webhook endpoint handlers
```

**Key Responsibilities:**
- OAuth 2.0 authentication and session management
- Fetch orders, products, customers
- Subscribe to webhooks (orders/create, orders/updated, etc.)
- Transform Shopify data into standardized format
- Handle rate limiting (40/min, 2/sec)
- Calculate ecommerce metrics (revenue, AOV, conversion)

**3. Supabase Service (Primary Data Layer)**
```
services/
├── supabase/
│   ├── __init__.py
│   ├── client.py          # Supabase client wrapper
│   ├── auth.py            # Authentication helpers
│   ├── models.py          # Database schema definitions
│   ├── queries.py         # Common query patterns
│   └── migrations/        # Database migrations
```

**Key Responsibilities:**
- User authentication and authorization
- Store all processed financial data
- Store AI analysis results and insights
- File storage (invoices, receipts, reports)
- Real-time updates for dashboards
- Row Level Security for multi-tenant isolation

### Data Flow Architecture

**Synchronization Flow:**
```python
# 1. Periodic Sync (scheduled jobs)
@scheduler.scheduled_job('interval', hours=1)
async def sync_financial_data():
    # Fetch from QuickBooks
    qb_data = await quickbooks_service.fetch_recent_data()

    # Fetch from Shopify
    shopify_data = await shopify_service.fetch_recent_orders()

    # Transform to standard format
    standardized_qb = transform_quickbooks_data(qb_data)
    standardized_shopify = transform_shopify_data(shopify_data)

    # Store in Supabase
    await supabase_service.upsert_financial_records(standardized_qb)
    await supabase_service.upsert_ecommerce_records(standardized_shopify)

    # Emit event for AI analysis
    await event_bus.publish('financial_data.updated', {
        'sources': ['quickbooks', 'shopify'],
        'timestamp': datetime.utcnow()
    })

# 2. Real-Time Webhook Handler
@app.post("/webhooks/shopify/orders")
async def handle_shopify_order(request: Request):
    # Verify HMAC
    if not verify_shopify_webhook(request):
        raise HTTPException(401)

    order_data = await request.json()

    # Transform and store
    standardized = transform_shopify_order(order_data)
    await supabase_service.insert_order(standardized)

    # Emit event
    await event_bus.publish('order.created', standardized)

    return {'status': 'success'}
```

### Standardized Data Models

**Create common Pydantic models for financial data:**

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class FinancialRecordType(str, Enum):
    INVOICE = "invoice"
    EXPENSE = "expense"
    PAYMENT = "payment"
    REFUND = "refund"

class FinancialRecord(BaseModel):
    """Standardized financial record across all sources."""
    id: str
    source: str = Field(..., description="quickbooks, shopify, etc.")
    source_id: str = Field(..., description="ID in source system")
    type: FinancialRecordType
    date: datetime
    amount: float
    currency: str = "USD"
    customer_id: str | None = None
    customer_name: str | None = None
    description: str | None = None
    category: str | None = None
    status: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class EcommerceOrder(BaseModel):
    """Standardized ecommerce order."""
    id: str
    source: str = "shopify"
    source_id: str
    order_number: str
    date: datetime
    customer_email: str
    customer_name: str
    total_amount: float
    subtotal: float
    tax: float
    shipping: float
    currency: str = "USD"
    items_count: int
    status: str
    line_items: list[dict]
    metadata: dict = Field(default_factory=dict)
```

### Configuration Management

**Use environment variables and Pydantic Settings:**

```python
from pydantic_settings import BaseSettings

class QuickBooksSettings(BaseSettings):
    client_id: str
    client_secret: str
    redirect_uri: str
    environment: str = "sandbox"  # or "production"

    class Config:
        env_prefix = "QUICKBOOKS_"
        env_file = ".env"

class ShopifySettings(BaseSettings):
    api_key: str
    api_secret: str
    shop_name: str
    access_token: str | None = None
    api_version: str = "2024-01"

    class Config:
        env_prefix = "SHOPIFY_"
        env_file = ".env"

class SupabaseSettings(BaseSettings):
    url: str
    anon_key: str
    service_role_key: str

    class Config:
        env_prefix = "SUPABASE_"
        env_file = ".env"

class AppSettings(BaseSettings):
    quickbooks: QuickBooksSettings = QuickBooksSettings()
    shopify: ShopifySettings = ShopifySettings()
    supabase: SupabaseSettings = SupabaseSettings()

    # App settings
    debug: bool = False
    log_level: str = "INFO"

# Usage
settings = AppSettings()
```

### Error Handling & Retry Logic

**Implement robust error handling for all external APIs:**

```python
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class RateLimitError(Exception):
    pass

class APIConnectionError(Exception):
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError))
)
async def fetch_quickbooks_data(endpoint: str, params: dict):
    try:
        response = await qb_client.get(endpoint, params=params)
        return response
    except QuickBooksRateLimitException:
        raise RateLimitError("QuickBooks rate limit exceeded")
    except QuickBooksConnectionException:
        raise APIConnectionError("Failed to connect to QuickBooks")

# Similar pattern for Shopify
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
async def fetch_shopify_orders(params: dict):
    try:
        orders = shopify.Order.find(**params)
        return orders
    except shopify.ShopifyResource.ClientError as e:
        if e.response.code == 429:
            retry_after = int(e.response.headers.get('Retry-After', 2))
            await asyncio.sleep(retry_after)
            raise RateLimitError("Shopify rate limit exceeded")
        raise
```

### Caching Strategy

**Implement caching to reduce API calls:**

```python
from functools import lru_cache
import redis
import json

# Redis for distributed caching
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache_financial_data(key: str, ttl: int = 3600):
    """Decorator for caching financial data in Redis."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Try to get from cache
            cached = redis_client.get(key)
            if cached:
                return json.loads(cached)

            # Fetch fresh data
            data = await func(*args, **kwargs)

            # Store in cache
            redis_client.setex(key, ttl, json.dumps(data))

            return data
        return wrapper
    return decorator

@cache_financial_data("quickbooks:accounts", ttl=3600)
async def get_quickbooks_accounts():
    """Cached for 1 hour since accounts don't change often."""
    return await fetch_quickbooks_data("/account", {})

@cache_financial_data("shopify:products", ttl=1800)
async def get_shopify_products():
    """Cached for 30 minutes."""
    return await shopify_client.fetch_products()
```

### Security Best Practices

**1. Secrets Management:**
```python
# NEVER hardcode credentials
# Use environment variables or secret management service

# Good:
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    quickbooks_client_id: str
    quickbooks_client_secret: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Bad:
QUICKBOOKS_CLIENT_ID = "hardcoded_value"  # NEVER DO THIS
```

**2. API Key Rotation:**
```python
# Implement token refresh logic
class QuickBooksClient:
    def __init__(self, auth_client):
        self.auth_client = auth_client
        self._access_token = None
        self._refresh_token = None
        self._token_expiry = None

    async def get_access_token(self):
        if self._token_expiry and datetime.now() > self._token_expiry:
            # Token expired, refresh it
            await self.refresh_access_token()
        return self._access_token

    async def refresh_access_token(self):
        response = self.auth_client.refresh(self._refresh_token)
        self._access_token = response['access_token']
        self._refresh_token = response['refresh_token']
        self._token_expiry = datetime.now() + timedelta(seconds=response['expires_in'])
```

**3. Webhook Verification:**
```python
import hmac
import hashlib

def verify_shopify_webhook(request, secret):
    """Verify Shopify webhook signature."""
    hmac_header = request.headers.get('X-Shopify-Hmac-Sha256')
    body = request.body

    calculated_hmac = hmac.new(
        secret.encode('utf-8'),
        body,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(hmac_header, calculated_hmac)
```

### Monitoring & Logging

**Implement comprehensive logging:**

```python
import logging
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class APIMetrics:
    """Track API call metrics."""

    def __init__(self):
        self.calls = {
            'quickbooks': {'count': 0, 'errors': 0},
            'shopify': {'count': 0, 'errors': 0}
        }

    def record_call(self, service: str, success: bool):
        self.calls[service]['count'] += 1
        if not success:
            self.calls[service]['errors'] += 1

        logger.info(f"API Call: {service} - Success: {success}")

        # Check rate limits
        if service == 'quickbooks' and self.calls[service]['count'] % 100 == 0:
            logger.warning(f"QuickBooks: {self.calls[service]['count']} calls made")

    def get_metrics(self):
        return self.calls

# Usage
metrics = APIMetrics()

async def fetch_data_with_metrics(service: str, fetch_func):
    try:
        data = await fetch_func()
        metrics.record_call(service, success=True)
        return data
    except Exception as e:
        metrics.record_call(service, success=False)
        logger.error(f"API Error for {service}: {str(e)}")
        raise
```

### Testing Strategy

**1. Unit Tests:**
```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_quickbooks_client():
    """Mock QuickBooks client for testing."""
    client = Mock()
    client.get_invoices.return_value = [
        {'id': '1', 'amount': 1000.00, 'status': 'paid'}
    ]
    return client

def test_transform_quickbooks_invoice(mock_quickbooks_client):
    """Test QuickBooks data transformation."""
    invoices = mock_quickbooks_client.get_invoices()
    transformed = transform_quickbooks_data(invoices)

    assert transformed[0]['type'] == 'invoice'
    assert transformed[0]['source'] == 'quickbooks'
    assert transformed[0]['amount'] == 1000.00
```

**2. Integration Tests:**
```python
@pytest.mark.integration
async def test_quickbooks_to_supabase_sync():
    """Test full sync from QuickBooks to Supabase."""
    # Requires QB sandbox and test Supabase instance
    qb_data = await quickbooks_service.fetch_invoices()
    await supabase_service.upsert_financial_records(qb_data)

    # Verify data in Supabase
    records = await supabase_service.get_financial_records()
    assert len(records) > 0
```

### Deployment Considerations

**Docker Composition:**
```yaml
version: '3.8'

services:
  api-gateway:
    build: ./api-gateway
    ports:
      - "8000:8000"
    environment:
      - QUICKBOOKS_CLIENT_ID=${QUICKBOOKS_CLIENT_ID}
      - SHOPIFY_API_KEY=${SHOPIFY_API_KEY}
      - SUPABASE_URL=${SUPABASE_URL}
    depends_on:
      - redis
      - postgres

  quickbooks-service:
    build: ./services/quickbooks
    environment:
      - QUICKBOOKS_CLIENT_ID=${QUICKBOOKS_CLIENT_ID}
      - REDIS_URL=redis://redis:6379

  shopify-service:
    build: ./services/shopify
    environment:
      - SHOPIFY_API_KEY=${SHOPIFY_API_KEY}
      - REDIS_URL=redis://redis:6379

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=cfo_system
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

---

## 5. Code Examples to Reference

### Comprehensive GitHub Examples

**QuickBooks Integration:**
1. **python-quickbooks library**: https://github.com/ej2/python-quickbooks
   - Complete API wrapper with OAuth 2.0
   - Integration tests for all major entities
   - Best maintained community library

2. **Intuit Official Sample App**: https://github.com/IntuitDeveloper/SampleApp-QuickBooksV3API-Python
   - Official example from Intuit (Flask)
   - OAuth 2.0 implementation reference
   - Customer API examples

3. **Financial Report Generation**: https://github.com/simonv3/quickbooks-python/blob/master/report.py
   - P&L, General Ledger implementations
   - Report parsing logic
   - Period-based tallying

4. **Custom PDF Reports**: https://github.com/Gradient-s-p/custom-quickbooks-report
   - QuickBooks API + FPDF integration
   - Customized financial report generation

**Shopify Integration:**
1. **Official shopify_python_api**: https://github.com/Shopify/shopify_python_api
   - Official Shopify library
   - REST and GraphQL examples
   - OAuth implementation

2. **API Cheatsheet**: https://github.com/troyshu/shopify-python-api-cheatsheet
   - Comprehensive method reference
   - Quick lookup for common operations

3. **Integration Examples**: https://github.com/ziplokk1/python-shopify-api
   - Practical integration patterns
   - Product management examples

**Supabase Integration:**
1. **Official supabase-py**: https://github.com/supabase/supabase-py
   - Official Python client
   - Complete API reference

2. **CRUD Tutorials**: https://github.com/AlvinKimata/supabase-tutorials
   - Basic CRUD operations
   - Authentication examples
   - Jupyter notebook format

3. **Python Tutorials**: https://github.com/kunal356/supabase-tuts
   - Learning path with Supabase
   - CRUD operations walkthrough

4. **FastAPI Template**: https://github.com/samyxdev/fastapi-supabase-template
   - Production-ready template
   - Authentication with Swagger
   - Bookmark app example (CRUD)

5. **Comprehensive Auth**: https://github.com/ruvnet/supabase-authentication
   - Streamlit and FastAPI examples
   - Secure authentication flows
   - Multiple framework demos

**Full Stack Integration Examples:**
- **FastAPI + Supabase Discussion**: https://github.com/orgs/supabase/discussions/33811
  - Best practices discussion
  - Community patterns

### Specific Code Patterns to Save

**1. OAuth 2.0 Token Management:**
- QuickBooks token refresh pattern
- Shopify session management
- Token storage and rotation

**2. Rate Limit Handling:**
- Exponential backoff implementation
- Request queue management
- Header monitoring patterns

**3. Webhook Verification:**
- HMAC signature verification (Shopify)
- Payload validation
- Replay attack prevention

**4. Data Transformation:**
- Source-to-standard format converters
- Pydantic model validation
- Error handling during transformation

**5. Multi-Tenant Architecture:**
- Supabase Row Level Security policies
- User-based data isolation
- Authentication flow with RLS

**6. Financial Calculations:**
- Revenue aggregation patterns
- Period-based reporting
- Customer lifetime value calculations

**7. Caching Strategies:**
- Redis integration patterns
- Cache invalidation logic
- TTL management by data type

**8. Error Recovery:**
- Retry mechanisms with tenacity
- Circuit breaker patterns
- Graceful degradation

---

## Summary & Recommendations

### Recommended Priority for Implementation

**Phase 1: Foundation (Week 1-2)**
1. Set up Supabase as primary database
2. Implement user authentication with Supabase
3. Create standardized data models (Pydantic)
4. Set up FastAPI gateway with basic endpoints

**Phase 2: QuickBooks Integration (Week 3-4)**
1. Implement OAuth 2.0 flow for QuickBooks
2. Build financial report fetching (P&L, Balance Sheet)
3. Transform QB data to standard format
4. Store in Supabase with scheduled sync

**Phase 3: Shopify Integration (Week 5-6)**
1. Implement OAuth 2.0 flow for Shopify
2. Build order and product data fetching
3. Set up webhooks for real-time updates
4. Transform and store ecommerce data

**Phase 4: AI CFO Engine (Week 7-8)**
1. Build PydanticAI agent with financial tools
2. Create analysis tools using standardized data
3. Implement CFO insights and recommendations
4. Build user-facing API endpoints

### Critical Success Factors

1. **Rate Limit Management**: Implement proper retry logic and request pacing from day one
2. **Security**: Never expose service keys; use proper OAuth flows and RLS
3. **Data Standardization**: Create Pydantic models early; transformations are crucial
4. **Error Handling**: Comprehensive logging and graceful degradation
5. **Testing**: Use sandbox environments; write integration tests
6. **Documentation**: Document all integrations, gotchas, and workarounds

### Documentation for PRPs/ai_docs/

**Essential Documents to Create:**

1. **`quickbooks_integration.md`**
   - OAuth 2.0 flow step-by-step
   - Key endpoints and rate limits
   - Error handling patterns
   - Data transformation examples

2. **`shopify_integration.md`**
   - OAuth setup and webhooks
   - Real-time data handling
   - Rate limit strategies
   - Ecommerce metrics calculations

3. **`supabase_architecture.md`**
   - Database schema design
   - RLS policies for multi-tenant
   - Authentication flows
   - Query patterns

4. **`data_models.md`**
   - Standardized Pydantic models
   - Transformation logic
   - Validation rules

5. **`api_gateway.md`**
   - FastAPI structure
   - Endpoint definitions
   - Authentication middleware
   - Error handling

6. **`deployment_guide.md`**
   - Docker composition
   - Environment variables
   - Secrets management
   - Monitoring setup

### Final Recommendation

All three APIs (QuickBooks, Shopify, Supabase) are **highly recommended** for the AI CFO system:

- **QuickBooks**: Essential for comprehensive financial data (P&L, Balance Sheet, invoices)
- **Shopify**: Critical for ecommerce businesses, provides revenue and customer insights
- **Supabase**: Perfect as the central data store with built-in auth and real-time capabilities

The integration architecture should follow a **microservices pattern** with:
- Individual services for each external API
- Supabase as the central data lake
- Event-driven communication for real-time updates
- Standardized Pydantic models for data consistency
- Comprehensive error handling and retry logic
- FastAPI as the API gateway

This architecture will enable the AI CFO system to provide powerful financial insights by combining data from multiple sources while maintaining security, scalability, and maintainability.
