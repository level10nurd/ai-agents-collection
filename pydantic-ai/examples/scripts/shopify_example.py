"""
Example usage of Shopify integration for CFO agent.

This script demonstrates how to use the Shopify API integration functions
to fetch orders, customers, and calculate metrics.
"""

import asyncio
import os
from agents.cfo.tools.shopify import (
    fetch_orders,
    fetch_customers,
    calculate_customer_metrics,
)


async def main():
    """
    Example: Fetch Shopify data and calculate customer metrics.
    
    Before running, set the following environment variables:
    - SHOPIFY_API_KEY: Your Shopify API access token
    - SHOPIFY_SHOP_NAME: Your shop name (without .myshopify.com)
    """
    # Get credentials from environment
    api_key = os.getenv("SHOPIFY_API_KEY")
    shop_name = os.getenv("SHOPIFY_SHOP_NAME")
    
    if not api_key or not shop_name:
        print("Error: SHOPIFY_API_KEY and SHOPIFY_SHOP_NAME environment variables must be set")
        print("\nExample:")
        print("export SHOPIFY_API_KEY='shpat_xxxxx'")
        print("export SHOPIFY_SHOP_NAME='myshop'")
        return
    
    print("Fetching Shopify data...\n")
    
    try:
        # Fetch orders from the last 30 days
        print("1. Fetching orders...")
        orders = await fetch_orders(
            api_key=api_key,
            shop_name=shop_name,
            start_date="2024-01-01",
            end_date="2024-01-31",
            limit=250
        )
        print(f"   ✓ Fetched {len(orders)} orders")
        
        # Display sample order
        if orders:
            sample_order = orders[0]
            print(f"\n   Sample order:")
            print(f"   - Order ID: {sample_order['order_id']}")
            print(f"   - Total Price: ${sample_order['total_price']:.2f}")
            print(f"   - Created: {sample_order['created_at']}")
            print(f"   - Line Items: {len(sample_order['line_items'])}")
        
        # Fetch customers
        print("\n2. Fetching customers...")
        customers = await fetch_customers(
            api_key=api_key,
            shop_name=shop_name,
            limit=250
        )
        print(f"   ✓ Fetched {len(customers)} customers")
        
        # Display sample customer
        if customers:
            sample_customer = customers[0]
            print(f"\n   Sample customer:")
            print(f"   - Customer ID: {sample_customer['customer_id']}")
            print(f"   - Email: {sample_customer.get('email', 'N/A')}")
            print(f"   - Orders Count: {sample_customer['orders_count']}")
            print(f"   - Total Spent: ${sample_customer['total_spent']:.2f}")
        
        # Calculate metrics
        print("\n3. Calculating customer metrics...")
        metrics = calculate_customer_metrics(orders, customers)
        
        print("\n   Customer Metrics:")
        print(f"   - Total Orders: {metrics['total_orders']}")
        print(f"   - Total Revenue: ${metrics['total_revenue']:,.2f}")
        print(f"   - Average Order Value: ${metrics['average_order_value']:.2f}")
        print(f"   - Unique Customers: {metrics['unique_customers']}")
        print(f"   - Orders per Customer: {metrics['orders_per_customer']:.2f}")
        print(f"   - Repeat Purchase Rate: {metrics['repeat_purchase_rate']:.1f}%")
        print(f"   - Avg Customer Lifetime Value: ${metrics['average_customer_lifetime_value']:.2f}")
        
        # Business insights
        print("\n4. Business Insights:")
        
        aov = metrics['average_order_value']
        rpr = metrics['repeat_purchase_rate']
        
        if aov < 100:
            print("   ⚠ Low AOV - Consider upselling strategies")
        else:
            print("   ✓ Healthy AOV")
        
        if rpr < 20:
            print("   ⚠ Low repeat purchase rate - Focus on customer retention")
        elif rpr < 40:
            print("   ↗ Moderate repeat purchase rate - Room for improvement")
        else:
            print("   ✓ Strong repeat purchase rate")
        
        print("\n✓ Analysis complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nMake sure your API credentials are correct and have the necessary permissions.")


if __name__ == "__main__":
    asyncio.run(main())
