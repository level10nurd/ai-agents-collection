"""
Inventory Analysis Model

Structured data model for inventory levels, reorder points, lead times,
and fulfillment metrics across SKUs and sales channels.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, date
from enum import Enum


class Channel(str, Enum):
    """Sales channel enumeration."""
    SHOPIFY = "shopify"
    AMAZON = "amazon"
    RETAIL = "retail"
    WHOLESALE = "wholesale"
    OTHER = "other"


class SKUInventory(BaseModel):
    """
    Inventory data for a single SKU.
    """

    sku: str = Field(
        ...,
        min_length=1,
        description="Stock Keeping Unit identifier"
    )
    product_name: str = Field(
        ...,
        description="Product name/description"
    )
    channel: Channel = Field(
        ...,
        description="Sales channel for this inventory"
    )
    units_on_hand: int = Field(
        ...,
        ge=0,
        description="Current units in stock"
    )
    units_reserved: int = Field(
        0,
        ge=0,
        description="Units reserved for pending orders"
    )
    units_available: int = Field(
        ...,
        ge=0,
        description="Units available for sale (on_hand - reserved)"
    )
    reorder_point: int = Field(
        ...,
        ge=0,
        description="Stock level triggering reorder"
    )
    reorder_quantity: int = Field(
        ...,
        gt=0,
        description="Quantity to order when reorder point is reached"
    )
    lead_time_days: int = Field(
        ...,
        ge=0,
        description="Days from order to delivery"
    )
    days_of_inventory: Optional[float] = Field(
        None,
        ge=0,
        description="Days of inventory remaining at current sales velocity"
    )
    needs_reorder: bool = Field(
        False,
        description="True if units_available <= reorder_point"
    )
    is_stockout_risk: bool = Field(
        False,
        description="True if days_of_inventory <= lead_time_days"
    )
    unit_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Unit cost (COGS)"
    )
    inventory_value: Optional[float] = Field(
        None,
        ge=0,
        description="Total value of inventory (units_on_hand * unit_cost)"
    )


class FulfillmentMetrics(BaseModel):
    """
    Fulfillment performance metrics.
    """

    total_orders: int = Field(
        ...,
        ge=0,
        description="Total orders in the period"
    )
    fulfilled_orders: int = Field(
        ...,
        ge=0,
        description="Orders successfully fulfilled"
    )
    pending_orders: int = Field(
        ...,
        ge=0,
        description="Orders awaiting fulfillment"
    )
    cancelled_orders: int = Field(
        0,
        ge=0,
        description="Orders cancelled"
    )
    fulfillment_rate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Fulfillment rate (fulfilled / total)"
    )
    average_fulfillment_days: Optional[float] = Field(
        None,
        ge=0,
        description="Average days from order to shipment"
    )
    on_time_shipment_rate: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Percentage of orders shipped on time"
    )
    period_start: date = Field(
        ...,
        description="Start date of metrics period"
    )
    period_end: date = Field(
        ...,
        description="End date of metrics period"
    )


class InventoryAnalysis(BaseModel):
    """
    Comprehensive inventory analysis across all SKUs and channels.

    Tracks stock levels, identifies reorder needs, calculates inventory value,
    and monitors fulfillment performance.
    """

    sku_inventories: list[SKUInventory] = Field(
        ...,
        min_length=1,
        description="Inventory data for all SKUs"
    )
    fulfillment_metrics: FulfillmentMetrics = Field(
        ...,
        description="Fulfillment performance metrics"
    )
    total_inventory_value: float = Field(
        ...,
        ge=0,
        description="Total value of all inventory"
    )
    skus_needing_reorder: list[str] = Field(
        default_factory=list,
        description="List of SKUs at or below reorder point"
    )
    stockout_risk_skus: list[str] = Field(
        default_factory=list,
        description="List of SKUs at risk of stockout"
    )
    inventory_turnover_ratio: Optional[float] = Field(
        None,
        ge=0,
        description="Inventory turnover ratio (COGS / avg inventory)"
    )
    days_inventory_outstanding: Optional[float] = Field(
        None,
        ge=0,
        description="Average days inventory held (DIO)"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this analysis was generated"
    )
    company_id: Optional[str] = Field(
        None,
        description="Company identifier for multi-tenant systems"
    )

    def get_skus_by_channel(self, channel: Channel) -> list[SKUInventory]:
        """
        Get all SKUs for a specific sales channel.

        Args:
            channel: The sales channel to filter by

        Returns:
            List of SKUInventory objects for that channel
        """
        return [sku for sku in self.sku_inventories if sku.channel == channel]

    def get_total_units_on_hand(self) -> int:
        """
        Calculate total units across all SKUs.

        Returns:
            Sum of units_on_hand for all SKUs
        """
        return sum(sku.units_on_hand for sku in self.sku_inventories)

    def get_total_units_available(self) -> int:
        """
        Calculate total available units across all SKUs.

        Returns:
            Sum of units_available for all SKUs
        """
        return sum(sku.units_available for sku in self.sku_inventories)

    def has_urgent_reorders(self) -> bool:
        """
        Check if any SKUs have urgent reorder needs.

        Returns:
            True if any SKUs need reorder or are at stockout risk
        """
        return len(self.skus_needing_reorder) > 0 or len(self.stockout_risk_skus) > 0

    def get_inventory_health_score(self) -> float:
        """
        Calculate an inventory health score (0-100).

        Factors:
        - Fulfillment rate (40%)
        - Reorder needs (30% - penalty for needed reorders)
        - Stockout risks (30% - penalty for stockout risks)

        Returns:
            Health score from 0 (poor) to 100 (excellent)
        """
        # Fulfillment component (0-40 points)
        fulfillment_score = self.fulfillment_metrics.fulfillment_rate * 40

        # Reorder component (0-30 points, penalize high reorder ratio)
        reorder_ratio = len(self.skus_needing_reorder) / len(self.sku_inventories)
        reorder_score = max(0, 30 - (reorder_ratio * 30))

        # Stockout risk component (0-30 points, penalize high risk ratio)
        stockout_ratio = len(self.stockout_risk_skus) / len(self.sku_inventories)
        stockout_score = max(0, 30 - (stockout_ratio * 30))

        return fulfillment_score + reorder_score + stockout_score

    def format_summary(self) -> str:
        """
        Format a concise text summary of inventory status.

        Returns:
            Multi-line string summary
        """
        health_score = self.get_inventory_health_score()

        lines = [
            "Inventory Analysis Summary",
            "=" * 40,
            f"Total SKUs: {len(self.sku_inventories)}",
            f"Total Units On Hand: {self.get_total_units_on_hand():,}",
            f"Total Units Available: {self.get_total_units_available():,}",
            f"Total Inventory Value: ${self.total_inventory_value:,.2f}",
            f"Inventory Health Score: {health_score:.1f}/100",
            "",
            "Fulfillment Performance:",
            f"  Fulfillment Rate: {self.fulfillment_metrics.fulfillment_rate*100:.1f}%",
            f"  Total Orders: {self.fulfillment_metrics.total_orders:,}",
            f"  Pending Orders: {self.fulfillment_metrics.pending_orders:,}",
            "",
        ]

        if self.has_urgent_reorders():
            lines.append("⚠️  ACTION REQUIRED:")
            if self.skus_needing_reorder:
                lines.append(f"  {len(self.skus_needing_reorder)} SKU(s) need reorder")
            if self.stockout_risk_skus:
                lines.append(f"  {len(self.stockout_risk_skus)} SKU(s) at stockout risk")
            lines.append("")

        return "\n".join(lines)

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "sku_inventories": [
                    {
                        "sku": "VC-COOLER-001",
                        "product_name": "VoChill Wine Chiller",
                        "channel": "shopify",
                        "units_on_hand": 150,
                        "units_reserved": 20,
                        "units_available": 130,
                        "reorder_point": 50,
                        "reorder_quantity": 200,
                        "lead_time_days": 30,
                        "days_of_inventory": 45,
                        "needs_reorder": False,
                        "is_stockout_risk": False,
                        "unit_cost": 25.00,
                        "inventory_value": 3750.00
                    }
                ],
                "fulfillment_metrics": {
                    "total_orders": 100,
                    "fulfilled_orders": 95,
                    "pending_orders": 5,
                    "cancelled_orders": 0,
                    "fulfillment_rate": 0.95,
                    "average_fulfillment_days": 2.5,
                    "on_time_shipment_rate": 0.92,
                    "period_start": "2024-11-01",
                    "period_end": "2024-11-30"
                },
                "total_inventory_value": 50000.00,
                "skus_needing_reorder": [],
                "stockout_risk_skus": []
            }
        }
