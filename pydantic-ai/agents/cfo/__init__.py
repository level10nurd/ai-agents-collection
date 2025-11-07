"""
CFO Agent Package

Multi-agent AI CFO system for VoChill e-commerce providing:
- Unit economics analysis (CAC, LTV, churn)
- Sales forecasting with Prophet (handles 70% seasonal concentration)
- 13-week cash flow forecasting
- Operations & inventory analysis
- Financial modeling and executive reporting

Architecture: Hierarchical Coordinator + Specialist Agents
"""

from agents.cfo.coordinator import cfo_coordinator_agent

__all__ = ["cfo_coordinator_agent"]
