## FEATURE:

- Super intelligent agent that is the AI CFO for a omni-channel startup ecommerce company. The CEO needs data-driven insights to balance growth, operations, and liqudidity.  
- The AI CFO should
    - Coordinate specialized subagents for different domains
    - Aggregate insights from multiple sources
    -   Provide executive summaries with actionable recommendations
- The AI CFO will be an expert and/or coordinate a team of experts in the domains of
    - ***HR:*** Oversee payroll costs, seasonal temps, hiring, and HR Compliance
    - ***Accounting:*** Ensure accurate, timely, and insightful accounting for the organization that holds up to the highest scrutiney under GAAP
    - ***SupplyChain Optimization:*** Optimize inventory levels, cashflows, production, and fulfillment schedules to ensure highest level of cashflow and operational efficiency. 
    - ***Budgeting:*** Maintain detailed 13-week cash forecast as well as rolling 12 month cash and sales forecast and report and provide insights to CEO weekly.
    - ***Forecasting:*** Timeseries and Demand forecasting based on historical sales, customer trends, marketing trends, product launches, etc. to generate a sales and item forecast, and coordinate with relevant supply chain and accounting domains to ensure all loops are closed.
    - ***Sales Analytics:*** Oversee world class sales analytics and reporting by combining omni-channel ecommerce data with paid and organice marketing data to provide revenue and growth opportunities to other domains. Work closely with Forecasting in the generation of timeseries/demand forecast.
    - ***Reporting:*** Unified reporting accross domains to CEO and Investors that both provide consistency, while also providing unique and actionable insight. 

## COMPANY & INDUSTRY:
### Company Overview
- ***Company:*** VoChill
- ***Website:*** www.vochill.com
- ***Founded:*** 2018 (first revenue in 2020)
- ***Stage:*** Series A (Closed total of $3.8mm since founding)
- ***Industry:*** Direct to Consumer E-Commerce
- ***HQ:*** Austin, TX

### Financial Snapshot
- ***Monthly Burn Rate:*** Jan-Apr & June-Oct ~($60,000): May ~$0: Nov-Dec ~$600,000
- ***Profits:*** $400k profit in 2023: ($110,000) loss in 2024
- ***Gross Margins:*** ~50%
- ***Customers:*** No concentration/highly distributed. ~98% of revenue from dtc
- ***Sales Channels:*** Amazon and Shopify
- ***Total Headcount:*** 8
- ***Highly Seasonal:*** ~70% of sales occure between 11/20 and 12/20 annually

### Operations:
- ***Product:*** Stemless Wine Chiller, Stemmed Wine Chiller, Spirits Chiller
- ***Sourcing:*** Manufactured at facility in Austin, TX
- ***Supply Chain:*** All significant parts sourced from USA
- 
## TOOLS & TECH:
- ***Sales & Customer:*** Shopify.com and Amazon Sellers Centralk
- ***Inventory & Manufacturing:*** InfoPlusWMS
- ***Marketing:*** Meta, Google, Klaviyo
- ***Accounting:*** Quickbooks Online, Bill.com
- ***Misc/Other:*** Excel, Supabase (postgresql), Python
- ***Payroll:*** Gusto

## DEPENDENCIES
- ***Transactional Data:*** Orders/OrderLines, Marketing Events/attribution
    - Sales data Supabase: Marketing data in platform
- ***Reference Data:*** Customers, Items, PriceLists, BillOfMaterials, etc.
    - Supabase
- ***Accounting Data:*** Bank/Credit Card transactions, AR/AP, Purchase Orders, recurring/known outflows
    - QBO & Supabase


## SYSTEM PROMPT(S)

[Describe the instructions for the agent(s) here - you can create the entire system prompt here or give a general description to guide the coding assistant]

## EXAMPLES:
- pydantic-ai/examples/scripts: a list of example python scripts 
    - pydantic-ai/examples/scripts/chat_agent.py- Basic chat agent with conversation memory
    - pydantic-ai/examples/scripts/tool_enabled_agent.py - Tool-enabled agent with web search capabilities  
    - pydantic-ai/examples/scripts/structued_agent_output.py - Structured output agent for data validation
    - pydantic-ai/examples/scripts/test_agent_patterns.py - Testing examples with TestModel and FunctionModel
    - pydantic-ai/examples/scripts/models.py - Best practices for building Pydantic AI agents
- pydantic-ai/examples/output-styles: sample report copy/tone
- pydantic-ai/examples/reports: sample deliverable/workproduct reports
    - FOR DEMONSTRATION ONLY: data not representative of business actual

## DOCUMENTATION:

[Add any additional documentation you want it to reference - this can be curated docs you put in PRPs/ai_docs, URLs, etc.]

- Pydantic AI Official Documentation: https://ai.pydantic.dev/
- Agent Creation Guide: https://ai.pydantic.dev/agents/
- Tool Integration: https://ai.pydantic.dev/tools/
- Testing Patterns: https://ai.pydantic.dev/testing/
- Model Providers: https://ai.pydantic.dev/models/
- SupaBase
- FASAB Handbook of Accounting Standards and Other Pronouncements
- See RAG tags for forecasting, accounting, cashflow
- ClaudeSDK Cookbook: https://github.com/anthropics/claude-cookbooks/tree/main

### Forecasting & Financial Modeling (2025-11-06)
- **Implementation Plan**: PRPs/FORECASTING_IMPLEMENTATION_PLAN.md - Complete roadmap for building forecasting agent
- **Comprehensive Research**: PRPs/forecasting_libraries_research.md - Deep dive into Prophet, SARIMA, XGBoost, TimesFM, etc.
- **Quick Reference**: PRPs/forecasting_quick_reference.md - Code snippets and best practices for daily use
- **Recommendation**: Use Prophet for extreme seasonality (70% in Nov-Dec), numpy-financial for metrics (NPV, IRR)

## OTHER CONSIDERATIONS:

- Use environment variables for API key configuration instead of hardcoded model strings
- Keep agents simple - default to string output unless structured output is specifically needed
- Follow the main_agent_reference patterns for configuration and providers
- Always include comprehensive testing with TestModel for development

