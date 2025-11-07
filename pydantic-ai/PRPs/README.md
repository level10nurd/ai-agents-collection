# Planning & Research Documents (PRPs)

This directory contains all planning, research, and reference documentation for the AI CFO System project.

## 📁 Directory Structure

```
PRPs/
├── 01_pydantic_ai/              # Pydantic AI framework documentation
├── 02_cfo_domain/               # CFO agent architecture & patterns
├── 03_forecasting/              # Time series forecasting research
├── 04_api_integrations/         # External API integration guides
├── 05_project_docs/             # Project-specific documentation
└── templates/                   # PRP templates
```

## 📚 Document Categories

### 01_pydantic_ai/ - Pydantic AI Framework
Core documentation for building multi-agent systems with Pydantic AI.

- **[PYDANTIC_AI_DOCUMENTATION_SUMMARY.md](01_pydantic_ai/PYDANTIC_AI_DOCUMENTATION_SUMMARY.md)** ⭐⭐ CRITICAL
  - Comprehensive guide to multi-agent coordination patterns
  - Dependency injection, tool design, structured outputs
  - Testing strategies with TestModel and FunctionModel
  - **When to read**: Before implementing any agent or tool

- **[PYDANTIC_AI_QUICK_REFERENCE.md](01_pydantic_ai/PYDANTIC_AI_QUICK_REFERENCE.md)** ⭐ REFERENCE
  - Copy-paste code templates for common patterns
  - Agent creation, tool registration, testing snippets
  - **When to use**: During daily implementation

- **[DOCUMENTATION_URLS.md](01_pydantic_ai/DOCUMENTATION_URLS.md)**
  - Direct links to official Pydantic AI docs with section anchors
  - Quick lookup for specific topics
  - **When to use**: When you need official docs for a specific feature

---

### 02_cfo_domain/ - CFO Agent Architecture
Domain-specific research on CFO agent design patterns and best practices.

- **[CFO_AGENT_RESEARCH.md](02_cfo_domain/CFO_AGENT_RESEARCH.md)** ⭐⭐ CRITICAL
  - Analysis of 10+ existing CFO agent implementations
  - Hierarchical Coordinator + Specialist pattern (recommended)
  - Domain-based decomposition strategies
  - 26 best practices and 20 anti-patterns
  - **When to read**: Before designing agent architecture

- **[CFO_AGENT_QUICK_REFERENCE.md](02_cfo_domain/CFO_AGENT_QUICK_REFERENCE.md)** ⭐ REFERENCE
  - Actionable CFO agent patterns with code snippets
  - Recommended architecture diagram
  - Critical validations and benchmarks
  - 4-phase implementation plan
  - **When to use**: During CFO agent implementation

---

### 03_forecasting/ - Time Series Forecasting
Research on forecasting libraries and implementation strategies for VoChill's extreme seasonality.

- **[FORECASTING_IMPLEMENTATION_PLAN.md](03_forecasting/FORECASTING_IMPLEMENTATION_PLAN.md)** ⭐⭐ CRITICAL
  - Complete 5-phase roadmap for Prophet-based forecasting
  - Handles VoChill's 70% Nov-Dec seasonal concentration
  - Pydantic models, system prompts, testing strategy
  - Prophet configuration for multiplicative seasonality
  - **When to read**: Before implementing forecasting specialist

- **[forecasting_libraries_research.md](03_forecasting/forecasting_libraries_research.md)**
  - Comprehensive evaluation of 8 forecasting libraries
  - Why Prophet wins for VoChill's use case
  - Code examples, pros/cons, gotchas
  - **When to read**: If considering alternative forecasting approaches

- **[forecasting_quick_reference.md](03_forecasting/forecasting_quick_reference.md)** ⭐ REFERENCE
  - Working code snippets for Prophet, pmdarima, numpy-financial
  - VoChill-specific Prophet configuration
  - **When to use**: During forecasting tool implementation

- **[FORECASTING_EXECUTIVE_SUMMARY.md](03_forecasting/FORECASTING_EXECUTIVE_SUMMARY.md)**
  - High-level overview of forecasting approach
  - Decision summary and rationale
  - **When to read**: For project overview or stakeholder communication

- **[README_FORECASTING.md](03_forecasting/README_FORECASTING.md)**
  - Navigation guide for forecasting documents
  - Reading order recommendations

---

### 04_api_integrations/ - External API Integration
Guides for integrating with QuickBooks, Shopify, Supabase, and other external services.

- **[financial_api_integration_research.md](04_api_integrations/financial_api_integration_research.md)** ⭐⭐ CRITICAL
  - QuickBooks Online API: OAuth 2.0, rate limits, key endpoints
  - Shopify Admin API: Authentication, pagination, webhooks
  - Supabase: JWT auth, query builder, real-time subscriptions
  - Python SDK examples and critical gotchas
  - **When to read**: Before implementing any API integration

---

### 05_project_docs/ - Project-Specific Documentation
Main project documentation, requirements, and status tracking.

- **[AI-CFO.md](05_project_docs/AI-CFO.md)** ⭐⭐⭐ PRIMARY PRP
  - Complete Product Requirements & Planning document
  - Goals, architecture, technical requirements
  - Success criteria, implementation blueprint
  - Task list in dependency order
  - **When to read**: Start here! Read at project start and reference frequently

- **[INITIAL.md](05_project_docs/INITIAL.md)**
  - Original VoChill requirements and context
  - Business needs and constraints
  - **When to read**: For historical context on project origins

- **[RESEARCH_COMPLETE.md](05_project_docs/RESEARCH_COMPLETE.md)**
  - Research completion summary
  - Links to all research documents
  - **When to read**: To understand research phase outcomes

---

### templates/ - Document Templates
Templates for creating new PRPs or documentation.

- **[prp_base.md](templates/prp_base.md)**
  - Standard PRP template structure
  - **When to use**: Creating new PRP documents

---

## 🎯 Quick Start Reading Guide

### New to the project?
Read in this order:
1. **[AI-CFO.md](05_project_docs/AI-CFO.md)** - Main PRP, understand the full project
2. **[PYDANTIC_AI_DOCUMENTATION_SUMMARY.md](01_pydantic_ai/PYDANTIC_AI_DOCUMENTATION_SUMMARY.md)** - Learn the framework
3. **[CFO_AGENT_RESEARCH.md](02_cfo_domain/CFO_AGENT_RESEARCH.md)** - Understand CFO agent patterns

### Implementing specific features?
- **Building agents**: [PYDANTIC_AI_QUICK_REFERENCE.md](01_pydantic_ai/PYDANTIC_AI_QUICK_REFERENCE.md) + [CFO_AGENT_QUICK_REFERENCE.md](02_cfo_domain/CFO_AGENT_QUICK_REFERENCE.md)
- **Forecasting**: [FORECASTING_IMPLEMENTATION_PLAN.md](03_forecasting/FORECASTING_IMPLEMENTATION_PLAN.md) + [forecasting_quick_reference.md](03_forecasting/forecasting_quick_reference.md)
- **API integrations**: [financial_api_integration_research.md](04_api_integrations/financial_api_integration_research.md)

### Need quick lookup?
- [PYDANTIC_AI_QUICK_REFERENCE.md](01_pydantic_ai/PYDANTIC_AI_QUICK_REFERENCE.md) - Code templates
- [CFO_AGENT_QUICK_REFERENCE.md](02_cfo_domain/CFO_AGENT_QUICK_REFERENCE.md) - CFO patterns
- [forecasting_quick_reference.md](03_forecasting/forecasting_quick_reference.md) - Forecasting code
- [DOCUMENTATION_URLS.md](01_pydantic_ai/DOCUMENTATION_URLS.md) - Official docs links

---

## 📝 Legend

- ⭐⭐⭐ PRIMARY: Main project document, read first
- ⭐⭐ CRITICAL: Must read before implementing related feature
- ⭐ REFERENCE: Quick lookup during implementation

---

## 🔄 Document Maintenance

- **Last Organization**: 2025-11-06
- **Status**: Documents reorganized into logical categories
- **Next Review**: As new research is added

When adding new documents:
1. Place in appropriate category folder
2. Update this README with description
3. Add to relevant reading guide section
4. Mark with appropriate priority (⭐⭐⭐, ⭐⭐, or ⭐)
