"""
Report Models

Executive and technical report models following mandatory format standards.
ExecutiveReport must follow the executive output style with action-oriented language.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class KeyMetric(BaseModel):
    """
    Single key metric for executive report.
    """

    label: str = Field(
        ...,
        min_length=1,
        description="Metric label (e.g., 'LTV:CAC Ratio')"
    )
    value: str = Field(
        ...,
        min_length=1,
        description="Metric value with formatting (e.g., '**2.1:1**')"
    )
    context: Optional[str] = Field(
        None,
        description="Brief context or comparison (e.g., 'vs 3.0 benchmark')"
    )
    is_positive: bool = Field(
        True,
        description="Whether this metric is positive/good news"
    )


class ExecutiveReport(BaseModel):
    """
    Executive report following mandatory format standards.

    MANDATORY FORMAT:
    - recommendation: Clear, action-oriented statement (1-2 sentences)
    - key_metrics: 3-5 critical metrics with bold numbers
    - rationale: Why this recommendation (2-3 bullets max)
    - next_steps: Specific actions (3-5 bullets, action verbs)
    - risks: Key risks to monitor (2-3 bullets max)

    Must be scannable in 30 seconds, use bold for numbers, action verbs throughout.
    """

    recommendation: str = Field(
        ...,
        min_length=10,
        description="Primary recommendation (action-oriented, 1-2 sentences)"
    )
    key_metrics: list[KeyMetric] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 critical metrics with bold numbers"
    )
    rationale: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="2-3 bullet points explaining the recommendation"
    )
    next_steps: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 specific action items (use action verbs)"
    )
    risks: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="2-3 key risks to monitor"
    )
    report_title: str = Field(
        ...,
        min_length=1,
        description="Report title (e.g., 'Weekly CFO Report - Q4 2024')"
    )
    report_date: datetime = Field(
        default_factory=datetime.now,
        description="When this report was generated"
    )
    company_id: Optional[str] = Field(
        None,
        description="Company identifier"
    )
    executive_summary: Optional[str] = Field(
        None,
        description="Optional 1-paragraph executive summary"
    )

    @field_validator('key_metrics')
    @classmethod
    def validate_key_metrics_count(cls, v: list[KeyMetric]) -> list[KeyMetric]:
        """
        Ensure 3-5 key metrics per executive format requirements.

        Args:
            v: List of key metrics

        Returns:
            Validated list

        Raises:
            ValueError: If not 3-5 metrics
        """
        if len(v) < 3:
            raise ValueError("Executive report must have at least 3 key metrics")
        if len(v) > 5:
            raise ValueError("Executive report must have no more than 5 key metrics")
        return v

    @field_validator('next_steps')
    @classmethod
    def validate_next_steps_have_action_verbs(cls, v: list[str]) -> list[str]:
        """
        Ensure next steps start with action verbs (soft validation).

        Args:
            v: List of next steps

        Returns:
            Validated list with warning if issues found
        """
        # List of common action verbs
        action_verbs = [
            'implement', 'create', 'develop', 'execute', 'analyze', 'review',
            'reduce', 'increase', 'optimize', 'launch', 'schedule', 'conduct',
            'finalize', 'complete', 'initiate', 'establish', 'prepare', 'monitor',
            'track', 'improve', 'streamline', 'consolidate', 'evaluate', 'assess',
            'build', 'design', 'test', 'deploy', 'update', 'enhance'
        ]

        for step in v:
            first_word = step.split()[0].lower().rstrip(',.:;')
            if first_word not in action_verbs:
                # Soft warning - don't fail validation
                print(f"Warning: Next step '{step}' may not start with action verb")

        return v

    def format_as_markdown(self) -> str:
        """
        Format the executive report as markdown following mandatory format.

        Returns:
            Markdown-formatted report string
        """
        lines = [
            f"# {self.report_title}",
            f"*{self.report_date.strftime('%B %d, %Y')}*",
            "",
            "---",
            "",
        ]

        # Executive Summary (if provided)
        if self.executive_summary:
            lines.extend([
                "## Executive Summary",
                "",
                self.executive_summary,
                "",
            ])

        # Recommendation
        lines.extend([
            "## Recommendation",
            "",
            self.recommendation,
            "",
        ])

        # Key Metrics
        lines.extend([
            "## Key Metrics",
            "",
        ])
        for metric in self.key_metrics:
            icon = "✅" if metric.is_positive else "⚠️"
            context_str = f" ({metric.context})" if metric.context else ""
            lines.append(f"- {icon} **{metric.label}:** {metric.value}{context_str}")
        lines.append("")

        # Rationale
        lines.extend([
            "## Rationale",
            "",
        ])
        for point in self.rationale:
            lines.append(f"- {point}")
        lines.append("")

        # Next Steps
        lines.extend([
            "## Next Steps",
            "",
        ])
        for i, step in enumerate(self.next_steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

        # Risks
        lines.extend([
            "## Risks to Monitor",
            "",
        ])
        for risk in self.risks:
            lines.append(f"- ⚠️  {risk}")
        lines.append("")

        return "\n".join(lines)

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "report_title": "Weekly CFO Report - Q4 2024",
                "recommendation": "Reduce CAC by 30% through channel optimization and raise $2M bridge round by Q1 2025 to extend runway to 18 months.",
                "key_metrics": [
                    {
                        "label": "LTV:CAC Ratio",
                        "value": "**2.1:1**",
                        "context": "vs 3.0 benchmark",
                        "is_positive": False
                    },
                    {
                        "label": "Cash Runway",
                        "value": "**4.6 months**",
                        "context": "at current burn",
                        "is_positive": False
                    },
                    {
                        "label": "Q4 Revenue",
                        "value": "**$850K**",
                        "context": "+120% YoY",
                        "is_positive": True
                    }
                ],
                "rationale": [
                    "LTV:CAC of 2.1:1 is below the 3.0 minimum benchmark, indicating inefficient customer acquisition",
                    "Current runway of 4.6 months requires immediate action to extend cash position",
                    "Q4 represents 70% of annual revenue due to shopping season concentration"
                ],
                "next_steps": [
                    "Reduce Facebook ad spend by 40% and reallocate to higher-ROI channels",
                    "Initiate bridge round conversations with existing investors by Dec 15",
                    "Implement inventory reduction plan to free up $150K in working capital",
                    "Launch referral program to reduce CAC by targeting word-of-mouth growth",
                    "Schedule weekly cash position reviews with CEO"
                ],
                "risks": [
                    "Q1 seasonality drop could accelerate cash burn if not mitigated",
                    "Bridge round may take 60-90 days, requiring interim cost reductions",
                    "Customer churn at 6% monthly may impact LTV calculations"
                ]
            }
        }


class TechnicalReport(BaseModel):
    """
    Detailed technical report with full methodology and calculations.

    Used for detailed analyses that require full transparency into:
    - Data sources and collection methods
    - Calculation formulas and assumptions
    - Validation checks and accuracy metrics
    - Detailed breakdowns and supporting data
    """

    report_title: str = Field(
        ...,
        min_length=1,
        description="Report title"
    )
    report_date: datetime = Field(
        default_factory=datetime.now,
        description="Report generation date"
    )
    company_id: Optional[str] = Field(
        None,
        description="Company identifier"
    )
    overview: str = Field(
        ...,
        min_length=10,
        description="High-level overview of the analysis"
    )
    data_sources: list[str] = Field(
        ...,
        min_length=1,
        description="List of data sources used"
    )
    methodology: str = Field(
        ...,
        min_length=10,
        description="Detailed methodology and approach"
    )
    key_assumptions: dict[str, str] = Field(
        default_factory=dict,
        description="Map of assumption names to values/descriptions"
    )
    calculations: Optional[dict[str, str]] = Field(
        None,
        description="Key calculation formulas and results"
    )
    detailed_findings: list[str] = Field(
        ...,
        min_length=1,
        description="Detailed findings with supporting data"
    )
    validation_checks: Optional[list[str]] = Field(
        None,
        description="Validation checks performed and results"
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Analysis limitations and caveats"
    )
    appendices: Optional[dict[str, str]] = Field(
        None,
        description="Additional supporting data and tables"
    )

    def format_as_markdown(self) -> str:
        """
        Format the technical report as markdown.

        Returns:
            Markdown-formatted report string
        """
        lines = [
            f"# {self.report_title}",
            f"*Technical Report - {self.report_date.strftime('%B %d, %Y')}*",
            "",
            "---",
            "",
            "## Overview",
            "",
            self.overview,
            "",
            "## Data Sources",
            "",
        ]

        for source in self.data_sources:
            lines.append(f"- {source}")
        lines.append("")

        lines.extend([
            "## Methodology",
            "",
            self.methodology,
            "",
        ])

        if self.key_assumptions:
            lines.extend([
                "## Key Assumptions",
                "",
            ])
            for key, value in self.key_assumptions.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        if self.calculations:
            lines.extend([
                "## Calculations",
                "",
            ])
            for calc_name, formula in self.calculations.items():
                lines.append(f"### {calc_name}")
                lines.append(f"```")
                lines.append(formula)
                lines.append(f"```")
                lines.append("")

        lines.extend([
            "## Detailed Findings",
            "",
        ])
        for i, finding in enumerate(self.detailed_findings, 1):
            lines.append(f"{i}. {finding}")
        lines.append("")

        if self.validation_checks:
            lines.extend([
                "## Validation Checks",
                "",
            ])
            for check in self.validation_checks:
                lines.append(f"- ✓ {check}")
            lines.append("")

        if self.limitations:
            lines.extend([
                "## Limitations & Caveats",
                "",
            ])
            for limitation in self.limitations:
                lines.append(f"- {limitation}")
            lines.append("")

        if self.appendices:
            lines.extend([
                "## Appendices",
                "",
            ])
            for appendix_name, content in self.appendices.items():
                lines.append(f"### {appendix_name}")
                lines.append("")
                lines.append(content)
                lines.append("")

        return "\n".join(lines)

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "report_title": "Unit Economics Analysis - Q4 2024",
                "overview": "Comprehensive analysis of customer acquisition costs and lifetime value",
                "data_sources": [
                    "QuickBooks Online (marketing expenses)",
                    "Shopify (customer orders and revenue)",
                    "Internal CRM (churn data)"
                ],
                "methodology": "Calculated CAC using total marketing expenses divided by new customers. LTV calculated using average revenue per account multiplied by gross margin, divided by annual churn rate.",
                "key_assumptions": {
                    "Gross Margin": "65%",
                    "Attribution Window": "30 days",
                    "Churn Calculation": "Trailing 12-month average"
                },
                "detailed_findings": [
                    "CAC increased 45% QoQ from $350 to $508",
                    "LTV remained stable at $1,200",
                    "Monthly churn rate decreased from 6.5% to 6.0%"
                ],
                "limitations": [
                    "Marketing attribution is based on last-touch model",
                    "LTV calculation assumes constant gross margin over customer lifetime"
                ]
            }
        }
