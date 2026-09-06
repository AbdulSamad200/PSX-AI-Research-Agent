"""Pydantic structured output models for OpenAI LLM reasoning nodes."""

from typing import List, Optional
from pydantic import BaseModel, Field


class StructuredReasoningOutput(BaseModel):
    """Typed Pydantic model for structured AI financial reasoning.
    
    Enforces evidence-grounded, analytical interpretation without generating
    unsupported BUY/SELL/HOLD scores.
    """
    investment_view: str = Field(
        ..., 
        description="High-level analytical summary of the investment view (e.g., 'High-Quality Dividend Growth with Margin Expansion')."
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0.0 and 1.0 based on evidence completeness and signal consistency."
    )
    fundamental_interpretation: str = Field(
        ..., 
        description="Analytical interpretation of earnings, profit margins, PEG, revenue growth, and financial statement integrity."
    )
    technical_interpretation: str = Field(
        ..., 
        description="Analytical interpretation of technical indicators (SMA 50/200, RSI 14, MACD, 52-week high/low distance, max drawdown, volatility)."
    )
    dividend_interpretation: str = Field(
        ..., 
        description="Analytical interpretation of trailing twelve month (TTM) dividend yield, annual historical dividend growth, and payout sustainability."
    )
    key_strengths: List[str] = Field(
        default_factory=list, 
        description="Key corporate and financial strengths directly supported by evidence."
    )
    key_risks: List[str] = Field(
        default_factory=list, 
        description="Key investment, valuation, operational, or macro risks directly supported by evidence."
    )
    thesis_assessment: str = Field(
        ..., 
        description="Objective evaluation of the user's investment hypothesis/thesis against provided evidence."
    )
    evidence_ids: List[str] = Field(
        default_factory=list, 
        description="List of Evidence IDs (e.g., ['EVD-FUND-001', 'EVD-TECH-001']) referenced in this reasoning output."
    )
