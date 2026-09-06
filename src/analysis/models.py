"""Data models for PSXenius financial & technical analysis engine."""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union


def get_utc_timestamp() -> str:
    """Returns ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SourceMetric:
    """Represents a metric directly provided by pypsx-toolkit or psx-mcp-server."""
    metric_name: str
    value: Any
    unit: Optional[str] = None
    financial_period: Optional[str] = None  # e.g., "FY2025", "Q2 2026", "TTM"
    source: str = "pypsx-toolkit"
    retrieved_at: str = field(default_factory=get_utc_timestamp)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DerivedMetric:
    """Represents a metric calculated by PSXenius Python engine."""
    metric_name: str
    value: Optional[Union[float, int, str]]
    unit: str  # e.g., "%", "PKR", "ratio", "points"
    calculation_period: str  # e.g., "5y", "1y", "14d"
    methodology: str
    input_references: List[str]
    calculated_at: str = field(default_factory=get_utc_timestamp)
    status: str = "success"  # "success", "insufficient_data", "error"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Represents a validation check comparing a calculated metric against a source metric."""
    metric_name: str
    source_value: Any
    calculated_value: Optional[float]
    difference: Optional[float]
    status: str  # "match", "discrepancy", "not_applicable", "error"
    notes: str
    validated_at: str = field(default_factory=get_utc_timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Unified container for source, derived, and validation metrics for a single ticker."""
    symbol: str
    retrieved_at: str = field(default_factory=get_utc_timestamp)
    source_metrics: Dict[str, Any] = field(default_factory=dict)
    derived_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "retrieved_at": self.retrieved_at,
            "source_metrics": self.source_metrics,
            "derived_metrics": self.derived_metrics,
            "validation_results": self.validation_results,
        }
