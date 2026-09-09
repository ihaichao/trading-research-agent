"""Report contract: the shape everything downstream agrees on."""

from tra.report.render import render_markdown
from tra.report.schema import (
    Claim,
    ClaimKind,
    Confidence,
    MetricPoint,
    MetricSeries,
    Report,
    ReportMeta,
    Section,
    SectionKind,
    Source,
    SourceKind,
    make_source_id,
)

__all__ = [
    "Claim",
    "ClaimKind",
    "Confidence",
    "MetricPoint",
    "MetricSeries",
    "Report",
    "ReportMeta",
    "Section",
    "SectionKind",
    "Source",
    "SourceKind",
    "make_source_id",
    "render_markdown",
]
