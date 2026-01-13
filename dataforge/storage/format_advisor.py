"""
DataForge Format Advisor

Recommend optimal file formats based on use case and requirements.

Features:
    - Format comparison
    - Use case recommendations
    - Migration guidance

Author: DataForge Team
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class FileFormat(Enum):
    """Supported file formats."""
    PARQUET = "parquet"
    DELTA = "delta"
    ORC = "orc"
    AVRO = "avro"
    JSON = "json"
    CSV = "csv"


class UseCase(Enum):
    """Data processing use cases."""
    ANALYTICS = "analytics"
    ETL = "etl"
    STREAMING = "streaming"
    ML = "ml"
    DATA_LAKE = "data_lake"
    INTERCHANGE = "interchange"


@dataclass
class FormatRecommendation:
    """
    Format recommendation with justification.

    Attributes:
        format: Recommended file format
        score: Suitability score (0-100)
        reasons: List of reasons for recommendation
        considerations: Things to consider
        compression: Recommended compression
    """
    format: FileFormat
    score: int
    reasons: List[str]
    considerations: List[str]
    compression: str = "snappy"


class FormatAdvisor:
    """
    Advisor for file format selection.

    Example:
        >>> advisor = FormatAdvisor()
        >>> rec = advisor.recommend(
        ...     use_case=UseCase.ANALYTICS,
        ...     data_size_gb=100,
        ...     need_acid=True
        ... )
        >>> print(f"Recommended: {rec.format.value}")
        >>> for reason in rec.reasons:
        ...     print(f"  - {reason}")
    """

    FORMAT_PROPERTIES = {
        FileFormat.PARQUET: {
            "columnar": True,
            "splittable": True,
            "schema_evolution": True,
            "compression_ratio": 0.25,
            "read_speed": 95,
            "write_speed": 80,
            "ecosystem_support": 95,
        },
        FileFormat.DELTA: {
            "columnar": True,
            "splittable": True,
            "schema_evolution": True,
            "acid": True,
            "time_travel": True,
            "compression_ratio": 0.25,
            "read_speed": 90,
            "write_speed": 85,
            "ecosystem_support": 85,
        },
        FileFormat.ORC: {
            "columnar": True,
            "splittable": True,
            "schema_evolution": True,
            "compression_ratio": 0.20,
            "read_speed": 90,
            "write_speed": 75,
            "ecosystem_support": 80,
        },
        FileFormat.AVRO: {
            "columnar": False,
            "splittable": True,
            "schema_evolution": True,
            "compression_ratio": 0.40,
            "read_speed": 75,
            "write_speed": 90,
            "ecosystem_support": 85,
        },
        FileFormat.JSON: {
            "columnar": False,
            "splittable": False,
            "schema_evolution": True,
            "compression_ratio": 0.60,
            "read_speed": 50,
            "write_speed": 70,
            "ecosystem_support": 100,
        },
        FileFormat.CSV: {
            "columnar": False,
            "splittable": True,
            "schema_evolution": False,
            "compression_ratio": 0.70,
            "read_speed": 40,
            "write_speed": 60,
            "ecosystem_support": 100,
        },
    }

    def recommend(
        self,
        use_case: UseCase,
        data_size_gb: float = 1.0,
        need_acid: bool = False,
        need_time_travel: bool = False,
        need_streaming: bool = False,
        interoperability: bool = False
    ) -> FormatRecommendation:
        """
        Recommend optimal file format.

        Args:
            use_case: Primary use case
            data_size_gb: Estimated data size in GB
            need_acid: Require ACID transactions
            need_time_travel: Require time travel capability
            need_streaming: Require streaming support
            interoperability: Need to share with non-Spark systems

        Returns:
            FormatRecommendation with details
        """
        scores = {}

        for fmt, props in self.FORMAT_PROPERTIES.items():
            score = 50  # Base score

            # Use case scoring
            if use_case == UseCase.ANALYTICS:
                if props.get("columnar"):
                    score += 30
                score += props.get("read_speed", 0) * 0.2

            elif use_case == UseCase.ETL:
                score += props.get("write_speed", 0) * 0.2
                if props.get("schema_evolution"):
                    score += 10

            elif use_case == UseCase.STREAMING:
                if fmt == FileFormat.DELTA:
                    score += 30  # Best for streaming
                score += props.get("write_speed", 0) * 0.1

            elif use_case == UseCase.DATA_LAKE:
                if fmt == FileFormat.DELTA:
                    score += 40  # Delta is ideal for data lakes
                elif props.get("columnar"):
                    score += 20

            elif use_case == UseCase.INTERCHANGE:
                score += props.get("ecosystem_support", 0) * 0.3

            # Requirement-based adjustments
            if need_acid and not props.get("acid"):
                if fmt != FileFormat.DELTA:
                    score -= 50

            if need_time_travel and not props.get("time_travel"):
                if fmt != FileFormat.DELTA:
                    score -= 30

            if need_streaming and fmt == FileFormat.DELTA:
                score += 20

            if interoperability:
                score += props.get("ecosystem_support", 0) * 0.1

            # Size considerations
            if data_size_gb > 10 and not props.get("columnar"):
                score -= 20

            scores[fmt] = max(0, min(100, int(score)))

        # Get best format
        best_format = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best_format]

        # Generate reasons
        reasons = self._generate_reasons(best_format, use_case, need_acid, need_time_travel)
        considerations = self._generate_considerations(best_format)
        compression = self._recommend_compression(best_format, use_case)

        return FormatRecommendation(
            format=best_format,
            score=best_score,
            reasons=reasons,
            considerations=considerations,
            compression=compression
        )

    def _generate_reasons(
        self,
        fmt: FileFormat,
        use_case: UseCase,
        need_acid: bool,
        need_time_travel: bool
    ) -> List[str]:
        """Generate recommendation reasons."""
        reasons = []
        props = self.FORMAT_PROPERTIES[fmt]

        if props.get("columnar"):
            reasons.append("Columnar format enables efficient column pruning")

        if fmt == FileFormat.DELTA:
            reasons.append("Delta Lake provides ACID transactions")
            reasons.append("Built-in OPTIMIZE and VACUUM for maintenance")
            if need_time_travel:
                reasons.append("Time travel for data versioning and recovery")

        if props.get("schema_evolution"):
            reasons.append("Supports schema evolution for flexibility")

        if use_case == UseCase.ANALYTICS:
            reasons.append("Optimized for analytical query patterns")

        return reasons

    def _generate_considerations(self, fmt: FileFormat) -> List[str]:
        """Generate considerations for the format."""
        considerations = []

        if fmt == FileFormat.DELTA:
            considerations.append("Requires Delta Lake runtime (Databricks or OSS)")
            considerations.append("Consider OPTIMIZE frequency for write-heavy tables")

        if fmt == FileFormat.PARQUET:
            considerations.append("No built-in ACID - consider Delta Lake for transactions")

        if fmt in [FileFormat.CSV, FileFormat.JSON]:
            considerations.append("Consider migrating to columnar format for better performance")

        return considerations

    def _recommend_compression(self, fmt: FileFormat, use_case: UseCase) -> str:
        """Recommend compression based on format and use case."""
        if fmt in [FileFormat.PARQUET, FileFormat.DELTA, FileFormat.ORC]:
            if use_case == UseCase.ANALYTICS:
                return "snappy"  # Fast decompression
            else:
                return "zstd"  # Better compression ratio
        elif fmt == FileFormat.AVRO:
            return "snappy"
        else:
            return "gzip"

    def compare_formats(self, formats: List[FileFormat]) -> str:
        """
        Generate comparison table for formats.

        Args:
            formats: List of formats to compare

        Returns:
            Formatted comparison string
        """
        lines = []
        lines.append("Format Comparison:")
        lines.append("-" * 70)
        lines.append(f"{'Property':<20} " + " ".join(f"{f.value:<12}" for f in formats))
        lines.append("-" * 70)

        properties = ["columnar", "splittable", "schema_evolution", "compression_ratio"]

        for prop in properties:
            values = []
            for fmt in formats:
                val = self.FORMAT_PROPERTIES[fmt].get(prop, "N/A")
                if isinstance(val, bool):
                    val = "Yes" if val else "No"
                elif isinstance(val, float):
                    val = f"{val:.0%}"
                values.append(f"{val:<12}")
            lines.append(f"{prop:<20} " + " ".join(values))

        return "\n".join(lines)
