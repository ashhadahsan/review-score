"""
Paper Review Result - Final aggregation of review points for complete paper evaluation.
Based on the ReviewScore paper methodology.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import statistics
from dataclasses import dataclass

from .core import ReviewPoint, ReviewScoreResult, ReviewPointType


class ReviewQualityLevel(str, Enum):
    """Overall review quality levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"


class PaperReviewSummary(BaseModel):
    """Summary of a complete paper review evaluation."""

    # Basic information
    paper_id: str
    total_review_points: int
    questions_count: int
    claims_count: int
    arguments_count: int

    # Misinformed statistics
    total_misinformed: int
    misinformed_questions: int
    misinformed_claims: int
    misinformed_arguments: int

    # Score statistics
    average_base_score: float
    average_advanced_score: float
    average_confidence: float

    # Quality assessment
    overall_quality: ReviewQualityLevel
    quality_score: float  # 1-5 scale

    # Detailed breakdown
    question_scores: List[float] = Field(default_factory=list)
    claim_scores: List[float] = Field(default_factory=list)
    argument_scores: List[float] = Field(default_factory=list)

    # Metadata
    evaluation_metadata: Dict[str, Any] = Field(default_factory=dict)


class PaperReviewResult(BaseModel):
    """Complete paper review evaluation result."""

    # Paper identification
    paper_id: str
    paper_title: Optional[str] = None

    # Individual review point results
    review_point_results: List[ReviewScoreResult]

    # Aggregated summary
    summary: PaperReviewSummary

    # Detailed analysis
    misinformed_review_points: List[ReviewScoreResult]
    high_quality_review_points: List[ReviewScoreResult]

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    # Metadata
    evaluation_metadata: Dict[str, Any] = Field(default_factory=dict)


class PaperReviewAggregator:
    """
    Aggregates individual review point results into final paper review assessment.
    Implements the methodology from the ReviewScore paper.
    """

    def __init__(self, quality_threshold: float = 3.0):
        """
        Initialize the aggregator.

        Args:
            quality_threshold: Minimum score for considering a review point as high quality
        """
        self.quality_threshold = quality_threshold

    def aggregate_paper_review(
        self,
        paper_id: str,
        review_point_results: List[ReviewScoreResult],
        paper_title: Optional[str] = None,
    ) -> PaperReviewResult:
        """
        Aggregate individual review point results into final paper review assessment.

        Args:
            paper_id: Unique identifier for the paper
            review_point_results: List of individual review point evaluation results
            paper_title: Optional title of the paper

        Returns:
            PaperReviewResult with complete assessment
        """
        if not review_point_results:
            return self._create_empty_result(paper_id, paper_title)

        # Separate by review point type
        questions, claims, arguments = self._categorize_review_points(
            review_point_results
        )

        # Calculate statistics
        summary = self._calculate_summary(
            paper_id, review_point_results, questions, claims, arguments
        )

        # Identify misinformed and high-quality review points
        misinformed_points = [r for r in review_point_results if r.is_misinformed]
        high_quality_points = [
            r
            for r in review_point_results
            if r.base_score >= self.quality_threshold and not r.is_misinformed
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            summary, misinformed_points, high_quality_points
        )

        return PaperReviewResult(
            paper_id=paper_id,
            paper_title=paper_title,
            review_point_results=review_point_results,
            summary=summary,
            misinformed_review_points=misinformed_points,
            high_quality_review_points=high_quality_points,
            recommendations=recommendations,
            evaluation_metadata={
                "aggregation_method": "paper_review_aggregator",
                "quality_threshold": self.quality_threshold,
                "total_evaluation_time": self._calculate_total_time(
                    review_point_results
                ),
            },
        )

    def _categorize_review_points(
        self, results: List[ReviewScoreResult]
    ) -> tuple[
        List[ReviewScoreResult], List[ReviewScoreResult], List[ReviewScoreResult]
    ]:
        """Categorize review points by type."""
        questions = []
        claims = []
        arguments = []

        for result in results:
            point_type = result.review_point.type
            if point_type == ReviewPointType.QUESTION:
                questions.append(result)
            elif point_type == ReviewPointType.CLAIM:
                claims.append(result)
            elif point_type == ReviewPointType.ARGUMENT:
                arguments.append(result)

        return questions, claims, arguments

    def _calculate_summary(
        self,
        paper_id: str,
        all_results: List[ReviewScoreResult],
        questions: List[ReviewScoreResult],
        claims: List[ReviewScoreResult],
        arguments: List[ReviewScoreResult],
    ) -> PaperReviewSummary:
        """Calculate comprehensive summary statistics."""

        # Count review points
        total_points = len(all_results)
        questions_count = len(questions)
        claims_count = len(claims)
        arguments_count = len(arguments)

        # Count misinformed points
        misinformed_questions = sum(1 for r in questions if r.is_misinformed)
        misinformed_claims = sum(1 for r in claims if r.is_misinformed)
        misinformed_arguments = sum(1 for r in arguments if r.is_misinformed)
        total_misinformed = (
            misinformed_questions + misinformed_claims + misinformed_arguments
        )

        # Calculate average scores
        base_scores = [r.base_score for r in all_results if r.base_score is not None]
        advanced_scores = [
            r.advanced_score for r in all_results if r.advanced_score is not None
        ]
        confidence_scores = [
            r.confidence for r in all_results if r.confidence is not None
        ]

        avg_base_score = statistics.mean(base_scores) if base_scores else 3.0
        avg_advanced_score = (
            statistics.mean(advanced_scores) if advanced_scores else 3.0
        )
        avg_confidence = (
            statistics.mean(confidence_scores) if confidence_scores else 0.0
        )

        # Determine overall quality
        overall_quality, quality_score = self._assess_overall_quality(
            avg_base_score, total_misinformed, total_points
        )

        # Extract scores by type
        question_scores = [r.base_score for r in questions if r.base_score is not None]
        claim_scores = [r.base_score for r in claims if r.base_score is not None]
        argument_scores = [r.base_score for r in arguments if r.base_score is not None]

        return PaperReviewSummary(
            paper_id=paper_id,
            total_review_points=total_points,
            questions_count=questions_count,
            claims_count=claims_count,
            arguments_count=arguments_count,
            total_misinformed=total_misinformed,
            misinformed_questions=misinformed_questions,
            misinformed_claims=misinformed_claims,
            misinformed_arguments=misinformed_arguments,
            average_base_score=avg_base_score,
            average_advanced_score=avg_advanced_score,
            average_confidence=avg_confidence,
            overall_quality=overall_quality,
            quality_score=quality_score,
            question_scores=question_scores,
            claim_scores=claim_scores,
            argument_scores=argument_scores,
            evaluation_metadata={
                "misinformed_rate": (
                    total_misinformed / total_points if total_points > 0 else 0.0
                ),
                "questions_misinformed_rate": (
                    misinformed_questions / questions_count
                    if questions_count > 0
                    else 0.0
                ),
                "claims_misinformed_rate": (
                    misinformed_claims / claims_count if claims_count > 0 else 0.0
                ),
                "arguments_misinformed_rate": (
                    misinformed_arguments / arguments_count
                    if arguments_count > 0
                    else 0.0
                ),
            },
        )

    def _assess_overall_quality(
        self, avg_score: float, misinformed_count: int, total_count: int
    ) -> tuple[ReviewQualityLevel, float]:
        """Assess overall review quality based on scores and misinformed rate."""

        misinformed_rate = misinformed_count / total_count if total_count > 0 else 0.0

        # Calculate quality score (1-5 scale)
        if avg_score >= 4.5 and misinformed_rate <= 0.1:
            return ReviewQualityLevel.EXCELLENT, 5.0
        elif avg_score >= 4.0 and misinformed_rate <= 0.2:
            return ReviewQualityLevel.GOOD, 4.0
        elif avg_score >= 3.0 and misinformed_rate <= 0.3:
            return ReviewQualityLevel.FAIR, 3.0
        elif avg_score >= 2.0 and misinformed_rate <= 0.5:
            return ReviewQualityLevel.POOR, 2.0
        else:
            return ReviewQualityLevel.VERY_POOR, 1.0

    def _generate_recommendations(
        self,
        summary: PaperReviewSummary,
        misinformed_points: List[ReviewScoreResult],
        high_quality_points: List[ReviewScoreResult],
    ) -> List[str]:
        """Generate recommendations based on the review assessment."""
        recommendations = []

        # Overall quality recommendations
        if summary.overall_quality == ReviewQualityLevel.EXCELLENT:
            recommendations.append(
                "This is an excellent review with high-quality feedback."
            )
        elif summary.overall_quality == ReviewQualityLevel.GOOD:
            recommendations.append(
                "This is a good review with mostly constructive feedback."
            )
        elif summary.overall_quality == ReviewQualityLevel.FAIR:
            recommendations.append(
                "This review has some valuable points but could be improved."
            )
        elif summary.overall_quality == ReviewQualityLevel.POOR:
            recommendations.append(
                "This review has significant issues and needs substantial improvement."
            )
        else:  # VERY_POOR
            recommendations.append(
                "This review is of very poor quality and should be rejected."
            )

        # Misinformed point recommendations
        if summary.total_misinformed > 0:
            recommendations.append(
                f"Found {summary.total_misinformed} misinformed review points that need correction."
            )

            if summary.misinformed_questions > 0:
                recommendations.append(
                    f"Reviewer asked {summary.misinformed_questions} questions that are already answered in the paper."
                )

            if summary.misinformed_claims > 0:
                recommendations.append(
                    f"Reviewer made {summary.misinformed_claims} factually incorrect claims."
                )

            if summary.misinformed_arguments > 0:
                recommendations.append(
                    f"Reviewer made {summary.misinformed_arguments} arguments with incorrect premises."
                )

        # Positive feedback
        if len(high_quality_points) > 0:
            recommendations.append(
                f"Review contains {len(high_quality_points)} high-quality review points."
            )

        # Specific recommendations based on paper findings
        if summary.questions_count > 0:
            question_misinformed_rate = (
                summary.misinformed_questions / summary.questions_count
            )
            if question_misinformed_rate > 0.3:
                recommendations.append(
                    "Reviewer should check if questions are already answered in the paper before asking them."
                )

        if summary.arguments_count > 0:
            argument_misinformed_rate = (
                summary.misinformed_arguments / summary.arguments_count
            )
            if argument_misinformed_rate > 0.3:
                recommendations.append(
                    "Reviewer should verify the factual accuracy of arguments before making them."
                )

        return recommendations

    def _calculate_total_time(self, results: List[ReviewScoreResult]) -> float:
        """Calculate total evaluation time from metadata."""
        total_time = 0.0
        for result in results:
            if "evaluation_time" in result.evaluation_metadata:
                total_time += result.evaluation_metadata["evaluation_time"]
        return total_time

    def _create_empty_result(
        self, paper_id: str, paper_title: Optional[str] = None
    ) -> PaperReviewResult:
        """Create empty result for papers with no review points."""
        empty_summary = PaperReviewSummary(
            paper_id=paper_id,
            total_review_points=0,
            questions_count=0,
            claims_count=0,
            arguments_count=0,
            total_misinformed=0,
            misinformed_questions=0,
            misinformed_claims=0,
            misinformed_arguments=0,
            average_base_score=3.0,
            average_advanced_score=3.0,
            average_confidence=0.0,
            overall_quality=ReviewQualityLevel.FAIR,
            quality_score=3.0,
            evaluation_metadata={"empty_review": True},
        )

        return PaperReviewResult(
            paper_id=paper_id,
            paper_title=paper_title,
            review_point_results=[],
            summary=empty_summary,
            misinformed_review_points=[],
            high_quality_review_points=[],
            recommendations=["No review points found for this paper."],
            evaluation_metadata={"empty_review": True},
        )


def create_paper_review_aggregator(
    quality_threshold: float = 3.0,
) -> PaperReviewAggregator:
    """
    Factory function to create a paper review aggregator.

    Args:
        quality_threshold: Minimum score for considering a review point as high quality

    Returns:
        PaperReviewAggregator instance
    """
    return PaperReviewAggregator(quality_threshold)


# Example usage and testing
if __name__ == "__main__":
    # Test the paper review aggregator
    print("Paper Review Result Aggregation Test")
    print("=" * 50)

    # This would be used with actual review point results
    print("Paper review aggregation functionality ready!")
    print(
        "Use this to aggregate individual review point results into final paper assessment."
    )
