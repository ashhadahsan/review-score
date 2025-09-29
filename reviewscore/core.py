"""
Core data structures and classes for ReviewScore implementation.
Based on the paper: ReviewScore: Misinformed Peer Review Detection with Large Language Models
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum
import json


class ReviewPointType(str, Enum):
    """Types of review points as defined in the paper."""

    QUESTION = "question"
    CLAIM = "claim"
    ARGUMENT = "argument"


class MisinformedStatus(str, Enum):
    """Misinformed status for review points."""

    MISINFORMED = "misinformed"
    NOT_MISINFORMED = "not_misinformed"


class ReviewPoint(BaseModel):
    """Base class for review points as defined in Definition 1."""

    id: str
    text: str
    type: ReviewPointType
    paper_context: str  # The submitted paper content for context
    review_context: str  # The full review text for context

    class Config:
        use_enum_values = True


class Question(ReviewPoint):
    """Question review point - can be answered by the paper."""

    type: Literal[ReviewPointType.QUESTION] = ReviewPointType.QUESTION
    is_answerable_by_paper: Optional[bool] = None
    answer_in_paper: Optional[str] = None


class Claim(ReviewPoint):
    """Claim review point - a weakness without supporting reasons."""

    type: Literal[ReviewPointType.CLAIM] = ReviewPointType.CLAIM
    is_factually_correct: Optional[bool] = None
    factuality_score: Optional[float] = None  # 1-5 scale


class Premise(BaseModel):
    """Individual premise in an argument."""

    text: str
    is_explicit: bool = True  # Whether this premise is explicitly stated
    factuality_score: Optional[float] = None  # 1-5 scale
    is_factually_correct: Optional[bool] = None


class Argument(ReviewPoint):
    """Argument review point - a weakness with supporting reasons."""

    type: Literal[ReviewPointType.ARGUMENT] = ReviewPointType.ARGUMENT
    premises: List[Premise] = Field(default_factory=list)
    conclusion: str = ""
    is_valid: Optional[bool] = None  # Logical validity
    is_faithful: Optional[bool] = None  # Faithfulness to original argument
    aggregated_factuality_score: Optional[float] = None  # Aggregated from premises


class ReviewScoreResult(BaseModel):
    """Result of ReviewScore evaluation."""

    review_point: ReviewPoint
    base_score: Optional[float] = None  # BASE REVIEW SCORE (1-5)
    advanced_score: Optional[float] = None  # ADVANCED REVIEW SCORE (1-5)
    is_misinformed: Optional[bool] = None  # Binary classification
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    model_used: Optional[str] = None
    evaluation_metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    """Metrics for evaluating ReviewScore performance."""

    f1_score: float
    kappa_score: float
    accuracy: float
    precision: float
    recall: float
    human_model_agreement: float


class AggregationMethod(str, Enum):
    """Methods for aggregating premise factuality scores."""

    LOGICAL_CONJUNCTION = "logical_conjunction"
    WEIGHTED_AVERAGE = "weighted_average"


class ModelConfig(BaseModel):
    """Configuration for LLM models used in evaluation."""

    model_name: str
    provider: str  # "openai", "anthropic", "google"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None


# Proprietary models used in the paper
PROPRIETARY_MODELS = [
    ModelConfig(
        model_name="claude-3-5-sonnet-20241022", provider="anthropic", temperature=0.0
    ),
    ModelConfig(
        model_name="claude-sonnet-4",
        provider="anthropic",
        temperature=0.0,
    ),
    ModelConfig(model_name="gpt-4o", provider="openai", temperature=0.0),
    ModelConfig(
        model_name="gpt-5",
        provider="openai",
    ),
    ModelConfig(model_name="gemini-2.5-flash", provider="google", temperature=0.0),
]


def create_review_point(
    text: str,
    point_type: ReviewPointType,
    paper_context: str,
    review_context: str,
    point_id: Optional[str] = None,
) -> Union[Question, Claim, Argument]:
    """Factory function to create review points based on type."""
    if point_id is None:
        point_id = f"{point_type}_{hash(text) % 10000}"

    if point_type == ReviewPointType.QUESTION:
        return Question(
            id=point_id,
            text=text,
            paper_context=paper_context,
            review_context=review_context,
        )
    elif point_type == ReviewPointType.CLAIM:
        return Claim(
            id=point_id,
            text=text,
            paper_context=paper_context,
            review_context=review_context,
        )
    elif point_type == ReviewPointType.ARGUMENT:
        return Argument(
            id=point_id,
            text=text,
            paper_context=paper_context,
            review_context=review_context,
        )
    else:
        raise ValueError(f"Unknown review point type: {point_type}")


def parse_review_points_from_text(
    review_text: str, paper_context: str
) -> List[ReviewPoint]:
    """
    Parse review text to extract individual review points.
    This is a simplified implementation - in practice, this would use
    more sophisticated NLP techniques or manual annotation.
    """
    # This is a placeholder implementation
    # In the actual paper, this would involve more sophisticated parsing
    sentences = review_text.split(". ")
    review_points = []

    for i, sentence in enumerate(sentences):
        if sentence.strip():
            # Simple heuristic to determine type
            if "?" in sentence:
                point_type = ReviewPointType.QUESTION
            elif any(
                word in sentence.lower()
                for word in ["however", "but", "although", "despite"]
            ):
                point_type = ReviewPointType.ARGUMENT
            else:
                point_type = ReviewPointType.CLAIM

            review_point = create_review_point(
                text=sentence.strip(),
                point_type=point_type,
                paper_context=paper_context,
                review_context=review_text,
                point_id=f"point_{i}",
            )
            review_points.append(review_point)

    return review_points
