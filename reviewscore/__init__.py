"""
ReviewScore: Misinformed Peer Review Detection with Large Language Models

A comprehensive implementation of the ReviewScore system as described in the paper:
"ReviewScore: Misinformed Peer Review Detection with Large Language Models"
(arXiv:2509.21679)

This package provides:
- BASE REVIEW SCORE evaluation for questions and claims
- ADVANCED REVIEW SCORE evaluation with premise-level factuality
- Automatic argument reconstruction engine
- LCEL workflows using LangChain
- LangGraph flows for complex evaluation pipelines
- Model evaluation system with multiple proprietary models
- Comprehensive evaluation metrics

Usage:
    from reviewscore import ReviewScoreEvaluator, create_review_point

    # Create a review point
    question = create_review_point(
        text="Does this paper address the scalability issues?",
        point_type=ReviewPointType.QUESTION,
        paper_context="...",
        review_context="..."
    )

    # Evaluate using LCEL workflow
    from reviewscore.lcel_workflows import create_lcel_workflow

    workflow = create_lcel_workflow("claude-3-5-sonnet-20241022")
    result = workflow.evaluate_review_point(question)

    print(f"Score: {result.base_score}, Misinformed: {result.is_misinformed}")
"""

from .core import (
    ReviewPoint,
    Question,
    Claim,
    Argument,
    Premise,
    ReviewPointType,
    MisinformedStatus,
    ReviewScoreResult,
    EvaluationMetrics,
    AggregationMethod,
    ModelConfig,
    PROPRIETARY_MODELS,
    create_review_point,
    parse_review_points_from_text,
)

from .base_evaluation import BaseReviewScoreEvaluator, create_base_evaluator

from .argument_reconstruction import (
    ArgumentReconstructionEngine,
    create_reconstruction_engine,
)

from .aggregation import (
    PremiseAggregator,
    AdvancedReviewScoreEvaluator,
    create_advanced_evaluator,
)

from .lcel_workflows import (
    ReviewScoreLCELWorkflow,
    ReviewScoreParallelWorkflow,
    create_lcel_workflow,
    create_parallel_workflow,
)

from .langgraph_flows import (
    ReviewScoreLangGraphFlow,
    ReviewScoreState,
    create_langgraph_flow,
)

from .evaluation_metrics import ReviewScoreEvaluator, create_evaluator

from .model_evaluation import ModelEvaluationSystem, create_model_evaluation_system

__version__ = "1.0.0"
__author__ = "ashhadahsan"
__email__ = "ashhadahsan@gmail.com"
__url__ = "https://github.com/ashhadahsan/reviewscore"
__description__ = "Misinformed Peer Review Detection with Large Language Models"

# Main classes for easy access
__all__ = [
    # Core classes
    "ReviewPoint",
    "Question",
    "Claim",
    "Argument",
    "Premise",
    "ReviewPointType",
    "MisinformedStatus",
    "ReviewScoreResult",
    "EvaluationMetrics",
    "AggregationMethod",
    "ModelConfig",
    "PROPRIETARY_MODELS",
    "create_review_point",
    "parse_review_points_from_text",
    # Evaluation classes
    "BaseReviewScoreEvaluator",
    "create_base_evaluator",
    "ArgumentReconstructionEngine",
    "create_reconstruction_engine",
    "PremiseAggregator",
    "AdvancedReviewScoreEvaluator",
    "create_advanced_evaluator",
    # Workflow classes
    "ReviewScoreLCELWorkflow",
    "ReviewScoreParallelWorkflow",
    "create_lcel_workflow",
    "create_parallel_workflow",
    "ReviewScoreLangGraphFlow",
    "ReviewScoreState",
    "create_langgraph_flow",
    # Evaluation and metrics
    "ReviewScoreEvaluator",
    "create_evaluator",
    "ModelEvaluationSystem",
    "create_model_evaluation_system",
]
