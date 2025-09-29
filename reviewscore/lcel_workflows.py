"""
LCEL (LangChain Expression Language) workflows for ReviewScore evaluation.
Implements complex evaluation pipelines using LangChain's LCEL framework.
"""

from typing import List, Dict, Any, Optional, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.runnables.utils import Input, Output
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import json

from .core import (
    ReviewPoint,
    Question,
    Claim,
    Argument,
    ReviewScoreResult,
    ReviewPointType,
    ModelConfig,
    PROPRIETARY_MODELS,
)
from .base_evaluation import BaseReviewScoreEvaluator
from .aggregation import (
    AdvancedReviewScoreEvaluator,
    PremiseAggregator,
    AggregationMethod,
)


class ReviewScoreLCELWorkflow:
    """
    LCEL workflow for comprehensive ReviewScore evaluation.
    Implements the complete evaluation pipeline using LangChain's LCEL.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.llm = self._initialize_llm()
        self.base_evaluator = BaseReviewScoreEvaluator(model_config)
        self.advanced_evaluator = self._create_advanced_evaluator()

        # Create LCEL chains
        self.question_evaluation_chain = self._create_question_evaluation_chain()
        self.claim_evaluation_chain = self._create_claim_evaluation_chain()
        self.argument_evaluation_chain = self._create_argument_evaluation_chain()
        self.comprehensive_evaluation_chain = (
            self._create_comprehensive_evaluation_chain()
        )

    def _initialize_llm(self):
        """Initialize the LLM based on the model configuration."""
        if self.model_config.provider == "openai":
            return ChatOpenAI(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                api_key=self.model_config.api_key,
            )
        elif self.model_config.provider == "anthropic":
            return ChatAnthropic(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens or 1000,
                api_key=self.model_config.api_key or "mock_key",
            )
        elif self.model_config.provider == "google":
            return ChatGoogleGenerativeAI(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens or 1000,
                api_key=self.model_config.api_key or "mock_key",
            )
        else:
            raise ValueError(f"Unsupported provider: {self.model_config.provider}")

    def _create_advanced_evaluator(self) -> AdvancedReviewScoreEvaluator:
        """Create advanced evaluator for arguments."""
        from .aggregation import create_advanced_evaluator

        return create_advanced_evaluator(
            model_name=self.model_config.model_name,
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
        )

    def _create_question_evaluation_chain(self):
        """Create LCEL chain for question evaluation."""
        question_prompt = ChatPromptTemplate.from_template(
            """
You are an expert reviewer evaluating whether a question in a peer review can be answered by the submitted paper.

PAPER CONTEXT:
{paper_context}

REVIEW CONTEXT:
{review_context}

QUESTION TO EVALUATE:
{question_text}

Evaluate if the question can be answered by the paper. Provide your response in JSON format:
{{
    "is_answerable": true/false,
    "unanswerability_score": 1-5 (1=definitely answerable, 5=definitely not answerable),
    "answer_in_paper": "specific text from paper that answers the question",
    "reasoning": "explanation of your evaluation"
}}
"""
        )

        return question_prompt | self.llm | JsonOutputParser()

    def _create_claim_evaluation_chain(self):
        """Create LCEL chain for claim evaluation."""
        claim_prompt = ChatPromptTemplate.from_template(
            """
You are an expert reviewer evaluating whether a claim in a peer review is factually correct regarding the submitted paper.

PAPER CONTEXT:
{paper_context}

REVIEW CONTEXT:
{review_context}

CLAIM TO EVALUATE:
{claim_text}

Evaluate if the claim is factually correct. Provide your response in JSON format:
{{
    "is_factually_correct": true/false,
    "factuality_score": 1-5 (1=definitely incorrect, 5=definitely correct),
    "evidence_from_paper": "specific text from paper that supports or contradicts the claim",
    "reasoning": "explanation of your evaluation"
}}
"""
        )

        return claim_prompt | self.llm | JsonOutputParser()

    def _create_argument_evaluation_chain(self):
        """Create LCEL chain for argument evaluation."""
        argument_prompt = ChatPromptTemplate.from_template(
            """
You are an expert reviewer evaluating an argument in a peer review. You need to:
1. Reconstruct the argument into premises and conclusion
2. Evaluate the factuality of each premise
3. Provide an overall assessment

PAPER CONTEXT:
{paper_context}

REVIEW CONTEXT:
{review_context}

ARGUMENT TO EVALUATE:
{argument_text}

Provide your evaluation in JSON format:
{{
    "conclusion": "the main conclusion of the argument",
    "premises": [
        {{
            "text": "premise text",
            "is_explicit": true/false,
            "factuality_score": 1-5
        }}
    ],
    "overall_factuality_score": 1-5,
    "reasoning": "explanation of your evaluation"
}}
"""
        )

        return argument_prompt | self.llm | JsonOutputParser()

    def _create_comprehensive_evaluation_chain(self):
        """Create comprehensive evaluation chain that routes to appropriate evaluation."""

        def route_evaluation(input_data: Dict[str, Any]) -> Dict[str, Any]:
            review_point = input_data["review_point"]

            if isinstance(review_point, Question):
                return self._evaluate_question_lcel(input_data)
            elif isinstance(review_point, Claim):
                return self._evaluate_claim_lcel(input_data)
            elif isinstance(review_point, Argument):
                return self._evaluate_argument_lcel(input_data)
            else:
                raise ValueError(f"Unknown review point type: {type(review_point)}")

        return RunnableLambda(route_evaluation)

    def _evaluate_question_lcel(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate question using LCEL chain."""
        question = input_data["review_point"]

        # Run the question evaluation chain
        result = self.question_evaluation_chain.invoke(
            {
                "paper_context": question.paper_context,
                "review_context": question.review_context,
                "question_text": question.text,
            }
        )

        # Convert to ReviewScoreResult format
        unanswerability_score = result.get("unanswerability_score", 3)
        is_misinformed = unanswerability_score <= 2

        return {
            "review_point": question,
            "base_score": unanswerability_score,
            "advanced_score": None,
            "is_misinformed": is_misinformed,
            "confidence": abs(unanswerability_score - 3) / 2,
            "reasoning": result.get("reasoning", ""),
            "model_used": self.model_config.model_name,
            "evaluation_metadata": result,
        }

    def _evaluate_claim_lcel(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate claim using LCEL chain."""
        claim = input_data["review_point"]

        # Run the claim evaluation chain
        result = self.claim_evaluation_chain.invoke(
            {
                "paper_context": claim.paper_context,
                "review_context": claim.review_context,
                "claim_text": claim.text,
            }
        )

        # Convert to ReviewScoreResult format
        factuality_score = result.get("factuality_score", 3)
        is_misinformed = factuality_score <= 2

        return {
            "review_point": claim,
            "base_score": factuality_score,
            "advanced_score": None,
            "is_misinformed": is_misinformed,
            "confidence": abs(factuality_score - 3) / 2,
            "reasoning": result.get("reasoning", ""),
            "model_used": self.model_config.model_name,
            "evaluation_metadata": result,
        }

    def _evaluate_argument_lcel(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate argument using LCEL chain."""
        argument = input_data["review_point"]

        # Run the argument evaluation chain
        result = self.argument_evaluation_chain.invoke(
            {
                "paper_context": argument.paper_context,
                "review_context": argument.review_context,
                "argument_text": argument.text,
            }
        )

        # Extract scores
        overall_score = result.get("overall_factuality_score", 3)
        is_misinformed = overall_score <= 2

        # Use advanced evaluator for more detailed analysis
        advanced_score = self.advanced_evaluator.evaluate_argument(argument)
        advanced_is_misinformed = advanced_score <= 2

        return {
            "review_point": argument,
            "base_score": overall_score,
            "advanced_score": advanced_score,
            "is_misinformed": advanced_is_misinformed,  # Use advanced result
            "confidence": abs(advanced_score - 3) / 2,
            "reasoning": result.get("reasoning", ""),
            "model_used": self.model_config.model_name,
            "evaluation_metadata": {
                **result,
                "advanced_evaluation": self.advanced_evaluator.get_evaluation_metadata(
                    argument
                ),
            },
        }

    def evaluate_review_point(self, review_point: ReviewPoint) -> ReviewScoreResult:
        """
        Evaluate a review point using the comprehensive LCEL workflow.

        Args:
            review_point: Review point to evaluate

        Returns:
            ReviewScoreResult with evaluation details
        """
        try:
            # Run the comprehensive evaluation chain
            result = self.comprehensive_evaluation_chain.invoke(
                {"review_point": review_point}
            )

            # Convert to ReviewScoreResult
            return ReviewScoreResult(
                review_point=result["review_point"],
                base_score=result["base_score"],
                advanced_score=result["advanced_score"],
                is_misinformed=result["is_misinformed"],
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                model_used=result["model_used"],
                evaluation_metadata=result["evaluation_metadata"],
            )

        except Exception as e:
            return ReviewScoreResult(
                review_point=review_point,
                base_score=3.0,
                advanced_score=3.0,
                is_misinformed=False,
                confidence=0.0,
                reasoning=f"Error in LCEL evaluation: {str(e)}",
                model_used=self.model_config.model_name,
                evaluation_metadata={"error": str(e)},
            )

    def evaluate_batch(
        self, review_points: List[ReviewPoint]
    ) -> List[ReviewScoreResult]:
        """
        Evaluate multiple review points using parallel processing.

        Args:
            review_points: List of review points to evaluate

        Returns:
            List of ReviewScoreResult objects
        """
        results = []

        for review_point in review_points:
            result = self.evaluate_review_point(review_point)
            results.append(result)

        return results


class ReviewScoreParallelWorkflow:
    """
    Parallel evaluation workflow using LCEL for multiple models.
    Implements the multi-model evaluation described in the paper.
    """

    def __init__(self, model_configs: List[ModelConfig]):
        self.model_configs = model_configs
        self.workflows = [ReviewScoreLCELWorkflow(config) for config in model_configs]

    def evaluate_with_multiple_models(
        self, review_point: ReviewPoint
    ) -> List[ReviewScoreResult]:
        """
        Evaluate a review point with multiple models in parallel.

        Args:
            review_point: Review point to evaluate

        Returns:
            List of results from each model
        """
        results = []

        for workflow in self.workflows:
            result = workflow.evaluate_review_point(review_point)
            results.append(result)

        return results

    def get_consensus_score(self, results: List[ReviewScoreResult]) -> Dict[str, Any]:
        """
        Calculate consensus score from multiple model results.

        Args:
            results: List of results from multiple models

        Returns:
            Dictionary with consensus information
        """
        if not results:
            return {"consensus_score": 3.0, "agreement": 0.0, "variance": 0.0}

        # Extract scores
        base_scores = [r.base_score for r in results if r.base_score is not None]
        advanced_scores = [
            r.advanced_score for r in results if r.advanced_score is not None
        ]
        misinformed_votes = [
            r.is_misinformed for r in results if r.is_misinformed is not None
        ]

        # Calculate consensus
        consensus_base = sum(base_scores) / len(base_scores) if base_scores else 3.0
        consensus_advanced = (
            sum(advanced_scores) / len(advanced_scores) if advanced_scores else 3.0
        )
        consensus_misinformed = (
            sum(misinformed_votes) / len(misinformed_votes) > 0.5
            if misinformed_votes
            else False
        )

        # Calculate agreement metrics
        base_variance = np.var(base_scores) if len(base_scores) > 1 else 0.0
        advanced_variance = np.var(advanced_scores) if len(advanced_scores) > 1 else 0.0

        return {
            "consensus_base_score": consensus_base,
            "consensus_advanced_score": consensus_advanced,
            "consensus_misinformed": consensus_misinformed,
            "base_variance": base_variance,
            "advanced_variance": advanced_variance,
            "num_models": len(results),
            "agreement_level": (
                "high" if max(base_variance, advanced_variance) < 1.0 else "low"
            ),
        }


def create_lcel_workflow(
    model_name: str = "claude-3-5-sonnet-20241022",
) -> ReviewScoreLCELWorkflow:
    """
    Factory function to create an LCEL workflow.

    Args:
        model_name: Name of the model to use

    Returns:
        ReviewScoreLCELWorkflow instance
    """
    # Find the model configuration
    model_config = None
    for config in PROPRIETARY_MODELS:
        if config.model_name == model_name:
            model_config = config
            break

    if model_config is None:
        raise ValueError(f"Model {model_name} not found in available models")

    return ReviewScoreLCELWorkflow(model_config)


def create_parallel_workflow(
    model_names: List[str] = None,
) -> ReviewScoreParallelWorkflow:
    """
    Factory function to create a parallel workflow with multiple models.

    Args:
        model_names: List of model names to use (defaults to all proprietary models)

    Returns:
        ReviewScoreParallelWorkflow instance
    """
    if model_names is None:
        model_names = [config.model_name for config in PROPRIETARY_MODELS]

    model_configs = []
    for model_name in model_names:
        for config in PROPRIETARY_MODELS:
            if config.model_name == model_name:
                model_configs.append(config)
                break

    if not model_configs:
        raise ValueError("No valid model configurations found")

    return ReviewScoreParallelWorkflow(model_configs)
