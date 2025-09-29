"""
LangGraph flows for complex ReviewScore evaluation pipelines.
Implements state-based evaluation workflows using LangGraph.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END

# ToolExecutor not available in current LangGraph version
# from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import operator

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
from .lcel_workflows import ReviewScoreLCELWorkflow


class ReviewScoreState(TypedDict):
    """State for ReviewScore evaluation workflow."""

    review_point: ReviewPoint
    paper_context: str
    review_context: str
    current_step: str
    base_evaluation: Optional[Dict[str, Any]]
    advanced_evaluation: Optional[Dict[str, Any]]
    final_result: Optional[ReviewScoreResult]
    error_message: Optional[str]
    evaluation_metadata: Dict[str, Any]


class ReviewScoreLangGraphFlow:
    """
    LangGraph-based workflow for ReviewScore evaluation.
    Implements complex state-based evaluation pipelines.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.workflow = self._create_workflow()
        self.app = self._compile_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(ReviewScoreState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_evaluation)
        workflow.add_node("classify_point", self._classify_review_point)
        workflow.add_node("evaluate_question", self._evaluate_question)
        workflow.add_node("evaluate_claim", self._evaluate_claim)
        workflow.add_node("evaluate_argument", self._evaluate_argument)
        workflow.add_node("reconstruct_argument", self._reconstruct_argument)
        workflow.add_node("evaluate_premises", self._evaluate_premises)
        workflow.add_node("aggregate_scores", self._aggregate_scores)
        workflow.add_node("finalize_result", self._finalize_result)
        workflow.add_node("handle_error", self._handle_error)

        # Add edges
        workflow.add_edge("initialize", "classify_point")

        # Set entry point
        workflow.set_entry_point("initialize")

        # Conditional routing based on review point type
        workflow.add_conditional_edges(
            "classify_point",
            self._route_evaluation,
            {
                "question": "evaluate_question",
                "claim": "evaluate_claim",
                "argument": "evaluate_argument",
                "error": "handle_error",
            },
        )

        # Question evaluation path
        workflow.add_edge("evaluate_question", "finalize_result")

        # Claim evaluation path
        workflow.add_edge("evaluate_claim", "finalize_result")

        # Argument evaluation path
        workflow.add_edge("evaluate_argument", "reconstruct_argument")
        workflow.add_edge("reconstruct_argument", "evaluate_premises")
        workflow.add_edge("evaluate_premises", "aggregate_scores")
        workflow.add_edge("aggregate_scores", "finalize_result")

        # Error handling
        workflow.add_edge("handle_error", END)
        workflow.add_edge("finalize_result", END)

        return workflow

    def _compile_workflow(self):
        """Compile the workflow into an executable app."""
        return self.workflow.compile()

    def _initialize_evaluation(self, state: ReviewScoreState) -> ReviewScoreState:
        """Initialize the evaluation process."""
        try:
            review_point = state["review_point"]

            return {
                **state,
                "current_step": "initialized",
                "paper_context": review_point.paper_context,
                "review_context": review_point.review_context,
                "evaluation_metadata": {
                    "model_used": self.model_config.model_name,
                    "workflow_type": "langgraph",
                },
            }
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Initialization error: {str(e)}",
            }

    def _classify_review_point(self, state: ReviewScoreState) -> ReviewScoreState:
        """Classify the type of review point."""
        try:
            review_point = state["review_point"]

            if isinstance(review_point, Question):
                point_type = "question"
            elif isinstance(review_point, Claim):
                point_type = "claim"
            elif isinstance(review_point, Argument):
                point_type = "argument"
            else:
                point_type = "error"

            return {
                **state,
                "current_step": f"classified_as_{point_type}",
                "evaluation_metadata": {
                    **state.get("evaluation_metadata", {}),
                    "point_type": point_type,
                },
            }
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Classification error: {str(e)}",
            }

    def _route_evaluation(self, state: ReviewScoreState) -> str:
        """Route to appropriate evaluation based on point type."""
        if state.get("current_step", "").startswith("error"):
            return "error"

        point_type = state.get("evaluation_metadata", {}).get("point_type", "error")
        return point_type

    def _evaluate_question(self, state: ReviewScoreState) -> ReviewScoreState:
        """Evaluate a question review point."""
        try:
            # Create LCEL workflow for question evaluation
            lcel_workflow = ReviewScoreLCELWorkflow(self.model_config)
            result = lcel_workflow.evaluate_review_point(state["review_point"])

            return {
                **state,
                "current_step": "question_evaluated",
                "base_evaluation": {
                    "score": result.base_score,
                    "is_misinformed": result.is_misinformed,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                },
                "final_result": result,
            }
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Question evaluation error: {str(e)}",
            }

    def _evaluate_claim(self, state: ReviewScoreState) -> ReviewScoreState:
        """Evaluate a claim review point."""
        try:
            # Create LCEL workflow for claim evaluation
            lcel_workflow = ReviewScoreLCELWorkflow(self.model_config)
            result = lcel_workflow.evaluate_review_point(state["review_point"])

            return {
                **state,
                "current_step": "claim_evaluated",
                "base_evaluation": {
                    "score": result.base_score,
                    "is_misinformed": result.is_misinformed,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                },
                "final_result": result,
            }
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Claim evaluation error: {str(e)}",
            }

    def _evaluate_argument(self, state: ReviewScoreState) -> ReviewScoreState:
        """Initial evaluation of an argument."""
        try:
            # Create LCEL workflow for argument evaluation
            lcel_workflow = ReviewScoreLCELWorkflow(self.model_config)
            result = lcel_workflow.evaluate_review_point(state["review_point"])

            return {
                **state,
                "current_step": "argument_initial_evaluation",
                "base_evaluation": {
                    "score": result.base_score,
                    "is_misinformed": result.is_misinformed,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                },
            }
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Argument evaluation error: {str(e)}",
            }

    def _reconstruct_argument(self, state: ReviewScoreState) -> ReviewScoreState:
        """Reconstruct argument into premises and conclusion."""
        try:
            from .argument_reconstruction import create_reconstruction_engine

            reconstruction_engine = create_reconstruction_engine(
                self.model_config.model_name
            )
            argument = state["review_point"]

            # Reconstruct the argument
            reconstructed_argument = reconstruction_engine.reconstruct_argument(
                argument
            )

            return {
                **state,
                "current_step": "argument_reconstructed",
                "review_point": reconstructed_argument,
                "evaluation_metadata": {
                    **state.get("evaluation_metadata", {}),
                    "reconstruction": {
                        "is_valid": reconstructed_argument.is_valid,
                        "is_faithful": reconstructed_argument.is_faithful,
                        "num_premises": len(reconstructed_argument.premises),
                        "conclusion": reconstructed_argument.conclusion,
                    },
                },
            }
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Argument reconstruction error: {str(e)}",
            }

    def _evaluate_premises(self, state: ReviewScoreState) -> ReviewScoreState:
        """Evaluate factuality of each premise."""
        try:
            from .argument_reconstruction import create_reconstruction_engine

            reconstruction_engine = create_reconstruction_engine(
                self.model_config.model_name
            )
            argument = state["review_point"]

            # Extract premises with factuality scores
            premises = reconstruction_engine.extract_premises_with_factuality(argument)

            return {
                **state,
                "current_step": "premises_evaluated",
                "review_point": argument,
                "evaluation_metadata": {
                    **state.get("evaluation_metadata", {}),
                    "premise_evaluation": {
                        "num_premises": len(premises),
                        "premise_scores": [
                            p.factuality_score
                            for p in premises
                            if p.factuality_score is not None
                        ],
                        "avg_premise_score": (
                            sum(
                                p.factuality_score
                                for p in premises
                                if p.factuality_score is not None
                            )
                            / len(
                                [p for p in premises if p.factuality_score is not None]
                            )
                            if premises
                            else 3.0
                        ),
                    },
                },
            }
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Premise evaluation error: {str(e)}",
            }

    def _aggregate_scores(self, state: ReviewScoreState) -> ReviewScoreState:
        """Aggregate premise scores into final argument score."""
        try:
            from .aggregation import create_advanced_evaluator

            advanced_evaluator = create_advanced_evaluator(
                model_name=self.model_config.model_name,
                aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            )
            argument = state["review_point"]

            # Get advanced evaluation
            advanced_score = advanced_evaluator.evaluate_argument(argument)
            advanced_is_misinformed = advanced_score <= 2

            return {
                **state,
                "current_step": "scores_aggregated",
                "advanced_evaluation": {
                    "score": advanced_score,
                    "is_misinformed": advanced_is_misinformed,
                    "aggregation_method": "weighted_average",
                },
                "evaluation_metadata": {
                    **state.get("evaluation_metadata", {}),
                    "advanced_evaluation": advanced_evaluator.get_evaluation_metadata(
                        argument
                    ),
                },
            }
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Score aggregation error: {str(e)}",
            }

    def _finalize_result(self, state: ReviewScoreState) -> ReviewScoreState:
        """Finalize the evaluation result."""
        try:
            review_point = state["review_point"]
            base_eval = state.get("base_evaluation", {})
            advanced_eval = state.get("advanced_evaluation", {})

            # Determine final scores
            base_score = base_eval.get("score", 3.0)
            advanced_score = advanced_eval.get("score", base_score)
            is_misinformed = advanced_eval.get(
                "is_misinformed", base_eval.get("is_misinformed", False)
            )

            # Create final result
            final_result = ReviewScoreResult(
                review_point=review_point,
                base_score=base_score,
                advanced_score=advanced_score if advanced_eval else None,
                is_misinformed=is_misinformed,
                confidence=max(
                    abs(base_score - 3) / 2,
                    abs(advanced_score - 3) / 2 if advanced_score else 0,
                ),
                reasoning=base_eval.get("reasoning", ""),
                model_used=self.model_config.model_name,
                evaluation_metadata=state.get("evaluation_metadata", {}),
            )

            return {**state, "current_step": "completed", "final_result": final_result}
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Finalization error: {str(e)}",
            }

    def _handle_error(self, state: ReviewScoreState) -> ReviewScoreState:
        """Handle errors in the evaluation process."""
        error_message = state.get("error_message", "Unknown error")

        # Create error result
        error_result = ReviewScoreResult(
            review_point=state["review_point"],
            base_score=3.0,
            advanced_score=3.0,
            is_misinformed=False,
            confidence=0.0,
            reasoning=f"Error: {error_message}",
            model_used=self.model_config.model_name,
            evaluation_metadata={
                "error": True,
                "error_message": error_message,
                "workflow_step": state.get("current_step", "unknown"),
            },
        )

        return {**state, "current_step": "error_handled", "final_result": error_result}

    def evaluate_review_point(self, review_point: ReviewPoint) -> ReviewScoreResult:
        """
        Evaluate a review point using the LangGraph workflow.

        Args:
            review_point: Review point to evaluate

        Returns:
            ReviewScoreResult with evaluation details
        """
        # Initialize state
        initial_state = {
            "review_point": review_point,
            "paper_context": "",
            "review_context": "",
            "current_step": "initial",
            "base_evaluation": None,
            "advanced_evaluation": None,
            "final_result": None,
            "error_message": None,
            "evaluation_metadata": {},
        }

        # Run the workflow
        final_state = self.app.invoke(initial_state)

        # Return the final result
        return final_state.get(
            "final_result",
            ReviewScoreResult(
                review_point=review_point,
                base_score=3.0,
                advanced_score=3.0,
                is_misinformed=False,
                confidence=0.0,
                reasoning="Workflow failed to produce result",
                model_used=self.model_config.model_name,
                evaluation_metadata={"error": True},
            ),
        )


def create_langgraph_flow(
    model_name: str = "claude-3-5-sonnet-20241022",
) -> ReviewScoreLangGraphFlow:
    """
    Factory function to create a LangGraph flow.

    Args:
        model_name: Name of the model to use

    Returns:
        ReviewScoreLangGraphFlow instance
    """
    # Find the model configuration
    model_config = None
    for config in PROPRIETARY_MODELS:
        if config.model_name == model_name:
            model_config = config
            break

    if model_config is None:
        raise ValueError(f"Model {model_name} not found in available models")

    return ReviewScoreLangGraphFlow(model_config)
