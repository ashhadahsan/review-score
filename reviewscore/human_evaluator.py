"""
Human evaluator using LangGraph's human-in-the-loop functionality.
Implements the human annotation process described in the paper.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END

# ToolExecutor not available in current LangGraph version
# from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .core import (
    ReviewPoint,
    Question,
    Claim,
    Argument,
    ReviewScoreResult,
    ReviewPointType,
)
from .paper_faithful import PaperFaithfulReviewScore, PaperFaithfulConfig


class HumanAnnotationState:
    """State for human annotation workflow."""

    review_point: ReviewPoint
    paper_content: str
    annotator_id: str
    annotation_step: str
    current_question: str
    human_response: str
    annotation_data: Dict[str, Any]
    is_complete: bool
    messages: List[BaseMessage]


class HumanAnnotationStep(Enum):
    """Steps in the human annotation process."""

    INITIALIZE = "initialize"
    CLASSIFY_POINT = "classify_point"
    EVALUATE_QUESTION = "evaluate_question"
    EVALUATE_CLAIM = "evaluate_claim"
    EVALUATE_ARGUMENT = "evaluate_argument"
    FINALIZE_ANNOTATION = "finalize_annotation"


class HumanEvaluator:
    """
    Human evaluator using LangGraph's human-in-the-loop functionality.
    Implements the human annotation process described in the paper.
    """

    def __init__(self, config: PaperFaithfulConfig = None):
        self.config = config or PaperFaithfulConfig()
        self.paper_evaluator = PaperFaithfulReviewScore(config)
        self.workflow = self._create_human_annotation_workflow()
        self.app = self._compile_workflow()

    def _create_human_annotation_workflow(self) -> StateGraph:
        """Create the human annotation workflow using LangGraph."""
        workflow = StateGraph(HumanAnnotationState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_annotation)
        workflow.add_node("classify_point", self._classify_review_point)
        workflow.add_node("evaluate_question", self._evaluate_question_human)
        workflow.add_node("evaluate_claim", self._evaluate_claim_human)
        workflow.add_node("evaluate_argument", self._evaluate_argument_human)
        workflow.add_node("finalize_annotation", self._finalize_annotation)
        workflow.add_node("human_input", self._get_human_input)

        # Set entry point
        workflow.set_entry_point("initialize")

        # Add edges
        workflow.add_edge("initialize", "classify_point")

        # Conditional routing based on review point type
        workflow.add_conditional_edges(
            "classify_point",
            self._route_evaluation,
            {
                "question": "evaluate_question",
                "claim": "evaluate_claim",
                "argument": "evaluate_argument",
                "human_input": "human_input",
            },
        )

        # Human input routing
        workflow.add_conditional_edges(
            "human_input",
            self._route_after_human_input,
            {
                "question": "evaluate_question",
                "claim": "evaluate_claim",
                "argument": "evaluate_argument",
                "finalize": "finalize_annotation",
            },
        )

        # Evaluation paths
        workflow.add_edge("evaluate_question", "finalize_annotation")
        workflow.add_edge("evaluate_claim", "finalize_annotation")
        workflow.add_edge("evaluate_argument", "finalize_annotation")
        workflow.add_edge("finalize_annotation", END)

        return workflow

    def _compile_workflow(self):
        """Compile the workflow into an executable app."""
        memory = MemorySaver()
        return self.workflow.compile(checkpointer=memory)

    def _initialize_annotation(
        self, state: HumanAnnotationState
    ) -> HumanAnnotationState:
        """Initialize the human annotation process."""
        return {
            **state,
            "annotation_step": HumanAnnotationStep.INITIALIZE.value,
            "annotation_data": {},
            "is_complete": False,
            "messages": [
                SystemMessage(
                    content="You are a human annotator for ReviewScore evaluation."
                )
            ],
        }

    def _classify_review_point(
        self, state: HumanAnnotationState
    ) -> HumanAnnotationState:
        """Classify the type of review point."""
        review_point = state["review_point"]

        if isinstance(review_point, Question):
            point_type = "question"
        elif isinstance(review_point, Claim):
            point_type = "claim"
        elif isinstance(review_point, Argument):
            point_type = "argument"
        else:
            point_type = "unknown"

        return {
            **state,
            "annotation_step": HumanAnnotationStep.CLASSIFY_POINT.value,
            "annotation_data": {
                **state.get("annotation_data", {}),
                "point_type": point_type,
            },
        }

    def _route_evaluation(self, state: HumanAnnotationState) -> str:
        """Route to appropriate evaluation based on point type."""
        point_type = state.get("annotation_data", {}).get("point_type", "unknown")

        if point_type in ["question", "claim", "argument"]:
            return point_type
        else:
            return "human_input"  # Need human input for unknown types

    def _route_after_human_input(self, state: HumanAnnotationState) -> str:
        """Route after human input."""
        human_response = state.get("human_response", "").lower()

        if "question" in human_response:
            return "question"
        elif "claim" in human_response:
            return "claim"
        elif "argument" in human_response:
            return "argument"
        else:
            return "finalize"

    def _evaluate_question_human(
        self, state: HumanAnnotationState
    ) -> HumanAnnotationState:
        """Evaluate question with human input."""
        review_point = state["review_point"]

        # Create human input prompt
        prompt = f"""
Please evaluate this question in a peer review:

QUESTION: {review_point.text}

PAPER CONTEXT: {state.get("paper_content", "")[:500]}...

Is this question answerable by the paper? Please provide:
1. Your score (1-5): 1=definitely answerable, 5=definitely not answerable
2. Your reasoning
3. Any specific text from the paper that answers the question (if answerable)

Format your response as JSON:
{{
    "score": 1-5,
    "reasoning": "your reasoning",
    "answer_in_paper": "specific text if answerable"
}}
"""

        return {
            **state,
            "annotation_step": HumanAnnotationStep.EVALUATE_QUESTION.value,
            "current_question": prompt,
            "messages": state.get("messages", []) + [HumanMessage(content=prompt)],
        }

    def _evaluate_claim_human(
        self, state: HumanAnnotationState
    ) -> HumanAnnotationState:
        """Evaluate claim with human input."""
        review_point = state["review_point"]

        # Create human input prompt
        prompt = f"""
Please evaluate this claim in a peer review:

CLAIM: {review_point.text}

PAPER CONTEXT: {state.get("paper_content", "")[:500]}...

Is this claim factually correct regarding the paper? Please provide:
1. Your score (1-5): 1=definitely incorrect, 5=definitely correct
2. Your reasoning
3. Any specific text from the paper that supports or contradicts the claim

Format your response as JSON:
{{
    "score": 1-5,
    "reasoning": "your reasoning",
    "evidence_from_paper": "specific text from paper"
}}
"""

        return {
            **state,
            "annotation_step": HumanAnnotationStep.EVALUATE_CLAIM.value,
            "current_question": prompt,
            "messages": state.get("messages", []) + [HumanMessage(content=prompt)],
        }

    def _evaluate_argument_human(
        self, state: HumanAnnotationState
    ) -> HumanAnnotationState:
        """Evaluate argument with human input."""
        review_point = state["review_point"]

        # Create human input prompt
        prompt = f"""
Please evaluate this argument in a peer review:

ARGUMENT: {review_point.text}

PAPER CONTEXT: {state.get("paper_content", "")[:500]}...

Please provide:
1. Your overall score (1-5): 1=definitely incorrect, 5=definitely correct
2. Identify the premises and conclusion
3. Evaluate each premise's factuality
4. Your reasoning

Format your response as JSON:
{{
    "score": 1-5,
    "premises": ["premise1", "premise2"],
    "conclusion": "conclusion",
    "premise_scores": [1-5, 1-5],
    "reasoning": "your reasoning"
}}
"""

        return {
            **state,
            "annotation_step": HumanAnnotationStep.EVALUATE_ARGUMENT.value,
            "current_question": prompt,
            "messages": state.get("messages", []) + [HumanMessage(content=prompt)],
        }

    def _get_human_input(self, state: HumanAnnotationState) -> HumanAnnotationState:
        """Get human input for unknown review point types."""
        prompt = f"""
Please classify this review point:

TEXT: {state["review_point"].text}

Is this a question, claim, or argument? Please respond with one of: question, claim, argument
"""

        return {
            **state,
            "current_question": prompt,
            "messages": state.get("messages", []) + [HumanMessage(content=prompt)],
        }

    def _finalize_annotation(self, state: HumanAnnotationState) -> HumanAnnotationState:
        """Finalize the human annotation."""
        # Parse human response if available
        human_response = state.get("human_response", "")
        annotation_data = state.get("annotation_data", {})

        try:
            if human_response:
                # Try to parse JSON response
                response_data = json.loads(human_response)
                annotation_data.update(response_data)
        except json.JSONDecodeError:
            # If not JSON, store as text
            annotation_data["human_response"] = human_response

        return {
            **state,
            "annotation_step": HumanAnnotationStep.FINALIZE_ANNOTATION.value,
            "annotation_data": annotation_data,
            "is_complete": True,
        }

    def annotate_review_point(
        self,
        review_point: ReviewPoint,
        paper_content: str,
        annotator_id: str = "human_annotator",
    ) -> Dict[str, Any]:
        """
        Annotate a review point using human-in-the-loop workflow.

        Args:
            review_point: Review point to annotate
            paper_content: Content of the submitted paper
            annotator_id: ID of the human annotator

        Returns:
            Dictionary with annotation results
        """
        # Initialize state
        initial_state = {
            "review_point": review_point,
            "paper_content": paper_content,
            "annotator_id": annotator_id,
            "annotation_step": "initial",
            "current_question": "",
            "human_response": "",
            "annotation_data": {},
            "is_complete": False,
            "messages": [],
        }

        # Run the workflow
        final_state = self.app.invoke(initial_state)

        return {
            "review_point_id": review_point.id,
            "review_point_type": review_point.type,
            "annotator_id": annotator_id,
            "annotation_data": final_state.get("annotation_data", {}),
            "is_complete": final_state.get("is_complete", False),
            "messages": final_state.get("messages", []),
        }

    def batch_annotate(
        self,
        review_points: List[ReviewPoint],
        paper_content: str,
        annotator_id: str = "human_annotator",
    ) -> List[Dict[str, Any]]:
        """
        Annotate multiple review points in batch.

        Args:
            review_points: List of review points to annotate
            paper_content: Content of the submitted paper
            annotator_id: ID of the human annotator

        Returns:
            List of annotation results
        """
        annotations = []

        for review_point in review_points:
            annotation = self.annotate_review_point(
                review_point, paper_content, annotator_id
            )
            annotations.append(annotation)

        return annotations


def create_human_evaluator(config: PaperFaithfulConfig = None) -> HumanEvaluator:
    """Factory function to create a human evaluator."""
    return HumanEvaluator(config)
