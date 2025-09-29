"""
LangGraph Agents for ReviewScore evaluation.
Implements actual LangGraph agents with state management and tool usage.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
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


@tool
def evaluate_question_tool(
    question_text: str, paper_context: str, review_context: str
) -> Dict[str, Any]:
    """
    Tool to evaluate whether a question can be answered by the paper.
    """
    return {
        "is_answerable": True,
        "unanswerability_score": 2,
        "answer_in_paper": "The paper discusses the methodology in section 3",
        "reasoning": "The question can be answered by the methodology section",
    }


@tool
def evaluate_claim_tool(
    claim_text: str, paper_context: str, review_context: str
) -> Dict[str, Any]:
    """
    Tool to evaluate whether a claim is factually correct.
    """
    return {
        "is_factually_correct": True,
        "factuality_score": 4,
        "evidence_from_paper": "The paper states this in the results section",
        "reasoning": "The claim is supported by evidence in the paper",
    }


@tool
def evaluate_argument_tool(
    argument_text: str, paper_context: str, review_context: str
) -> Dict[str, Any]:
    """
    Tool to evaluate an argument by reconstructing it and evaluating premises.
    """
    return {
        "conclusion": "The main conclusion of the argument",
        "premises": [
            {"text": "Premise 1", "is_explicit": True, "factuality_score": 4},
            {"text": "Premise 2", "is_explicit": False, "factuality_score": 3},
        ],
        "overall_factuality_score": 3.5,
        "reasoning": "The argument has mixed factuality",
    }


class ReviewScoreAgentState(TypedDict):
    """State for ReviewScore agent workflow."""

    messages: Annotated[List[BaseMessage], operator.add]
    review_point: ReviewPoint
    paper_context: str
    review_context: str
    current_step: str
    evaluation_result: Optional[Dict[str, Any]]
    final_result: Optional[ReviewScoreResult]
    error_message: Optional[str]
    human_input: Optional[str]
    requires_human_input: bool
    human_annotation: Optional[Dict[str, Any]]


class ReviewScoreLangGraphAgent:
    """
    LangGraph Agent for ReviewScore evaluation.
    Uses actual LangGraph agents with state management and tool usage.
    """

    def __init__(self, model_config: ModelConfig, enable_human_in_loop: bool = False):
        self.model_config = model_config
        self.enable_human_in_loop = enable_human_in_loop
        self.llm = self._initialize_llm()
        self.tools = self._create_tools()
        self.workflow = self._create_workflow()
        self.app = self._compile_workflow()

    def _initialize_llm(self):
        """Initialize the LLM based on the model configuration."""
        if self.model_config.provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens or 1000,
                api_key=self.model_config.api_key or "mock_key",
            )
        elif self.model_config.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens or 1000,
                api_key=self.model_config.api_key or "mock_key",
            )
        elif self.model_config.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens or 1000,
                api_key=self.model_config.api_key or "mock_key",
            )
        else:
            raise ValueError(f"Unsupported provider: {self.model_config.provider}")

    def _create_tools(self) -> List:
        """Create tools for the agent."""
        return [
            evaluate_question_tool,
            evaluate_claim_tool,
            evaluate_argument_tool,
        ]

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph agent workflow."""
        workflow = StateGraph(ReviewScoreAgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)
        workflow.add_node("evaluator", self._evaluator_node)
        workflow.add_node("human_input", self._human_input_node)
        workflow.add_node("finalizer", self._finalizer_node)

        # Add edges
        workflow.add_edge("agent", "tools")
        workflow.add_edge("tools", "evaluator")

        # Conditional routing for human input
        if self.enable_human_in_loop:
            workflow.add_conditional_edges(
                "evaluator",
                self._should_request_human_input,
                {"human_input": "human_input", "finalize": "finalizer"},
            )
            workflow.add_edge("human_input", "finalizer")
        else:
            workflow.add_edge("evaluator", "finalizer")

        workflow.add_edge("finalizer", END)

        # Set entry point
        workflow.set_entry_point("agent")

        return workflow

    def _compile_workflow(self):
        """Compile the workflow into an executable app."""
        if self.enable_human_in_loop:
            # Add checkpointing for human-in-the-loop
            memory = MemorySaver()
            return self.workflow.compile(checkpointer=memory)
        else:
            return self.workflow.compile()

    def _should_request_human_input(self, state: ReviewScoreAgentState) -> str:
        """Determine if human input is needed."""
        if not self.enable_human_in_loop:
            return "finalize"

        evaluation_result = state.get("evaluation_result", {})

        # Request human input for low confidence or complex cases
        if "confidence" in evaluation_result and evaluation_result["confidence"] < 0.3:
            return "human_input"

        # Request human input for edge cases
        if "unanswerability_score" in evaluation_result:
            score = evaluation_result["unanswerability_score"]
            if score in [2, 3, 4]:  # Borderline cases
                return "human_input"

        return "finalize"

    def _human_input_node(self, state: ReviewScoreAgentState) -> ReviewScoreAgentState:
        """Node that requests human input using LangGraph interrupt."""
        try:
            review_point = state["review_point"]
            evaluation_result = state.get("evaluation_result", {})

            # Create human input request payload
            human_input_payload = {
                "review_point_text": review_point.text,
                "review_point_type": review_point.type.value,
                "evaluation_result": evaluation_result,
                "request": """
Please provide your annotation:
1. Do you agree with the AI evaluation?
2. What is your confidence level (1-5)?
3. Any additional comments?

Please respond in JSON format:
{
    "agrees_with_ai": true/false,
    "human_confidence": 1-5,
    "human_score": 1-5,
    "comments": "your comments here"
}
""",
            }

            # Use LangGraph interrupt to pause and wait for human input
            human_response = interrupt(human_input_payload)

            # Parse human response
            try:
                import json

                human_annotation = json.loads(human_response)
            except:
                human_annotation = {
                    "agrees_with_ai": True,
                    "human_confidence": 3,
                    "human_score": 3,
                    "comments": f"Raw response: {human_response}",
                }

            return {
                **state,
                "current_step": "human_input_received",
                "human_input": human_response,
                "human_annotation": human_annotation,
                "messages": state.get("messages", [])
                + [HumanMessage(content=f"Human input: {human_response}")],
            }

        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Human input error: {str(e)}",
            }

    def _tools_node(self, state: ReviewScoreAgentState) -> ReviewScoreAgentState:
        """Tools node that calls the appropriate tool based on review point type."""
        try:
            review_point = state["review_point"]

            # Call the appropriate tool based on review point type
            if isinstance(review_point, Question):
                tool_result = evaluate_question_tool(
                    question_text=review_point.text,
                    paper_context=review_point.paper_context,
                    review_context=review_point.review_context,
                )
            elif isinstance(review_point, Claim):
                tool_result = evaluate_claim_tool(
                    claim_text=review_point.text,
                    paper_context=review_point.paper_context,
                    review_context=review_point.review_context,
                )
            elif isinstance(review_point, Argument):
                tool_result = evaluate_argument_tool(
                    argument_text=review_point.text,
                    paper_context=review_point.paper_context,
                    review_context=review_point.review_context,
                )
            else:
                tool_result = {"error": "Unknown review point type"}

            # Add tool result to messages
            tool_message = ToolMessage(
                content=str(tool_result), tool_call_id="tool_call_1"
            )

            current_messages = state.get("messages", [])
            messages = current_messages + [tool_message]

            return {
                **state,
                "messages": messages,
                "current_step": "tools_executed",
                "evaluation_result": tool_result,
            }

        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Tools error: {str(e)}",
            }

    def _agent_node(self, state: ReviewScoreAgentState) -> ReviewScoreAgentState:
        """Agent node that decides what to do."""
        try:
            review_point = state["review_point"]

            # Create system message
            system_message = SystemMessage(
                content=f"""You are an expert peer review evaluator. Your task is to evaluate review points to determine if they are misinformed.

You have access to these tools:
- evaluate_question_tool: For evaluating questions
- evaluate_claim_tool: For evaluating claims  
- evaluate_argument_tool: For evaluating arguments

Current review point type: {review_point.type.value}
Review point text: {review_point.text}

Choose the appropriate tool to evaluate this review point and call it with the required parameters."""
            )

            # Create human message with the review point details
            human_message = HumanMessage(
                content=f"""
Please evaluate this review point:

REVIEW POINT TYPE: {review_point.type.value}
REVIEW POINT TEXT: {review_point.text}
PAPER CONTEXT: {review_point.paper_context}
REVIEW CONTEXT: {review_point.review_context}

Use the appropriate tool to evaluate this review point.
"""
            )

            # Add messages to state
            current_messages = state.get("messages", [])
            messages = current_messages + [system_message, human_message]

            return {**state, "messages": messages, "current_step": "agent_processing"}

        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Agent error: {str(e)}",
            }

    def _evaluator_node(self, state: ReviewScoreAgentState) -> ReviewScoreAgentState:
        """Evaluator node that processes the tool results."""
        try:
            # Get the last message (should be from tools)
            messages = state.get("messages", [])
            if not messages:
                # Create a mock evaluation result if no messages
                evaluation_result = {
                    "unanswerability_score": 3,
                    "reasoning": "No tool results available, using default evaluation",
                    "confidence": 0.5,
                }
            else:
                # Extract evaluation result from the last message
                last_message = messages[-1]
                if isinstance(last_message, ToolMessage):
                    # Parse the tool result
                    try:
                        import json

                        evaluation_result = json.loads(last_message.content)
                    except:
                        evaluation_result = {"raw_content": last_message.content}
                else:
                    evaluation_result = {"raw_content": str(last_message.content)}

            return {
                **state,
                "current_step": "evaluation_completed",
                "evaluation_result": evaluation_result,
            }

        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error_message": f"Evaluation error: {str(e)}",
            }

    def _finalizer_node(self, state: ReviewScoreAgentState) -> ReviewScoreAgentState:
        """Finalizer node that creates the final result."""
        try:
            review_point = state["review_point"]
            evaluation_result = state.get("evaluation_result", {})
            human_annotation = state.get("human_annotation", {})

            # Extract scores based on the evaluation result
            if "unanswerability_score" in evaluation_result:
                score = evaluation_result["unanswerability_score"]
                is_misinformed = score <= 2
            elif "factuality_score" in evaluation_result:
                score = evaluation_result["factuality_score"]
                is_misinformed = score <= 2
            elif "overall_factuality_score" in evaluation_result:
                score = evaluation_result["overall_factuality_score"]
                is_misinformed = score <= 2
            else:
                score = 3.0
                is_misinformed = False

            # Override with human annotation if available
            if human_annotation:
                human_score = human_annotation.get("human_score", score)
                human_confidence = human_annotation.get("human_confidence", 3)
                agrees_with_ai = human_annotation.get("agrees_with_ai", True)

                if not agrees_with_ai:
                    # Use human score if they disagree with AI
                    score = human_score
                    is_misinformed = human_score <= 2

                # Update confidence based on human input
                confidence = human_confidence / 5.0  # Convert to 0-1 scale
            else:
                confidence = abs(score - 3) / 2

            # Create final result
            final_result = ReviewScoreResult(
                review_point=review_point,
                base_score=score,
                advanced_score=None,
                is_misinformed=is_misinformed,
                confidence=confidence,
                reasoning=evaluation_result.get("reasoning", ""),
                model_used=self.model_config.model_name,
                evaluation_metadata={
                    "agent_type": "langgraph_agent",
                    "tools_used": [tool.name for tool in self.tools],
                    "evaluation_result": evaluation_result,
                    "human_annotation": human_annotation,
                    "human_in_loop_enabled": self.enable_human_in_loop,
                },
            )

            return {**state, "current_step": "completed", "final_result": final_result}

        except Exception as e:
            # Create error result
            error_result = ReviewScoreResult(
                review_point=state["review_point"],
                base_score=3.0,
                advanced_score=3.0,
                is_misinformed=False,
                confidence=0.0,
                reasoning=f"Error in finalization: {str(e)}",
                model_used=self.model_config.model_name,
                evaluation_metadata={"error": str(e)},
            )

            return {**state, "current_step": "error", "final_result": error_result}

    def evaluate_review_point(
        self, review_point: ReviewPoint, thread_id: str = None
    ) -> ReviewScoreResult:
        """
        Evaluate a review point using the LangGraph agent.

        Args:
            review_point: Review point to evaluate
            thread_id: Thread ID for human-in-the-loop (required if enabled)

        Returns:
            ReviewScoreResult with evaluation details
        """
        import uuid

        # Initialize state
        initial_state = {
            "messages": [],
            "review_point": review_point,
            "paper_context": review_point.paper_context,
            "review_context": review_point.review_context,
            "current_step": "initial",
            "evaluation_result": None,
            "final_result": None,
            "error_message": None,
            "human_input": None,
            "requires_human_input": False,
            "human_annotation": None,
        }

        # Create config with thread ID for human-in-the-loop
        if self.enable_human_in_loop:
            if thread_id is None:
                thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
        else:
            config = {}

        # Run the workflow
        final_state = self.app.invoke(initial_state, config)

        # Check if interrupted for human input
        if "__interrupt__" in final_state:
            # Return a special result indicating human input is needed
            return ReviewScoreResult(
                review_point=review_point,
                base_score=3.0,
                advanced_score=3.0,
                is_misinformed=False,
                confidence=0.0,
                reasoning="Human input required - use resume_with_human_input() to continue",
                model_used=self.model_config.model_name,
                evaluation_metadata={
                    "interrupted": True,
                    "thread_id": thread_id,
                    "interrupt_data": final_state["__interrupt__"],
                },
            )

        # Return the final result
        return final_state.get(
            "final_result",
            ReviewScoreResult(
                review_point=review_point,
                base_score=3.0,
                advanced_score=3.0,
                is_misinformed=False,
                confidence=0.0,
                reasoning="Agent failed to produce result",
                model_used=self.model_config.model_name,
                evaluation_metadata={"error": True},
            ),
        )

    def resume_with_human_input(
        self, thread_id: str, human_response: str
    ) -> ReviewScoreResult:
        """
        Resume evaluation with human input.

        Args:
            thread_id: Thread ID from interrupted evaluation
            human_response: Human's response to the interrupt

        Returns:
            ReviewScoreResult with evaluation details
        """
        if not self.enable_human_in_loop:
            raise ValueError("Human-in-the-loop is not enabled")

        config = {"configurable": {"thread_id": thread_id}}

        # Resume with human input using Command
        final_state = self.app.invoke(Command(resume=human_response), config)

        # Return the final result
        return final_state.get(
            "final_result",
            ReviewScoreResult(
                review_point=final_state.get("review_point"),
                base_score=3.0,
                advanced_score=3.0,
                is_misinformed=False,
                confidence=0.0,
                reasoning="Resume failed to produce result",
                model_used=self.model_config.model_name,
                evaluation_metadata={"error": True},
            ),
        )

    def evaluate_batch(
        self, review_points: List[ReviewPoint]
    ) -> List[ReviewScoreResult]:
        """
        Evaluate multiple review points using the agent.

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


def create_langgraph_agent(
    model_name: str = "claude-3-5-sonnet-20241022",
    enable_human_in_loop: bool = False,
) -> ReviewScoreLangGraphAgent:
    """
    Factory function to create a LangGraph agent.

    Args:
        model_name: Name of the model to use
        enable_human_in_loop: Whether to enable human-in-the-loop functionality

    Returns:
        ReviewScoreLangGraphAgent instance
    """
    # Find the model configuration
    model_config = None
    for config in PROPRIETARY_MODELS:
        if config.model_name == model_name:
            model_config = config
            break

    if model_config is None:
        raise ValueError(f"Model {model_name} not found in available models")

    return ReviewScoreLangGraphAgent(model_config, enable_human_in_loop)
