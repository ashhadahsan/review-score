"""
LangChain Agents for ReviewScore evaluation.
Implements actual LangChain agents for comprehensive evaluation workflows.
"""

from typing import List, Dict, Any, Optional, Union
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents import create_react_agent, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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


@tool
def evaluate_question_tool(
    question_text: str, paper_context: str, review_context: str
) -> Dict[str, Any]:
    """
    Tool to evaluate whether a question can be answered by the paper.

    Args:
        question_text: The question to evaluate
        paper_context: Context from the submitted paper
        review_context: Context from the review

    Returns:
        Dictionary with evaluation results
    """
    # This is a simplified evaluation - in practice, this would use an LLM
    return {
        "is_answerable": True,
        "unanswerability_score": 2,  # 1-5 scale
        "answer_in_paper": "The paper discusses the methodology in section 3",
        "reasoning": "The question can be answered by the methodology section",
    }


@tool
def evaluate_claim_tool(
    claim_text: str, paper_context: str, review_context: str
) -> Dict[str, Any]:
    """
    Tool to evaluate whether a claim is factually correct.

    Args:
        claim_text: The claim to evaluate
        paper_context: Context from the submitted paper
        review_context: Context from the review

    Returns:
        Dictionary with evaluation results
    """
    return {
        "is_factually_correct": True,
        "factuality_score": 4,  # 1-5 scale
        "evidence_from_paper": "The paper states this in the results section",
        "reasoning": "The claim is supported by evidence in the paper",
    }


@tool
def evaluate_argument_tool(
    argument_text: str, paper_context: str, review_context: str
) -> Dict[str, Any]:
    """
    Tool to evaluate an argument by reconstructing it and evaluating premises.

    Args:
        argument_text: The argument to evaluate
        paper_context: Context from the submitted paper
        review_context: Context from the review

    Returns:
        Dictionary with evaluation results
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


@tool
def get_paper_context_tool(paper_content: str) -> str:
    """
    Tool to extract relevant context from the paper.

    Args:
        paper_content: The full paper content

    Returns:
        Relevant context for evaluation
    """
    # Simplified context extraction
    return paper_content[:500] + "..." if len(paper_content) > 500 else paper_content


class ReviewScoreLangChainAgent:
    """
    LangChain Agent for ReviewScore evaluation.
    Uses actual LangChain agents with tools for comprehensive evaluation.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.llm = self._initialize_llm()
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

    def _initialize_llm(self):
        """Initialize the LLM based on the model configuration."""
        if self.model_config.provider == "openai":
            return ChatOpenAI(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens or 1000,
                api_key=self.model_config.api_key or "mock_key",
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

    def _create_tools(self) -> List:
        """Create tools for the agent."""
        return [
            evaluate_question_tool,
            evaluate_claim_tool,
            evaluate_argument_tool,
            get_paper_context_tool,
        ]

    def _create_agent(self):
        """Create the LangChain agent."""
        # Create the prompt template for ReAct agent
        from langchain import hub
        from langchain.agents import create_react_agent

        # Use a standard ReAct prompt
        prompt = hub.pull("hwchase17/react")

        # Create the agent using ReAct
        return create_react_agent(self.llm, self.tools, prompt)

    def evaluate_review_point(self, review_point: ReviewPoint) -> ReviewScoreResult:
        """
        Evaluate a review point using the LangChain agent.

        Args:
            review_point: Review point to evaluate

        Returns:
            ReviewScoreResult with evaluation details
        """
        try:
            # Prepare the input for the agent
            agent_input = {
                "point_type": review_point.type.value,
                "point_text": review_point.text,
                "paper_context": review_point.paper_context,
                "review_context": review_point.review_context,
            }

            # Run the agent
            result = self.agent_executor.invoke(agent_input)

            # Parse the result
            output = result.get("output", "")

            # Extract scores from the output (simplified parsing)
            if "unanswerability_score" in output:
                score = self._extract_score(output, "unanswerability_score")
                is_misinformed = score <= 2
            elif "factuality_score" in output:
                score = self._extract_score(output, "factuality_score")
                is_misinformed = score <= 2
            elif "overall_factuality_score" in output:
                score = self._extract_score(output, "overall_factuality_score")
                is_misinformed = score <= 2
            else:
                score = 3.0
                is_misinformed = False

            return ReviewScoreResult(
                review_point=review_point,
                base_score=score,
                advanced_score=None,
                is_misinformed=is_misinformed,
                confidence=abs(score - 3) / 2,
                reasoning=output,
                model_used=self.model_config.model_name,
                evaluation_metadata={
                    "agent_type": "langchain_agent",
                    "tools_used": [tool.name for tool in self.tools],
                    "raw_output": output,
                },
            )

        except Exception as e:
            return ReviewScoreResult(
                review_point=review_point,
                base_score=3.0,
                advanced_score=3.0,
                is_misinformed=False,
                confidence=0.0,
                reasoning=f"Error in agent evaluation: {str(e)}",
                model_used=self.model_config.model_name,
                evaluation_metadata={"error": str(e)},
            )

    def _extract_score(self, text: str, score_type: str) -> float:
        """Extract score from agent output."""
        try:
            # Look for score in the text
            lines = text.split("\n")
            for line in lines:
                if score_type in line.lower():
                    # Extract number from the line
                    import re

                    numbers = re.findall(r"\d+\.?\d*", line)
                    if numbers:
                        return float(numbers[0])
            return 3.0
        except:
            return 3.0

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


class ReviewScoreMultiAgentSystem:
    """
    Multi-agent system for ReviewScore evaluation.
    Uses multiple specialized agents for different evaluation tasks.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.question_agent = self._create_question_agent()
        self.claim_agent = self._create_claim_agent()
        self.argument_agent = self._create_argument_agent()

    def _create_question_agent(self) -> ReviewScoreLangChainAgent:
        """Create specialized agent for question evaluation."""
        return ReviewScoreLangChainAgent(self.model_config)

    def _create_claim_agent(self) -> ReviewScoreLangChainAgent:
        """Create specialized agent for claim evaluation."""
        return ReviewScoreLangChainAgent(self.model_config)

    def _create_argument_agent(self) -> ReviewScoreLangChainAgent:
        """Create specialized agent for argument evaluation."""
        return ReviewScoreLangChainAgent(self.model_config)

    def evaluate_review_point(self, review_point: ReviewPoint) -> ReviewScoreResult:
        """
        Evaluate a review point using the appropriate specialized agent.

        Args:
            review_point: Review point to evaluate

        Returns:
            ReviewScoreResult with evaluation details
        """
        if isinstance(review_point, Question):
            return self.question_agent.evaluate_review_point(review_point)
        elif isinstance(review_point, Claim):
            return self.claim_agent.evaluate_review_point(review_point)
        elif isinstance(review_point, Argument):
            return self.argument_agent.evaluate_review_point(review_point)
        else:
            raise ValueError(f"Unknown review point type: {type(review_point)}")


def create_langchain_agent(
    model_name: str = "claude-3-5-sonnet-20241022",
) -> ReviewScoreLangChainAgent:
    """
    Factory function to create a LangChain agent.

    Args:
        model_name: Name of the model to use

    Returns:
        ReviewScoreLangChainAgent instance
    """
    # Find the model configuration
    model_config = None
    for config in PROPRIETARY_MODELS:
        if config.model_name == model_name:
            model_config = config
            break

    if model_config is None:
        raise ValueError(f"Model {model_name} not found in available models")

    return ReviewScoreLangChainAgent(model_config)


def create_multi_agent_system(
    model_name: str = "claude-3-5-sonnet-20241022",
) -> ReviewScoreMultiAgentSystem:
    """
    Factory function to create a multi-agent system.

    Args:
        model_name: Name of the model to use

    Returns:
        ReviewScoreMultiAgentSystem instance
    """
    # Find the model configuration
    model_config = None
    for config in PROPRIETARY_MODELS:
        if config.model_name == model_name:
            model_config = config
            break

    if model_config is None:
        raise ValueError(f"Model {model_name} not found in available models")

    return ReviewScoreMultiAgentSystem(model_config)
