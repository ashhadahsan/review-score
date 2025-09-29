"""
BASE REVIEW SCORE implementation as defined in the paper.
Evaluates questions and claims directly without premise-level analysis.
"""

from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re

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


class BaseReviewScoreEvaluator:
    """
    Implements BASE REVIEW SCORE evaluation as defined in Definition 3 of the paper.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.llm = self._initialize_llm()
        self.question_prompt = self._create_question_prompt()
        self.claim_prompt = self._create_claim_prompt()
        self.output_parser = JsonOutputParser()

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
                max_tokens=self.model_config.max_tokens,
                api_key=self.model_config.api_key,
            )
        elif self.model_config.provider == "google":
            return ChatGoogleGenerativeAI(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                api_key=self.model_config.api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.model_config.provider}")

    def _create_question_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for evaluating questions."""
        template = """
You are an expert reviewer evaluating whether a question in a peer review can be answered by the submitted paper.

TASK: Determine if the question can be answered by the paper content.

PAPER CONTENT:
{paper_context}

REVIEW CONTEXT:
{review_context}

QUESTION TO EVALUATE:
{question_text}

EVALUATION CRITERIA:
- A question is MISINFORMED if it can be answered by the paper
- A question is NOT MISINFORMED if it cannot be answered by the paper
- Look for direct answers, related information, or implicit answers in the paper

Please provide your evaluation in the following JSON format:
{{
    "is_answerable": true/false,
    "answer_in_paper": "specific text from paper that answers the question (if answerable)",
    "unanswerability_score": 1-5 (1=definitely answerable, 5=definitely not answerable),
    "reasoning": "explanation of your evaluation"
}}

Focus on factual accuracy and whether the paper contains information that directly or indirectly answers the question.
"""
        return ChatPromptTemplate.from_template(template)

    def _create_claim_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for evaluating claims."""
        template = """
You are an expert reviewer evaluating whether a claim in a peer review is factually correct regarding the submitted paper.

TASK: Determine if the claim is factually correct based on the paper content.

PAPER CONTENT:
{paper_context}

REVIEW CONTEXT:
{review_context}

CLAIM TO EVALUATE:
{claim_text}

EVALUATION CRITERIA:
- A claim is MISINFORMED if it is factually incorrect regarding the paper
- A claim is NOT MISINFORMED if it is factually correct regarding the paper
- Check if the claim accurately represents what the paper states or does

Please provide your evaluation in the following JSON format:
{{
    "is_factually_correct": true/false,
    "factuality_score": 1-5 (1=definitely incorrect, 5=definitely correct),
    "evidence_from_paper": "specific text from paper that supports or contradicts the claim",
    "reasoning": "explanation of your evaluation"
}}

Focus on factual accuracy and whether the claim correctly represents the paper's content.
"""
        return ChatPromptTemplate.from_template(template)

    def evaluate_question(self, question: Question) -> ReviewScoreResult:
        """
        Evaluate a question using BASE REVIEW SCORE.
        Returns unanswerability score (1-5) where 1=answerable (misinformed), 5=not answerable.
        """
        try:
            # Create the evaluation chain
            chain = self.question_prompt | self.llm | self.output_parser

            # Run evaluation
            result = chain.invoke(
                {
                    "paper_context": question.paper_context,
                    "review_context": question.review_context,
                    "question_text": question.text,
                }
            )

            # Extract scores and reasoning
            unanswerability_score = result.get("unanswerability_score", 3)
            is_answerable = result.get("is_answerable", False)
            reasoning = result.get("reasoning", "")

            # Convert to BASE REVIEW SCORE (1-5 scale)
            # 1-2 = Misinformed, 3-5 = Not misinformed
            base_score = unanswerability_score
            is_misinformed = base_score <= 2

            return ReviewScoreResult(
                review_point=question,
                base_score=base_score,
                is_misinformed=is_misinformed,
                confidence=abs(base_score - 3)
                / 2,  # Confidence based on distance from neutral
                reasoning=reasoning,
                model_used=self.model_config.model_name,
                evaluation_metadata={
                    "is_answerable": is_answerable,
                    "answer_in_paper": result.get("answer_in_paper", ""),
                    "unanswerability_score": unanswerability_score,
                },
            )

        except Exception as e:
            return ReviewScoreResult(
                review_point=question,
                base_score=3.0,  # Neutral score on error
                is_misinformed=False,
                confidence=0.0,
                reasoning=f"Error in evaluation: {str(e)}",
                model_used=self.model_config.model_name,
                evaluation_metadata={"error": str(e)},
            )

    def evaluate_claim(self, claim: Claim) -> ReviewScoreResult:
        """
        Evaluate a claim using BASE REVIEW SCORE.
        Returns factuality score (1-5) where 1=incorrect (misinformed), 5=correct.
        """
        try:
            # Create the evaluation chain
            chain = self.claim_prompt | self.llm | self.output_parser

            # Run evaluation
            result = chain.invoke(
                {
                    "paper_context": claim.paper_context,
                    "review_context": claim.review_context,
                    "claim_text": claim.text,
                }
            )

            # Extract scores and reasoning
            factuality_score = result.get("factuality_score", 3)
            is_factually_correct = result.get("is_factually_correct", True)
            reasoning = result.get("reasoning", "")

            # Convert to BASE REVIEW SCORE (1-5 scale)
            # 1-2 = Misinformed, 3-5 = Not misinformed
            base_score = factuality_score
            is_misinformed = base_score <= 2

            return ReviewScoreResult(
                review_point=claim,
                base_score=base_score,
                is_misinformed=is_misinformed,
                confidence=abs(base_score - 3)
                / 2,  # Confidence based on distance from neutral
                reasoning=reasoning,
                model_used=self.model_config.model_name,
                evaluation_metadata={
                    "is_factually_correct": is_factually_correct,
                    "evidence_from_paper": result.get("evidence_from_paper", ""),
                    "factuality_score": factuality_score,
                },
            )

        except Exception as e:
            return ReviewScoreResult(
                review_point=claim,
                base_score=3.0,  # Neutral score on error
                is_misinformed=False,
                confidence=0.0,
                reasoning=f"Error in evaluation: {str(e)}",
                model_used=self.model_config.model_name,
                evaluation_metadata={"error": str(e)},
            )

    def evaluate_review_point(self, review_point: ReviewPoint) -> ReviewScoreResult:
        """
        Evaluate any review point using BASE REVIEW SCORE.
        Routes to appropriate evaluation method based on type.
        """
        if isinstance(review_point, Question):
            return self.evaluate_question(review_point)
        elif isinstance(review_point, Claim):
            return self.evaluate_claim(review_point)
        elif isinstance(review_point, Argument):
            # For arguments, we use a simplified approach in BASE REVIEW SCORE
            # This would be improved with ADVANCED REVIEW SCORE
            return self._evaluate_argument_base(review_point)
        else:
            raise ValueError(f"Unknown review point type: {type(review_point)}")

    def _evaluate_argument_base(self, argument: Argument) -> ReviewScoreResult:
        """
        Simplified BASE REVIEW SCORE evaluation for arguments.
        This treats the argument as a single claim for evaluation.
        """
        try:
            # Create a temporary claim from the argument text
            temp_claim = Claim(
                id=argument.id,
                text=argument.text,
                paper_context=argument.paper_context,
                review_context=argument.review_context,
            )

            # Evaluate as a claim
            result = self.evaluate_claim(temp_claim)

            # Update the result to reflect it's an argument
            result.review_point = argument
            result.evaluation_metadata["evaluation_type"] = "base_argument"

            return result

        except Exception as e:
            return ReviewScoreResult(
                review_point=argument,
                base_score=3.0,
                is_misinformed=False,
                confidence=0.0,
                reasoning=f"Error in argument evaluation: {str(e)}",
                model_used=self.model_config.model_name,
                evaluation_metadata={"error": str(e)},
            )


def create_base_evaluator(
    model_name: str = "claude-3-5-sonnet-20241022",
) -> BaseReviewScoreEvaluator:
    """
    Factory function to create a BASE REVIEW SCORE evaluator.
    """
    # Find the model configuration
    model_config = None
    for config in PROPRIETARY_MODELS:
        if config.model_name == model_name:
            model_config = config
            break

    if model_config is None:
        raise ValueError(f"Model {model_name} not found in available models")

    return BaseReviewScoreEvaluator(model_config)
