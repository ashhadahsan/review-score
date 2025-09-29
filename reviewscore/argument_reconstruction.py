"""
Automatic Argument Reconstruction Engine as described in the paper.
Reconstructs arguments into premise-conclusion structures for ADVANCED REVIEW SCORE.
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re

from .core import (
    Argument, Premise, ModelConfig, PROPRIETARY_MODELS
)


class ArgumentReconstructionEngine:
    """
    Automatic argument reconstruction engine that extracts explicit and implicit premises
    from arguments as described in Section 3.2 of the paper.
    """
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.llm = self._initialize_llm()
        self.reconstruction_prompt = self._create_reconstruction_prompt()
        self.validation_prompt = self._create_validation_prompt()
        self.output_parser = JsonOutputParser()
    
    def _initialize_llm(self):
        """Initialize the LLM based on the model configuration."""
        if self.model_config.provider == "openai":
            return ChatOpenAI(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                api_key=self.model_config.api_key
            )
        elif self.model_config.provider == "anthropic":
            return ChatAnthropic(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                api_key=self.model_config.api_key
            )
        elif self.model_config.provider == "google":
            return ChatGoogleGenerativeAI(
                model=self.model_config.model_name,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                api_key=self.model_config.api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {self.model_config.provider}")
    
    def _create_reconstruction_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for argument reconstruction."""
        template = """
You are an expert in logic and critical thinking. Your task is to reconstruct an argument into its explicit and implicit premises and conclusion.

TASK: Analyze the argument and extract all premises (both explicit and implicit) and the conclusion.

PAPER CONTEXT:
{paper_context}

REVIEW CONTEXT:
{review_context}

ARGUMENT TO RECONSTRUCT:
{argument_text}

INSTRUCTIONS:
1. Identify the main conclusion of the argument
2. Extract all explicit premises (directly stated)
3. Identify all implicit premises (assumed but not stated)
4. Ensure the argument is logically valid (premises should support the conclusion)
5. Ensure the reconstruction is faithful to the original argument

Please provide your reconstruction in the following JSON format:
{{
    "conclusion": "the main conclusion of the argument",
    "explicit_premises": [
        {{
            "text": "premise text",
            "is_explicit": true,
            "reasoning": "why this premise supports the conclusion"
        }}
    ],
    "implicit_premises": [
        {{
            "text": "premise text",
            "is_explicit": false,
            "reasoning": "why this premise is assumed and supports the conclusion"
        }}
    ],
    "logical_structure": "description of how premises lead to conclusion",
    "validity_assessment": "assessment of logical validity"
}}

Focus on:
- Completeness: Don't miss any important premises
- Faithfulness: Accurately represent the original argument
- Clarity: Make implicit assumptions explicit
- Logical coherence: Ensure premises support the conclusion
"""
        return ChatPromptTemplate.from_template(template)
    
    def _create_validation_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for validating reconstructed arguments."""
        template = """
You are an expert in logic and critical thinking. Your task is to validate a reconstructed argument.

TASK: Check if the reconstructed argument is both valid and faithful to the original.

ORIGINAL ARGUMENT:
{original_argument}

RECONSTRUCTED ARGUMENT:
{reconstructed_argument}

VALIDATION CRITERIA:
1. VALIDITY: Do the premises logically support the conclusion?
2. FAITHFULNESS: Does the reconstruction accurately represent the original argument?
3. COMPLETENESS: Are all important premises included?
4. CLARITY: Are implicit premises made explicit?

Please provide your validation in the following JSON format:
{{
    "is_valid": true/false,
    "is_faithful": true/false,
    "validity_score": 1-5 (1=invalid, 5=highly valid),
    "faithfulness_score": 1-5 (1=unfaithful, 5=highly faithful),
    "missing_premises": ["any important premises that were missed"],
    "invalid_premises": ["any premises that don't support the conclusion"],
    "reasoning": "detailed explanation of your validation"
}}
"""
        return ChatPromptTemplate.from_template(template)
    
    def reconstruct_argument(self, argument: Argument) -> Argument:
        """
        Reconstruct an argument into its premise-conclusion structure.
        Returns the argument with populated premises and conclusion.
        """
        try:
            # Create the reconstruction chain
            chain = (
                self.reconstruction_prompt
                | self.llm
                | self.output_parser
            )
            
            # Run reconstruction
            result = chain.invoke({
                "paper_context": argument.paper_context,
                "review_context": argument.review_context,
                "argument_text": argument.text
            })
            
            # Extract premises and conclusion
            conclusion = result.get("conclusion", "")
            explicit_premises = result.get("explicit_premises", [])
            implicit_premises = result.get("implicit_premises", [])
            
            # Create Premise objects
            premises = []
            for premise_data in explicit_premises:
                premise = Premise(
                    text=premise_data.get("text", ""),
                    is_explicit=True
                )
                premises.append(premise)
            
            for premise_data in implicit_premises:
                premise = Premise(
                    text=premise_data.get("text", ""),
                    is_explicit=False
                )
                premises.append(premise)
            
            # Update the argument
            argument.conclusion = conclusion
            argument.premises = premises
            
            # Validate the reconstruction
            validation_result = self._validate_reconstruction(argument, result)
            argument.is_valid = validation_result.get("is_valid", False)
            argument.is_faithful = validation_result.get("is_faithful", False)
            
            return argument
            
        except Exception as e:
            # On error, return the original argument with error metadata
            argument.conclusion = ""
            argument.premises = []
            argument.is_valid = False
            argument.is_faithful = False
            return argument
    
    def _validate_reconstruction(self, argument: Argument, reconstruction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the reconstructed argument for validity and faithfulness.
        """
        try:
            # Create the validation chain
            chain = (
                self.validation_prompt
                | self.llm
                | self.output_parser
            )
            
            # Prepare reconstructed argument text
            reconstructed_text = f"Conclusion: {argument.conclusion}\n"
            reconstructed_text += "Premises:\n"
            for premise in argument.premises:
                reconstructed_text += f"- {premise.text} (explicit: {premise.is_explicit})\n"
            
            # Run validation
            result = chain.invoke({
                "original_argument": argument.text,
                "reconstructed_argument": reconstructed_text
            })
            
            return result
            
        except Exception as e:
            return {
                "is_valid": False,
                "is_faithful": False,
                "validity_score": 1,
                "faithfulness_score": 1,
                "reasoning": f"Validation error: {str(e)}"
            }
    
    def extract_premises_with_factuality(self, argument: Argument) -> List[Premise]:
        """
        Extract premises and evaluate their factuality.
        This is used in ADVANCED REVIEW SCORE evaluation.
        """
        # First reconstruct the argument
        reconstructed_argument = self.reconstruct_argument(argument)
        
        # Then evaluate factuality of each premise
        for premise in reconstructed_argument.premises:
            premise.factuality_score = self._evaluate_premise_factuality(premise, argument)
            premise.is_factually_correct = premise.factuality_score > 2.5 if premise.factuality_score else None
        
        return reconstructed_argument.premises
    
    def _evaluate_premise_factuality(self, premise: Premise, argument: Argument) -> float:
        """
        Evaluate the factuality of a single premise.
        Returns a score from 1-5 where 1=incorrect, 5=correct.
        """
        try:
            # Create a simple factuality evaluation prompt
            factuality_prompt = ChatPromptTemplate.from_template("""
You are an expert reviewer evaluating the factuality of a premise in a peer review.

PAPER CONTENT:
{paper_context}

PREMISE TO EVALUATE:
{premise_text}

TASK: Determine if this premise is factually correct regarding the paper.

Please provide your evaluation in the following JSON format:
{{
    "factuality_score": 1-5 (1=definitely incorrect, 5=definitely correct),
    "is_factually_correct": true/false,
    "evidence": "evidence from paper that supports or contradicts the premise",
    "reasoning": "explanation of your evaluation"
}}
""")
            
            # Create the evaluation chain
            chain = (
                factuality_prompt
                | self.llm
                | self.output_parser
            )
            
            # Run evaluation
            result = chain.invoke({
                "paper_context": argument.paper_context,
                "premise_text": premise.text
            })
            
            return result.get("factuality_score", 3.0)
            
        except Exception as e:
            return 3.0  # Neutral score on error


def create_reconstruction_engine(model_name: str = "claude-3-5-sonnet-20241022") -> ArgumentReconstructionEngine:
    """
    Factory function to create an argument reconstruction engine.
    """
    # Find the model configuration
    model_config = None
    for config in PROPRIETARY_MODELS:
        if config.model_name == model_name:
            model_config = config
            break
    
    if model_config is None:
        raise ValueError(f"Model {model_name} not found in available models")
    
    return ArgumentReconstructionEngine(model_config)
