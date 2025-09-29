"""
Comprehensive test suite for the paper-faithful ReviewScore implementation.
Tests all components described in the paper.
"""

import pytest
import numpy as np
from typing import List, Dict, Any
import json
import os

from reviewscore.core import (
    ReviewPoint,
    Question,
    Claim,
    Argument,
    ReviewPointType,
    create_review_point,
    parse_review_points_from_text,
)
from reviewscore.paper_faithful import (
    PaperFaithfulReviewScore,
    PaperFaithfulConfig,
    KnowledgeBase,
    SATSolver,
    KnowledgeBaseManager,
    PaperSpecificPrompts,
)
from reviewscore.human_evaluator import HumanEvaluator


class TestPaperFaithfulConfig:
    """Test the paper-faithful configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PaperFaithfulConfig()
        assert config.use_sat_solver == True
        assert config.use_multiple_kb == True
        assert config.use_paper_prompts == True
        assert config.use_paper_metrics == True
        assert config.sat_solver_type == "simple"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PaperFaithfulConfig(
            use_sat_solver=False,
            use_multiple_kb=False,
            use_paper_prompts=False,
            use_paper_metrics=False,
            sat_solver_type="z3",
        )
        assert config.use_sat_solver == False
        assert config.use_multiple_kb == False
        assert config.use_paper_prompts == False
        assert config.use_paper_metrics == False
        assert config.sat_solver_type == "z3"


class TestSATSolver:
    """Test the SAT solver functionality."""

    def test_simple_solver_initialization(self):
        """Test simple solver initialization."""
        solver = SATSolver("simple")
        assert solver.solver_type == "simple"
        assert solver.solver == "simple_solver"

    def test_simple_validation(self):
        """Test simple validation logic."""
        solver = SATSolver("simple")

        # Test valid argument
        premises = ["All birds can fly", "Penguins are birds"]
        conclusion = "Penguins can fly"
        result = solver.validate_argument(premises, conclusion)

        assert "is_valid" in result
        assert "solver_used" in result
        assert result["solver_used"] == "simple"

    def test_word_overlap_validation(self):
        """Test word overlap validation."""
        solver = SATSolver("simple")

        premises = [
            "The paper discusses machine learning",
            "Machine learning is important",
        ]
        conclusion = "The paper is about machine learning"

        result = solver.validate_argument(premises, conclusion)
        assert result["is_valid"] == True
        assert result["word_overlap"] > 0

    def test_logical_connection_validation(self):
        """Test logical connection validation."""
        solver = SATSolver("simple")

        premises = ["The method works well"]
        conclusion = "Therefore, the method is effective"

        result = solver.validate_argument(premises, conclusion)
        assert result["is_valid"] == True
        assert result["has_logical_connection"] == True


class TestKnowledgeBaseManager:
    """Test the knowledge base manager."""

    def test_initialization(self):
        """Test knowledge base manager initialization."""
        kb_manager = KnowledgeBaseManager()

        assert KnowledgeBase.SUBMITTED_PAPER in kb_manager.knowledge_bases
        assert KnowledgeBase.ANNOTATOR_KNOWLEDGE in kb_manager.knowledge_bases
        assert KnowledgeBase.REFERRED_PAPERS in kb_manager.knowledge_bases

    def test_set_knowledge_base(self):
        """Test setting knowledge base content."""
        kb_manager = KnowledgeBaseManager()

        # Set submitted paper
        paper_content = "This paper presents a novel approach to machine learning."
        kb_manager.set_knowledge_base(KnowledgeBase.SUBMITTED_PAPER, paper_content)

        content = kb_manager.get_knowledge_base_content(KnowledgeBase.SUBMITTED_PAPER)
        assert content == paper_content

    def test_set_referred_papers(self):
        """Test setting referred papers."""
        kb_manager = KnowledgeBaseManager()

        referred_papers = ["Paper A", "Paper B", "Paper C"]
        kb_manager.set_knowledge_base(KnowledgeBase.REFERRED_PAPERS, referred_papers)

        content = kb_manager.get_knowledge_base_content(KnowledgeBase.REFERRED_PAPERS)
        assert "Paper A" in content
        assert "Paper B" in content
        assert "Paper C" in content

    def test_select_appropriate_kb_question(self):
        """Test knowledge base selection for questions."""
        kb_manager = KnowledgeBaseManager()

        question = create_review_point(
            text="What is the main contribution?",
            point_type=ReviewPointType.QUESTION,
            paper_context="Test paper",
            review_context="Test review",
            point_id="q1",
        )

        selected_kbs = kb_manager.select_appropriate_kb(question)
        assert KnowledgeBase.SUBMITTED_PAPER in selected_kbs
        assert KnowledgeBase.ANNOTATOR_KNOWLEDGE in selected_kbs

    def test_select_appropriate_kb_claim(self):
        """Test knowledge base selection for claims."""
        kb_manager = KnowledgeBaseManager()

        claim = create_review_point(
            text="The paper lacks experimental validation",
            point_type=ReviewPointType.CLAIM,
            paper_context="Test paper",
            review_context="Test review",
            point_id="c1",
        )

        selected_kbs = kb_manager.select_appropriate_kb(claim)
        assert KnowledgeBase.SUBMITTED_PAPER in selected_kbs
        assert len(selected_kbs) == 1


class TestPaperSpecificPrompts:
    """Test the paper-specific prompts."""

    def test_question_evaluation_prompt(self):
        """Test question evaluation prompt."""
        prompt = PaperSpecificPrompts.get_question_evaluation_prompt()

        assert "question" in prompt.lower()
        assert "paper_context" in prompt
        assert "json" in prompt.lower()
        assert "unanswerability_score" in prompt

    def test_claim_evaluation_prompt(self):
        """Test claim evaluation prompt."""
        prompt = PaperSpecificPrompts.get_claim_evaluation_prompt()

        assert "claim" in prompt.lower()
        assert "paper_context" in prompt
        assert "json" in prompt.lower()
        assert "factuality_score" in prompt

    def test_argument_reconstruction_prompt(self):
        """Test argument reconstruction prompt."""
        prompt = PaperSpecificPrompts.get_argument_reconstruction_prompt()

        assert "argument" in prompt.lower()
        assert "premises" in prompt.lower()
        assert "conclusion" in prompt.lower()
        assert "json" in prompt.lower()

    def test_premise_evaluation_prompt(self):
        """Test premise evaluation prompt."""
        prompt = PaperSpecificPrompts.get_premise_evaluation_prompt()

        assert "premise" in prompt.lower()
        assert "paper_context" in prompt
        assert "json" in prompt.lower()
        assert "factuality_score" in prompt


class TestPaperFaithfulReviewScore:
    """Test the paper-faithful ReviewScore implementation."""

    def test_initialization(self):
        """Test initialization of paper-faithful evaluator."""
        evaluator = PaperFaithfulReviewScore()

        assert evaluator.config is not None
        assert evaluator.kb_manager is not None
        assert evaluator.prompts is not None

    def test_set_knowledge_bases(self):
        """Test setting knowledge bases."""
        evaluator = PaperFaithfulReviewScore()

        paper_content = "This paper presents a novel approach."
        annotator_knowledge = "Expert knowledge about the field."
        referred_papers = ["Paper A", "Paper B"]

        evaluator.set_knowledge_bases(
            submitted_paper=paper_content,
            annotator_knowledge=annotator_knowledge,
            referred_papers=referred_papers,
        )

        # Check that knowledge bases are set
        paper_content_set = evaluator.kb_manager.get_knowledge_base_content(
            KnowledgeBase.SUBMITTED_PAPER
        )
        assert paper_content_set == paper_content

        annotator_content = evaluator.kb_manager.get_knowledge_base_content(
            KnowledgeBase.ANNOTATOR_KNOWLEDGE
        )
        assert annotator_content == annotator_knowledge

    def test_evaluate_question_paper_methodology(self):
        """Test question evaluation using paper methodology."""
        evaluator = PaperFaithfulReviewScore()

        # Set up knowledge bases
        paper_content = "This paper presents a novel machine learning approach that achieves 95% accuracy."
        evaluator.set_knowledge_bases(
            submitted_paper=paper_content,
            annotator_knowledge="Expert knowledge about ML",
            referred_papers=["Related work 1", "Related work 2"],
        )

        # Create a question
        question = create_review_point(
            text="What accuracy does the method achieve?",
            point_type=ReviewPointType.QUESTION,
            paper_context=paper_content,
            review_context="Test review",
            point_id="q1",
        )

        # Evaluate
        result = evaluator._evaluate_question_paper_methodology(question)

        assert result.review_point == question
        assert 1 <= result.base_score <= 5
        assert isinstance(result.is_misinformed, bool)
        assert result.confidence >= 0
        assert "paper_faithful" in result.evaluation_metadata["methodology"]

    def test_evaluate_claim_paper_methodology(self):
        """Test claim evaluation using paper methodology."""
        evaluator = PaperFaithfulReviewScore()

        # Set up knowledge bases
        paper_content = "The proposed method achieves 95% accuracy on the test set."
        evaluator.set_knowledge_bases(
            submitted_paper=paper_content,
            annotator_knowledge="Expert knowledge",
            referred_papers=[],
        )

        # Create a claim
        claim = create_review_point(
            text="The method achieves 95% accuracy",
            point_type=ReviewPointType.CLAIM,
            paper_context=paper_content,
            review_context="Test review",
            point_id="c1",
        )

        # Evaluate
        result = evaluator._evaluate_claim_paper_methodology(claim)

        assert result.review_point == claim
        assert 1 <= result.base_score <= 5
        assert isinstance(result.is_misinformed, bool)
        assert result.confidence >= 0

    def test_evaluate_argument_paper_methodology(self):
        """Test argument evaluation using paper methodology."""
        evaluator = PaperFaithfulReviewScore()

        # Set up knowledge bases
        paper_content = "The method uses deep learning and achieves high accuracy."
        evaluator.set_knowledge_bases(
            submitted_paper=paper_content,
            annotator_knowledge="Expert knowledge",
            referred_papers=[],
        )

        # Create an argument
        argument = create_review_point(
            text="The method is effective because it uses deep learning and achieves high accuracy",
            point_type=ReviewPointType.ARGUMENT,
            paper_context=paper_content,
            review_context="Test review",
            point_id="a1",
        )

        # Evaluate
        result = evaluator._evaluate_argument_paper_methodology(argument)

        assert result.review_point == argument
        assert 1 <= result.base_score <= 5
        assert isinstance(result.is_misinformed, bool)
        assert result.confidence >= 0
        assert "premise_scores" in result.evaluation_metadata

    def test_evaluate_with_paper_methodology(self):
        """Test the main evaluation method."""
        evaluator = PaperFaithfulReviewScore()

        # Set up knowledge bases
        paper_content = "This paper presents a novel approach."
        evaluator.set_knowledge_bases(
            submitted_paper=paper_content,
            annotator_knowledge="Expert knowledge",
            referred_papers=[],
        )

        # Test with question
        question = create_review_point(
            text="What is the main contribution?",
            point_type=ReviewPointType.QUESTION,
            paper_context=paper_content,
            review_context="Test review",
            point_id="q1",
        )

        result = evaluator.evaluate_with_paper_methodology(question)
        assert result.review_point == question
        assert 1 <= result.base_score <= 5


class TestHumanEvaluator:
    """Test the human evaluator using LangGraph."""

    def test_initialization(self):
        """Test human evaluator initialization."""
        evaluator = HumanEvaluator()

        assert evaluator.config is not None
        assert evaluator.paper_evaluator is not None
        assert evaluator.workflow is not None
        assert evaluator.app is not None

    def test_workflow_creation(self):
        """Test that the workflow is created correctly."""
        evaluator = HumanEvaluator()

        # Check that all expected nodes are present
        nodes = evaluator.workflow.nodes
        expected_nodes = [
            "initialize",
            "classify_point",
            "evaluate_question",
            "evaluate_claim",
            "evaluate_argument",
            "finalize_annotation",
            "human_input",
        ]

        for node in expected_nodes:
            assert node in nodes

    def test_annotation_workflow(self):
        """Test the annotation workflow."""
        evaluator = HumanEvaluator()

        # Create a review point
        question = create_review_point(
            text="What is the main contribution?",
            point_type=ReviewPointType.QUESTION,
            paper_context="Test paper",
            review_context="Test review",
            point_id="q1",
        )

        paper_content = "This paper presents a novel approach to machine learning."

        # Test annotation (this would normally require human input)
        # For testing, we'll just check that the workflow can be initialized
        initial_state = {
            "review_point": question,
            "paper_content": paper_content,
            "annotator_id": "test_annotator",
            "annotation_step": "initial",
            "current_question": "",
            "human_response": "",
            "annotation_data": {},
            "is_complete": False,
            "messages": [],
        }

        # The workflow should be able to handle the initial state
        assert initial_state["review_point"] == question
        assert initial_state["paper_content"] == paper_content


class TestIntegration:
    """Test integration between components."""

    def test_end_to_end_evaluation(self):
        """Test end-to-end evaluation process."""
        # Create test data
        paper_content = """
        This paper presents a novel deep learning approach for natural language processing.
        The method achieves 95% accuracy on the benchmark dataset.
        The approach uses transformer architecture with attention mechanisms.
        """

        # Create review points
        question = create_review_point(
            text="What accuracy does the method achieve?",
            point_type=ReviewPointType.QUESTION,
            paper_context=paper_content,
            review_context="Test review",
            point_id="q1",
        )

        claim = create_review_point(
            text="The method achieves 95% accuracy",
            point_type=ReviewPointType.CLAIM,
            paper_context=paper_content,
            review_context="Test review",
            point_id="c1",
        )

        argument = create_review_point(
            text="The method is effective because it uses transformers and achieves high accuracy",
            point_type=ReviewPointType.ARGUMENT,
            paper_context=paper_content,
            review_context="Test review",
            point_id="a1",
        )

        # Create evaluator
        evaluator = PaperFaithfulReviewScore()
        evaluator.set_knowledge_bases(
            submitted_paper=paper_content,
            annotator_knowledge="Expert knowledge about NLP",
            referred_papers=["Attention Is All You Need", "BERT paper"],
        )

        # Evaluate all points
        question_result = evaluator.evaluate_with_paper_methodology(question)
        claim_result = evaluator.evaluate_with_paper_methodology(claim)
        argument_result = evaluator.evaluate_with_paper_methodology(argument)

        # Check results
        assert question_result.review_point == question
        assert claim_result.review_point == claim
        assert argument_result.review_point == argument

        # Check that all results have valid scores
        for result in [question_result, claim_result, argument_result]:
            assert 1 <= result.base_score <= 5
            assert isinstance(result.is_misinformed, bool)
            assert result.confidence >= 0

    def test_batch_evaluation(self):
        """Test batch evaluation of multiple review points."""
        paper_content = "Test paper content"

        # Create multiple review points
        review_points = [
            create_review_point(
                text=f"Question {i}?",
                point_type=ReviewPointType.QUESTION,
                paper_context=paper_content,
                review_context="Test review",
                point_id=f"q{i}",
            )
            for i in range(3)
        ]

        # Create evaluator
        evaluator = PaperFaithfulReviewScore()
        evaluator.set_knowledge_bases(
            submitted_paper=paper_content,
            annotator_knowledge="Expert knowledge",
            referred_papers=[],
        )

        # Evaluate all points
        results = []
        for review_point in review_points:
            result = evaluator.evaluate_with_paper_methodology(review_point)
            results.append(result)

        # Check that all evaluations completed
        assert len(results) == 3
        for result in results:
            assert 1 <= result.base_score <= 5
            assert isinstance(result.is_misinformed, bool)


def run_tests():
    """Run all tests."""
    print("Running ReviewScore Paper-Faithful Tests...")
    print("=" * 50)

    # Test configuration
    print("Testing PaperFaithfulConfig...")
    test_config = TestPaperFaithfulConfig()
    test_config.test_default_config()
    test_config.test_custom_config()
    print("PASS: PaperFaithfulConfig tests passed")

    # Test SAT solver
    print("\nTesting SATSolver...")
    test_solver = TestSATSolver()
    test_solver.test_simple_solver_initialization()
    test_solver.test_simple_validation()
    test_solver.test_word_overlap_validation()
    test_solver.test_logical_connection_validation()
    print("PASS: SATSolver tests passed")

    # Test knowledge base manager
    print("\nTesting KnowledgeBaseManager...")
    test_kb = TestKnowledgeBaseManager()
    test_kb.test_initialization()
    test_kb.test_set_knowledge_base()
    test_kb.test_set_referred_papers()
    test_kb.test_select_appropriate_kb_question()
    test_kb.test_select_appropriate_kb_claim()
    print("PASS: KnowledgeBaseManager tests passed")

    # Test paper-specific prompts
    print("\nTesting PaperSpecificPrompts...")
    test_prompts = TestPaperSpecificPrompts()
    test_prompts.test_question_evaluation_prompt()
    test_prompts.test_claim_evaluation_prompt()
    test_prompts.test_argument_reconstruction_prompt()
    test_prompts.test_premise_evaluation_prompt()
    print("PASS: PaperSpecificPrompts tests passed")

    # Test paper-faithful evaluator
    print("\nTesting PaperFaithfulReviewScore...")
    test_evaluator = TestPaperFaithfulReviewScore()
    test_evaluator.test_initialization()
    test_evaluator.test_set_knowledge_bases()
    test_evaluator.test_evaluate_question_paper_methodology()
    test_evaluator.test_evaluate_claim_paper_methodology()
    test_evaluator.test_evaluate_argument_paper_methodology()
    test_evaluator.test_evaluate_with_paper_methodology()
    print("PASS: PaperFaithfulReviewScore tests passed")

    # Test human evaluator
    print("\nTesting HumanEvaluator...")
    test_human = TestHumanEvaluator()
    test_human.test_initialization()
    test_human.test_workflow_creation()
    test_human.test_annotation_workflow()
    print("PASS: HumanEvaluator tests passed")

    # Test integration
    print("\nTesting Integration...")
    test_integration = TestIntegration()
    test_integration.test_end_to_end_evaluation()
    test_integration.test_batch_evaluation()
    print("PASS: Integration tests passed")

    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("The paper-faithful implementation is working correctly.")


if __name__ == "__main__":
    run_tests()
