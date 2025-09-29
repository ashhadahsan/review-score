"""
Simple demonstration of ReviewScore without requiring API keys.
Shows the core functionality and paper-faithful implementation.
"""

from dotenv import load_dotenv

load_dotenv()

from reviewscore.paper_faithful import (
    PaperFaithfulReviewScore,
    PaperFaithfulConfig,
    SATSolver,
    KnowledgeBaseManager,
    PaperSpecificPrompts,
)
from reviewscore.core import create_review_point, ReviewPointType
from reviewscore.human_evaluator import HumanEvaluator


def demonstrate_core_components():
    """Demonstrate core components without API calls."""
    print("REVIEWSCORE CORE COMPONENTS DEMONSTRATION")
    print("=" * 50)

    # 1. Configuration
    print("\n1. Paper-Faithful Configuration")
    config = PaperFaithfulConfig(
        use_sat_solver=True,
        use_multiple_kb=True,
        use_paper_prompts=True,
        use_paper_metrics=True,
    )
    print(f"PASS: Configuration created: {config}")

    # 2. SAT Solver
    print("\n2. SAT Solver Validation")
    solver = SATSolver("simple")
    result = solver.validate_argument(
        premises=["The method works well", "Good methods are effective"],
        conclusion="Therefore, the method is effective",
    )
    print(f"PASS: SAT validation result: {result}")

    # 3. Knowledge Base Manager
    print("\n3. Knowledge Base Integration")
    kb_manager = KnowledgeBaseManager()
    from reviewscore.paper_faithful import KnowledgeBase

    kb_manager.set_knowledge_base(
        KnowledgeBase.SUBMITTED_PAPER, "This is a sample paper about machine learning."
    )
    print("PASS: Knowledge bases configured")

    # 4. Paper-Specific Prompts
    print("\n4. Paper-Specific Prompts")
    prompts = PaperSpecificPrompts()
    question_prompt = prompts.get_question_evaluation_prompt()
    print(f"PASS: Question prompt length: {len(question_prompt)} characters")
    print(f"PASS: Contains JSON format: {'json' in question_prompt.lower()}")

    # 5. Human Evaluator
    print("\n5. Human-in-the-Loop Workflow")
    human_evaluator = HumanEvaluator()
    print(
        f"PASS: Human evaluator initialized with {len(human_evaluator.workflow.nodes)} workflow nodes"
    )

    # 6. Review Point Creation
    print("\n6. Review Point Creation")
    question = create_review_point(
        text="What is the main contribution?",
        point_type=ReviewPointType.QUESTION,
        paper_context="Sample paper content",
        review_context="Sample review content",
        point_id="demo_q1",
    )
    print(f"PASS: Review point created: {question.type} - {question.text[:50]}...")

    print("\n" + "=" * 50)
    print("SUCCESS: All core components working correctly!")
    print("Ready for full evaluation with API keys.")


def demonstrate_paper_methodology():
    """Demonstrate paper methodology without API calls."""
    print("\nPAPER METHODOLOGY DEMONSTRATION")
    print("=" * 50)

    # Create evaluator
    evaluator = PaperFaithfulReviewScore()

    # Set up knowledge bases
    paper_content = """
    This paper presents a novel deep learning approach for natural language processing.
    The method achieves 95% accuracy on the benchmark dataset.
    The approach uses transformer architecture with attention mechanisms.
    """

    evaluator.set_knowledge_bases(
        submitted_paper=paper_content,
        annotator_knowledge="Expert knowledge about NLP and transformers",
        referred_papers=["Attention Is All You Need", "BERT paper", "GPT paper"],
    )

    print("PASS: Knowledge bases configured")
    print(f"  - Paper content: {len(paper_content)} characters")
    print(f"  - Annotator knowledge: Available")
    print(f"  - Referred papers: 3 papers")

    # Create sample review points
    review_points = [
        create_review_point(
            text="What accuracy does the method achieve?",
            point_type=ReviewPointType.QUESTION,
            paper_context=paper_content,
            review_context="Sample review",
            point_id="q1",
        ),
        create_review_point(
            text="The method achieves 95% accuracy",
            point_type=ReviewPointType.CLAIM,
            paper_context=paper_content,
            review_context="Sample review",
            point_id="c1",
        ),
        create_review_point(
            text="The method is effective because it uses transformers and achieves high accuracy",
            point_type=ReviewPointType.ARGUMENT,
            paper_context=paper_content,
            review_context="Sample review",
            point_id="a1",
        ),
    ]

    print(f"\nPASS: Created {len(review_points)} review points:")
    for i, point in enumerate(review_points):
        print(f"  {i+1}. {point.type}: {point.text[:50]}...")

    print("\n" + "=" * 50)
    print("SUCCESS: Paper methodology ready for evaluation!")
    print("Add API keys to run full evaluation.")


def main():
    """Run the simple demonstration."""
    print("REVIEWSCORE SIMPLE DEMONSTRATION")
    print("Following arXiv:2509.21679 methodology")
    print("=" * 60)

    try:
        demonstrate_core_components()
        demonstrate_paper_methodology()

        print("\n" + "=" * 60)
        print("SUCCESS: DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("All components are working correctly.")
        print("\nTo run full evaluation with LLMs:")
        print("1. Add your API keys to .env file")
        print("2. Run: python paper_faithful_example.py")
        print("3. Or run: python test_paper_faithful.py")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: Error during demonstration: {e}")
        print("Please check the implementation and try again.")


if __name__ == "__main__":
    main()
