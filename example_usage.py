"""
Example usage of the ReviewScore system.
Demonstrates how to use the different evaluation methods and workflows.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from reviewscore import (
    create_review_point,
    ReviewPointType,
    Question,
    Claim,
    Argument,
    create_lcel_workflow,
    create_langgraph_flow,
    create_model_evaluation_system,
    create_evaluator,
)


def create_sample_data():
    """Create sample review points for demonstration."""

    # Sample paper content
    paper_content = """
    Title: Efficient Neural Architecture Search for Computer Vision
    
    Abstract: We propose a novel neural architecture search (NAS) method that reduces 
    search time by 50% while maintaining competitive accuracy. Our approach uses 
    reinforcement learning to guide the search process and introduces a new 
    hierarchical search space that enables efficient exploration.
    
    Introduction: Neural Architecture Search (NAS) has become a crucial technique 
    for automating the design of neural networks. However, existing methods suffer 
    from high computational costs and long search times. In this paper, we address 
    these limitations by proposing a more efficient search strategy.
    
    Method: Our approach consists of three main components: (1) a hierarchical 
    search space that reduces the number of possible architectures, (2) a 
    reinforcement learning agent that learns to select promising architectures, 
    and (3) a progressive search strategy that starts with simple architectures 
    and gradually increases complexity.
    
    Experiments: We evaluate our method on CIFAR-10 and ImageNet datasets. 
    Results show that our approach achieves 2.3% better accuracy than 
    state-of-the-art methods while reducing search time by 50%.
    
    Conclusion: We have presented an efficient NAS method that significantly 
    reduces computational costs while maintaining high performance. Future work 
    will explore the application of our method to other domains.
    """

    # Sample review content
    review_content = """
    This paper presents an interesting approach to neural architecture search. 
    However, I have several concerns about the methodology and experimental setup.
    
    The authors claim to reduce search time by 50%, but this comparison is not 
    fair since they use a different search space. The baseline methods they 
    compare against use much larger search spaces, which naturally take longer 
    to explore.
    
    The experimental results are promising, but I'm concerned about the 
    reproducibility. The authors don't provide sufficient details about the 
    hyperparameters used in their experiments, particularly for the 
    reinforcement learning component.
    
    Additionally, the paper lacks a thorough analysis of the computational 
    complexity of their approach. While they claim efficiency improvements, 
    they don't provide a detailed complexity analysis or comparison with 
    other efficient NAS methods.
    
    The hierarchical search space is interesting, but I'm not convinced 
    that it doesn't limit the expressiveness of the search. The authors 
    should provide more analysis of what architectures are excluded by 
    their hierarchical structure.
    """

    # Create sample review points
    review_points = [
        # Question
        create_review_point(
            text="What is the computational complexity of the proposed method?",
            point_type=ReviewPointType.QUESTION,
            paper_context=paper_content,
            review_context=review_content,
            point_id="question_1",
        ),
        # Claim
        create_review_point(
            text="The authors claim to reduce search time by 50%, but this comparison is not fair since they use a different search space.",
            point_type=ReviewPointType.CLAIM,
            paper_context=paper_content,
            review_context=review_content,
            point_id="claim_1",
        ),
        # Argument
        create_review_point(
            text="The experimental results are promising, but I'm concerned about the reproducibility. The authors don't provide sufficient details about the hyperparameters used in their experiments, particularly for the reinforcement learning component.",
            point_type=ReviewPointType.ARGUMENT,
            paper_context=paper_content,
            review_context=review_content,
            point_id="argument_1",
        ),
    ]

    return review_points


def demonstrate_base_evaluation():
    """Demonstrate BASE REVIEW SCORE evaluation."""
    print("=== BASE REVIEW SCORE Evaluation ===")

    from reviewscore.base_evaluation import create_base_evaluator

    # Create evaluator
    evaluator = create_base_evaluator("claude-3-5-sonnet-20241022")

    # Get sample data
    review_points = create_sample_data()

    # Evaluate each review point
    for i, review_point in enumerate(review_points):
        print(f"\n--- Review Point {i+1} ({review_point.type}) ---")
        print(f"Text: {review_point.text[:100]}...")

        try:
            result = evaluator.evaluate_review_point(review_point)
            print(f"Base Score: {result.base_score}")
            print(f"Is Misinformed: {result.is_misinformed}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Reasoning: {result.reasoning[:200]}...")
        except Exception as e:
            print(f"Error: {e}")


def demonstrate_lcel_workflow():
    """Demonstrate LCEL workflow evaluation."""
    print("\n=== LCEL Workflow Evaluation ===")

    # Create LCEL workflow
    workflow = create_lcel_workflow("claude-3-5-sonnet-20241022")

    # Get sample data
    review_points = create_sample_data()

    # Evaluate each review point
    for i, review_point in enumerate(review_points):
        print(f"\n--- Review Point {i+1} ({review_point.type}) ---")
        print(f"Text: {review_point.text[:100]}...")

        try:
            result = workflow.evaluate_review_point(review_point)
            print(f"Base Score: {result.base_score}")
            print(f"Advanced Score: {result.advanced_score}")
            print(f"Is Misinformed: {result.is_misinformed}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Model Used: {result.model_used}")
        except Exception as e:
            print(f"Error: {e}")


def demonstrate_langgraph_flow():
    """Demonstrate LangGraph flow evaluation."""
    print("\n=== LangGraph Flow Evaluation ===")

    # Create LangGraph flow
    flow = create_langgraph_flow("claude-3-5-sonnet-20241022")

    # Get sample data
    review_points = create_sample_data()

    # Evaluate each review point
    for i, review_point in enumerate(review_points):
        print(f"\n--- Review Point {i+1} ({review_point.type}) ---")
        print(f"Text: {review_point.text[:100]}...")

        try:
            result = flow.evaluate_review_point(review_point)
            print(f"Base Score: {result.base_score}")
            print(f"Advanced Score: {result.advanced_score}")
            print(f"Is Misinformed: {result.is_misinformed}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Model Used: {result.model_used}")
        except Exception as e:
            print(f"Error: {e}")


def demonstrate_model_comparison():
    """Demonstrate model comparison evaluation."""
    print("\n=== Model Comparison Evaluation ===")

    # Create model evaluation system
    evaluation_system = create_model_evaluation_system()

    # Get sample data
    review_points = create_sample_data()

    # Create sample human annotations
    human_annotations = [
        {
            "is_misinformed": False,  # Question is not answerable by paper
            "score": 4.0,
            "reasoning": "The paper does not provide detailed complexity analysis",
        },
        {
            "is_misinformed": True,  # Claim is incorrect
            "score": 2.0,
            "reasoning": "The paper does provide fair comparison with baseline methods",
        },
        {
            "is_misinformed": False,  # Argument is valid
            "score": 4.0,
            "reasoning": "The concern about reproducibility is valid and not addressed in the paper",
        },
    ]

    # Evaluate with multiple models
    print("Evaluating with multiple models...")
    all_results = evaluation_system.evaluate_all_models(review_points, "lcel")

    # Compare models
    comparison_results = evaluation_system.compare_models(
        review_points, human_annotations, "lcel"
    )

    print("\nModel Comparison Results:")
    for model_name, metrics in comparison_results["model_metrics"].items():
        f1 = metrics.get("binary_classification", {}).get("f1_score", 0.0)
        kappa = metrics.get("binary_classification", {}).get("kappa_score", 0.0)
        print(f"{model_name}: F1={f1:.3f}, Kappa={kappa:.3f}")

    print(f"\nBest Model: {comparison_results['best_model']}")
    print(f"Best F1 Score: {comparison_results['best_f1_score']:.3f}")


def demonstrate_consensus_evaluation():
    """Demonstrate consensus evaluation."""
    print("\n=== Consensus Evaluation ===")

    # Create model evaluation system
    evaluation_system = create_model_evaluation_system()

    # Get sample data
    review_points = create_sample_data()

    # Evaluate with consensus
    print("Evaluating with consensus from multiple models...")
    consensus_results = evaluation_system.evaluate_with_consensus(review_points, "lcel")

    for i, result in enumerate(consensus_results):
        print(f"\n--- Consensus Result {i+1} ---")
        print(f"Base Score: {result.base_score}")
        print(f"Advanced Score: {result.advanced_score}")
        print(f"Is Misinformed: {result.is_misinformed}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Model Used: {result.model_used}")
        print(
            f"Agreement Rate: {result.evaluation_metadata.get('agreement_rate', 0):.2f}"
        )


def main():
    """Main demonstration function."""
    print("ReviewScore System Demonstration")
    print("=" * 50)

    # Check if API keys are set
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print(f"Warning: Missing API keys: {missing_keys}")
        print("Please set the required API keys in your environment or .env file")
        print("Continuing with demonstration (some features may not work)...")

    try:
        # Demonstrate different evaluation methods
        demonstrate_base_evaluation()
        demonstrate_lcel_workflow()
        demonstrate_langgraph_flow()
        demonstrate_model_comparison()
        demonstrate_consensus_evaluation()

        print("\n" + "=" * 50)
        print("Demonstration completed successfully!")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("This is likely due to missing API keys or network issues.")


if __name__ == "__main__":
    main()
