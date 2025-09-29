#!/usr/bin/env python3
"""
ReviewScore Plot Creation Example.
Demonstrates how to create and save ReviewScore visualizations.
"""

import os
import random
from datetime import datetime, timedelta
from typing import List

from reviewscore.visualization import create_visualizer
from reviewscore.paper_review_result import create_paper_review_aggregator
from reviewscore.core import create_review_point, ReviewPointType, ReviewScoreResult


def create_sample_review_data(num_points: int = 30) -> List[ReviewScoreResult]:
    """Create sample review data for demonstration."""
    results = []

    # Sample review point texts
    question_texts = [
        "What methodology does this paper use?",
        "How does this compare to existing approaches?",
        "What are the limitations of this work?",
        "Can you provide more details on the experimental setup?",
        "What datasets were used for evaluation?",
    ]

    claim_texts = [
        "The experimental results are not convincing.",
        "The paper lacks proper evaluation.",
        "The methodology is novel and well-designed.",
        "The experimental setup is insufficient.",
        "The paper makes significant contributions.",
    ]

    argument_texts = [
        "The paper lacks proper evaluation because the dataset is too small.",
        "The experimental results are not convincing because the baseline comparison is inadequate.",
        "The methodology is novel because it introduces a new attention mechanism.",
        "The paper makes significant contributions because it achieves state-of-the-art performance.",
        "The experimental setup is insufficient because it doesn't include ablation studies.",
    ]

    for i in range(num_points):
        # Randomly select type and text
        point_type = random.choice(list(ReviewPointType))

        if point_type == ReviewPointType.QUESTION:
            text = random.choice(question_texts)
        elif point_type == ReviewPointType.CLAIM:
            text = random.choice(claim_texts)
        else:  # ARGUMENT
            text = random.choice(argument_texts)

        # Create review point
        review_point = create_review_point(
            text=text,
            point_type=point_type,
            paper_context="Sample paper content for demonstration.",
            review_context="Sample review context for demonstration.",
            point_id=f"sample_{i}",
        )

        # Generate realistic scores
        base_score = random.uniform(1.0, 5.0)
        advanced_score = base_score + random.uniform(-0.5, 0.5)
        advanced_score = max(1.0, min(5.0, advanced_score))

        is_misinformed = base_score <= 2.5
        confidence = random.uniform(0.3, 1.0)

        # Create result
        result = ReviewScoreResult(
            review_point=review_point,
            base_score=base_score,
            advanced_score=advanced_score,
            is_misinformed=is_misinformed,
            confidence=confidence,
            reasoning=f"Sample reasoning for {point_type.value} evaluation.",
            model_used="sample_model",
            evaluation_metadata={
                "timestamp": (
                    datetime.now() - timedelta(days=random.randint(0, 7))
                ).isoformat(),
                "evaluation_time": random.uniform(0.5, 3.0),
            },
        )

        results.append(result)

    return results


def example_basic_plotting():
    """Example: Basic plotting with individual review results."""
    print("Example 1: Basic Review Score Plotting")
    print("=" * 50)

    # Create sample data
    results = create_sample_review_data(50)

    # Create visualizer
    visualizer = create_visualizer()

    # Create output directory
    output_dir = "example_plots"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Review Score Distribution
    print("Creating review score distribution plot...")
    fig1 = visualizer.plot_review_score_distribution(
        results, save_path=os.path.join(output_dir, "review_score_distribution.png")
    )
    print(f"Saved: {output_dir}/review_score_distribution.png")

    # 2. Model Comparison (simulate multiple models)
    print("Creating model comparison plot...")
    model_results = {
        "Claude-3.5-Sonnet": results[:15],
        "GPT-4o": results[15:30],
        "Gemini-2.5-Flash": results[30:45],
    }

    fig2 = visualizer.plot_model_comparison(
        model_results, save_path=os.path.join(output_dir, "model_comparison.png")
    )
    print(f"Saved: {output_dir}/model_comparison.png")

    # 3. Temporal Analysis
    print("Creating temporal analysis plot...")
    fig3 = visualizer.plot_temporal_analysis(
        results, save_path=os.path.join(output_dir, "temporal_analysis.png")
    )
    print(f"Saved: {output_dir}/temporal_analysis.png")

    return output_dir


def example_paper_analysis():
    """Example: Paper-level analysis and visualization."""
    print("\nExample 2: Paper-Level Analysis")
    print("=" * 50)

    # Create sample data for a paper
    results = create_sample_review_data(25)

    # Aggregate into paper result
    aggregator = create_paper_review_aggregator()
    paper_result = aggregator.aggregate_paper_review(
        paper_id="example_paper",
        review_point_results=results,
        paper_title="Example Research Paper",
    )

    # Create visualizer
    visualizer = create_visualizer()

    # Create output directory
    output_dir = "example_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Create paper summary plot
    print("Creating paper review summary plot...")
    fig = visualizer.plot_paper_review_summary(
        paper_result, save_path=os.path.join(output_dir, "paper_summary.png")
    )
    print(f"Saved: {output_dir}/paper_summary.png")

    # Show paper summary
    summary = paper_result.summary
    print(f"\nPaper Summary:")
    print(f"  Total Review Points: {summary.total_review_points}")
    print(f"  Questions: {summary.questions_count}")
    print(f"  Claims: {summary.claims_count}")
    print(f"  Arguments: {summary.arguments_count}")
    print(f"  Misinformed Points: {summary.total_misinformed}")
    print(f"  Quality Score: {summary.quality_score:.2f}/5.0")
    print(f"  Quality Level: {summary.overall_quality.value.title()}")

    return paper_result


def example_multi_paper_dashboard():
    """Example: Multi-paper dashboard creation."""
    print("\nExample 3: Multi-Paper Dashboard")
    print("=" * 50)

    # Create multiple papers
    paper_results = []
    aggregator = create_paper_review_aggregator()

    paper_titles = [
        "Machine Learning for NLP",
        "Computer Vision Applications",
        "Deep Learning Optimization",
    ]

    for i, title in enumerate(paper_titles):
        # Create sample data for each paper
        results = create_sample_review_data(20)

        # Aggregate into paper result
        paper_result = aggregator.aggregate_paper_review(
            paper_id=f"paper_{i+1}", review_point_results=results, paper_title=title
        )
        paper_results.append(paper_result)

    # Create visualizer
    visualizer = create_visualizer()

    # Create output directory
    output_dir = "example_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Create multi-paper dashboard
    print("Creating multi-paper dashboard...")
    fig = visualizer.create_dashboard(
        paper_results, save_path=os.path.join(output_dir, "multi_paper_dashboard.png")
    )
    print(f"Saved: {output_dir}/multi_paper_dashboard.png")

    # Show summary statistics
    total_papers = len(paper_results)
    total_points = sum(p.summary.total_review_points for p in paper_results)
    total_misinformed = sum(p.summary.total_misinformed for p in paper_results)

    print(f"\nDashboard Summary:")
    print(f"  Total Papers: {total_papers}")
    print(f"  Total Review Points: {total_points}")
    print(f"  Total Misinformed: {total_misinformed}")
    print(f"  Overall Misinformed Rate: {total_misinformed/total_points*100:.1f}%")

    return paper_results


def example_custom_visualization():
    """Example: Custom visualization with specific requirements."""
    print("\nExample 4: Custom Visualization")
    print("=" * 50)

    # Create sample data
    results = create_sample_review_data(40)

    # Create visualizer with custom settings
    visualizer = create_visualizer(
        style="darkgrid", figsize=(16, 10)  # Different style  # Larger figure size
    )

    # Create output directory
    output_dir = "example_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Create custom plot with specific focus
    print("Creating custom visualization...")
    fig = visualizer.plot_review_score_distribution(
        results, save_path=os.path.join(output_dir, "custom_visualization.png")
    )
    print(f"Saved: {output_dir}/custom_visualization.png")

    return output_dir


def main():
    """Main function demonstrating all plotting examples."""
    print("ReviewScore Plot Creation Examples")
    print("=" * 60)
    print("This script demonstrates how to create and save ReviewScore visualizations.")
    print("All plots will be saved as high-quality PNG files.")
    print("=" * 60)

    try:
        # Run all examples
        example_basic_plotting()
        example_paper_analysis()
        example_multi_paper_dashboard()
        example_custom_visualization()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 60)
        print("Generated files in 'example_plots/' directory:")
        print("  - review_score_distribution.png")
        print("  - model_comparison.png")
        print("  - temporal_analysis.png")
        print("  - paper_summary.png")
        print("  - multi_paper_dashboard.png")
        print("  - custom_visualization.png")
        print("\nThese examples show how to:")
        print("  1. Create basic review score plots")
        print("  2. Generate paper-level analysis")
        print("  3. Build multi-paper dashboards")
        print("  4. Customize visualization settings")
        print("  5. Save plots as high-quality PNG files")

    except Exception as e:
        print(f"Error during example execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
