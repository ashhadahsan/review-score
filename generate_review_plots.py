#!/usr/bin/env python3
"""
Generate comprehensive ReviewScore plots for demonstration.
Creates all visualization types and saves them as high-quality PNG files.
"""

import os
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

from reviewscore.visualization import create_visualizer
from reviewscore.paper_review_result import create_paper_review_aggregator
from reviewscore.core import create_review_point, ReviewPointType, ReviewScoreResult


def create_comprehensive_sample_data():
    """Create comprehensive sample data for demonstration."""
    print("Creating comprehensive sample data...")

    # Create multiple papers with different characteristics
    all_results = []
    paper_results = []

    # Paper 1: High quality review
    print("  - Creating high-quality paper review...")
    high_quality_results = []
    for i in range(20):
        point_type = random.choice(list(ReviewPointType))
        review_point = create_review_point(
            text=f"High-quality {point_type.value} review point {i+1}",
            point_type=point_type,
            paper_context="High-quality paper content",
            review_context="High-quality review context",
            point_id=f"high_quality_{i}",
        )

        # Higher scores for high-quality review
        base_score = random.uniform(3.5, 5.0)
        advanced_score = base_score + random.uniform(-0.3, 0.3)
        advanced_score = max(1.0, min(5.0, advanced_score))

        result = ReviewScoreResult(
            review_point=review_point,
            base_score=base_score,
            advanced_score=advanced_score,
            is_misinformed=random.random() < 0.1,  # 10% misinformed
            confidence=random.uniform(0.7, 1.0),
            reasoning=f"High-quality reasoning for {point_type.value}",
            model_used="claude-3.5-sonnet",
            evaluation_metadata={
                "timestamp": (
                    datetime.now() - timedelta(days=random.randint(0, 7))
                ).isoformat(),
                "evaluation_time": random.uniform(1.0, 2.5),
            },
        )
        high_quality_results.append(result)
        all_results.append(result)

    # Aggregate high-quality paper
    aggregator = create_paper_review_aggregator()
    high_quality_paper = aggregator.aggregate_paper_review(
        paper_id="high_quality_paper",
        review_point_results=high_quality_results,
        paper_title="High-Quality Research Paper",
    )
    paper_results.append(high_quality_paper)

    # Paper 2: Poor quality review
    print("  - Creating poor-quality paper review...")
    poor_quality_results = []
    for i in range(15):
        point_type = random.choice(list(ReviewPointType))
        review_point = create_review_point(
            text=f"Poor-quality {point_type.value} review point {i+1}",
            point_type=point_type,
            paper_context="Poor-quality paper content",
            review_context="Poor-quality review context",
            point_id=f"poor_quality_{i}",
        )

        # Lower scores for poor-quality review
        base_score = random.uniform(1.0, 3.0)
        advanced_score = base_score + random.uniform(-0.5, 0.5)
        advanced_score = max(1.0, min(5.0, advanced_score))

        result = ReviewScoreResult(
            review_point=review_point,
            base_score=base_score,
            advanced_score=advanced_score,
            is_misinformed=random.random() < 0.4,  # 40% misinformed
            confidence=random.uniform(0.3, 0.7),
            reasoning=f"Poor-quality reasoning for {point_type.value}",
            model_used="gpt-4o",
            evaluation_metadata={
                "timestamp": (
                    datetime.now() - timedelta(days=random.randint(0, 7))
                ).isoformat(),
                "evaluation_time": random.uniform(0.5, 1.5),
            },
        )
        poor_quality_results.append(result)
        all_results.append(result)

    # Aggregate poor-quality paper
    poor_quality_paper = aggregator.aggregate_paper_review(
        paper_id="poor_quality_paper",
        review_point_results=poor_quality_results,
        paper_title="Poor-Quality Research Paper",
    )
    paper_results.append(poor_quality_paper)

    # Paper 3: Mixed quality review
    print("  - Creating mixed-quality paper review...")
    mixed_quality_results = []
    for i in range(25):
        point_type = random.choice(list(ReviewPointType))
        review_point = create_review_point(
            text=f"Mixed-quality {point_type.value} review point {i+1}",
            point_type=point_type,
            paper_context="Mixed-quality paper content",
            review_context="Mixed-quality review context",
            point_id=f"mixed_quality_{i}",
        )

        # Mixed scores
        base_score = random.uniform(2.0, 4.5)
        advanced_score = base_score + random.uniform(-0.4, 0.4)
        advanced_score = max(1.0, min(5.0, advanced_score))

        result = ReviewScoreResult(
            review_point=review_point,
            base_score=base_score,
            advanced_score=advanced_score,
            is_misinformed=random.random() < 0.25,  # 25% misinformed
            confidence=random.uniform(0.4, 0.9),
            reasoning=f"Mixed-quality reasoning for {point_type.value}",
            model_used="gemini-2.5-flash",
            evaluation_metadata={
                "timestamp": (
                    datetime.now() - timedelta(days=random.randint(0, 7))
                ).isoformat(),
                "evaluation_time": random.uniform(0.8, 2.0),
            },
        )
        mixed_quality_results.append(result)
        all_results.append(result)

    # Aggregate mixed-quality paper
    mixed_quality_paper = aggregator.aggregate_paper_review(
        paper_id="mixed_quality_paper",
        review_point_results=mixed_quality_results,
        paper_title="Mixed-Quality Research Paper",
    )
    paper_results.append(mixed_quality_paper)

    return all_results, paper_results


def generate_all_plots():
    """Generate and save all ReviewScore plots."""
    print("ReviewScore Plot Generation")
    print("=" * 60)

    # Create output directory
    output_dir = "reviewscore_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create sample data
    all_results, paper_results = create_comprehensive_sample_data()

    # Create visualizer
    visualizer = create_visualizer(figsize=(15, 10))

    # 1. Review Score Distribution
    print("\n1. Generating Review Score Distribution Plot...")
    fig1 = visualizer.plot_review_score_distribution(
        all_results,
        save_path=os.path.join(output_dir, "01_review_score_distribution.png"),
    )
    print("   ✓ Saved: 01_review_score_distribution.png")

    # 2. Model Comparison
    print("\n2. Generating Model Comparison Plot...")
    model_results = {
        "Claude-3.5-Sonnet": [
            r for r in all_results if r.model_used == "claude-3.5-sonnet"
        ],
        "GPT-4o": [r for r in all_results if r.model_used == "gpt-4o"],
        "Gemini-2.5-Flash": [
            r for r in all_results if r.model_used == "gemini-2.5-flash"
        ],
    }

    fig2 = visualizer.plot_model_comparison(
        model_results, save_path=os.path.join(output_dir, "02_model_comparison.png")
    )
    print("   ✓ Saved: 02_model_comparison.png")

    # 3. Temporal Analysis
    print("\n3. Generating Temporal Analysis Plot...")
    fig3 = visualizer.plot_temporal_analysis(
        all_results, save_path=os.path.join(output_dir, "03_temporal_analysis.png")
    )
    print("   ✓ Saved: 03_temporal_analysis.png")

    # 4. Individual Paper Summaries
    print("\n4. Generating Individual Paper Summary Plots...")
    for i, paper_result in enumerate(paper_results, 1):
        fig = visualizer.plot_paper_review_summary(
            paper_result,
            save_path=os.path.join(output_dir, f"04_paper_{i}_summary.png"),
        )
        print(f"   ✓ Saved: 04_paper_{i}_summary.png ({paper_result.paper_id})")

    # 5. Multi-Paper Dashboard
    print("\n5. Generating Multi-Paper Dashboard...")
    fig5 = visualizer.create_dashboard(
        paper_results,
        save_path=os.path.join(output_dir, "05_multi_paper_dashboard.png"),
    )
    print("   ✓ Saved: 05_multi_paper_dashboard.png")

    # 6. Advanced Analysis - Score Distribution by Quality
    print("\n6. Generating Advanced Analysis Plots...")

    # Separate results by paper quality
    high_quality_results = paper_results[0].review_point_results
    poor_quality_results = paper_results[1].review_point_results
    mixed_quality_results = paper_results[2].review_point_results

    # Create quality comparison plot
    import matplotlib.pyplot as plt

    fig6, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig6.suptitle(
        "Review Quality Analysis by Paper Type", fontsize=16, fontweight="bold"
    )

    # Score distribution by quality
    ax1 = axes[0, 0]
    ax1.hist(
        [r.base_score for r in high_quality_results],
        alpha=0.7,
        label="High Quality",
        bins=5,
    )
    ax1.hist(
        [r.base_score for r in poor_quality_results],
        alpha=0.7,
        label="Poor Quality",
        bins=5,
    )
    ax1.hist(
        [r.base_score for r in mixed_quality_results],
        alpha=0.7,
        label="Mixed Quality",
        bins=5,
    )
    ax1.set_xlabel("Base Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Score Distribution by Paper Quality")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Misinformed rate by quality
    ax2 = axes[0, 1]
    quality_types = ["High Quality", "Poor Quality", "Mixed Quality"]
    misinformed_rates = [
        sum(1 for r in high_quality_results if r.is_misinformed)
        / len(high_quality_results),
        sum(1 for r in poor_quality_results if r.is_misinformed)
        / len(poor_quality_results),
        sum(1 for r in mixed_quality_results if r.is_misinformed)
        / len(mixed_quality_results),
    ]
    bars = ax2.bar(
        quality_types,
        [r * 100 for r in misinformed_rates],
        color=["green", "red", "orange"],
        alpha=0.7,
    )
    ax2.set_ylabel("Misinformed Rate (%)")
    ax2.set_title("Misinformed Rate by Paper Quality")
    ax2.set_ylim(0, 100)

    # Add value labels
    for bar, rate in zip(bars, misinformed_rates):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate*100:.1f}%",
            ha="center",
            va="bottom",
        )

    # Confidence vs Score by quality
    ax3 = axes[1, 0]
    ax3.scatter(
        [r.base_score for r in high_quality_results],
        [r.confidence for r in high_quality_results],
        alpha=0.7,
        label="High Quality",
        color="green",
    )
    ax3.scatter(
        [r.base_score for r in poor_quality_results],
        [r.confidence for r in poor_quality_results],
        alpha=0.7,
        label="Poor Quality",
        color="red",
    )
    ax3.scatter(
        [r.base_score for r in mixed_quality_results],
        [r.confidence for r in mixed_quality_results],
        alpha=0.7,
        label="Mixed Quality",
        color="orange",
    )
    ax3.set_xlabel("Base Score")
    ax3.set_ylabel("Confidence")
    ax3.set_title("Confidence vs Score by Paper Quality")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Quality summary table
    ax4 = axes[1, 1]
    summary_data = []
    for i, paper in enumerate(paper_results):
        summary_data.append(
            [
                paper.paper_id,
                f"{paper.summary.quality_score:.2f}",
                f"{paper.summary.total_misinformed}/{paper.summary.total_review_points}",
                f"{paper.summary.average_base_score:.2f}",
                paper.summary.overall_quality.value.title(),
            ]
        )

    table = ax4.table(
        cellText=summary_data,
        colLabels=[
            "Paper ID",
            "Quality Score",
            "Misinformed",
            "Avg Score",
            "Quality Level",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax4.axis("off")
    ax4.set_title("Paper Quality Summary")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "06_quality_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("   ✓ Saved: 06_quality_analysis.png")

    # 7. Create summary report
    print("\n7. Generating Summary Report...")
    create_summary_report(paper_results, all_results, output_dir)
    print("   ✓ Saved: 07_summary_report.txt")

    return output_dir


def create_summary_report(paper_results, all_results, output_dir):
    """Create a comprehensive summary report."""
    report = []
    report.append("ReviewScore Analysis Summary Report")
    report.append("=" * 60)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Overall statistics
    total_papers = len(paper_results)
    total_review_points = sum(p.summary.total_review_points for p in paper_results)
    total_misinformed = sum(p.summary.total_misinformed for p in paper_results)
    avg_quality = np.mean([p.summary.quality_score for p in paper_results])

    report.append("Overall Statistics:")
    report.append(f"  Total Papers Analyzed: {total_papers}")
    report.append(f"  Total Review Points: {total_review_points}")
    report.append(f"  Total Misinformed Points: {total_misinformed}")
    report.append(
        f"  Overall Misinformed Rate: {total_misinformed/total_review_points*100:.1f}%"
    )
    report.append(f"  Average Quality Score: {avg_quality:.2f}/5.0")
    report.append("")

    # Paper-by-paper analysis
    report.append("Paper-by-Paper Analysis:")
    report.append("-" * 40)
    for i, paper in enumerate(paper_results, 1):
        report.append(f"Paper {i}: {paper.paper_id}")
        report.append(f"  Title: {paper.paper_title}")
        report.append(f"  Review Points: {paper.summary.total_review_points}")
        report.append(f"    - Questions: {paper.summary.questions_count}")
        report.append(f"    - Claims: {paper.summary.claims_count}")
        report.append(f"    - Arguments: {paper.summary.arguments_count}")
        report.append(f"  Misinformed Points: {paper.summary.total_misinformed}")
        report.append(f"  Quality Score: {paper.summary.quality_score:.2f}/5.0")
        report.append(f"  Quality Level: {paper.summary.overall_quality.value.title()}")
        report.append(f"  Average Base Score: {paper.summary.average_base_score:.2f}")
        report.append("")

    # Quality level distribution
    quality_levels = [p.summary.overall_quality.value for p in paper_results]
    quality_counts = {}
    for level in quality_levels:
        quality_counts[level] = quality_counts.get(level, 0) + 1

    report.append("Quality Level Distribution:")
    for level, count in quality_counts.items():
        report.append(
            f"  {level.title()}: {count} papers ({count/total_papers*100:.1f}%)"
        )
    report.append("")

    # Model performance
    model_stats = {}
    for result in all_results:
        model = result.model_used
        if model not in model_stats:
            model_stats[model] = {"scores": [], "misinformed": [], "confidence": []}
        model_stats[model]["scores"].append(result.base_score)
        model_stats[model]["misinformed"].append(result.is_misinformed)
        model_stats[model]["confidence"].append(result.confidence)

    report.append("Model Performance:")
    for model, stats in model_stats.items():
        avg_score = np.mean(stats["scores"])
        misinformed_rate = np.mean(stats["misinformed"]) * 100
        avg_confidence = np.mean(stats["confidence"])
        report.append(f"  {model}:")
        report.append(f"    Average Score: {avg_score:.2f}")
        report.append(f"    Misinformed Rate: {misinformed_rate:.1f}%")
        report.append(f"    Average Confidence: {avg_confidence:.2f}")
    report.append("")

    # Generated plots
    report.append("Generated Plots:")
    report.append(
        "  01_review_score_distribution.png - Score distribution by review point type"
    )
    report.append("  02_model_comparison.png - Performance comparison across models")
    report.append("  03_temporal_analysis.png - Review quality trends over time")
    report.append("  04_paper_*_summary.png - Individual paper review summaries")
    report.append("  05_multi_paper_dashboard.png - Comprehensive multi-paper analysis")
    report.append("  06_quality_analysis.png - Advanced quality analysis by paper type")
    report.append("  07_summary_report.txt - This summary report")
    report.append("")

    # Recommendations
    report.append("Key Recommendations:")
    all_recommendations = []
    for paper in paper_results:
        all_recommendations.extend(paper.recommendations)

    unique_recommendations = list(set(all_recommendations))
    for i, rec in enumerate(unique_recommendations[:10], 1):
        report.append(f"  {i}. {rec}")

    # Save report
    with open(os.path.join(output_dir, "07_summary_report.txt"), "w") as f:
        f.write("\n".join(report))


def main():
    """Main function to generate all plots."""
    print("ReviewScore Plot Generation System")
    print("=" * 60)
    print(
        "This script generates comprehensive visualizations for ReviewScore analysis."
    )
    print("All plots will be saved as high-quality PNG files.")
    print("=" * 60)

    try:
        # Generate all plots
        output_dir = generate_all_plots()

        print("\n" + "=" * 60)
        print("PLOT GENERATION COMPLETED!")
        print("=" * 60)
        print(f"All plots saved in: {output_dir}/")
        print("\nGenerated files:")
        print("  01_review_score_distribution.png")
        print("  02_model_comparison.png")
        print("  03_temporal_analysis.png")
        print("  04_paper_*_summary.png (3 files)")
        print("  05_multi_paper_dashboard.png")
        print("  06_quality_analysis.png")
        print("  07_summary_report.txt")
        print("\nThese plots demonstrate the complete ReviewScore methodology")
        print("with comprehensive metrics visualization and analysis.")

    except Exception as e:
        print(f"Error during plot generation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
