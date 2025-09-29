"""
ReviewScore Visualization - Generate plots and charts for review analysis.
Based on the ReviewScore paper methodology and metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
from datetime import datetime

from .core import ReviewScoreResult, ReviewPointType
from .paper_review_result import PaperReviewResult, ReviewQualityLevel


class ReviewScoreVisualizer:
    """
    Comprehensive visualization system for ReviewScore analysis.
    Generates plots for individual and aggregated review metrics.
    """

    def __init__(self, style: str = "whitegrid", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style for plots
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.figsize = figsize
        self._setup_plotting()

    def _setup_plotting(self):
        """Setup plotting configuration."""
        # Use seaborn style instead of matplotlib style
        sns.set_style(self.style)
        sns.set_palette("husl")

        # Set default figure size
        plt.rcParams["figure.figsize"] = self.figsize
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.fontsize"] = 10

    def plot_review_score_distribution(
        self, results: List[ReviewScoreResult], save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of review scores by type.

        Args:
            results: List of ReviewScoreResult objects
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "ReviewScore Distribution Analysis", fontsize=16, fontweight="bold"
        )

        # Prepare data
        data = []
        for result in results:
            data.append(
                {
                    "type": result.review_point.type.value,
                    "base_score": result.base_score,
                    "advanced_score": result.advanced_score,
                    "is_misinformed": result.is_misinformed,
                    "confidence": result.confidence,
                }
            )

        df = pd.DataFrame(data)

        # 1. Score distribution by type
        ax1 = axes[0, 0]
        for point_type in df["type"].unique():
            type_data = df[df["type"] == point_type]
            ax1.hist(
                type_data["base_score"], alpha=0.7, label=point_type.title(), bins=5
            )
        ax1.set_xlabel("Base Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Score Distribution by Review Point Type")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Misinformed rate by type
        ax2 = axes[0, 1]
        misinformed_by_type = df.groupby("type")["is_misinformed"].mean() * 100
        bars = ax2.bar(misinformed_by_type.index, misinformed_by_type.values)
        ax2.set_ylabel("Misinformed Rate (%)")
        ax2.set_title("Misinformed Rate by Review Point Type")
        ax2.set_ylim(0, 100)

        # Add value labels on bars
        for bar, value in zip(bars, misinformed_by_type.values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
            )

        # 3. Confidence vs Score scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(
            df["base_score"],
            df["confidence"],
            c=df["is_misinformed"],
            cmap="RdYlGn",
            alpha=0.7,
        )
        ax3.set_xlabel("Base Score")
        ax3.set_ylabel("Confidence")
        ax3.set_title("Confidence vs Score (Color: Misinformed)")
        ax3.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label("Misinformed")

        # 4. Score comparison (Base vs Advanced)
        ax4 = axes[1, 1]
        has_advanced = df["advanced_score"].notna()
        if has_advanced.any():
            ax4.scatter(
                df.loc[has_advanced, "base_score"],
                df.loc[has_advanced, "advanced_score"],
                alpha=0.7,
            )
            ax4.plot([1, 5], [1, 5], "r--", alpha=0.5, label="Perfect Agreement")
            ax4.set_xlabel("Base Score")
            ax4.set_ylabel("Advanced Score")
            ax4.set_title("Base vs Advanced Score Comparison")
            ax4.legend()
        else:
            ax4.text(
                0.5,
                0.5,
                "No Advanced Scores Available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Advanced Score Comparison")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_paper_review_summary(
        self, paper_result: PaperReviewResult, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comprehensive summary for a paper review.

        Args:
            paper_result: PaperReviewResult object
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Paper Review Summary: {paper_result.paper_id}",
            fontsize=16,
            fontweight="bold",
        )

        summary = paper_result.summary

        # 1. Review Point Type Distribution
        ax1 = axes[0, 0]
        types = ["Questions", "Claims", "Arguments"]
        counts = [
            summary.questions_count,
            summary.claims_count,
            summary.arguments_count,
        ]
        colors = ["skyblue", "lightcoral", "lightgreen"]

        wedges, texts, autotexts = ax1.pie(
            counts, labels=types, colors=colors, autopct="%1.1f%%", startangle=90
        )
        ax1.set_title("Review Point Type Distribution")

        # 2. Misinformed Points Breakdown
        ax2 = axes[0, 1]
        misinformed_data = {
            "Questions": summary.misinformed_questions,
            "Claims": summary.misinformed_claims,
            "Arguments": summary.misinformed_arguments,
        }
        total_misinformed = sum(misinformed_data.values())

        if total_misinformed > 0:
            bars = ax2.bar(
                misinformed_data.keys(),
                misinformed_data.values(),
                color=["red", "orange", "yellow"],
            )
            ax2.set_ylabel("Number of Misinformed Points")
            ax2.set_title("Misinformed Points by Type")

            # Add value labels
            for bar, value in zip(bars, misinformed_data.values()):
                if value > 0:
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        str(value),
                        ha="center",
                        va="bottom",
                    )
        else:
            ax2.text(
                0.5,
                0.5,
                "No Misinformed Points",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=14,
                color="green",
            )
            ax2.set_title("Misinformed Points by Type")

        # 3. Quality Assessment
        ax3 = axes[0, 2]
        quality_colors = {
            ReviewQualityLevel.EXCELLENT: "green",
            ReviewQualityLevel.GOOD: "lightgreen",
            ReviewQualityLevel.FAIR: "yellow",
            ReviewQualityLevel.POOR: "orange",
            ReviewQualityLevel.VERY_POOR: "red",
        }

        quality_text = f"Overall Quality: {summary.overall_quality.value.title()}\n"
        quality_text += f"Quality Score: {summary.quality_score:.1f}/5.0\n"
        quality_text += f"Average Base Score: {summary.average_base_score:.2f}\n"
        quality_text += (
            f"Average Advanced Score: {summary.average_advanced_score:.2f}\n"
        )
        quality_text += f"Average Confidence: {summary.average_confidence:.2f}"

        ax3.text(
            0.1,
            0.5,
            quality_text,
            transform=ax3.transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=quality_colors.get(summary.overall_quality, "gray"),
                alpha=0.3,
            ),
        )
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis("off")
        ax3.set_title("Quality Assessment")

        # 4. Score Distribution by Type
        ax4 = axes[1, 0]
        all_scores = []
        all_types = []

        for result in paper_result.review_point_results:
            all_scores.append(result.base_score)
            all_types.append(result.review_point.type.value)

        if all_scores:
            df_scores = pd.DataFrame({"score": all_scores, "type": all_types})
            sns.boxplot(data=df_scores, x="type", y="score", ax=ax4)
            ax4.set_title("Score Distribution by Type")
            ax4.set_ylabel("Base Score")
            ax4.set_xlabel("Review Point Type")
            ax4.grid(True, alpha=0.3)

        # 5. Misinformed Rate Analysis
        ax5 = axes[1, 1]
        if summary.total_review_points > 0:
            rates = {
                "Overall": summary.total_misinformed / summary.total_review_points,
                "Questions": summary.misinformed_questions
                / max(summary.questions_count, 1),
                "Claims": summary.misinformed_claims / max(summary.claims_count, 1),
                "Arguments": summary.misinformed_arguments
                / max(summary.arguments_count, 1),
            }

            bars = ax5.bar(
                rates.keys(),
                [r * 100 for r in rates.values()],
                color=["blue", "red", "orange", "green"],
            )
            ax5.set_ylabel("Misinformed Rate (%)")
            ax5.set_title("Misinformed Rate by Category")
            ax5.set_ylim(0, 100)

            # Add value labels
            for bar, value in zip(bars, rates.values()):
                ax5.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{value*100:.1f}%",
                    ha="center",
                    va="bottom",
                )

        # 6. Recommendations Summary
        ax6 = axes[1, 2]
        recommendations_text = "Recommendations:\n\n"
        for i, rec in enumerate(paper_result.recommendations[:5], 1):  # Show first 5
            recommendations_text += f"{i}. {rec}\n"

        if len(paper_result.recommendations) > 5:
            recommendations_text += (
                f"\n... and {len(paper_result.recommendations) - 5} more"
            )

        ax6.text(
            0.05,
            0.95,
            recommendations_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
        )
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis("off")
        ax6.set_title("Key Recommendations")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_model_comparison(
        self,
        model_results: Dict[str, List[ReviewScoreResult]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot comparison between different models.

        Args:
            model_results: Dictionary mapping model names to their results
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Comparison Analysis", fontsize=16, fontweight="bold")

        # Prepare data
        comparison_data = []
        for model_name, results in model_results.items():
            for result in results:
                comparison_data.append(
                    {
                        "model": model_name,
                        "type": result.review_point.type.value,
                        "base_score": result.base_score,
                        "advanced_score": result.advanced_score,
                        "is_misinformed": result.is_misinformed,
                        "confidence": result.confidence,
                    }
                )

        df = pd.DataFrame(comparison_data)

        # 1. Average scores by model
        ax1 = axes[0, 0]
        model_stats = (
            df.groupby("model")
            .agg({"base_score": "mean", "advanced_score": "mean", "confidence": "mean"})
            .reset_index()
        )

        x = np.arange(len(model_stats))
        width = 0.25

        ax1.bar(
            x - width, model_stats["base_score"], width, label="Base Score", alpha=0.8
        )
        ax1.bar(
            x, model_stats["advanced_score"], width, label="Advanced Score", alpha=0.8
        )
        ax1.bar(
            x + width, model_stats["confidence"], width, label="Confidence", alpha=0.8
        )

        ax1.set_xlabel("Model")
        ax1.set_ylabel("Average Score")
        ax1.set_title("Average Scores by Model")
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_stats["model"], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Misinformed rate by model
        ax2 = axes[0, 1]
        misinformed_by_model = df.groupby("model")["is_misinformed"].mean() * 100
        bars = ax2.bar(misinformed_by_model.index, misinformed_by_model.values)
        ax2.set_ylabel("Misinformed Rate (%)")
        ax2.set_title("Misinformed Rate by Model")
        ax2.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, value in zip(bars, misinformed_by_model.values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
            )

        # 3. Score distribution comparison
        ax3 = axes[1, 0]
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            ax3.hist(model_data["base_score"], alpha=0.6, label=model, bins=5)
        ax3.set_xlabel("Base Score")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Score Distribution by Model")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Performance metrics table
        ax4 = axes[1, 1]
        metrics_data = []
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            metrics_data.append(
                {
                    "Model": model,
                    "Avg Score": f"{model_data['base_score'].mean():.2f}",
                    "Misinformed %": f"{model_data['is_misinformed'].mean()*100:.1f}%",
                    "Avg Confidence": f"{model_data['confidence'].mean():.2f}",
                    "Count": len(model_data),
                }
            )

        metrics_df = pd.DataFrame(metrics_data)

        # Create table
        table_data = []
        for _, row in metrics_df.iterrows():
            table_data.append(
                [
                    row["Model"],
                    row["Avg Score"],
                    row["Misinformed %"],
                    row["Avg Confidence"],
                    str(row["Count"]),
                ]
            )

        table = ax4.table(
            cellText=table_data,
            colLabels=["Model", "Avg Score", "Misinformed %", "Confidence", "Count"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.axis("off")
        ax4.set_title("Model Performance Metrics")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_temporal_analysis(
        self, results: List[ReviewScoreResult], save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot temporal analysis of review scores.

        Args:
            results: List of ReviewScoreResult objects with temporal metadata
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Temporal Analysis of Review Scores", fontsize=16, fontweight="bold"
        )

        # Prepare data with timestamps
        data = []
        for result in results:
            timestamp = result.evaluation_metadata.get(
                "timestamp", datetime.now().isoformat()
            )
            data.append(
                {
                    "timestamp": pd.to_datetime(timestamp),
                    "base_score": result.base_score,
                    "is_misinformed": result.is_misinformed,
                    "type": result.review_point.type.value,
                    "confidence": result.confidence,
                }
            )

        df = pd.DataFrame(data)
        df = df.sort_values("timestamp")

        # 1. Score trends over time
        ax1 = axes[0, 0]
        ax1.plot(df["timestamp"], df["base_score"], marker="o", alpha=0.7)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Base Score")
        ax1.set_title("Score Trends Over Time")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        # 2. Misinformed rate over time
        ax2 = axes[0, 1]
        df["misinformed_numeric"] = df["is_misinformed"].astype(int)
        rolling_misinformed = (
            df.set_index("timestamp")["misinformed_numeric"].rolling("1H").mean()
        )
        ax2.plot(
            rolling_misinformed.index,
            rolling_misinformed.values * 100,
            marker="o",
            color="red",
            alpha=0.7,
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Misinformed Rate (%)")
        ax2.set_title("Misinformed Rate Over Time (Rolling Average)")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        # 3. Score distribution by hour
        ax3 = axes[1, 0]
        df["hour"] = df["timestamp"].dt.hour
        hourly_scores = df.groupby("hour")["base_score"].mean()
        ax3.bar(hourly_scores.index, hourly_scores.values, alpha=0.7)
        ax3.set_xlabel("Hour of Day")
        ax3.set_ylabel("Average Score")
        ax3.set_title("Average Score by Hour of Day")
        ax3.grid(True, alpha=0.3)

        # 4. Type distribution over time
        ax4 = axes[1, 1]
        type_counts = (
            df.groupby([df["timestamp"].dt.date, "type"]).size().unstack(fill_value=0)
        )
        type_counts.plot(kind="bar", stacked=True, ax=ax4)
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Number of Review Points")
        ax4.set_title("Review Point Types Over Time")
        ax4.legend(title="Type")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_dashboard(
        self, paper_results: List[PaperReviewResult], save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard for multiple paper reviews.

        Args:
            paper_results: List of PaperReviewResult objects
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(
            "ReviewScore Dashboard - Multi-Paper Analysis",
            fontsize=18,
            fontweight="bold",
        )

        # Prepare aggregated data
        dashboard_data = []
        for paper in paper_results:
            dashboard_data.append(
                {
                    "paper_id": paper.paper_id,
                    "total_points": paper.summary.total_review_points,
                    "misinformed_rate": paper.summary.total_misinformed
                    / max(paper.summary.total_review_points, 1),
                    "avg_score": paper.summary.average_base_score,
                    "quality_score": paper.summary.quality_score,
                    "questions": paper.summary.questions_count,
                    "claims": paper.summary.claims_count,
                    "arguments": paper.summary.arguments_count,
                }
            )

        df = pd.DataFrame(dashboard_data)

        # 1. Papers overview
        ax1 = axes[0, 0]
        ax1.bar(range(len(df)), df["total_points"], alpha=0.7, color="skyblue")
        ax1.set_xlabel("Paper Index")
        ax1.set_ylabel("Number of Review Points")
        ax1.set_title("Review Points per Paper")
        ax1.grid(True, alpha=0.3)

        # 2. Misinformed rate distribution
        ax2 = axes[0, 1]
        ax2.hist(df["misinformed_rate"] * 100, bins=10, alpha=0.7, color="red")
        ax2.set_xlabel("Misinformed Rate (%)")
        ax2.set_ylabel("Number of Papers")
        ax2.set_title("Distribution of Misinformed Rates")
        ax2.grid(True, alpha=0.3)

        # 3. Quality score vs misinformed rate
        ax3 = axes[0, 2]
        scatter = ax3.scatter(
            df["misinformed_rate"] * 100,
            df["quality_score"],
            s=df["total_points"] * 10,
            alpha=0.7,
            c=df["avg_score"],
            cmap="RdYlGn",
        )
        ax3.set_xlabel("Misinformed Rate (%)")
        ax3.set_ylabel("Quality Score")
        ax3.set_title(
            "Quality vs Misinformed Rate\n(Size = Review Points, Color = Avg Score)"
        )
        ax3.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label("Average Score")

        # 4. Review point type distribution
        ax4 = axes[1, 0]
        type_totals = df[["questions", "claims", "arguments"]].sum()
        ax4.pie(
            type_totals.values,
            labels=type_totals.index,
            autopct="%1.1f%%",
            colors=["lightblue", "lightcoral", "lightgreen"],
        )
        ax4.set_title("Overall Review Point Type Distribution")

        # 5. Score distribution
        ax5 = axes[1, 1]
        ax5.hist(df["avg_score"], bins=10, alpha=0.7, color="green")
        ax5.set_xlabel("Average Score")
        ax5.set_ylabel("Number of Papers")
        ax5.set_title("Distribution of Average Scores")
        ax5.grid(True, alpha=0.3)

        # 6. Quality level distribution
        ax6 = axes[1, 2]
        quality_levels = [
            paper.summary.overall_quality.value for paper in paper_results
        ]
        quality_counts = pd.Series(quality_levels).value_counts()
        ax6.bar(quality_counts.index, quality_counts.values, alpha=0.7, color="purple")
        ax6.set_xlabel("Quality Level")
        ax6.set_ylabel("Number of Papers")
        ax6.set_title("Distribution of Quality Levels")
        ax6.tick_params(axis="x", rotation=45)
        ax6.grid(True, alpha=0.3)

        # 7. Top performing papers
        ax7 = axes[2, 0]
        top_papers = df.nlargest(5, "quality_score")
        ax7.barh(
            top_papers["paper_id"], top_papers["quality_score"], alpha=0.7, color="gold"
        )
        ax7.set_xlabel("Quality Score")
        ax7.set_ylabel("Paper ID")
        ax7.set_title("Top 5 Papers by Quality Score")
        ax7.grid(True, alpha=0.3)

        # 8. Worst performing papers
        ax8 = axes[2, 1]
        worst_papers = df.nsmallest(5, "quality_score")
        ax8.barh(
            worst_papers["paper_id"],
            worst_papers["quality_score"],
            alpha=0.7,
            color="red",
        )
        ax8.set_xlabel("Quality Score")
        ax8.set_ylabel("Paper ID")
        ax8.set_title("Bottom 5 Papers by Quality Score")
        ax8.grid(True, alpha=0.3)

        # 9. Summary statistics
        ax9 = axes[2, 2]
        summary_text = f"""
        Dashboard Summary:
        
        Total Papers: {len(paper_results)}
        Total Review Points: {df['total_points'].sum()}
        Average Misinformed Rate: {df['misinformed_rate'].mean()*100:.1f}%
        Average Quality Score: {df['quality_score'].mean():.2f}
        Average Base Score: {df['avg_score'].mean():.2f}
        
        Best Paper: {df.loc[df['quality_score'].idxmax(), 'paper_id']}
        Worst Paper: {df.loc[df['quality_score'].idxmin(), 'paper_id']}
        """

        ax9.text(
            0.05,
            0.95,
            summary_text,
            transform=ax9.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
        )
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.axis("off")
        ax9.set_title("Summary Statistics")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


def create_visualizer(
    style: str = "whitegrid", figsize: Tuple[int, int] = (12, 8)
) -> ReviewScoreVisualizer:
    """
    Factory function to create a ReviewScore visualizer.

    Args:
        style: Matplotlib style for plots
        figsize: Default figure size (width, height)

    Returns:
        ReviewScoreVisualizer instance
    """
    return ReviewScoreVisualizer(style, figsize)


# Example usage and testing
if __name__ == "__main__":
    print("ReviewScore Visualization System")
    print("=" * 50)
    print("This module provides comprehensive visualization for ReviewScore analysis.")
    print("Use create_visualizer() to get started with plotting review metrics.")
