"""
Evaluation metrics for ReviewScore human-model agreement.
Implements F1, Kappa, accuracy, and other metrics as described in the paper.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)
from scipy.stats import pearsonr, spearmanr
import pandas as pd

from .core import ReviewScoreResult, EvaluationMetrics


class ReviewScoreEvaluator:
    """
    Evaluator for measuring human-model agreement on ReviewScore.
    Implements the evaluation metrics described in Section 4 of the paper.
    """

    def __init__(self):
        self.results = []
        self.human_annotations = []

    def add_evaluation_result(
        self, model_result: ReviewScoreResult, human_annotation: Dict[str, Any]
    ):
        """
        Add a model evaluation result with corresponding human annotation.

        Args:
            model_result: Result from model evaluation
            human_annotation: Human annotation with ground truth
        """
        self.results.append(model_result)
        self.human_annotations.append(human_annotation)

    def calculate_metrics(self) -> Dict[str, EvaluationMetrics]:
        """
        Calculate comprehensive evaluation metrics.

        Returns:
            Dictionary with metrics for different evaluation types
        """
        if not self.results:
            raise ValueError("No evaluation results available")

        metrics = {}

        # Binary classification metrics
        metrics["binary_classification"] = self._calculate_binary_metrics()

        # 5-point scale metrics
        metrics["five_point_scale"] = self._calculate_five_point_metrics()

        # Agreement metrics
        metrics["agreement"] = self._calculate_agreement_metrics()

        # Model-specific metrics
        metrics["model_comparison"] = self._calculate_model_comparison_metrics()

        return metrics

    def _calculate_binary_metrics(self) -> EvaluationMetrics:
        """Calculate binary classification metrics (Misinformed vs Not misinformed)."""
        if not self.results:
            return EvaluationMetrics(
                f1_score=0.0,
                kappa_score=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                human_model_agreement=0.0,
            )

        # Extract binary labels
        model_labels = [1 if result.is_misinformed else 0 for result in self.results]
        human_labels = [
            1 if annotation.get("is_misinformed", False) else 0
            for annotation in self.human_annotations
        ]

        # Calculate metrics
        f1 = f1_score(human_labels, model_labels, average="weighted")
        kappa = cohen_kappa_score(human_labels, model_labels)
        accuracy = accuracy_score(human_labels, model_labels)
        precision = precision_score(
            human_labels, model_labels, average="weighted", zero_division=0
        )
        recall = recall_score(
            human_labels, model_labels, average="weighted", zero_division=0
        )

        # Human-model agreement (same as accuracy for binary)
        agreement = accuracy

        return EvaluationMetrics(
            f1_score=f1,
            kappa_score=kappa,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            human_model_agreement=agreement,
        )

    def _calculate_five_point_metrics(self) -> EvaluationMetrics:
        """Calculate 5-point scale metrics."""
        if not self.results:
            return EvaluationMetrics(
                f1_score=0.0,
                kappa_score=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                human_model_agreement=0.0,
            )

        # Extract 5-point scale scores
        model_scores = []
        human_scores = []

        for result, annotation in zip(self.results, self.human_annotations):
            # Use advanced score if available, otherwise base score
            model_score = (
                result.advanced_score
                if result.advanced_score is not None
                else result.base_score
            )
            human_score = annotation.get("score", 3.0)

            if model_score is not None and human_score is not None:
                model_scores.append(model_score)
                human_scores.append(human_score)

        if not model_scores:
            return EvaluationMetrics(
                f1_score=0.0,
                kappa_score=0.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                human_model_agreement=0.0,
            )

        # Convert to binary for F1 calculation (1-2 = misinformed, 3-5 = not misinformed)
        model_binary = [1 if score <= 2 else 0 for score in model_scores]
        human_binary = [1 if score <= 2 else 0 for score in human_scores]

        # Calculate metrics
        f1 = f1_score(human_binary, model_binary, average="weighted")
        kappa = cohen_kappa_score(human_binary, model_binary)
        accuracy = accuracy_score(human_binary, model_binary)
        precision = precision_score(
            human_binary, model_binary, average="weighted", zero_division=0
        )
        recall = recall_score(
            human_binary, model_binary, average="weighted", zero_division=0
        )

        # Calculate correlation for 5-point scale
        correlation = (
            pearsonr(model_scores, human_scores)[0] if len(model_scores) > 1 else 0.0
        )
        agreement = correlation if not np.isnan(correlation) else 0.0

        return EvaluationMetrics(
            f1_score=f1,
            kappa_score=kappa,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            human_model_agreement=agreement,
        )

    def _calculate_agreement_metrics(self) -> Dict[str, float]:
        """Calculate agreement metrics between human and model evaluations."""
        if not self.results:
            return {"overall_agreement": 0.0}

        # Extract scores and labels
        model_scores = []
        human_scores = []
        model_labels = []
        human_labels = []

        for result, annotation in zip(self.results, self.human_annotations):
            # Scores
            model_score = (
                result.advanced_score
                if result.advanced_score is not None
                else result.base_score
            )
            human_score = annotation.get("score", 3.0)

            if model_score is not None and human_score is not None:
                model_scores.append(model_score)
                human_scores.append(human_score)

            # Binary labels
            model_labels.append(1 if result.is_misinformed else 0)
            human_labels.append(1 if annotation.get("is_misinformed", False) else 0)

        # Calculate agreement metrics
        exact_agreement = sum(
            1 for m, h in zip(model_labels, human_labels) if m == h
        ) / len(model_labels)

        # Score correlation
        score_correlation = 0.0
        if len(model_scores) > 1:
            correlation = pearsonr(model_scores, human_scores)[0]
            score_correlation = correlation if not np.isnan(correlation) else 0.0

        # Kappa agreement
        kappa_agreement = cohen_kappa_score(human_labels, model_labels)

        return {
            "exact_agreement": exact_agreement,
            "score_correlation": score_correlation,
            "kappa_agreement": kappa_agreement,
            "overall_agreement": (
                exact_agreement + abs(score_correlation) + abs(kappa_agreement)
            )
            / 3,
        }

    def _calculate_model_comparison_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for comparing different models."""
        if not self.results:
            return {"model_performance": {}}

        # Group results by model
        model_results = {}
        for result in self.results:
            model_name = result.model_used
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)

        # Calculate metrics for each model
        model_performance = {}
        for model_name, results in model_results.items():
            # Get corresponding human annotations
            model_indices = [
                i
                for i, result in enumerate(self.results)
                if result.model_used == model_name
            ]
            model_human_annotations = [self.human_annotations[i] for i in model_indices]

            # Calculate metrics for this model
            model_metrics = self._calculate_model_specific_metrics(
                results, model_human_annotations
            )
            model_performance[model_name] = model_metrics

        return {"model_performance": model_performance}

    def _calculate_model_specific_metrics(
        self, results: List[ReviewScoreResult], annotations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate metrics for a specific model."""
        if not results:
            return {"f1_score": 0.0, "kappa_score": 0.0, "accuracy": 0.0}

        # Extract labels
        model_labels = [1 if result.is_misinformed else 0 for result in results]
        human_labels = [
            1 if annotation.get("is_misinformed", False) else 0
            for annotation in annotations
        ]

        # Calculate metrics
        f1 = f1_score(human_labels, model_labels, average="weighted")
        kappa = cohen_kappa_score(human_labels, model_labels)
        accuracy = accuracy_score(human_labels, model_labels)

        return {"f1_score": f1, "kappa_score": kappa, "accuracy": accuracy}

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix for binary classification."""
        if not self.results:
            return np.array([[0, 0], [0, 0]])

        model_labels = [1 if result.is_misinformed else 0 for result in self.results]
        human_labels = [
            1 if annotation.get("is_misinformed", False) else 0
            for annotation in self.human_annotations
        ]

        return confusion_matrix(human_labels, model_labels)

    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if not self.results:
            return "No results available"

        model_labels = [1 if result.is_misinformed else 0 for result in self.results]
        human_labels = [
            1 if annotation.get("is_misinformed", False) else 0
            for annotation in self.human_annotations
        ]

        return classification_report(
            human_labels, model_labels, target_names=["Not Misinformed", "Misinformed"]
        )

    def get_disagreement_analysis(self) -> Dict[str, Any]:
        """Analyze disagreements between human and model evaluations."""
        if not self.results:
            return {"disagreements": []}

        disagreements = []

        for i, (result, annotation) in enumerate(
            zip(self.results, self.human_annotations)
        ):
            model_misinformed = result.is_misinformed
            human_misinformed = annotation.get("is_misinformed", False)

            if model_misinformed != human_misinformed:
                disagreement = {
                    "index": i,
                    "review_point_text": result.review_point.text[:100] + "...",
                    "model_prediction": model_misinformed,
                    "human_annotation": human_misinformed,
                    "model_score": (
                        result.advanced_score
                        if result.advanced_score is not None
                        else result.base_score
                    ),
                    "human_score": annotation.get("score", 3.0),
                    "model_reasoning": result.reasoning,
                    "human_reasoning": annotation.get("reasoning", ""),
                }
                disagreements.append(disagreement)

        return {
            "total_disagreements": len(disagreements),
            "disagreement_rate": len(disagreements) / len(self.results),
            "disagreements": disagreements,
        }

    def export_results(self, filename: str = "reviewscore_evaluation_results.csv"):
        """Export evaluation results to CSV file."""
        if not self.results:
            return

        data = []
        for i, (result, annotation) in enumerate(
            zip(self.results, self.human_annotations)
        ):
            row = {
                "index": i,
                "review_point_id": result.review_point.id,
                "review_point_type": result.review_point.type,
                "review_point_text": result.review_point.text,
                "model_used": result.model_used,
                "model_base_score": result.base_score,
                "model_advanced_score": result.advanced_score,
                "model_is_misinformed": result.is_misinformed,
                "model_confidence": result.confidence,
                "model_reasoning": result.reasoning,
                "human_score": annotation.get("score", 3.0),
                "human_is_misinformed": annotation.get("is_misinformed", False),
                "human_reasoning": annotation.get("reasoning", ""),
                "agreement": result.is_misinformed
                == annotation.get("is_misinformed", False),
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")


def create_evaluator() -> ReviewScoreEvaluator:
    """Factory function to create a ReviewScore evaluator."""
    return ReviewScoreEvaluator()
