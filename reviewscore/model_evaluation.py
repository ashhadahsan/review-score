"""
Model evaluation system for ReviewScore using proprietary models.
Implements the evaluation framework described in Section 4 of the paper.
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .core import ReviewPoint, ReviewScoreResult, ModelConfig, PROPRIETARY_MODELS
from .base_evaluation import BaseReviewScoreEvaluator
from .lcel_workflows import ReviewScoreLCELWorkflow, ReviewScoreParallelWorkflow
from .langgraph_flows import ReviewScoreLangGraphFlow
from .evaluation_metrics import ReviewScoreEvaluator


class ModelEvaluationSystem:
    """
    Comprehensive model evaluation system for ReviewScore.
    Implements the evaluation framework described in the paper.
    """

    def __init__(self, model_configs: List[ModelConfig] = None):
        self.model_configs = model_configs or PROPRIETARY_MODELS
        self.evaluators = {}
        self.evaluation_results = {}
        self.metrics_calculator = ReviewScoreEvaluator()

        # Initialize evaluators for each model
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        """Initialize evaluators for each model configuration."""
        for config in self.model_configs:
            try:
                # Create different types of evaluators
                self.evaluators[config.model_name] = {
                    "base": BaseReviewScoreEvaluator(config),
                    "lcel": ReviewScoreLCELWorkflow(config),
                    "langgraph": ReviewScoreLangGraphFlow(config),
                }
            except Exception as e:
                print(
                    f"Warning: Failed to initialize evaluator for {config.model_name}: {e}"
                )

    def evaluate_single_model(
        self,
        model_name: str,
        review_points: List[ReviewPoint],
        evaluation_type: str = "lcel",
    ) -> List[ReviewScoreResult]:
        """
        Evaluate review points using a single model.

        Args:
            model_name: Name of the model to use
            review_points: List of review points to evaluate
            evaluation_type: Type of evaluation ("base", "lcel", "langgraph")

        Returns:
            List of evaluation results
        """
        if model_name not in self.evaluators:
            raise ValueError(f"Model {model_name} not available")

        if evaluation_type not in self.evaluators[model_name]:
            raise ValueError(
                f"Evaluation type {evaluation_type} not available for {model_name}"
            )

        evaluator = self.evaluators[model_name][evaluation_type]
        results = []

        print(
            f"Evaluating {len(review_points)} review points with {model_name} ({evaluation_type})"
        )

        for i, review_point in enumerate(review_points):
            try:
                if evaluation_type == "base":
                    result = evaluator.evaluate_review_point(review_point)
                elif evaluation_type == "lcel":
                    result = evaluator.evaluate_review_point(review_point)
                elif evaluation_type == "langgraph":
                    result = evaluator.evaluate_review_point(review_point)
                else:
                    raise ValueError(f"Unknown evaluation type: {evaluation_type}")

                results.append(result)

                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(review_points)} evaluations")

            except Exception as e:
                print(f"Error evaluating review point {i}: {e}")
                # Create error result
                error_result = ReviewScoreResult(
                    review_point=review_point,
                    base_score=3.0,
                    advanced_score=3.0,
                    is_misinformed=False,
                    confidence=0.0,
                    reasoning=f"Evaluation error: {str(e)}",
                    model_used=model_name,
                    evaluation_metadata={"error": True, "error_message": str(e)},
                )
                results.append(error_result)

        return results

    def evaluate_all_models(
        self, review_points: List[ReviewPoint], evaluation_type: str = "lcel"
    ) -> Dict[str, List[ReviewScoreResult]]:
        """
        Evaluate review points using all available models.

        Args:
            review_points: List of review points to evaluate
            evaluation_type: Type of evaluation to use

        Returns:
            Dictionary mapping model names to their results
        """
        all_results = {}

        for model_name in self.evaluators.keys():
            try:
                results = self.evaluate_single_model(
                    model_name, review_points, evaluation_type
                )
                all_results[model_name] = results
                print(f"Completed evaluation with {model_name}")
            except Exception as e:
                print(f"Error evaluating with {model_name}: {e}")
                all_results[model_name] = []

        return all_results

    def evaluate_parallel(
        self,
        review_points: List[ReviewPoint],
        evaluation_type: str = "lcel",
        max_workers: int = 3,
    ) -> Dict[str, List[ReviewScoreResult]]:
        """
        Evaluate review points using multiple models in parallel.

        Args:
            review_points: List of review points to evaluate
            evaluation_type: Type of evaluation to use
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary mapping model names to their results
        """
        all_results = {}

        def evaluate_model(model_name):
            try:
                return model_name, self.evaluate_single_model(
                    model_name, review_points, evaluation_type
                )
            except Exception as e:
                print(f"Error in parallel evaluation with {model_name}: {e}")
                return model_name, []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            future_to_model = {
                executor.submit(evaluate_model, model_name): model_name
                for model_name in self.evaluators.keys()
            }

            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_name, results = future.result()
                all_results[model_name] = results
                print(f"Completed parallel evaluation with {model_name}")

        return all_results

    def compare_models(
        self,
        review_points: List[ReviewPoint],
        human_annotations: List[Dict[str, Any]],
        evaluation_type: str = "lcel",
    ) -> Dict[str, Any]:
        """
        Compare performance of different models.

        Args:
            review_points: List of review points to evaluate
            human_annotations: Human annotations for ground truth
            evaluation_type: Type of evaluation to use

        Returns:
            Dictionary with model comparison results
        """
        # Evaluate with all models
        all_results = self.evaluate_all_models(review_points, evaluation_type)

        # Calculate metrics for each model
        model_metrics = {}

        for model_name, results in all_results.items():
            if not results:
                continue

            # Create evaluator for this model
            evaluator = ReviewScoreEvaluator()

            # Add results to evaluator
            for result, annotation in zip(results, human_annotations):
                evaluator.add_evaluation_result(result, annotation)

            # Calculate metrics
            metrics = evaluator.calculate_metrics()
            model_metrics[model_name] = metrics

        # Find best performing model
        best_model = None
        best_f1 = 0.0

        for model_name, metrics in model_metrics.items():
            f1 = metrics.get("binary_classification", {}).get("f1_score", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name

        return {
            "model_metrics": model_metrics,
            "best_model": best_model,
            "best_f1_score": best_f1,
            "comparison_summary": self._create_comparison_summary(model_metrics),
        }

    def _create_comparison_summary(
        self, model_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a summary of model comparison results."""
        summary = {"model_rankings": {}, "performance_gaps": {}, "recommendations": []}

        # Rank models by F1 score
        f1_scores = {}
        for model_name, metrics in model_metrics.items():
            f1 = metrics.get("binary_classification", {}).get("f1_score", 0.0)
            f1_scores[model_name] = f1

        sorted_models = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
        summary["model_rankings"]["f1_score"] = sorted_models

        # Calculate performance gaps
        if len(sorted_models) > 1:
            best_f1 = sorted_models[0][1]
            for model_name, f1 in sorted_models[1:]:
                gap = best_f1 - f1
                summary["performance_gaps"][model_name] = gap

        # Generate recommendations
        if sorted_models:
            best_model = sorted_models[0][0]
            summary["recommendations"].append(f"Best performing model: {best_model}")

            if len(sorted_models) > 1:
                second_best = sorted_models[1][0]
                gap = sorted_models[0][1] - sorted_models[1][1]
                if gap < 0.05:  # Less than 5% difference
                    summary["recommendations"].append(
                        f"Consider ensemble of {best_model} and {second_best}"
                    )

        return summary

    def evaluate_with_consensus(
        self, review_points: List[ReviewPoint], evaluation_type: str = "lcel"
    ) -> List[ReviewScoreResult]:
        """
        Evaluate review points using consensus from multiple models.

        Args:
            review_points: List of review points to evaluate
            evaluation_type: Type of evaluation to use

        Returns:
            List of consensus evaluation results
        """
        # Get results from all models
        all_results = self.evaluate_all_models(review_points, evaluation_type)

        # Calculate consensus for each review point
        consensus_results = []

        for i, review_point in enumerate(review_points):
            # Collect results from all models for this review point
            model_results = []
            for model_name, results in all_results.items():
                if i < len(results):
                    model_results.append(results[i])

            if not model_results:
                continue

            # Calculate consensus
            consensus_result = self._calculate_consensus_result(
                review_point, model_results
            )
            consensus_results.append(consensus_result)

        return consensus_results

    def _calculate_consensus_result(
        self, review_point: ReviewPoint, model_results: List[ReviewScoreResult]
    ) -> ReviewScoreResult:
        """Calculate consensus result from multiple model results."""
        if not model_results:
            return ReviewScoreResult(
                review_point=review_point,
                base_score=3.0,
                advanced_score=3.0,
                is_misinformed=False,
                confidence=0.0,
                reasoning="No model results available",
                model_used="consensus",
                evaluation_metadata={"consensus": True},
            )

        # Calculate consensus scores
        base_scores = [r.base_score for r in model_results if r.base_score is not None]
        advanced_scores = [
            r.advanced_score for r in model_results if r.advanced_score is not None
        ]
        misinformed_votes = [
            r.is_misinformed for r in model_results if r.is_misinformed is not None
        ]

        consensus_base = sum(base_scores) / len(base_scores) if base_scores else 3.0
        consensus_advanced = (
            sum(advanced_scores) / len(advanced_scores) if advanced_scores else 3.0
        )
        consensus_misinformed = (
            sum(misinformed_votes) / len(misinformed_votes) > 0.5
            if misinformed_votes
            else False
        )

        # Calculate confidence based on agreement
        agreement_rate = sum(
            1 for r in model_results if r.is_misinformed == consensus_misinformed
        ) / len(model_results)
        confidence = agreement_rate

        # Combine reasoning from all models
        reasoning_parts = [r.reasoning for r in model_results if r.reasoning]
        combined_reasoning = (
            " | ".join(reasoning_parts) if reasoning_parts else "Consensus evaluation"
        )

        return ReviewScoreResult(
            review_point=review_point,
            base_score=consensus_base,
            advanced_score=consensus_advanced,
            is_misinformed=consensus_misinformed,
            confidence=confidence,
            reasoning=f"Consensus from {len(model_results)} models: {combined_reasoning}",
            model_used="consensus",
            evaluation_metadata={
                "consensus": True,
                "num_models": len(model_results),
                "agreement_rate": agreement_rate,
                "model_results": [r.model_used for r in model_results],
            },
        )

    def export_evaluation_results(
        self,
        results: Dict[str, List[ReviewScoreResult]],
        filename: str = "model_evaluation_results.json",
    ):
        """Export evaluation results to JSON file."""
        export_data = {}

        for model_name, model_results in results.items():
            export_data[model_name] = []
            for result in model_results:
                result_data = {
                    "review_point_id": result.review_point.id,
                    "review_point_type": result.review_point.type,
                    "review_point_text": result.review_point.text,
                    "base_score": result.base_score,
                    "advanced_score": result.advanced_score,
                    "is_misinformed": result.is_misinformed,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "model_used": result.model_used,
                    "evaluation_metadata": result.evaluation_metadata,
                }
                export_data[model_name].append(result_data)

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Evaluation results exported to {filename}")


def create_model_evaluation_system(
    model_names: List[str] = None, custom_configs: List[ModelConfig] = None
) -> ModelEvaluationSystem:
    """
    Factory function to create a model evaluation system.

    Args:
        model_names: List of model names to use (defaults to all proprietary models)
        custom_configs: Custom model configurations

    Returns:
        ModelEvaluationSystem instance
    """
    if custom_configs:
        return ModelEvaluationSystem(custom_configs)

    if model_names:
        # Filter to specified models
        selected_configs = []
        for config in PROPRIETARY_MODELS:
            if config.model_name in model_names:
                selected_configs.append(config)

        if not selected_configs:
            raise ValueError(f"No valid model configurations found for: {model_names}")

        return ModelEvaluationSystem(selected_configs)

    return ModelEvaluationSystem()
