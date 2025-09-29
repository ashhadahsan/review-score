"""
Aggregation methods for premise factuality scores in ADVANCED REVIEW SCORE.
Implements logical conjunction and weighted average as described in the paper.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .core import Premise, Argument, AggregationMethod


class PremiseAggregator:
    """
    Aggregates premise factuality scores using different methods.
    Implements the aggregation methods described in Section 3.1 of the paper.
    """
    
    def __init__(self, method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE):
        self.method = method
    
    def aggregate_premise_scores(self, premises: List[Premise]) -> float:
        """
        Aggregate premise factuality scores into a single score.
        
        Args:
            premises: List of premises with factuality scores
            
        Returns:
            Aggregated score (1-5 scale)
        """
        if not premises:
            return 3.0  # Neutral score if no premises
        
        # Filter out premises without scores
        scored_premises = [p for p in premises if p.factuality_score is not None]
        
        if not scored_premises:
            return 3.0  # Neutral score if no scored premises
        
        if self.method == AggregationMethod.LOGICAL_CONJUNCTION:
            return self._logical_conjunction(scored_premises)
        elif self.method == AggregationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(scored_premises)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
    
    def _logical_conjunction(self, premises: List[Premise]) -> float:
        """
        Logical conjunction aggregation: all premises must be correct.
        Returns 1 if any premise is incorrect, 5 if all are correct.
        """
        # Check if any premise is incorrect (score <= 2.5)
        for premise in premises:
            if premise.factuality_score <= 2.5:
                return 1.0  # Misinformed if any premise is incorrect
        
        # All premises are correct
        return 5.0
    
    def _weighted_average(self, premises: List[Premise]) -> float:
        """
        Weighted average aggregation: average of premise scores.
        Maintains the 1-5 scale while considering all premises.
        """
        scores = [p.factuality_score for p in premises]
        
        # Simple average (could be enhanced with weights based on premise importance)
        return np.mean(scores)
    
    def aggregate_with_weights(self, premises: List[Premise], weights: Optional[List[float]] = None) -> float:
        """
        Weighted average with custom weights.
        
        Args:
            premises: List of premises with factuality scores
            weights: Optional list of weights for each premise
            
        Returns:
            Weighted average score (1-5 scale)
        """
        if not premises:
            return 3.0
        
        scored_premises = [p for p in premises if p.factuality_score is not None]
        
        if not scored_premises:
            return 3.0
        
        scores = [p.factuality_score for p in scored_premises]
        
        if weights is None:
            # Equal weights
            weights = [1.0] * len(scores)
        else:
            # Ensure weights match number of scored premises
            if len(weights) != len(scores):
                weights = weights[:len(scores)] if len(weights) > len(scores) else weights + [1.0] * (len(scores) - len(weights))
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        return np.average(scores, weights=weights)
    
    def get_aggregation_metadata(self, premises: List[Premise]) -> Dict[str, Any]:
        """
        Get metadata about the aggregation process.
        
        Args:
            premises: List of premises that were aggregated
            
        Returns:
            Dictionary with aggregation metadata
        """
        scored_premises = [p for p in premises if p.factuality_score is not None]
        
        if not scored_premises:
            return {
                "method": self.method.value,
                "num_premises": len(premises),
                "num_scored_premises": 0,
                "scores": [],
                "aggregated_score": 3.0
            }
        
        scores = [p.factuality_score for p in scored_premises]
        aggregated_score = self.aggregate_premise_scores(premises)
        
        return {
            "method": self.method.value,
            "num_premises": len(premises),
            "num_scored_premises": len(scored_premises),
            "scores": scores,
            "aggregated_score": aggregated_score,
            "score_range": (min(scores), max(scores)) if scores else (3.0, 3.0),
            "score_std": np.std(scores) if len(scores) > 1 else 0.0
        }


class AdvancedReviewScoreEvaluator:
    """
    Implements ADVANCED REVIEW SCORE evaluation with premise-level factuality.
    Uses argument reconstruction and premise aggregation.
    """
    
    def __init__(self, reconstruction_engine, aggregator: PremiseAggregator):
        self.reconstruction_engine = reconstruction_engine
        self.aggregator = aggregator
    
    def evaluate_argument(self, argument: Argument) -> float:
        """
        Evaluate an argument using ADVANCED REVIEW SCORE.
        
        Args:
            argument: Argument to evaluate
            
        Returns:
            ADVANCED REVIEW SCORE (1-5 scale)
        """
        # Extract premises with factuality scores
        premises = self.reconstruction_engine.extract_premises_with_factuality(argument)
        
        # Aggregate premise scores
        aggregated_score = self.aggregator.aggregate_premise_scores(premises)
        
        # Update the argument with aggregated score
        argument.aggregated_factuality_score = aggregated_score
        
        return aggregated_score
    
    def get_evaluation_metadata(self, argument: Argument) -> Dict[str, Any]:
        """
        Get detailed metadata about the ADVANCED REVIEW SCORE evaluation.
        
        Args:
            argument: Argument that was evaluated
            
        Returns:
            Dictionary with evaluation metadata
        """
        premises = argument.premises
        aggregation_metadata = self.aggregator.get_aggregation_metadata(premises)
        
        # Add reconstruction metadata
        reconstruction_metadata = {
            "is_valid": argument.is_valid,
            "is_faithful": argument.is_faithful,
            "conclusion": argument.conclusion,
            "num_premises": len(premises),
            "num_explicit_premises": len([p for p in premises if p.is_explicit]),
            "num_implicit_premises": len([p for p in premises if not p.is_explicit])
        }
        
        return {
            **aggregation_metadata,
            **reconstruction_metadata,
            "premise_details": [
                {
                    "text": p.text,
                    "is_explicit": p.is_explicit,
                    "factuality_score": p.factuality_score,
                    "is_factually_correct": p.is_factually_correct
                }
                for p in premises
            ]
        }


def create_advanced_evaluator(
    model_name: str = "claude-3-5-sonnet-20241022",
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
) -> AdvancedReviewScoreEvaluator:
    """
    Factory function to create an ADVANCED REVIEW SCORE evaluator.
    
    Args:
        model_name: Name of the model to use for reconstruction
        aggregation_method: Method for aggregating premise scores
        
    Returns:
        AdvancedReviewScoreEvaluator instance
    """
    from .argument_reconstruction import create_reconstruction_engine
    
    reconstruction_engine = create_reconstruction_engine(model_name)
    aggregator = PremiseAggregator(aggregation_method)
    
    return AdvancedReviewScoreEvaluator(reconstruction_engine, aggregator)
