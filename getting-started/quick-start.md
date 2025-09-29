# Quick Start Guide

Get up and running with ReviewScore in just a few minutes!

## 5-Minute Setup

### Step 1: Basic Evaluation

```python
from reviewscore import create_review_point, ReviewPointType
from reviewscore.paper_faithful import create_paper_faithful_evaluator

# Create a review point
question = create_review_point(
    text="What methodology was used in this paper?",
    point_type=ReviewPointType.QUESTION,
    paper_context="This paper uses transformer architecture for NLP tasks...",
    review_context="The reviewer asks about the methodology used...",
    point_id="q1"
)

# Create evaluator
evaluator = create_paper_faithful_evaluator()

# Evaluate
result = evaluator.evaluate_review_point(question)
print(f"Score: {result.base_score}/5.0")
print(f"Advanced Score: {result.advanced_score}/5.0")
print(f"Misinformed: {result.is_misinformed}")
print(f"Confidence: {result.confidence}")
```

### Step 2: Batch Evaluation

```python
# Create multiple review points
review_points = [
    create_review_point(
        text="The experimental results are not convincing.",
        point_type=ReviewPointType.CLAIM,
        paper_context="Paper content...",
        review_context="Review content...",
        point_id="c1"
    ),
    create_review_point(
        text="The methodology is novel and well-designed.",
        point_type=ReviewPointType.ARGUMENT,
        paper_context="Paper content...",
        review_context="Review content...",
        point_id="a1"
    )
]

# Evaluate all points
results = evaluator.evaluate_batch(review_points)

for result in results:
    print(f"Point {result.review_point.point_id}: {result.base_score}/5.0")
```

### Step 3: Paper-Level Analysis

```python
from reviewscore.paper_review_result import create_paper_review_aggregator

# Aggregate results into paper-level analysis
aggregator = create_paper_review_aggregator()
paper_result = aggregator.aggregate_paper_review(
    paper_id="paper_123",
    review_point_results=results,
    paper_title="Sample Research Paper"
)

print(f"Overall Quality: {paper_result.summary.overall_quality.value}")
print(f"Quality Score: {paper_result.summary.quality_score}/5.0")
print(f"Misinformed Rate: {paper_result.summary.misinformed_rate:.1%}")
```

## Advanced Features

### SAT Solver Integration

```python
from reviewscore.paper_faithful import PaperFaithfulConfig

# Configure SAT solver
config = PaperFaithfulConfig(
    sat_solver="z3",  # or "pysat" or "simple"
    enable_sat_validation=True
)

evaluator = create_paper_faithful_evaluator(config)
```

### Human-in-the-Loop Workflows

```python
from reviewscore.langgraph_agents import create_langgraph_agent

# Create agent with human-in-the-loop
agent = create_langgraph_agent(
    model_name="claude-3-5-sonnet-20241022",
    enable_human_in_loop=True
)

# Evaluate with potential human intervention
result = agent.evaluate_review_point(question)

# If human input is required
if result.requires_human_input:
    human_response = input("Please provide your annotation: ")
    result = agent.resume_with_human_input(human_response)
```

### PDF Knowledge Base

```python
from reviewscore.pdf_knowledge_base import create_pdf_knowledge_base

# Create knowledge base from PDF
kb = create_pdf_knowledge_base("paper.pdf")

# Use in evaluation
config = PaperFaithfulConfig(
    knowledge_base=kb,
    enable_knowledge_base=True
)

evaluator = create_paper_faithful_evaluator(config)
```

### Visualization

```python
from reviewscore.visualization import create_visualizer

# Create visualizer
visualizer = create_visualizer()

# Generate plots
fig1 = visualizer.plot_review_score_distribution(
    results,
    save_path="scores.png"
)

fig2 = visualizer.plot_paper_review_summary(
    paper_result,
    save_path="paper_summary.png"
)
```

## Complete Example

Here's a complete example that demonstrates all major features:

```python
#!/usr/bin/env python3
"""
Complete ReviewScore example demonstrating all features.
"""

import os
from dotenv import load_dotenv
from reviewscore import create_review_point, ReviewPointType
from reviewscore.paper_faithful import create_paper_faithful_evaluator, PaperFaithfulConfig
from reviewscore.paper_review_result import create_paper_review_aggregator
from reviewscore.visualization import create_visualizer

# Load environment variables
load_dotenv()

def main():
    print("ReviewScore Complete Example")
    print("=" * 50)

    # 1. Create review points
    review_points = [
        create_review_point(
            text="What methodology was used?",
            point_type=ReviewPointType.QUESTION,
            paper_context="This paper uses transformer architecture...",
            review_context="The reviewer asks about methodology...",
            point_id="q1"
        ),
        create_review_point(
            text="The experimental results are not convincing.",
            point_type=ReviewPointType.CLAIM,
            paper_context="The paper presents experimental results...",
            review_context="The reviewer claims results are not convincing...",
            point_id="c1"
        ),
        create_review_point(
            text="The methodology is novel because it introduces attention mechanisms.",
            point_type=ReviewPointType.ARGUMENT,
            paper_context="The paper introduces novel attention mechanisms...",
            review_context="The reviewer argues the methodology is novel...",
            point_id="a1"
        )
    ]

    # 2. Configure evaluator
    config = PaperFaithfulConfig(
        sat_solver="z3",
        enable_sat_validation=True,
        enable_knowledge_base=False
    )

    evaluator = create_paper_faithful_evaluator(config)

    # 3. Evaluate review points
    print("Evaluating review points...")
    results = evaluator.evaluate_batch(review_points)

    for result in results:
        print(f"  {result.review_point.point_id}: {result.base_score}/5.0 "
              f"(Misinformed: {result.is_misinformed})")

    # 4. Aggregate into paper result
    print("\nAggregating paper results...")
    aggregator = create_paper_review_aggregator()
    paper_result = aggregator.aggregate_paper_review(
        paper_id="example_paper",
        review_point_results=results,
        paper_title="Example Research Paper"
    )

    print(f"Overall Quality: {paper_result.summary.overall_quality.value}")
    print(f"Quality Score: {paper_result.summary.quality_score}/5.0")
    print(f"Misinformed Rate: {paper_result.summary.misinformed_rate:.1%}")

    # 5. Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = create_visualizer()

    # Score distribution
    fig1 = visualizer.plot_review_score_distribution(
        results,
        save_path="example_scores.png"
    )
    print("Saved: example_scores.png")

    # Paper summary
    fig2 = visualizer.plot_paper_review_summary(
        paper_result,
        save_path="example_paper_summary.png"
    )
    print("Saved: example_paper_summary.png")

    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
```

## Next Steps

Now that you have the basics:

1. **[Configuration Guide](configuration.md)** - Learn about advanced configuration options
2. **[Core Concepts](core-concepts/overview.md)** - Understand the system architecture
3. **[Examples](examples/basic-usage.md)** - See more detailed examples
4. **[API Reference](api-reference/core.md)** - Explore the complete API
