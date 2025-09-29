# Basic Usage Examples

Learn how to use ReviewScore with practical examples.

## Simple Evaluation

### Single Review Point

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
print(f"Reasoning: {result.reasoning}")
```

### Multiple Review Points

```python
# Create multiple review points
review_points = [
    create_review_point(
        text="What methodology was used?",
        point_type=ReviewPointType.QUESTION,
        paper_context="Paper content...",
        review_context="Review content...",
        point_id="q1"
    ),
    create_review_point(
        text="The experimental results are not convincing.",
        point_type=ReviewPointType.CLAIM,
        paper_context="Paper content...",
        review_context="Review content...",
        point_id="c1"
    ),
    create_review_point(
        text="The methodology is novel because it introduces attention mechanisms.",
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

## Paper-Level Analysis

### Aggregate Results

```python
from reviewscore.paper_review_result import create_paper_review_aggregator

# Aggregate results into paper-level analysis
aggregator = create_paper_review_aggregator()
paper_result = aggregator.aggregate_paper_review(
    paper_id="paper_123",
    review_point_results=results,
    paper_title="Sample Research Paper"
)

# Display paper summary
summary = paper_result.summary
print(f"Overall Quality: {summary.overall_quality.value}")
print(f"Quality Score: {summary.quality_score}/5.0")
print(f"Total Review Points: {summary.total_review_points}")
print(f"Misinformed Points: {summary.total_misinformed}")
print(f"Misinformed Rate: {summary.misinformed_rate:.1%}")

# Display recommendations
print("\nRecommendations:")
for i, rec in enumerate(paper_result.recommendations, 1):
    print(f"{i}. {rec}")
```

### Quality Analysis

```python
# Analyze quality by review point type
questions = [r for r in results if r.review_point.point_type == ReviewPointType.QUESTION]
claims = [r for r in results if r.review_point.point_type == ReviewPointType.CLAIM]
arguments = [r for r in results if r.review_point.point_type == ReviewPointType.ARGUMENT]

print("Quality by Type:")
print(f"Questions: {len(questions)} points, avg score: {sum(r.base_score for r in questions)/len(questions):.2f}")
print(f"Claims: {len(claims)} points, avg score: {sum(r.base_score for r in claims)/len(claims):.2f}")
print(f"Arguments: {len(arguments)} points, avg score: {sum(r.base_score for r in arguments)/len(arguments):.2f}")
```

## Model Comparison

### Compare Different Models

```python
from reviewscore.model_evaluation import create_model_evaluation_system

# Create evaluators for different models
models = {
    "GPT-4o": create_model_evaluation_system("gpt-4o"),
    "Claude-3.5-Sonnet": create_model_evaluation_system("claude-3-5-sonnet-20241022"),
    "Gemini-2.5-Flash": create_model_evaluation_system("gemini-2.5-flash")
}

# Evaluate with each model
model_results = {}
for model_name, evaluator in models.items():
    results = evaluator.evaluate_batch(review_points)
    model_results[model_name] = results
    
    avg_score = sum(r.base_score for r in results) / len(results)
    print(f"{model_name}: Average score = {avg_score:.2f}")
```

### Model Performance Analysis

```python
# Analyze model performance
for model_name, results in model_results.items():
    scores = [r.base_score for r in results]
    misinformed = [r.is_misinformed for r in results]
    confidence = [r.confidence for r in results]
    
    print(f"\n{model_name} Performance:")
    print(f"  Average Score: {sum(scores)/len(scores):.2f}")
    print(f"  Misinformed Rate: {sum(misinformed)/len(misinformed)*100:.1f}%")
    print(f"  Average Confidence: {sum(confidence)/len(confidence):.2f}")
    print(f"  Score Range: {min(scores):.2f} - {max(scores):.2f}")
```

## SAT Solver Integration

### Argument Validation

```python
from reviewscore.paper_faithful import PaperFaithfulConfig

# Configure with SAT solver
config = PaperFaithfulConfig(
    sat_solver="z3",  # Use Z3 solver
    enable_sat_validation=True
)

evaluator = create_paper_faithful_evaluator(config)

# Evaluate argument
argument = create_review_point(
    text="The methodology is novel because it introduces attention mechanisms.",
    point_type=ReviewPointType.ARGUMENT,
    paper_context="The paper introduces novel attention mechanisms...",
    review_context="The reviewer argues the methodology is novel...",
    point_id="a1"
)

result = evaluator.evaluate_review_point(argument)
print(f"Argument Score: {result.base_score}/5.0")
print(f"SAT Validation: {result.sat_validation_result}")
```

### Different SAT Solvers

```python
# Test different SAT solvers
solvers = ["z3", "pysat", "simple"]

for solver in solvers:
    config = PaperFaithfulConfig(
        sat_solver=solver,
        enable_sat_validation=True
    )
    
    evaluator = create_paper_faithful_evaluator(config)
    result = evaluator.evaluate_review_point(argument)
    
    print(f"{solver.upper()} Solver:")
    print(f"  Score: {result.base_score}/5.0")
    print(f"  Validation: {result.sat_validation_result}")
    print(f"  Time: {result.evaluation_metadata.get('sat_time', 'N/A')}s")
```

## Human-in-the-Loop Workflows

### Basic Human-in-the-Loop

```python
from reviewscore.langgraph_agents import create_langgraph_agent

# Create agent with human-in-the-loop
agent = create_langgraph_agent(
    model_name="claude-3-5-sonnet-20241022",
    enable_human_in_loop=True
)

# Evaluate with potential human intervention
result = agent.evaluate_review_point(question)

if result.requires_human_input:
    print("Human input required!")
    print(f"AI Score: {result.base_score}/5.0")
    print(f"AI Reasoning: {result.reasoning}")
    
    human_response = input("Please provide your annotation: ")
    result = agent.resume_with_human_input(human_response)
    
    print(f"Final Score: {result.base_score}/5.0")
    print(f"Human Annotation: {result.human_annotation}")
```

### Batch Human-in-the-Loop

```python
# Process multiple points with human intervention
results = []
for point in review_points:
    result = agent.evaluate_review_point(point)
    
    if result.requires_human_input:
        print(f"\nHuman input needed for {point.point_id}")
        print(f"Text: {point.text}")
        print(f"AI Score: {result.base_score}/5.0")
        
        human_response = input("Your annotation: ")
        result = agent.resume_with_human_input(human_response)
    
    results.append(result)
    print(f"Completed {point.point_id}: {result.base_score}/5.0")
```

## PDF Knowledge Base Integration

### Single PDF Knowledge Base

```python
from reviewscore.pdf_knowledge_base import create_pdf_knowledge_base
from reviewscore.paper_faithful import PaperFaithfulConfig

# Create knowledge base from PDF
kb = create_pdf_knowledge_base("paper.pdf")

# Use in evaluation
config = PaperFaithfulConfig(
    knowledge_base=kb,
    enable_knowledge_base=True
)

evaluator = create_paper_faithful_evaluator(config)

# Evaluate with PDF context
result = evaluator.evaluate_review_point(question)
print(f"Score with PDF context: {result.base_score}/5.0")
```

### Multi-PDF Knowledge Base

```python
from reviewscore.pdf_knowledge_base import create_multi_pdf_knowledge_base

# Create knowledge base from multiple PDFs
kb = create_multi_pdf_knowledge_base([
    "paper1.pdf",
    "paper2.pdf", 
    "paper3.pdf"
])

# Use in evaluation
config = PaperFaithfulConfig(
    knowledge_base=kb,
    enable_knowledge_base=True
)

evaluator = create_paper_faithful_evaluator(config)
```

### Knowledge Base Search

```python
# Search knowledge base
search_results = kb.search("transformer architecture", top_k=5)

print("Search Results:")
for i, result in enumerate(search_results, 1):
    print(f"{i}. {result['content'][:100]}...")
    print(f"   Relevance: {result['relevance']:.3f}")
    print(f"   Source: {result['source']}")
```

## Visualization Examples

### Basic Plotting

```python
from reviewscore.visualization import create_visualizer

# Create visualizer
visualizer = create_visualizer()

# Plot score distribution
fig1 = visualizer.plot_review_score_distribution(
    results,
    save_path="score_distribution.png"
)

# Plot model comparison
model_results = {
    "GPT-4o": gpt_results,
    "Claude-3.5-Sonnet": claude_results,
    "Gemini-2.5-Flash": gemini_results
}

fig2 = visualizer.plot_model_comparison(
    model_results,
    save_path="model_comparison.png"
)
```

### Paper Summary Visualization

```python
# Create paper summary plot
fig = visualizer.plot_paper_review_summary(
    paper_result,
    save_path="paper_summary.png"
)

# Create multi-paper dashboard
paper_results = [paper1_result, paper2_result, paper3_result]
fig = visualizer.create_dashboard(
    paper_results,
    save_path="multi_paper_dashboard.png"
)
```

### Custom Visualization

```python
# Custom visualizer settings
visualizer = create_visualizer(
    style="darkgrid",
    figsize=(16, 10),
    dpi=300
)

# Generate custom plots
fig = visualizer.plot_review_score_distribution(
    results,
    title="Custom Score Distribution",
    save_path="custom_plot.png"
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
from reviewscore.pdf_knowledge_base import create_pdf_knowledge_base

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

- **[Paper Analysis](paper-analysis.md)** - Learn about paper-level analysis
- **[Visualization](visualization.md)** - Explore visualization options
- **[PDF Integration](pdf-integration.md)** - Learn about PDF knowledge bases
- **[API Reference](api-reference/core.md)** - Explore the complete API
