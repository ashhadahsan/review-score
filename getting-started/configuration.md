# Configuration

Learn how to configure ReviewScore for your specific needs.

## Basic Configuration

### Model Configuration

```python
from reviewscore.model_evaluation import ModelConfig

# OpenAI configuration
openai_config = ModelConfig(
    model_name="gpt-4o",
    temperature=0.1,
    max_tokens=1000,
    api_key="your-openai-key"
)

# Anthropic configuration
anthropic_config = ModelConfig(
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.1,
    max_tokens=1000,
    api_key="your-anthropic-key"
)

# Google configuration
google_config = ModelConfig(
    model_name="gemini-2.5-flash",
    temperature=0.1,
    max_tokens=1000,
    api_key="your-google-key"
)
```

### Paper Faithful Configuration

```python
from reviewscore.paper_faithful import PaperFaithfulConfig

config = PaperFaithfulConfig(
    # Model settings
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.1,
    max_tokens=1000,

    # SAT solver settings
    sat_solver="z3",  # "z3", "pysat", or "simple"
    enable_sat_validation=True,

    # Knowledge base settings
    enable_knowledge_base=True,
    knowledge_base=None,  # Will be set when needed

    # Evaluation settings
    confidence_threshold=0.7,
    enable_human_annotation=False,

    # Advanced settings
    custom_prompts=None,
    evaluation_metadata={}
)
```

## Advanced Configuration

### Custom Prompts

```python
from reviewscore.paper_faithful import PaperSpecificPrompts

custom_prompts = PaperSpecificPrompts(
    question_evaluation_prompt="""
    You are evaluating a question in an academic paper review.

    Question: {question_text}
    Paper Context: {paper_context}
    Review Context: {review_context}

    Rate the quality of this question on a scale of 1-5.
    Consider: clarity, relevance, depth, and appropriateness.
    """,

    claim_evaluation_prompt="""
    You are evaluating a claim in an academic paper review.

    Claim: {claim_text}
    Paper Context: {paper_context}
    Review Context: {review_context}

    Rate the validity and quality of this claim on a scale of 1-5.
    Consider: accuracy, evidence, reasoning, and fairness.
    """,

    argument_evaluation_prompt="""
    You are evaluating an argument in an academic paper review.

    Argument: {argument_text}
    Paper Context: {paper_context}
    Review Context: {review_context}

    Rate the strength and quality of this argument on a scale of 1-5.
    Consider: logic, evidence, reasoning, and persuasiveness.
    """
)

config = PaperFaithfulConfig(
    custom_prompts=custom_prompts
)
```

### SAT Solver Configuration

```python
# Z3 Solver (recommended for complex arguments)
config = PaperFaithfulConfig(
    sat_solver="z3",
    enable_sat_validation=True,
    sat_timeout=30,  # seconds
    sat_verbose=False
)

# PySAT Solver (faster for simple arguments)
config = PaperFaithfulConfig(
    sat_solver="pysat",
    enable_sat_validation=True,
    sat_timeout=10,  # seconds
    sat_verbose=False
)

# Simple Solver (fastest, least accurate)
config = PaperFaithfulConfig(
    sat_solver="simple",
    enable_sat_validation=True
)
```

### Knowledge Base Configuration

```python
from reviewscore.pdf_knowledge_base import create_pdf_knowledge_base

# Single PDF knowledge base
kb = create_pdf_knowledge_base(
    pdf_path="paper.pdf",
    chunk_size=1000,
    chunk_overlap=200
)

# Multi-PDF knowledge base
from reviewscore.pdf_knowledge_base import create_multi_pdf_knowledge_base

kb = create_multi_pdf_knowledge_base(
    pdf_paths=["paper1.pdf", "paper2.pdf", "paper3.pdf"],
    chunk_size=1000,
    chunk_overlap=200
)

config = PaperFaithfulConfig(
    knowledge_base=kb,
    enable_knowledge_base=True
)
```

### Human-in-the-Loop Configuration

```python
from reviewscore.langgraph_agents import create_langgraph_agent

# Enable human-in-the-loop
agent = create_langgraph_agent(
    model_name="claude-3-5-sonnet-20241022",
    enable_human_in_loop=True,
    human_input_threshold=0.5,  # Request human input if confidence < 0.5
    checkpoint_dir="./checkpoints"  # Directory for saving state
)

# Configure when human input is requested
def should_request_human_input(result):
    return (
        result.confidence < 0.5 or
        result.base_score < 2.0 or
        result.is_misinformed
    )
```

## Environment Variables

### Required API Keys

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### Optional Configuration

```bash
# Default model settings
DEFAULT_MODEL=claude-3-5-sonnet-20241022
DEFAULT_TEMPERATURE=0.1
DEFAULT_MAX_TOKENS=1000

# SAT solver settings
DEFAULT_SAT_SOLVER=z3
SAT_TIMEOUT=30

# Knowledge base settings
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200

# Visualization settings
DEFAULT_PLOT_STYLE=whitegrid
DEFAULT_FIGURE_SIZE=12,8
```

## Configuration Files

### YAML Configuration

Create a `config.yaml` file:

```yaml
models:
  openai:
    model_name: "gpt-4o"
    temperature: 0.1
    max_tokens: 1000

  anthropic:
    model_name: "claude-3-5-sonnet-20241022"
    temperature: 0.1
    max_tokens: 1000

  google:
    model_name: "gemini-2.5-flash"
    temperature: 0.1
    max_tokens: 1000

sat_solver:
  solver: "z3"
  timeout: 30
  enable_validation: true

knowledge_base:
  enable: true
  chunk_size: 1000
  chunk_overlap: 200

evaluation:
  confidence_threshold: 0.7
  enable_human_annotation: false

visualization:
  style: "whitegrid"
  figure_size: [12, 8]
  dpi: 300
```

### Loading Configuration

```python
import yaml
from reviewscore.paper_faithful import PaperFaithfulConfig

# Load from YAML
with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)

config = PaperFaithfulConfig(
    model_name=config_data["models"]["anthropic"]["model_name"],
    temperature=config_data["models"]["anthropic"]["temperature"],
    max_tokens=config_data["models"]["anthropic"]["max_tokens"],
    sat_solver=config_data["sat_solver"]["solver"],
    enable_sat_validation=config_data["sat_solver"]["enable_validation"]
)
```

## Performance Tuning

### Model Selection

```python
# For speed (faster, less accurate)
config = PaperFaithfulConfig(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=500
)

# For accuracy (slower, more accurate)
config = PaperFaithfulConfig(
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.1,
    max_tokens=2000
)

# For cost-effectiveness
config = PaperFaithfulConfig(
    model_name="gemini-2.5-flash",
    temperature=0.1,
    max_tokens=1000
)
```

### Batch Processing

```python
# Process multiple review points efficiently
evaluator = create_paper_faithful_evaluator(config)

# Batch evaluation (recommended)
results = evaluator.evaluate_batch(review_points)

# Parallel processing
from concurrent.futures import ThreadPoolExecutor

def evaluate_single(point):
    return evaluator.evaluate_review_point(point)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(evaluate_single, review_points))
```

### Caching

```python
# Enable result caching
config = PaperFaithfulConfig(
    enable_caching=True,
    cache_dir="./cache",
    cache_ttl=3600  # 1 hour
)
```

## Troubleshooting Configuration

### Common Issues

#### 1. API Key Errors

```python
import os
from reviewscore.model_evaluation import create_model_evaluation_system

# Check API keys
required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    print(f"Missing API keys: {missing_keys}")
```

#### 2. SAT Solver Issues

```python
# Test SAT solver availability
try:
    import z3
    print("Z3 solver available")
except ImportError:
    print("Z3 solver not installed. Install with: pip install z3-solver")

try:
    from pysat.solvers import Glucose3
    print("PySAT solver available")
except ImportError:
    print("PySAT solver not installed. Install with: pip install python-sat")
```

#### 3. Memory Issues

```python
# Reduce memory usage
config = PaperFaithfulConfig(
    max_tokens=500,  # Reduce token limit
    enable_caching=False,  # Disable caching
    batch_size=1  # Process one at a time
)
```

### Configuration Validation

```python
from reviewscore.paper_faithful import PaperFaithfulConfig

# Validate configuration
try:
    config = PaperFaithfulConfig(
        model_name="invalid-model",
        sat_solver="invalid-solver"
    )
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Best Practices

### 1. Model Selection

- **Speed**: Use `gpt-3.5-turbo` or `gemini-2.5-flash`
- **Accuracy**: Use `claude-3-5-sonnet-20241022` or `gpt-4o`
- **Cost**: Use `gemini-2.5-flash` for cost-effective evaluation

### 2. SAT Solver Selection

- **Complex Arguments**: Use `z3` solver
- **Simple Arguments**: Use `pysat` or `simple` solver
- **Speed**: Use `simple` solver

### 3. Knowledge Base Usage

- **Single Paper**: Use `create_pdf_knowledge_base`
- **Multiple Papers**: Use `create_multi_pdf_knowledge_base`
- **Large Documents**: Increase `chunk_size` and `chunk_overlap`

### 4. Performance Optimization

- Use batch processing for multiple review points
- Enable caching for repeated evaluations
- Use appropriate model for your accuracy needs
- Consider parallel processing for large datasets

## Next Steps

- **[Core Concepts](core-concepts/overview.md)** - Understand the system architecture
- **[Examples](examples/basic-usage.md)** - See configuration in action
- **[API Reference](api-reference/core.md)** - Explore all configuration options
