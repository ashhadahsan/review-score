# Contributing to ReviewScore

We welcome contributions to ReviewScore! This guide will help you get started.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/your-username/reviewscore.git
cd reviewscore
```

### 2. Set Up Development Environment

```bash
# Install in development mode
pip install -e ".[dev]"

# Install documentation dependencies
pip install -r docs-requirements.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Guidelines

### Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Automated checks

```bash
# Format code
black reviewscore/

# Lint code
flake8 reviewscore/

# Type checking
mypy reviewscore/
```

### Testing

```bash
# Run all tests
python -m pytest

# Run specific test modules
python test_paper_faithful.py
python test_dataset.py

# Run with coverage
python -m pytest --cov=reviewscore
```

### Documentation

```bash
# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve

# Deploy documentation
mkdocs gh-deploy
```

## Contribution Types

### Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal steps to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, dependencies
6. **Code**: Minimal code example if applicable

### Feature Requests

When requesting features, please include:

1. **Description**: Clear description of the feature
2. **Use Case**: Why this feature is needed
3. **Proposed Solution**: How you think it should work
4. **Alternatives**: Other solutions you've considered
5. **Additional Context**: Any other relevant information

### Code Contributions

#### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests for new functionality
5. **Update** documentation if needed
6. **Run** all tests and checks
7. **Submit** a pull request

#### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## Code Structure

### Core Modules

- **`core.py`**: Core data structures and functions
- **`paper_faithful.py`**: Paper-faithful implementation
- **`model_evaluation.py`**: LLM evaluation system
- **`argument_reconstruction.py`**: Argument parsing and validation
- **`human_evaluator.py`**: Human evaluation interface

### Workflow Modules

- **`lcel_workflows.py`**: LangChain Expression Language workflows
- **`langgraph_flows.py`**: LangGraph state-based workflows
- **`langchain_agents.py`**: LangChain ReAct agents
- **`langgraph_agents.py`**: LangGraph agents with human-in-the-loop

### Utility Modules

- **`pdf_knowledge_base.py`**: PDF processing and knowledge base
- **`visualization.py`**: Plotting and visualization
- **`paper_review_result.py`**: Paper-level aggregation
- **`evaluation_metrics.py`**: Evaluation metrics and scoring

## Testing Guidelines

### Unit Tests

```python
def test_review_point_creation():
    """Test review point creation."""
    point = create_review_point(
        text="Test question",
        point_type=ReviewPointType.QUESTION,
        paper_context="Test context",
        review_context="Test review",
        point_id="test1"
    )
    
    assert point.text == "Test question"
    assert point.point_type == ReviewPointType.QUESTION
    assert point.point_id == "test1"
```

### Integration Tests

```python
def test_evaluation_workflow():
    """Test complete evaluation workflow."""
    # Create review point
    point = create_review_point(...)
    
    # Create evaluator
    evaluator = create_paper_faithful_evaluator()
    
    # Evaluate
    result = evaluator.evaluate_review_point(point)
    
    # Assertions
    assert result.base_score >= 1.0
    assert result.base_score <= 5.0
    assert isinstance(result.is_misinformed, bool)
```

### Mock Testing

```python
from unittest.mock import Mock, patch

@patch('reviewscore.model_evaluation.OpenAI')
def test_openai_evaluation(mock_openai):
    """Test OpenAI evaluation with mocked API."""
    mock_openai.return_value.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="4.5"))]
    )
    
    evaluator = create_model_evaluation_system("gpt-4o")
    result = evaluator.evaluate_review_point(point)
    
    assert result.base_score == 4.5
```

## Documentation Guidelines

### Code Documentation

```python
def evaluate_review_point(self, review_point: ReviewPoint) -> ReviewScoreResult:
    """
    Evaluate a review point using the configured model and settings.
    
    Args:
        review_point: The review point to evaluate
        
    Returns:
        ReviewScoreResult: Evaluation result with scores and metadata
        
    Raises:
        ValueError: If review point is invalid
        APIError: If model API call fails
        
    Example:
        >>> point = create_review_point(...)
        >>> evaluator = create_paper_faithful_evaluator()
        >>> result = evaluator.evaluate_review_point(point)
        >>> print(result.base_score)
        4.2
    """
```

### API Documentation

Use MkDocs with mkdocstrings for automatic API documentation:

```python
#: This module provides core functionality for ReviewScore.
#: It includes data structures, evaluation methods, and utilities.

class ReviewPoint:
    """Represents a review point in the evaluation system."""
    
    def __init__(self, text: str, point_type: ReviewPointType, ...):
        """
        Initialize a review point.
        
        Args:
            text: The text content of the review point
            point_type: The type of review point (QUESTION, CLAIM, ARGUMENT)
            ...
        """
```

## Release Process

### Version Bumping

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes written
- [ ] Tag created
- [ ] PyPI package built
- [ ] Documentation deployed

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Discord**: Real-time chat and support
- **Email**: support@reviewscore.ai

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributors highlighted
- **Documentation**: Contributors mentioned in relevant sections

## Questions?

If you have questions about contributing:

1. Check existing [GitHub Issues](https://github.com/your-username/reviewscore/issues)
2. Join our [Discord Community](https://discord.gg/reviewscore)
3. Email us: support@reviewscore.ai

Thank you for contributing to ReviewScore! 🎉
