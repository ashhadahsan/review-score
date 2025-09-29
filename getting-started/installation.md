# Installation

This guide will help you install ReviewScore and set up your environment.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for development)

## Installation Methods

### Method 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/ashhadahsan/review-score.git
cd reviewscore

# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"
```

### Method 2: Install from PyPI (Future)

```bash
pip install reviewscore
```

## Dependencies

### Core Dependencies

ReviewScore requires the following core dependencies:

- `langchain>=0.1.0` - LangChain framework
- `langgraph>=0.1.0` - LangGraph workflows
- `pydantic>=2.0.0` - Data validation
- `openai>=1.0.0` - OpenAI API client
- `anthropic>=0.3.0` - Anthropic API client
- `google-generativeai>=0.3.0` - Google AI client

### Optional Dependencies

#### SAT Solver Support

```bash
# For Z3 solver
pip install z3-solver

# For PySAT solver
pip install python-sat
```

#### PDF Knowledge Base

```bash
pip install PyPDF2 pdfplumber PyMuPDF reportlab
```

#### Visualization

```bash
pip install matplotlib seaborn pandas numpy
```

#### All Optional Dependencies

```bash
pip install -e ".[all]"
```

## Environment Setup

### 1. Create Environment File

Create a `.env` file in your project root:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Custom model configurations
DEFAULT_MODEL=claude-3-5-sonnet-20241022
DEFAULT_TEMPERATURE=0.1
DEFAULT_MAX_TOKENS=1000
```

### 2. Load Environment Variables

```python
from dotenv import load_dotenv
load_dotenv()
```

## Verification

### Test Installation

```python
# Test basic imports
from reviewscore import create_review_point, ReviewPointType
from reviewscore.paper_faithful import create_paper_faithful_evaluator

print("ReviewScore installed successfully!")
```

### Test API Keys

```python
import os
from reviewscore.model_evaluation import create_model_evaluation_system

# Test OpenAI
if os.getenv("OPENAI_API_KEY"):
    evaluator = create_model_evaluation_system("gpt-4o")
    print("OpenAI API key configured")

# Test Anthropic
if os.getenv("ANTHROPIC_API_KEY"):
    evaluator = create_model_evaluation_system("claude-3-5-sonnet-20241022")
    print("Anthropic API key configured")

# Test Google
if os.getenv("GOOGLE_API_KEY"):
    evaluator = create_model_evaluation_system("gemini-2.5-flash")
    print("Google API key configured")
```

## Development Setup

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
python -m pytest

# Run specific test modules
python test_paper_faithful.py
python test_dataset.py
```

### Code Quality

```bash
# Format code
black reviewscore/

# Lint code
flake8 reviewscore/

# Type checking
mypy reviewscore/
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Ensure you're in the correct directory
cd /path/to/reviewscore

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 2. API Key Issues

```python
# Check environment variables
import os
print("OpenAI:", bool(os.getenv("OPENAI_API_KEY")))
print("Anthropic:", bool(os.getenv("ANTHROPIC_API_KEY")))
print("Google:", bool(os.getenv("GOOGLE_API_KEY")))
```

#### 3. SAT Solver Issues

```python
# Test SAT solver installation
try:
    import z3
    print("Z3 solver available")
except ImportError:
    print("Z3 solver not installed")

try:
    from pysat.solvers import Glucose3
    print("PySAT solver available")
except ImportError:
    print("PySAT solver not installed")
```

#### 4. PDF Processing Issues

```python
# Test PDF libraries
try:
    import PyPDF2
    print("PyPDF2 available")
except ImportError:
    print("PyPDF2 not installed")

try:
    import pdfplumber
    print("pdfplumber available")
except ImportError:
    print("pdfplumber not installed")
```

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search [GitHub Issues](https://github.com/your-username/reviewscore/issues)
3. Join our [Discord Community](https://discord.gg/reviewscore)
4. Email support: support@reviewscore.ai

## Next Steps

Once you have ReviewScore installed:

1. **[Quick Start Guide](quick-start.md)** - Get up and running in 5 minutes
2. **[Configuration Guide](configuration.md)** - Configure your evaluation settings
3. **[Examples](examples/basic-usage.md)** - See ReviewScore in action
