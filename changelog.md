# Changelog

All notable changes to ReviewScore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive MkDocs documentation
- Material Design theme for documentation
- API reference with automatic generation
- Interactive examples and tutorials
- Contributing guidelines
- Code of conduct

### Changed

- Improved documentation structure
- Enhanced code examples
- Better navigation and search

## [1.0.0] - 2024-01-15

### Added

- Initial release of ReviewScore
- Core evaluation system with multiple LLM providers
- SAT solver integration (Z3, PySAT, simple)
- Human-in-the-loop workflows with LangGraph
- PDF knowledge base integration
- Comprehensive visualization system
- Paper-level aggregation and analysis
- LangChain and LangGraph agent implementations
- Support for OpenAI, Anthropic, and Google models
- Argument validation and consistency checking
- Multi-paper dashboard and analytics
- Quality assessment and recommendations

### Features

- **Multi-Model Evaluation**: Support for GPT-4o, Claude-3.5-Sonnet, Gemini-2.5-Flash
- **SAT Solver Integration**: Z3, PySAT, and simple solvers for argument validation
- **Human-in-the-Loop**: LangGraph-based workflows with checkpointing
- **PDF Knowledge Base**: Multi-PDF support with content extraction
- **Visualization**: Rich plots and analytics for review analysis
- **Paper Aggregation**: Overall quality assessment and recommendations
- **Agent Workflows**: LangChain and LangGraph agent implementations

### API

- `create_review_point()`: Create review points for evaluation
- `create_paper_faithful_evaluator()`: Main evaluation system
- `create_langgraph_agent()`: LangGraph agents with human-in-the-loop
- `create_langchain_agent()`: LangChain ReAct agents
- `create_visualizer()`: Visualization and plotting system
- `create_pdf_knowledge_base()`: PDF knowledge base creation

### Documentation

- Comprehensive README with examples
- API documentation with type hints
- Usage examples and tutorials
- Installation and configuration guides
- Contributing guidelines

## [0.9.0] - 2024-01-10

### Added

- Initial implementation of core functionality
- Basic LLM evaluation system
- Simple SAT solver integration
- Basic visualization capabilities
- Test suite and examples

### Changed

- Refactored core architecture
- Improved error handling
- Enhanced documentation

## [0.8.0] - 2024-01-05

### Added

- LangGraph workflow implementation
- Human-in-the-loop functionality
- PDF knowledge base support
- Advanced visualization features
- Paper-level aggregation

### Changed

- Updated to latest LangGraph version
- Improved SAT solver integration
- Enhanced visualization system

## [0.7.0] - 2024-01-01

### Added

- LangChain agent implementation
- ReAct pattern for agent workflows
- Tool integration for evaluation
- Batch processing capabilities

### Changed

- Refactored agent architecture
- Improved workflow management
- Enhanced error handling

## [0.6.0] - 2023-12-25

### Added

- PDF processing and knowledge base
- Multi-PDF support
- Content extraction and chunking
- Knowledge base search functionality

### Changed

- Updated PDF processing libraries
- Improved content extraction
- Enhanced search capabilities

## [0.5.0] - 2023-12-20

### Added

- Visualization system
- Plotting and analytics
- Score distribution analysis
- Model comparison charts
- Temporal analysis plots

### Changed

- Refactored visualization architecture
- Improved plot quality and styling
- Enhanced analytics capabilities

## [0.4.0] - 2023-12-15

### Added

- Paper-level aggregation
- Quality assessment system
- Recommendation generation
- Overall quality classification

### Changed

- Improved aggregation algorithms
- Enhanced quality metrics
- Better recommendation system

## [0.3.0] - 2023-12-10

### Added

- SAT solver integration
- Argument validation
- Consistency checking
- Multiple solver backends

### Changed

- Refactored SAT solver architecture
- Improved validation algorithms
- Enhanced error handling

## [0.2.0] - 2023-12-05

### Added

- Multi-model evaluation system
- OpenAI, Anthropic, Google model support
- Model configuration and management
- Batch evaluation capabilities

### Changed

- Refactored model architecture
- Improved API design
- Enhanced error handling

## [0.1.0] - 2023-12-01

### Added

- Initial release
- Core evaluation system
- Basic LLM integration
- Simple scoring system
- Basic test suite

### Features

- Review point creation and evaluation
- Basic scoring (1-5 scale)
- Simple misinformed detection
- Basic confidence scoring

## [0.0.1] - 2023-11-25

### Added

- Project initialization
- Basic structure
- Initial documentation
- Development setup

---

## Version History

| Version | Date       | Description                     |
| ------- | ---------- | ------------------------------- |
| 1.0.0   | 2024-01-15 | Initial stable release          |
| 0.9.0   | 2024-01-10 | Pre-release with core features  |
| 0.8.0   | 2024-01-05 | LangGraph and human-in-the-loop |
| 0.7.0   | 2024-01-01 | LangChain agents                |
| 0.6.0   | 2023-12-25 | PDF knowledge base              |
| 0.5.0   | 2023-12-20 | Visualization system            |
| 0.4.0   | 2023-12-15 | Paper aggregation               |
| 0.3.0   | 2023-12-10 | SAT solver integration          |
| 0.2.0   | 2023-12-05 | Multi-model evaluation          |
| 0.1.0   | 2023-12-01 | Core evaluation system          |
| 0.0.1   | 2023-11-25 | Project initialization          |

## Breaking Changes

### Version 1.0.0

- None (initial stable release)

### Version 0.9.0

- Changed `evaluate()` method to `evaluate_review_point()`
- Updated model configuration structure
- Refactored SAT solver interface

### Version 0.8.0

- Updated LangGraph workflow structure
- Changed human-in-the-loop API
- Refactored checkpointing system

### Version 0.7.0

- Updated LangChain agent architecture
- Changed tool integration interface
- Refactored workflow management

## Migration Guide

### From 0.9.0 to 1.0.0

No breaking changes. All existing code should work without modification.

### From 0.8.0 to 0.9.0

```python
# Old
result = evaluator.evaluate(review_point)

# New
result = evaluator.evaluate_review_point(review_point)
```

### From 0.7.0 to 0.8.0

```python
# Old
workflow = create_langgraph_workflow(model_name)

# New
workflow = create_langgraph_flow(model_name)
```

## Deprecations

### Version 1.0.0

- None

### Version 0.9.0

- `evaluate()` method deprecated in favor of `evaluate_review_point()`
- Old model configuration format deprecated

### Version 0.8.0

- Old LangGraph workflow structure deprecated
- Legacy human-in-the-loop API deprecated

## Security

### Version 1.0.0

- Fixed API key exposure in logs
- Enhanced input validation
- Improved error handling

### Version 0.9.0

- Added input sanitization
- Enhanced security for API calls
- Improved error messages

## Performance

### Version 1.0.0

- Optimized batch processing
- Improved memory usage
- Enhanced caching system

### Version 0.9.0

- Faster SAT solver integration
- Optimized model API calls
- Improved visualization rendering

## Dependencies

### Version 1.0.0

- Python 3.8+
- LangChain 0.1.0+
- LangGraph 0.1.0+
- Pydantic 2.0.0+

### Version 0.9.0

- Python 3.8+
- LangChain 0.0.350+
- LangGraph 0.0.50+
- Pydantic 1.10.0+

## Known Issues

### Version 1.0.0

- None

### Version 0.9.0

- SAT solver timeout issues with complex arguments
- Memory usage with large PDF documents
- Visualization rendering on some systems

## Future Plans

### Version 1.1.0 (Planned)

- Enhanced model support
- Improved SAT solver performance
- Advanced visualization features
- Better error handling

### Version 1.2.0 (Planned)

- Multi-language support
- Advanced analytics
- Custom model integration
- Enhanced human-in-the-loop workflows

### Version 2.0.0 (Planned)

- Complete architecture redesign
- Advanced AI integration
- Real-time evaluation
- Cloud deployment support
