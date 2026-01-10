# Package Structure

How Medea is organized for pip installation.

## Structure

```
Medea/
├── medea/              # pip-installable package
│   ├── __init__.py        # Public API
│   ├── core.py            # Core functions (medea, experiment_analysis, etc.)
│   ├── agents/            # Agent implementations
│   └── tool_space/        # Tools and utilities
│
├── examples/              # Usage examples
├── evaluation/            # Benchmarks (not in package)
├── main.py                # Evaluation CLI
├── pyproject.toml         # Package config
└── README.md
```

## Usage Modes

**As a library** (recommended):
```python
from medea import medea, AgentLLM
```

**For evaluation** (development):
```bash
python main.py --task targetID --disease ra
```

## Public API

```python
# Core functions
from medea import medea, experiment_analysis, literature_reasoning

# Agents
from medea import ResearchPlanning, Analysis, LiteratureReasoning

# LLM
from medea import AgentLLM, LLMConfig

# Actions (if you need custom agents)
from medea import ResearchPlanDraft, CodeGenerator, LiteratureSearch

# Utils
from medea import Proposal, CodeSnippet, multi_round_discussion
```

## Installation

```bash
pip install -e .  # Editable mode
```

## Benefits

- ✅ Clean imports: `from medea import medea`
- ✅ No code duplication
- ✅ Works with existing evaluation code
- ✅ Easy to extend

## Development

Edit code in `medea/agents/` or `medea/tool_space/` - changes apply immediately in editable mode.

## Publishing (Future)

```bash
python -m build
python -m twine upload dist/*
```

Then users install with: `pip install medea`
