# Medea Examples

Quick examples to get you started.

## Setup

```bash
# Install
pip install -e .

# Configure (.env file)
MEDEADB_PATH=/path/to/MedeaDB
BACKBONE_LLM=gpt-4o
OPENROUTER_API_KEY=your-key
```

## Run Examples

**Full system**:
```bash
python examples/quickstart.py
```

**Custom workflows**:
```bash
python examples/custom_workflow.py
```

## Quick Snippets

### 1. Full Medea System

```python
from medea import medea, AgentLLM, LLMConfig
from medea import ResearchPlanning, Analysis, LiteratureReasoning

llm = AgentLLM(LLMConfig({"temperature": 0.4}), llm_name="gpt-4o")

result = medea(
    user_instruction="Which gene is best for RA?",
    research_planning_module=ResearchPlanning(llm),
    analysis_module=Analysis(llm),
    literature_module=LiteratureReasoning(llm)
)

print(result['CGRH'])
```

### 2. Research Planning Only

```python
from medea import ResearchPlanning, AgentLLM, LLMConfig
from agentlite.commons import TaskPackage

llm = AgentLLM(LLMConfig({"temperature": 0.4}), llm_name="gpt-4o")
agent = ResearchPlanning(llm)

task = TaskPackage(instruction=str({"user_query": "Your question"}))
result = agent(task)
print(result['proposal_draft'].proposal)
```

### 3. Experiment Only

```python
from medea import experiment_analysis
from medea import ResearchPlanning, Analysis, AgentLLM, LLMConfig

llm = AgentLLM(LLMConfig({"temperature": 0.4}), llm_name="gpt-4o")

plan, result = experiment_analysis(
    query="Your question",
    research_planning_module=ResearchPlanning(llm),
    analysis_module=Analysis(llm)
)
```

## Customization

**Temperature**:
```python
LLMConfig({"temperature": 0.2})  # Focused
LLMConfig({"temperature": 0.7})  # Creative
```

**Different models**:
```python
research_llm = AgentLLM(LLMConfig({"temperature": 0.3}), llm_name="gpt-4o")
analysis_llm = AgentLLM(LLMConfig({"temperature": 0.5}), llm_name="claude")
```

**Max iterations**:
```python
from medea import IntegrityVerification, AnalysisQualityChecker

IntegrityVerification(max_iter=3)  # Research plan quality
AnalysisQualityChecker(max_iter=2)      # Code quality
```

## Troubleshooting

**Import error**: `pip install -e .`

**Missing MedeaDB**: `huggingface-cli download psui3905/MedeaDB --local-dir ./MedeaDB`

**API key**: Check `.env` file exists with correct format

## More Info

- [Quickstart Guide](../docs/QUICKSTART.md)
- [Main README](../README.md)
- [Package Structure](../docs/PACKAGE_STRUCTURE.md)
