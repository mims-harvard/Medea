# Medea Quickstart

Get Medea running in 5 minutes.

## Install

```bash
git clone https://github.com/mims-harvard/Medea.git
cd Medea
pip install -e .
```

## Configure

```bash
cp env_template.txt .env
```

Edit `.env` - set these 3 required variables:

```bash
MEDEADB_PATH=/path/to/MedeaDB
BACKBONE_LLM=gpt-4o
OPENROUTER_API_KEY=your-key-here
```

**Get MedeaDB**: `huggingface-cli download psui3905/MedeaDB --local-dir ./MedeaDB`  
**Get API key**: https://openrouter.ai/keys

## Run

```python
from medea import medea, AgentLLM, LLMConfig
from medea import ResearchPlanning, Analysis, LiteratureReasoning

# Setup LLM
llm = AgentLLM(LLMConfig({"temperature": 0.4}), llm_name="gpt-4o")

# Create agents (use default actions)
research_planning_module = ResearchPlanning(llm)
analysis_module = Analysis(llm)
literature_module = LiteratureReasoning(llm)

# Run Medea
result = medea(
    user_instruction="Which gene is the best therapeutic target for rheumatoid arthritis?",
    research_planning_module=research_planning_module,
    analysis_module=analysis_module,
    literature_module=literature_module
)

print(result['CGRH'])  # Final answer
```

## Examples

See [examples/](../examples/) for more patterns.

## Config Options

See [env_template.txt](../env_template.txt) for all options.

**Common models**:
- OpenAI: `gpt-4o`, `gpt-4o-mini`, `o1-mini`
- Anthropic: `claude-3.5-sonnet`
- Google: `gemini-2.0-flash`
- DeepSeek: `deepseek-r1:671b`

## Troubleshooting

**Import error?**  
```bash
pip install -e .
uv pip install openai==1.82.1 # Make sure correct version of OpenAI package installed
```

**Missing MedeaDB?**  
```bash
huggingface-cli download psui3905/MedeaDB --local-dir ./MedeaDB
```

**API key not working?**  
Check `.env` file exists and has correct format.

## Next Steps

- Browse [examples/README.md](../examples/README.md)
- Check [PACKAGE_STRUCTURE.md](PACKAGE_STRUCTURE.md) (developers)
- Visit [project website](https://mims-harvard.github.io/Medea/)

