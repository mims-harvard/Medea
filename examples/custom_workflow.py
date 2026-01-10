"""
Custom Agent System Example

This example demonstrates how to build a custom agent system using
individual Medea components.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Medea components
from medea import (
    experiment_analysis,
    literature_reasoning,
    AgentLLM,
    LLMConfig,
    ResearchPlanning,
    Analysis,
    LiteratureReasoning,
    # Research planning actions
    ResearchPlanDraft,
    ContextVerification,
    IntegrityVerification,
    # Analysis actions
    CodeGenerator,
    AnalysisExecution,
    CodeDebug,
    AnalysisQulityChecker,
    # Literature reasoning actions
    LiteratureSearch,
    PaperJudge,
    OpenScholarReasoning,
)


def example_1_research_planning_only():
    """
    Example 1: Use only the research planning agent.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Research Planning Only")
    print("=" * 80)
    
    from agentlite.commons import TaskPackage
    
    # Configure LLM
    llm_config = LLMConfig({"temperature": 0.4})
    llm = AgentLLM(llm_config, llm_name="gpt-4o", verbose=True)
    
    # Configure actions
    actions = [
        ResearchPlanDraft(tmp=0.4, llm_provider="gpt-4o"),
        ContextVerification(tmp=0.4, llm_provider="gpt-4o"),
        IntegrityVerification(tmp=0.4, llm_provider="gpt-4o", max_iter=2),
    ]
    
    # Initialize agent
    agent = ResearchPlanning(llm=llm, actions=actions)
    
    # Run
    query = "Which gene is the best therapeutic target for rheumatoid arthritis?"
    task_dict = {"user_query": query}
    taskpack = TaskPackage(instruction=str(task_dict))
    
    result = agent(taskpack)
    
    print("\n[Result]")
    if isinstance(result, dict) and 'proposal_draft' in result:
        print(result['proposal_draft'].proposal)
    else:
        print(result)


def example_2_experiment_analysis_system():
    """
    Example 2: Use research planning + in-silico experiment agent system.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Research Planning + In-Silico Experiment")
    print("=" * 80)
    
    # Configure LLMs
    llm_config = LLMConfig({"temperature": 0.4})
    
    research_llm = AgentLLM(llm_config, llm_name="gpt-4o", verbose=True)
    analysis_llm = AgentLLM(llm_config, llm_name="gpt-4o", verbose=True)
    
    # Configure research planning actions
    research_actions = [
        ResearchPlanDraft(tmp=0.4, llm_provider="gpt-4o"),
        ContextVerification(tmp=0.4, llm_provider="gpt-4o"),
        IntegrityVerification(tmp=0.4, llm_provider="gpt-4o", max_iter=2),
    ]
    
    # Configure analysis actions
    analysis_actions = [
        CodeGenerator(tmp=0.4, llm_provider="gpt-4o"),
        AnalysisExecution(),
        CodeDebug(tmp=0.4, llm_provider="gpt-4o"),
        AnalysisQulityChecker(tmp=0.4, llm_provider="gpt-4o", max_iter=2),
    ]
    
    # Initialize agents
    research_planning_module = ResearchPlanning(llm=research_llm, actions=research_actions)
    analysis_module = Analysis(llm=analysis_llm, actions=analysis_actions)
    
    # Run experiment analysis agent system
    query = "Which gene is the best therapeutic target for rheumatoid arthritis in CD4+ T cells?"
    
    plan, result = experiment_analysis(
        query=query,
        research_planning_module=research_planning_module,
        analysis_module=analysis_module
    )
    
    print("\n[Research Plan]")
    print(plan)
    print("\n[Analysis Result]")
    print(result)


def example_3_literature_reasoning_only():
    """
    Example 3: Use only literature reasoning.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Literature Reasoning Only")
    print("=" * 80)
    
    # Configure LLM
    llm_config = LLMConfig({"temperature": 0.4})
    llm = AgentLLM(llm_config, llm_name="gpt-4o", verbose=True)
    
    # Configure actions
    actions = [
        LiteratureSearch(model_name="gpt-4o", verbose=False),
        PaperJudge(model_name="gpt-4o", verbose=True),
        OpenScholarReasoning(tmp=0.4, llm_provider="gpt-4o", verbose=True),
    ]
    
    # Initialize agent
    agent = LiteratureReasoning(llm=llm, actions=actions)
    
    # Run
    query = "What are the latest therapeutic targets for rheumatoid arthritis?"
    
    result = literature_reasoning(
        query=query,
        literature_module=agent
    )
    
    print("\n[Result]")
    print(result)


def example_4_custom_temperature():
    """
    Example 4: Use different temperatures for different modules.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Custom Temperature Settings")
    print("=" * 80)
    
    # Configure LLMs with different temperatures
    research_config = LLMConfig({"temperature": 0.3})  # Lower for planning
    analysis_config = LLMConfig({"temperature": 0.5})   # Higher for code generation
    
    research_llm = AgentLLM(research_config, llm_name="gpt-4o", verbose=True)
    analysis_llm = AgentLLM(analysis_config, llm_name="gpt-4o", verbose=True)
    
    # Configure actions
    research_actions = [
        ResearchPlanDraft(tmp=0.3, llm_provider="gpt-4o"),
        ContextVerification(tmp=0.3, llm_provider="gpt-4o"),
        IntegrityVerification(tmp=0.3, llm_provider="gpt-4o", max_iter=2),
    ]
    
    analysis_actions = [
        CodeGenerator(tmp=0.5, llm_provider="gpt-4o"),
        AnalysisExecution(),
        CodeDebug(tmp=0.5, llm_provider="gpt-4o"),
        AnalysisQulityChecker(tmp=0.5, llm_provider="gpt-4o", max_iter=1),
    ]
    
    # Initialize agents
    research_planning_module = ResearchPlanning(llm=research_llm, actions=research_actions)
    analysis_module = Analysis(llm=analysis_llm, actions=analysis_actions)
    
    # Run
    query = "Which gene is the best therapeutic target for rheumatoid arthritis?"
    
    plan, result = experiment_analysis(
        query=query,
        research_planning_module=research_planning_module,
        analysis_module=analysis_module
    )
    
    print("\n[Research Plan] (temp=0.3)")
    print(plan)
    print("\n[Analysis Result] (temp=0.5)")
    print(result)


def main():
    """Run all examples."""
    
    print("=" * 80)
    print("CUSTOM AGENT SYSTEM EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate how to use individual Medea components")
    print("to build custom agent systems tailored to your specific needs.")
    
    # Uncomment the examples you want to run:
    
    # example_1_research_planning_only()
    # example_2_experiment_analysis_system()
    # example_3_literature_reasoning_only()
    example_4_custom_temperature()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

