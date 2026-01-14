"""
Medea Quickstart Example

This example demonstrates the simplest way to use Medea for therapeutic target discovery.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Medea components
from medea import (
    medea,
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
    AnalysisQualityChecker,
    # Literature reasoning actions
    LiteratureSearch,
    PaperJudge,
    OpenScholarReasoning,
)


def main():
    """
    Quickstart example: Run Medea on a therapeutic target discovery query.
    """
    
    # ========================================================================
    # Step 1: Configure LLMs
    # ========================================================================
    print("=" * 80)
    print("MEDEA QUICKSTART EXAMPLE")
    print("=" * 80)
    
    temperature = 0.4
    llm_config = LLMConfig({"temperature": temperature})
    
    # Initialize LLM for each agent
    research_plan_llm = AgentLLM(
        llm_config=llm_config,
        llm_name=os.getenv("BACKBONE_LLM", "gpt-4o"),
        verbose=True
    )
    
    analysis_llm = AgentLLM(
        llm_config=llm_config,
        llm_name=os.getenv("BACKBONE_LLM", "gpt-4o"),
        verbose=True
    )
    
    literature_llm = AgentLLM(
        llm_config=llm_config,
        llm_name=os.getenv("BACKBONE_LLM", "gpt-4o"),
        verbose=True
    )
    
    # ========================================================================
    # Step 2: Configure Agent Actions
    # ========================================================================
    print("\n[Setup] Configuring agent actions...")
    
    # Research planning actions
    research_plan_actions = [
        ResearchPlanDraft(tmp=temperature, llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o")),
        ContextVerification(tmp=temperature, llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o")),
        IntegrityVerification(tmp=temperature, llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o"), max_iter=2),
    ]
    
    # Analysis actions
    analysis_actions = [
        CodeGenerator(tmp=temperature, llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o")),
        AnalysisExecution(),
        CodeDebug(tmp=temperature, llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o")),
        AnalysisQualityChecker(tmp=temperature, llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o"), max_iter=2),
    ]
    
    # Literature reasoning actions
    literature_actions = [
        LiteratureSearch(model_name=os.getenv("PAPER_JUDGE_LLM", "gpt-4o"), verbose=False),
        PaperJudge(model_name=os.getenv("PAPER_JUDGE_LLM", "gpt-4o"), verbose=True),
        OpenScholarReasoning(tmp=temperature, llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o"), verbose=True),
    ]
    
    # ========================================================================
    # Step 3: Initialize Agents
    # ========================================================================
    print("[Setup] Initializing modules...")
    
    research_planning_module = ResearchPlanning(
        llm=research_plan_llm,
        actions=research_plan_actions
    )
    
    analysis_module = Analysis(
        llm=analysis_llm,
        actions=analysis_actions
    )
    
    literature_module = LiteratureReasoning(
        llm=literature_llm,
        actions=literature_actions
    )
    
    # ========================================================================
    # Step 4: Define Research Query
    # ========================================================================
    user_instruction = """
    Considering the following gene list: CD79A, MS4A1, IGJ, CD3D, FCER1A, and FCGR3A.
    Which gene is the most potential therapeutic target for rheumatoid arthritis (RA) 
    in CD4+ alpha-beta T cells?
    """
    
    experiment_instruction = None  # Optional: additional experiment context
    
    print(f"\n[Query] {user_instruction.strip()}")
    
    # ========================================================================
    # Step 5: Run Medea
    # ========================================================================
    print("\n[Execution] Running Medea multi-agent system...")
    print("-" * 80)
    
    result = medea(
        user_instruction=user_instruction,
        experiment_instruction=experiment_instruction,
        research_planning_module=research_planning_module,
        analysis_module=analysis_module,
        literature_module=literature_module,
        debate_rounds=2,
        panelist_llms=['gemini-2.5-flash', 'o3-mini-0131', 'gpt-4o'],
        include_backbone_llm=True,
        vote_merge=True,
        timeout=800
    )
    
    # ========================================================================
    # Step 6: Display Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if 'P' in result:
        print("\n[Research Plan]")
        print("-" * 80)
        print(result['P'])
    
    if 'PA' in result:
        print("\n[In-Silico Experiment Result]")
        print("-" * 80)
        print(result['PA'])
    
    if 'R' in result:
        print("\n[Literature Reasoning]")
        print("-" * 80)
        print(result['R'])
    
    if 'final' in result:
        print("\n[Panel Discussion Hypothesis]")
        print("-" * 80)
        print(result['final'])
    
    if 'llm' in result:
        print("\n[LLM Panel Responses]")
        print("-" * 80)
        print(result['llm'])
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

