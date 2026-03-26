"""
Core Medea agent system implementations.

This module provides the main entry points for running Medea:
- medea(): Full multi-agent system with parallel execution
  Takes user_instruction and experiment_instruction as separate parameters
- experiment_analysis(): Research planning + in-silico experiment
- literature_reasoning(): Literature search and reasoning
"""

import os
import multiprocessing as mp
from typing import Dict, Optional, Any
from agentlite.commons import TaskPackage

# Optional: psutil for better process management
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def experiment_analysis(
    query: str, 
    research_planning_module, 
    analysis_module
) -> tuple:
    """
    Execute research planning and code analysis using the agent system.
    
    Args:
        query: User's research question
        research_planning_module: Agent for generating research plans
        analysis_module: Agent for generating and executing in-silico experiments
        
    Returns:
        Tuple of (research_plan_text, analysis_response)
        
    Example:
        >>> from medea import experiment_analysis, ResearchPlanning, Analysis
        >>> research_planning_module = ResearchPlanning(...)
        >>> analysis_module = Analysis(...)
        >>> plan, result = experiment_analysis(
        ...     "Which gene is the best therapeutic target for RA?",
        ...     research_planning_module,
        ...     analysis_module
        ... )
    """
    from .modules.utils import Proposal
    
    # Generate research plan
    research_plan_task_dict = {"user_query": query}
    research_plan_taskpack = TaskPackage(instruction=str(research_plan_task_dict))
    
    max_agent_retries = int(os.environ.get("AGENT_MAX_RETRIES", "1"))
    research_plan_response = None
    for attempt in range(max_agent_retries):
        try: 
            research_plan_response = research_planning_module(research_plan_taskpack)
            if research_plan_response is not None:
                break
        except Exception as e:
            print(f"Research plan agent call failed (attempt {attempt+1}/{max_agent_retries}): {e}", flush=True)
            if attempt == max_agent_retries - 1:
                return "None", "None"

    # Execute experiment analysis if research plan is valid
    analysis_response, research_plan_text = "None", "None"
    if isinstance(research_plan_response, dict) and isinstance(research_plan_response.get('proposal_draft'), Proposal):
        research_plan_text = research_plan_response['proposal_draft'].proposal

        analysis_task_dict = {"task": query, "instruction": research_plan_text}
        analysis_taskpack = TaskPackage(instruction=str(analysis_task_dict))
        
        for attempt in range(max_agent_retries):
            try:
                analysis_response = analysis_module(analysis_taskpack)
                if analysis_response is not None and analysis_response != "None":
                    break
            except Exception as e:
                print(f"Analysis agent call failed (attempt {attempt+1}/{max_agent_retries}): {e}", flush=True)
                if attempt == max_agent_retries - 1:
                    return research_plan_text, "None"
            
    return research_plan_text, analysis_response


def literature_reasoning(
    query: str, 
    literature_module
) -> Any:
    """
    Execute literature-based reasoning using the agent system.
    
    Args:
        query: User's research question
        literature_module: Agent for literature search and reasoning
        
    Returns:
        Reasoning response from the agent
        
    Example:
        >>> from medea import literature_reasoning, LiteratureReasoning
        >>> agent = LiteratureReasoning(...)
        >>> result = literature_reasoning(
        ...     "What are the therapeutic targets for RA?",
        ...     agent
        ... )
    """
    task_dict = {"user_query": query, "hypothesis": None}
    reason_taskpack = TaskPackage(instruction=str(task_dict))

    max_agent_retries = int(os.environ.get("AGENT_MAX_RETRIES", "1"))
    reasoning_response = "None"
    for attempt in range(max_agent_retries):
        try:
            reasoning_response = literature_module(reason_taskpack)
            if reasoning_response is not None and reasoning_response != "None":
                break
        except Exception as e:
            print(f"Reasoning agent call failed (attempt {attempt+1}/{max_agent_retries}): {e}", flush=True)
        
    return reasoning_response


# ============================================================================
# MULTIPROCESSING WRAPPERS
# ============================================================================

def _experiment_wrapper(inputs_for_coding, coding_result):
    """Wrapper for experiment analysis module in multiprocessing context.
    Runs research planning and analysis in two phases so the research plan
    is saved even if analysis times out."""
    from .modules.utils import Proposal
    
    query, research_planning_module, analysis_module = inputs_for_coding
    
    try:
        # Phase 1: Research planning (save immediately)
        research_plan_task_dict = {"user_query": query}
        research_plan_taskpack = TaskPackage(instruction=str(research_plan_task_dict))
        
        max_retries = int(os.environ.get("AGENT_MAX_RETRIES", "1"))
        research_plan_response = None
        for attempt in range(max_retries):
            try:
                research_plan_response = research_planning_module(research_plan_taskpack)
                if research_plan_response is not None:
                    break
            except Exception as e:
                print(f"Research plan agent call failed (attempt {attempt+1}/{max_retries}): {e}", flush=True)
        
        research_plan_text = "None"
        if isinstance(research_plan_response, dict) and isinstance(research_plan_response.get('proposal_draft'), Proposal):
            research_plan_text = research_plan_response['proposal_draft'].proposal
        
        # Save research plan immediately — survives if analysis times out
        coding_result['research_plan'] = research_plan_text
        
        # Phase 2: Analysis (may be killed by timeout)
        analysis_response = "None"
        if research_plan_text != "None":
            analysis_task_dict = {"task": query, "instruction": research_plan_text}
            analysis_taskpack = TaskPackage(instruction=str(analysis_task_dict))
            
            for attempt in range(max_retries):
                try:
                    analysis_response = analysis_module(analysis_taskpack)
                    if analysis_response is not None and analysis_response != "None":
                        break
                except Exception as e:
                    print(f"Analysis agent call failed (attempt {attempt+1}/{max_retries}): {e}", flush=True)
        
        coding_result['data'] = (research_plan_text, analysis_response)
        coding_result['success'] = True
    except Exception as e:
        print(f"[CODING_PROCESS] Error: {e}", flush=True)
        coding_result['error'] = str(e)
        coding_result['success'] = False


def _reasoning_wrapper(inputs_for_reasoning, reasoning_result):
    """Wrapper for literature reasoning module in multiprocessing context."""
    try:
        result = literature_reasoning(*inputs_for_reasoning)
        reasoning_result['data'] = result
        reasoning_result['success'] = True
    except Exception as e:
        print(f"[REASONING_PROCESS] Error: {e}", flush=True)
        reasoning_result['error'] = str(e)
        reasoning_result['success'] = False


def medea(
    user_instruction: str,
    experiment_instruction: Optional[str] = None,
    research_planning_module = None,
    analysis_module = None,
    literature_module = None,
    debate_rounds: int = 1,
    panelist_llms: list = None,
    include_backbone_llm: bool = True,
    vote_merge: bool = True,
    full_instruction: bool = False,
    timeout: int = 1200
) -> Dict[str, Any]:
    """
    Execute full Medea multi-agent system with parallel execution.
    
    Runs research planning, in-silico experiment, and literature reasoning in parallel,
    then synthesizes results through multi-round panel discussion.

    Args:
        user_instruction: User's research question/instruction
        experiment_instruction: Optional additional experiment context and instructions (default: None)
        research_planning_module: Agent for generating research plans (default: None)
        analysis_module: Agent for in-silico experiment analysis (default: None)
        literature_module: Agent for literature-based reasoning (default: None)
        debate_rounds: Number of panel discussion rounds (default: 2)
        panelist_llms: List of LLM models for panel discussion (default: None)
        include_backbone_llm: Include backbone LLM in panel (default: True)
        vote_merge: Merge similar votes from different panelists (default: True)
        full_instruction: Use full query in panel or user instruction only (default: False)
        timeout: Timeout in seconds for each parallel process (default: 800)
    
    Returns:
        Dictionary containing:
            - 'P': Research plan text
            - 'PA': Analysis response
            - 'R': Literature reasoning response
            - 'final': Panel discussion hypothesis
            - 'llm': Panel LLM responses
            
    Example:
        >>> from medea import medea, AgentLLM, LLMConfig
        >>> from medea import ResearchPlanning, Analysis, LiteratureReasoning
        >>> 
        >>> # Initialize agents
        >>> llm_config = LLMConfig({"temperature": 0.4})
        >>> llm = AgentLLM(llm_config)
        >>> 
        >>> research_planning_module = ResearchPlanning(llm, actions=[...])
        >>> analysis_module = Analysis(llm, actions=[...])
        >>> literature_module = LiteratureReasoning(llm, actions=[...])
        >>> 
        >>> # Run Medea (Option 1: Combined instruction)
        >>> result = medea(
        ...     user_instruction="Which gene is the best therapeutic target for RA in CD4+ T cells?",
        ...     experiment_instruction=None,
        ...     research_planning_module=research_planning_module,
        ...     analysis_module=analysis_module,
        ...     literature_module=literature_module,
        ...     panelist_llms=['gemini-2.5-flash', 'gpt-4o', 'claude']
        ... )
        >>> 
        >>> # Alternative (Option 2: Separate instructions)
        >>> result = medea(
        ...     user_instruction="Which gene is the best therapeutic target for RA?",
        ...     experiment_instruction="in CD4+ T cells",
        ...     research_planning_module=research_planning_module,
        ...     analysis_module=analysis_module,
        ...     literature_module=literature_module
        ... )
        >>> 
        >>> print(result['final'])  # Final hypothesis from panel discussion
    """
    from .modules.discussion import multi_round_discussion
    from .tool_space.agentic_tool import reset_call_budget
    
    # Reset per-sample caches and call budgets
    reset_call_budget()
    
    print(f"\n[MEDEA] Starting parallel execution: Research Planning + In-silico Experiment + Literature Reasoning", flush=True)
    
    # Combine user instruction with experiment instruction for full query
    full_query = user_instruction if experiment_instruction is None else user_instruction + " " + experiment_instruction
    
    # Record the output from all agents
    agent_output_dict = {}
    inputs_for_coding = (full_query, research_planning_module, analysis_module)
    inputs_for_reasoning = (user_instruction, literature_module)

    research_plan_text, analysis_response, literature_response = "None", "None", "None"
    
    # Shared data structures for results
    mgr = mp.Manager()
    analysis_result = mgr.dict()
    literature_result = mgr.dict()
    
    # Start both processes with module-level wrapper functions
    print(f"[MEDEA] Launching in-silico experiment process...", flush=True)
    analysis_process = mp.Process(target=_experiment_wrapper, args=(inputs_for_coding, analysis_result))
    
    print(f"[MEDEA] Launching literature reasoning process...", flush=True)
    literature_process = mp.Process(target=_reasoning_wrapper, args=(inputs_for_reasoning, literature_result))
    
    analysis_process.start()
    print(f"[MEDEA] Experiment analysis process started (PID: {analysis_process.pid})", flush=True)
    
    literature_process.start()
    print(f"[MEDEA] Literature reasoning process started (PID: {literature_process.pid})", flush=True)
    print(f"[MEDEA] Both processes running in parallel...", flush=True)
    
    # Wait for analysis process with timeout
    analysis_process.join(timeout=timeout)
    if analysis_process.is_alive():
        print(f"Error: Analysis task exceeded {timeout}s timeout. Forcefully killing process...", flush=True)
        try:
            if PSUTIL_AVAILABLE:
                parent = psutil.Process(analysis_process.pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                parent.kill()
                analysis_process.join(timeout=5)
            else:
                analysis_process.terminate()
                analysis_process.join(timeout=2)
                if analysis_process.is_alive():
                    analysis_process.kill()
                    analysis_process.join()
        except (psutil.NoSuchProcess if PSUTIL_AVAILABLE else Exception):
            analysis_process.terminate()
            analysis_process.join(timeout=2)
            if analysis_process.is_alive():
                analysis_process.kill()
                analysis_process.join()
    
    # Wait for literature reasoning process with timeout  
    literature_process.join(timeout=timeout)
    if literature_process.is_alive():
        print(f"Error: Literature reasoning task exceeded {timeout}s timeout. Forcefully killing process...", flush=True)
        try:
            if PSUTIL_AVAILABLE:
                parent = psutil.Process(literature_process.pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                parent.kill()
                literature_process.join(timeout=5)
            else:
                literature_process.terminate()
                literature_process.join(timeout=2)
                if literature_process.is_alive():
                    literature_process.kill()
                    literature_process.join()
        except (psutil.NoSuchProcess if PSUTIL_AVAILABLE else Exception):
            literature_process.terminate()
            literature_process.join(timeout=2)
            if literature_process.is_alive():
                literature_process.kill()
                literature_process.join()
    
    # Extract results
    print(f"\n[MEDEA] Collecting results from parallel processes...", flush=True)
    
    if analysis_result.get('success', False):
        research_plan_text, analysis_response = analysis_result['data']
        print(f"[MEDEA] ✓ Analysis process completed successfully", flush=True)
    else:
        # Even if analysis timed out or failed, try to recover the research plan
        if 'research_plan' in analysis_result and analysis_result['research_plan'] != "None":
            research_plan_text = analysis_result['research_plan']
            print(f"[MEDEA] ⚠ Analysis timed out, but research plan was recovered", flush=True)
        elif 'error' in analysis_result:
            print(f"[MEDEA] ✗ Analysis process failed: {analysis_result['error']}", flush=True)
        else:
            print(f"[MEDEA] ⚠ Analysis process: no result", flush=True)
    
    if literature_result.get('success', False):
        literature_response = literature_result['data']
        print(f"[MEDEA] ✓ Literature reasoning process completed successfully", flush=True)
    elif 'error' in literature_result:
        print(f"[MEDEA] ✗ Literature reasoning process failed: {literature_result['error']}", flush=True)
    else:
        print(f"[MEDEA] ⚠ Literature reasoning process: no result", flush=True)

    # Save outputs
    if research_plan_text != "None":
        agent_output_dict['P'] = research_plan_text

    if analysis_response != "None":
        agent_output_dict['PA'] = analysis_response

    if literature_response != "None":
        agent_output_dict['R'] = literature_response

    # Log the generated research plan if available
    if research_plan_text:
        print(f"[Research Plan]: {research_plan_text}\n", flush=True)

    # LLM-based Panel Discussion
    panel_query = user_instruction if not full_instruction else full_query
    
    # Default panelist LLMs if not provided
    if panelist_llms is None:
        import os
        panelist_llms = ['gemini-2.5-flash', 'o3-mini-0131', os.getenv("BACKBONE_LLM", "gpt-4o")]
        
    # Each agent output is assigned an LLM to join panel discussion
    hypothesis_response, llm_hypothesis_response, evidence_grounding = multi_round_discussion(
        query=panel_query,
        include_llm=include_backbone_llm,
        mod='diff_context', 
        panelist_llms=panelist_llms,
        proposal_response=research_plan_text,
        coding_response=analysis_response, 
        reasoning_response=literature_response, 
        vote_merge=vote_merge,
        round=debate_rounds
    )
    
    agent_output_dict['llm'] = llm_hypothesis_response
    agent_output_dict['final'] = hypothesis_response
    agent_output_dict['evidence_grounding'] = evidence_grounding

    return agent_output_dict

