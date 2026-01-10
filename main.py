# ============================================================================
# ENVIRONMENT SETUP (must be first)
# ============================================================================
import dotenv
dotenv.load_dotenv('.env')

# Enable Hugging Face download progress bars and logging
import os
import logging

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'  # Show progress bars
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'     # Show transformers info
os.environ['HF_HUB_VERBOSITY'] = 'info'           # Show hub info


# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================
import argparse
import copy
import multiprocessing as mp
import random
import time
from pathlib import Path

# ============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# ============================================================================
import pandas as pd
from agentlite.commons import TaskPackage

# Optional: psutil for better process management
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Process killing might be less effective.", flush=True)

# ============================================================================
# LOCAL APPLICATION IMPORTS
# ============================================================================
# Agent LLM infrastructure
from medea.modules.agent_llms import LLMConfig, AgentLLM

# Agent implementations
from medea.modules.experiment_analysis import Analysis, CodeDebug, AnalysisExecution, CodeGenerator, AnalysisQulityChecker
from medea.modules.discussion import multi_round_discussion
from medea.modules.literature_reasoning import LiteratureSearch, OpenScholarReasoning, PaperJudge, LiteratureReasoning
from medea.modules.research_planning import ContextVerification, ResearchPlanDraft, IntegrityVerification, ResearchPlanning
from medea.modules.utils import Proposal
from medea.core import medea

# Evaluation and utilities
from evaluation.query_template import *
from medea.tool_space.gpt_utils import chat_completion
from medea.tool_space.env_utils import get_env_with_error, get_seed, get_backbone_llm
from utils import (
    evaluate_prediction,
    input_loader,
    log_acc_dict,
    log_check,
    log_saver,
    split_df_after_checkpoint,
    update_acc_dict
)

# ============================================================================
# HELPER UTILITIES FOR MODEL DOWNLOAD MONITORING
# ============================================================================

def check_model_cache(model_name: str, cache_dir: str = None) -> bool:
    """
    Check if a Hugging Face model is already cached locally.
    
    Args:
        model_name: Model identifier (e.g., 'OpenSciLM/OpenScholar_Reranker')
        cache_dir: Custom cache directory (uses HF_HOME or default if None)
        
    Returns:
        True if model is cached, False if it needs to be downloaded
    """
    if cache_dir is None:
        cache_dir = os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    
    # Check if model directory exists in cache
    model_cache_path = Path(cache_dir) / 'hub' / f"models--{model_name.replace('/', '--')}"
    
    is_cached = model_cache_path.exists()
    if is_cached:
        print(f"[MODEL CACHE] ✓ {model_name} found in cache", flush=True)
    else:
        print(f"[MODEL CACHE] x {model_name} NOT in cache - will download on first use", flush=True)
        print(f"[MODEL CACHE] Download progress bars are enabled (see logs below)", flush=True)
    
    return is_cached


# ============================================================================
# COMMAND-LINE ARGUMENT PARSER
# ============================================================================

def parse_arguments():
    """
    Parse command-line arguments for configuration.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Medea: Multi-Agent Research Planning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (medea, targetID task)
  python main.py
  
  # Run specific task
  python main.py --task targetID --disease ra
  
  # Run with custom settings
  python main.py --setting medea --task targetID --disease t1dm --sample-seed 43
  
  # Synthetic Lethality task with custom cell line
  python main.py --task sl --cell-line CAL27 --sl-source samson
  
  # Immune Therapy task with custom dataset
  python main.py --task immune_response --immune-dataset IMVigor210 --patient-tpm-root /path/to/tpm/data
        """
    )
    
    # General settings
    parser.add_argument('--setting', type=str, default='medea',
                        help='Evaluation setting (default: medea). Options: medea, gpt-4o, o3-mini-0131, deepseek-r1:70b, claude')
    parser.add_argument('--task', type=str, default='targetID',
                        help='Task type (default: targetID). Options: targetID, sl, immune_response')
    parser.add_argument('--sample-seed', type=int, default=42,
                        help='Dataset sampling seed (default: 42)')
    parser.add_argument('--evaluation-folder', type=str, default='./evaluation',
                        help='Path to evaluation data folder (default: ./evaluation)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to resume from (format: "GENE1,GENE2,CELLLINE,TYPE")')
    
    # TargetID task specific
    parser.add_argument('--disease', type=str, default='ra',
                        help='Disease context for targetID task (default: ra). Options: ra, t1dm, ss, blastoma, fl')
    parser.add_argument('--scfm', type=str, default='PINNACLE',
                        help='Single-cell foundation model (default: PINNACLE). Options: PINNACLE, TranscriptFormer')
    
    # Synthetic Lethality task specific
    parser.add_argument('--cell-line', type=str, default='MCF7',
                        help='Cell line for sl task (default: MCF7)')
    parser.add_argument('--sl-source', type=str, default='samson',
                        help='SL data source (default: samson). Options: samson (default)')
    
    # Immune therapy task specific
    parser.add_argument('--immune-dataset', type=str, default='IMVigor210',
                        help='Immune therapy dataset (default: IMVigor210)')
    parser.add_argument('--patient-tpm-root', type=str, default=None,
                        help='Path to patient TPM data (default: from MEDEADB_PATH env var)')
    
    # Agent configuration
    parser.add_argument('--temperature', type=float, default=0.4,
                        help='LLM temperature for all agents (default: 0.4)')
    parser.add_argument('--quality-max-iter', type=int, default=2,
                        help='Max iterations for quality checks (default: 2)')
    parser.add_argument('--code-quality-max-iter', type=int, default=2,
                        help='Max iterations for code quality checks (default: 2)')
    
    # Panel discussion
    parser.add_argument('--debate-rounds', type=int, default=2,
                        help='Number of panel discussion rounds (default: 2)')
    parser.add_argument('--panelists', type=str, nargs='+', default=None,
                        help='LLM models for panel discussion (default: gemini-2.5-flash, o3-mini-0131, BACKBONE_LLM)')
    
    return parser.parse_args()


# ============================================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================================

# Parse arguments (only if running as main script)
args = parse_arguments()

# ============================================================================
# CONFIGURATION: EXPERIMENT SETTINGS
# ============================================================================

# Model identifier mapping for result file naming
setting_naming_dict = {
    'deepseek-r1:70b': 'deepseek70b',
    'deepseek-r1:671b': 'deepseek671b',
    'o1-mini-2024-09-12': 'o1_mini',
    'gpt-5': 'gpt5',
    'gpt-4o': 'gpt4o',
    'o3-mini-0131': 'o3_mini',
    'claude': 'claude',
    'medea': 'medea'
}

# Experiment configurations: {setup_name: {user_template, agent_template, judge_prompt}}
# Results saved to: results/PROMPT_SETTING/setting_naming_dict[SETTING]-context_dict[TASK]-SAMPLE_SEED.csv
experiment_setup_dict = {
    # Target Normination (Hard multi-choice)
    'targetid_gpt4o_multi': {
        'user': target_id_query_temp_5, 
        'agent': target_id_instruction_1,
        'judge_prompt': TARGETID_REASON_CHECK
    },
    'targetid_gpt4o_pinnacle_multi': {
        'user': target_id_query_temp_5, 
        'agent': target_id_instruction_1,
        'judge_prompt': TARGETID_REASON_CHECK
    },
    # TargetID-TranscriptFormer (Hard multi-choice)
    'targetid_gpt4o_transcriptformer_multi': {
        'user': target_id_query_temp_5, 
        'agent': target_id_instruction_1,
        'judge_prompt': TARGETID_REASON_CHECK
    },
    'targetid_claude_pinnacle_multi': {
        'user': target_id_query_temp_5, 
        'agent': target_id_instruction_1,
        'judge_prompt': TARGETID_REASON_CHECK
    },
    'targetid_gpt4o_noInstruction_multi': {
        'user': target_id_query_temp_5, 
        'agent': None,
        'judge_prompt': TARGETID_REASON_CHECK
    },
    # Synthetic Lethality Prediction (Open-ended) [Enrichr only]
    'sl_e2_open_end':  {
        'user': sl_query_lineage_openend,
        'agent': sl_instruction_e2,
        'judge_prompt': SL_REASON_CHECK
    },
    # Synthetic Lethality Prediction (Open-ended) [Enrichr+DepMap]
    'sl_paperFilter_open_end':  {
        'user': sl_query_lineage_openend, 
        'agent': sl_instruction_e5, 
        'judge_prompt': SL_REASON_CHECK 
    },
    'sl_claude_open_end':  {
        'user': sl_query_lineage_openend, 
        'agent': sl_instruction_e5, 
        'judge_prompt': SL_REASON_CHECK 
    },
    # ICI Prediction 
    'immune_gpt4o_tmp1': {
        'user': immune_query_temp_1, 
        'agent':immune_instruction_1,
        'judge_prompt': IMMUNE_ANS_CHECK
    },
    'immune_gpt4o_tmp2': {
        'user': immune_query_temp_2, 
        'agent':immune_instruction_2,
        'judge_prompt': IMMUNE_ANS_CHECK
    },
    'immune_gpt4o_tmp3': {
        'user': immune_query_temp_3, 
        'agent':immune_instruction_3,
        'judge_prompt': IMMUNE_ANS_CHECK
    },
    'immune_gpt4o_tmp4': {
        'user': immune_query_temp_4, 
        'agent':immune_instruction_4,
        'judge_prompt': IMMUNE_ANS_CHECK
    },
    'immune_gpt4o_tmp5': {
        'user': immune_query_temp_5, 
        'agent':immune_instruction_5,
        'judge_prompt': IMMUNE_ANS_CHECK
    },
    'immune_gpt4o_tmp6': {
        'user': immune_query_temp_6, 
        'agent':immune_instruction_6,
        'judge_prompt': IMMUNE_ANS_CHECK
    },
    'immune_claude_tmp4': {
        'user': immune_query_temp_4, 
        'agent':immune_instruction_4,
        'judge_prompt': IMMUNE_ANS_CHECK
    }
}

# ============================================================================
# CONFIGURATION: GENERAL PARAMETERS
# ============================================================================

# Random seeds
try:
    SEED = get_seed(default=42)  # Seed for backbone LLM
except Exception as e:
    print(f"⚠️  Warning: Could not get SEED from environment, using default 42")
    SEED = 42

SAMPLE_SEED = args.sample_seed

# Evaluation settings (from command-line args)
SETTING = args.setting
TASK = args.task
CHECKPOINT = tuple(args.checkpoint.split(',')) if args.checkpoint else None

# Data and logging
QUERY_MODE = True  # True: query input, False: data + template
EVALUATION_FOLDER = args.evaluation_folder
FINAL_LOG = "target-summary"  # Summary log file name
AGENT_OUTPUT_PATH = None  # Path to agent output CSV for panel discussion mode

# LLM configuration
FULL_INSTURCTION = False  # For panel discussion: True = user + experiment instruction, False = user query only
LLM_JUDGE_MODEL = get_env_with_error("EVALUATION_JUDGE_LLM", default=None, required=False)  # Judge LLM for evaluation

# ============================================================================
# CONFIGURATION: PROMPT SELECTION
# ============================================================================

# Select prompt configuration based on task
if TASK == "targetID":
    PROMPT_SETTING = "targetid_gpt4o_multi"
elif TASK == "sl":
    PROMPT_SETTING = "sl_claude_open_end"
else:
    PROMPT_SETTING = "immune_gpt4o_tmp4"

# Load prompt templates from selected configuration
USER_PROMPT = experiment_setup_dict[PROMPT_SETTING]['user']
AGENT_PROMPT = experiment_setup_dict[PROMPT_SETTING]['agent']
LLM_JUDGE_PROMPT = experiment_setup_dict[PROMPT_SETTING]['judge_prompt']


# ============================================================================
# CONFIGURATION: TASK-SPECIFIC DATA PATHS
# ============================================================================

# Task 1: TargetID (from args)
SCFM = args.scfm
DISEASE = args.disease
TARGET_DB_QUERY_PATH = f"targetid-{DISEASE}-query-{SAMPLE_SEED}.csv"
TARGET_DB_PATH = f"targetid-{DISEASE}-{SAMPLE_SEED}.csv"

# Task 2: Synthetic Lethality (from args)
SL_SOURCE = args.sl_source
CELL_LINE = args.cell_line
SL_DB_QUERY_PATH = f"{SL_SOURCE}-{CELL_LINE}-query-{SAMPLE_SEED}.csv"
SL_DB_PATH = f"{SL_SOURCE}-{CELL_LINE}-{SAMPLE_SEED}.csv"

# Task 3: Immune Therapy Response (from args)
IMMUNE_DATASET = args.immune_dataset
IMMUNE_TMP = PROMPT_SETTING.split("_")[-1]
PATIENT_TABLE_QUERY_PATH = f"{IMMUNE_DATASET}-{IMMUNE_TMP}-query-{SAMPLE_SEED}.csv"
if args.patient_tpm_root:
    ICI_PATIENT_TPM_ROOT = args.patient_tpm_root
else:
    medeadb_path = get_env_with_error("MEDEADB_PATH", default=".", required=False)
    ICI_PATIENT_TPM_ROOT = os.path.join(medeadb_path, "immune-compass/patients")
PATIENT_TABLE_PATH = f"{IMMUNE_DATASET}-patient-{SAMPLE_SEED}.csv"

# Context mapping for each task
context_dict = {
    'targetID': DISEASE,
    'sl': CELL_LINE,
    'immune_response': IMMUNE_DATASET
}

# Set patient data path for immune task
PATIENT_TPM_ROOT = ICI_PATIENT_TPM_ROOT if TASK == "immune_response" else None

# ============================================================================
# CONFIGURATION: LOGGING PATHS
# ============================================================================

LOG_ROOT = f"results/{PROMPT_SETTING}"
Path(LOG_ROOT).mkdir(parents=True, exist_ok=True)

# Add suffix if resuming from checkpoint
suffix = "-continue" if CHECKPOINT else ""
TASK_LOG_FILE = f"{LOG_ROOT}/{setting_naming_dict[SETTING]}-{context_dict[TASK]}-{SAMPLE_SEED}{suffix}.csv"

# ============================================================================
# CONFIGURATION: AGENT TEMPERATURES (from args)
# ============================================================================

# Default temperature from args
DEFAULT_TEMPERATURE = args.temperature

# Agent backbone temperatures
research_planning_module_tmp = DEFAULT_TEMPERATURE
analysis_module_tmp = DEFAULT_TEMPERATURE
literature_module_tmp = DEFAULT_TEMPERATURE

# Action temperatures
research_plan_act_tmp = DEFAULT_TEMPERATURE
analysis_act_tmp = DEFAULT_TEMPERATURE
literature_reason_act_tmp = DEFAULT_TEMPERATURE

# ============================================================================
# CONFIGURATION: AGENT ACTION PARAMETERS (from args)
# ============================================================================

# Iteration limits
QUALITY_MAX_ITER = args.quality_max_iter
CODE_QAULITY_MAX_ITER = args.code_quality_max_iter

# Panel discussion settings
DEBATE_ROUND = args.debate_rounds
PANELIST_LLM = args.panelists or ['gemini-2.5-flash', 'o3-mini-0131', get_backbone_llm("gpt-4o")]
INCLUDE_BACKBONE_LLM = True  # Include backbone LLM in panel
VOTE_MERGE = True  # Merge similar votes from different panelists

# ============================================================================
# MAIN TESTING
# ============================================================================

def medea_unittest(df, user_template=None, agent_template=None):
    """
    Run Medea multi-agent system test on evaluation dataset.
    
    Args:
        df: DataFrame containing evaluation queries
        user_template: Template for user query formatting
        agent_template: Template for agent instruction formatting
    """
    
    # The succ count for each agent (p - Proposal, cg - coding, r - reasoning, h - hypothesis)
    success_count = {}
    print("User Template: ", user_template, flush=True)
    print("Agent Template: ", agent_template, flush=True)

    # Initialized the agents and actions
    # Display LLM provider configuration
    llm_provider = get_env_with_error("LLM_PROVIDER_NAME", default="OpenRouter")
    print(f"=== LLM Provider: {llm_provider} ===", flush=True)
    
    backbone_llm = get_backbone_llm("gpt-4o")
    paper_judge_llm = get_env_with_error("PAPER_JUDGE_LLM", default=backbone_llm)
    
    print("=== Init Research Planning Agent Backbone ===", flush=True)
    research_plan_llm_config = LLMConfig({"temperature": research_planning_module_tmp})
    research_plan_llm = AgentLLM(
        llm_config=research_plan_llm_config,
        llm_name=backbone_llm,
        verbose=True
    )
    
    print("=== Init Analysis Agent Backbone ===", flush=True)
    analysis_llm_config = LLMConfig({"temperature": analysis_module_tmp})
    analysis_llm = AgentLLM(
        llm_config=analysis_llm_config,
        llm_name=backbone_llm,
        verbose=True
    )
    
    print("=== Init Literature Reasoning Agent Backbone ===", flush=True)
    literature_reason_llm_config = LLMConfig({"temperature": literature_module_tmp})
    literature_reason_llm = AgentLLM(
        llm_config=literature_reason_llm_config,
        llm_name=backbone_llm,
        verbose=True
    )
    
    print("=== Init Research Planning Actions ===", flush=True)
    research_plan_actions = [
        ResearchPlanDraft(tmp=research_plan_act_tmp, llm_provider=backbone_llm),
        ContextVerification(tmp=research_plan_act_tmp, llm_provider=backbone_llm),
        IntegrityVerification(tmp=research_plan_act_tmp, llm_provider=backbone_llm, max_iter=QUALITY_MAX_ITER),
    ]
    
    print("=== Init Analysis Actions ===", flush=True)
    analysis_actions = [
        CodeGenerator(tmp=analysis_act_tmp, llm_provider=backbone_llm), 
        AnalysisExecution(),
        CodeDebug(tmp=analysis_act_tmp, llm_provider=backbone_llm), 
        AnalysisQulityChecker(tmp=analysis_act_tmp, llm_provider=backbone_llm, max_iter=CODE_QAULITY_MAX_ITER), 
    ]
    
    print("=== Init Literature Reasoning Actions ===", flush=True)
    literature_reason_actions = [
        LiteratureSearch(model_name=paper_judge_llm, verbose=False),
        PaperJudge(model_name=paper_judge_llm, verbose=True),
        OpenScholarReasoning(tmp=DEFAULT_TEMPERATURE, llm_provider=backbone_llm, verbose=True)
    ]
    
    print("LLM Backbone Seed: ", SEED, flush=True)
    print("[Research Plan] Agent Temp: ", research_planning_module_tmp, flush=True)
    print("[Research Plan] Action Temp: ", research_plan_act_tmp, flush=True)
    print("[Research Plan] Quality Iterations: ", QUALITY_MAX_ITER, flush=True)
    print("[Analysis] Agent Temp: ", analysis_module_tmp, flush=True)
    print("[Analysis] Action Temp: ", analysis_act_tmp, flush=True)
    print("[Panel] Panelists: ", PANELIST_LLM, flush=True)
    
    research_planning_module = ResearchPlanning(llm=research_plan_llm, actions=research_plan_actions)
    analysis_module = Analysis(llm=analysis_llm, actions=analysis_actions)
    literature_module = LiteratureReasoning(llm=literature_reason_llm, actions=literature_reason_actions)

    acc_dict = {}
    log_df = pd.DataFrame()
    cg_count, reason_count, hypo_count, gpt_count = 0, 0, 0, 0

    # Load inputs based on configed settings
    paraphraser_llm = get_env_with_error("PARAPHRASER_LLM", default=backbone_llm)
    inputs = input_loader(
        df, 
        task=TASK, 
        rephrase_model=paraphraser_llm, 
        user_template=user_template, 
        agent_template=agent_template, 
        scfm=SCFM, 
        sl_source=SL_SOURCE,
        patient_tpm_root=PATIENT_TPM_ROOT
    )

    # X is the full query (user_instruction + experiment_instruction), user_instruction is just the user question
    for candidate_genes, X, user_instruction, y, *attribute in inputs:
        try:
            # Record the system execution
            start_time = time.time()
            print(f"[User Query]: {X}", flush=True)
            
            # Extract experiment instruction from full query
            experiment_instruction = X[len(user_instruction):].strip() if X.startswith(user_instruction) else None
            
            response = medea(user_instruction, experiment_instruction, research_planning_module, analysis_module, literature_module)
            end_time = time.time()

            # Log check for the query and response
            success_count, executed_output, reason_output, hypo_output, gpt_feedback = log_check(
                task=TASK,
                query=X,
                gpt_query=user_instruction,
                llm_judge_prompt=LLM_JUDGE_PROMPT,
                response_dict=response,
                start_time=start_time,
                end_time=end_time,
                success_count=success_count,
                gene_list=candidate_genes,
                llm_judge=LLM_JUDGE_MODEL
            )
            
            # Save log
            log_pack = [executed_output, reason_output, hypo_output, gpt_feedback]
            log_df = log_saver(
                log_df, 
                "medea", 
                TASK, 
                candidate_genes, 
                y, 
                log_pack,
                TASK_LOG_FILE,
                *attribute
            )

            # Evaluate the prediction and update metrics
            acc_dict, cg_count, reason_count, hypo_count, gpt_count = evaluate_prediction(
                TASK,
                y,
                acc_dict,
                executed_output,
                reason_output,
                hypo_output,
                gpt_feedback,
                success_count,
                cg_count,
                reason_count,
                hypo_count,
                gpt_count,
                *attribute
            )
        except Exception as e:
            print(f"Agent Error: {e}", flush=True)
            if "Incapsula Resource Error" in str(e):
                print("[medea_unittest] Incapsula Resource Error. Agent terminated.", flush=True)
                break
            continue
        
    log_acc_dict(acc_dict, FINAL_LOG, None)


def llm_unitest(df, mod='llm', agent_output_df=None, model=None, user_template=None, agent_template=None, attempts=3):
    """
    Run LLM-only or panel discussion test on evaluation dataset.
    
    Args:
        df: DataFrame containing evaluation queries
        mod: Mode - 'llm' for direct LLM, 'multi_round_discussion' for panel
        agent_output_df: Agent outputs for panel discussion (required if mod='multi_round_discussion')
        model: LLM model identifier
        user_template: Template for user query formatting
        agent_template: Template for agent instruction formatting
        attempts: Number of retry attempts for LLM calls
    """
    count, success_count = 0, {}
    acc_dict = {} 
    log_df = pd.DataFrame()

    if mod == 'multi_round_discussion':
        assert agent_output_df is not None, "Agent output dataframe required for panel discussion mode"

    print("User Template: ", user_template, flush=True)
    print("Agent Template: ", agent_template, flush=True)
    # Load inputs based on configed settings
    backbone_llm = get_backbone_llm("gpt-4o")
    inputs = input_loader(
        df, 
        task=TASK, 
        rephrase_model=backbone_llm,
        user_template=user_template, 
        agent_template=agent_template,
        scfm=SCFM, 
        sl_source=SL_SOURCE,
        patient_tpm_root=PATIENT_TPM_ROOT
    )

    for candidate_genes, _, user_instruction, y, *attribute in inputs:
        i = 0
        y = y.strip("['']").lower()
        
        if mod == "llm":
            assert model != None, "Model must not be None."
            
            response = None
            while i < attempts:
                try:
                    response = chat_completion(user_instruction, model=model)
                    break  # exit loop if successful
                except Exception as e:
                    print(f"Attempt {i}/{attempts} failed: {e}")
                    i += 1
            
            if response is None:
                response = {'llm': "ERROR: All attempts failed"}
            else:
                response = {'llm': response}


        elif mod == 'multi_round_discussion':
            # Get the agent output from the dataframe
            agent_output = agent_output_df.loc[i]
            research_plan_text = agent_output['proposal']  # Column name stays 'proposal' for compatibility
            code_snippet = agent_output['code_snippet']
            executed_output = agent_output['executed_output']
            reasoning_output = agent_output['reasoning_output']

            # Formulate agent output for panel discussion
            analysis_response_dict = {
                'code_snippet': code_snippet,
                'executed_output': executed_output
            }
            literature_response_dict = {
                'user_query': reasoning_output
            }
            
            hypothesis_response, llm_hypothesis_response = multi_round_discussion(
                query=user_instruction, 
                include_llm=INCLUDE_BACKBONE_LLM,
                mod='diff_context', 
                panelist_llms=PANELIST_LLM,
                proposal_response=research_plan_text, 
                coding_response=analysis_response_dict, 
                reasoning_response=literature_response_dict, 
                vote_merge=VOTE_MERGE,
                round=DEBATE_ROUND
            )
            response = {
                'llm': llm_hypothesis_response,
                'CGRH': hypothesis_response
            }
        
        if i == attempts: continue

        success_count, _, _, hypo_output, llm_feedback = log_check(
            task=TASK,
            query=user_instruction,
            gpt_query=user_instruction,
            llm_judge_prompt=LLM_JUDGE_PROMPT,
            response_dict=response,
            start_time=0,
            end_time=0,
            success_count=success_count,
            gene_list=candidate_genes,
            mod=mod,
            llm_judge=LLM_JUDGE_MODEL
        )

        # Save log
        log_pack = [hypo_output, llm_feedback]
        log_df = log_saver(
            log_df, 
            mod, 
            TASK, 
            candidate_genes, 
            y, 
            log_pack, 
            TASK_LOG_FILE, 
            *attribute 
        )
        
        # Determine the prediction output based on mode
        if mod == 'multi_round_discussion':
            print(f"[response] {response['CGRH']}", flush=True)
            prediction_output = hypo_output
        else:
            print(f"[response] {response['llm']}", flush=True)
            prediction_output = llm_feedback
        
        # Check if prediction is correct and update counters
        is_correct = y.lower() in prediction_output.lower() if prediction_output else False
        
        if is_correct:
            count += 1
            
        # Update accuracy dictionary based on mode
        if mod == 'multi_round_discussion':
            update_acc_dict(acc_dict, TASK, 'pannel_count', *attribute)
        else:
            update_acc_dict(acc_dict, TASK, 'gpt_count', *attribute)

        # Update total count
        total = success_count.get('total', 0)
        update_acc_dict(acc_dict, TASK, 'total', *attribute)

        # Calculate accuracy with division by zero protection
        acc = count / total if total > 0 else 0.0
        
        # Logging output
        print('Input Query: ', user_instruction, flush=True)
        print(
            f"\nContext: {candidate_genes, attribute} "
            f"| y: {y} "
            f"| LLM-predict-y: {prediction_output} "
            f"| LLM Acc: {acc:.4f} ({count}/{total})\n",
            flush=True
        )
        
        log_acc_dict(acc_dict, TASK, *attribute)
    log_acc_dict(acc_dict, FINAL_LOG, None)


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def get_evaluation_data_path(task: str, query_mode: bool = True) -> str:
    """
    Get the appropriate evaluation data path based on task and mode.
    
    Args:
        task: Task type ('targetID', 'sl', 'immune_response')
        query_mode: Whether to use query mode
        evaluation_db_path: Custom evaluation database path (if provided)
    
    Returns:
        Path to the evaluation data file
    """
    # Task-specific file path configuration
    task_config = {
        'targetID': {
            'query': TARGET_DB_QUERY_PATH,
            'source': TARGET_DB_PATH
        },
        'sl': {
            'query': SL_DB_QUERY_PATH,
            'source': SL_DB_PATH
        },
        'immune_response': {
            'query': PATIENT_TABLE_QUERY_PATH,
            'source': PATIENT_TABLE_PATH
        }
    }
    
    if task not in task_config:
        raise ValueError(f"Unsupported task: {task}")
    
    config = task_config[task]
    subfolder = 'evaluation_samples' if query_mode else 'source'
    filename = config['query'] if query_mode else config['source']
    
    return os.path.join(EVALUATION_FOLDER, task, subfolder, filename)


def load_evaluation_data(task: str, query_mode: bool = True, checkpoint: str = None) -> pd.DataFrame:
    """
    Load evaluation data for the specified task.
    
    Args:
        task: Task type ('targetID', 'sl', 'immune_response')
        query_mode: Whether to use query mode (default: True for evaluation_samples)
        checkpoint: Checkpoint to split data after (if provided)
    
    Returns:
        Loaded DataFrame
    """
    data_path = get_evaluation_data_path(task, query_mode)
    print(f"Loading evaluation data from: {data_path}")
    print(f"  Query mode: {query_mode} → Folder: {'evaluation_samples' if query_mode else 'source'}", flush=True)
    
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {len(df)} rows")
        
        if checkpoint:
            df = split_df_after_checkpoint(df, checkpoint, task)
            print(f"After checkpoint split: {len(df)} rows")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Evaluation data file not found: {data_path}")
        raise
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        raise


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Initialize random seeds
    random.seed(SEED)
    
    # Display configuration
    print("=" * 80, flush=True)
    print(" MEDEA EVALUATION CONFIGURATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Setting:           {SETTING}", flush=True)
    print(f"Task:              {TASK}", flush=True)
    print(f"Dataset Seed:      {SAMPLE_SEED}", flush=True)
    print(f"LLM Backbone Seed: {SEED}", flush=True)
    print(f"Temperature:       {DEFAULT_TEMPERATURE}", flush=True)
    print(f"Evaluation Folder: {EVALUATION_FOLDER}", flush=True)
    print("-" * 80, flush=True)
    
    # Task-specific configuration
    if TASK == "targetID":
        print(f"Disease:           {DISEASE}", flush=True)
        print(f"SCFM:              {SCFM}", flush=True)
    elif TASK == "sl":
        print(f"Cell Line:         {CELL_LINE}", flush=True)
        print(f"SL Source:         {SL_SOURCE}", flush=True)
    elif TASK == "immune_response":
        print(f"Immune Dataset:    {IMMUNE_DATASET}", flush=True)
        print(f"Patient TPM Root:  {PATIENT_TPM_ROOT}", flush=True)
    
    print("-" * 80, flush=True)
    backbone_llm_display = get_backbone_llm("gpt-4o")
    paraphraser_llm_display = get_env_with_error("PARAPHRASER_LLM", default=backbone_llm_display)
    print(f"LLM Backbone:      {backbone_llm_display}", flush=True)
    print(f"LLM Paraphraser:   {paraphraser_llm_display}", flush=True)
    print(f"LLM Judge:         {LLM_JUDGE_MODEL}", flush=True)
    print(f"Quality Max Iter:  {QUALITY_MAX_ITER}", flush=True)
    print(f"Debate Rounds:     {DEBATE_ROUND}", flush=True)
    print(f"Panelists:         {PANELIST_LLM}", flush=True)
    print("=" * 80, flush=True)
    
    # Load evaluation data
    df = load_evaluation_data(
        task=TASK,
        query_mode=QUERY_MODE,
        checkpoint=CHECKPOINT
    )
    
    # Run appropriate agent system based on setting
    if 'medea' in SETTING:
        medea_unittest(
            df, 
            user_template=copy.deepcopy(USER_PROMPT),
            agent_template=copy.deepcopy(AGENT_PROMPT)
        )
    elif 'multi_round_discussion' in SETTING:
        agent_output_df = pd.read_csv(os.path.join(EVALUATION_FOLDER, TASK, 'source', AGENT_OUTPUT_PATH))
        llm_unitest(
            df, 
            agent_output_df=agent_output_df,
            mod='multi_round_discussion', 
            model=SETTING, 
            user_template=copy.deepcopy(USER_PROMPT), 
            agent_template=copy.deepcopy(AGENT_PROMPT)
        )
    else:
        llm_unitest(
            df, 
            mod='llm', 
            model=SETTING, 
            user_template=copy.deepcopy(USER_PROMPT), 
            agent_template=copy.deepcopy(AGENT_PROMPT)
        )
