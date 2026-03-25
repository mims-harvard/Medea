#!/usr/bin/env python3
"""
Script to generate CSV query files following the same format as the sample.
Uses the input_loader function from utils.py to generate queries.
"""

import os
import sys
import pandas as pd
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import input_loader
from query_template import *
from dotenv import load_dotenv
load_dotenv()

def log(msg):
    """Print with flush for immediate output"""
    print(msg, flush=True)


def process_single_row_sl(args, max_retries=3, retry_delay=5):
    """Process a single SL row for parallel execution with retry logic"""
    import time
    from medea.tool_space.gpt_utils import chat_completion
    from query_template import EXPERIMENT_INSTRUCTION_REPHRASE_TEMPLATE
    
    row_idx, row_data, user_template, agent_template, rephrase_model, sl_source = args
    
    if sl_source == 'yeast':
        g_a = row_data.get('array_gene', row_data.get('gene_a', ''))
        g_b = row_data.get('query_gene', row_data.get('gene_b', ''))
        context = row_data.get('condition', '')
        y = row_data.get('label', row_data.get('interaction', ''))

        gpt_prompt = user_template.format(
            gene_a=g_a,
            gene_b=g_b,
            condition=expand_yeast_condition(context)
        )
    else:
        g_a = row_data.get('gene_a', '')
        g_b = row_data.get('gene_b', '')
        context = row_data.get('cell_line', '')
        y = row_data.get('interaction', '')
        
        gpt_prompt = user_template.format(
            gene_a=g_a,
            gene_b=g_b,
            cell_line=context
        )
    
    def _is_error_response(text: str) -> bool:
        """Return True if the LLM returned an error string instead of a real answer."""
        if not text:
            return True
        lowered = text.lower()
        return any(kw in lowered for kw in [
            "i cannot help", "azure error", "ratelimitreached",
            "error code", "rate limit", "too many requests",
        ])

    # Call LLM for rephrasing user query with retry logic
    user_instruction = None
    for attempt in range(max_retries):
        try:
            user_instruction = chat_completion(gpt_prompt, temperature=1, model=rephrase_model)
            if user_instruction and not _is_error_response(user_instruction):
                user_instruction = user_instruction.strip('""')
                break
            # Treat error-string responses the same as rate-limit exceptions
            wait = retry_delay * 2 ** attempt
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise Exception(f"Error response after {max_retries} attempts for {g_a}-{g_b}: {user_instruction}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * 2 ** attempt)
            else:
                raise Exception(f"Failed after {max_retries} retries for {g_a}-{g_b}: {e}")

    if not user_instruction:
        raise Exception(f"Empty response for {g_a}-{g_b}")
    
    # Only generate experiment instruction if agent_template is provided
    if agent_template:
        experiment_instruction = None
        for attempt in range(max_retries):
            try:
                experiment_instruction = chat_completion(
                    EXPERIMENT_INSTRUCTION_REPHRASE_TEMPLATE.format(
                        role="biologist", 
                        task_instruction_template=agent_template
                    ), 
                    model=rephrase_model, 
                    temperature=1,
                )
                if experiment_instruction and not _is_error_response(experiment_instruction):
                    break
                wait = retry_delay * 2 ** attempt
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    raise Exception(f"Error response for agent instruction after {max_retries} attempts")
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * 2 ** attempt)
                else:
                    raise Exception(f"Failed agent instruction after {max_retries} retries: {e}")
        
        full_query = user_instruction + " " + experiment_instruction if experiment_instruction else user_instruction
    else:
        # No agent template - full_query equals user_question
        full_query = user_instruction
    
    interaction = str(y).strip("['']").lower()
    
    if sl_source == 'yeast':
        sample_entity = {
            'gene_a': g_a,
            'gene_b': g_b,
            'condition': context,
            'interaction': interaction,
            'user_question': user_instruction,
            'full_query': full_query
        }
    else:
        sample_entity = {
            'gene_a': g_a,
            'gene_b': g_b,
            'cell_line': context,
            'interaction': interaction,
            'user_question': user_instruction,
            'full_query': full_query
        }
    
    return row_idx, sample_entity

YEAST_CONDITION_NAME_MAP = {
    'BLEO': 'bleomycin',
    'DMS': 'dimethyl sulfate',
}

def expand_yeast_condition(condition: str) -> str:
    """Expand yeast condition abbreviations to their formal chemical names."""
    return YEAST_CONDITION_NAME_MAP.get(condition.strip().upper(), condition)


experiment_setup_dict = {
    # Target Normination
    'targetid_analysis_gpt4o': {
        'user': target_id_query_temp, 
        'agent': target_id_instruction,
        'judge_prompt': TARGETID_REASON_CHECK
    },
    'targetid_analysis_claude': {
        'user': target_id_query_temp, 
        'agent': target_id_instruction,
        'judge_prompt': TARGETID_REASON_CHECK
    },
    # TargetID-PINNACLE
    'targetid_analysis_pinnacle': {
        'user': target_id_query_temp, 
        'agent': target_id_instruction,
        'judge_prompt': TARGETID_REASON_CHECK
    },
    # TargetID-TranscriptFormer
    'targetid_analysis_transcriptformer': {
        'user': target_id_query_temp, 
        'agent': target_id_instruction,
        'judge_prompt': TARGETID_REASON_CHECK
    },
    # Synthetic Lethality Prediction (Cell Line)
    'sl_analysis_gpt4o':  {
        'user': sl_query_lineage_openend, 
        'agent': sl_instruction_default, 
        'judge_prompt': SL_REASON_CHECK 
    },
    'sl_analysis_claude':  {
        'user': sl_query_lineage_openend, 
        'agent': sl_instruction_default, 
        'judge_prompt': SL_REASON_CHECK 
    },
    # Synthetic Lethality Prediction (Yeast)
    'sl_yeast_analysis_gpt4o': {
        'user': sl_query_yeast_openend,
        'agent': None,
        'judge_prompt': SL_REASON_CHECK
    },
    'sl_yeast_analysis_claude': {
        'user': sl_query_yeast_openend,
        'agent': None,
        'judge_prompt': SL_REASON_CHECK
    },
    # ICI Prediction 
    'immune_no_instruction_gpt4o': {
        'user': immune_query_default, 
        'agent':immune_no_instruction,
        'judge_prompt': IMMUNE_ANS_CHECK
    },
    'immune_instruction_a_gpt4o': {
        'user': immune_query_temp_a, 
        'agent':immune_instruction_a,
        'judge_prompt': IMMUNE_ANS_CHECK
    },
    'immune_instruction_b_gpt4o': {
        'user': immune_query_temp_b, 
        'agent':immune_instruction_b,
        'judge_prompt': IMMUNE_ANS_CHECK
    },
    'immune_instruction_b_claude': {
        'user': immune_query_temp_b, 
        'agent':immune_instruction_b,
        'judge_prompt': IMMUNE_ANS_CHECK
    }
}

PROMPT_SETTING = "sl_yeast_analysis_gpt4o"
USER_PROMPT = experiment_setup_dict[PROMPT_SETTING]['user'] # Pick user intruction template (Path: evaluation/query_template.py)
AGENT_PROMPT = experiment_setup_dict[PROMPT_SETTING]['agent'] # Pick agent intruction template (Path: evaluation/query_template.py)
LLM_JUDGE_PROMPT = experiment_setup_dict[PROMPT_SETTING]['judge_prompt'] # LLM judge prompt template (Path: evaluation/query_template.py)
# Use the directory where this script is located
EVALUATION_FOLDER = os.path.dirname(os.path.abspath(__file__))

def generate_queries_for_task(task: str, 
                            input_file: str, 
                            output_file: str,
                            rephrase_model: str = "gpt-4o",
                            user_template: str = None,
                            agent_template: str = None,
                            scfm: str = None,
                            sl_source: str = None,
                            patient_tpm_root: str = None,
                            query_mode: bool = False,
                            num_workers: int = 8) -> None:
    """
    Generate CSV query file for the specified task.
    
    Args:
        task: Task type ("targetID", "sl", or "immune_response")
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        rephrase_model: Model to use for rephrasing
        user_template: User query template
        agent_template: Agent instruction template
        scfm: SCFM source for targetID task
        sl_source: Source for SL data ("biogrid", "samson", or "yeast")
        patient_tpm_root: Root path for patient TPM data
        query_mode: If True, use existing user_question and full_query columns directly
    """
    
    # Read input data
    df = pd.read_csv(input_file)
    log(f"[INFO] Loaded {len(df)} rows from {input_file}")
    log(f"[INFO] sl_source: {sl_source}")
    
    if query_mode:
        log("[INFO] Running in query mode - using existing user_question and full_query columns")
        # In query mode, we just need to format the existing data
        queries = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            if task == "sl":
                # Check if this is yeast data (has 'condition' column instead of 'cell_line')
                if sl_source == 'yeast' or 'condition' in row.index:
                    queries.append({
                        'gene_a': row.get('gene_a', row.get('array_gene', '')),
                        'gene_b': row.get('gene_b', row.get('query_gene', '')),
                        'condition': row.get('condition', ''),
                        'interaction': row.get('interaction', row.get('label', '')),
                        'user_question': row.get('user_question', ''),
                        'full_query': row.get('full_query', '')
                    })
                else:
                    queries.append({
                        'gene_a': row.get('gene_a', ''),
                        'gene_b': row.get('gene_b', ''),
                        'cell_line': row.get('cell_line', ''),
                        'interaction': row.get('interaction', ''),
                        'user_question': row.get('user_question', ''),
                        'full_query': row.get('full_query', '')
                    })
            elif task == "targetID":
                queries.append({
                    'candidate_genes': row.get('candidate_genes', ''),  # Use candidate_genes as is
                    'disease': row.get('disease', ''),
                    'celltype': row.get('celltype', ''),
                    'y': row.get('y', ''),
                    'user_question': row.get('user_question', ''),
                    'full_query': row.get('full_query', '')
                })
            elif task == "immune_response":
                queries.append({
                    'cancer_type': row.get('cancer_type', ''),
                    'TMB (FMOne mutation burden per MB)': row.get('TMB (FMOne mutation burden per MB)', ''),
                    'Neoantigen burden per MB': row.get('Neoantigen burden per MB', ''),
                    'Immune phenotype': row.get('Immune phenotype', ''),
                    'response_label': row.get('response_label', ''),
                    'user_question': row.get('user_question', ''),
                    'full_query': row.get('full_query', '')
                })
    else:
        # Generate queries using parallel processing for SL task
        if task == "sl":
            log(f"[INFO] Generating queries using LLM rephrasing (model: {rephrase_model})...")
            log(f"[INFO] Using {num_workers} parallel workers for faster processing")
            
            # Prepare arguments for parallel processing
            row_args = []
            for idx, row in df.iterrows():
                row_args.append((idx, row.to_dict(), user_template, agent_template, rephrase_model, sl_source))
            
            # Process in parallel with progress bar
            queries = [None] * len(row_args)  # Pre-allocate to maintain order
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_single_row_sl, args): args[0] for args in row_args}
                
                failed_args = []  # Store failed row args for retry
                with tqdm(total=len(futures), desc="Generating queries") as pbar:
                    for future in as_completed(futures):
                        row_idx = futures[future]
                        try:
                            idx, sample_entity = future.result()
                            queries[idx] = sample_entity
                        except Exception as e:
                            # Find the original args for this row
                            for args in row_args:
                                if args[0] == row_idx:
                                    failed_args.append(args)
                                    break
                            tqdm.write(f"[WARN] Row {row_idx} failed, will retry: {e}")
                        pbar.update(1)
            
            # Retry failed rows sequentially with longer delays until all succeed
            if failed_args:
                log(f"\n[INFO] Retrying {len(failed_args)} failed rows sequentially...")
                import time
                
                retry_round = 1
                while failed_args:
                    log(f"[RETRY] Round {retry_round}: {len(failed_args)} rows remaining")
                    still_failed = []
                    
                    for args in tqdm(failed_args, desc=f"Retry round {retry_round}"):
                        row_idx = args[0]
                        try:
                            # Longer delay between sequential retries
                            time.sleep(2)
                            idx, sample_entity = process_single_row_sl(args, max_retries=5, retry_delay=10)
                            queries[idx] = sample_entity
                            log(f"[SUCCESS] Row {row_idx} succeeded on retry round {retry_round}")
                        except Exception as e:
                            still_failed.append(args)
                            tqdm.write(f"[WARN] Row {row_idx} still failed: {e}")
                    
                    failed_args = still_failed
                    retry_round += 1
                    
                    # Safety limit to prevent infinite loop
                    if retry_round > 10:
                        log(f"[ERROR] Giving up after 10 retry rounds. {len(failed_args)} rows still failed.")
                        break
                    
                    if failed_args:
                        log(f"[INFO] Waiting 30 seconds before next retry round...")
                        time.sleep(30)
            
            # Remove any None entries (failed rows)
            queries = [q for q in queries if q is not None]
        
        else:
            # For non-SL tasks, use sequential processing with input_loader
            log(f"[INFO] Generating queries using LLM rephrasing (model: {rephrase_model})...")
            queries = []
            
            results_iter = input_loader(
                df=df,
                task=task,
                rephrase_model=rephrase_model,
                user_template=user_template,
                agent_template=agent_template,
                scfm=scfm,
                sl_source=sl_source,
                patient_tpm_root=patient_tpm_root,
                query_mode=query_mode
            )
            
            for i, result in enumerate(tqdm(results_iter, total=len(df), desc="Generating queries")):
                if task == "targetID":
                    candidate_genes, agent_X, X, y, celltype, disease = result
                    sample_entity = {
                        'candidate_genes': candidate_genes,
                        'disease': disease,
                        'celltype': celltype,
                        'y': y,
                        'user_question': X,
                        'full_query': agent_X
                    }
                elif task == "immune_response":
                    _, agent_X, X, y, cancer_type, tmb, nmb, pheno, _ = result
                    sample_entity = {
                        'cancer_type': cancer_type,
                        'TMB (FMOne mutation burden per MB)': tmb,
                        'Neoantigen burden per MB': nmb,
                        'Immune phenotype': pheno,
                        'response_label': y,
                        'user_question': X,
                        'full_query': agent_X
                    }
                queries.append(sample_entity)
    # Save to CSV
    log(f"\n[INFO] Saving {len(queries)} queries to {output_file}...")
    save_queries_to_csv(task, queries, output_file)
    log(f"[SUCCESS] Generated {len(queries)} queries and saved to {output_file}")

def save_queries_to_csv(task: str, queries: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save queries to CSV file using pandas DataFrame.
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            log(f"[INFO] Created directory: {output_dir}")
        
        # Convert queries list to DataFrame
        df = pd.DataFrame(queries)
        
        # Save to CSV using pandas
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        log(f"[INFO] Successfully saved {len(queries)} queries to {output_file}")
        
    except Exception as e:
        log(f"[ERROR] Error saving to {output_file}: {e}")

def get_default_templates(task: str, sl_source: str = None) -> tuple:
    """
    Get default templates for the specified task.
    
    Args:
        task: Task type ("targetID", "sl", or "immune_response")
        sl_source: Source for SL data ("biogrid", "samson", or "yeast")
    """
    if task == "sl":
        if sl_source == "yeast":
            # Use yeast-specific template with condition instead of cell_line
            # No agent template for yeast - full_query should equal user_question
            user_template = sl_query_yeast_openend
            agent_template = None
        else:
            user_template = """We have introduced concurrent mutations in {gene_a} and {gene_b} in the {cell_line} cell line. Could you describe the resulting synthetic genetic interaction observed between these two genes, if any?"""
            agent_template = """ Use DepMap data to retrieve correlation metrics reflecting the co-dependency of the gene pair on cell viability. Next, perform pathway enrichment analysis with Enrichr to identify whether pathways associated with cell viability are significantly enriched and could be impacted by the gene pair. Synthesize the DepMap and Enrichr results, evaluate whether the combined perturbation of these genes is likely to induce a significant effect on cell viability, and find literature support if exist."""
    
    elif task == "targetID":
        user_template = """What are the potential therapeutic targets for {disease} in {celltype} cells, specifically focusing on {candidate_genes}?"""
        agent_template = """ Use DepMap data to analyze gene dependency scores and identify potential therapeutic targets. Perform pathway enrichment analysis with Enrichr to understand the biological context. Synthesize the results to evaluate the therapeutic potential of the candidate genes."""
    
    elif task == "immune_response":
        user_template = """Analyze the immune response patterns in {disease} patients treated with {treatment} in {tissue} tissue, considering TMB of {tmb}, neoantigen burden of {nmb}, and patient demographics (sex: {sex}, race: {race})."""
        agent_template = """"""
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return user_template, agent_template

def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV query files using input_loader function",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--seed', default=42, help='Random seed')
    parser.add_argument(
        '--rephrase-model', 
        default='o3-mini-0131',
        help='Model to use for rephrasing (default: o3-mini-0131)'
    )
    parser.add_argument('--user-template', default=None, help='Custom user query template (if not set, uses task-specific default)')
    parser.add_argument('--agent-template', default=None, help='Custom agent instruction template (if not set, uses task-specific default)')
    
    parser.add_argument(
        '--task', 
        choices=['sl', 'targetID', 'immune_response'],
        help='Task type'
    )
    parser.add_argument(
        '--sl-source', 
        default='samson',
        help='Source for SL data: "biogrid", "samson", or "yeast" (required for sl task)'
    )
    parser.add_argument(
        '--disease',
        default='blastoma',
        help='Disease for targetID task'
    )
    parser.add_argument(
        '--cell-line', 
        default="MCF7",
        help='Cell line (for biogrid/samson SL data)'
    )
    parser.add_argument(
        '--condition',
        default="BLEO",
        help='Experimental condition for yeast SL data (e.g., BLEO, DMS)'
    )
    parser.add_argument(
        '--patient-tpm-root', 
        default=os.path.expanduser(os.getenv("MEDEADB_PATH", "~/MedeaDB")) + "/compass/patients",
        help='Root path for patient TPM data (required for immune_response task)'
    )
    parser.add_argument(
        '--scfm', 
        default='PINNACLE',
        help='Source for SCFM data (default: PINNACLE)'
    )
    parser.add_argument(
        '--immune-dataset', 
        default='IMVigor210',
        help='Dataset name for immune_response task (default: IMVigor210)'
    )
    parser.add_argument(
        '--immune-tmp', 
        default='tmp1',
        help='Immune tmp for immune_response task (default: tmp1)'
    )
    parser.add_argument(
        '--query-mode', 
        default=False,
        help='If set, use existing user_question and full_query columns directly'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of parallel workers for query generation (default: 8)'
    )
    args = parser.parse_args()
    
    # Validate required arguments
    if args.task == 'sl' and not args.sl_source:
        parser.error("--sl-source is required for sl task")
    if args.task == 'immune_response' and not args.patient_tpm_root:
        parser.error("--patient-tpm-root is required for immune_response task")
    
    # Get default templates if not provided
    if not args.user_template or not args.agent_template:
        default_user, default_agent = get_default_templates(args.task, sl_source=args.sl_source)
        user_template = args.user_template or default_user
        agent_template = args.agent_template or default_agent
    else:
        user_template = args.user_template
        agent_template = args.agent_template
    
    if args.task == 'targetID':
        source_name = f"targetid-{args.disease}-{args.seed}.csv"
        output_name = f"targetid-{args.disease}-query-{args.seed}.csv"
    elif args.task == 'sl':
        if args.sl_source == 'yeast':
            # Yeast data uses condition instead of cell_line
            source_name = f"yeast-{args.condition}-{args.seed}.csv"
            output_name = f"yeast-{args.condition}-query-{args.seed}.csv"
        else:
            source_name = f"{args.sl_source}-{args.cell_line}-{args.seed}.csv"
            output_name = f"{args.sl_source}-{args.cell_line}-query-{args.seed}.csv"
    elif args.task == 'immune_response':
        source_name = f"{args.immune_dataset}-patient.csv"
        output_name = f"{args.immune_dataset}-{args.immune_tmp}-query-{args.seed}.csv"
    
    input_file = os.path.join(EVALUATION_FOLDER, args.task, 'source', source_name)
    output_file = os.path.join(EVALUATION_FOLDER, args.task, 'evaluation_samples', output_name)
    
    log("=" * 60)
    log(f"[CONFIG] Task: {args.task}")
    log(f"[CONFIG] SL Source: {args.sl_source}")
    log(f"[CONFIG] Rephrase Model: {args.rephrase_model}")
    log(f"[CONFIG] Num Workers: {args.num_workers}")
    log(f"[CONFIG] Input file: {input_file}")
    log(f"[CONFIG] Output file: {output_file}")
    log(f"[CONFIG] EVALUATION_FOLDER: {EVALUATION_FOLDER}")
    log("=" * 60)
    
    # Show template preview (first 200 chars)
    log(f"\n[TEMPLATE] User template preview:\n{user_template[:200] if user_template else 'None'}...")
    log(f"\n[TEMPLATE] Agent template: {agent_template[:200] + '...' if agent_template else 'None (full_query = user_question)'}")
    log("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        log(f"[ERROR] Input file does not exist: {input_file}")
        return
    
    # Generate queries
    generate_queries_for_task(
        task=args.task,
        input_file=input_file,
        output_file=output_file,
        rephrase_model=args.rephrase_model,
        user_template=user_template,
        agent_template=agent_template,
        scfm=args.scfm,
        sl_source=args.sl_source,
        patient_tpm_root=args.patient_tpm_root,
        query_mode=args.query_mode,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main() 