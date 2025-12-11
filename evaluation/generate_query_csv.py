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

# Add the parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import input_loader
from query_template import *
from dotenv import load_dotenv
load_dotenv()

experiment_setup_dict = {
    # Target Normination (Hard multi-choice)
    'targetid_gpt4o_multi': {
        'user': target_id_query_temp_5, 
        'agent': target_id_instruction_1,
        'judge_prompt': TARGETID_REASON_CHECK
    },
    # Target Normination (Open-ended)
    'targetid_open_end': {
        'user': target_id_open_end_query_temp_1, 
        'agent': target_id_instruction_2, 
        'judge_prompt': TARGETID_OPEN_END_REASON_CHECK
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
    # Synthetic Lethality Prediction (Multi-choice) [Enrichr+DepMap]
    'sl_e5_multi':  {
        'user': sl_query_lineage_multi, 
        'agent': sl_instruction_e5, 
        'judge_prompt': MULTI_CHOICE_SL_REASON_CHECK
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
}

PROMPT_SETTING = "immune_gpt4o_tmp6"
USER_PROMPT = experiment_setup_dict[PROMPT_SETTING]['user'] # Pick user intruction template (Path: evaluation/query_template.py)
AGENT_PROMPT = experiment_setup_dict[PROMPT_SETTING]['agent'] # Pick agent intruction template (Path: evaluation/query_template.py)
LLM_JUDGE_PROMPT = experiment_setup_dict[PROMPT_SETTING]['judge_prompt'] # LLM judge prompt template (Path: evaluation/query_template.py)
EVALUATION_FOLDER = os.path.expanduser('~/Medea/evaluation')

def generate_queries_for_task(task: str, 
                            input_file: str, 
                            output_file: str,
                            rephrase_model: str = "gpt-4o",
                            user_template: str = None,
                            agent_template: str = None,
                            scfm: str = None,
                            sl_source: str = None,
                            patient_tpm_root: str = None,
                            query_mode: bool = False) -> None:
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
        sl_source: Source for SL data ("biogrid" or "samson")
        patient_tpm_root: Root path for patient TPM data
        query_mode: If True, use existing user_question and full_query columns directly
    """
    
    # Read input data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
    
    if query_mode:
        print("Running in query mode - using existing user_question and full_query columns")
        # In query mode, we just need to format the existing data
        queries = []
        for _, row in df.iterrows():
            if task == "sl":
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
        # Generate queries using input_loader
        queries = []
        
        for result in input_loader(
            df=df,
            task=task,
            rephrase_model=rephrase_model,
            user_template=user_template,
            agent_template=agent_template,
            scfm=scfm,
            sl_source=sl_source,
            patient_tpm_root=patient_tpm_root,
            query_mode=query_mode
        ):
            if task == "sl":
                gene_pair, agent_X, X, interaction, _, _, cell_line = result
                sample_entity = {
                    'gene_a': gene_pair[0],
                    'gene_b': gene_pair[1],
                    'cell_line': cell_line,
                    'interaction': interaction,
                    'user_question': X,
                    'full_query': agent_X
                }
            elif task == "targetID":
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
            print(sample_entity)
    # Save to CSV
    save_queries_to_csv(task, queries, output_file)
    print(f"Generated {len(queries)} queries and saved to {output_file}")

def save_queries_to_csv(task: str, queries: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save queries to CSV file using pandas DataFrame.
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
        
        # Convert queries list to DataFrame
        df = pd.DataFrame(queries)
        
        # Save to CSV using pandas
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"Successfully saved {len(queries)} queries to {output_file}")
        
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")

def get_default_templates(task: str) -> tuple:
    """
    Get default templates for the specified task.
    """
    if task == "sl":
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
    parser.add_argument('--user-template', default=USER_PROMPT, help='Custom user query template')
    parser.add_argument('--agent-template', default=AGENT_PROMPT, help='Custom agent instruction template')
    
    parser.add_argument(
        '--task', 
        choices=['sl', 'targetID', 'immune_response'],
        help='Task type'
    )
    parser.add_argument(
        '--sl-source', 
        default='samson',
        help='Source for SL data (required for sl task)'
    )
    parser.add_argument(
        '--disease',
        default='blastoma',
        help='Disease for targetID task'
    )
    parser.add_argument(
        '--cell-line', 
        default="MCF7",
        help='Cell line'
    )
    parser.add_argument(
        '--patient-tpm-root', 
        default=os.path.expanduser(os.getenv("MEDEADB_PATH", "~/MedeaDB")) + "/immune-compass/patients",
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
    args = parser.parse_args()
    
    # Validate required arguments
    if args.task == 'sl' and not args.sl_source:
        parser.error("--sl-source is required for sl task")
    if args.task == 'immune_response' and not args.patient_tpm_root:
        parser.error("--patient-tpm-root is required for immune_response task")
    
    # Get default templates if not provided
    if not args.user_template or not args.agent_template:
        default_user, default_agent = get_default_templates(args.task)
        user_template = args.user_template or default_user
        agent_template = args.agent_template or default_agent
    else:
        user_template = args.user_template
        agent_template = args.agent_template
    
    if args.task == 'targetID':
        source_name = f"targetid-{args.disease}-{args.seed}.csv"
        output_name = f"targetid-{args.disease}-query-{args.seed}.csv"
    elif args.task == 'sl':
        source_name = f"{args.sl_source}-{args.cell_line}-{args.seed}.csv"
        output_name = f"{args.sl_source}-{args.cell_line}-query-{args.seed}.csv"
    elif args.task == 'immune_response':
        source_name = f"{args.immune_dataset}-patient.csv"
        output_name = f"{args.immune_dataset}-{args.immune_tmp}-query-{args.seed}.csv"
    
    input_file = os.path.join(EVALUATION_FOLDER, args.task, 'source', source_name)
    output_file = os.path.join(EVALUATION_FOLDER, args.task, 'evaluation_samples', output_name)
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"user_template: {user_template}")
    print(f"agent_template: {agent_template}")
    print(f"EVALUATION_FOLDER: {EVALUATION_FOLDER}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}")
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
        query_mode=args.query_mode
    )

if __name__ == "__main__":
    main() 