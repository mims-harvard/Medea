from medea.tool_space.gpt_utils import chat_completion
from evaluation.query_template import *
import ast, os, random
import numpy as np
import pandas as pd

ABSTAIN_VALS = {"abstain"}

def _classify_decision(d):
    d = str(d).strip().lower()
    if d in ABSTAIN_VALS:
        return "abstain"
    if d == "failed":
        return "failed"
    return d

# Loads input data for the specified task from a DataFrame.
def input_loader(
    df: pd.DataFrame, 
    task:str, 
    rephrase_model:str,
    user_template:str=None, 
    agent_template:str=None,
    scfm:str=None, 
    sl_source:str=None,
    patient_tpm_root=None,
    query_mode=True
):
    """
    Loads input data for the specified task from a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame containing the necessary data.
        task (str): The type of task to perform, either "targetID", "sl", or "immune_response".
        rephrase_model (str): Model to use for rephrasing queries.
        user_template (str): Template for user queries (not used in query_mode).
        agent_template (str): Template for experiment instructions (not used in query_mode).
        scfm (str): SCFM source for targetID task.
        sl_source (str): Source for SL data ("biogrid" or "samson").
        patient_tpm_root (str): Root path for patient TPM data.
        query_mode (bool): If True, use existing user_question and full_query columns directly.

    Returns:
        generator: A generator yielding tuples containing (candidate_genes, user_instruction, experiment_instruction, ...).
        
    The function processes the DataFrame based on the specified task:
    - For "targetID":
        - Returns: (candidate_genes, user_instruction, experiment_instruction, y, cell_type, disease)
    - For "sl":
        - Returns: (gene_pair, user_instruction, experiment_instruction, interaction, gene_pair, interaction, cell_line)
    - For "immune_response":
        - Returns: (None, user_instruction, experiment_instruction, y, cancer_type, tmb, nmb, pheno, y)
    """
    
    if query_mode:
        # Query mode: use existing user_question and full_query columns directly
        if task == "targetID":
            for _, row in df.iterrows():
                g = row.get('candidate_genes', '')
                user_instruction = row.get('user_question', '')
                y = row.get('y', '')
                c = row.get('celltype', '')
                d = row.get('disease', '')
                
                # Extract experiment instruction from full_query
                full_query = row.get('full_query', '')
                if full_query and full_query != user_instruction:
                    # Extract experiment_instruction by removing user_instruction from full_query
                    experiment_instruction = full_query[len(user_instruction):].strip()
                elif scfm != 'PINNACLE' and agent_template != None:
                    # Generate experiment instruction
                    instruction = agent_template.format(
                        disease=d, 
                        scfm=scfm, 
                        cell_type=c
                    )
                    experiment_instruction = chat_completion(
                        EXPERIMENT_INSTRUCTION_REPHRASE_TEMPLATE.format(
                            role="biologist", 
                            task_instruction_template=instruction
                        ), 
                        model=rephrase_model, 
                        temperature=1
                    )
                else:
                    experiment_instruction = None
                
                yield (g, user_instruction, experiment_instruction, y, c, d)
        
        elif task == "sl":
            for _, row in df.iterrows():
                # Detect if this is yeast data or cell line data based on column presence
                is_yeast_data = 'condition' in row.index or 'array_gene' in row.index
                
                if is_yeast_data:
                    # Yeast data columns: array_gene, query_gene, condition, label
                    g_a = row.get('gene_a', row.get('array_gene', ''))
                    g_b = row.get('gene_b', row.get('query_gene', ''))
                    context = row.get('condition', '')
                    interaction = row.get('interaction', row.get('label', ''))
                else:
                    # Cell line data columns: gene_a, gene_b, cell_line, interaction
                    g_a = row.get('gene_a', '')
                    g_b = row.get('gene_b', '')
                    context = row.get('cell_line', '')
                    interaction = row.get('interaction', '')
                
                user_instruction = row.get('user_question', '')
                
                # Get full_query (user question + experiment instruction)
                full_query = row.get('full_query', '')
                # Use full_query if available, otherwise fall back to user_instruction
                X = full_query if full_query else user_instruction
                    
                # yield: (candidate_genes, X=full_query, user_instruction, y, *attributes)
                yield ([g_a, g_b], X, user_instruction, interaction, [g_a, g_b], interaction, context)
        
        elif task == "immune_response":
            for _, row in df.iterrows():
                tmp_p = row.get('Index', '')
                cancer_type = row.get('cancer_type', '')
                tmb = row.get('TMB (FMOne mutation burden per MB)', '')
                nmb = row.get('Neoantigen burden per MB', '')
                pheno = row.get('Immune phenotype', '')
                y = row.get('response_label', '')
                user_instruction = row.get('user_question', '')
                
                # Extract experiment instruction from full_query
                full_query = row.get('full_query', '')
                if full_query and full_query != user_instruction:
                    experiment_instruction = full_query[len(user_instruction):].strip()
                else:
                    experiment_instruction = None
                    
                yield (None, user_instruction, experiment_instruction, y, cancer_type, tmb, nmb, pheno, y)
    
    else:
        # Non-query mode: generate queries using templates
        if task == "targetID":
            for d, c, g, y in df[["disease", "celltype", "candidate_genes", "y"]].values:
                user_instruction = user_template.format(disease=d, celltype=c, candidate_genes=g.strip("['']"))
                experiment_instruction = agent_template.format(disease=d, scfm=scfm, cell_type=c)
                
                # Paraphrase
                user_instruction = chat_completion(REPHRASE_TEMPLATE.format(role="biologist", task_instruction_template=user_instruction), model=rephrase_model, temperature=1)
                experiment_instruction = chat_completion(EXPERIMENT_INSTRUCTION_REPHRASE_TEMPLATE.format(role="biologist", task_instruction_template=experiment_instruction), model=rephrase_model, temperature=1)
                
                y = y.strip("['']")
                yield (g, user_instruction, experiment_instruction, y, c, d)

        elif task == "sl":
            setting_dict = {
                "pheno": "- Phenotype: [Phenotype; if provided]", 
                "pheno_sample": "- Phenotype: {phenotype}",
                "notes": "- Additional Notes: [Additional Notes; if provided]",
                "notes_sample": "- Additional Notes: {additional_notes}"
            }
            
            addition_context_temp = ""
            addition_context_sample = ""

            if sl_source == 'biogrid':
                columns = ["gene_a", "gene_b", "synonym_a", "synonym_b", "cell_line", "phenotype", "interaction"]
            elif sl_source == 'samson':
                columns = ["gene_a", "gene_b", "cell_line", "interaction"]
            elif sl_source == 'yeast':
                columns = ["array_gene", "query_gene", "condition", "label"]
            else:
                raise ValueError(f"Unsupported SL source: {sl_source}")

            for sample in df[columns].values:
                if sl_source == 'biogrid':
                    g_a, g_b, s_a, s_b, cell_line, pheno, y = sample
                    addition_context_temp = setting_dict['pheno']
                    addition_context_sample = setting_dict['pheno_sample'].format(phenotype=pheno)

                    gpt_prompt = user_template.format(
                        gene_a=g_a,
                        gene_b=g_b,
                        synonyms_a=s_a,
                        synonyms_b=s_b,
                        cell_line=cell_line,
                        addition_context_temp=addition_context_temp,
                        addition_context_sample=addition_context_sample
                    )
                    context = cell_line

                elif sl_source == 'samson':
                    g_a, g_b, cell_line, y = sample

                    gpt_prompt = user_template.format(
                        gene_a=g_a,
                        gene_b=g_b,
                        cell_line=cell_line
                    )
                    context = cell_line

                elif sl_source == 'yeast':
                    g_a, g_b, condition, y = sample

                    gpt_prompt = user_template.format(
                        gene_a=g_a,
                        gene_b=g_b,
                        condition=condition
                    )
                    context = condition

                user_instruction = chat_completion(gpt_prompt, temperature=1, model=rephrase_model).strip('""')
                experiment_instruction = chat_completion(EXPERIMENT_INSTRUCTION_REPHRASE_TEMPLATE.format(role="biologist", task_instruction_template=agent_template), model=rephrase_model, temperature=1)
                interaction = y.strip("['']").lower()  # Normalize to lowercase (SL -> sl, non_SL -> non_sl)

                yield ([g_a, g_b], user_instruction, experiment_instruction, interaction, [g_a, g_b], interaction, context)

        elif task == 'immune_response':
            cols = [
                'Index', 'cancer_type', 'ICI', 
                'Tissue', 'Immune phenotype', 
                'TMB (FMOne mutation burden per MB)',
                'Neoantigen burden per MB', 'Sex',
                'Race', 'response_label'
            ]
            for tmp_p, cancer_type, drug, tissue, pheno, tmb, nmb, sex, race, y in df[cols].values:
                tpm_path = os.path.join(patient_tpm_root, tmp_p)
                disease = "Bladder Urothelial Carcinoma (BLCA)"
                # Use the actual cancer_type from the data instead of hardcoding
                user_instruction = user_template.format(
                    disease=disease, treatment=drug, tissue=tissue, 
                    tmb=tmb, nmb=nmb, sex=sex, race=race, 
                )
                ici_file_path = ICI_FILE_PATH.format(tpm_path=tpm_path)

                user_instruction = chat_completion(IMMUNE_REPHRASE_TEMPLATE.format(role="clinician", task_instruction_template=user_instruction), model=rephrase_model, temperature=1)
                if len(agent_template) > 0:
                    experiment_instruction = chat_completion(EXPERIMENT_INSTRUCTION_REPHRASE_TEMPLATE.format(role="clinician", task_instruction_template=agent_template), model=rephrase_model, temperature=1)
                    experiment_instruction = " ".join([experiment_instruction, ici_file_path])
                else:
                    experiment_instruction = ici_file_path
                yield (None, user_instruction, experiment_instruction, y, cancer_type, tmb, nmb, pheno, y)


def ensure_columns(df: pd.DataFrame, cols: list, default=np.nan) -> None:
    """
    Ensure that DataFrame `df` contains all columns in `cols`.
    If a column is missing, add it with the given `default` value.
    """
    for col in cols:
        if col not in df.columns:
            df[col] = default


def log_saver(
    df: pd.DataFrame,
    mod: str,
    task: str,
    candidate_genes,
    y,
    log_pack,
    csv_path,
    *args
) -> pd.DataFrame:
    """
    Append a new log row to `df` based on `task` and `mod`, ensuring required columns.

    Parameters:
    - df: DataFrame to append to.
    - mod: 'medea' or other model identifier.
    - task: one of ['targetID', 'sl', 'immune_response'].
    - candidate_genes: genes involved (varies by task).
    - y: ground truth label or value.
    - log_pack: model outputs (tuple for 'medea', single value otherwise).
    - args: additional context-specific arguments.

    Returns:
    - Updated DataFrame with the new row appended.
    """
    # Define expected columns per task
    expected_columns = {
        "targetID":          ['candidate_genes', 'cell_type', 'disease', 'y'],
        "sl":                ['gene a', 'gene b', 'cell line', 'y'],
        "immune_response":   ['cancer_type', 'tmb', 'nmb', 'pheno', 'y'],
        "medea":             ['pa', 'r', 'full', 'llm-bb'],
        "multi_round_discussion": ['full', 'llm'],
        "llm":               ['llm']
    }

    # Ensure base columns exist for the given task
    ensure_columns(df, expected_columns.get(task, []))

    # Build the new row dict based on task
    new_row = {}
    if task == "targetID":
        cell_type, disease = args
        new_row = {
            'candidate_genes': candidate_genes,
            'cell_type': cell_type,
            'disease': disease,
            'y': y
        }
    elif task == "sl":
        pair, _, cell_line = args
        g_a, g_b = pair
        new_row = {
            'gene a': g_a,
            'gene b': g_b,
            'cell line': cell_line,
            'y': y
        }
    elif task == "immune_response":
        cancer_type, tmb, nmb, pheno, _ = args
        new_row = {
            'cancer_type': cancer_type,
            'tmb': tmb,
            'nmb': nmb,
            'pheno': pheno,
            'y': y
        }
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Handle model-specific output columns
    if mod == "medea":
        medea_cols = expected_columns['medea']
        ensure_columns(df, medea_cols)
        # Unpack log_pack as a tuple of four outputs
        pa_out, r_out, full_out, llm_bb_out = log_pack
        new_row.update({
            'pa': pa_out,
            'r': r_out,
            'full': full_out,
            'llm-bb': llm_bb_out
        })
    elif mod == 'multi_round_discussion':
        panel_cols = expected_columns['multi_round_discussion']
        ensure_columns(df, panel_cols)
        hyp_out, llm_out = log_pack
        new_row.update({
            'full': hyp_out,
            'llm': llm_out
        })
    elif mod == 'llm':
        llm_cols = expected_columns['llm']
        ensure_columns(df, llm_cols)
        new_row.update({
            'llm': log_pack[-1]
        })
    # Append new row
    updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print(f"[LogSaver] Updated new row to log csv: {new_row}", flush=True)

    # Flush updated DataFrame to CSV
    updated_df.to_csv(csv_path, index=False)

    return updated_df
            

        
        
# Check success rate and log the Code snippet and output
def log_check(
    task,
    query, 
    llm_query,
    llm_judge_prompt,
    response_dict, 
    start_time, 
    end_time, 
    success_count, 
    gene_list=None,
    mod="medea",
    llm_judge='o1-mini'
):
    """
    Logs the success rate and extracts code snippet, output, and timing information.

    Parameters:
        task (str): The task type (targetID, sl, immune_response).
        query (str): The user instruction only (user_instruction).
        llm_query (str): The full query including experiment instruction for LLM judge.
        llm_judge_prompt (str): Template for LLM judge evaluation.
        response_dict (dict or str): The response containing code snippet and output or the direct output.
        start_time (float): The timestamp when the process started.
        end_time (float): The timestamp when the process ended.
        success_count (dict): Dictionary tracking success counts.
        gene_list (list, optional): A list of genes to check for promotion.
        mod (str): Model mode (medea, multi_round_discussion, llm).
        llm_judge (str): LLM model for judging outputs.

    Returns:
        tuple: Updated success count and decision outputs (cg_decision, reason_decision, hypo_decision, llm_decision).
    """
    
    success_count["total"] = success_count.get("total", 0) + 1

    if mod == "medea":
        # Update success count if the response is a dictionary with valid keys
        p_response, cg_response, reason_response, hyp_response, llm_response = "None", "None", "None", "None", "None"

        if isinstance(response_dict, dict):
            if "P" in response_dict:
                if response_dict["P"] != None:
                    p_response = response_dict["P"] 
            # Support both 'CG' (legacy) and 'PA' (new medea core) keys
            if "CG" in response_dict:
                if response_dict["CG"] != None:
                    cg_response = response_dict["CG"]
            elif "PA" in response_dict:
                if response_dict["PA"] != None:
                    cg_response = response_dict["PA"]
            if "R" in response_dict:
                if response_dict["R"] != None:
                    reason_response = response_dict["R"]
            if "final" in response_dict:
                hyp_response = response_dict["final"]
            if "llm" in response_dict:
                llm_response = response_dict["llm"]
        else:
            cg_response = reason_response = hyp_response = response_dict = llm_response
        
        if p_response != "None":
            success_count["P"] = success_count.get("P", 0) + 1
        if isinstance(cg_response, dict):
            success_count["CG"] = success_count.get("CG", 0) + 1
            code_snippet = cg_response.get('code_snippet', 'None')
            executed_output = cg_response.get('executed_output', 'None')
        else:
            code_snippet = executed_output = cg_decision = cg_response

        if isinstance(reason_response, dict):
            reason_dict = reason_response.get('user_query', {})
            if reason_dict is not None:
                success_count["R"] = success_count.get("R", 0) + 1
                reason_output = reason_dict.get('answer', 'None')
            else:
                reason_output = "None"
        else:
            reason_output = reason_response
        

        llm_decision = chat_completion(llm_judge_prompt.format(user_query=llm_query, reasoning_result=llm_response), temperature=0, model=llm_judge)
        cg_decision = chat_completion(llm_judge_prompt.format(user_query=llm_query, reasoning_result=executed_output), temperature=0, model=llm_judge)
        reason_decision = chat_completion(llm_judge_prompt.format(user_query=llm_query, reasoning_result=reason_output), temperature=0, model=llm_judge)
        hypo_decision = chat_completion(llm_judge_prompt.format(user_query=llm_query, reasoning_result=hyp_response), temperature=0, model=llm_judge)
    elif mod == 'multi_round_discussion':
        hyp_response, llm_hyp_response = response_dict['final'], response_dict['llm']
        hypo_decision = chat_completion(llm_judge_prompt.format(user_query=llm_query, reasoning_result=hyp_response), temperature=0, model=llm_judge)
        llm_decision = chat_completion(llm_judge_prompt.format(user_query=llm_query, reasoning_result=llm_hyp_response), temperature=0, model=llm_judge)
        return success_count, None, None, hypo_decision.lower(), llm_decision.lower()
    else:
        llm_response = response_dict['llm']
        llm_decision = chat_completion(llm_judge_prompt.format(user_query=llm_query, reasoning_result=llm_response), temperature=0, model=llm_judge)
        return success_count, None, None, None, llm_decision.lower()


    # Calculate the execution duration
    duration = end_time - start_time

    cg_decision = cg_decision.lower().strip('\n')
    reason_decision = reason_decision.lower().strip('\n')
    hypo_decision = hypo_decision.lower().strip('\n')
    llm_decision = llm_decision.lower().strip('\n')
    

    # Log the code snippet, output, and duration
    print(f"[User Input]: {query}\n", flush=True)
    print(f"[CG] Code: {code_snippet}\n", flush=True)
    print(f"[CG] Output: {executed_output}\n", flush=True)
    print(f"[Reason] Output: {reason_output}\n", flush=True)
    print(f"[Hypo] Output: {hyp_response}\n", flush=True)
    print(f'[LLM (backbone)] Output: {llm_response}\n', flush=True)

    print(f'[LLM (backbone)] Decision: {llm_decision}', flush=True)
    print(f"[CG] Decision: {cg_decision}", flush=True)
    print(f"[Reason] Decision: {reason_decision}", flush=True)
    print(f"[Hypo] Decision: {hypo_decision}", flush=True)
    print(f"[Total] Time: {duration}s\n", flush=True)
    
    # if gene_list and type(gene_list) == str:
    #     gene_list = ast.literal_eval(gene_list)

    if cg_decision is None: cg_decision = "None"
    if reason_decision is None: reason_decision = "None"
    if hypo_decision is None: hypo_decision = "None"
    return success_count, cg_decision, reason_decision, hypo_decision, llm_decision


def update_acc_dict(acc_dict, task, counter, *args):
    # print("update_acc_dict called with args:", args)  # Debug print
    init_dict = {
        'cg_count': 0, 'reason_count': 0, 'hypo_count': 0, 'gpt_count': 0, 'total': 0,
        'cg_abstain': 0, 'reason_abstain': 0, 'hypo_abstain': 0, 'gpt_abstain': 0,
        'cg_failed': 0, 'reason_failed': 0, 'hypo_failed': 0, 'gpt_failed': 0,
    }
    
    if task == "targetID":
        if "disease" not in acc_dict:
            acc_dict["disease"] = {}
        if "celltype" not in acc_dict:
            acc_dict["celltype"] = {}
    
        if args != None:
            celltype, disease = args
            if disease not in acc_dict["disease"]:
                acc_dict["disease"][disease] = init_dict.copy()
            acc_dict["disease"][disease][counter] += 1

            if celltype not in acc_dict["celltype"]:
                acc_dict["celltype"][celltype] = init_dict.copy()
            acc_dict["celltype"][celltype][counter] += 1
        
    
    if task == "sl":
        if "cell_line" not in acc_dict:
            acc_dict["cell_line"] = {}
        if "gene" not in acc_dict:
            acc_dict["gene"] = {}
        if "interaction" not in acc_dict:
            acc_dict["interaction"] = {}
        if args != None:
            candidate_genes, interaction, cell_line = args
            # print("Candidate Genes:", candidate_genes)  # Debug print
            # print("Cell Line:", cell_line)  # Debug print
            if cell_line not in acc_dict["cell_line"]:
                acc_dict["cell_line"][cell_line] = init_dict.copy()
            acc_dict["cell_line"][cell_line][counter] += 1

            if interaction not in acc_dict["interaction"]:
                acc_dict["interaction"][interaction] = init_dict.copy()
            acc_dict["interaction"][interaction][counter] += 1

            for gene in candidate_genes:
                if gene not in acc_dict["gene"]:
                    acc_dict["gene"][gene] = init_dict.copy()
                acc_dict["gene"][gene][counter] += 1
    
    if task == "immune_response":
        if args != None:
            _, tmb, ngb, pheno, y = args
            if tmb < 10 and y == 'R':
                if "ICI=R, tmb<10" not in acc_dict:
                    acc_dict["ICI=R, tmb<10"] = init_dict.copy()
                acc_dict["ICI=R, tmb<10"][counter] += 1

            if tmb >= 10 and y == 'NR':
                if "ICI=NR, tmb>=10" not in acc_dict:
                    acc_dict["ICI=NR, tmb>=10"] = init_dict.copy()
                acc_dict["ICI=NR, tmb>=10"][counter] += 1

            if (tmb < 10 and ngb < 10) and y == 'R':
                if "ICI=R, tmb/ngb<10" not in acc_dict:
                    acc_dict["ICI=R, tmb/ngb<10"] = init_dict.copy()
                acc_dict["ICI=R, tmb/ngb<10"][counter] += 1

            if (tmb >= 10 and ngb >= 10) and y == 'NR':
                if "ICI=NR, tmb/ngb>=10" not in acc_dict:
                    acc_dict["ICI=NR, tmb/ngb>=10"] = init_dict.copy()
                acc_dict["ICI=NR, tmb/ngb>=10"][counter] += 1

            if pheno == 'inflamed' and y == 'NR':
                if "inflamed-NR" not in acc_dict:
                    acc_dict["inflamed-NR"] = init_dict.copy()
                acc_dict["inflamed-NR"][counter] += 1

            if pheno == 'desert' and y == 'R':
                if "desert-R" not in acc_dict:
                    acc_dict["desert-R"] = init_dict.copy()
                acc_dict["desert-R"][counter] += 1

            if pheno == 'excluded' and y == 'R':
                if "excluded-R" not in acc_dict:
                    acc_dict["excluded-R"] = init_dict.copy()
                acc_dict["excluded-R"][counter] += 1


def _print_perf_table(label, v):
    """Print a formatted performance table for a given acc_dict entry."""
    t = v['total']
    if t == 0:
        return
    modules = [
        ("LLM (bb)", 'gpt_count', 'gpt_abstain', 'gpt_failed'),
        ("Medea CG", 'cg_count', 'cg_abstain', 'cg_failed'),
        ("Medea R",  'reason_count', 'reason_abstain', 'reason_failed'),
        ("Medea full", 'hypo_count', 'hypo_abstain', 'hypo_failed'),
    ]
    print(f"\n======== {label} (n={t}) ========", flush=True)
    print(f"{'Module':<14} {'Acc (non-abs)':>16} {'Answered':>10} {'Abstain%':>10} {'Failed%':>10}", flush=True)
    print("-" * 64, flush=True)
    for name, cnt_k, abs_k, fail_k in modules:
        correct = v.get(cnt_k, 0)
        abstain = v.get(abs_k, 0)
        failed = v.get(fail_k, 0)
        answered = t - abstain
        acc_str = f"{correct/answered:.4f} ({correct}/{answered})" if answered > 0 else "N/A"
        abs_pct = abstain / t * 100
        fail_pct = failed / t * 100
        print(f"{name:<14} {acc_str:>16} {answered:>4}/{t:<4} {abs_pct:>8.1f}% {fail_pct:>8.1f}%", flush=True)


def log_acc_dict(acc_dict, task, *args):
    if task == "targetID":
        celltype, disease = args
        for d, v in acc_dict["disease"].items():
            _print_perf_table(f"Disease: {d}", v)
        for c, v in acc_dict["celltype"].items():
            _print_perf_table(f"Celltype: {c}", v)

    if task == "sl":
        candidate_genes, interaction, cell_line = args
        for c, v in acc_dict["interaction"].items():
            _print_perf_table(f"Interaction: {c}", v)
        for c, v in acc_dict["cell_line"].items():
            _print_perf_table(f"Cell line: {c}", v)
        for g in candidate_genes:
            v = acc_dict["gene"][g]
            _print_perf_table(f"Gene: {g}", v)
    
    if task == "immune_response":
        for k, v in acc_dict.items():
            if isinstance(v, dict) and 'total' in v:
                _print_perf_table(k, v)

    if task == "sl-summary":
        acc_set = set()
        for k, v in acc_dict["gene"].items():
            acc_set.add((k, v['gpt_count']/v['total'], v['total']))
        filtered_gene_data = {item for item in acc_set if item[2] >= 1}
        sorted_gene_data = sorted(filtered_gene_data, key=lambda x: x[1], reverse=True)
        print("Gene acc:", flush=True)
        for k, acc, count in sorted_gene_data:
            print(f"Gene: {k} | Acc: {acc:4} | Appearance: {count}", flush=True)

def evaluate_prediction(
        task,
        y,  
        acc_dict, 
        executed_output, 
        reason_output,
        final_output, 
        llm_feedback,
        success_count,
        cg_count,
        reason_count,
        hypo_count,
        gpt_count,
        *args
    ):  

    def _check_and_count(output, y, task, count, count_key, abstain_key, failed_key):
        """Classify output and update counters. Returns updated count."""
        cls = _classify_decision(output) if isinstance(output, str) else "abstain"
        if cls == "abstain":
            update_acc_dict(acc_dict, task, abstain_key, *args)
            return count, cls
        if cls == "failed":
            update_acc_dict(acc_dict, task, failed_key, *args)
            return count, cls
        if isinstance(output, str) and y.lower() in output.lower():
            if task != "immune_response" or y.lower() == output.lower():
                count += 1
                update_acc_dict(acc_dict, task, count_key, *args)
        return count, cls

    cg_count, cg_cls = _check_and_count(executed_output, y, task, cg_count, 'cg_count', 'cg_abstain', 'cg_failed')
    reason_count, r_cls = _check_and_count(reason_output, y, task, reason_count, 'reason_count', 'reason_abstain', 'reason_failed')
    hypo_count, h_cls = _check_and_count(final_output, y, task, hypo_count, 'hypo_count', 'hypo_abstain', 'hypo_failed')
    gpt_count, g_cls = _check_and_count(llm_feedback, y, task, gpt_count, 'gpt_count', 'gpt_abstain', 'gpt_failed')

    total = success_count.get('total', 0)
    update_acc_dict(acc_dict, task, 'total', *args)

    def _sym(cls, output, y):
        if cls == "abstain": return "~"
        if cls == "failed": return "!"
        if isinstance(output, str) and y.lower() in output.lower(): return "\u2713"
        return "\u2717"

    case_num = total
    print(
        f"[Case {case_num}] y={y} | "
        f"bb={g_cls} {_sym(g_cls, llm_feedback, y)} | "
        f"CG={cg_cls} {_sym(cg_cls, executed_output, y)} | "
        f"R={r_cls} {_sym(r_cls, reason_output, y)} | "
        f"full={h_cls} {_sym(h_cls, final_output, y)}",
        flush=True
    )
    
    log_acc_dict(acc_dict, task, *args)
    return acc_dict, cg_count, reason_count, hypo_count, gpt_count


def split_df_after_checkpoint(
    df: pd.DataFrame,
    checkpoint: tuple,
    task: str
) -> pd.DataFrame:
    """
    Return the slice of `df` that comes after the first occurrence of `checkpoint`.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, assumed to contain all columns in `cols` in order.
    checkpoint : tuple
        A 4-tuple of values (one per column in `cols`) that marks the split point.
    cols : list of str, optional
        The column names, in order, against which to match the checkpoint.

    Returns
    -------
    pd.DataFrame
        The sub-DataFrame starting immediately after the checkpoint row.
        If no match is found, returns the original df unmodified.
    """
    # Build a boolean mask of rows matching all checkpoint values
    mask = True
    if task == "targetID":
        cols = ['candidate_genes', 'celltype', 'disease', 'y']
    elif task == "sl":
        if 'cell_line' in df.columns:
            cols = ['gene_a', 'gene_b', 'cell_line', 'interaction']
        else:
            cols = ['gene_a', 'gene_b', 'condition', 'interaction']
    elif task == "immune_response":
        cols = ["TMB (FMOne mutation burden per MB)", "Neoantigen burden per MB", "Immune phenotype", "response_label"]

    
    for col_name, value in zip(cols, checkpoint):
        if pd.isna(value):
            # Handle NaN values specially
            mask &= pd.isna(df[col_name])
        else:
            mask &= (df[col_name] == value)

    # Find the first index where all match
    matches = df.index[mask]
    if matches.empty:
        # checkpoint not found → return original
        print(f"[split_df_after_checkpoint] Checkpoint not found", flush=True)
        return df.copy().reset_index(drop=True)

    split_idx = matches[0] + 1
    print(f"[split_df_after_checkpoint] Statring from idx [{split_idx} / {len(df)}]", flush=True)
    return df.iloc[split_idx: ].reset_index(drop=True)
