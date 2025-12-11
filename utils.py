from medea.tool_space.gpt_utils import chat_completion
from evaluation.query_template import *
import ast, os, random
import numpy as np
import pandas as pd

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
                g_a = row.get('gene_a', '')
                g_b = row.get('gene_b', '')
                cell_line = row.get('cell_line', '')
                interaction = row.get('interaction', '')
                user_instruction = row.get('user_question', '')
                
                # Extract experiment instruction from full_query
                full_query = row.get('full_query', '')
                if full_query and full_query != user_instruction:
                    experiment_instruction = full_query[len(user_instruction):].strip()
                else:
                    experiment_instruction = None
                    
                yield ([g_a, g_b], user_instruction, experiment_instruction, interaction, [g_a, g_b], interaction, cell_line)
        
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

                elif sl_source == 'samson':
                    g_a, g_b, cell_line, y = sample

                    gpt_prompt = user_template.format(
                        gene_a=g_a,
                        gene_b=g_b,
                        cell_line=cell_line
                    )

                user_instruction = chat_completion(gpt_prompt, temperature=1, model=rephrase_model).strip('""')
                experiment_instruction = chat_completion(EXPERIMENT_INSTRUCTION_REPHRASE_TEMPLATE.format(role="biologist", task_instruction_template=agent_template), model=rephrase_model, temperature=1)
                interaction = y.strip("['']").lower()

                yield ([g_a, g_b], user_instruction, experiment_instruction, interaction, [g_a, g_b], interaction, cell_line)

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
        "medea":             ['cg', 'r', 'full', 'llm-bb'],
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
        cg_out, r_out, full_out, llm_bb_out = log_pack
        new_row.update({
            'cg': cg_out,
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
    gpt_query,
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
        gpt_query (str): The full query including experiment instruction for LLM judge.
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
            if "CG" in response_dict:
                if response_dict["CG"] != None:
                    cg_response = response_dict["CG"] 
            if "R" in response_dict:
                if response_dict["R"] != None:
                    reason_response = response_dict["R"]
            if "CGRH" in response_dict:
                hyp_response = response_dict["CGRH"]
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
        

        llm_decision = chat_completion(llm_judge_prompt.format(user_query=gpt_query, reasoning_result=llm_response), temperature=0, model=llm_judge)
        cg_decision = chat_completion(llm_judge_prompt.format(user_query=gpt_query, reasoning_result=executed_output), temperature=0, model=llm_judge)
        reason_decision = chat_completion(llm_judge_prompt.format(user_query=gpt_query, reasoning_result=reason_output), temperature=0, model=llm_judge)
        hypo_decision = chat_completion(llm_judge_prompt.format(user_query=gpt_query, reasoning_result=hyp_response), temperature=0, model=llm_judge)
    elif mod == 'multi_round_discussion':
        hyp_response, llm_hyp_response = response_dict['CGRH'], response_dict['llm']
        hypo_decision = chat_completion(llm_judge_prompt.format(user_query=gpt_query, reasoning_result=hyp_response), temperature=0, model=llm_judge)
        llm_decision = chat_completion(llm_judge_prompt.format(user_query=gpt_query, reasoning_result=llm_hyp_response), temperature=0, model=llm_judge)
        return success_count, None, None, hypo_decision.lower(), llm_decision.lower()
    else:
        llm_response = response_dict['llm']
        llm_decision = chat_completion(llm_judge_prompt.format(user_query=gpt_query, reasoning_result=llm_response), temperature=0, model=llm_judge)
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
    init_dict = {'cg_count': 0, 'reason_count': 0, 'hypo_count': 0, 'gpt_count': 0, 'total': 0,}
    
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


def log_acc_dict(acc_dict, task, *args):
    if task == "targetID":
        celltype, disease = args
        print(f"\nDiseases: {disease} | Celltype: {celltype}", flush=True)
        
        for d, v in acc_dict["disease"].items():
            print(
                f"Disease: {d} | "
                f"LLM (backbone) Acc: {v['gpt_count']/v['total']:.4} | "
                f"CG Acc: {v['cg_count']/v['total']:.4} | "
                f"R Acc: {v['reason_count']/v['total']:.4} | "
                f"CGRH Acc: {v['hypo_count']/v['total']:.4}", 
                flush=True
            )
        for c, v in acc_dict["celltype"].items():
            print(
                f"Cell type: {c} | "
                f"LLM (backbone) Acc: {v['gpt_count']/v['total']:.4} | "
                f"CG Acc: {v['cg_count']/v['total']:.4} | "
                f"R Acc: {v['reason_count']/v['total']:.4} | "
                f"CGRH Acc: {v['hypo_count']/v['total']:.4}", 
                flush=True
            )

    if task == "sl":
        candidate_genes, interaction, cell_line = args
        print(f"\nGene Pair: {candidate_genes} | Cell line: {cell_line} | Interaction: {interaction}", flush=True)
        print("======================================")
        for c, v in acc_dict["interaction"].items():
            print(
                f"Interaction: {c} | "
                f"LLM (backbone) Acc: {v['gpt_count']/v['total']:.4} | "
                f"CG Acc: {v['cg_count']/v['total']:.4} | "
                f"R Acc: {v['reason_count']/v['total']:.4} | "
                f"CGRH Acc: {v['hypo_count']/v['total']:.4}", 
                flush=True
            )
        print("======================================")
        for c, v in acc_dict["cell_line"].items():
            print(
                f"Cell line: {c} | "
                f"LLM (backbone) Acc: {v['gpt_count']/v['total']:.4} | "
                f"CG Acc: {v['cg_count']/v['total']:.4} | "
                f"R Acc: {v['reason_count']/v['total']:.4} | "
                f"CGRH Acc: {v['hypo_count']/v['total']:.4}", 
                flush=True
            )
        for g in candidate_genes:
            v = acc_dict["gene"][g]
            print(
                f"Gene: {g} | "
                f"LLM (backbone) Acc: {v['gpt_count']/v['total']:.4} | "
                f"CG Acc: {v['cg_count']/v['total']:.4} | "
                f"R Acc: {v['reason_count']/v['total']:.4} | "
                f"CGRH Acc: {v['hypo_count']/v['total']:.4}", 
                flush=True
            )
    
    if task == "immune_response":
        # print(acc_dict)
        for k, v in acc_dict.items():
            print("======================================")
            print(
                f"Cases: {k} ({v['total']}) | "
                f"LLM (backbone) Acc: {v['gpt_count']/v['total']:.4} | "
                f"CG Acc: {v['cg_count']/v['total']:.4} | "
                f"R Acc: {v['reason_count']/v['total']:.4} | "
                f"CGRH Acc: {v['hypo_count']/v['total']:.4}", 
                flush=True
            )
        print('\n')

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
        hypo_output, 
        gpt_feedback,
        success_count,
        cg_count,
        reason_count,
        hypo_count,
        gpt_count,
        *args
    ):  
        
    if type(executed_output) is str and y.lower() in executed_output.lower():
        if task != "immune_response" or y.lower() == executed_output.lower():
            cg_count += 1
            update_acc_dict(acc_dict, task, 'cg_count', *args)

    if type(reason_output) is str and y.lower() in reason_output.lower():
        if task != "immune_response" or y.lower() == reason_output.lower():
            reason_count += 1
            update_acc_dict(acc_dict, task, 'reason_count', *args)

    if type(hypo_output) is str and y.lower() in hypo_output.lower():
        if task != "immune_response" or y.lower() == hypo_output.lower():
            hypo_count += 1
            update_acc_dict(acc_dict, task, 'hypo_count', *args)
    
    if type(gpt_feedback) is str and y.lower() in gpt_feedback.lower():
        if task != "immune_response" or y.lower() == gpt_feedback.lower():
            gpt_count += 1
            update_acc_dict(acc_dict, task, 'gpt_count', *args)
    
    total = success_count.get('total', 0)
    update_acc_dict(acc_dict, task, 'total', *args)

    cg_acc = cg_count/total
    reason_acc = reason_count/total
    hypo_acc = hypo_count/total
    gpt_acc = gpt_count/total

    p_sc = success_count.get('P', 0) / total
    cg_sc = success_count.get('CG', 0) / total
    r_sc = success_count.get('R', 0) / total
    
    print(
        f"\n[P] Succ Rate: {p_sc:.4f} | "
        f"[CG] Succ Rate: {cg_sc:.4f} | "
        f"[R] Succ Rate: {r_sc:.4f}"
    )
    
    print(
        f"\nContext: {args} |"
        f"y: {y} | "
        f"LLM (backbone): {gpt_feedback} | "
        f"Medea (CG): {executed_output} | "
        f"Medea (R): {reason_output} | "
        f"Medea (CGRH): {hypo_output}\n"
        f"LLM (backbone): {gpt_acc} | "
        f"Medea (CG): {cg_acc:.4f} | "
        f"Medea (R): {reason_acc:.4f} | "
        f"Medea (CGRH): {hypo_acc:.4f}\n ",
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
        cols = ['gene_a', 'gene_b', 'cell_line', 'interaction']
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
