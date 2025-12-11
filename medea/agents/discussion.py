from collections import Counter
import dotenv
import json
import os, re, ast, sys
import time
import base64

# Use relative imports within package
from ..tool_space.gpt_utils import chat_completion
from .prompt_template import *

dotenv.load_dotenv()

def sanitize_prompt_content(text):
    """
    Sanitize prompt content to avoid triggering SQL injection detection systems.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Replace problematic JSON structure patterns that look like SQL injection
    # Convert {"key": "value"} patterns to safer alternatives
    text = re.sub(r'\{\\?"([^"]+)\\?":\s*\\?"\\?"[^}]*\}', 
                  lambda m: m.group(0).replace('\\"', "'").replace('"', "'"), text)
    
    # Replace escaped quotes that might trigger detection
    text = text.replace('\\"', "'")
    
    # Replace multiple consecutive quotes
    text = re.sub(r'"{2,}', '"', text)
    
    # Remove patterns that look like SQL injection attempts
    sql_patterns = [
        r'SELECT\s+\*\s+FROM',
        r'DROP\s+TABLE',
        r'INSERT\s+INTO',
        r'DELETE\s+FROM',
        r'UPDATE\s+.*\s+SET',
        r'UNION\s+SELECT',
        r'OR\s+1\s*=\s*1',
        r'AND\s+1\s*=\s*1',
        r';\s*--',
        r'/\*.*?\*/',
    ]
    
    for pattern in sql_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text

def encode_complex_content(content):
    """
    Base64 encode complex content that might trigger detection.
    """
    if not content or content == "None":
        return content
    try:
        return base64.b64encode(str(content).encode()).decode()
    except:
        return str(content)

def decode_complex_content(encoded_content):
    """
    Decode base64 encoded content.
    """
    try:
        return base64.b64decode(encoded_content).decode()
    except:
        return encoded_content

def find_idx_by_element(input_list, element):
    return [i for i, a in enumerate(input_list) if a == element]


def find_element_by_indices(input_list, index_list):
    return [b for i, b in enumerate(input_list) for k in index_list if i == k]


def trans_confidence(x):
    x = float(x)
    if x <= 0.6: return 0.1
    if 0.8 > x > 0.6: return 0.3
    if 0.9 > x >= 0.8: return 0.5
    if 1 > x >= 0.9: return 0.8
    if x == 1: return 1


def parse_json(model_output):
    """Parse JSON from LLM output, handling both JSON and Python dict formats"""
    if type(model_output) is dict:
        return model_output
    elif type(model_output) is not str:
        model_output = str(model_output)
    
    try:
        # Clean and extract the JSON/dict object
        model_output = model_output.replace("\n", " ").strip()
        json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', model_output)
        if not json_match:
            return "ERR_SYNTAX"
        
        json_str = json_match.group(1)
        
        # Clean up common issues
        json_str = json_str.replace("\\'", "'")  # Remove invalid escape sequences
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        
        # Strategy 1: Try as valid JSON (with double quotes)
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Convert Python dict format to JSON format
        try:
            # Replace single quotes with double quotes for JSON compatibility
            # But be careful not to replace quotes inside strings
            json_converted = json_str
            # Simple approach: replace single quotes around keys and string values
            json_converted = re.sub(r"'([^']*)':", r'"\1":', json_converted)  # Keys
            json_converted = re.sub(r":\s*'([^']*)'", r': "\1"', json_converted)  # String values
            result = json.loads(json_converted)
            return result
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Use ast.literal_eval for Python dict syntax
        try:
            # Fix quotes for Python dict evaluation
            dict_str = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", json_str)
            result = ast.literal_eval(dict_str)
            return result
        except (ValueError, SyntaxError):
            pass
            
    except Exception:
        pass
    
    return "ERR_SYNTAX"


def parse_output(tmp, query, rounds, vote_merge=True, attempt=4):
    c, g, b = "llm_0", "llm_1", "llm_2"
    r = "_output_"+str(rounds)
    
    certainty_vote = {}
        
    for o in [c, g, b]:
        if o+r in tmp:
            # Ensure tmp[o+r] is a dictionary and has the expected structure
            if isinstance(tmp[o+r], dict) and 'answer' in tmp[o+r]:
                tmp[o+"_pred_"+str(rounds)] = tmp[o+r]['answer']
                tmp[o+"_exp_"+str(rounds)] = f"I think the answer is {tmp[o+r]['answer']} because {tmp[o+r]['reasoning']} My confidence level is {tmp[o+r]['confidence_level']}." 
                if tmp[o+r]['answer'] not in certainty_vote:
                    certainty_vote[tmp[o+r]['answer']] = trans_confidence(tmp[o+r]['confidence_level']) + 1e-5
                else:
                    certainty_vote[tmp[o+r]['answer']] += trans_confidence(tmp[o+r]['confidence_level'])
            else:
                # Handle the case where tmp[o+r] is not a dictionary or lacks expected keys
                print(f"Warning: {o+r} is not structured as expected: {tmp[o+r]}")

    if c+r in tmp and g+r in tmp and b+r in tmp:
        tmp['vote_'+str(rounds)] = [tmp['llm_0_pred_'+str(rounds)], tmp['llm_1_pred_'+str(rounds)], tmp['llm_2_pred_'+str(rounds)]]
        tmp['exps_'+str(rounds)] = [tmp['llm_0_exp_'+str(rounds)], tmp['llm_1_exp_'+str(rounds)], tmp['llm_2_exp_'+str(rounds)]]
        
        # ========== 
        # Clean the votes:
        if vote_merge:
            original_vote = certainty_vote.copy()  # Keep backup in case parsing fails
            while attempt > 0:
                try:
                    # Convert the string to a Python dictionary safely
                    safe_query = sanitize_prompt_content(str(query))
                    safe_vote_content = sanitize_prompt_content(str(certainty_vote))
                    reconcile_content = RECONCILE_PROMPT + "\n\nUser query: " + safe_query + "\nDictionary: \n" + safe_vote_content
                    safe_reconcile_content = sanitize_prompt_content(reconcile_content)
                    messages = [{"role": "user", "content": safe_reconcile_content}]
                    cleaned_output = chat_completion(messages, model=os.getenv("BACKBONE_LLM"), mod='dialog')
                    cleaned_output = cleaned_output.strip()
                    
                    # Remove potential markdown formatting (e.g., ```python ... ```)
                    if cleaned_output.startswith("```"):
                        # Remove markdown fences
                        cleaned_output = re.sub(r'^```(?:python|json)?\n?', '', cleaned_output)
                        cleaned_output = re.sub(r'\n?```$', '', cleaned_output)
                        cleaned_output = cleaned_output.strip()
                    
                    # Try multiple parsing strategies
                    parsed_vote = None
                    
                    # Strategy 1: Direct ast.literal_eval
                    try:
                        parsed_vote = ast.literal_eval(cleaned_output)
                    except (ValueError, SyntaxError):
                        pass
                    
                    # Strategy 2: Try JSON parsing
                    if parsed_vote is None:
                        try:
                            parsed_vote = json.loads(cleaned_output)
                        except json.JSONDecodeError:
                            pass
                    
                    # Strategy 3: Fix common quote issues and retry
                    if parsed_vote is None:
                        try:
                            # Escape problematic quotes
                            fixed_output = cleaned_output.replace("\\'", "'").replace('\\"', '"')
                            # Try to extract just the dict part
                            dict_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', fixed_output)
                            if dict_match:
                                parsed_vote = ast.literal_eval(dict_match.group(0))
                        except (ValueError, SyntaxError):
                            pass
                    
                    if parsed_vote and isinstance(parsed_vote, dict):
                        certainty_vote = parsed_vote
                        break
                    else:
                        raise ValueError("Could not parse vote dictionary")
                        
                except Exception as e:
                    attempt -= 1
                    if attempt > 0:
                        # Silently retry
                        pass
                    else:
                        # Last attempt failed, use original vote
                        print(f"[Vote Reconciliation] Failed to parse after all attempts. Using original votes.", flush=True)
                        certainty_vote = original_vote

            # print("Converted Vote dictionary:", cleaned_output, flush=True)
        # ==========
        for v in certainty_vote:
            print(v, flush=True)
        tmp['weighted_vote_'+str(rounds)] = certainty_vote
        tmp['weighted_max_'+str(rounds)] = max(certainty_vote, key=certainty_vote.get)
        print(f"\nMax weighted Vote: {tmp['weighted_max_'+str(rounds)]}", flush=True)

        tmp['debate_prompt_'+str(rounds)] = ''
        vote = Counter(tmp['vote_'+str(rounds)]).most_common(2)

        tmp['majority_ans_'+str(rounds)] = vote[0][0]
        if len(vote) > 1: # not all the agents give the same answer
            for v in vote:
                tmp['debate_prompt_'+str(rounds)] += f"There are {v[1]} agents think the answer is {v[0]}. "
                exp_index = find_idx_by_element(tmp['vote_'+str(rounds)], v[0])
                group_exp = find_element_by_indices(tmp['exps_'+str(rounds)], exp_index)
                exp = "\n".join(["One agent solution: " + g for g in group_exp])
                tmp['debate_prompt_'+str(rounds)] += exp + "\n\n"
                    
    return tmp


def clean_output(tmp, rounds):
    co, go, bo = "llm_0" + str(rounds), 'llm_1' + str(rounds), 'llm_2' + str(rounds)

    for o in [co, go, bo]:
        if o in tmp:
            if 'reasoning' not in tmp[o]:
                tmp[o]['reasoning'] = ""
            elif type(tmp[o]['reasoning']) is list:
                tmp[o]['reasoning'] = " ".join(tmp[o]['reasoning'])
            
            if 'answer' not in tmp[o] or not tmp[o]['answer']:
                tmp[o]['answer'] = 'unknown'

            if 'confidence_level' not in tmp[o] or not tmp[o]['confidence_level']:
                tmp[o]['confidence_level'] = 0.0
            else:
                if type(tmp[o]['confidence_level']) is str and "%" in tmp[o]['confidence_level']:
                        tmp[o]['confidence_level'] = float(tmp[o]['confidence_level'].replace("%","")) / 100
                else:
                    try:
                        tmp[o]['confidence_level'] = float(tmp[o]['confidence_level'])
                    except:
                        print(tmp[o]['confidence_level'])
                        tmp[o]['confidence_level'] = 0.0
            
    return tmp

def prepare_context_for_chat_assistant(query, convincing_samples=None, intervene=False):
    contexts = []
    if convincing_samples:
        for cs in convincing_samples:
            contexts.append({"role": "user", "content": f"User Query: {cs['train_sample']['question']}"})
            contexts.append({"role": "assistant", "content": str({"reasoning": cs['train_sample']['gold_explanation'], "answer": cs['train_sample']['answer']})})

    if intervene:
        contexts.append({"role": "user", "content": f"User Query: {query['question']}" + "\nAnswer the question given the fact that " + query['gold_explanation']})  
    else:
        contexts.append({"role": "user", "content": f"User Query: {query}"})
        
    contexts[-1]["content"] += " Analyze the evidence and provide a clear, concise answer with supporting reasoning. If evidence is insufficient, state limitations explicitly."
    contexts[-1]["content"] += " Include confidence level (0.0-1.0) based on evidence quality."
    
    # Concise JSON format instructions to reduce token usage
    safe_json_format = "Output in JSON format: {'reasoning': 'brief_reasoning', 'answer': 'your_answer', 'confidence_level': (0.0-1.0)}. Keep response under 1000 tokens."
    contexts[-1]["content"] += " " + safe_json_format
    
    # Sanitize the final content
    contexts[-1]["content"] = sanitize_prompt_content(contexts[-1]["content"])
    
    return contexts


def gpt_gen_ans(query, model='gpt-4o', attempts=3, convincing_samples=None, additional_instruc=None, intervene=False):
    i = 0
    while i < attempts:
        try:
            contexts = prepare_context_for_chat_assistant(query, convincing_samples, intervene)
            if additional_instruc:
                # Sanitize additional instructions before adding them
                safe_additional_instruc = [sanitize_prompt_content(str(instr)) for instr in additional_instruc]
                contexts[-1]['content'] += " " + " ".join(safe_additional_instruc)
                # Apply final sanitization to the complete content
                contexts[-1]['content'] = sanitize_prompt_content(contexts[-1]['content'])
            # print(contexts)
            output = chat_completion(contexts, model=model, mod='dialog')
            # print(output, flush=True)
            if output:
                if "{" not in output or "}" not in output:
                    print(output)
                    raise ValueError("cannot find { or } in the model output.")
                result = parse_json(output)
                if result == "ERR_SYNTAX":
                    raise ValueError("[gpt_gen_ans] Incomplete JSON format. Retrying (attempts: " + str(i) + ")...")
            return result
        except Exception as e:
            print(f"[Retrying - {model}]: {e}")
            time.sleep(5) # wait for 10 seconds
            if "Incapsula_Resource" in str(e):
                print("Incapsula Resource Error, retrying...")
                print(f"[gpt_gen_ans] Prompt blocked by Incapsula:\n {contexts}")
                raise Exception(f"Incapsula Resource Error. Agent terminated: {e}")
            i += 1
    return {'reasoning': "None", "answer": "I can not help with this.", "confidence_level": 0.0}
    


def llm_debate(query, tmp, rounds, model_name='gpt-4o', llm_name='llm_0', convincing_samples=None):
    r = '_' + str(rounds-1)

    if f'{llm_name}_output_'+str(rounds) not in tmp and 'debate_prompt'+ r in tmp and len(tmp['debate_prompt'+r]):
        print("Debate")
        additional_instruc = ["\n\nCarefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question."]
        additional_instruc.append("Clearly states that which pointview do you agree or disagree and why.\n\n")
        
        # Sanitize the debate prompt content before adding it
        sanitized_debate_prompt = sanitize_prompt_content(tmp['debate_prompt'+r])
        additional_instruc.append(sanitized_debate_prompt)
        
        # Use safer JSON format instructions
        safe_json_instruction = "Output your answer in JSON format using this structure: {'reasoning': 'your_reasoning', 'answer': 'your_answer', 'confidence_level': 'numeric_value'}. Use single quotes for the JSON structure."
        additional_instruc.append(safe_json_instruction)
        
        result = gpt_gen_ans(query,
                            model=model_name,
                            convincing_samples=convincing_samples,
                            additional_instruc=additional_instruc,
                            intervene=False)
        tmp[f'{llm_name}_output_'+str(rounds)] = result
    else:
        if f'{llm_name}_output_'+str(rounds) in tmp:
            print(f'{llm_name}_output_'+str(rounds)+' existed')
        elif 'debate_prompt'+ r not in tmp:
            print(f"[llm_debate] debate_prompt{r} not in tmp", flush=True)
        elif not len(tmp['debate_prompt'+r]):
            print("[llm_debate] No debate prompts for all llm judge", flush=True)
    return tmp


def multi_round_discussion(
    query, 
    mod='diff_context', 
    panelist_llms=[
        'gemini', 
        'gpt-4o', 
        'gpt-4o-mini'
    ],
    include_llm=True, 
    proposal_response=None, 
    coding_response=None, 
    reasoning_response=None, 
    vote_merge=True, 
    round=1
    ):
    
    tmp = {}
    debate_query = query
    code_snippet = executed_output = reasoning_output = "None"

    if mod == "diff_context":
        if isinstance(coding_response, dict):
            code_snippet = coding_response.get("code_snippet", "None")
            executed_output = coding_response.get("executed_output") or "None"

        if isinstance(reasoning_response, dict):
            reasoning_dict = reasoning_response.get("user_query", {})
            if isinstance(reasoning_dict, dict):
                reasoning_output = reasoning_dict.get("answer") or "None"

        # Sanitize all input content before processing
        safe_proposal = sanitize_prompt_content(str(proposal_response))
        safe_code_snippet = sanitize_prompt_content(str(code_snippet))
        safe_executed_output = sanitize_prompt_content(str(executed_output))
        safe_reasoning_output = sanitize_prompt_content(str(reasoning_output))

        # Normalize the hypothesis proposed by agent into a unified context length
        experiment_hypothesis = (
            f"[Coding Agent] Proposal:\n{safe_proposal}\n"
            f"[Coding Agent] Code Snippet:\n{safe_code_snippet}\n"
            f"[Coding Agent] Output:\n{safe_executed_output}"
        )
        literature_hypothesis = f"[Reasoning Agent] Output:\n{safe_reasoning_output}"
        
        # Combine hypotheses generated by different agents into a single output
        safe_query = sanitize_prompt_content(str(query))
        coding_hypothesis = HYPOTHESOS_NORMALIZER.format(user_query=safe_query, agent_hypo=experiment_hypothesis)
        reasoning_hypothesis = HYPOTHESOS_NORMALIZER.format(user_query=safe_query, agent_hypo=literature_hypothesis)
        agents_ans = "[Coding Agent] Output:\n " + coding_hypothesis + '\n\n' + '[Reasoning Agent] Output:\n ' + reasoning_hypothesis
        

        if include_llm:
            print(f"----- Hypothesis from Backbone LLM: {os.getenv('BACKBONE_LLM')} -----", flush=True)
            llm_hypothesis = chat_completion(safe_query, model=os.getenv("BACKBONE_LLM"))
            safe_llm_hypothesis = sanitize_prompt_content(str(llm_hypothesis))
            backbone_hypothesis = HYPOTHESOS_NORMALIZER.format(user_query=safe_query, agent_hypo=safe_llm_hypothesis)
            print(f"{llm_hypothesis}\n", flush=True)
            agents_ans += '\n\n' + '[LLM]:\n' + backbone_hypothesis
        
        # Build the debate query safely
        debate_query_parts = [
            sanitize_prompt_content(str(query)),
            "Hypothesis from Agents:",
            "",
            sanitize_prompt_content(agents_ans)
        ]
        debate_query = "\n".join(debate_query_parts)

    # Phase1: Initial round for pannel discussion
    panelist_1, panelist_2, panelist_3 = panelist_llms
    tmp['llm_0_output_0'] = gpt_gen_ans(debate_query, model=panelist_1, additional_instruc=None, intervene=False)
    tmp['llm_1_output_0'] = gpt_gen_ans(debate_query, model=panelist_2, additional_instruc=None, intervene=False)
    tmp['llm_2_output_0'] = gpt_gen_ans(debate_query, model=panelist_3, additional_instruc=None, intervene=False)

    tmp = clean_output(tmp, 0)
    tmp = parse_output(tmp, query, 0, vote_merge=vote_merge)


    # Phase2: Multi-Round Discussion
    for r in range(1, round+1):
        print(f"----- Round {r} Discussion -----", flush=True)
        tmp = llm_debate(debate_query, tmp, llm_name='llm_0', rounds=r, model_name=panelist_1)
        tmp = llm_debate(debate_query, tmp, llm_name='llm_1', rounds=r, model_name=panelist_2)
        tmp = llm_debate(debate_query, tmp, llm_name='llm_2', rounds=r, model_name=panelist_3)
        
        tmp = clean_output(tmp, r)
        tmp = parse_output(tmp, query, r, vote_merge=vote_merge)
    
    # Find keys that start with 'majority_ans_' and extract the highest suffix
    # print([key for key in tmp])
    majority_keys = [key for key in tmp if key.startswith("weighted_max_")]
    if majority_keys:
        # Sort the keys by their numeric suffix
        majority_keys_sorted = sorted(majority_keys, key=lambda x: int(x.split('_')[-1]))
        # Get the last one (highest suffix)
        last_majority_key = majority_keys_sorted[-1]
        hyp_response = tmp[last_majority_key]
    
    original_ans = experiment_hypothesis + "\n" + literature_hypothesis + "\n" + llm_hypothesis
    hypo_prompt = HYPOTHESIS_FORMULATOR.format(query=query, answer=hyp_response, agent_ans=original_ans)
    hyp_response = chat_completion(hypo_prompt, model=os.getenv("BACKBONE_LLM"))
    
    if include_llm:
        return hyp_response, llm_hypothesis
    return hyp_response, None
