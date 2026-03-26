from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import dotenv
import json
import os, re, ast, sys
import time
import base64
from typing import Optional

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


def parse_llm_dict_output(output_str: str) -> Optional[dict]:
    """
    Robustly parse dictionary from LLM output using multiple strategies.
    
    Args:
        output_str: LLM output string potentially containing a dictionary
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not output_str or not isinstance(output_str, str):
        return None
    
    cleaned = output_str.strip()
    
    # Remove markdown code fences
    if "```" in cleaned:
        # Remove markdown with optional language specifier
        cleaned = re.sub(r'```(?:python|json|dict)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n?```\s*', '', cleaned)
        cleaned = cleaned.strip()
    
    # Extract first JSON/dict object from text
    # This handles cases like "Here is the result: {dict}"
    dict_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    match = re.search(dict_pattern, cleaned)
    if match:
        cleaned = match.group(0).strip()
    
    # Try parsing with multiple methods
    parse_methods = [
        # Method 1: Direct JSON parsing (most reliable)
        ("json.loads", lambda s: json.loads(s)),
        
        # Method 2: AST literal eval (handles Python dicts)
        ("ast.literal_eval", lambda s: ast.literal_eval(s)),
        
        # Method 3: Handle nested JSON (for {"normalized_votes": {...}})
        ("nested json", lambda s: json.loads(s).get("normalized_votes", json.loads(s))),
        
        # Method 4: Convert single quotes to double quotes for JSON
        ("json with quote fix", lambda s: json.loads(s.replace("'", '"'))),
        
        # Method 5: Handle escaped quotes
        ("unescape quotes", lambda s: ast.literal_eval(
            s.replace('\\"', '"').replace("\\'", "'")
        )),
        
        # Method 6: Aggressive cleanup for malformed JSON
        ("aggressive cleanup", lambda s: json.loads(
            s.replace("'", '"')
             .replace('True', 'true')
             .replace('False', 'false')
             .replace('None', 'null')
        )),
    ]
    
    for method_name, parser_func in parse_methods:
        try:
            result = parser_func(cleaned)
            if isinstance(result, dict) and len(result) > 0:
                return result
        except (json.JSONDecodeError, ValueError, SyntaxError, TypeError, AttributeError):
            continue
    
    return None


def reconcile_votes_with_llm(certainty_vote: dict, query: str, max_attempts: int = 4) -> dict:
    """
    Reconcile and normalize vote dictionary using LLM with robust parsing.
    
    Args:
        certainty_vote: Dictionary of {answer: weight}
        query: User query for context
        max_attempts: Maximum parsing attempts
        
    Returns:
        Reconciled dictionary or original if all attempts fail
    """
    # Skip if only one unique answer
    if len(certainty_vote) <= 1:
        print(f"[Vote Reconciliation] Only one unique answer, skipping.", flush=True)
        return certainty_vote
    
    original_vote = certainty_vote.copy()
    model_name = os.getenv("BACKBONE_LLM", "gpt-4o")
    # Only use JSON mode for models that support response_format (not o-series or gpt-5)
    _no_json_mode = ('o1' in model_name or 'o3' in model_name or 'o4' in model_name
                     or 'gpt-5' in model_name)
    use_json_mode = not _no_json_mode and (
        "gpt" in model_name.lower() or "openai" in model_name.lower()
    )
    
    for attempt_num in range(1, max_attempts + 1):
        output = None
        try:
            # Prepare prompt
            safe_query = sanitize_prompt_content(str(query))
            safe_vote_content = sanitize_prompt_content(json.dumps(certainty_vote, indent=2))
            
            # Use the same prompt but enable JSON mode for better structured output
            prompt = RECONCILE_PROMPT + f"\n\nUser query: {safe_query}\n\nDictionary:\n{safe_vote_content}"
            messages = [{"role": "user", "content": prompt}]
            
            # Call LLM with JSON mode if supported (for guaranteed valid JSON output)
            if use_json_mode:
                output = chat_completion(
                    messages, 
                    model=model_name, 
                    mod='dialog',
                    response_format={"type": "json_object"},
                    reasoning_effort="low"
                )
            else:
                output = chat_completion(messages, model=model_name, mod='dialog', reasoning_effort="low")
            
            # Parse the output
            parsed_vote = parse_llm_dict_output(output)
            
            # Handle nested {"normalized_votes": {...}} format from JSON mode
            if parsed_vote and isinstance(parsed_vote, dict):
                if "normalized_votes" in parsed_vote and isinstance(parsed_vote["normalized_votes"], dict):
                    parsed_vote = parsed_vote["normalized_votes"]
            
            # Validate parsed result
            if parsed_vote and isinstance(parsed_vote, dict) and len(parsed_vote) > 0:
                # Ensure all values are numeric
                if all(isinstance(v, (int, float)) for v in parsed_vote.values()):
                    print(f"[Vote Reconciliation] ✓ Successfully reconciled on attempt {attempt_num}", flush=True)
                    return parsed_vote
                else:
                    non_numeric = {k: type(v).__name__ for k, v in parsed_vote.items() if not isinstance(v, (int, float))}
                    raise ValueError(f"Non-numeric values in dictionary: {non_numeric}")
            else:
                raise ValueError("Failed to parse valid dictionary")
                
        except Exception as e:
            if attempt_num < max_attempts:
                print(f"[Vote Reconciliation] Attempt {attempt_num} failed: {str(e)[:120]}", flush=True)
                # Always show LLM output on failure for easier debugging
                if output:
                    print(f"[Vote Reconciliation] LLM output preview: {str(output)[:300]}", flush=True)
            else:
                print(f"[Vote Reconciliation] All attempts failed. Using original votes.", flush=True)
                if output:
                    print(f"[Vote Reconciliation] Last LLM output: {str(output)[:300]}", flush=True)
                if os.getenv("MEDEA_DEBUG", "").lower() == "true":
                    print(f"[Debug] Final output: {output[:400]}", flush=True)
    
    return original_vote

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
    if x > 10: x = x / 100.0  # Handle LLM returning e.g. 85 instead of 0.85
    x = min(x, 1.0)  # Clamp values like 1.5 to 1.0
    x = max(0.0, min(1.0, x))  # Clamp to [0, 1]
    if x <= 0.6: return 0.1
    if 0.8 > x > 0.6: return 0.3
    if 0.9 > x >= 0.8: return 0.5
    if 1 > x >= 0.9: return 0.8
    return 1


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
            if isinstance(tmp[o+r], dict) and 'answer' in tmp[o+r]:
                tmp[o+"_pred_"+str(rounds)] = tmp[o+r]['answer']
                evidence_basis = tmp[o+r].get('evidence_basis', 'unknown')
                tmp[o+"_eb_"+str(rounds)] = evidence_basis
                tmp[o+"_exp_"+str(rounds)] = (
                    f"Answer: {tmp[o+r]['answer']} | "
                    f"Evidence basis: {evidence_basis} | "
                    f"Confidence: {tmp[o+r]['confidence_level']} | "
                    f"Reasoning: {tmp[o+r]['reasoning']}"
                )
                if tmp[o+r]['answer'] not in certainty_vote:
                    certainty_vote[tmp[o+r]['answer']] = trans_confidence(tmp[o+r]['confidence_level']) + 1e-5
                else:
                    certainty_vote[tmp[o+r]['answer']] += trans_confidence(tmp[o+r]['confidence_level'])
            else:
                print(f"Warning: {o+r} is not structured as expected: {tmp[o+r]}")
                tmp[o+"_pred_"+str(rounds)] = 'unknown'
                tmp[o+"_eb_"+str(rounds)] = 'unknown'
                tmp[o+"_exp_"+str(rounds)] = (
                    f"Answer: unknown | Evidence basis: unknown | Confidence: 0.0 | "
                    f"Reasoning: malformed LLM output after all retries"
                )

    if c+r in tmp and g+r in tmp and b+r in tmp:
        tmp['vote_'+str(rounds)] = [tmp['llm_0_pred_'+str(rounds)], tmp['llm_1_pred_'+str(rounds)], tmp['llm_2_pred_'+str(rounds)]]
        tmp['exps_'+str(rounds)] = [tmp['llm_0_exp_'+str(rounds)], tmp['llm_1_exp_'+str(rounds)], tmp['llm_2_exp_'+str(rounds)]]
        
        # ========== VOTE RECONCILIATION ==========
        if vote_merge:
            certainty_vote = reconcile_votes_with_llm(certainty_vote, query, max_attempts=attempt)
        # ==========
        for v in certainty_vote:
            print(v, flush=True)
        tmp['weighted_vote_'+str(rounds)] = certainty_vote
        tmp['weighted_max_'+str(rounds)] = max(certainty_vote, key=certainty_vote.get) if certainty_vote else "unknown"
        print(f"\nMax weighted Vote: {tmp['weighted_max_'+str(rounds)]}", flush=True)

        tmp['debate_prompt_'+str(rounds)] = ''
        vote = Counter(tmp['vote_'+str(rounds)]).most_common(2)

        tmp['majority_ans_'+str(rounds)] = vote[0][0]
        if len(vote) > 1: # not all the agents give the same answer
            # Present each viewpoint neutrally without revealing vote counts
            # to avoid social conformity pressure
            view_num = 0
            for v in vote:
                view_num += 1
                exp_index = find_idx_by_element(tmp['vote_'+str(rounds)], v[0])
                group_exp = find_element_by_indices(tmp['exps_'+str(rounds)], exp_index)
                exp = "\n".join(["Supporting argument: " + g for g in group_exp])
                tmp['debate_prompt_'+str(rounds)] += (
                    f"Viewpoint {view_num}: \"{v[0]}\"\n{exp}\n\n"
                )
            tmp['debate_prompt_'+str(rounds)] += (
                "Critically evaluate ALL viewpoints above. Consider: "
                "What evidence supports each view? What evidence contradicts it? "
                "Pay attention to each argument's evidence basis — arguments grounded in "
                "empirical or literature evidence should carry more weight than those based on "
                "backbone or own_knowledge. "
                "If you change your answer, explain what specific argument convinced you."
            )
                    
    return tmp


def clean_output(tmp, rounds):
    co, go, bo = f"llm_0_output_{rounds}", f"llm_1_output_{rounds}", f"llm_2_output_{rounds}"

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
    contexts = [{"role": "system", "content": PANEL_SYSTEM_PROMPT}]

    if convincing_samples:
        for cs in convincing_samples:
            contexts.append({"role": "user", "content": f"User Query: {cs['train_sample']['question']}"})
            contexts.append({"role": "assistant", "content": str({"reasoning": cs['train_sample']['gold_explanation'], "answer": cs['train_sample']['answer']})})

    if intervene:
        contexts.append({"role": "user", "content": f"User Query: {query['question']}" + "\nAnswer the question given the fact that " + query['gold_explanation']})  
    else:
        contexts.append({"role": "user", "content": f"User Query: {query}"})
        
    contexts[-1]["content"] += (
        " Analyze the evidence from all sources. Follow the evidence hierarchy and confidence calibration "
        "guidelines from your system instructions. Label the source of each key claim in your reasoning."
    )
    
    safe_json_format = (
        " Output in JSON format: {'reasoning': 'step_by_step_reasoning_with_source_labels', "
        "'answer': 'your_answer', 'confidence_level': (0.0-1.0), "
        "'evidence_basis': 'empirical|literature|backbone|own_knowledge|insufficient'}. "
        "Keep response under 1000 tokens."
    )
    contexts[-1]["content"] += safe_json_format
    
    contexts[-1]["content"] = sanitize_prompt_content(contexts[-1]["content"])
    
    return contexts


def _extract_answer_from_plaintext(text):
    """
    Fallback: extract a structured answer from plain-text LLM output
    when the model fails to produce valid JSON. This is a general recovery
    mechanism for models that don't reliably follow JSON format instructions.
    """
    text = text.strip()
    # Try to find any JSON-like structure in the text (even partial)
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if json_match:
        result = parse_json(json_match.group())
        if result != "ERR_SYNTAX":
            return result
    
    # Last resort: use the raw text as the answer with low confidence
    return {
        'reasoning': text[:500] if len(text) > 500 else text,
        'answer': text[:300] if len(text) > 300 else text,
        'confidence_level': 0.3  # Low confidence since format was wrong
    }


_REQUIRED_PANELIST_KEYS = {'answer', 'confidence_level', 'reasoning'}

def gpt_gen_ans(query, model='gpt-4o', attempts=3, convincing_samples=None, additional_instruc=None, intervene=False):
    i = 0
    last_output = None
    while i < attempts:
        try:
            contexts = prepare_context_for_chat_assistant(query, convincing_samples, intervene)
            if additional_instruc:
                # Sanitize additional instructions before adding them
                safe_additional_instruc = [sanitize_prompt_content(str(instr)) for instr in additional_instruc]
                contexts[-1]['content'] += " " + " ".join(safe_additional_instruc)
                # Apply final sanitization to the complete content
                contexts[-1]['content'] = sanitize_prompt_content(contexts[-1]['content'])
            output = chat_completion(contexts, model=model, mod='dialog', reasoning_effort="medium")
            if output:
                last_output = output
                if "{" not in output or "}" not in output:
                    raise ValueError("cannot find { or } in the model output.")
                result = parse_json(output)
                if result == "ERR_SYNTAX":
                    raise ValueError("[gpt_gen_ans] Incomplete JSON format. Retrying (attempts: " + str(i) + ")...")
                if not isinstance(result, dict) or not _REQUIRED_PANELIST_KEYS.issubset(result.keys()):
                    raise ValueError(
                        f"[gpt_gen_ans] Response missing required keys {_REQUIRED_PANELIST_KEYS - set(result.keys() if isinstance(result, dict) else [])}. "
                        f"Got keys: {list(result.keys()) if isinstance(result, dict) else type(result)}. Retrying (attempt: {i})..."
                    )
            return result
        except Exception as e:
            print(f"[Retrying - {model}]: {e}")
            time.sleep(5)
            if "Incapsula_Resource" in str(e):
                print("Incapsula Resource Error, retrying...")
                print(f"[gpt_gen_ans] Prompt blocked by Incapsula:\n {contexts}")
                raise Exception(f"Incapsula Resource Error. Agent terminated: {e}")
            i += 1
    
    # All attempts failed: try to recover answer from last raw output
    if last_output:
        print(f"[gpt_gen_ans] All {attempts} JSON attempts failed for {model}. Extracting from plain text.", flush=True)
        return _extract_answer_from_plaintext(last_output)
    
    return {'reasoning': "None", "answer": "I can not help with this.", "confidence_level": 0.0}
    


def llm_debate(query, tmp, rounds, model_name='gpt-4o', llm_name='llm_0', convincing_samples=None):
    r = '_' + str(rounds-1)

    if f'{llm_name}_output_'+str(rounds) not in tmp and 'debate_prompt'+ r in tmp and len(tmp['debate_prompt'+r]):
        print("Debate")
        additional_instruc = [
            "\n\nBelow are different viewpoints from other panelists. "
            "Critically evaluate each viewpoint based on the EVIDENCE presented, not on how many panelists hold that view. "
            "Pay attention to each viewpoint's evidence_basis — viewpoints backed by empirical/literature evidence "
            "deserve more weight than those based on backbone or own_knowledge. "
            "Provide your own independent answer and step-by-step reasoning."
        ]
        additional_instruc.append(
            "Clearly state which viewpoint you agree or disagree with and WHY, citing specific evidence. "
            "If no strong empirical evidence supports any viewpoint, say so and assign low confidence.\n\n"
        )
        
        sanitized_debate_prompt = sanitize_prompt_content(tmp['debate_prompt'+r])
        additional_instruc.append(sanitized_debate_prompt)
        
        safe_json_instruction = (
            "Output your answer in JSON format: {'reasoning': 'your_reasoning_with_source_labels', "
            "'answer': 'your_answer', 'confidence_level': numeric_value, "
            "'evidence_basis': 'empirical|literature|backbone|own_knowledge|insufficient'}."
        )
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
    panelist_llms=None,
    include_llm=True, 
    proposal_response=None, 
    coding_response=None, 
    reasoning_response=None, 
    vote_merge=True,
    round=1
    ):
    import builtins
    _round = builtins.round  # Preserve built-in round() since parameter 'round' shadows it
    if panelist_llms is None:
        panelist_llms = ['gemini', 'gpt-4o', 'gpt-4o-mini']

    tmp = {}
    debate_query = query
    code_snippet = executed_output = reasoning_output = "None"
    llm_hypothesis = ""
    experiment_hypothesis = ""
    literature_hypothesis = ""
    coding_failed = False
    reasoning_failed = False

    if mod == "diff_context":
        if isinstance(coding_response, dict):
            code_snippet = coding_response.get("code_snippet", "None")
            executed_output = coding_response.get("executed_output") or "None"

        reasoning_citation = "None"
        if isinstance(reasoning_response, dict):
            reasoning_dict = reasoning_response.get("user_query", {})
            if isinstance(reasoning_dict, dict):
                reasoning_output = reasoning_dict.get("answer") or "None"
                reasoning_citation = reasoning_dict.get("citation") or "None"

        # Sanitize all input content before processing
        safe_proposal = sanitize_prompt_content(str(proposal_response))
        safe_code_snippet = sanitize_prompt_content(str(code_snippet))
        safe_executed_output = sanitize_prompt_content(str(executed_output))
        safe_reasoning_output = sanitize_prompt_content(str(reasoning_output))
        safe_reasoning_citation = sanitize_prompt_content(str(reasoning_citation))
        safe_query = sanitize_prompt_content(str(query))

        # --- Evidence grading: detect agent success/failure before normalization ---
        def _is_agent_failed(output_str):
            """
            Fast deterministic check for obvious agent failures.
            A negative finding ('no papers found') is NOT a failure — it's valid evidence.
            Only true failures (None, empty, explicit failure messages) count as failed.
            """
            low = output_str.lower().strip()
            # Exact match for empty/null outputs
            if low in ("none", "null", "", "failed"):
                return True
            # Explicit failure patterns — only unambiguous agent-level messages
            _hard_fail_patterns = [
                "i cannot help",
                "agent could not complete",
                "call budget exceeded",
            ]
            return any(pat in low for pat in _hard_fail_patterns)

        def _is_agent_failed_llm(output_str, agent_name):
            """
            LLM-based check for ambiguous cases where the fast path returns False
            but the output might still indicate a true failure (e.g., wall of tracebacks).
            Uses a fast, cheap model. Defaults to False (not failed) on any error.
            """
            try:
                # Truncate to first 1000 + last 1000 chars for efficiency
                if len(output_str) > 2000:
                    excerpt = output_str[:1000] + "\n...[truncated]...\n" + output_str[-1000:]
                else:
                    excerpt = output_str

                judge_model = os.getenv("AGENT_FAILURE_JUDGE_LLM", "gpt-4.1-nano")
                judge_prompt = AGENT_FAILURE_JUDGE_PROMPT.format(
                    agent_name=agent_name,
                    output_excerpt=excerpt
                )
                result = chat_completion(judge_prompt, model=judge_model, temperature=0, reasoning_effort="low")
                is_failed = "FAILED" in result.upper().strip()
                if is_failed:
                    print(f"[Evidence Grading] LLM judge determined {agent_name}: FAILED", flush=True)
                return is_failed
            except Exception as e:
                print(f"[Evidence Grading] LLM failure judge error for {agent_name}: {e}. Defaulting to not-failed.", flush=True)
                return False  # Safe default: preserve evidence

        coding_failed = _is_agent_failed(safe_executed_output)
        if not coding_failed:
            coding_failed = _is_agent_failed_llm(safe_executed_output, "Analysis Agent")

        reasoning_failed = _is_agent_failed(safe_reasoning_output)
        if not reasoning_failed:
            reasoning_failed = _is_agent_failed_llm(safe_reasoning_output, "Literature Reasoning Agent")

        # Build raw hypothesis text for each agent
        experiment_hypothesis = (
            f"[Analysis Agent] Proposal:\n{safe_proposal}\n"
            f"[Analysis Agent] Code Snippet:\n{safe_code_snippet}\n"
            f"[Analysis Agent] Output:\n{safe_executed_output}"
        )
        literature_hypothesis = f"[Literature Reasoning Agent] Output:\n{safe_reasoning_output}"
        if safe_reasoning_citation != "None":
            literature_hypothesis += f"\n[Literature Reasoning Agent] Citations:\n{safe_reasoning_citation}"

        # Normalize agent outputs; for failed agents, preserve partial evidence via summarization
        if coding_failed:
            print("[Evidence Grading] Analysis agent: NO CONCLUSION", flush=True)
            if len(safe_executed_output) > 200:
                summarizer_model = os.getenv("AGENT_FAILURE_JUDGE_LLM", "gpt-4.1-mini")
                evidence_summary = chat_completion(
                    FAILED_AGENT_EVIDENCE_SUMMARIZER.format(user_query=safe_query, raw_output=safe_executed_output[:6000]),
                    model=summarizer_model, reasoning_effort="low"
                )
                coding_hypothesis = (
                    f"[Analysis Agent] [STATUS: NO CONCLUSION] "
                    f"The agent did not reach a conclusion, but collected the following evidence:\n{evidence_summary}"
                )
            else:
                coding_hypothesis = (
                    "[Analysis Agent] [STATUS: NO CONCLUSION] "
                    "The analysis agent could not produce results. No empirical evidence from this agent."
                )
        else:
            coding_hypothesis = ANALYSIS_AGENT_NORMALIZER.format(user_query=safe_query, agent_hypo=experiment_hypothesis)
        
        if reasoning_failed:
            print("[Evidence Grading] Literature reasoning agent: NO CONCLUSION", flush=True)
            if len(safe_reasoning_output) > 200:
                summarizer_model = os.getenv("AGENT_FAILURE_JUDGE_LLM", "gpt-4.1-mini")
                evidence_summary = chat_completion(
                    FAILED_AGENT_EVIDENCE_SUMMARIZER.format(user_query=safe_query, raw_output=safe_reasoning_output[:6000]),
                    model=summarizer_model, reasoning_effort="low"
                )
                reasoning_hypothesis = (
                    f"[Literature Reasoning Agent] [STATUS: NO CONCLUSION] "
                    f"The agent did not reach a conclusion, but collected the following evidence:\n{evidence_summary}"
                )
            else:
                reasoning_hypothesis = (
                    "[Literature Reasoning Agent] [STATUS: NO CONCLUSION] "
                    "The literature reasoning agent found no relevant papers or could not complete analysis. "
                    "No literature evidence from this agent."
                )
        else:
            reasoning_hypothesis = HYPOTHESOS_NORMALIZER.format(user_query=safe_query, agent_hypo=literature_hypothesis)

        # Evidence from computational agents
        agents_ans_for_panel = (
            "[Analysis Agent] Output:\n " + coding_hypothesis + '\n\n'
            + '[Literature Reasoning Agent] Output:\n ' + reasoning_hypothesis
        )
        
        # Full evidence including backbone
        agents_ans = agents_ans_for_panel
        
        if include_llm:
            print(f"----- Hypothesis from Backbone LLM: {os.getenv('BACKBONE_LLM')} -----", flush=True)
            backbone_query = BACKBONE_QUERY_PROMPT.format(query=safe_query)
            llm_hypothesis = chat_completion(backbone_query, model=os.getenv("BACKBONE_LLM"), reasoning_effort="medium")
            safe_llm_hypothesis = sanitize_prompt_content(str(llm_hypothesis))
            backbone_hypothesis = BACKBONE_NORMALIZER.format(user_query=safe_query, agent_hypo=safe_llm_hypothesis)
            print(f"{llm_hypothesis}\n", flush=True)
            # Backbone hypothesis is shared with both panelists and formulator as an evidence source
            agents_ans += '\n\n' + '[LLM Backbone] Output:\n' + backbone_hypothesis
            agents_ans_for_panel += '\n\n' + '[LLM Backbone] Output:\n' + backbone_hypothesis
        
        # --- Uncertainty metadata for panelists ---
        evidence_note = ""
        if coding_failed and reasoning_failed:
            evidence_note = (
                "\n\n[EVIDENCE NOTE] WARNING: Both the analysis agent and literature reasoning agent "
                "FAILED to produce results. The hypotheses below are based on general LLM knowledge only, "
                "NOT on empirical evidence. Assign appropriately LOW confidence to your answer."
            )
        elif coding_failed or reasoning_failed:
            failed_agent = "analysis agent" if coding_failed else "literature reasoning agent"
            evidence_note = (
                f"\n\n[EVIDENCE NOTE] The {failed_agent} FAILED to produce results. "
                f"Only partial agent evidence is available. Consider this limitation in your confidence level."
            )
        
        # --- Evidence Auditor: cross-source analysis before panel ---
        audit_report = ""
        if include_llm:
            print("----- Evidence Auditor: cross-source analysis -----", flush=True)
            auditor_model = os.getenv("AUDITOR_LLM", "gpt-4.1-mini")
            audit_prompt = EVIDENCE_AUDITOR_PROMPT.format(
                query=safe_query,
                coding_output=coding_hypothesis,
                reasoning_output=reasoning_hypothesis,
                backbone_output=backbone_hypothesis,
            )
            audit_report = chat_completion(audit_prompt, model=auditor_model, temperature=0, reasoning_effort="medium")
            audit_report = sanitize_prompt_content(str(audit_report))
            print(f"{audit_report}\n", flush=True)

        # Build the debate query for panelists
        debate_query_parts = [
            sanitize_prompt_content(str(query)),
            "\nEvidence from Agents and LLM Backbone:",
            evidence_note,
            sanitize_prompt_content(agents_ans_for_panel),
        ]
        if audit_report:
            debate_query_parts.append("\n[Evidence Audit Report]\n" + audit_report)
        debate_query = "\n".join(debate_query_parts)

    # Phase1: Initial round for panel discussion — 3 panelists in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(gpt_gen_ans, debate_query, model=p, additional_instruc=None, intervene=False): f'llm_{i}_output_0'
            for i, p in enumerate(panelist_llms)
        }
        for fut in as_completed(futures):
            tmp[futures[fut]] = fut.result()

    tmp = clean_output(tmp, 0)
    tmp = parse_output(tmp, query, 0, vote_merge=vote_merge)


    # Phase2: Multi-Round Discussion — 3 panelists in parallel per round
    for r in range(1, round+1):
        print(f"----- Round {r} Discussion -----", flush=True)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(llm_debate, debate_query, tmp, llm_name=f'llm_{i}', rounds=r, model_name=p)
                for i, p in enumerate(panelist_llms)
            ]
            for fut in as_completed(futures):
                fut.result()  # Each call updates tmp in place with unique keys
        
        tmp = clean_output(tmp, r)
        tmp = parse_output(tmp, query, r, vote_merge=vote_merge)
    
    # Find keys that start with 'weighted_max_' and extract the highest suffix
    majority_keys = [key for key in tmp if key.startswith("weighted_max_")]
    if majority_keys:
        majority_keys_sorted = sorted(majority_keys, key=lambda x: int(x.split('_')[-1]))
        last_majority_key = majority_keys_sorted[-1]
        panel_conclusion = tmp[last_majority_key]
    else:
        panel_conclusion = "No panel conclusion available."
    
    # --- Uncertainty propagation: check average panelist confidence ---
    # If both agents failed AND average panelist confidence is low, flag as insufficient evidence
    last_round = round  # the last debate round number
    avg_confidence = 0.0
    confidence_count = 0
    for panelist_id in ["llm_0", "llm_1", "llm_2"]:
        output_key = f"{panelist_id}_output_{last_round}"
        if output_key in tmp and isinstance(tmp[output_key], dict):
            conf = tmp[output_key].get('confidence_level', 0.0)
            try:
                avg_confidence += float(conf)
                confidence_count += 1
            except (ValueError, TypeError):
                pass
    if confidence_count > 0:
        avg_confidence /= confidence_count
    
    # --- Evidence grounding: deterministic provenance tracking ---
    both_agents_failed = (mod == "diff_context" and coding_failed and reasoning_failed)

    evidence_grounding = {
        "analysis_agent": "FAILED" if coding_failed else "SUCCEEDED",
        "literature_agent": "FAILED" if reasoning_failed else "SUCCEEDED",
        "llm_backbone": "SUCCEEDED" if llm_hypothesis else "NOT_INCLUDED",
        "grounding_level": "full",
        "warning": None,
        "avg_panelist_confidence": _round(avg_confidence, 2),
    }
    if both_agents_failed:
        evidence_grounding["grounding_level"] = "llm_only"
        evidence_grounding["warning"] = (
            "Both the analysis agent and literature reasoning agent failed to produce results. "
            "Conclusion is based entirely on LLM prior knowledge with no empirical grounding."
        )
    elif coding_failed:
        evidence_grounding["grounding_level"] = "partial"
        evidence_grounding["warning"] = (
            "Analysis agent failed. Conclusion lacks computational/database evidence "
            "and relies on literature reasoning and LLM knowledge only."
        )
    elif reasoning_failed:
        evidence_grounding["grounding_level"] = "partial"
        evidence_grounding["warning"] = (
            "Literature reasoning agent failed. Conclusion lacks literature-grounded evidence "
            "and relies on computational analysis and LLM knowledge only."
        )

    # Uncertainty propagation: tag panel_conclusion when agent(s) failed
    if both_agents_failed:
        print(
            f"[Uncertainty Propagation] Both agents failed (avg panelist confidence={avg_confidence:.2f}). "
            f"Tagging panel conclusion.",
            flush=True
        )
        panel_conclusion = (
            "[CAUTION: No empirical evidence — both agents failed] " + panel_conclusion
        )
    elif coding_failed or reasoning_failed:
        failed_name = "analysis" if coding_failed else "literature reasoning"
        print(
            f"[Uncertainty Propagation] {failed_name.capitalize()} agent failed "
            f"(avg panelist confidence={avg_confidence:.2f}). Tagging panel conclusion.",
            flush=True
        )
        panel_conclusion = (
            f"[NOTE: {failed_name} agent failed — partial evidence only] " + panel_conclusion
        )

    # Build agent evidence summary for the formulator
    original_ans = experiment_hypothesis + "\n\n" + literature_hypothesis
    if llm_hypothesis:
        original_ans += "\n\n[LLM Backbone] Output:\n" + llm_hypothesis
    
    # Format the faithful-reporter prompt (panel conclusion is anchored as the immutable constraint)
    hypo_prompt = HYPOTHESIS_FORMULATOR.format(
        query=query, answer=panel_conclusion, agent_ans=original_ans
    )
    
    # Use a separate formulator model if configured, otherwise use backbone
    formulator_model = os.getenv("FORMULATOR_LLM", os.getenv("BACKBONE_LLM"))
    formulated = chat_completion(hypo_prompt, model=formulator_model, reasoning_effort="medium")

    # Deterministic grounding block — appended by code, not LLM-generated
    if evidence_grounding["warning"]:
        formulated += (
            f"\n\n---\n"
            f"Evidence Grounding: {evidence_grounding['grounding_level'].upper()}\n"
            f"- Analysis Agent: {evidence_grounding['analysis_agent']}\n"
            f"- Literature Agent: {evidence_grounding['literature_agent']}\n"
            f"- LLM Backbone: {evidence_grounding['llm_backbone']}\n"
            f"- Warning: {evidence_grounding['warning']}\n"
            f"---"
        )

    if include_llm:
        return formulated, llm_hypothesis, evidence_grounding
    return formulated, None, evidence_grounding
