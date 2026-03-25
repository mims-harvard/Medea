import os
import json
import time
import hashlib
import tempfile
from multiprocessing import freeze_support
from agentlite.commons import TaskPackage

# Use relative imports within the package
from ..modules.literature_reasoning import *
from ..modules.agent_llms import LLMConfig, AgentLLM
from ..modules.utils import FlushAgentLogger as AgentLogger, ReasoningPackage

# ============================================================================
# FILE-BASED LITERATURE CACHE + CALL BUDGET
# Works across subprocess boundaries (subprocess.Popen, multiprocessing.Process)
# Each top-level process (main.py invocation) gets its own isolated directory
# so parallel tmux experiments don't interfere with each other's budgets.
# ============================================================================
_SESSION_ID = str(os.getpid())
_CACHE_DIR = os.path.join(tempfile.gettempdir(), f"medea_lit_cache_{_SESSION_ID}")
_BUDGET_FILE = os.path.join(_CACHE_DIR, "_call_count.json")


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _normalize_query(query: str) -> str:
    """Normalize query for cache lookup: lowercase, deduplicate, sort key terms."""
    words = sorted(set(query.lower().split()))
    return " ".join(words)


def _cache_key_to_path(cache_key: str) -> str:
    """Convert a cache key to a file path."""
    h = hashlib.md5(cache_key.encode()).hexdigest()
    return os.path.join(_CACHE_DIR, f"lit_{h}.json")


def _read_call_count() -> int:
    try:
        with open(_BUDGET_FILE, 'r') as f:
            return json.load(f).get("count", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def _write_call_count(count: int):
    _ensure_cache_dir()
    with open(_BUDGET_FILE, 'w') as f:
        json.dump({"count": count}, f)


def reset_call_budget():
    """Reset per-sample call counter and literature cache. Call between samples."""
    _ensure_cache_dir()
    _write_call_count(0)
    for f in os.listdir(_CACHE_DIR):
        fp = os.path.join(_CACHE_DIR, f)
        try:
            os.remove(fp)
        except OSError:
            pass


def reasoning_module(query: str, reason_agent):
    """
    Execute reasoning agent and return the reasoning content as a string.
    
    Args:
        query: The research question to analyze
        reason_agent: The reasoning agent instance
        
    Returns:
        str: The reasoning content with citations, or error message
    """
    task_dict = {"user_query": query, "hypothesis": None}
    reason_taskpack = TaskPackage(instruction=str(task_dict))
    
    reasoning_response = reason_agent(reason_taskpack)
            
    # Handle dict response (new format after ReasonFinishAction fix)
    if isinstance(reasoning_response, dict) and "user_query" in reasoning_response:
        user_query_data = reasoning_response["user_query"]
        if user_query_data and isinstance(user_query_data, dict):
            reasoning_ans = user_query_data.get("answer", "")
            return reasoning_ans
    return reasoning_response


def scientific_reasoning_agent(
        user_instruction, 
        llm_name=os.getenv("BACKBONE_LLM"), 
        reason_agent_tmp=0.4,
        reason_action_tmp=0.4,
        verbose=False,
    ):
    """
    Scientific reasoning agent with file-based caching and call budget.
    
    Uses file-system cache (works across subprocess.Popen and multiprocessing).
    Enforces a per-session call budget (MAX_REASONING_AGENT_CALLS env var, default 2).
    Call reset_call_budget() between samples to reset.
    """
    _ensure_cache_dir()
    max_calls = int(os.environ.get("MAX_REASONING_AGENT_CALLS", "6"))
    
    # --- File-based cache check ---
    cache_key = _normalize_query(user_instruction)
    cache_path = _cache_key_to_path(cache_key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            print(f"[scientific_reasoning_agent] Cache HIT — reusing previous result", flush=True)
            return cached["result"]
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupted cache, re-run
    
    # --- File-based call budget check ---
    call_count = _read_call_count() + 1
    _write_call_count(call_count)
    
    if call_count > max_calls:
        msg = (
            f"[scientific_reasoning_agent] Call budget exceeded ({max_calls} calls max per session). "
            f"Reuse results from previous calls or reduce the number of literature queries in your code."
        )
        print(msg, flush=True)
        return msg
    
    print(f"[scientific_reasoning_agent] Call {call_count}/{max_calls} — running full pipeline", flush=True)
    
    freeze_support()
    reason_llm_config_dict = {"temperature": reason_agent_tmp}
    reason_llm_config = LLMConfig(reason_llm_config_dict)
    reason_llm = AgentLLM(llm_config=reason_llm_config, llm_name=llm_name)
    logger = AgentLogger(FLAG_PRINT=False, PROMPT_DEBUG_FLAG=False)

    reason_actions = [
        LiteratureSearch(model_name=llm_name, verbose=verbose),
        PaperJudge(model_name=llm_name, verbose=verbose, max_workers=10),
        OpenScholarReasoning(tmp=reason_action_tmp, llm_provider=llm_name, model_name=llm_name, verbose=verbose)
    ]
    reason_agent = LiteratureReasoning(llm=reason_llm, actions=reason_actions, logger=logger)
    result = reasoning_module(user_instruction, reason_agent)
    
    # --- Cache result to file ---
    result_str = str(result) if not isinstance(result, str) else result
    try:
        with open(cache_path, 'w') as f:
            json.dump({"result": result_str, "timestamp": time.time()}, f)
    except Exception:
        pass  # Non-critical if cache write fails
    
    return result



if __name__ == "__main__":
    print(scientific_reasoning_agent("what is ICI treatment?"))