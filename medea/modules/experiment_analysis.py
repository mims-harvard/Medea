import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

import dotenv
from agentlite.actions import ThinkAct
from agentlite.actions.BaseAction import BaseAction
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.agents.agent_utils import *
try:
    from agentlite.agents.agent_utils import ACTION_NOT_FOUND_MESS
except ImportError:
    ACTION_NOT_FOUND_MESS = "[Error] Action not found in action list."
from agentlite.commons import AgentAct, TaskPackage
from agentlite.commons.AgentAct import ActObsChainType

dotenv.load_dotenv()

# Use relative imports within package
from ..tool_space.gpt_utils import chat_completion
from ..tool_space.env_utils import get_backbone_llm, get_utility_llm

from .agent_llms import LLMConfig, AgentLLM, parse_action
from .BasePrompt import BasePromptGen
from .prompt_template import *
from .utils import CodeSnippet, FlushAgentLogger as AgentLogger, Proposal

# Load available tools using package-relative path
def _load_tool_info():
    """Load tool_info.json from package directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from agents/ to get to the package root
    package_root = os.path.dirname(current_dir)
    tool_info_path = os.path.join(package_root, "tool_space", "tool_info.json")
    
    # Fallback: if not found, try relative to current working directory (for development)
    if not os.path.exists(tool_info_path):
        tool_info_path = "tool_space/tool_info.json"
    
    with open(tool_info_path, 'r') as tool_file:
        return json.load(tool_file)

AVALIBLE_TOOL = _load_tool_info()

def stream_reader(pipe, output_list, source):
    """
    Read lines from a pipe and append them to a list.
    
    Args:
        pipe: File object to read from
        output_list: List to append lines to
        source: Source identifier for logging (e.g., 'stdout', 'stderr')
    """
    try:
        with pipe:
            for line in iter(pipe.readline, ''):
                print(f'[{source}] {line}', end='', flush=True)
                output_list.append(line)
    except Exception as e:
        print(f"Error reading pipe: {e}", flush=True)


class ToolSelector:
    """Selects relevant tools for code generation based on task instructions."""
    
    def __init__(self, llm_provider: str, tmp: float = 0.4) -> None:
        """
        Initialize the ToolSelector.

        Args:
            llm_provider: LLM provider/model name
            tmp: Temperature setting for the LLM
        """
        self.tool_list = AVALIBLE_TOOL
        agent_config = LLMConfig({'temperature': tmp})
        self.pattern = r'```json\n(.*?)```'
        self.selector = AgentLLM(
            llm_config=agent_config,
            llm_name=llm_provider,
            system_prompt=TOOL_SELECTION_TEMPLATE, 
            input_variables=["instruction", "tool_info"],
            reasoning_effort="low"
        )
        
    def __call__(self, instruction: str, max_attempts: int=3) -> List[Dict]:
        """
        Executes the tool selection process based on the given instruction.

        Args:
            instruction (str): The instruction to process.
            max_attempts (int): Number of attempts to retry LLM execution in case of failure. Default is 3.

        Returns:
            List[Dict]: A list of JSON objects representing the relevant tools.
        """
        input_prompt = {"instruction": instruction, "tool_info": self.tool_list}
        for attempt in range(max_attempts):
            try:
                # Run the LLM with the input prompt
                response = self.selector.run(input_prompt)
                if "```json" in response:
                    matches = re.findall(self.pattern, response, re.DOTALL)
                    response = matches[0]
                if "```python" in response:
                    pattern = r'^```python\s*(.*?)\s*```$'
                    matches = re.search(pattern, response, re.DOTALL)
                    response = matches.group(1).strip() if matches else response.strip()
                print(f"LLM Response: {response}", flush=True)

                # Safely evaluate the LLM's response
                relevant_tools = ast.literal_eval(response)
                if isinstance(relevant_tools, list):
                    break
                else:
                    raise ValueError("LLM response is not a valid list.")
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}", flush=True)
                if attempt == max_attempts - 1:
                    raise RuntimeError("Failed to retrieve tools after multiple attempts.") from e

        # Filter the available tools based on the response
        tool_info_json = [
            tool for tool in self.tool_list if tool.get("name") in relevant_tools
        ]
        return tool_info_json


class CodeGenerator(BaseAction):
    """Generates Python code based on task instructions and available tools."""
    
    def __init__(self, llm_provider: str = None, tmp: float = 0.4) -> None:
        # Get LLM provider with helpful error message if not provided
        if llm_provider is None:
            llm_provider = get_backbone_llm("gpt-4o")
        
        action_name = "CodeGenerator"
        action_desc = "Using this action generates the code snippet based on the given instruction (a Proposal object)."
        params_doc = {
            "instruction": "Supplied Proposal Object (e.g., <Proposal:xxxx>) - the Proposal object that contains step-by-step procedure about how to address the task",
            "code_draft": "CodeSnippet Object (e.g., <CodeSnippet:xxxx>) - the CodeSnippet draft on last iteration with feedback from the AnalysisQualityChecker action. This parameter can only be 'None' (e.g., code_draft=None) during the first iteration."
        }

        # Default temperature is 0.0
        agent_config = LLMConfig({'temperature': tmp})
        self.tool_selector = ToolSelector(llm_provider=get_utility_llm())

        self.CodeGenerator_agent = AgentLLM(
            llm_config=agent_config, 
            llm_name=llm_provider,
            system_prompt=CODE_GENERATION_TEMPLATE, 
            input_variables=["instruction", "user_query", "tools"],
            reasoning_effort="medium"
        )

        # Flexible pattern: matches ```python, ```Python, ```py, ``` python, etc.
        self.pattern = r'```\s*[Pp]y(?:thon)?\s*\n(.*?)```'
        self.max_instruction_tokens = 100000  # Conservative limit to avoid context issues
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )
        

    def _regex_extract_tools(self, instruction_text: str) -> List[Dict]:
        """
        Deterministically extract tool names from Proposal text using regex.
        The Proposal format uses 'Tool: exact_tool_name' fields.
        Returns filtered tool info dicts matching AVALIBLE_TOOL, or empty list if none found.
        """
        # Match "Tool: tool_name" patterns (case-insensitive, handles whitespace)
        matches = re.findall(r'(?:^|\n)\s*Tool:\s*(\w+)', instruction_text, re.IGNORECASE)
        if not matches:
            return []
        tool_names = list(set(matches))
        tool_info_json = [
            tool for tool in AVALIBLE_TOOL if tool.get("name") in tool_names
        ]
        if tool_info_json:
            print(f"[CodeGenerator] Regex extracted {len(tool_info_json)} tools: {[t['name'] for t in tool_info_json]}", flush=True)
        return tool_info_json

    @staticmethod
    def _is_tool_related_feedback(feedback: str) -> bool:
        """Check if quality feedback indicates a tool selection problem (vs. code logic problem)."""
        if not feedback:
            return False
        low = feedback.lower()
        tool_indicators = [
            "wrong tool", "missing tool", "tool not", "incorrect tool",
            "not in the documentation", "unavailable tool", "tool mismatch",
            "missing required call", "required tool", "should use",
        ]
        return any(ind in low for ind in tool_indicators)

    def __call__(
        self,
        instruction: Proposal,
        code_draft: CodeSnippet,
        quality_flag: bool = False,
        attempt: int = 4
    ) -> CodeSnippet:

        # Extract instruction text from Proposal object
        if not isinstance(instruction, Proposal):
            return f"Invalid instruction type: {type(instruction)}. Please provide a Proposal object."

        feedback, i = "", 0
        if code_draft is not None and type(code_draft) == CodeSnippet:
            code_last_round = "Code from last iteration:\n" + code_draft.get_code()
            feedback_last_round = "Feedback:\n" + code_draft.get_feedback()
            feedback = code_last_round + feedback_last_round

        user_query = instruction.user_query
        instruction_text = instruction.proposal

        print(f"User query:\n{user_query}", flush=True)
        print(f"Instruction text:\n{instruction_text}", flush=True)

        # Tool selection: regex first, LLM fallback, re-select only on tool-related feedback
        tool_json = self._regex_extract_tools(instruction_text)
        if not tool_json:
            print("[CodeGenerator] Regex found no tools, falling back to LLM ToolSelector", flush=True)
            tool_json = self.tool_selector(instruction=instruction_text)

        while not quality_flag and i < attempt:
            # Only re-select tools if previous feedback specifically indicates tool issues
            if i > 0 and self._is_tool_related_feedback(feedback):
                print(f"[CodeGenerator] Tool-related feedback detected, re-selecting tools (attempt {i+1})", flush=True)
                tool_json = self.tool_selector(instruction=instruction_text)

            input_prompt = {
                "instruction": instruction_text + "\n" + feedback,
                "user_query": user_query,
                "tools": tool_json
            }
            
            try:
                raw_code_snippet = self.CodeGenerator_agent.run(input_prompt)
                matches = re.findall(self.pattern, raw_code_snippet, re.DOTALL)
                if not matches:
                    # Fallback 1: try generic ``` fence without language tag
                    generic_matches = re.findall(r'```\n(.*?)```', raw_code_snippet, re.DOTALL)
                    if generic_matches:
                        # Pick the longest block (most likely the actual code)
                        matches = [max(generic_matches, key=len)]
                        print("[CodeGenerator] Found code in generic ``` fence (no language tag).", flush=True)
                    
                if not matches:
                    # Fallback 2: response contains Python code but no fences at all
                    stripped = raw_code_snippet.strip()
                    python_indicators = ['import ', 'from ', 'def ', 'class ', 'print(', 'if __name__']
                    has_python = sum(1 for kw in python_indicators if kw in stripped)
                    if has_python >= 2:
                        # Extract from first import/from/def line to end
                        lines = stripped.split('\n')
                        code_start = next(
                            (i for i, line in enumerate(lines)
                             if any(line.strip().startswith(kw) for kw in ['import ', 'from ', 'def ', '#!', '#'])),
                            None
                        )
                        if code_start is not None:
                            matches = ['\n'.join(lines[code_start:])]
                            print("[CodeGenerator] No code fence found, extracted Python code from response.", flush=True)
                    
                if not matches:
                    print("[CodeGenerator] No code block found in response, retrying...", flush=True)
                    i += 1
                    continue
                    
                code_snippet = matches[0]
                print("========Code========", flush=True)
                print(code_snippet, flush=True)
                
                quality_flag, feedback = self.check_code_quality(
                    instruction_text, tool_json, code_snippet
                )

                if quality_flag:
                    return CodeSnippet(
                        task=user_query,
                        instruction=instruction_text,  # Store instruction text
                        tool_info=tool_json,
                        code_snippet=code_snippet
                    )
                    
            except Exception as e:
                print(f"[CodeGenerator] Error on attempt {i+1}: {e}")
                feedback = f"Error: {str(e)}"
                
            i += 1
        
        return CodeSnippet(
            task=user_query,
            instruction=instruction_text,
            tool_info=tool_json,
            code_snippet=code_snippet if 'code_snippet' in locals() else "# Error: Could not generate code"
        )
    
    def check_code_quality(self, instruction: str, tool_json: List[Dict], code_snippet: str) -> tuple:
        """
        Check the quality of generated code.
        
        Args:
            instruction: Task instruction text
            tool_json: List of tools used
            code_snippet: Generated code to check
            
        Returns:
            Tuple of (is_quality_ok: bool, feedback: str)
        """
        pattern = r"\[(\w+)\]\s*-\s*(.*)"
        
        checker_prompt = CODE_PRECHECK_TEMPLATE.format(
            instruction=instruction,
            tool_info=tool_json,
            code_snippet=code_snippet
        )
        
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                output = chat_completion(checker_prompt, model=get_utility_llm(), reasoning_effort="low")
                print(output, flush=True)
                
                match = re.match(pattern, output)
                if match:
                    decision = match.group(1)
                    feedback = match.group(2)
                    break
                
                # Try alternative patterns
                output_lower = output.lower()
                if "approved" in output_lower or "pass" in output_lower:
                    decision = "Approved"
                    feedback = "Code quality check passed"
                    break
                if "[minor]" in output_lower or "minor" == output_lower.split("]")[0].strip("[").strip():
                    decision = "Minor"
                    feedback = output
                    break
                if "failed" in output_lower or "error" in output_lower:
                    decision = "Failed"
                    feedback = output
                    break
                        
            except Exception as e:
                print(f"[CodeGenerator] Quality check error (attempt {attempt+1}): {e}")
                if attempt >= max_attempts - 1:
                    decision = "Approved"
                    feedback = "Quality check failed, defaulting to approval"
                    break
        
        if decision and decision.lower() in ["failed", "fail"]:
            return False, f"Feedback:\n{feedback}"
        if decision and decision.lower() == "minor":
            print(f"[CodeGenerator] Quality check: MINOR issues (non-blocking). Proceeding.", flush=True)
        return True, feedback




class AnalysisExecution(BaseAction):
    """Executes Python code and captures output/errors."""
    
    def __init__(self) -> None:
        action_name = "AnalysisExecution"
        action_desc = "Using this action interprets the code and gets the executed output."
        params_doc = {
            "code_snippet": "EXACTLY '<CodeSnippet:xxxx>' format with NO additional text"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )
        self.failure_iter = 0

    def __call__(self, code_snippet: CodeSnippet, file_path: str = 'session1.py'):
        if not isinstance(code_snippet, CodeSnippet):
            return f"Invalid code snippet type: {type(code_snippet)}. Please provide a CodeSnippet object."
        
        # Prepend path setup to make tool_space imports work
        path_setup = """import sys
import os
import threading
# Add parent directory to path so tool_space can be imported as medea.tool_space
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pre-warm the OpenScholar reranker in the background while Steps 1-5 run.
# By the time scientific_reasoning_agent is called, the model is already loaded.
def _prewarm_reranker():
    try:
        from medea.modules.literature_reasoning import get_reranker
        get_reranker("OpenSciLM/OpenScholar_Reranker", use_fp16=True)
    except Exception:
        pass
threading.Thread(target=_prewarm_reranker, daemon=True).start()

"""
        
        # Replace tool_space imports with medea.tool_space
        code_to_run = code_snippet.code_snippet.replace(
            'from tool_space.', 'from medea.tool_space.'
        ).replace(
            'import tool_space.', 'import medea.tool_space.'
        )
        
        # Use a unique temp file to avoid race conditions across parallel experiments
        fd, tmp_path = tempfile.mkstemp(suffix='.py', prefix='medea_session_')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(path_setup + code_to_run)
        except Exception:
            os.close(fd)
            raise

        stdout_lines, stderr_lines = [], []
        p = subprocess.Popen(
            ["python", tmp_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            bufsize=1, 
            universal_newlines=True
        )

        stdout_thread = threading.Thread(target=stream_reader, args=(p.stdout, stdout_lines, 'stdout'))
        stderr_thread = threading.Thread(target=stream_reader, args=(p.stderr, stderr_lines, 'stderr'))
        stdout_thread.start()
        stderr_thread.start()

        try:
            # Wait for the process to complete with a timeout
            p.wait(timeout=600)  # wait up to 10 minutes
        except subprocess.TimeoutExpired:
            p.kill()
            print("Subprocess timed out and was killed.")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Ensure threads have finished reading
        stdout_thread.join()
        stderr_thread.join()

        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)
        
        # Keep only real errors; drop warnings, library logs, config dumps, and progress bars
        def _is_noise(line):
            s = line.strip()
            return (not s
                    or "Warning:" in line or "warn(" in line
                    or s[0] in '"{}'
                    or s.startswith(("loading ", "Model config", "You're using"))
                    or "it/s]" in line or "%|" in line)
        
        filtered_stderr = '\n'.join(l for l in stderr.splitlines() if not _is_noise(l))
        
        # Check for errors based on return code and filtered stderr content
        if p.returncode != 0:
            code_snippet.status = "error"
            code_snippet.stderr = filtered_stderr or stderr
            print(f"[Coding Error] {filtered_stderr or stderr}", flush=True)
            if self.failure_iter < 3:
                self.failure_iter += 1
                return f'{stderr}\n{code_snippet}: Error occurred during code execution, call CodeDebugger next.'
            self.failure_iter = 0
            return f'{code_snippet}: The system cannot help with it, call Finish next.'
        
        # If no errors, proceed (store filtered stderr for transparency but mark as non-error)
        code_snippet.stderr = filtered_stderr if filtered_stderr else None
        code_snippet.code_output = stdout
        code_snippet.status = "executed"
        return f'{code_snippet}: Successfully executed, call AnalysisQualityChecker action next.'
        


class CodeDebug(BaseAction):
    """Debugs and fixes code errors using LLM-based analysis."""
    
    def __init__(self, llm_provider: str = None, tmp: float = 0.4) -> None:
        # Get LLM provider with helpful error message if not provided
        if llm_provider is None:
            llm_provider = get_backbone_llm("gpt-4o")
        
        action_name = "CodeDebugger"
        action_desc = "When observed any issues from AnalysisExecution output, using this action to debug the code and get refined code snippet."
        params_doc = {
            "code_snippet": "EXACTLY '<CodeSnippet:xxxx>' format with NO additional text"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )
        
        agent_config = LLMConfig({'temperature': tmp})
        self.debugger_agent = AgentLLM(
            llm_config=agent_config, 
            llm_name=llm_provider, 
            system_prompt=DEBUGGER_TEMPLATE,
            reasoning_effort="medium"
        )
        self.pattern = r'```\s*[Pp]y(?:thon)?\s*\n(.*?)```'
        
    @staticmethod
    def _compress_for_debugger(instruction: str, tool_info, error_msg: str,
                                feedback: str, max_instruction_chars: int = 2000,
                                max_error_chars: int = 1500,
                                max_feedback_chars: int = 800) -> tuple:
        """Compress debugger inputs to prevent context overflow.

        Priority (highest → lowest):
          1. code_snippet — always kept in full (passed separately)
          2. error_msg    — keep last N chars (tail has the actual traceback)
          3. tool_info    — keep only name + parameters (strip verbose descriptions)
          4. instruction  — keep first + last N/2 chars
          5. feedback     — truncate to N chars
        """
        # --- error_msg: keep the tail (traceback is at the end) ---
        if error_msg and len(error_msg) > max_error_chars:
            error_msg = "...[truncated]...\n" + error_msg[-max_error_chars:]

        # --- tool_info: strip verbose descriptions, keep signatures ---
        if tool_info:
            compact_tools = []
            for tool in (tool_info if isinstance(tool_info, list) else [tool_info]):
                if isinstance(tool, dict):
                    compact = {"name": tool.get("name", ""),
                               "parameters": tool.get("parameters", tool.get("params", {}))}
                    compact_tools.append(compact)
                else:
                    compact_tools.append(tool)
            tool_info = compact_tools

        # --- instruction: keep head + tail ---
        if instruction and len(instruction) > max_instruction_chars:
            half = max_instruction_chars // 2
            instruction = (instruction[:half]
                           + "\n...[truncated — middle omitted for brevity]...\n"
                           + instruction[-half:])

        # --- feedback: simple truncation ---
        if feedback and len(feedback) > max_feedback_chars:
            feedback = feedback[:max_feedback_chars] + "\n...[truncated]"

        return instruction, tool_info, error_msg, feedback

    def __call__(self, code_snippet: CodeSnippet):
        if code_snippet.status != "error":
            return f"{code_snippet}: No error occurred, call AnalysisExecution next."
        feedback = None
        task = code_snippet.task
        instruction = code_snippet.instruction
        tool_info = code_snippet.tool_info
        snippet = code_snippet.code_snippet
        error_msg = code_snippet.stderr

        if code_snippet.feedback:
            feedback = code_snippet.feedback

        # Compress fields to prevent context overflow while preserving key info
        instruction, tool_info, error_msg, feedback = self._compress_for_debugger(
            instruction, tool_info, error_msg, feedback
        )

        task_prompt = DEBUGGER_CHAT_TEMPLATE.format(
            user_query=task,
            instruction=instruction,
            tool_info=tool_info,
            code_snippet=snippet,
            error_msg=error_msg,
            feedback=feedback
        )
        print(task_prompt, flush=True)
        raw_code_snippet = self.debugger_agent.run(task_prompt)

        # Extract code with fallbacks for malformed LLM output
        matches = re.findall(self.pattern, raw_code_snippet, re.DOTALL)
        if not matches:
            # Fallback 1: generic ``` fence without language tag
            matches = re.findall(r'```\n(.*?)```', raw_code_snippet, re.DOTALL)
        if not matches:
            # Fallback 2: raw response looks like Python code (no fences)
            stripped = raw_code_snippet.strip()
            python_indicators = ['import ', 'from ', 'def ', 'print(', 'if __name__']
            if sum(1 for kw in python_indicators if kw in stripped) >= 2:
                lines = stripped.split('\n')
                code_start = next(
                    (i for i, line in enumerate(lines)
                     if any(line.strip().startswith(kw) for kw in ['import ', 'from ', 'def ', '#'])),
                    None
                )
                if code_start is not None:
                    matches = ['\n'.join(lines[code_start:])]
                    print("[CodeDebugger] No code fence found, extracted Python code from response.", flush=True)
        if not matches:
            # Fallback 3: return original code unchanged so agent can retry or finish
            print("[CodeDebugger] WARNING: Could not extract code from LLM response. Keeping original code.", flush=True)
            code_snippet.status = "error"
            return f"{code_snippet}: CodeDebugger failed to produce valid code, call Finish next."

        code_snippet.code_snippet = matches[0]
        code_snippet.status = "unexecuted"
        return f"{code_snippet}: debugged, call AnalysisExecution next."


class AnalysisQualityChecker(BaseAction):
    """Checks code quality and provides feedback for improvement."""
    
    def __init__(self, llm_provider: str = None, tmp: float = 0.4, max_iter: int = 3) -> None:
        # Quality checking is a utility task — use UTILITY_LLM if available
        if llm_provider is None:
            llm_provider = get_utility_llm()
        
        action_name = "AnalysisQualityChecker"
        action_desc = "After each successful code execution using the AnalysisExecution action, using this action to provide feedback on the code snippet, evaluating its informativeness and correctness"
        params_doc = {
            "code_snippet": "EXACTLY '<CodeSnippet:xxxx>' format with NO additional text"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )
        
        agent_config = LLMConfig({'temperature': tmp})
        self.quality_checker_agent = AgentLLM(
            llm_config=agent_config, 
            llm_name=llm_provider,
            system_prompt=QUALITY_ASSURANCE_TEMPLATE,
            input_variables=["user_query", "instruction", "tool_info", "code_snippet", "code_output"],
            reasoning_effort="medium"
        )
        self.pattern = r"\[(\w+)\]\s*-\s*(.*)"
        self.iterations = 0
        self.max_iter = max_iter
    
    def __call__(self, code_snippet: CodeSnippet):
        if code_snippet.status != "executed":
            return f"{code_snippet}: Code snippet has not been executed, call AnalysisExecution next."
        
        # Skip directly to the approval iteration exceeds the max_iter
        if self.iterations >= self.max_iter:
            self.iterations = 0
            code_snippet.status = "approved"
            return f"{code_snippet}: Passed, call Finish action next."
        
        decision = None
        feedback = None
        user_query = code_snippet.task
        instruction = code_snippet.instruction
        tool_info = code_snippet.tool_info
        code_snippet_str = code_snippet.code_snippet
        code_output = code_snippet.code_output

        input_prompt = {
            "user_query": user_query, 
            "instruction": instruction, 
            "tool_info": tool_info,
            "code_snippet": code_snippet_str, 
            "code_output": code_output
        }

        max_parse_retries = 5
        for _retry in range(max_parse_retries):
            output = self.quality_checker_agent.run(input_prompt)
            print(output, flush=True)
            match = re.match(self.pattern, output)
            if match:
                decision = match.group(1)  # Decision: Approved, Minor, or Failed
                feedback = match.group(2)  # Feedback
                code_snippet.update_feedback(feedback)
                break
        else:
            # Could not parse a valid decision after retries — approve to avoid hang
            print(f"[AnalysisQualityChecker] Could not parse decision after {max_parse_retries} retries. Auto-approving.", flush=True)
            decision = "Approved"
            feedback = "Auto-approved: quality checker output could not be parsed."
        
        if decision == "Failed":
            self.iterations += 1
            code_snippet.status = "error"
            return f"{output}\n{code_snippet}: Failed, call CodeDebugger action next."
        
        if decision == "Minor":
            print(f"[AnalysisQualityChecker] Non-critical issues found (iteration {self.iterations+1}/{self.max_iter}). Approving with feedback.", flush=True)
            print(f"Feeback from AnalysisQualityChecker (if any):\n{feedback}\n", flush=True)
        
        self.iterations = 0
        code_snippet.status = "approved"
        return f"{code_snippet}: Passed, call Finish action next."


class AnalysisFinishAction(BaseAction):
    """Completes the coding task with the final refined code snippet."""
    
    def __init__(self) -> None:
        action_name = "Finish"
        action_desc = "Complete the task with a refined CodeSnippet Object"
        params_doc = {
            "code_snippet": "EXACTLY '<CodeSnippet:xxxx>' format with NO additional text"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, code_snippet: CodeSnippet) -> Dict:
        """Return the final code and execution output."""
        return {
            "code_snippet": code_snippet.code_snippet,
            "executed_output": code_snippet.code_output,
        }


FinishAct = AnalysisFinishAction()


class Analysis(BaseAgent):
    """
    Agent responsible for generating, executing, and debugging code.
    
    Uses a react-style reasoning approach with actions for code generation,
    execution, debugging, and quality checking.
    """
    
    def __init__(
        self,
        llm: AgentLLM = AgentLLM(
            llm_config=LLMConfig({"temperature": 0.4}),
            llm_name=os.getenv("BACKBONE_LLM")
        ),
        actions: List[BaseAction] = None,
        manager: ABCAgent = None,
        logger: AgentLogger = AgentLogger(FLAG_PRINT=True, PROMPT_DEBUG_FLAG=False),
        reasoning_type: str = "react"
    ):
        if actions is None:
            actions = [
                CodeGenerator(llm_provider=os.getenv("BACKBONE_LLM")), 
                AnalysisExecution(), 
                CodeDebug(llm_provider=os.getenv("BACKBONE_LLM")),
                AnalysisQualityChecker(llm_provider=os.getenv("BACKBONE_LLM")), 
            ]
        
        name = "analysis_agent"
        role = CODE_GENERATION_AGENT_TEMPLATE
                
        super().__init__(
            name=name,
            role=role,
            reasoning_type=reasoning_type,
            llm=llm,
            actions=actions, 
            manager=manager, 
            max_exec_steps=20,
            logger=logger,
        )
        self.prompt_gen = BasePromptGen(
            agent_role=self.role,
            constraint=self.constraint,
            instruction=self.instruction,
        )
        self.context_proposal = None
    
    def _parse_task_package(self, task: TaskPackage) -> Tuple[str, Proposal]:
        """
        Parse TaskPackage to extract user query and instruction.
        
        Args:
            task: TaskPackage containing the task information
            
        Returns:
            Tuple of (user_query, proposal_object)
        """
        try:
            instruction_str = str(task.instruction)
            
            # Try JSON parsing first
            try:
                task_dict = json.loads(instruction_str)
            except json.JSONDecodeError:
                # If JSON fails, try Python literal evaluation
                try:
                    task_dict = ast.literal_eval(instruction_str)
                except (ValueError, SyntaxError):
                    # Manual regex parsing for common pattern
                    task_dict = self._parse_instruction_regex(instruction_str)
            
            # Extract user query and instruction text
            user_query = task_dict.get("task", "")
            instruction_text = task_dict.get("instruction", "")
            
            # Create Proposal object
            proposal = Proposal(
                user_query=user_query,
                proposal=instruction_text
            )
            
            return user_query, proposal
            
        except Exception as e:
            print(f"[Analysis] Error parsing TaskPackage: {e}")
            # Fallback: use entire instruction
            fallback_proposal = Proposal(
                user_query=str(task.instruction),
                proposal=str(task.instruction)
            )
            return str(task.instruction), fallback_proposal
    
    def _parse_instruction_regex(self, instruction_str: str) -> Dict:
        """Parse instruction string using regex patterns."""
        if not (instruction_str.startswith('{') and instruction_str.endswith('}')):
            raise ValueError("Not a dict-like string")
        
        # Find task value
        task_pattern = r"'task':\s*'(.*?)(?=',\s*'instruction')"
        task_match = re.search(task_pattern, instruction_str, re.DOTALL)
        
        if not task_match:
            task_match = re.search(r"'task':\s*'([^']*(?:''[^']*)*)'", instruction_str)
        if not task_match:
            task_match = re.search(r'"task":\s*"([^"]*(?:""[^"]*)*)"', instruction_str)
        
        # Find instruction value
        instruction_match = re.search(r"'instruction':\s*(<Proposal:\d+>)", instruction_str)
        if not instruction_match:
            instruction_match = re.search(r"'instruction':\s*'([^']*(?:''[^']*)*)'", instruction_str)
        if not instruction_match:
            instruction_match = re.search(r'"instruction":\s*"([^"]*(?:""[^"]*)*)"', instruction_str)
        
        if task_match:
            return {
                "task": task_match.group(1),
                "instruction": instruction_match.group(1) if instruction_match else ""
            }
        
        raise ValueError("Could not parse instruction format")
    
    def __call__(self, task: TaskPackage):
        """
        Process a task by parsing it and executing the parent's workflow.
        
        Args:
            task: TaskPackage containing the task information
            
        Returns:
            The result from the parent __call__ method
        """
        user_query, proposal = self._parse_task_package(task)
        self.user_query = user_query
        self.context_proposal = proposal
        task.instruction = str({"task": user_query, "instruction": proposal})
        
        print(f"Task instruction:\n {task.instruction}", flush=True)
        return super().__call__(task)
    
    def __add_inner_actions__(self):
        """Add inner action types based on the reasoning_type."""
        if self.reasoning_type == "react":
            self.actions += [ThinkAct, FinishAct]
        self.actions = list(set(self.actions))
    
    def __next_act__(self, task: TaskPackage, action_chain: ActObsChainType) -> AgentAct:
        """
        Generate the next action for the agent.

        Uses a sliding window over action history: recent steps are kept in full
        detail, while older steps have their observations compressed to one-line
        summaries. This preserves long-horizon reasoning context (what was tried
        and what happened) while reducing prompt size from verbose tool outputs.

        Args:
            task: The task which agent receives and solves
            action_chain: History actions and observations from memory

        Returns:
            The action for agent to execute
        """
        # Compress old history for prompt construction
        compressed_chain = self._compress_action_chain(action_chain)

        action_prompt = self.prompt_gen.action_prompt(
            task=task,
            actions=self.actions,
            action_chain=compressed_chain,
        )
        self.logger.get_prompt(action_prompt)
        raw_action = self.llm_layer(action_prompt)
        self.logger.get_llm_output(raw_action)

        # Pass ORIGINAL action_chain for CodeSnippet reference resolution
        return self.__action_parser__(raw_action, action_chain)

    def _compress_action_chain(
        self, action_chain: ActObsChainType, recent_k: int = 3
    ) -> ActObsChainType:
        """Compress old action-observation pairs via sliding window.

        Recent ``recent_k`` steps are kept verbatim.  Older steps retain their
        full action (so the LLM can see *what* was attempted) but have their
        observation replaced with a deterministic one-line summary (so verbose
        stderr / quality-checker feedback doesn't bloat the prompt).

        This preserves the narrative arc for long-horizon reasoning while
        cutting the bulk of prompt growth in multi-step ReAct loops.
        """
        if len(action_chain) <= recent_k:
            return action_chain

        compressed = []
        cutoff = len(action_chain) - recent_k

        for i, (act, obs) in enumerate(action_chain):
            if i < cutoff:
                summary = self._summarize_observation(act.name, str(obs))
                compressed.append((act, summary))
            else:
                compressed.append((act, obs))

        return compressed

    @staticmethod
    def _summarize_observation(action_name: str, obs: str) -> str:
        """Return a deterministic one-line summary of an observation string."""
        if not obs:
            return "[No output]"

        # ── CodeGenerator ───────────────────────────────────────────────
        if action_name == "CodeGenerator":
            m = re.search(r'(<CodeSnippet:\w+>)', obs)
            sid = m.group(1) if m else "code"
            return f"[Summary] Generated {sid}"

        # ── AnalysisExecution ───────────────────────────────────────────
        if action_name == "AnalysisExecution":
            if "Successfully executed" in obs:
                m = re.search(r'(<CodeSnippet:\w+>)', obs)
                sid = m.group(1) if m else "code"
                return f"[Summary] {sid} executed successfully"
            if "Error occurred" in obs:
                # Surface the first real error/exception line
                for line in obs.splitlines():
                    line = line.strip()
                    if line and ("Error" in line or "Exception" in line) \
                            and not line.startswith("<CodeSnippet"):
                        return f"[Summary] Execution error: {line[:200]}"
                return "[Summary] Execution failed with errors"
            if "cannot help" in obs:
                return "[Summary] Execution failed permanently"
            return f"[Summary] Execution: {obs[:150]}"

        # ── AnalysisQualityChecker ──────────────────────────────────────
        if action_name == "AnalysisQualityChecker":
            if "Passed" in obs:
                return "[Summary] Quality check passed"
            if "Failed" in obs:
                m = re.search(r'\[Failed\]\s*-\s*(.+?)(?:\n|$)', obs)
                fb = m.group(1)[:200] if m else "see feedback"
                return f"[Summary] Quality check failed: {fb}"
            if "has not been executed" in obs:
                return "[Summary] Quality check skipped — code not yet executed"
            return f"[Summary] Quality check: {obs[:150]}"

        # ── CodeDebugger ────────────────────────────────────────────────
        if action_name == "CodeDebugger":
            m = re.search(r'(<CodeSnippet:\w+>)', obs)
            sid = m.group(1) if m else "code"
            if "debugged" in obs:
                return f"[Summary] {sid} debugged, ready for re-execution"
            return f"[Summary] Debug: {obs[:150]}"

        # ── Think — keep short thoughts verbatim ────────────────────────
        if action_name == "Think":
            if len(obs) <= 300:
                return obs
            return f"[Summary] {obs[:250]}..."

        # ── Default fallback ────────────────────────────────────────────
        if len(obs) <= 300:
            return obs
        return f"[Summary] {obs[:250]}..."
    
    def __action_parser__(self, raw_action: str, action_chain: ActObsChainType) -> AgentAct:
        """
        Parse the generated content to an executable action.

        Args:
            raw_action: LLM generated text
            action_chain: Action history chain

        Returns:
            An executable action wrapper
        """
        action_name, args, PARSE_FLAG = parse_action(raw_action)
        
        # Resolve code_snippet references from action chain
        if 'code_snippet' in args and args['code_snippet'] is not None:
            cs_ref = str(args['code_snippet']).strip()
            resolved = False
            # Try exact match first
            for _, p_obs in reversed(action_chain):
                if isinstance(p_obs, CodeSnippet) and p_obs.get_id() == cs_ref:
                    args["code_snippet"] = p_obs
                    resolved = True
                    break
            # Fallback: find any CodeSnippet in action chain (observations are often strings)
            if not resolved:
                for _, p_obs in reversed(action_chain):
                    if isinstance(p_obs, CodeSnippet):
                        args["code_snippet"] = p_obs
                        resolved = True
                        break
            # Last resort: check if any observation string contains a CodeSnippet ID pattern
            if not resolved and "<CodeSnippet:" in cs_ref:
                for a_act, a_obs in reversed(action_chain):
                    if hasattr(a_act, 'params') and 'code_snippet' in a_act.params:
                        param = a_act.params['code_snippet']
                        if isinstance(param, CodeSnippet):
                            args["code_snippet"] = param
                            resolved = True
                            break
        
        # Resolve instruction references from action chain or context
        if 'instruction' in args and args['instruction'] is not None:
            found_proposal = self._find_proposal(args['instruction'], action_chain)
            if found_proposal is not None:
                args["instruction"] = found_proposal
        
        return AgentAct(name=action_name, params=args)
    
    def _find_proposal(self, instruction_ref: str, action_chain: ActObsChainType) -> Optional[Proposal]:
        """Find a Proposal object from action chain or context."""
        instruction_ref = str(instruction_ref)
        
        # Try to find from action chain with ID matching
        for _, p_obs in reversed(action_chain):
            if isinstance(p_obs, Proposal) and p_obs.get_id() == instruction_ref:
                return p_obs
        
        # Use stored context proposal
        if hasattr(self, 'context_proposal') and self.context_proposal is not None:
            return self.context_proposal
        
        # Find any Proposal object in action chain
        for _, p_obs in reversed(action_chain):
            if isinstance(p_obs, Proposal):
                return p_obs
        
        return None

    def forward(self, task: TaskPackage, agent_act: AgentAct) -> str:
        """
        Forward the action to get the observation.

        Args:
            task: The task which agent receives and solves
            agent_act: The action wrapper for execution

        Returns:
            Observation from action execution
        """
        act_found_flag = False
        param_parse_flag = False
        observation = None
        
        for action in self.actions:
            if act_match(agent_act.name, action):
                act_found_flag = True
                try:
                    observation = action(**agent_act.params)
                except Exception as e:
                    observation = MISS_ACTION_PARAM.format(
                        param_doc=action.params_doc,
                        failed_param=agent_act.params
                    )
                    return observation
                
                if agent_act.name == FinishAct.action_name:
                    task.answer = observation
                    task.completion = "completed"
                    
            if action.action_name in agent_act.name:
                param_parse_flag = True
        
        if act_found_flag:
            return observation
        if param_parse_flag:
            return WRONG_ACTION_PARAM
        return ACTION_NOT_FOUND_MESS
    