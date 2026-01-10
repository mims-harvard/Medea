import ast
import importlib
import inspect
import json
import os
import re
import sys
from typing import Dict, List, Tuple

from agentlite.actions import PlanAct, ThinkAct
from agentlite.actions.BaseAction import BaseAction
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.agents.agent_utils import *
from agentlite.commons import AgentAct, TaskPackage
from agentlite.commons.AgentAct import ActObsChainType

# Use relative imports within package
from ..tool_space.action_functions import *
from ..tool_space.humanbase import *
from ..tool_space.id_checkers import *
from ..tool_space.env_utils import get_backbone_llm

from .agent_llms import AgentLLM, LLMConfig, parse_action
from .BasePrompt import BasePromptGen
from .prompt_template import *
from .utils import FlushAgentLogger as AgentLogger
from .utils import Proposal


# Load tool_info.json using package-relative path
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


# Constants
JSON_CODE_BLOCK_PATTERN = r'```json\n(.*?)```'
PYTHON_CODE_BLOCK_PATTERN = r'^```python\s*(.*?)\s*```$'
STATUS_PATTERN = r"^(?:-\s*)?\[(\w+)\]\s*-\s*(.*)"


class ProposalToolSelector:
    """Selects relevant tools for a given user query using LLM-based analysis."""
    
    def __init__(self, llm_provider: str, tmp: float = 0.4) -> None:
        """
        Initialize the ProposalToolSelector for proposal generation.

        Args:
            llm_provider: LLM provider name
            tmp: Temperature setting for the LLM. Default is 0.4.
        """
        self.tool_list = AVALIBLE_TOOL
        agent_config = LLMConfig({'temperature': tmp})
        self.selector = AgentLLM(
            llm_config=agent_config,
            llm_name=llm_provider,
            system_prompt=PROPOSAL_TOOL_SELECTION_TEMPLATE,
            input_variables=["user_query", "tool_info"]
        )
    
    def _extract_code_block(self, response: str) -> str:
        """Extract code from JSON or Python code blocks."""
        if "```json" in response:
            matches = re.findall(JSON_CODE_BLOCK_PATTERN, response, re.DOTALL)
            return matches[0] if matches else response
        if "```python" in response:
            matches = re.search(PYTHON_CODE_BLOCK_PATTERN, response, re.DOTALL)
            return matches.group(1).strip() if matches else response.strip()
        return response
    
    def __call__(self, user_query: str, max_attempts: int = 3) -> List[Dict]:
        """
        Execute the tool selection process based on the user query.

        Args:
            user_query: The user query to analyze for tool selection.
            max_attempts: Number of attempts to retry LLM execution in case of failure.

        Returns:
            List of JSON objects representing the relevant tools.
        """
        input_prompt = {"user_query": user_query, "tool_info": self.tool_list}
        relevant_tools = []
        
        for attempt in range(max_attempts):
            try:
                response = self.selector.run(input_prompt)
                response = self._extract_code_block(response)
                
                relevant_tools = ast.literal_eval(response)
                if isinstance(relevant_tools, list):
                    break
                raise ValueError("LLM response is not a valid list.")
                
            except Exception as e:
                print(f"Tool selection error on attempt {attempt + 1}: {e}", flush=True)
                if attempt == max_attempts - 1:
                    print("Failed to select tools, using all available tools as fallback", flush=True)
                    relevant_tools = [tool.get("name") for tool in self.tool_list]

        tool_info_json = [
            tool for tool in self.tool_list if tool.get("name") in relevant_tools
        ]
        print(f"Selected Tools for Proposal: {[tool.get('name') for tool in tool_info_json]}", flush=True)
        return tool_info_json


class ResearchPlanDraft(BaseAction):
    """Generates a proposal draft with step-by-step procedures to solve the user query."""
    
    def __init__(self, llm_provider: str = None, tmp: float = 0.4):
        # Get LLM provider with helpful error message if not provided
        if llm_provider is None:
            llm_provider = get_backbone_llm("gpt-4o")
        
        action_name = "ResearchPlanDraft"
        action_desc = "Generate a proposal draft based on the task. The proposal should outline the objective, step-by-step procedure to achieve the objective."
        params_doc = {
            "user_query": "the original user query provided by user",
            "proposal_draft": "Proposal Object reference - MUST be EXACTLY '<Proposal:xxxx>' format (e.g., '<Proposal:2362>') with NO additional text. Use None for first iteration. The Proposal object ALREADY contains all feedback internally - do NOT append feedback text to the reference."
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )
        
        agent_config = LLMConfig({"temperature": tmp})
        self.tool_selector = ProposalToolSelector(llm_provider=llm_provider, tmp=tmp)
        self.proposal_draft = AgentLLM(
            llm_config=agent_config,
            llm_name=llm_provider,
            system_prompt=PROPOSAL_DRAFT_TEMPLATE,
            input_variables=["user_query", "tool_list", "proposal_feedback"]
        )

    def __call__(self, user_query: str, proposal_draft: Proposal = None) -> Proposal:
        """
        Generate or refine a proposal draft.
        
        Args:
            user_query: The original user query
            proposal_draft: Existing proposal to refine (None for first iteration)
            
        Returns:
            A Proposal object containing the generated or refined proposal
        """
        proposal_feedback = self._prepare_feedback(proposal_draft)
        selected_tools = self._select_tools(user_query)
        
        input_prompt = {
            "user_query": user_query,
            "tool_list": selected_tools,
            "proposal_feedback": proposal_feedback
        }
        
        proposal_draft_result = self.proposal_draft.run(input_prompt)
        return Proposal(user_query=user_query, proposal=proposal_draft_result)
    
    def _prepare_feedback(self, proposal_draft: Proposal) -> str:
        """Prepare feedback content from previous proposal iterations."""
        if not proposal_draft:
            return ""
        
        if proposal_draft.get_current_mapper_feedback() is None:
            return f"{proposal_draft}: Please check {proposal_draft} use ContextVerification action."
        
        return f"-----\nProposal Draft from last generation:\n\n{proposal_draft.get_summary()}"
    
    def _select_tools(self, user_query: str) -> List[Dict]:
        """Select relevant tools based on user query with fallback to all tools."""
        try:
            selected_tools = self.tool_selector(user_query)
            print(f"[ResearchPlanDraft] Selected {len(selected_tools)} tools out of {len(AVALIBLE_TOOL)} available", flush=True)
            return selected_tools
        except Exception as e:
            print(f"[ResearchPlanDraft] Tool selection failed, using all tools: {e}", flush=True)
            return AVALIBLE_TOOL



class ContextVerification(BaseAction):
    """Validates context compatibility for proposed tools and parameters."""
    
    def __init__(self, llm_provider: str = None, tmp: float = 0.4):
        # Get LLM provider with helpful error message if not provided
        if llm_provider is None:
            llm_provider = get_backbone_llm("gpt-4o")
        
        action_name = "ContextVerification"
        action_desc = "Validate context compatibility for proposed tools and parameters"
        params_doc = {
            "proposal_draft": "Proposal object containing the draft to validate"
        }
        super().__init__(action_name=action_name, action_desc=action_desc, params_doc=params_doc)
        
        self._load_configurations()
        
        agent_config = LLMConfig({"temperature": tmp})
        self.context_checker = AgentLLM(
            llm_config=agent_config,
            llm_name=llm_provider,
            system_prompt=CONTEXT_CHECKER_TEMPLATE.format(
                tool_id_checker=json.dumps(self.checker_configs, indent=2)
            )
        )

    def _load_configurations(self):
        """Load and organize checker configurations."""
        # Use package-relative path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.dirname(current_dir)
        checker_path = os.path.join(package_root, "tool_space", "tool_id_checker.json")
        
        # Fallback: if not found, try relative to current working directory (for development)
        if not os.path.exists(checker_path):
            checker_path = "tool_space/tool_id_checker.json"
        
        with open(checker_path, 'r') as f:
            self.checker_configs = json.load(f)
        
        self.checker_to_tools = {}
        self.tool_to_checker = {}
        self.checker_functions = self._load_checker_functions()
        
        # Build mappings from config
        for config in self.checker_configs:
            tools = config.get('tool', [])
            for checker_info in config.get('associated_id_checker', []):
                checker_name = checker_info['checker_name']
                self.checker_to_tools[checker_name] = tools
                for tool in tools:
                    self.tool_to_checker[tool] = checker_name

    def _load_checker_functions(self) -> Dict:
        """Dynamically load all checker functions from id_checkers module."""
        try:
            id_checkers_module = importlib.import_module('tool_space.id_checkers')
            
            checker_functions = {
                name: obj
                for name, obj in inspect.getmembers(id_checkers_module, inspect.isfunction)
                if name.endswith('_checker')
            }
            
            print(f"Loaded {len(checker_functions)} checker functions: {list(checker_functions.keys())}", flush=True)
            return checker_functions
            
        except Exception as e:
            print(f"Failed to dynamically load checker functions: {e}", flush=True)
            return {}
    
    def _convert_null_to_none(self, obj):
        """Recursively convert null-like values to None in nested structures."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {key: self._convert_null_to_none(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._convert_null_to_none(item) for item in obj]
        if isinstance(obj, str) and obj.lower() in ['null', 'none']:
            return None
        return obj

    def _extract_context_pairs(self, proposal_text: str) -> List[Dict]:
        """Extract tool-checker pairs from proposal text."""
        try:
            response = self.context_checker.run(proposal_text).strip()
            
            # Clean code blocks
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            pairs = json.loads(response.strip())
            pairs = self._convert_null_to_none(pairs)
            print(f"Context check output: {json.dumps(pairs, indent=2)}", flush=True)
            return pairs
            
        except Exception as e:
            print(f"Failed to extract context pairs: {e}", flush=True)
            return []

    def _get_param_config(self, checker_name: str) -> Dict:
        """Get parameter configuration for a checker."""
        for config in self.checker_configs:
            for checker_info in config.get('associated_id_checker', []):
                if checker_info['checker_name'] == checker_name:
                    return {p['param_name']: p for p in checker_info['input_params']}
        return {}

    def _prepare_parameters(self, checker_name: str, input_params: Dict, tool_name: str = None) -> Dict:
        """Prepare and validate parameters for checker."""
        param_config = self._get_param_config(checker_name)
        prepared_params = {}
        
        for param_name, param_def in param_config.items():
            if param_name in input_params:
                prepared_params[param_name] = input_params[param_name]
            elif param_name == 'model_name' and tool_name:
                inferred_model = self._infer_model_from_tool(tool_name, param_def.get('allowed_values', []))
                if inferred_model:
                    prepared_params[param_name] = inferred_model
                elif param_def.get('default') is not None:
                    prepared_params[param_name] = param_def['default']
            elif param_def.get('default') is not None:
                prepared_params[param_name] = param_def['default']
        
        return prepared_params

    def _infer_model_from_tool(self, tool_name: str, allowed_values: List[str]) -> str:
        """Dynamically infer model name from tool name using allowed values."""
        if not allowed_values:
            return None
            
        tool_name_lower = tool_name.lower()
        
        for model in allowed_values:
            model_lower = model.lower()
            if model_lower in tool_name_lower or tool_name_lower in model_lower:
                return model
        
        return None

    def _validate_parameters(self, checker_name: str, params: Dict) -> List[str]:
        """Validate parameters against configuration."""
        param_config = self._get_param_config(checker_name)
        errors = []
        
        for param_name, param_def in param_config.items():
            if not param_def.get('required', True):
                continue
                
            if param_name not in params or params[param_name] is None:
                errors.append(f"Missing required parameter: {param_name}")
                continue
            
            value = params[param_name]
            
            # Check constraints
            if not param_def.get('allow_empty', True):
                is_empty = (isinstance(value, str) and not value.strip()) or \
                           (isinstance(value, list) and len(value) == 0)
                if is_empty:
                    errors.append(f"{param_name} cannot be empty")
            
            if param_def.get('min_length') and isinstance(value, list):
                if len(value) < param_def['min_length']:
                    errors.append(f"{param_name} requires at least {param_def['min_length']} items")
            
            if param_def.get('allowed_values'):
                if isinstance(value, str) and value.lower() not in [v.lower() for v in param_def['allowed_values']]:
                    errors.append(f"{param_name} must be one of {param_def['allowed_values']}")
        
        return errors

    def _run_checker(self, checker_name: str, params: Dict) -> Tuple[bool, str]:
        """Run the actual checker function."""
        try:
            checker_func = self.checker_functions[checker_name]
            print(f"Running {checker_name} with params: {params}", flush=True)
            return checker_func(**params)
        except Exception as e:
            print(f"Error running {checker_name}: {e}", flush=True)
            return False, f"Checker error: {str(e)}"
    
    def __call__(self, proposal_draft: Proposal, attempts: int = 3) -> str:
        """
        Main validation workflow.
        
        Args:
            proposal_draft: The proposal to validate
            attempts: Number of retry attempts for validation
            
        Returns:
            Validation result message
        """
        if not isinstance(proposal_draft, Proposal):
            return f"Invalid input: Expected Proposal object, got {type(proposal_draft).__name__}"

        proposal_text = f"[User Query]:\n{proposal_draft.get_query()}\n\n[Proposal Draft]:\n{proposal_draft.get_proposal()}"
        feedbacks, all_valid = self._validate_with_retries(proposal_text, attempts)
        
        proposal_draft.update_id_feedback(feedbacks)

        if all_valid:
            return f"{proposal_draft}: All context validations passed. Proceed to IntegrityVerification."
        return f"{proposal_draft}: Context validation issues found. {'; '.join(feedbacks)}. Please refine the proposal."
    
    def _validate_with_retries(self, proposal_text: str, attempts: int) -> Tuple[List[str], bool]:
        """Execute validation with retry logic."""
        feedbacks = []
        all_valid = True
        
        for attempt in range(attempts):
            try:
                pairs_to_check = self._extract_context_pairs(proposal_text)
                if not pairs_to_check:
                    feedbacks = ["No validation requirements found in proposal"]
                    break
                
                feedbacks, all_valid = self._process_validation_pairs(pairs_to_check)
                break
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}", flush=True)
                if attempt == attempts - 1:
                    feedbacks = [f"Context validation failed after {attempts} attempts: {str(e)}"]
                    all_valid = False
        
        return feedbacks, all_valid
    
    def _process_validation_pairs(self, pairs_to_check: List[Dict]) -> Tuple[List[str], bool]:
        """Process each validation pair and collect results."""
        feedbacks = []
        all_valid = True
        
        for pair in pairs_to_check:
            tool_name = pair.get('tool')
            checker_name = pair.get('checker_name')
            input_params = pair.get('input_params', {})
            
            if not self._is_valid_checker_association(checker_name, tool_name):
                continue
            
            prepared_params = self._prepare_parameters(checker_name, input_params, tool_name)
            
            validation_errors = self._validate_parameters(checker_name, prepared_params)
            if validation_errors:
                error_msg = f"Parameter validation failed for {checker_name}: {'; '.join(validation_errors)}"
                print(error_msg, flush=True)
                feedbacks.append(error_msg)
                all_valid = False
                continue
            
            is_available, feedback = self._run_checker(checker_name, prepared_params)
            feedbacks.append(feedback)
            if not is_available:
                all_valid = False
        
        return feedbacks, all_valid
    
    def _is_valid_checker_association(self, checker_name: str, tool_name: str) -> bool:
        """Validate that the checker exists and is associated with the tool."""
        if checker_name not in self.checker_functions:
            print(f"Unknown checker: {checker_name}", flush=True)
            return False
        
        if tool_name not in self.checker_to_tools.get(checker_name, []):
            print(f"Tool {tool_name} not associated with checker {checker_name}", flush=True)
            return False
        
        return True


class IntegrityVerification(BaseAction):
    """Checks the quality of proposal drafts and provides feedback."""
    
    def __init__(self, llm_provider: str = None, tmp: float = 0.4, max_iter: int = 1) -> None:
        # Get LLM provider with helpful error message if not provided
        if llm_provider is None:
            llm_provider = get_backbone_llm("gpt-4o")
        
        action_name = "IntegrityVerification"
        action_desc = "Check the quality of the proposal draft and provide feedback on the quality of the proposal draft."
        params_doc = {
            "proposal_draft": "Proposal object (e.g., <Proposal:xxxx>) - the proposal draft from the proposal_draft action"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )
        agent_config = LLMConfig({"temperature": tmp})

        tool_list_json = json.dumps(AVALIBLE_TOOL, indent=2)
        tool_list_escaped = tool_list_json.replace("{", "{{").replace("}", "}}")

        self.llm_evaluater = AgentLLM(
            llm_config=agent_config,
            llm_name=llm_provider,
            system_prompt=PROPOSAL_QUALITY_TEMPLATE.format(tool_list=tool_list_escaped)
        )
        self.iterations = 0
        self.max_iter = max_iter
        

    def __call__(self, proposal_draft: Proposal) -> Proposal:
        """
        Evaluate the quality of a proposal draft.
        
        Args:
            proposal_draft: The proposal to evaluate
            
        Returns:
            The proposal with updated status, or a feedback message string
        """
        if proposal_draft.get_current_mapper_feedback() is None:
            return f"{proposal_draft}: Please call ContextVerification action first."
        
        if self.iterations >= self.max_iter:
            self.iterations = 0
            proposal_draft.update_status("Approved")
            return proposal_draft
        
        previous_feedback, current_feedback = proposal_draft.retrieve_mapper_feedback_trace()
        print(f"[User query]: {proposal_draft.get_query()}", flush=True)
        
        prompt = self._build_evaluation_prompt(proposal_draft, previous_feedback, current_feedback)
        result = self._evaluate_with_retries(prompt, proposal_draft)
        
        return result
    
    def _build_evaluation_prompt(self, proposal_draft: Proposal, previous_feedback: str, current_feedback: str) -> str:
        """Build the evaluation prompt from proposal and feedback."""
        return (
            f"User Query:\n{proposal_draft.get_query()}\n\n"
            f"Proposal Draft:\n{proposal_draft.get_proposal()}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"CONTEXT VERIFICATION FEEDBACK (Alternative contexts may have been suggested):\n"
            f"{current_feedback}\n\n"
            f"Previous Iteration Feedback:\n{previous_feedback}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"IMPORTANT: If ContextVerification suggested alternative entities (cell types, genes, etc.),\n"
            f"these are VALIDATED and ACCEPTABLE to use. Do NOT fail the proposal just because\n"
            f"it uses alternatives - instead, verify that the proposal clearly documents why\n"
            f"alternatives were chosen and that all tool parameters are correct for the alternatives used."
        )
    
    def _evaluate_with_retries(self, prompt: str, proposal_draft: Proposal, max_retries: int = 3) -> Proposal:
        """Evaluate proposal with retry logic."""
        for retry_count in range(max_retries):
            try:
                print(f"[IntegrityVerification] Attempt {retry_count + 1}/{max_retries}", flush=True)
                feedback = self.llm_evaluater.run(prompt)
                print(f"[IntegrityVerification] Raw response: {feedback[:200]}...", flush=True)
                
                result = self._process_feedback(feedback, proposal_draft)
                if result is not None:
                    return result
                
            except Exception as e:
                print(f"[IntegrityVerification] Error on attempt {retry_count + 1}: {e}", flush=True)
        
        print(f"[IntegrityVerification] Max retries reached, auto-approving proposal", flush=True)
        proposal_draft.update_status("Approved")
        return proposal_draft
    
    def _process_feedback(self, feedback: str, proposal_draft: Proposal):
        """Process LLM feedback and update proposal status."""
        match = re.match(STATUS_PATTERN, feedback, flags=re.DOTALL)
        if not match:
            print(f"[IntegrityVerification] Response doesn't match expected pattern", flush=True)
            print(f"[IntegrityVerification] Full response: {feedback}", flush=True)
            return None
        
        status = match.group(1)
        detail_feedback = match.group(2)
        
        if status not in ["Failed", "Approved"]:
            print(f"[IntegrityVerification] Invalid status '{status}', expected 'Failed' or 'Approved'", flush=True)
            return None
        
        proposal_draft.update_status(status)
        proposal_draft.add_feedback(detail_feedback)
        print(feedback, flush=True)
        
        if status == "Failed":
            self.iterations += 1
            return f'The proposal has several issues that need to be addressed:\n {detail_feedback}.\nPlease refine {proposal_draft} use ResearchPlanDraft action.'
        
        self.iterations = 0
        proposal_draft.update_status("Approved")
        return proposal_draft.log_summary()


class ProposalFinishAction(BaseAction):
    """Completes the proposal generation task with the final refined proposal."""
    
    def __init__(self) -> None:
        action_name = "Finish"
        action_desc = "Complete the task with a refined Proposal Object"
        params_doc = {
            "proposal_draft": "the Proposal Object (e.g., <Proposal:xxxx>)"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, proposal_draft: Proposal) -> Dict:
        """Return the final proposal as a dictionary."""
        response = {"proposal_draft": proposal_draft}
        return response


FinishAct = ProposalFinishAction()


class ResearchPlanning(BaseAgent):
    """
    Agent responsible for generating and refining research proposals.
    
    Uses a react-style reasoning approach with actions for drafting,
    context validation, and quality assurance.
    """
    
    def __init__(
        self,
        llm: AgentLLM = AgentLLM(
            LLMConfig({"temperature": 0.4}),
            llm_name=os.getenv("BACKBONE_LLM"),
        ),
        actions: List[BaseAction] = None,
        manager: ABCAgent = None,
        logger: AgentLogger = AgentLogger(FLAG_PRINT=True, PROMPT_DEBUG_FLAG=False),
    ):
        if actions is None:
            actions = [
                ResearchPlanDraft(llm_provider=os.getenv("BACKBONE_LLM")),
                ContextVerification(llm_provider=os.getenv("BACKBONE_LLM")),
                IntegrityVerification(llm_provider=os.getenv("BACKBONE_LLM")),
            ]
        
        name = "research_plan_agent"
        reasoning_type = "react"
        role = RESEARCH_PLAN_AGENT_TEMPLATE

        super().__init__(
            name=name,
            role=role,
            reasoning_type=reasoning_type,
            llm=llm,
            actions=actions,
            manager=manager,
            max_exec_steps=60,
            logger=logger,
        )
        self.prompt_gen = BasePromptGen(
            agent_role=self.role,
            constraint=self.constraint,
            instruction=self.instruction,
        )
    
    def __next_act__(self, task: TaskPackage, action_chain: ActObsChainType) -> AgentAct:
        """
        Generate the next action for the agent.

        Args:
            task: The task which agent receives and solves
            action_chain: History actions and observations of this task from memory

        Returns:
            The action for agent to execute
        """
        action_prompt = self.prompt_gen.action_prompt(
            task=task,
            actions=self.actions,
            action_chain=action_chain,
        )
        
        self.logger.get_prompt(action_prompt)
        raw_action = self.llm_layer(action_prompt)
        self.logger.get_llm_output(raw_action)
        
        return self.__action_parser__(raw_action, action_chain)
    
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
        
        # Resolve proposal_draft references from action chain
        if 'proposal_draft' in args and args['proposal_draft'] is not None:
            for _, p_obs in reversed(action_chain):
                if isinstance(p_obs, Proposal) and p_obs.get_id() == args['proposal_draft']:
                    args["proposal_draft"] = p_obs
                    break
        
        return AgentAct(name=action_name, params=args)
    

    def forward(self, task: TaskPackage, agent_act: AgentAct):
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
                    print(e, flush=True)
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
        return ACION_NOT_FOUND_MESS
    
    def __add_inner_actions__(self):
        """Add inner action types based on the reasoning_type."""
        action_map = {
            "react": [ThinkAct, FinishAct],
            "act": [FinishAct],
            "planact": [PlanAct, FinishAct],
            "planreact": [PlanAct, ThinkAct, FinishAct],
        }
        
        if self.reasoning_type in action_map:
            self.actions += action_map[self.reasoning_type]
        else:
            Warning("Not yet supported. Will using react instead.")
            self.actions += [ThinkAct, ProposalFinishAction()]
        
        self.actions = list(set(self.actions))
        
if __name__ == "__main__":
    # Example usage
    test_task = (
        "Which of these genes ['NTMT1', 'C1QBP', 'MPDU1', 'B4GALT7', 'UBE2E1'] "
        "in granulocyte is the strongest candidate as a therapeutic target for RA? "
        "Use cosine similarity to rank the genes"
    )
    
    llm_config_dict = {"temperature": 0.5}
    llm_config = LLMConfig(llm_config_dict)
    llm = AgentLLM(llm_config=llm_config)
    
    test_task_pack = TaskPackage(instruction=test_task)
    research_plan_agent = ResearchPlanning(llm=llm)
    response = research_plan_agent(test_task_pack)
    print(response)