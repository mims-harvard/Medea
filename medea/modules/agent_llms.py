import ast
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import dotenv

dotenv.load_dotenv()

# Import chat_completion from gpt_utils
from ..tool_space.gpt_utils import chat_completion
from ..tool_space.env_utils import get_backbone_llm, get_env_with_error

# Constants
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TEMPERATURE = 0.4  # Default temperature for all LLM calls


class LLMConfig:
    """Simplified configuration for LLM models."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize LLM configuration.
        
        Args:
            config_dict: Optional dictionary to override default values
        """
        self.temperature: float = DEFAULT_TEMPERATURE
        self.llm_name: Optional[str] = None
        
        if config_dict is not None:
            self.__dict__.update(config_dict)


class BaseLLM:
    """Base class for LLM implementations."""
    
    def __init__(self, llm_config: LLMConfig) -> None:
        """Initialize base LLM."""
        self.temperature: float = llm_config.temperature

    def __call__(self, prompt: str) -> str:
        """Allow calling the LLM instance directly."""
        return self.run(prompt)

    def run(self, prompt: str) -> str:
        """Execute the LLM with the given prompt."""
        raise NotImplementedError("Subclasses must implement run()")


class AgentLLM(BaseLLM):
    """
    Simplified LLM interface using chat_completion backend.
    
    This is a thin wrapper around the chat_completion function from gpt_utils,
    which handles all provider logic (OpenRouter, Azure, OpenAI, Claude, etc.)
    """
    
    def __init__(
        self, 
        llm_config: LLMConfig, 
        llm_name: Optional[str] = None,
        provider_name: Optional[str] = None, 
        system_prompt: Optional[str] = None,
        input_variables: Optional[List[str]] = None,
        verbose: bool = False
    ):
        """
        Initialize the LLM interface.
        
        Args:
            llm_config: Configuration for the LLM
            llm_name: Name of the LLM model to use (e.g., 'gpt-4o', 'claude')
            provider_name: Ignored - kept for backward compatibility
            system_prompt: Optional system prompt template with {variable} placeholders
            input_variables: List of variable names expected in the system prompt
            verbose: If True, print initialization messages
        """
        super().__init__(llm_config)
        
        self.model = llm_name or get_backbone_llm("gpt-4o")
        self.system_prompt = system_prompt
        self.input_variables = input_variables or ["prompt"]
        self.use_openrouter = get_env_with_error("USE_OPENROUTER", default="true").lower() == "true"
        
        if verbose:
            provider = "OpenRouter" if self.use_openrouter else "Direct API"
            print(f"Initialized LLM: {self.model} (via {provider})")
    
    def run(
        self, 
        input_data: Union[str, Dict[str, Any]], 
        max_attempts: int = DEFAULT_MAX_ATTEMPTS, 
        retry_delay: float = 1.0
    ) -> str:
        """
        Execute the prompt with retry logic.
        
        Args:
            input_data: Either a string prompt or a dictionary of input variables
            max_attempts: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds (unused, chat_completion handles retries)
            
        Returns:
            The LLM response content
            
        Raises:
            Exception: If all retry attempts fail
            ValueError: If input_data is invalid or missing required variables
        """
        # Prepare messages from input
        messages = self._prepare_messages(input_data)
        
        # Call chat_completion (handles all provider logic, retries, etc.)
        try:
            response = chat_completion(
                messages=messages,
                temperature=self.temperature,
                model=self.model,
                mod='chat',  # Already in message format
                attempts=max_attempts,
                use_openrouter=self.use_openrouter
            )
            return response
        except Exception as e:
            print(f"[AgentLLM] Error calling {self.model}: {e}", flush=True)
            raise
    
    def _prepare_messages(self, input_data: Union[str, Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Prepare messages for the LLM from input data.
        
        Args:
            input_data: Either a string prompt or a dictionary of input variables
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
            
        Raises:
            ValueError: If input_data is invalid or missing required variables
        """
        messages = []
        
        # Add system message if system prompt exists
        if self.system_prompt:
            system_content = self._format_system_prompt(input_data)
            messages.append({"role": "system", "content": system_content})
        
        # Add user message
        user_message = self._extract_user_message(input_data)
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _format_system_prompt(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """Format system prompt with input variables."""
        if not isinstance(input_data, dict):
            return self.system_prompt
        
        system_content = self.system_prompt
        for key, value in input_data.items():
            placeholder = "{" + key + "}"
            system_content = system_content.replace(placeholder, str(value))
        
        return system_content
    
    def _extract_user_message(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """Extract user message content from input data."""
        if isinstance(input_data, str):
            return input_data
        
        if isinstance(input_data, dict):
            if "prompt" in input_data:
                return input_data["prompt"]
            
            # Check if all required variables are provided
            if self.system_prompt and self.input_variables:
                missing_vars = [var for var in self.input_variables if var not in input_data]
                if missing_vars:
                    raise ValueError(f"Missing required input variables: {missing_vars}")
                return "Please process the information provided in the system prompt."
        
        raise ValueError("Invalid input_data format")


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

class LLMProviderRegistry:
    """
    Simplified registry for backward compatibility.
    
    Provider logic is now handled by chat_completion in gpt_utils.
    """
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return ["OpenRouter", "AzureOpenAI", "OpenAI", "Claude", "Gemini", "Ollama"]
    
    @classmethod
    def use_openrouter_globally(cls) -> None:
        """Enable OpenRouter globally."""
        os.environ["USE_OPENROUTER"] = "true"
        print("[LLMProviderRegistry] ✓ OpenRouter enabled globally")


# ============================================================================
# ACTION PARSING UTILITIES
# ============================================================================

def parse_action(string: str) -> Tuple[str, Dict, bool]:
    """
    Parse an action string into action type, arguments, and parse status.
    
    Expected format: ActionName[{param1: value1, param2: value2}]
    
    Args:
        string: The action string to parse
        
    Returns:
        Tuple of (action_type, arguments_dict, parse_success_flag)
    """
    # Extract action line if multiple lines present
    string = _extract_action_line(string)
    string = _clean_action_string(string)
    
    # Parse action structure
    action_match = re.match(r'(\w+)\[(.*)\]$', string, flags=re.DOTALL)
    
    if not action_match:
        return string, {}, False
    
    action_type = action_match.group(1).strip()
    bracket_content = action_match.group(2).strip()
    
    # Extract and parse arguments
    arguments, PARSE_FLAG = _extract_arguments(bracket_content)
    
    if not PARSE_FLAG:
        return string, {}, False
    
    # Validate special argument patterns
    PARSE_FLAG = _validate_special_arguments(arguments)
    
    return action_type, arguments, PARSE_FLAG


def _extract_action_line(string: str) -> str:
    """Extract the line containing the action from multi-line input."""
    for line in string.split("\n"):
        if 'Action:' in line:
            return line
    return string


def _clean_action_string(string: str) -> str:
    """Clean and normalize the action string."""
    string = string.strip("Action").strip(":").strip()
    # Remove malformed patterns like ][{...}]
    string = re.sub(r'\]\[\{.*?\}\]$', '', string)
    return string


def _extract_arguments(bracket_content: str) -> Tuple[Dict, bool]:
    """Extract and parse JSON arguments from bracket content."""
    if not bracket_content.startswith('{'):
        return {}, False
    
    # Find complete JSON using brace counting
    arguments_str = _extract_complete_json(bracket_content)
    
    if not arguments_str:
        return {}, False
    
    # Clean up escape sequences
    arguments_str = arguments_str.replace("\\'", "'")
    
    # Try parsing the JSON/dict
    return _parse_json_or_dict(arguments_str)


def _extract_complete_json(content: str) -> str:
    """Extract complete JSON object using brace counting."""
    brace_count = 0
    end_pos = 0
    
    for i, char in enumerate(content):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_pos = i + 1
                break
    
    return content[:end_pos] if end_pos > 0 else ""


def _parse_json_or_dict(arguments_str: str) -> Tuple[Dict, bool]:
    """Parse arguments string as JSON or Python dict."""
    # Try JSON first
    try:
        return json.loads(arguments_str), True
    except json.JSONDecodeError:
        pass
    
    # Fallback to ast.literal_eval for Python dict syntax
    try:
        return ast.literal_eval(arguments_str), True
    except (ValueError, SyntaxError):
        return {}, False


def _validate_special_arguments(arguments: Dict) -> bool:
    """
    Validate special argument patterns (proposal_draft, code_snippet, instruction).
    
    Ensures object references are EXACT format with no additional text.
    """
    validators = {
        'proposal_draft': r'^<Proposal:\d{4}>$',
        'code_snippet': r'^<CodeSnippet:\d{4}>$',
        'instruction': r'^<Proposal:\d{4}>$'
    }
    
    for arg_name, pattern in validators.items():
        if arg_name in arguments and arguments[arg_name] is not None:
            value = arguments[arg_name]
            
            if not isinstance(value, str):
                return False
            
            # Check if it matches the exact pattern
            if not re.match(pattern, value):
                # Provide helpful error message if common mistake detected
                if value.startswith('<Proposal:') or value.startswith('<CodeSnippet:'):
                    print(f"[parse_action] ERROR: '{arg_name}' has extra text: '{value}'", flush=True)
                    print(f"[parse_action] Should be EXACTLY '<Proposal:xxxx>' or '<CodeSnippet:xxxx>' with NO additional text", flush=True)
                return False
    
    return True


if __name__ == "__main__":
    # Test the simplified interface
    print("=== Testing Simplified AgentLLM ===\n")
    
    # Example 1: Simple string prompt
    config = LLMConfig({"temperature": 0.7})
    llm = AgentLLM(
        llm_config=config,
        llm_name="gpt-4o",
        verbose=True
    )
    
    response = llm("What is 2+2?")
    print(f"Response: {response}\n")
    
    # Example 2: With system prompt and variables
    llm2 = AgentLLM(
        llm_config=config,
        llm_name="gpt-4o",
        system_prompt="You are a {role}. Answer in {style} style.",
        input_variables=["role", "style"],
        verbose=True
    )
    
    response2 = llm2.run({"role": "math teacher", "style": "simple"})
    print(f"Response: {response2}")

