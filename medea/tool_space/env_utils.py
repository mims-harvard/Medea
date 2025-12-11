"""
Centralized environment variable management with helpful error messages.
"""
import os
from typing import Optional


def get_env_with_error(
    var_name: str,
    default: Optional[str] = None,
    required: bool = False,
    description: str = None
) -> Optional[str]:
    """
    Get an environment variable with a helpful error message if not set.
    
    Args:
        var_name: Name of the environment variable
        default: Default value if not set
        required: Whether this variable is required
        description: Description of what this variable is used for
        
    Returns:
        The environment variable value or default
        
    Raises:
        EnvironmentError: If required=True and variable is not set
    """
    value = os.getenv(var_name)
    
    if value is None:
        if required:
            error_msg = f"\n\n❌ {var_name} environment variable is not set!\n\n"
            
            if description:
                error_msg += f"This variable is required for: {description}\n\n"
            
            error_msg += "To fix this issue:\n"
            error_msg += "1. Create a .env file in your project root directory\n"
            error_msg += f"2. Add the following line to the .env file:\n"
            error_msg += f"   {var_name}=<your-value-here>\n"
            error_msg += "3. Or set it directly in your terminal:\n"
            error_msg += f"   export {var_name}=<your-value-here>\n\n"
            
            # Add specific guidance for common variables
            if var_name == "MEDEADB_PATH":
                error_msg += "Example:\n"
                error_msg += "   MEDEADB_PATH=/path/to/your/MedeaDB\n\n"
            elif var_name == "BACKBONE_LLM":
                error_msg += "Example:\n"
                error_msg += "   BACKBONE_LLM=gpt-4o\n"
                error_msg += "   or BACKBONE_LLM=claude-3-5-sonnet-20241022\n\n"
            elif var_name in ["AZURE_OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]:
                error_msg += f"You need to obtain an API key from the respective provider.\n\n"
            elif var_name == "SEED":
                error_msg += "Example:\n"
                error_msg += "   SEED=42\n\n"
            
            error_msg += "For more details, see the README.md or examples/README.md file.\n"
            
            raise EnvironmentError(error_msg)
        
        return default
    
    return value


def get_medeadb_path() -> str:
    """
    Get the MEDEADB_PATH environment variable with proper error handling.
    
    Returns:
        str: The path to the MedeaDB directory.
        
    Raises:
        EnvironmentError: If MEDEADB_PATH is not set.
    """
    return get_env_with_error(
        "MEDEADB_PATH",
        required=True,
        description="accessing Medea database files (embeddings, datasets, models)"
    )


def get_backbone_llm(default: str = "gpt-4o") -> str:
    """
    Get the BACKBONE_LLM environment variable with proper error handling.
    
    Args:
        default: Default LLM to use if not set
        
    Returns:
        str: The LLM model name
    """
    return get_env_with_error(
        "BACKBONE_LLM",
        default=default,
        required=False,
        description="specifying the main LLM model for agents"
    )


def get_seed(default: int = 42) -> int:
    """
    Get the SEED environment variable with proper error handling.
    
    Args:
        default: Default seed value if not set
        
    Returns:
        int: The seed value
    """
    seed_str = get_env_with_error(
        "SEED",
        default=str(default),
        required=False,
        description="setting random seed for reproducibility"
    )
    
    try:
        return int(seed_str)
    except (ValueError, TypeError):
        print(f"⚠️  Warning: Invalid SEED value '{seed_str}', using default {default}")
        return default


def get_api_key(provider: str, required: bool = False) -> Optional[str]:
    """
    Get API key for a specific provider with helpful error messages.
    
    Args:
        provider: Provider name (e.g., 'AZURE_OPENAI', 'OPENROUTER', 'GEMINI', 'ANTHROPIC')
        required: Whether this API key is required
        
    Returns:
        Optional[str]: The API key if found
        
    Raises:
        EnvironmentError: If required=True and API key is not set
    """
    var_name = f"{provider}_API_KEY"
    return get_env_with_error(
        var_name,
        required=required,
        description=f"authenticating with {provider} API"
    )


def validate_environment(required_vars: list = None) -> dict:
    """
    Validate that all required environment variables are set.
    
    Args:
        required_vars: List of required variable names. If None, checks common ones.
        
    Returns:
        dict: Dictionary of variable names to values
        
    Raises:
        EnvironmentError: If any required variables are missing
    """
    if required_vars is None:
        required_vars = ["BACKBONE_LLM", "SEED"]
    
    results = {}
    missing = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            missing.append(var)
        else:
            results[var] = value
    
    if missing:
        error_msg = f"\n\n❌ Missing required environment variables: {', '.join(missing)}\n\n"
        error_msg += "To fix this issue:\n"
        error_msg += "1. Create a .env file in your project root directory\n"
        error_msg += "2. Add the following lines to the .env file:\n"
        for var in missing:
            error_msg += f"   {var}=<your-value-here>\n"
        error_msg += "\n3. Or set them in your terminal:\n"
        for var in missing:
            error_msg += f"   export {var}=<your-value-here>\n"
        error_msg += "\nFor more details, see the README.md or examples/README.md file.\n"
        
        raise EnvironmentError(error_msg)
    
    return results

