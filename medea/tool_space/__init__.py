"""
Medea Tool Space

This package contains all the tools available in Medea for biological research tasks.
Tools are registered with ToolUniverse framework following the BaseTool pattern.

Import this module to register tools with ToolUniverse:
    from medea.tool_space import tooluniverse_tools
"""

# Import ToolUniverse-compatible tools (this triggers @register_tool decorators)
try:
    from . import tooluniverse_tools
    from .tooluniverse_tools import get_registered_tools, list_medea_tools
    
    __all__ = [
        'tooluniverse_tools',
        'get_registered_tools',
        'list_medea_tools'
    ]
except ImportError as e:
    # If ToolUniverse is not installed, provide informative message
    print(f"[Warning] ToolUniverse not available: {e}")
    print("[Warning] Install tooluniverse to use Medea tools: pip install tooluniverse")
    
    __all__ = []

