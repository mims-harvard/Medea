"""
Example: Using Medea Tools with ToolUniverse

This example demonstrates the proper way to use Medea's tools in global tool space
with ToolUniverse framework.
"""

from tooluniverse.tool_registry import get_tool_registry


def main():
    print("="*80)
    print("Medea ToolUniverse Integration")
    print("Following the Official ToolUniverse Pattern")
    print("="*80)
    
    # Step 1: Import Medea tools (this triggers @register_tool decorators)
    print("\n[Step 1] Importing Medea tools...")
    from medea.tool_space import tooluniverse_tools
    print("✓ Medea tools imported and registered with ToolUniverse")
    
    # Step 2: Get all registered tools from ToolUniverse
    print("\n[Step 2] Getting registered tools from ToolUniverse...")
    all_tool_classes = get_tool_registry()
    medea_tool_names = tooluniverse_tools.list_medea_tools()
    print(f"✓ Found {len(medea_tool_names)} Medea tools in ToolUniverse registry")
    print(f"   Sample tools: {medea_tool_names[:5]}")
    
    # Step 3: Use a tool via ToolUniverse
    print("\n[Step 3] Using a Medea tool via ToolUniverse...")
    
    # Get the tool class from ToolUniverse registry
    tool_name = 'load_disease_targets'
    if tool_name in all_tool_classes:
        tool_class = all_tool_classes[tool_name]
        tool = tool_class({})  # Instantiate with empty config
        print(f"✓ Retrieved tool: {tool_name}")
        print(f"   Tool class: {tool.__class__.__name__}")
        
        # Use the tool
        print(f"\n[Step 4] Executing tool: {tool_name}")
        try:
            result = tool.run({
                'disease_name': 'rheumatoid arthritis',
                'use_api': True,
                'attributes': ['otGeneticsPortal', 'chembl']
            })
            
            if result.get('success'):
                print(f"✓ Tool executed successfully!")
                print(f"   Disease: {result.get('disease')}")
                print(f"   Targets found: {result.get('count')}")
                print(f"   Sample targets: {result.get('targets', [])[:5]}")
            else:
                print(f"✗ Tool execution failed: {result.get('error')}")
        except Exception as e:
            print(f"✗ Tool execution error: {e}")
    else:
        print(f"✗ Tool '{tool_name}' not found in registry")
    
    # Step 5: List all available Medea tools
    print("\n[Step 5] All Available Medea Tools:")
    print("-" * 80)
    for i, name in enumerate(medea_tool_names, 1):
        if name in all_tool_classes:
            tool_class = all_tool_classes[name]
            doc = tool_class.__doc__ or "No description"
            print(f"{i:2d}. {name:40s} - {doc.strip()}")
    
    # Step 6: Demonstrate input validation
    print("\n[Step 6] Testing input validation...")
    try:
        tool_class = all_tool_classes.get('load_disease_targets')
        if tool_class:
            tool = tool_class({})  # Instantiate with empty config
            # This should raise a validation error
            tool.validate_input()  # Missing required parameter
            print("✗ Validation should have failed")
    except ValueError as e:
        print(f"✓ Validation working correctly: {e}")
    
    print("\n" + "="*80)
    print("Integration Complete!")
    print("="*80)
    print("\nKey Features:")
    print("  ✓ Tools follow BaseTool pattern")
    print("  ✓ Registered with @register_tool decorator")
    print("  ✓ Configuration in separate JSON file")
    print("  ✓ Input validation included")
    print("  ✓ Original tool functions unchanged")
    print("\nNext Steps:")
    print("  1. Use tools via ToolUniverse's tool management")
    print("  2. Integrate with LLM agents")
    print("  3. Add custom tools following the same pattern")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease install ToolUniverse:")
        print("  pip install tooluniverse")
        print("  # or")
        print("  uv pip install tooluniverse")

