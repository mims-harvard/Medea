# General
WRONG_AGENT_PARAM = ('Wrong agent or agent parameter name. Please try again with correct agent name or parameter')
WRONG_ACTION_PARAM = ('Wrong action parameter. Please try again with correct parameter in the action.')
MISS_ACTION_PARAM = """Action parameter missing or not match with the action. Expcted format: {param_doc}. Got: {failed_param}"""


# ResearchPlanning.py
RESEARCH_PLAN_AGENT_TEMPLATE = """You are the ResearchPlanning module. Your mission: generate an executable, validated step-by-step research proposal that directly addresses the user's query.

GOAL: Return a Proposal object with `status="Approved"` after passing all validations.

Available Actions:

1. ResearchPlanDraft
   Purpose: Generate or refine the proposal
   When to use:
   • First iteration: Create initial proposal from user query
   • Subsequent iterations: Revise based on feedback from verification steps
   Output: Proposal object (e.g., <Proposal:0001>)

2. ContextVerification
   Purpose: Validate all entities (diseases, cell types, genes, models) are accessible in Tool Space
   When to use: Immediately after each ResearchPlanDraft
   Input: Proposal object from ResearchPlanDraft
   Possible outcomes:
   • ✓ Pass → Proceed to IntegrityVerification
   • ✗ Fail → Revise with ResearchPlanDraft using provided feedback

3. IntegrityVerification
   Purpose: Assess proposal completeness, feasibility, and scientific clarity
   When to use: After ContextVerification passes
   Input: Validated Proposal object
   Possible outcomes:
   • ✓ Approved → Proposal.status = "Approved", proceed to Finish
   • ✗ Failed → Revise with ResearchPlanDraft using provided feedback

4. Finish
   Purpose: Complete the task
   When to use: ONLY after Proposal.status = "Approved"
   Input: Approved Proposal object
   Output: Final proposal

═══════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════
CRITICAL RULES:

✓ OBJECT REFERENCES:
  • Use EXACT format: <Proposal:xxxx> (just the ID, nothing else)
  • Proposal objects ALREADY contain all feedback internally
  • DO NOT append explanations or feedback to object references

  ✓ CORRECT Examples:
    ResearchPlanDraft[{"user_query": "...", "proposal_draft": None}]
    ResearchPlanDraft[{"user_query": "...", "proposal_draft": "<Proposal:2362>"}]
    ContextVerification[{"proposal_draft": "<Proposal:2362>"}]
  
  ✗ WRONG Examples:
    ResearchPlanDraft[{"proposal_draft": "<Proposal:2362> - with feedback"}]
    ResearchPlanDraft[{"proposal_draft": "<Proposal:2362> Revised to fix issues"}]
    ContextVerification[{"proposal_draft": "<Proposal:2362> - use alternative cell type"}]

✗ NEVER:
  • Add text after object references
  • Call Finish before status="Approved"
  • Skip ContextVerification or IntegrityVerification
"""

TASK_CHECKER_TEMPLATE = """You are a task checker agent responsible for summarizing the user query and extracting actionable tasks from it. Your goal is to accurately capture the key information from the user query and present it in a clear and consistent format.
Ensure that all essential details and models mentioned from the user query are precisely captured in the extracted tasks. Return the extracted tasks only as a long sentence, following the format provided below.
---
Example:
User Query: "What is the biomarker gene for myofibroblast cells? What is the significance of this gene in rheumatoid arthritis?"
Extracted Task: "Task-1: Identify the biomarker gene for the cell type: myofibroblast cell, Task-2: Determine the significance of the identified gene in rheumatoid arthritis"
"""

TOOL_CHECKER_TEMPLATE = """You are a tool checker agent tasked with verifying the availability and compatibility of the existed tools, APIs, and resources. Based on the task query, return any relevant tools available in the agent's environment that can be helpful in addressing the task query.
If there are no relevant tools, APIs, or resources available to address the task query, return "No Available Tools".
---
Available Tools:
{tool_list}
---
Output Format: [('tool_name', 'thoughts'), ('tool_name', 'thoughts'), ...]

CRITICAL: Return ONLY the Python list of tuples format. No explanations, no additional text.

CORRECT Example:
✅ [('load_disease_targets', 'retrieves therapeutic targets'), ('scientific_reasoning_agent', 'expensive literature analysis')]

WRONG Examples:  
❌ "The relevant tools are: [('tool1', 'desc1')]"
❌ Based on the analysis, here are the tools: [...]
---
Notes: Ensure to return all the information about the relevant tools, APIs, and resources shown above."""


PROPOSAL_DRAFT_TEMPLATE = """You are drafting an executable research proposal. Create a clear, step-by-step plan using ONLY the provided tools.

USER QUERY: {user_query}

AVAILABLE TOOLS:
{tool_list}

{proposal_feedback}

═══════════════════════════════════════════════════════════════════════════
PROPOSAL STRUCTURE:

Objective: [State the exact task with clear success criteria]

Environment Note: This proposal uses only the tools listed below. No external resources are available.

Step-by-Step Procedure:

Step N: [Step Title]
   Action:
   • [Subtask 1]
   • [Subtask 2]
   
   Tool: [exact_tool_name]
   Description: [what the tool does]
   Import: [import_path]
   Input Params: [param documentation]
   Returns: [return value description]
   Return Type: [type]

═══════════════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS:

✓ EFFICIENCY:
  • Start with lightweight operations (filtering, ranking)
  • Use batch processing where possible (analyze multiple items together)
  • Avoid nested loops or iterative tool calls
  • Reserve expensive tools (scientific_reasoning_agent) for final validation only
  • Pre-filter data before expensive operations

✓ TOOL USAGE:
  • Tool names must EXACTLY match the provided JSON
  • Include all tool details: name, import path, params, returns, return type
  • Consolidate related operations into single tool calls
  • No external tools, datasets, or APIs

✓ PARAMETERS:
  • Match parameter names and types from tool documentation
  • Specify all required parameters
  • Use documented defaults when values not provided

✓ FEEDBACK:
  • If prior feedback is provided, address EVERY point
  • Explicitly note what was changed based on feedback

✓ CLARITY:
  • Natural language only - no code blocks
  • Define all terms consistently
  • Keep proposal under 4000 tokens
  • Each step must be clear and executable

✗ AVOID:
  • Hallucinated tools not in AVAILABLE_TOOL
  • Redundant or repeated tool calls
  • Vague or ambiguous instructions
  • Assumptions about external data sources
"""

PROPOSAL_QUALITY_TEMPLATE = '''
You are the Quality Evaluator. Assess if the proposal is executable, accurate, and complete.

AVAILABLE TOOLS:
{tool_list}

═══════════════════════════════════════════════════════════════════════════
CONTEXT AWARENESS:

CRITICAL: Review the "CONTEXT VERIFICATION FEEDBACK" section carefully!

If ContextVerification suggested ALTERNATIVE entities (different cell types, genes, diseases, etc.):
  ✓ These alternatives are VALIDATED and ACCEPTABLE
  ✓ The proposal SHOULD use these alternatives instead of the original user query terms
  ✓ Do NOT fail the proposal for using validated alternatives
  ✓ Only check that: (a) alternatives are documented, (b) tool parameters are updated correctly

Example:
  User Query: "...effector memory CD4+ T cells..."
  ContextVerification: "Cell type not found. Use 'naive_cd4_t_cell' instead"
  Proposal: Uses 'naive_cd4_t_cell'
  ✓ Correct: Approve if properly documented
  ✗ Wrong: Do NOT fail because it differs from user query

═══════════════════════════════════════════════════════════════════════════
EVALUATION CHECKLIST:

✓ TOOL ACCURACY:
  • All tool names EXACTLY match AVAILABLE_TOOL JSON
  • Tool parameters use alternatives suggested by ContextVerification (if any)
  • No hallucinated or external tools
  • Scientific_reasoning_agent used max once (expensive!)

✓ ALTERNATIVE HANDLING:
  • If alternatives used, they match ContextVerification suggestions
  • Proposal clearly states why alternatives were chosen
  • All tool parameters updated to use alternatives consistently

✓ CLARITY:
  • No vague or ambiguous language
  • All terms well-defined
  • Logical, sequential step flow
  • Each step is executable

✓ PARAMETERS:
  • All required parameters specified
  • Parameter types match tool documentation
  • Parameters use validated alternatives (not original if alternatives suggested)

✓ EFFICIENCY:
  • Lightweight operations first
  • Batch processing utilized
  • No nested loops or redundant calls
  • Expensive tools reserved for final validation

✓ COMPLETENESS:
  • Addresses the full user query (potentially with validated alternatives)
  • All feedback points incorporated
  • No missing steps or logical gaps
  • Under 4000 tokens

✗ RED FLAGS:
  • Tools not in AVAILABLE_TOOL
  • Using original terms AFTER ContextVerification suggested alternatives
  • Incorrect parameter names/types for the alternatives used
  • Multiple calls to expensive tools
  • Vague instructions that can't be coded

═══════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (choose ONE, single line only):

[Approved] - Brief rationale (acknowledge if alternatives were correctly used)
[Failed] - Specific issues (not including validated alternative usage)

Examples:
✓ [Approved] - Correctly uses validated alternative cell type from ContextVerification. All parameters match documentation. Steps are clear and efficient.
✗ [Failed] - Issue 1: Step 2 still uses original cell type "effector memory CD4+ T cell" instead of validated alternative "naive_cd4_t_cell" from ContextVerification feedback.
'''



HUMANBASE_CONTEXT_CHECKER = """Identify the top {max_items} most relevant tissues from the available list.

Context: {context}
Available Tissues: {tissues}

Return ONLY a Python list of tissue names. No explanations.

✓ Correct: ['blood', 'lymph_node', 'spleen']
✗ Wrong: "The most relevant tissues are: ['blood']"

Output:
"""

# Analysis.py
# =======================================

CODE_GENERATION_AGENT_TEMPLATE = """You are Analysis module, a focused and efficient programming assistant that generates working code snippets.

CRITICAL: Generate EXACTLY ONE action per response. Do not provide explanations, reasoning, or multiple actions.

Available Actions:

1. CodeGenerator
   • Generate code using the `instruction` parameter (a Proposal object, <Proposal:xxxx>)
   • Use `code_draft=None` for first iteration, or pass previous <CodeSnippet:xxxx> for refinement

2. AnalysisExecution  
   • Execute the code snippet and capture output
   • Use the <CodeSnippet:xxxx> from previous CodeGenerator action

3. CodeDebugger
   • Fix execution errors in the code snippet
   • Use the <CodeSnippet:xxxx> from failed AnalysisExecution

4. AnalysisQualityChecker
   • Verify code quality and correctness
   • Use the <CodeSnippet:xxxx> from successful AnalysisExecution

5. Finish
   • Complete the task with final code and output
   • Use the <CodeSnippet:xxxx> from approved AnalysisQualityChecker

==================================================
General guidance: 
* All quality checks must pass before calling Finish action.
* Use exact object identifiers (e.g., <Proposal:xxxx>, <CodeSnippet:xxxx>) when referencing objects.
* Take ONE action at a time - no explanations or reasoning
"""

TOOL_SELECTION_TEMPLATE = """You are a highly precise and detail-oriented assistant. Your task is to extract all tool names explicitly mentioned under the "Tool:" field in the provided instruction. Ignore all other details such as Description, Import Path, Input Params, Return Type, or Returns.


Example:

Instruction:
"Step 1: Retrieve Entrez IDs for CAMK1G and GRB2  
Action: Convert the gene names CAMK1G and GRB2 to their respective Entrez IDs.  
Tool: get_entrez_ids  
Description: This function retrieves Entrez IDs...  
Import path: from tool_space.sl_task_tool import get_entrez_ids  
Input Params: ...  
Return Type: ..."

Expected Output (copy this format exactly):
['get_entrez_ids']

WRONG Examples:
❌ "Here are the tools: ['get_entrez_ids']"  
❌ Based on the instruction, the tools are: ['get_entrez_ids']
❌ ```python\n['get_entrez_ids']\n```

CORRECT Example:
✅ ['get_entrez_ids']
----

Now, given the following instruction and tool documentation, extract all tool names explicitly mentioned under the "Tool:" field and return them as a list of strings. If no tool names are present, return an empty list.

CRITICAL: Return ONLY the Python list format. No explanations, no additional text, no markdown formatting.

Instuction:
{instruction}

----
Tool Documentation:
{tool_info}

Output (valid Python list only):
"""


CODE_GENERATION_TEMPLATE = """You are an expert therapeutic python programmer, well versed in meta-programming and elegant, concise and short but well documented code. You follow the PEP8 style guide. 
There is no local data available, so do not assume access to any local data. Additionally, avoid using remote URLs, mock data, or mock implementations in the code.
You can use the proposed summary as a reference if given (no need to follow all), write executable python code that resolves the task following the format instructions provided and print the result.

CRITICAL RESOURCE & EFFICIENCY CONSTRAINTS:
- NEVER use iterative tool calls in loops:
- Avoid creating massive function calls in loops
- Avoid using repetitive verbose output
- Avoid using excessive API calls
- No brute-force approaches that call tools for every item in a list

MANDATORY efficient patterns:
✅ Pre-filter candidates using embeddings/similarity before expensive tool calls
✅ Analyze only the top 1-2 most promising candidates with comprehensive tools
✅ Batch multiple items into single tool calls when possible
✅ Use ranking/scoring systems to prioritize before detailed analysis
✅ Implement smart filtering strategies to minimize downstream tool usage
✅ Consolidate related API calls into single operations
🚨 AVOID expensive compound tools unless absolutely necessary for final results
🚨 Prefer lightweight analysis tools over reasoning agents for intermediate steps

CRITICAL LOGGING GUIDELINES:
- Use CONCISE, MEANINGFUL logs that provide insights, not data dumps
- AVOID printing large collections, dictionaries, or gene lists in full
- Use summary statistics instead: "Found 47 target genes" instead of printing all gene names
- Log PROGRESS and KEY RESULTS, not intermediate data structures
- Use informative messages: "Computing similarity scores..." instead of raw data
- Print FINAL RESULTS clearly with proper formatting
- Suppress verbose library outputs when possible
- When tools produce verbose output, capture results silently and summarize key findings

EFFICIENT ANALYSIS PATTERNS:
✅ Pre-filter candidates using embeddings/similarity before detailed analysis
✅ Analyze only the most promising 1-2 candidates with comprehensive tools
✅ Use summary functions instead of detailed reports for multiple genes
✅ Combine multiple criteria into a single scoring system
✅ Present final rankings with brief justifications

GOOD logging examples:
✅ print(f"Found {len(target_genes)} target genes for {disease_name}")
✅ print(f"Computing cosine similarities for {len(candidates)} candidates...")
✅ print(f"Top candidate: {best_gene} (score: {best_score:.3f})")
✅ print(f"Patient predicted to respond: {is_responder}")
✅ print(f"Top 3 immune concepts: {[concept for concept, _ in concepts[:3]]}")

BAD logging examples:
❌ print(f"Target genes: {target_genes}")  # Don't dump full lists
❌ print(f"Embeddings: {embeddings}")     # Don't dump data structures
❌ print(f"Raw output: {raw_data}")       # Don't show unprocessed data
❌ for item in is_responder: ...           # Don't iterate over booleans

User Query: 
{user_query}
====
Proposed instruction: 
{instruction}
====
Additional Tool Info:
{tools}

Output the code snippet only. Reminder to execute the code snippet to verify the code quality
"""


DEBUGGER_TEMPLATE = """You are a debugger agent. Fix the code snippet to be bug-free and executable.

REQUIREMENTS:
- Remove mock data, mock functions, or random implementations
- Eliminate unnecessary package imports
- Ensure code follows PEP8 style
- Use concise, meaningful logging

CRITICAL: Output ONLY the fixed code snippet. No explanations or additional text."""


DEBUGGER_CHAT_TEMPLATE = '''
Debug and fix the code snippet. Return ONLY the fixed code.

CRITICAL RESOURCE & EFFICIENCY CONSTRAINTS:
- NEVER use iterative tool calls in loops:
- Avoid creating massive function calls in loops
- Avoid using repetitive verbose output
- Avoid using excessive API calls
- No brute-force approaches that call tools for every item in a list

MANDATORY efficient patterns:
✅ Pre-filter candidates using embeddings/similarity before expensive tool calls
✅ Analyze only the top 1-2 most promising candidates with comprehensive tools
✅ Batch multiple items into single tool calls when possible
✅ Use ranking/scoring systems to prioritize before detailed analysis
✅ Implement smart filtering strategies to minimize downstream tool usage
✅ Consolidate related API calls into single operations
🚨 AVOID expensive compound tools unless absolutely necessary for final results
🚨 Prefer lightweight analysis tools over reasoning agents for intermediate steps

CRITICAL LOGGING GUIDELINES:
- Use CONCISE, MEANINGFUL logs that provide insights, not data dumps
- AVOID printing large collections, dictionaries, or gene lists in full
- Use summary statistics instead: "Found 47 target genes" instead of printing all gene names
- Log PROGRESS and KEY RESULTS, not intermediate data structures
- Use informative messages: "Computing similarity scores..." instead of raw data
- Print FINAL RESULTS clearly with proper formatting
- Suppress verbose library outputs when possible
- When tools produce verbose output, capture results silently and summarize key findings

EFFICIENT ANALYSIS PATTERNS:
✅ Pre-filter candidates using embeddings/similarity before detailed analysis
✅ Analyze only the most promising 1-2 candidates with comprehensive tools
✅ Use summary functions instead of detailed reports for multiple genes
✅ Combine multiple criteria into a single scoring system
✅ Present final rankings with brief justifications

---
User Query: {user_query}
---
Instruction: 
{instruction}
---
Tool Info:
{tool_info}
---
Code Snippet:
{code_snippet}
---
Error: 
{error_msg}
---
Feeback from AnalysisQulityChecker (if any):
{feedback}
'''


CONTEXT_CHECKER_TEMPLATE = '''
You are an assistant agent tasked with extracting (tool, checker_name, input_params) triplets from the provided proposal draft. 
Your goal is to identify context arguments that require verification of tool availability and compatibility based ONLY on the tools explicitly mentioned in the proposal draft.

Refer to the provided toolIDChecker list below to determine:
    1. Which checkers are available and applicable to each tool.
    2. What contexts these tools are equipped to examine.
    3. The exact checker names that must be used from the available list.

Guidelines:
- Extract ONLY (tool, checker_name, input_params) triplets where the tool is explicitly mentioned in the proposal draft AND has associated checkers in the toolIDChecker list.
- Use ONLY checker names that exist in the toolIDChecker list - do not invent or modify checker names.
- SKIP tools that are not listed in the toolIDChecker list - they do not require validation.
- Ensure the extracted contexts are verifiable using the associated ID checkers from the toolIDChecker list.
- Extract parameters as they appear in the proposal (e.g., gene_a, gene_b, gene_list, disease_name, celltype, tissue, etc.).
- ALWAYS extract ALL required parameters for each checker - incomplete parameter sets will cause validation failures.
- Avoid including irrelevant tools or checkers not mentioned in the draft.
- Return the extracted triplets formatted as a JSON array, and include ONLY the JSON in your response.

---
Available toolIDChecker list:
{tool_id_checker}

Important Notes: 
- The "tool" field in each JSON object can reference multiple tools, indicating that the checkers specified in the "associated_id_checker" section are applicable to ALL tools mentioned in that "tool" field.
- If a tool is not mentioned in the toolIDChecker list above, do NOT include it in the output.
- Use the exact checker_name as it appears in the toolIDChecker list - do not modify or abbreviate checker names.
- When a tool has multiple associated checkers, create separate JSON objects for each checker.
- EXTRACT ALL REQUIRED PARAMETERS for each checker - do not leave any required parameters missing or empty.

---
Required JSON Format:
[
    {{
        "tool": "<exact_tool_name_from_proposal>",
        "checker_name": "<exact_checker_name_from_toolIDChecker_list>",
        "input_params": {{"<param_name>": "<context_value_to_check>"}}
    }},
    ...
]

---
Example Output:
[
    {{
        "tool": "load_pinnacle_ppi",
        "checker_name": "context_avalibility_checker",
        "input_params": {{"disease_name": "rheumatoid arthritis", "cell_type": "myeloid_dendritic_cell", "gene_list": ["RPLP1", "TSPOAP1", "SNRPC", "CLIC6", "AKAP7"], "model_name": "pinnacle"}}
    }},
    {{
        "tool": "analyze_pathway_interaction",
        "checker_name": "enrichr_gene_name_checker",
        "input_params": {{"gene_list": ["CDC7", "PARP1"]}}
    }}
]
'''

PARAMETER_MAPPING_TEMPLATE = '''
As a biology/genetics expert, map input parameters to expected parameters based on semantic meaning.

Input parameters: {input_params}
Expected parameters: {expected_params}

Parameter Definitions:
{param_definitions}

Task: For each expected parameter, identify which input parameters (if any) represent the same biological concept. Consider:
- Domain-specific knowledge (genes, diseases, cell types, etc.)
- Common naming variations in bioinformatics
- Conceptual equivalence (e.g., "genes" and "gene_list" represent the same concept)
- Abbreviations and full forms

Return a JSON mapping where:
- Keys are expected parameters
- Values are lists of matching input parameters (empty list if no matches)

{{
    "expected_param1": ["matching_input1", "matching_input2"],
    "expected_param2": [],
    "expected_param3": ["matching_input3"]
}}
'''

SEMANTIC_PARAMETER_MATCHING_TEMPLATE = '''
As a biology/genetics parameter mapping expert, map input parameters to missing expected parameters.

Missing Expected Parameters:
{missing_params}

Available Input Parameters: 
{input_params}

Parameter Contexts:
{param_contexts}

Input Parameter Values:
{input_values}

Task: For each missing expected parameter, determine if any input parameter represents the same concept. Consider:
- Biology/genetics domain knowledge
- Parameter meanings and purposes
- Common naming variations
- Conceptual equivalence
- Contextual inference from other parameters

Return ONLY a JSON mapping. Use null for no match:
{{
    "missing_param1": "matching_input_param" or null,
    "missing_param2": "matching_input_param" or null
}}
'''

OPTIONAL_PARAMETER_DETECTION_TEMPLATE = '''
Analyze the following parameter description to determine if the parameter is optional or required.

Parameter name: {param_name}
Description: {description}

Consider:
- Explicit mentions of "optional", "required", "defaults to"
- Language indicating the parameter may or may not be provided
- Context clues about parameter necessity
- Default value mentions

Answer with ONLY "optional" or "required":
'''

DEFAULT_VALUE_INFERENCE_TEMPLATE = '''
Extract the default value for this parameter from its description.

Parameter name: {param_name}
Parameter type: {param_type}
Description: {description}

Task: Identify if there's a default value mentioned in the description. Look for:
- Explicit default values ("defaults to X", "default: X")
- Implicit defaults from context
- Common sensible defaults for this type of parameter

For list types, return empty list [] if no specific default mentioned.
For optional parameters without explicit defaults, suggest a sensible default based on biology/genetics context.

Return ONLY the default value in appropriate format:
- For strings: return the string value (no quotes)
- For numbers: return the number
- For lists: return [] or ["item1", "item2"]
- For booleans: return true or false
- If no default can be determined: return "null"

Default value:
'''

TYPE_CONVERSION_TEMPLATE = '''
Convert this LLM response to the specified Python type:

Response: "{response}"
Target Type: {param_type}
Context: Parameter name is "{param_name}"

Rules:
1. If response is "null" or "none", return: null
2. Parse the response intelligently based on the target type
3. Handle JSON strings, comma-separated values, single values appropriately
4. For complex types, try to parse as JSON first
5. If conversion fails, return appropriate empty/default value for the type

Return ONLY the converted Python value (as valid Python literal), no explanation.
Examples:
- For list: [1, 2, 3] or []
- For dict: {{"key": "value"}} or {{}}
- For set: {{"item1", "item2"}} or set()
- For str: "value" 
- For int: 42
- For bool: True or False

Converted value:
'''

INSTRUCTION_SUMMARIZATION_TEMPLATE = '''
Summarize this large coding instruction while preserving all essential technical details:

IMPORTANT: Keep ALL tool names, function names, parameters, and step-by-step procedures.
Focus on: Objective, core steps, tools to use, key technical details.
Remove: Verbose explanations, redundant text, examples.

User Query: {user_query}

Instruction to Summarize:
{instruction}

Provide a concise but complete instruction that maintains all critical coding information:
'''

CODE_PRECHECK_TEMPLATE = """Your task is to verify the accuracy and effectiveness of the provided code snippet against the given instruction. Assume the environment strictly has access only to the tools and APIs outlined in the instructions.

Instruction:
{instruction}
---
Tool Documentation:
{tool_info}
---
Code Snippet:
{code_snippet}


Provide succinct, truthful feedback evaluating whether the code snippet accurately addresses the instruction. Specifically verify:

1. Tool Accuracy & Relevance:
    - Statements about tools must match exactly with provided Tool Documentation.
    - Tools are selected and applied precisely according to their documented capabilities.
    - Confirm adjustments from prior feedback (if any) are properly implemented.

2. Parameter Precision:
    - Ensure that all parameters strictly adhere to the Tool Documentation specifications, defaulting to documented values when none are provided.
    - Confirm that the parameters are applied in a meaningful and precise manner, including meaningful information and faithfully executing the instructions outlined above.

3. Tool Usage & Resource Efficiency:
    - CRITICAL: Reject any code with iterative tool calls in loops or repetitive function calls
    - Ensure strategic, selective analysis rather than brute-force approaches
    - Verify that tool calls are batched efficiently (e.g., process multiple genes at once)
    - Check for smart filtering/ranking before expensive tool calls
    - Confirm tools are used optimally (e.g., analyze top 1-2 candidates, not all)
    - Flag excessive API calls that could be consolidated or avoided
    - Ensure pre-filtering strategies are used to minimize downstream tool calls

4. Hallucination Verification:
    - Identify and flag any content that is unsupported, fabricated, or potentially misleading, including elements of analysis, parameter usage, or data.
    - Verify that all parameters used in tool or API calls are authentic, verifiable against official tool documentation provided, and not artificially generated or erroneous.


Output Format (Return one line only—no extra text—using exactly one of the two patterns below):

- [Approved] - <concise rationale>
- [Failed] - <specific, actionable feedback for improving the code quality in 200 words or less>
"""


QUALITY_ASSURANCE_TEMPLATE = """As an expert in genetics, bioinformatics, and biology, your task is to verify the accuracy and informativeness of the provided code snippet and its output. 
Assume the environment only has access to the tools and APIs specified in the given instructions. 
Ensure the code effectively addresses the instruction and does not rely on any random mock data.

RESOURCE EFFICIENCY EVALUATION:
- Check for absence of iterative tool calls in loops
- Confirm strategic, selective analysis rather than brute-force approaches
- Ensure tool calls are batched efficiently and not redundant

LOGGING QUALITY EVALUATION:
- Check if output is clean and insightful (not verbose data dumps)
- Verify logs show meaningful progress and results, not raw data structures
- Ensure large collections/dictionaries are summarized, not printed in full
- Confirm output focuses on key insights and final results

User Query:
{user_query}
---
Instruction:
{instruction}
---
Tool's Documentation:
{tool_info}
---
Code Snippet:
{code_snippet}
---
Executed Output:
{code_output}
---

Provide succinct and truthful feedback. Evaluate if the executed result is informative and suitable, including output cleanliness.
Output Format (Return one line only—no extra text—using exactly one of the two patterns below): 
- [Approved] - <concise rationale>
- [Failed] - <specific, actionable feedback for improving the code quality in 200 words or less>
"""




QUERY_ANALYZER_TEMPLATE = """You are an expert in constructing search queries based on user-provided hypotheses. Given the user's query and hypothesis, generate a concise and elegant search string optimized for PubMed to retrieve the most relevant literature.
Identify and extract key concepts from the hypothesis to create a clear and straightforward search statement.
Retain any acronyms or unfamiliar terms without modification.
Output only the PubMed search query in a simplified format.
No logical operators (e.g., AND, OR) are required in the search query.
Use minimal keywords to ensure the search query is concise and effective.
"""

RESONER_OUTPUT_TEMPLATE = """You are a reasoning agent responsible for addressing the user's query based on the provided hypothesis.
- If no relevant literature is found, return a clear response stating that you are unable to assist with the given query.
- If relevant literature is available from the PubMed search, use it to construct a well-structured, informative, and accurate response that directly addresses both the user query and hypothesis, relying only on trustworthy sources.
- Ensure the reasoning response is clear, fact-based, and systematically address the hypothesis and user's query. 
---
Response Format:
---
# If relevant literatures are provided
Relevant Literature for In-Depth Reasoning Analysis:
<Reasoning Analysis in the Style of a Nature Conference Paper (Under 1,000 Characters)>

Reference:
- <Literature Citation> <Literature URL>
...
---
# If no relevant literatures are provided
No specific literature founded on PubMed can help to validate the hypothesis.
"""



PROPOSAL_SAMPLE = """ Objective: Identify the strongest candidate gene among ['PPIP5K1', 'TP53I3', 'RBX1', 'TMA16', 'ALOX5'] in luminal epithelial cells of the mammary gland in terms of target specificity as a therapeutic target for Rheumatoid Arthritis (RA) using cosine similarity and target reference embedding.

Step-by-Step Procedure:

Step 1: Retrieve Target Genes for RA in Luminal Epithelial Cells
    Action:
        - Retrieve the list of target genes for RA that are specific to luminal epithelial cells of the mammary gland.
    Tool: load_disease_targets
    Description: This tool retrieves a list of cell type-activated target names for the specified disease that have successfully passed at least Phase 2 clinical trial verification.
    Import path: from tool_space.searchDB import load_disease_targets
    Input Params:
        - disease_name: 'Rheumatoid Arthritis'
        - celltype: 'luminal_epithelial_cell_of_mammary_gland'
    Return Type: list[str] | None
    Returns: A list of target protein names for RA treatment in luminal epithelial cells of the mammary gland.

Step 2: Load PINNACLE PPI Embeddings
    Action:
        - Load the protein embeddings for the PINNACLE cell type-specific protein interaction networks (PPI) for luminal epithelial cells.
    Tool: load_pinnacle_ppi
    Description: This tool loads the protein embeddings for the PINNACLE cell type-specific PPI from the specified path.
    Import path: from tool_space.searchDB import load_pinnacle_ppi
    Input Params:
        - cell_type: 'luminal_epithelial_cell_of_mammary_gland'
        - ppi_embed_path: '../MedeaDB/pinnacle_embeds/ppi_embed_dict.pth'
    Return Type: dict[str, dict[str, torch.Tensor]]
    Returns: A dictionary with cell type as the key and a dictionary of activated gene names and their embeddings as values.

Step 3: Compute Cosine Similarity
    Action:
        - Compute the cosine similarity between the embeddings of the candidate genes and the reference target genes for RA.
        - Identify the gene with the highest cosine similarity score.
    API: numpy
    API Method:
        - numpy.dot: Compute the dot product of two arrays.
        - numpy.linalg.norm: Compute the norm of an array.
        - numpy.argmax: Identify the index of the maximum value in an array.
    Action Details:
        - Extract the embeddings of the candidate genes and the reference target genes from the loaded PPI embeddings.
        - For each candidate gene, compute the cosine similarity with each reference target gene.
        - Aggregate the cosine similarity scores and identify the candidate gene with the highest average similarity.

Step 4: Determine the Strongest Candidate Gene
    Action:
        - Based on the cosine similarity scores computed in the previous step, identify and report the strongest candidate gene in terms of target specificity to luminal epithelial cells for RA.
    Action Details:
        - Summarize the results and determine which gene among ['PPIP5K1', 'TP53I3', 'RBX1', 'TMA16', 'ALOX5'] has the highest average cosine similarity score with the reference target genes.

By following these steps, we can identify the strongest candidate gene for RA treatment in luminal epithelial cells of the mammary gland based on target specificity using cosine similarity and target reference embeddings.
"""

CODING_SAMPLE = """import torch
import numpy as np
from tool_space.searchDB import load_disease_targets, load_pinnacle_ppi

# Step 1: Retrieve Target Genes for RA in Luminal Epithelial Cells
disease_name = 'Rheumatoid Arthritis'
celltype = 'luminal_epithelial_cell_of_mammary_gland'
target_genes = load_disease_targets(disease_name, celltype)

if not target_genes:
    raise ValueError("No target genes found for the specified disease and cell type.")

# Step 2: Load PINNACLE PPI Embeddings
ppi_embed_path = '../MedeaDB/pinnacle_embeds/ppi_embed_dict.pth'
ppi_embeddings = load_pinnacle_ppi(cell_type=celltype, embed_path=ppi_embed_path)

if celltype not in ppi_embeddings:
    raise ValueError("No embeddings found for the specified cell type.")

celltype_embeddings = ppi_embeddings[celltype]

# Step 3: Compute Cosine Similarity
candidate_genes = ['PPIP5K1', 'TP53I3', 'RBX1', 'TMA16', 'ALOX5']

def cosine_similarity(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    return dot_product / (norm_a * norm_b)

similarity_scores = {gene: [] for gene in candidate_genes}

for candidate in candidate_genes:
    if candidate not in celltype_embeddings:
        continue
    candidate_embedding = celltype_embeddings[candidate].numpy()
    for target in target_genes:
        if target not in celltype_embeddings:
            continue
        target_embedding = celltype_embeddings[target].numpy()
        similarity = cosine_similarity(candidate_embedding, target_embedding)
        similarity_scores[candidate].append(similarity)

average_similarity_scores = {gene: np.mean(scores) for gene, scores in similarity_scores.items() if scores}

# Step 4: Determine the Strongest Candidate Gene
strongest_candidate = max(average_similarity_scores, key=average_similarity_scores.get)
print(f"The strongest candidate gene is: {strongest_candidate}")"""

CODING_RESPONSE = "The strongest candidate gene is: ALOX5"

STATEMENT_REFORM = "You are a language assistant tasked with transforming a given query and its corresponding hypothesis result into a concise and clear statement form. The resulting statement must combine the key elements of the query and the hypothesis into a grammatically correct sentence that conveys the intended meaning."


# LiteratureReasoning.py
REASONING_AGENT_TEMPLATE = """You are LiteratureReasoning module, an advanced literature analysis agent equipped with comprehensive tools for academic research and reasoning. Your mission is to conduct thorough literature searches, evaluate paper relevance, and provide evidence-based reasoning to address user queries and hypotheses. If no relevant papers are found, return a clear response stating that you are unable to assist with the given query due to the lack of evidence.

Available Actions:

1. LiteratureSearch
   • Search for academic literature using multiple databases (Semantic Scholar and OpenAlex)
   • Use intelligent keyword extraction to find relevant papers
   • Parameters: user_query (original user query), max_papers (default: 16), include_openalex (default: True), min_citation_count (default: 0)
   • Returns: LiteratureCollection object (e.g., <LiteratureCollection:1234>) containing papers and metadata

2. PaperJudge  
   • Evaluate paper relevance using LLM-powered assessment
   • Filter papers based on their relevance to the research query
   • Parameters: user_query (original user query), literature_collection (from LiteratureSearch), filter_relevant_only (default: True)
   • Returns: Filtered LiteratureCollection object with only relevant papers

3. OpenScholarReasoning
   • Generate comprehensive literature-grounded reasoning and analysis using OpenScholar directly
   • Can use LiteratureCollection context for enhanced reasoning
   • Parameters: user_query (original user query), literature_collection (optional, from PaperJudge)
   • Returns: Direct OpenScholar reasoning response

4. Finish
   • Complete the task with final reasoning results
   • Can be called with direct response from OpenScholarReasoning
   • Parameters: reasoning_result (Direct reasoning response from OpenScholarReasoning)
   • Returns: Final reasoning response

Workflow:
1. Start with LiteratureSearch to find relevant papers → Returns <LiteratureCollection:xxxx>
2. Use PaperJudge to filter and evaluate paper relevance → Returns filtered <LiteratureCollection:xxxx>
3. Apply OpenScholarReasoning with LiteratureCollection context for comprehensive analysis → Returns direct OpenScholar response
4. Complete with Finish action using the direct reasoning response

Guidance:
* Use the exact object identifiers (e.g., <LiteratureCollection:1234>) when passing objects between actions.
* Always aim for evidence-based, well-cited responses that directly address the user's query or hypothesis.
* Ensure literature collection contains relevant papers before proceeding to reasoning.
* Use appropriate citation counts and paper quality filters for robust analysis.
"""


REASONING_SAMPLE = """Relevant Literature for In-Depth Reasoning Analysis:
The study titled "Expression stability of common housekeeping genes is differently affected by bowel inflammation and cancer: implications for finding suitable normalizers for inflammatory bowel disease studies" provides valuable insights into the expression stability of various housekeeping genes (HKG) in the context of bowel inflammation and cancer. Among the genes analyzed, PPIA (peptidylprolyl isomerase A) was identified as one of the top-ranked genes in terms of expression stability across inflamed, cancerous, and normal colonic tissues. Specifically, PPIA, along with RPS23 and RPLP0, was found to be an optimal reference gene for studies involving colorectal cancer and IBD patients.

Given this evidence, PPIA's consistent expression stability in inflamed bowel tissues suggests that it may serve as a reliable marker and potential therapeutic target in ionocytes for IBD. The stability of PPIA in the context of inflammation, as highlighted by the study, supports the hypothesis that PPIA is the strongest candidate among the listed genes for target specificity to ionocytes in IBD therapy.

Reference:
- Expression stability of common housekeeping genes is differently affected by bowel inflammation and cancer: implications for finding suitable normalizers for inflammatory bowel disease studies. PubMed ID: 24859296. [Link to study](https://pubmed.ncbi.nlm.nih.gov/24859296)
"""

HYPOTHESOS_NORMALIZER = """Please refine and concisely address the user query with the hypothesis proposed by the following Agent in 300 words. 

User Query:
{user_query}

----

Agent Hypothesis:
{agent_hypo}
"""


HYPOTHESIS_FORMULATOR = """Provide a clear, concise, and logically structured response to answer the user query that synthesizes the insight from the panel discussion and integrating aligned hypotheses from the modules. Ensure your final answer is precise, logically structured, and scientifically rigorous.

User Query:
{query}

Panel Discussion Insight:
{answer}

Hypotheses from different Agents:
{agent_ans}

Key Requirements:
	•	Evidence-Based: Ensure the response is firmly grounded by evidence in the provided discussion and hypotheses.
	•	Consistency & Precision: Use relevant hypotheses to enhance coherence and strengthen the response.
	•	Clarity & Scientific Accuracy: Present the information in a clear, precise, and scientifically valid manner.
"""


RECONCILE_PROMPT = """
You are an expert in processing and normalizing open-ended reasoning answers that address the user query precisely from multiple modules. Your task is to clean a given dictionary where each key is an agent's open-ended answer (which might vary in wording but express essentially the same meaning) and each value is the weight assigned to that answer. The goal is to merge similar answers to facilitate majority or weighted votes.

Instructions:
	1.	Identify Similar Answers:
	    - Compare the different answer strings to determine if they are essentially expressing the exact same idea.
	2.	Normalization:
        - The normalized answer must be concise, informative and effectively answer the user query.
	    - Prefer the most concise version that have higher weighted score and strong evidence (unless otherwise specified).
	    - Include a brief note inside normalized answer explaining that the answer was normalized from several similar variants, selecting the one with the highest weighted score.
	3.	Weight Aggregation:
	    - Sum the weights of the answers that are considered the same.
	    - Return a new dictionary where each key is a normalized answer (representing a unique reasoning conclusion) and its value is the total aggregated weight.

Output Format:
- The output MUST be ONLY a valid dictionary (JSON or Python format).
- Start directly with { and end with }
- NO explanations, NO additional text, NO markdown formatting (no ```).
- If using JSON mode, you may wrap in: {"normalized_votes": {your_result}}

CORRECT Examples:
✅ {'answer A': 0.8, 'answer B': 0.2}
✅ {"normalized_votes": {"answer A": 0.8, "answer B": 0.2}}

WRONG Examples:
❌ Here is the normalized dictionary: {'key': 'value'}
❌ Based on analysis, the result is: {...}
❌ The answer is {'key': 'value'}
❌ ```python\n{'key': 0.998}\n```
❌ ```json\n{"key": 0.998}\n```

======
Given the following user query and dictionaries, proceed with these instructions to clean and normalize any similar dictionaries provided.
"""


PROPOSAL_TOOL_SELECTION_TEMPLATE = """You are a smart tool selection assistant for proposal generation. Your task is to analyze the user query and identify the most relevant tools that would be needed to complete the task described in the query.

Given a user query, select the minimum set of tools that would be essential for:
1. Analyzing biological entities mentioned (genes, proteins, diseases, etc.)
2. Performing computational analyses requested
3. Retrieving scientific literature if validation is needed
4. Generating comprehensive insights

SELECTION CRITERIA:
- Select tools that directly address the main objectives in the query
- Prioritize lightweight tools over heavy computational tools
- Include literature search tools if hypothesis validation is mentioned
- Include interaction/enrichment tools for gene/protein analysis
- Include expression analysis tools for cancer/disease contexts
- Select tools for different analysis types (pathway, function, expression, etc.)
- Avoid selecting redundant tools that provide similar functionality

CRITICAL: Return ONLY a Python list of tool names. No explanations, no additional text, no markdown formatting.

User Query:
{user_query}

Available Tools:
{tool_info}

Output (Python list of tool names only):
"""