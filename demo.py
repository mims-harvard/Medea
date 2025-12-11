from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import os
import sys
import signal
import threading
import queue
from pathlib import Path
from concurrent import futures
import time

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from agentlite.commons import TaskPackage
from medea.agents.experiment_analysis import Analysis, CodeGenerator, AnalysisExecution, CodeDebug, CodeQulityChecker
from medea.agents.research_planning import ResearchPlanning, ResearchPlanDraft, ContextVerification, IntegrityVerification
from medea.agents.literature_reasoning import LiteratureReasoning, OpenScholarReasoning, LiteratureSearch, PaperJudge
from medea.agents.discussion import multi_round_discussion
from medea.agents.agent_llms import LLMConfig, AgentLLM
from medea.agents.utils import Proposal, FlushAgentLogger
from medea.tool_space.gpt_utils import chat_completion

# Global flag for graceful shutdown
shutdown_flag = threading.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    shutdown_flag.set()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Initialize FastAPI app
app = FastAPI(title="Medea API", description="Multi-agent system for therapeutic discovery")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    userID: Optional[str] = None
    teamID: Optional[str] = None
    template: str = "medea"
    model: str = "medea"
    config: Dict[str, Any] = {}

class MedeaResponse(BaseModel):
    commentary: str
    template: str
    title: str
    description: str
    additional_dependencies: List[str]
    has_additional_dependencies: bool
    install_dependencies_command: str
    port: Optional[int]
    file_path: str
    code: str
    # New fields for streamlined output
    agent_outputs: Dict[str, Any]
    has_code_execution: bool
    code_snippet: Optional[str]
    execution_result: Optional[str]

# Global agent instances
research_planning_agent = None
analysis_module = None
literature_reasoning_agent = None

# Agent configuration parameters
research_planning_tmp = 0.4
analysis_tmp = 0.4
literature_tmp = 0.4
QUALITY_MAX_ITER = 2
CODE_QUALITY_MAX_ITER = 2
DEBATE_ROUND = 2
PANELIST_LLM = [os.getenv('BACKBONE_LLM', 'gpt-4o'), 'gemini-2.5-flash', 'o3-mini-0131']
INCLUDE_BACKBONE_LLM = True
VOTE_MERGE = True

def initialize_medea_agents():
    """Initialize the Medea agents with the workflow approach"""
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv('.env')
        
        print("=== Init Research Planning Agent Backbone ===")
        research_llm_config = LLMConfig({"temperature": research_planning_tmp})
        research_llm = AgentLLM(
            llm_config=research_llm_config,
            llm_name=os.getenv("BACKBONE_LLM", "gpt-4o"),
        )
        
        print("=== Init Analysis Agent Backbone ===")
        analysis_llm_config = LLMConfig({"temperature": analysis_tmp})
        analysis_llm = AgentLLM(
            llm_config=analysis_llm_config,
            llm_name=os.getenv("BACKBONE_LLM", "gpt-4o"),
        )
        
        print("=== Init Literature Reasoning Agent Backbone ===")
        literature_llm_config = LLMConfig({"temperature": literature_tmp})
        literature_llm = AgentLLM(
            llm_config=literature_llm_config,
            llm_name=os.getenv("BACKBONE_LLM", "gpt-4o"),
        )
        
        print("=== Init Research Planning Actions ===")
        research_actions = [
            ResearchPlanDraft(llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o")),
            ContextVerification(llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o")),
            IntegrityVerification(llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o"), max_iter=QUALITY_MAX_ITER),
        ]
        
        print("=== Init Analysis Actions ===")
        analysis_actions = [
            CodeGenerator(llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o")), 
            AnalysisExecution(),
            CodeDebug(llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o")), 
            CodeQulityChecker(llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o"), max_iter=CODE_QUALITY_MAX_ITER), 
        ]
        
        print("=== Init Literature Reasoning Actions ===")
        literature_actions = [
            LiteratureSearch(model_name=os.getenv("PAPER_JUDGE_LLM", "gpt-4o"), verbose=False),
            PaperJudge(model_name=os.getenv("PAPER_JUDGE_LLM", "gpt-4o"), verbose=True),
            OpenScholarReasoning(llm_provider=os.getenv("BACKBONE_LLM", "gpt-4o"), verbose=True)
        ]
        
        # Initialize agents with enhanced loggers
        research_planning_agent = ResearchPlanning(
            llm=research_llm, 
            actions=research_actions,
            logger=FlushAgentLogger(FLAG_PRINT=True, PROMPT_DEBUG_FLAG=False)
        )
        analysis_module = Analysis(
            llm=analysis_llm, 
            actions=analysis_actions,
            logger=FlushAgentLogger(FLAG_PRINT=True, PROMPT_DEBUG_FLAG=False)
        )
        literature_reasoning_agent = LiteratureReasoning(
            llm=literature_llm, 
            actions=literature_actions,
            logger=FlushAgentLogger(FLAG_PRINT=True, PROMPT_DEBUG_FLAG=False)
        )
        
        return research_planning_agent, analysis_module, literature_reasoning_agent
        
    except Exception as e:
        print(f"Error initializing Medea agents: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def research_and_analysis_module(query: str, research_planning_module, analysis_module):
    """Execute research planning and analysis modules"""
    # Create a task package for the research planning agent
    research_task_dict = {"user_query": query}
    research_taskpack = TaskPackage(instruction=str(research_task_dict))
    try: 
        research_response = research_planning_module(research_taskpack)
    except Exception as e:
        print(f"Research planning agent call failed: {e}", flush=True)
        return "None", "None"

    # Extract the proposal if valid, otherwise return the response directly
    analysis_response, proposal_text = "None", "None"
    if isinstance(research_response, dict) and isinstance(research_response.get('proposal_draft'), Proposal):
        proposal_text = research_response['proposal_draft'].proposal

        # Create an analysis task package with the proposal text as instruction
        analysis_task_dict = {"task": query, "instruction": proposal_text}
        analysis_taskpack = TaskPackage(instruction=str(analysis_task_dict))
        
        # Set a timeout for the analysis agent call
        try:
            analysis_response = analysis_module(analysis_taskpack)
        except Exception as e:
            print(f"Analysis agent call failed: {e}", flush=True)
            return proposal_text, "None"
    return proposal_text, analysis_response

def literature_module(query: str, literature_module):
    """Execute literature reasoning module"""
    # Literature reasoning
    task_dict = {"user_query": query, "hypothesis": None}
    literature_taskpack = TaskPackage(instruction=str(task_dict))

    # Set a timeout for the literature agent call
    try:
        literature_response = literature_module(literature_taskpack)
    except Exception as e:
        print(f"Literature reasoning agent call failed: {e}", flush=True)
        literature_response = "None"
    return literature_response

def workflow_query_test(query, full_query, research_planning_module, analysis_module, literature_module):
    """
    Executes a Medea workflow that tests a query by generating a research plan
    and executing analysis and literature reasoning.
    """
    # Check for shutdown flag
    if shutdown_flag.is_set():
        return {"error": "Server is shutting down"}
    
    # Record the output from all agents
    agent_output_dict = {}
    inputs_for_research_analysis = (query, research_planning_module, analysis_module)
    inputs_for_literature = (query, literature_module)

    proposal_text, analysis_response, literature_response = "None", "None", "None"

    # Use ThreadPoolExecutor with explicit max_workers
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        
        # Submit both agent functions to the executor
        research_future = executor.submit(research_and_analysis_module, *inputs_for_research_analysis)
        literature_future = executor.submit(literature_module, *inputs_for_literature)

        try:
            # Try retrieving research and analysis response first
            proposal_text, analysis_response = research_future.result(timeout=1600)
        except futures.TimeoutError:
            print("Error: Research and analysis task exceeded the timeout limit.", flush=True)
            research_future.cancel()
        except Exception as e:
            print(f"Error occurred in research/analysis agents: {e}", flush=True)

        try:
            # Retrieve literature response separately
            literature_response = literature_future.result(timeout=500)
        except futures.TimeoutError:
            print("Error: Literature task exceeded the timeout limit.", flush=True)
            literature_future.cancel()
        except Exception as e:
            print(f"Error occurred in literature task: {e}", flush=True)

        # Save whatever output is available
        if proposal_text != "None":
            agent_output_dict['P'] = proposal_text

        if analysis_response != "None":
            agent_output_dict['CG'] = analysis_response

        if literature_response != "None":
            agent_output_dict['R'] = literature_response

        # Log the generated proposal if available
        if proposal_text:
            print(f"[Research Plan]: {proposal_text}\n", flush=True)
        
    # Check for shutdown flag before panel discussion
    if shutdown_flag.is_set():
        return agent_output_dict
        
    # LLM-based Panel Discussion
    panel_query = full_query
        
    # Each agent output will be assigned an LLM to join panel discussion
    hyp_response, llm_hyp_response = multi_round_discussion(
        query=panel_query,
        include_llm=INCLUDE_BACKBONE_LLM,
        mod='diff_context', 
        panelist_llms=PANELIST_LLM,
        proposal_response=proposal_text,
        coding_response=analysis_response, 
        reasoning_response=literature_response, 
        vote_merge=VOTE_MERGE,
        round=DEBATE_ROUND
    )
    
    agent_output_dict['llm'] = llm_hyp_response
    agent_output_dict['CGRH'] = hyp_response

    return agent_output_dict

@app.on_event("startup")
async def startup_event():
    """Initialize Medea agents on startup"""
    global research_planning_agent, analysis_module, literature_reasoning_agent
    research_planning_agent, analysis_module, literature_reasoning_agent = initialize_medea_agents()
    if any(agent is None for agent in [research_planning_agent, analysis_module, literature_reasoning_agent]):
        print("Warning: Failed to initialize one or more Medea agents")

def extract_code_and_execution(coding_response):
    """Extract code snippet and execution result from coding response"""
    code_snippet = None
    execution_result = None
    
    if isinstance(coding_response, dict):
        code_snippet = coding_response.get('code_snippet')
        execution_result = coding_response.get('executed_output')
    elif isinstance(coding_response, str):
        # Try to extract code blocks from string response
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', coding_response, re.DOTALL)
        if code_blocks:
            code_snippet = code_blocks[0]
        execution_result = coding_response
    
    return code_snippet, execution_result

def format_medea_response(agent_output_dict: dict, user_query: str) -> MedeaResponse:
    """Format Medea output into the expected schema format with streamlined agent outputs"""
    
    # Extract the final hypothesis from the panel discussion
    final_response = agent_output_dict.get('CGRH', 'No response generated')
    
    # Extract code and execution results
    coding_response = agent_output_dict.get('CG', 'None')
    code_snippet, execution_result = extract_code_and_execution(coding_response)
    
    # Create streamlined agent outputs
    streamlined_outputs = {
        "P": agent_output_dict.get('P', 'Not available'),
        "CG": agent_output_dict.get('CG', 'Not available'),
        "R": agent_output_dict.get('R', 'Not available'),
        "llm": agent_output_dict.get('llm', 'Not available'),
        "CGRH": agent_output_dict.get('CGRH', 'Not available')
    }
    
    # Create the main code content for the sandbox
    if code_snippet:
        # Use the existing sandbox template for code execution
        main_code = f'''
Medea Multi-Agent Analysis Result
Generated for query: {user_query}

This analysis was performed using the Medea multi-agent system with:
- Proposal Agent: Generated research proposals and plans
- Coding Agent: Executed data analysis and modeling  
- Reasoning Agent: Performed literature search and reasoning
- Panel Discussion: Combined insights from all agents

FINAL ANALYSIS:
{final_response}

AGENT STATUS:
{json.dumps(streamlined_outputs, indent=2)}

# Generated Code from Medea Coding Agent:
{code_snippet}

# Execution Results:
if __name__ == "__main__":
    print("=== MEDEA ANALYSIS EXECUTION ===")
    print("Query:", "{user_query}")
    print("\\n=== EXECUTION RESULTS ===")
    print("""{execution_result}""")
    print("\\n=== FINAL ANALYSIS ===")
    print("""{final_response}""")
'''
    else:
        # Fallback when no code is generated
        main_code = f"""
Medea Multi-Agent Analysis Result
Generated for query: {user_query}

This analysis was performed using the Medea multi-agent system.

FINAL ANALYSIS:
{final_response}

AGENT STATUS:
{json.dumps(streamlined_outputs, indent=2)}
"""

    return MedeaResponse(
        commentary=final_response,
        template="medea",
        title="Medea Multi-Agent Analysis",
        description=f"Multi-agent analysis of: {user_query}",
        additional_dependencies=[],
        has_additional_dependencies=False,
        install_dependencies_command="",
        port=None,
        file_path="medea_analysis.py",
        code=main_code,
        agent_outputs=streamlined_outputs,
        has_code_execution=code_snippet is not None,
        code_snippet=code_snippet,
        execution_result=execution_result
    )

async def stream_medea_response(agent_output_dict: dict, user_query: str):
    """Stream the Medea response in the expected format"""
    response = format_medea_response(agent_output_dict, user_query)
    
    # Convert to JSON and stream
    response_dict = response.dict()
    
    # Stream the response as JSON
    yield f"data: {json.dumps(response_dict)}\n\n"
    yield "data: [DONE]\n\n"

def stream_medea_workflow(query, full_query, research_planning_module, analysis_module, literature_module):
    """Stream the Medea workflow with real-time agent updates"""
    stream_queue = queue.Queue()
    
    # Store original print function
    original_print = print
    
    def stream_callback(data):
        """Callback to send data to the stream queue"""
        try:
            print(f"[DEBUG] Stream callback called with data: {data.get('type', 'unknown')}", flush=True)
            stream_queue.put(data)
            print(f"[DEBUG] Data put in queue, queue size: {stream_queue.qsize()}", flush=True)
        except Exception as e:
            original_print(f"Error in stream callback: {e}", flush=True)
    
    def intercepted_print(*args, **kwargs):
        """Intercept print calls to capture agent logs"""
        # Call original print
        original_print(*args, **kwargs)
        
        # Try to capture all logs (not just agent-specific ones)
        if args:
            log_str = " ".join(str(arg) for arg in args)
            
            # Skip debug messages from this streaming code itself
            if '[DEBUG]' in log_str:
                return
            
            # Check if this looks like an agent log or tool execution
            keywords = [
                '_agent', 'Agent', 'Action:', 'Observation:', 'Context check',
                'research_planning', 'analysis', 'literature', 
                '[', 'Step', 'Finish', 'TaskPackage',
                'checker', 'Tool', 'function'
            ]
            
            if any(keyword in log_str for keyword in keywords):
                try:
                    # Determine agent type from log content
                    agent_name = "medea_system"
                    if "research_planning" in log_str.lower():
                        agent_name = "research_planning"
                    elif "analysis" in log_str.lower():
                        agent_name = "analysis"
                    elif "literature" in log_str.lower():
                        agent_name = "literature"
                    
                    # Stream agent logs directly
                    stream_callback({
                        "type": "agent_log",
                        "agent": agent_name,
                        "content": log_str,
                        "timestamp": time.time()
                    })
                except Exception as e:
                    original_print(f"Error intercepting log: {e}", flush=True)
    
    def run_workflow():
        try:
            print(f"[DEBUG] Starting workflow for query: {query}", flush=True)
            
            # Note: Streaming handled through intercepted print and direct queue puts
            print(f"[DEBUG] Starting workflow execution", flush=True)
            
            # Temporarily replace print function
            import builtins
            builtins.print = intercepted_print
            
            try:
                # Start workflow notification
                print(f"[DEBUG] Putting workflow_start in queue", flush=True)
                stream_queue.put({
                    "type": "workflow_start",
                    "content": f"Starting Medea System for query: {query}...",
                    "timestamp": time.time()
                })
                
                # Execute research planning module
                print(f"[DEBUG] Putting agent_step (Research Planning start) in queue", flush=True)
                stream_queue.put({
                    "type": "agent_step",
                    "agent": "ResearchPlanning",
                    "action": "start",
                    "status": "started",
                    "details": "Initializing research plan generation"
                })
                
                research_task_dict = {"user_query": query}
                research_taskpack = TaskPackage(instruction=str(research_task_dict))
                
                try:
                    print(f"[DEBUG] Calling research planning agent", flush=True)
                    
                    # Send a progress update
                    stream_queue.put({
                        "type": "agent_log",
                        "agent": "research_planning",
                        "content": "Research planning agent is analyzing the query...",
                        "timestamp": time.time()
                    })
                    
                    research_response = research_planning_module(research_taskpack)
                    proposal_text = "None"
                    
                    if isinstance(research_response, dict) and isinstance(research_response.get('proposal_draft'), Proposal):
                        proposal_text = research_response['proposal_draft'].proposal
                        stream_queue.put({
                            "type": "agent_output",
                            "agent": "research_planning",
                            "content": f"{proposal_text}"
                        })
                    else:
                        stream_queue.put({
                            "type": "agent_output",
                            "agent": "research_planning",
                            "content": f"Research plan response: {str(research_response)}"
                        })
                        
                    stream_queue.put({
                        "type": "agent_step",
                        "agent": "ResearchPlanning",
                        "action": "complete",
                        "status": "completed",
                        "details": "Research plan generation completed"
                    })
                    
                except Exception as e:
                    original_print(f"Research planning agent call failed: {e}", flush=True)
                    stream_queue.put({
                        "type": "agent_step",
                        "agent": "ResearchPlanning",
                        "action": "complete",
                        "status": "failed",
                        "details": f"Error: {str(e)}"
                    })
                    proposal_text = "None"
                
                # Execute analysis module if proposal is available
                if proposal_text != "None":
                    stream_queue.put({
                        "type": "agent_step",
                        "agent": "Analysis",
                        "action": "start",
                        "status": "started",
                        "details": "Starting analysis based on research plan"
                    })
                    
                    analysis_task_dict = {"task": query, "instruction": proposal_text}
                    analysis_taskpack = TaskPackage(instruction=str(analysis_task_dict))
                    
                    try:
                        # Send a progress update
                        stream_queue.put({
                            "type": "agent_log",
                            "agent": "analysis",
                            "content": "Analysis agent is executing computational experiments...",
                            "timestamp": time.time()
                        })
                        
                        analysis_response = analysis_module(analysis_taskpack)
                        stream_queue.put({
                            "type": "agent_step",
                            "agent": "Analysis",
                            "action": "complete",
                            "status": "completed",
                            "details": "Analysis completed"
                        })
                        
                        # Extract code snippet if available
                        code_snippet, execution_result = extract_code_and_execution(analysis_response)
                        if code_snippet:
                            stream_queue.put({
                                "type": "agent_output",
                                "agent": "analysis",
                                "content": f"{str(code_snippet)}\n\nExecution Result:\n{str(execution_result)}"
                            })
                    except Exception as e:
                        original_print(f"Analysis agent call failed: {e}", flush=True)
                        analysis_response = "None"
                        stream_queue.put({
                            "type": "agent_step",
                            "agent": "Analysis",
                            "action": "complete",
                            "status": "failed",
                            "details": f"Error: {str(e)}"
                        })
                else:
                    analysis_response = "None"
                    stream_queue.put({
                        "type": "agent_step",
                        "agent": "Analysis",
                        "action": "complete",
                        "status": "failed",
                        "details": "No research plan available"
                    })
                
                # Execute literature reasoning module
                stream_queue.put({
                    "type": "agent_step",
                    "agent": "LiteratureReasoning",
                    "action": "start",
                    "status": "started",
                    "details": "Starting literature analysis and reasoning"
                })
                
                literature_task_dict = {"user_query": query, "hypothesis": None}
                literature_taskpack = TaskPackage(instruction=str(literature_task_dict))
                
                try:
                    # Send a progress update
                    stream_queue.put({
                        "type": "agent_log",
                        "agent": "literature_reasoning",
                        "content": "Literature reasoning agent is searching and analyzing papers...",
                        "timestamp": time.time()
                    })
                    
                    literature_response = literature_module(literature_taskpack)
                    stream_queue.put({
                        "type": "agent_step",
                        "agent": "LiteratureReasoning",
                        "action": "complete",
                        "status": "completed",
                        "details": "Literature analysis completed"
                    })
                    
                    stream_queue.put({
                        "type": "agent_output",
                        "agent": "literature_reasoning",
                        "content": f"Literature-based Reasoning Analysis:\n{str(literature_response)}"
                    })
                except Exception as e:
                    original_print(f"Literature reasoning agent call failed: {e}", flush=True)
                    literature_response = "None"
                    stream_queue.put({
                        "type": "agent_step",
                        "agent": "LiteratureReasoning",
                        "action": "complete",
                        "status": "failed",
                        "details": f"Error: {str(e)}"
                    })
                
                # Panel discussion
                stream_queue.put({
                    "type": "agent_step",
                    "agent": "Panel",
                    "action": "start",
                    "status": "started",
                    "details": f"Starting panel discussion with {len(PANELIST_LLM)} LLMs"
                })
                
                # Send a progress update
                stream_queue.put({
                    "type": "agent_log",
                    "agent": "panel_discussion",
                    "content": f"Panel discussion starting with {len(PANELIST_LLM)} LLM models for consensus...",
                    "timestamp": time.time()
                })
                
                try:
                    hyp_response, llm_hyp_response = multi_round_discussion(
                        query=full_query,
                        include_llm=INCLUDE_BACKBONE_LLM,
                        mod='diff_context', 
                        panelist_llms=PANELIST_LLM,
                        proposal_response=proposal_text,
                        coding_response=analysis_response, 
                        reasoning_response=literature_response, 
                        vote_merge=VOTE_MERGE,
                        round=DEBATE_ROUND
                    )
                    
                    stream_queue.put({
                        "type": "agent_step",
                        "agent": "Panel",
                        "action": "complete",
                        "status": "completed",
                        "details": "Panel discussion completed"
                    })
                    
                    stream_queue.put({
                        "type": "agent_output",
                        "agent": "panel_discussion",
                        "content": f"Panel discussion completed with {len(PANELIST_LLM)} LLMs"
                    })
                except Exception as e:
                    original_print(f"Panel discussion failed: {e}", flush=True)
                    hyp_response, llm_hyp_response = "None", "None"
                    stream_queue.put({
                        "type": "agent_step",
                        "agent": "Panel",
                        "action": "complete",
                        "status": "failed",
                        "details": f"Error: {str(e)}"
                    })
                
                # Final result
                agent_output_dict = {
                    'P': proposal_text,
                    'CG': analysis_response,
                    'R': literature_response,
                    'llm': llm_hyp_response,
                    'CGRH': hyp_response
                }
                
                print(f"[DEBUG] Putting workflow_complete in queue", flush=True)
                stream_queue.put({
                    "type": "workflow_complete",
                    "agent_outputs": agent_output_dict
                })
                
            finally:
                # Restore original print function
                builtins.print = original_print
                
        except Exception as e:
            print(f"[DEBUG] Workflow error: {e}", flush=True)
            stream_queue.put({
                "type": "error",
                "error": str(e)
            })
    
    # Start workflow in background thread
    print(f"[DEBUG] Starting workflow thread", flush=True)
    workflow_thread = threading.Thread(target=run_workflow)
    workflow_thread.start()
    
    # Stream updates from the queue
    print(f"[DEBUG] Starting streaming loop", flush=True)
    while True:
        try:
            # Wait for data with minimal timeout for immediate streaming
            # print(f"[DEBUG] Waiting for data from queue (timeout=0.1)", flush=True)
            data = stream_queue.get(timeout=0.1)  # Reduced from 1.0 to 0.1 seconds
            # print(f"[DEBUG] Got data from queue: {data.get('type', 'unknown')}", flush=True)
            
            if data.get("type") == "workflow_complete":
                # Format final response
                agent_output_dict = data["agent_outputs"]
                print(f"[DEBUG] Final agent outputs: {agent_output_dict}", flush=True)
                response = format_medea_response(agent_output_dict, query)
                response_dict = response.dict()
                print(f"[DEBUG] Formatted response agent_outputs: {response_dict.get('agent_outputs')}", flush=True)
                
                print(f"[DEBUG] Yielding final_response", flush=True)
                yield f"data: {json.dumps({'type': 'final_response', 'data': response_dict})}\n\n"
                yield "data: [DONE]\n\n"
                break
            elif data.get("type") == "error":
                print(f"[DEBUG] Yielding error", flush=True)
                yield f"data: {json.dumps({'type': 'error', 'error': data['error']})}\n\n"
                yield "data: [DONE]\n\n"
                break
            else:
                # Stream intermediate updates immediately
                print(f"[DEBUG] Yielding intermediate update: {data.get('type', 'unknown')}", flush=True)
                try:
                    # Ensure data is JSON serializable
                    json_str = json.dumps(data)
                    yield f"data: {json_str}\n\n"
                except (TypeError, ValueError) as e:
                    print(f"[DEBUG] JSON serialization error: {e}", flush=True)
                    # Try to create a simplified version
                    try:
                        simplified_data = {
                            "type": data.get("type", "unknown"),
                            "agent": data.get("agent", "unknown"),
                            "content": str(data.get("content", "")),
                            "timestamp": data.get("timestamp", time.time())
                        }
                        json_str = json.dumps(simplified_data)
                        yield f"data: {json_str}\n\n"
                    except Exception as e2:
                        print(f"[DEBUG] Failed to create simplified data: {e2}", flush=True)
                        # Send a basic error message
                        error_data = {
                            "type": "stream_error",
                            "content": f"Error serializing data: {str(e)}",
                            "timestamp": time.time()
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
                
        except queue.Empty:
            # Check if workflow thread is still running
            if not workflow_thread.is_alive():
                print(f"[DEBUG] Workflow thread finished, breaking", flush=True)
                break
            continue
        except Exception as e:
            print(f"[DEBUG] Exception in streaming loop: {e}", flush=True)
            # Check if workflow thread is still running
            if not workflow_thread.is_alive():
                break
            continue

@app.post("/api/medea/chat")
async def medea_chat(request: ChatRequest):
    """Chat endpoint for Medea multi-agent system"""
    global research_planning_agent, analysis_module, literature_reasoning_agent
    
    if any(agent is None for agent in [research_planning_agent, analysis_module, literature_reasoning_agent]):
        raise HTTPException(status_code=500, detail="Medea agents not initialized")
    
    try:
        # Extract the user's query from the messages
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        user_query = user_messages[-1].content
        
        # Execute the Medea workflow
        agent_output_dict = workflow_query_test(
            query=user_query,
            full_query=user_query,
            research_planning_module=research_planning_agent,
            analysis_module=analysis_module,
            literature_module=literature_reasoning_agent
        )
        
        # Format and return the response
        formatted_response = format_medea_response(agent_output_dict, user_query)
        
        return formatted_response
        
    except Exception as e:
        print(f"Error in Medea chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/api/medea/chat/stream")
async def medea_chat_stream(request: ChatRequest):
    """Streaming chat endpoint for Medea multi-agent system with real-time agent updates"""
    global research_planning_agent, analysis_module, literature_reasoning_agent
    
    if any(agent is None for agent in [research_planning_agent, analysis_module, literature_reasoning_agent]):
        raise HTTPException(status_code=500, detail="Medea agents not initialized")
    
    try:
        # Extract the user's query from the messages
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        user_query = user_messages[-1].content
        
        # Return streaming response with real-time agent updates
        return StreamingResponse(
            stream_medea_workflow(user_query, user_query, research_planning_agent, analysis_module, literature_reasoning_agent),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        print(f"Error in Medea chat stream: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/api/medea/health")
async def health_check():
    """Health check endpoint"""
    agents_initialized = all(agent is not None for agent in [research_planning_agent, analysis_module, literature_reasoning_agent])
    return {
        "status": "healthy" if agents_initialized and not shutdown_flag.is_set() else "shutting_down",
        "agents_initialized": agents_initialized,
        "research_planning_agent": research_planning_agent is not None,
        "analysis_module": analysis_module is not None,
        "literature_reasoning_agent": literature_reasoning_agent is not None,
        "shutdown_requested": shutdown_flag.is_set()
    }

@app.get("/api/medea/shutdown")
async def shutdown_endpoint():
    """Manual shutdown endpoint"""
    shutdown_flag.set()
    return {"message": "Shutdown requested"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medea API Server",
        "endpoints": {
            "/api/medea/chat": "Chat with Medea (non-streaming)",
            "/api/medea/chat/stream": "Chat with Medea (streaming)",
            "/api/medea/health": "Health check",
            "/api/medea/shutdown": "Manual shutdown"
        }
    }

if __name__ == "__main__":
    import uvicorn
    try:
        print("Starting Medea API Server...")
        print("Press Ctrl+C to stop the server")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\nShutting down Medea API Server...")
        shutdown_flag.set()
        sys.exit(0) 