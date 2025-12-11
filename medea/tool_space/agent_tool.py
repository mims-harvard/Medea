import sys
sys.path.insert(0, '.')
from multiprocessing import freeze_support
from agents.literature_reasoning import *
from agentlite.commons import TaskPackage
from agents.agent_llms import LLMConfig, AgentLLM
from agents.utils import FlushAgentLogger as AgentLogger, ReasoningPackage


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
    Scientific reasoning agent that returns ReasoningPackage on success, string on failure.
    
    Returns:
        ReasoningPackage: When reasoning is successful with literature grounding
        str: When no literature found, OpenScholar fails, or other errors occur
    """
    
    freeze_support()
    reason_llm_config_dict = {"temperature": reason_agent_tmp}
    reason_llm_config = LLMConfig(reason_llm_config_dict)
    reason_llm = AgentLLM(llm_config=reason_llm_config, llm_name=llm_name)
    logger = AgentLogger(FLAG_PRINT=False, PROMPT_DEBUG_FLAG=False)

    # Pass verbose parameter to reasoning actions
    reason_actions = [
        LiteratureSearch(model_name=llm_name, verbose=verbose),
        PaperJudge(model_name=llm_name, verbose=verbose),
        OpenScholarReasoning(tmp=reason_action_tmp, llm_provider=llm_name, model_name=llm_name, verbose=verbose)
    ]
    reason_agent = LiteratureReasoning(llm=reason_llm, actions=reason_actions, logger=logger)
    return reasoning_module(user_instruction, reason_agent)



if __name__ == "__main__":
    print(scientific_reasoning_agent("what is ICI treatment?"))