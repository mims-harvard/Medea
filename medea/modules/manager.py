from agentlite.agents.agent_utils import *
from agentlite.agents import BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.actions import ThinkAct
from agentlite.commons.AgentAct import ActObsChainType
from agentlite.actions.BaseAction import BaseAction

from typing import List
import streamlit as st
from modules.utils import FlushAgentLogger as AgentLogger
from modules.utils import Proposal
from modules.proposal import ProposalAgent, ResearchPlanDraft, ContextVerification, IntegrityVerification
from modules.experiment_analysis import CodeDebug, AnalysisExecution, CodeGenerator, AnalysisQulityChecker, Analysis
from modules.literature_reasoning import LiteratureReasoning, OpenScholarReasoning
from modules.logger import UILogger
from modules.agent_llms import AgentLLM, LLMConfig, parse_action
from modules.BasePrompt import BasePromptGen
from modules.prompt_template import *
from modules.agent_llms import *
from tool_space.gpt_utils import chat_completion
# Agent temp
research_plan_agent_tmp = 0.4
analysis_agent_tmp = 0.6
reasoning_agent_tmp = 0.4
proposal_act_tmp = 0.4
coding_act_tmp = 0.4

LLM_PROVIDER = "claude"

class MedeaSystem(BaseAction):
    def __init__(self) -> None:
        action_name = "MedeaSystem"
        action_desc = "Takes the original user queries as input, call Medea Agent system to generates insightful, multi-phase reasoning to address them, leveraging advanced tools, models, and online publications."
        params_doc = {"user_query": "The original query provided by the user"}
        super().__init__(
            action_name=action_name,
            action_desc=action_desc, 
            params_doc=params_doc,
        )
        proposal_llm_config_dict = {"temperature": research_plan_agent_tmp}
        proposal_llm_config = LLMConfig(proposal_llm_config_dict)
        proposal_llm = AgentLLM(llm_config=proposal_llm_config, llm_name=LLM_PROVIDER)

        coding_llm_config_dict = {"temperature": analysis_agent_tmp}
        coding_llm_config = LLMConfig(coding_llm_config_dict)
        coding_llm = AgentLLM(llm_config=coding_llm_config, llm_name=LLM_PROVIDER)

        reasoning_llm_config_dict = {"temperature": reasoning_agent_tmp}
        reasoning_llm_config = LLMConfig(reasoning_llm_config_dict)
        reasoning_llm = AgentLLM(llm_config=reasoning_llm_config, llm_name=LLM_PROVIDER)


        proposal_actions = [ResearchPlanDraft(tmp=proposal_act_tmp),
                            ContextVerification(tmp=proposal_act_tmp),
                            IntegrityVerification(tmp=proposal_act_tmp, max_iter=0),]
    
        coding_actions = [CodeGenerator(tmp=coding_act_tmp), 
                          AnalysisQulityChecker(tmp=coding_act_tmp), 
                          CodeDebug(tmp=coding_act_tmp), 
                          AnalysisExecution()]
        
        reason_actions = [OpenScholarReasoning(tmp=0.4)]

        self.research_plan_agent = ProposalAgent(llm=proposal_llm, actions=proposal_actions)
        self.analysis_agent = Analysis(llm=coding_llm, actions=coding_actions)
        self.reasoning_agent = LiteratureReasoning(llm=reasoning_llm, actions=reason_actions)

    
    def __call__(self, user_query: str) -> str:
        task_pack = TaskPackage(instruction=user_query)
        proposal_response = self.research_plan_agent(task_pack)
        # Extract the proposal if valid, otherwise return the response directly
        if isinstance(proposal_response, dict) and isinstance(proposal_response.get('proposal_draft'), Proposal):
            proposal_text = proposal_response['proposal_draft'].proposal
            proposal_log = f"Proposal Agent:\n```json\n{proposal_text}\n```"
            with st.chat_message(name="assistant"):
                st.markdown(proposal_log)
            st.session_state.messages.append({"role": "assistant", "content": proposal_log})

            task_dict = str({"task": user_query, "instruction": proposal_text})
            coding_taskpack = TaskPackage(instruction=task_dict)
            cg_response = self.analysis_agent(coding_taskpack)
            if isinstance(cg_response, dict):
                code_snippet = cg_response.get('code_snippet')
                executed_output = cg_response.get('executed_output')
                code_log = f"Coding Agent:\n```python\n{code_snippet}\n```"
                code_result_log = f"Executed: \n```json\n{executed_output}\n```"
                with st.chat_message(name="assistant"):
                    st.markdown(code_log)
                    st.markdown(code_result_log)
                st.session_state.messages.append({"role": "assistant", "content": code_log})
                st.session_state.messages.append({"role": "assistant", "content": code_result_log})
        else:
            proposal_log = f"Proposal Agent:```json\n{proposal_response}\n```"
            with st.chat_message(name="assistant"):
                st.markdown(proposal_log)
            st.session_state.messages.append({"role": "assistant", "content": proposal_log})

        task_dict = str({"user_query": user_query, "hypothesis": None})
        reasoning_taskpack = TaskPackage(instruction=task_dict)
        
        reason_response = self.reasoning_agent(reasoning_taskpack)
        reason_log = f"Reasoning Agent: \n{reason_response['user_query']}"
        with st.chat_message(name="assistant"):
            st.markdown(reason_log)
        st.session_state.messages.append({"role": "assistant", "content": reason_log})

        if isinstance(cg_response, dict):
            code_snippet = cg_response.get('code_snippet')
            executed_output = cg_response.get('executed_output')
        else:
            code_snippet = executed_output = cg_response
        
        if isinstance(reason_response, dict):
            reason_output = reason_response['user_query']
        else:
            reason_output = reason_response
        
        gpt_output = chat_completion(user_query, temperature=1, model='gpt-4o')

        hyp_response = chat_completion(hypothesis_prompt, temperature=1, model='gpt-4o')
        hyp_log = f"Final Hypothesis: \n{hyp_response}"
        with st.chat_message(name="assistant"):
            st.markdown(hyp_log)
        st.session_state.messages.append({"role": "assistant", "content": hyp_log})
        return hyp_response


class ChatFinishAction(BaseAction):
    def __init__(self) -> None:
        action_name = "Finish"
        action_desc = "Complete the chat with a response"
        params_doc = {
            "response": "response to address the user query"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, response: str) -> dict:
        return response

FinishAct = ChatFinishAction()

class MedeaManager(BaseAgent):
    def __init__(self,
        llm: AgentLLM = AgentLLM(
            LLMConfig({"temperature": 0.4}),
            llm_name=LLM_PROVIDER
        ),
        actions: List[BaseAction] = [
            MedeaSystem()
        ], 
        manager = None,
        logger: AgentLogger = AgentLogger(FLAG_PRINT=True, PROMPT_DEBUG_FLAG=False),
    ):
        name = "Medea"
        reasoning_type = "react"
        role = """You are a therapeutic agent chatbot with access to MedeaSystem, an LLM-based multi-agent system. Engage in general conversational interactions with users normally and without invoking tools unless explicitly required. If the user asks a therapeutic-related question or requires assistance with biological queries, leverage MedeaSystem to provide accurate and insightful answers.
	        - MedeaSystem: A specialized multi-agent system designed for therapeutic discovery. Taking the original user query as input, it processes therapeutic-related queries to generate insightful, multi-phase reasoning by utilizing advanced tools, models, and online publications. Only use MedeaSystem when addressing therapeutic or biological queries.
            - Finish: Use Finish action to finish the chat with a response to answer user query
        """

        super().__init__(
            name=name,
            role=role,
            reasoning_type=reasoning_type,
            llm=llm,
            actions=actions, 
            manager=manager, 
            max_exec_steps = 10,
            logger=logger,
        )
        self.prompt_gen = BasePromptGen(
            agent_role=self.role,
            constraint=self.constraint,
            instruction=self.instruction,
        )

    def __next_act__(
        self, task: TaskPackage, action_chain: ActObsChainType
    ) -> AgentAct:
        """one-step action generation

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :param action_chain: history actions and observation of this task from memory
        :type action_chain: ActObsChainType
        :return: the action for agent to execute
        :rtype: AgentAct
        """
        # print(action_chain)
        action_prompt = self.prompt_gen.action_prompt(
            task=task,
            actions=self.actions,
            action_chain=action_chain,
        )
        # print("Action Prompt: \n", action_prompt)
        self.logger.get_prompt(action_prompt)
        raw_action = self.llm_layer(action_prompt)
        self.logger.get_llm_output(raw_action)
        return self.__action_parser__(raw_action, action_chain)
    
    def __action_parser__(self, raw_action: str, action_chain:ActObsChainType) -> AgentAct:
        """parse the generated content to an executable action

        :param raw_action: llm generated text
        :type raw_action: str
        :return: an executable action wrapper
        :rtype: AgentAct
        """
        # print(raw_action)
        action_name, args, PARSE_FLAG = parse_action(raw_action)
        agent_act = AgentAct(name=action_name, params=args)
        return agent_act
    
    
    def forward(self, task: TaskPackage, agent_act: AgentAct) -> str:
        """forward the action to get the observation or response from other agent

        :param task: the task to forward
        :type task: TaskPackage
        :param agent_act: the action to forward
        :type agent_act: AgentAct
        :return: the observation or response from other agent
        :rtype: str
        """
        act_found_flag = False
        param_parse_flag = False
        # if match one in self.actions
        for action in self.actions:
            if act_match(agent_act.name, action):
                act_found_flag = True
                try:
                    observation = action(**agent_act.params)
                except Exception as e:
                    print(e)
                    observation = (MISS_ACTION_PARAM.format(param_doc=action.params_doc, failed_param=agent_act.params))
                    return observation
                # if action is Finish Action
                if agent_act.name == FinishAct.action_name:
                    task.answer = observation
                    task.completion = "completed"
            if action.action_name in agent_act.name:
                param_parse_flag = True
        # if not find this action
        if act_found_flag:
            return observation
        if param_parse_flag:
            return WRONG_ACTION_PARAM
        return ACION_NOT_FOUND_MESS
    
    def __add_inner_actions__(self):
        """adding the inner action types into agent, which is based on the `self.reasoning_type`"""
        if self.reasoning_type == "react":
            self.actions += [ThinkAct, FinishAct]
        if self.reasoning_type == "act":
            self.actions += [FinishAct]
        self.actions = list(set(self.actions))
    
if __name__ == "__main__":
    proposal_llm_config_dict = {"temperature": research_plan_agent_tmp}
    proposal_llm_config = LLMConfig(proposal_llm_config_dict)
    proposal_llm = AgentLLM(llm_config=proposal_llm_config)

    coding_llm_config_dict = {"temperature": analysis_agent_tmp}
    coding_llm_config = LLMConfig(coding_llm_config_dict)
    coding_llm = AgentLLM(llm_config=coding_llm_config)

    reasoning_llm_config_dict = {"temperature": reasoning_agent_tmp}
    reasoning_llm_config = LLMConfig(reasoning_llm_config_dict)
    reasoning_llm = AgentLLM(llm_config=reasoning_llm_config)


    proposal_actions = [ResearchPlanDraft(tmp=proposal_act_tmp),
                        ContextVerification(tmp=proposal_act_tmp),
                        IntegrityVerification(tmp=proposal_act_tmp, max_iter=0),
                        ]

    coding_actions = [CodeGenerator(tmp=coding_act_tmp), 
                        AnalysisQulityChecker(tmp=coding_act_tmp), 
                        CodeDebug(tmp=coding_act_tmp), 
                        AnalysisExecution()
                    ]
    reason_actions = [OpenScholarReasoning(tmp=0.4)]

    research_plan_agent = ProposalAgent(llm=proposal_llm, actions=proposal_actions, logger=UILogger(FLAG_PRINT=True))
    analysis_agent = Analysis(llm=coding_llm, actions=coding_actions, logger=UILogger(FLAG_PRINT=True))
    reasoning_agent = LiteratureReasoning(llm=reasoning_llm, actions=reason_actions, logger=UILogger(FLAG_PRINT=True))

    user_query = """Considering that CD140B and nPKC-theta are both implicated in signaling pathways, what is the most specific genetic interaction that might arise from their mutations in K562 cells? 
                    Could you provide the most likely genetic interaction that could occur, particularly in the absence of a defined phenotype, and how these signaling pathways might influence each other? """
    task_pack = TaskPackage(instruction=user_query)
    proposal_response = research_plan_agent(task_pack)
    # Extract the proposal if valid, otherwise return the response directly
    if isinstance(proposal_response, dict) and isinstance(proposal_response.get('proposal_draft'), Proposal):
        proposal_text = proposal_response['proposal_draft'].proposal
        # with st.chat_message(name="assistant"):
        #     st.markdown(proposal_text)
    
        task_dict = str({"task": user_query, "instruction": proposal_text})
        coding_taskpack = TaskPackage(instruction=task_dict)
        cg_response = analysis_agent(coding_taskpack)
        if isinstance(cg_response, dict):
            code_snippet = cg_response.get('code_snippet')
            executed_output = cg_response.get('executed_output')
            print(code_snippet, flush=True)
            print(executed_output, flush=True)
            # with st.chat_message(name="assistant"):
            #     st.markdown(code_snippet)
            #     st.markdown(executed_output)
    # else:
    #     with st.chat_message(name="assistant"):
    #         st.markdown(proposal_response)

    task_dict = str({"user_query": user_query, "hypothesis": None})
    reasoning_taskpack = TaskPackage(instruction=task_dict)
    
    reason_response = reasoning_agent(reasoning_taskpack)
    # with st.chat_message(name="assistant"):
    #     st.markdown(reason_response)
    
    if isinstance(cg_response, dict):
        code_snippet = cg_response.get('code_snippet')
        executed_output = cg_response.get('executed_output')
    else:
        code_snippet = executed_output = cg_response
    
    if isinstance(reason_response, dict):
        reason_output = reason_response['user_query']
    else:
        reason_output = reason_response
    
    gpt_output = chat_completion(user_query, temperature=1, model='gpt-4o')
