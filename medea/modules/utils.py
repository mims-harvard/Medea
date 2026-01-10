from agentlite.logging.terminal_logger import AgentLogger
from agentlite.actions.BaseAction import BaseAction
from agentlite.logging.utils import *
from pydantic import BaseModel

import time
import uuid

class TaskPackage(BaseModel):
    task: str
    instruction: str
    completion: str = "active"
    creator: str = ""
    timestamp: str = time.time()
    answer: str = ""
    executor: str = ""
    priority: int = 5
    task_id: str = str(uuid.uuid4())

    def __str__(self):
        task_dict = str({"task": self.task, "instruction": self.instruction})
        return f"""Task ID: {self.task_id}\nUser Query: {self.task}\nInstruction: {task_dict}\nTask Creator: {self.creator}\nTask Completion:{self.completion}\nAnswer: {self.answer}\nTask Executor: {self.executor}"""
    

class Tool:
    def __init__(self, info: dict=None):
        self.name = None
        self.type = None
        self.description = None
        self.import_path = None
        self.input_params = None
        self.output_params = None
        self.input_type = None
        self.output_type = None
        
        if info:
            self.__dict__.update(info)
    
    def __str__(self):
        return f"{self.name} - {self.description}"
    
    def __repr__(self):
        return f"{self.name} - {self.description}"
    
    def get_info(self):
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "import_path": self.import_path,
            "input_params": self.input_params,
            "output_params": self.output_params,
            "input_type": self.input_type,
            "output_type": self.output_type
        }
        
        
class Proposal:
    def __init__(self, user_query=None, proposal=None):
        self.user_query = user_query
        self.proposal = proposal
        self.feedback = None
        self.id_mapping_feedback = [None]
        self.status = "Failed"
        self.proposal_id = str(uuid.uuid4().int)[:4]
    
    def __str__(self):
        return f"<Proposal:{self.proposal_id}>"
    
    def __repr__(self):
        return '<Proposal:' + self.proposal_id + '>'
    
    def __len__(self):
        """Return the length of the proposal content"""
        return len(self.proposal) if self.proposal else 0
    
    def __dict__(self):
        return {
            "proposal": self.proposal,
            "status": self.status,
            "proposal_id": self.proposal_id
        }
    
    def update_id_feedback(self, id_mapping_feedback):
        self.id_mapping_feedback.append(id_mapping_feedback)
        
    def get_id(self):
        return '<Proposal:' + self.proposal_id + '>'
    
    def add_feedback(self, feedback: str):
        self.feedback = feedback
    
    def get_proposal(self):
        return self.proposal
    
    def get_query(self):
        return self.user_query
    
    def get_status(self):
        return self.status
    
    def update_status(self, status: str):
        if status in ["Failed", "Approved"]:
            self.status = status
        else:
            raise ValueError("Invalid status")
    
    def log_summary(self):
        if self.status == 'Failed' and self.id_mapping_feedback[-1] is None:
            return f"{self.get_id()} created. Call ContextVerification action next."
        elif self.status == 'Failed' and self.feedback is None:
            return f"{self.get_id()} created. Call IntegrityVerification action next."
        elif self.status == 'Failed':
            return f"{self.get_id()} refined. Please perfrom IntegrityVerification action later."
        return f"{self.get_id()} approved, please do Finish action."
    
    def get_summary(self):
        summary = f"{self.proposal}\n"
        if self.feedback is not None:
            summary += f"Feedback: {self.feedback}\n"
        if self.id_mapping_feedback[-1] is not None:
            summary += f"ID Mapping Feedback: {self.id_mapping_feedback[-1]}"
        return summary

    def retrieve_mapper_feedback_trace(self):
        return self.id_mapping_feedback[-2], self.id_mapping_feedback[-1]
    
    def get_current_mapper_feedback(self):
        return self.id_mapping_feedback[-1]


# Code Snippet Object to store the code snippet, execution output, and quality feedback
class CodeSnippet:
    def __init__(self, task: str, 
                instruction: str, 
                tool_info: list,
                code_snippet: str):
        
        self.task = task
        self.instruction = instruction
        self.code_snippet = code_snippet
        self.tool_info = tool_info
        self.feedback = None
        self.code_output = None
        self.stderr = None
        self.status = "unexecuted"
        self.snippet_id = str(uuid.uuid4().int)[:4]
        self.status_set = {"unexecuted", "executed", "error", "approved"}
        
    def __str__(self):
        return f"<CodeSnippet:{self.snippet_id}>"
    
    def __dict__(self):
        return {
            "code_snippet": self.code_snippet,
            "status": self.status,
            "proposal_id": self.snippet_id
        }
    
    def __repr__(self):
        return f"<CodeSnippet:{self.snippet_id}>"
    
    def update_feedback(self, feedback):
        self.feedback = feedback
        self.status = "quality_checked"
    
    def update_status(self, status):
        if status in self.status_set:
            self.status = status
        else:
            raise ValueError(f"{status} is not a valid status.")
    
    def get_id(self):
        return f"<CodeSnippet:{self.snippet_id}>"
    
    def get_code(self):
        return self.code_snippet
    
    def get_feedback(self):
        return self.feedback


# LiteratureCollection class for managing literature efficiently
class LiteratureCollection:
    def __init__(self, search_query: str = "", papers: list = None):
        self.search_query = search_query
        self.papers = papers or []
        self.sources_used = []
        self.total_found = len(self.papers)
        self.relevant_count = None
        self.assessments = []
        self.id = str(uuid.uuid4().int)[:4]
        self.status = "unprocessed"  # unprocessed, judged, filtered
        
    def __str__(self):
        return f"<LiteratureCollection:{self.id}>"
    
    def __repr__(self):
        return f"<LiteratureCollection:{self.id}>"
    
    def __len__(self):
        """Return the number of papers in the collection"""
        return len(self.papers)
    
    def get_id(self):
        return f"<LiteratureCollection:{self.id}>"
    
    def add_papers(self, papers: list, source: str = "unknown"):
        """Add papers from a specific source"""
        self.papers.extend(papers)
        if source not in self.sources_used:
            self.sources_used.append(source)
        self.total_found = len(self.papers)
    
    def set_papers(self, papers: list):
        """Replace all papers with new list"""
        self.papers = papers
        self.total_found = len(self.papers)
    
    def filter_papers(self, relevant_papers: list, assessments: list = None):
        """Filter to keep only relevant papers"""
        self.papers = relevant_papers
        self.relevant_count = len(relevant_papers)
        self.status = "filtered"
        if assessments:
            self.assessments = assessments
    
    def get_papers(self):
        """Get the list of papers"""
        return self.papers
    
    def get_paper_count(self):
        """Get count of papers"""
        return len(self.papers)
    
    def get_summary(self):
        """Get a summary of the collection for context"""
        status_info = f"Status: {self.status}"
        if self.relevant_count is not None:
            status_info += f" ({self.relevant_count}/{self.total_found} relevant)"
        
        papers_preview = []
        for i, paper in enumerate(self.papers[:3]):  # Show first 3 papers
            title = paper.get('title', 'Unknown title')[:60] + "..."
            papers_preview.append(f"{i+1}. {title}")
        
        preview_text = "\n".join(papers_preview)
        if len(self.papers) > 3:
            preview_text += f"\n... and {len(self.papers) - 3} more papers"
        
        return f"""Literature Collection Summary:
Query: {self.search_query}
{status_info}
Sources: {', '.join(self.sources_used) if self.sources_used else 'Unknown'}
Papers Preview:
{preview_text}"""
    
    def get_context_summary(self):
        """Get a brief context summary for LLM consumption"""
        paper_titles = [p.get('title', 'Unknown')[:50] for p in self.papers[:5]]
        return f"Literature collection with {len(self.papers)} papers including: {', '.join(paper_titles)}"


# ReasoningPackage class for internal use
class ReasoningPackage:
    def __init__(self) -> None:
        self.papers = []
        self.task = None
        self.hypothesis = None
        self.id = str(uuid.uuid4().int)[:4]
        self.reasoning = {"user_query": None, "hypothesis": None}
        self.reasoning_type = ["user_query", "hypothesis", "gpt"]
        
    def __str__(self):
        return f"<ReasoningPackage:{self.id}>"
    
    def update_reasoning(self, reasoning: str, citation: str, track: str):
        if track not in self.reasoning_type:
            raise ValueError(f"Track must be one of {self.reasoning_type}")
        self.reasoning[track] = {"answer": reasoning, "citation": citation}

    def update_papers(self, papers: list):
        self.papers.append(papers)

    def get_papers(self):
        return self.papers
    
    def get_id(self):
        return f"<ReasoningPackage:{self.id}>"
    
    def log_summary(self):
        if len(self.reasoning) != 0:
            return f"{self.get_id()} created. Please do Finish action"
        else:
            return "ReasoningPackage is empty. Please retry the last action."
    
class FlushAgentLogger(AgentLogger):
    def __init__(
        self,
        log_file_name: str = "agent.log",
        FLAG_PRINT: bool = True,
        PROMPT_DEBUG_FLAG: bool = False,
        OBS_OFFSET: int = None,  # None means no truncation
    ) -> None:
        super().__init__(
            log_file_name=log_file_name,
            FLAG_PRINT=FLAG_PRINT,
            PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        self.OBS_OFFSET = OBS_OFFSET  # Override parent's OBS_OFFSET
        
    def __save_log__(self, log_str: str):
        if self.FLAG_PRINT:
            print(log_str, flush=True)
        with open(self.log_file_name, "a") as f:
            f.write(str_color_remove(log_str) + "\n")
    
    def get_obs(self, obs):
        if type(obs) == Proposal or type(obs) == ReasoningPackage:
            obs = obs.log_summary()
        if type(obs) == CodeSnippet:
            obs = f"{obs} created."
        # Only truncate if OBS_OFFSET is set
        if self.OBS_OFFSET is not None and len(obs) > self.OBS_OFFSET:
            obs = obs[: self.OBS_OFFSET] + "[TLDR]"
        log_str = f"""Observation: {self.__color_obs_str__(obs)}"""
        self.__save_log__(log_str)
