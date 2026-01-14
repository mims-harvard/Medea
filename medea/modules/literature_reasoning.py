from agentlite.actions.BaseAction import BaseAction
from agentlite.actions import ThinkAct
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.commons.AgentAct import ActObsChainType
from agentlite.agents.agent_utils import act_match
try:
    from agentlite.agents.agent_utils import ACTION_NOT_FOUND_MESS
except ImportError:
    ACTION_NOT_FOUND_MESS = "[Error] Action not found in action list."
from typing import List, Dict, Any
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Use relative imports within package
from ..tool_space.search_api import KeywordExtractor, SemanticScholarSearch, LLMPaperJudge, PaperQueryAligner
from ..tool_space.open_alex import search_openalex_papers
from ..tool_space.open_scholar import OpenScholar

# FlagEmbedding may download models on first import - this can take time
print("[INFO] Loading FlagEmbedding library (may download models on first run)...", flush=True)
from FlagEmbedding import FlagReranker
print("[INFO] FlagEmbedding loaded successfully", flush=True)

from .agent_llms import LLMConfig, AgentLLM
from .agent_llms import parse_action
from .utils import FlushAgentLogger as AgentLogger
from .BasePrompt import BasePromptGen
from .utils import ReasoningPackage, LiteratureCollection
from .prompt_template import *
from .prompt_template import MISS_ACTION_PARAM, WRONG_ACTION_PARAM


class LiteratureSearch(BaseAction):
    """Comprehensive literature search using multiple academic databases."""
    
    def __init__(self, model_name=os.getenv("BACKBONE_LLM"), verbose=True) -> None:
        action_name = "LiteratureSearch"
        action_desc = "Search for academic literature using Semantic Scholar and OpenAlex databases with intelligent keyword extraction. Returns a LiteratureCollection object."
        params_doc = {
            "user_query": "The research question or hypothesis to analyze",
            "max_papers": "Maximum number of papers to retrieve (default: 16)",
            "include_openalex": "Whether to include OpenAlex search results (default: True)",
            "min_citation_count": "Minimum citation count filter for papers (default: 0)"
        }
        super().__init__(
            action_name=action_name, 
            action_desc=action_desc, 
            params_doc=params_doc
        )
        self.model_name = model_name
        self.verbose = verbose
        self.keyword_extractor = KeywordExtractor(verbose=verbose)
        self.semantic_scholar = SemanticScholarSearch(extractor=self.keyword_extractor, verbose=verbose)

    def __call__(self, user_query: str, max_papers: int = 16, include_openalex: bool = True, min_citation_count: int = 0):
        """Search for literature using multiple academic databases."""
        if self.verbose:
            print(f"[LiteratureSearch] Searching for: {user_query}")
        
        # Ensure parameters are correct types (in case they come as strings)
        max_papers = int(max_papers) if isinstance(max_papers, str) else (max_papers or 16)
        min_citation_count = int(min_citation_count) if isinstance(min_citation_count, str) else (min_citation_count or 0)
        
        # Create literature collection
        collection = LiteratureCollection(search_query=user_query)
        
        try:
            # Search OpenAlex if requested
            if include_openalex:
                if self.verbose:
                    print("[LiteratureSearch] Searching OpenAlex...")
                openalex_papers, keywords_used = search_openalex_papers(
                    question=user_query,
                    max_results=max_papers//2,
                    model_name=self.model_name,
                    verbose=self.verbose
                )
                if openalex_papers and isinstance(openalex_papers, list):
                    collection.add_papers(openalex_papers, "OpenAlex")
                    if self.verbose:
                        print(f"[LiteratureSearch] Found {len(openalex_papers)} papers from OpenAlex")
            
            # Search Semantic Scholar     
            if self.verbose:
                print("[LiteratureSearch] Searching Semantic Scholar...")
            ss_papers = self.semantic_scholar.search(
                question=user_query,
                model_name=self.model_name,
                max_paper_num=max_papers//2,
                min_citation_count=min_citation_count
            )
            if ss_papers:
                collection.add_papers(ss_papers, "Semantic Scholar")
                if self.verbose:
                    print(f"[LiteratureSearch] Found {len(ss_papers)} papers from Semantic Scholar")
            
            
            # Deduplicate papers by title and DOI
            unique_papers = self._deduplicate_papers(collection.get_papers())
            
            # Limit results and update collection
            final_papers = unique_papers[:max_papers]
            collection.set_papers(final_papers)
            
            if self.verbose:
                print(f"[LiteratureSearch] Final results: {len(final_papers)} unique papers from {', '.join(collection.sources_used)}")
                print(f"[LiteratureSearch] Created {collection.get_id()}")
            return collection
            
        except Exception as e:
            if self.verbose:
                print(f"[LiteratureSearch] Error during search: {e}")
            # Return empty collection with error info
            collection.status = "error"
            collection.error = str(e)
            return collection
    
    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity and DOI."""
        if not papers:
            return []
        
        unique_papers = []
        seen_titles = set()
        seen_dois = set()
        
        for paper in papers:
            title = (paper.get('title') or '').lower().strip()
            doi = (paper.get('doi') or '').strip()
            
            # Skip if title or DOI already seen
            if title in seen_titles or (doi and doi in seen_dois):
                continue
                
            if title:
                seen_titles.add(title)
            if doi:
                seen_dois.add(doi)
                
            unique_papers.append(paper)
        
        return unique_papers


class PaperJudge(BaseAction):
    """Evaluate paper relevance using LLM-powered assessment."""
    
    def __init__(self, model_name=os.getenv("PAPER_JUDGE_LLM"), verbose=True, max_workers=None) -> None:
        action_name = "PaperJudge"
        action_desc = "Evaluate whether papers in a LiteratureCollection are relevant to a research query and return a filtered LiteratureCollection"
        params_doc = {
            "user_query": "The research question or hypothesis to evaluate against",
            "literature_collection": "LiteratureCollection object from LiteratureSearch (e.g., <LiteratureCollection:xxxx>)",
            "filter_relevant_only": "Whether to return only relevant papers (default: True)"
        }
        super().__init__(
            action_name=action_name, 
            action_desc=action_desc, 
            params_doc=params_doc
        )
        self.model_name = model_name
        self.verbose = verbose
        self.paper_judge = LLMPaperJudge()
        self.query_aligner = PaperQueryAligner(judge=self.paper_judge)
        # Default to 5 workers if not specified
        self.max_workers = max_workers if max_workers is not None else 5
        # Note: No threading.Lock() here to keep agent picklable for multiprocessing

    def _evaluate_single_paper(self, paper_idx: int, paper: Dict, user_query: str, total_papers: int):
        """Helper method to evaluate a single paper (used for parallel processing)."""
        try:
            # Simple print without lock - minor output interleaving is acceptable
            if self.verbose:
                print(f"[PaperJudge] Evaluating paper {paper_idx+1}/{total_papers}", flush=True)
            
            is_relevant, explanation = self.paper_judge.judge_paper(
                query=user_query,
                paper_entity=paper,
                model_name=self.model_name,
                verbose=False  # Disable verbose in paper_judge to avoid thread-unsafe printing
            )
            
            assessment = {
                "paper": paper,
                "is_relevant": is_relevant,
                "explanation": explanation,
                "title": paper.get('title', 'Unknown title')[:50] + "...",
                "index": paper_idx  # Keep track of original order
            }
            
            return assessment, None
        except Exception as e:
            return None, (paper_idx, str(e))

    def __call__(self, user_query: str, literature_collection: LiteratureCollection, filter_relevant_only: bool = True):
        """Judge paper relevance and provide detailed assessments (parallelized)."""
        papers = literature_collection.get_papers()
        if self.verbose:
            print(f"[PaperJudge] Evaluating {len(papers)} papers in {literature_collection.get_id()} for relevance to: {user_query}")
            print(f"[PaperJudge] Using parallel processing with max_workers={self.max_workers}")
        
        if not papers:
            if self.verbose:
                print("[PaperJudge] No papers to evaluate")
            literature_collection.status = "judged"
            return "[PaperJudge] No papers to evaluate, indicating that there is no existing literature that is relevant to the user query. Suggest to abstain from reasoning."
        
        try:
            assessments = []
            relevant_papers = []
            errors = []
            
            # Evaluate papers in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all papers for evaluation
                future_to_paper = {
                    executor.submit(self._evaluate_single_paper, i, paper, user_query, len(papers)): i 
                    for i, paper in enumerate(papers)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_paper):
                    assessment, error = future.result()
                    if error:
                        errors.append(error)
                        if self.verbose:
                            print(f"[PaperJudge] Error evaluating paper {error[0]+1}: {error[1]}", flush=True)
                    else:
                        assessments.append(assessment)
                        if assessment["is_relevant"]:
                            relevant_papers.append(assessment["paper"])
            
            # Sort assessments by original order
            assessments.sort(key=lambda x: x["index"])
            
            # Remove index from final assessments
            for assessment in assessments:
                del assessment["index"]
            
            # Create new filtered collection or update existing one
            if filter_relevant_only:
                literature_collection.filter_papers(relevant_papers, assessments)
            else:
                literature_collection.assessments = assessments
                literature_collection.relevant_count = len(relevant_papers)
                literature_collection.status = "judged"
            
            if self.verbose:
                print(f"[PaperJudge] Assessment complete: {len(relevant_papers)}/{len(papers)} papers deemed relevant")
                if errors:
                    print(f"[PaperJudge] {len(errors)} errors occurred during evaluation")
                print(f"[PaperJudge] Updated {literature_collection.get_id()}")
            return f"[PaperJudge] Finished evaluating {literature_collection} for user_query: {user_query}"
            
        except Exception as e:
            if self.verbose:
                msg = f"[PaperJudge] Error during evaluation of {literature_collection}: {e}"
                print(msg, flush=True)
            literature_collection.status = "error"
            literature_collection.error = str(e)
            literature_collection.papers = []
            return msg


class OpenScholarReasoning(BaseAction):
    """Advanced reasoning using OpenScholar with literature-grounded responses."""
    
    def __init__(self, llm_provider=os.getenv("BACKBONE_LLM"), model_name=os.getenv("BACKBONE_LLM"), tmp=0.4, verbose=True) -> None:
        action_name = "OpenScholarReasoning"
        action_desc = "Generate comprehensive reasoning and analysis grounded in literature using OpenScholar with LiteratureCollection context"
        params_doc = {
            "user_query": "The research question or hypothesis to analyze",
            "literature_collection": "Optional: LiteratureCollection object to use as context (e.g., <LiteratureCollection:xxxx>)"
        }
        super().__init__(
            action_name=action_name, 
            action_desc=action_desc, 
            params_doc=params_doc
        )
        self.model_name = model_name
        self.default_reranker = "OpenSciLM/OpenScholar_Reranker"
        self.verbose = verbose
        
    def __call__(self, user_query: str, literature_collection: LiteratureCollection = None, **kwargs):
        """Generate literature-grounded reasoning using OpenScholar directly."""
        
        try:
            if self.verbose:
                print(f"[OpenScholarReasoning] Analyzing query: {user_query}")
            
            # Check if we have literature - abstain if none found
            if literature_collection is None or literature_collection.get_paper_count() == 0:
                if self.verbose:
                    print("[OpenScholarReasoning] No literature found, abstaining from reasoning")
                return "[OpenScholarReasoning]: No relevant scientific studies found to ground the analysis. I would suggest to abstain the reasoning."
            
            # Prepare papers for OpenScholar
            papers = literature_collection.get_papers()
            if self.verbose:
                print(f"[OpenScholarReasoning] Using {len(papers)} papers from {literature_collection.get_id()}")
            
            # Prepare data for OpenScholar
            reason_dict = [{'input': user_query, 'ctxs': papers}]
            
            # Initialize reranker and OpenScholar
            # Auto-detect device: use CUDA if available, otherwise CPU
            if 'devices' in kwargs:
                device = kwargs['devices']
                use_fp16 = True  # User specified device, assume they know what they want
            else:
                if torch.cuda.is_available():
                    device = 'cuda:0'
                    use_fp16 = True
                    if self.verbose:
                        print(f"[OpenScholarReasoning] CUDA detected, using GPU", flush=True)
                else:
                    device = 'cpu'
                    use_fp16 = False  # FP16 not supported on CPU
                    if self.verbose:
                        print(f"[OpenScholarReasoning] No CUDA detected, using CPU", flush=True)
            
            if self.verbose:
                print(f"[OpenScholarReasoning] Initializing reranker: {self.default_reranker}", flush=True)
                print(f"[OpenScholarReasoning] Device: {device}, FP16: {use_fp16}", flush=True)
                print(f"[OpenScholarReasoning] First-time use may download model files (~1GB)...", flush=True)
            
            reranker = FlagReranker(
                self.default_reranker, 
                use_fp16=use_fp16
            )
            
            if self.verbose:
                print(f"[OpenScholarReasoning] Reranker initialized successfully", flush=True)
            
            open_scholar = OpenScholar(
                model=kwargs.get('model'),
                tokenizer=kwargs.get('tokenizer'),
                client_llm=self.model_name, 
                reranker=reranker,
                use_contexts=kwargs.get('use_contexts', True),
                top_n=kwargs.get('top_n', 4),
                min_citation=kwargs.get('min_citation', 5),
                norm_cite=kwargs.get('norm_cite', False),
                ss_retriever=kwargs.get('ss_retriever', False)
            )
            
            # Prepare the item for OpenScholar (it expects a single item, not a list)
            item = reason_dict[0]  # Extract the single item from the list
            
            # Generate response using OpenScholar
            result_item = open_scholar.run(
                item,
                ranking_ce=kwargs.get('ranking_ce', True),
                use_feedback=kwargs.get('feedback', True),
                skip_generation=kwargs.get('skip_generation', False),
                posthoc_at=kwargs.get('posthoc_at', True),
                llama3_chat=False,
                task_name=kwargs.get('task_name', "default"),
                zero_shot=kwargs.get('zero_shot', True),
                max_per_paper=kwargs.get('max_per_paper', 3),
                use_abstract=kwargs.get('use_abstract', True),
                max_tokens=kwargs.get('max_tokens', 2000),
            )
            
            # Get the reasoning output
            reasoning_output = result_item.get('output', 'No output generated')
            
            if not reasoning_output or reasoning_output == 'No output generated':
                if self.verbose:
                    print(f"[OpenScholarReasoning] OpenScholar failed to generate output")
                return "Unable to provide reasoning: OpenScholar failed to generate a response."
            
            # Create ReasoningPackage for successful reasoning
            reasoning_package = ReasoningPackage()
            reasoning_package.task = user_query
            reasoning_package.update_papers(papers)
            
            # Extract citations if available
            citations = result_item.get('citations', '')
            if not citations and 'ctxs' in result_item:
                # Build citations from context papers
                citations = "\n".join([
                    f"[{i+1}] {ctx.get('title', 'Unknown Title')} ({ctx.get('year', 'Unknown Year')})"
                    for i, ctx in enumerate(result_item.get('ctxs', [])[:5])  # Limit to first 5
                ])
            
            # Update reasoning with OpenScholar results
            reasoning_package.update_reasoning(
                reasoning=reasoning_output,
                citation=citations,
                track="user_query"  # Using "gpt" track for OpenScholar reasoning
            )
            
            if self.verbose:
                print(f"[OpenScholarReasoning] Created {reasoning_package.get_id()} successfully")
            
            return reasoning_package
            
        except Exception as e:
            if self.verbose:
                print(f"[OpenScholarReasoning] Error during reasoning: {e}")
            return f"Unable to provide reasoning: OpenScholar processing failed with error - {str(e)}"


class ReasonFinishAction(BaseAction):
    """Complete the reasoning task with final results."""
    
    def __init__(self) -> None:
        action_name = "Finish"
        action_desc = "Complete the task with comprehensive reasoning results"
        params_doc = {
            "reasoning_result": "The reasoning result from OpenScholarReasoning (ReasoningPackage object or error string)"
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, reasoning_result: ReasoningPackage):
        """Finalize and return the reasoning results."""
        # Handle string error cases first
        if isinstance(reasoning_result, str):
            return reasoning_result  # Return error message as-is
        
        # Handle ReasoningPackage objects
        if isinstance(reasoning_result, ReasoningPackage):
            if hasattr(reasoning_result, 'reasoning') and reasoning_result.reasoning:
                return reasoning_result.reasoning  # Return the dict containing reasoning results
            else:
                return "No reasoning content available in the ReasoningPackage."
        
        # Handle invalid types
        return f"Invalid reasoning result type: {type(reasoning_result)}. Please provide a ReasoningPackage object or an error string."


ReasonFinishAct = ReasonFinishAction()


class LiteratureReasoning(BaseAgent):
    """Advanced reasoning agent with literature search, paper evaluation, and OpenScholar integration."""
    
    def __init__(self,
        llm: AgentLLM = AgentLLM(
            LLMConfig({"temperature": 0.4}),
            llm_name=os.getenv("BACKBONE_LLM")
        ),
        actions: List[BaseAction] = None, 
        manager: ABCAgent = None,
        logger: AgentLogger = AgentLogger(FLAG_PRINT=True, PROMPT_DEBUG_FLAG=False),
    ):
        name = "reasoning_agent"
        reasoning_type = "react"
        role = REASONING_AGENT_TEMPLATE
        
        # Set default actions if none provided
        if actions is None:
            actions = [
                LiteratureSearch(model_name=os.getenv("BACKBONE_LLM")),
                PaperJudge(model_name=os.getenv("BACKBONE_LLM")),
                OpenScholarReasoning(llm_provider=os.getenv("BACKBONE_LLM"))
            ]

        super().__init__(
            name=name,
            role=role,
            reasoning_type=reasoning_type,
            llm=llm,
            actions=actions, 
            manager=manager, 
            max_exec_steps=20,  # Increased for more complex workflows
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
        """Generate the next action based on current task and action history."""
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
        """Parse LLM output into executable action."""
        action_name, args, PARSE_FLAG = parse_action(raw_action)
        
        # Handle special parameter parsing for ReasoningPackage objects and direct responses
        if 'reasoning_result' in args:
            if args['reasoning_result'] is not None:
                for _, p_obs in reversed(action_chain):
                    if type(p_obs) == ReasoningPackage and p_obs.get_id() == args['reasoning_result']:
                        args["reasoning_result"] = p_obs
                        break
                    # Also check for direct OpenScholar responses (strings)
                    elif isinstance(p_obs, str) and str(p_obs) == args['reasoning_result']:
                        args["reasoning_result"] = p_obs
                        break
        
        # Handle special parameter parsing for LiteratureCollection objects
        if 'literature_collection' in args:
            if args['literature_collection'] is not None:
                for _, p_obs in reversed(action_chain):
                    if type(p_obs) == LiteratureCollection and p_obs.get_id() == args['literature_collection']:
                        args["literature_collection"] = p_obs
        
        agent_act = AgentAct(name=action_name, params=args)
        return agent_act
    
    def forward(self, task: TaskPackage, agent_act: AgentAct):
        """Execute the action and return observation."""
        act_found_flag = False
        param_parse_flag = False
        
        # Match action to available actions
        for action in self.actions:
            if act_match(agent_act.name, action):
                act_found_flag = True
                try:
                    observation = action(**agent_act.params)
                except Exception as e:
                    # print(f"Action execution error: {e}")
                    observation = (MISS_ACTION_PARAM.format(param_doc=action.params_doc, failed_param=agent_act.params))
                    return observation
                
                # Handle Finish action
                if agent_act.name == ReasonFinishAct.action_name:
                    task.answer = observation
                    task.completion = "completed"
                    
            if action.action_name in agent_act.name:
                param_parse_flag = True
        
        if act_found_flag:
            return observation
        if param_parse_flag:
            return WRONG_ACTION_PARAM
        return ACTION_NOT_FOUND_MESS

    def __add_inner_actions__(self):
        """Add inner action types based on reasoning type."""
        if self.reasoning_type == "react":
            self.actions += [ThinkAct, ReasonFinishAct]
        self.actions = list(set(self.actions))


if __name__ == "__main__":
    user_query = "Which of these genes ['ZNF763', 'FIBP', 'GMCL2', 'PPIA', 'PRKCI'] in ionocyte is the strongest candidate in terms of target specificity to the cell type as a therapeutic target for IBD? Find the best one by comparing to a reference embedding of IBD targets using cosine similarity."
    hypothesis = None
    
    llm_config_dict = {"temperature": 0.5}
    llm_config = LLMConfig(llm_config_dict)
    llm = AgentLLM(llm_config=llm_config)
    
    task_dict = str({"user_query": user_query, "hypothesis": hypothesis})
    task = TaskPackage(instruction=task_dict)
    agent = LiteratureReasoning(llm=llm)
    response = agent(task)
    print(response)
    
