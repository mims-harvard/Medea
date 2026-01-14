"""
Academic Literature Search API

A comprehensive toolkit for searching, filtering, and analyzing academic literature
using semantic search, LLM-based relevance assessment, and intelligent keyword extraction.

Key Components:
- KeywordExtractor: Intelligent query generation for academic databases
- LLMPaperJudge: LLM-powered relevance assessment with explanations
- SemanticScholarSearch: Search interface for Semantic Scholar API
- OpenScholarReasoning: End-to-end literature analysis pipeline
"""

import os
import sys
import warnings
import time
import re
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Iterable, Tuple, Optional

import requests
from bs4 import BeautifulSoup

# FlagEmbedding may download models on first import - this can take time
print("[INFO] Loading FlagEmbedding library (may download reranker models)...", flush=True)
from FlagEmbedding import FlagReranker
print("[INFO] FlagEmbedding loaded successfully", flush=True)

from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .gpt_utils import chat_completion
from .open_scholar import OpenScholar, search_paper_via_query
from . import instructions


# =============================================================================
# KEYWORD EXTRACTION
# =============================================================================

class KeywordExtractor:
    """Intelligent keyword extraction for academic literature search."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def extract_keywords(
        self, 
        question: str, 
        model_name: str = 'gpt-4o',
        query_num: int = 4,
        platform: str = 'semantic_scholar',
        max_retries: int = 3
    ) -> List[str]:
        """
        Generate optimized search keywords for academic literature retrieval.
        
        Args:
            question: The research question to generate keywords for
            model_name: The LLM model to use for keyword generation
            query_num: Number of queries to return
            platform: Target platform ('semantic_scholar', 'openalex', or 'auto')
            max_retries: Maximum retries for API calls
        
        Returns:
            List of optimized search queries
        """
        if self.verbose:
            print(f"[KEYWORD_GEN] Generating {query_num} keywords for {platform} using {model_name}")
        
        enhanced_question = self._enhance_question_for_platform(question, platform)
        
        for attempt in range(max_retries):
            try:
                keywords = chat_completion(
                    instructions.keyword_extraction_prompt.format_map({"question": enhanced_question}), 
                    model=model_name
                )
                
                queries = self._parse_keyword_response(keywords, query_num)
                
                if queries and len(queries) >= min(query_num, 2):
                    if self.verbose:
                        print(f"[KEYWORD_GEN] Generated {len(queries)} keywords")
                        print(f"[KEYWORD_GEN] Keywords: {queries}")
                    return queries
                else:
                    if self.verbose:
                        print(f"[KEYWORD_GEN] WARNING: Insufficient keywords on attempt {attempt + 1}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"[KEYWORD_GEN] ERROR: Attempt {attempt + 1} failed - {e}")
                
            if attempt < max_retries - 1:
                if self.verbose:
                    print("[KEYWORD_GEN] Retrying keyword generation...")
                time.sleep(1)
            
        if self.verbose:
            print("[KEYWORD_GEN] WARNING: All attempts failed, using fallback extraction")
        return self._extract_fallback_keywords(question, query_num, model_name)
    
    def _enhance_question_for_platform(self, question: str, platform: str) -> str:
        """Enhance question with platform-specific optimization."""
        if platform == 'openalex':
            return f"For OpenAlex academic database search: {question}"
        elif platform == 'semantic_scholar':
            return f"For Semantic Scholar academic search: {question}"
        return question
    
    def _parse_keyword_response(self, response: str, target_num: int = 4) -> List[str]:
        """Parse the LLM response to extract search keywords with multiple strategies."""
        if not response:
            return []
        
        # Strategy 1: Look for [Response_Start] and [Response_End] markers
        if "[Response_Start]" in response and "[Response_End]" in response:
            start_idx = response.find("[Response_Start]") + len("[Response_Start]")
            end_idx = response.find("[Response_End]")
            keywords_text = response[start_idx:end_idx].strip()
        else:
            # Strategy 2: Look for comma-separated terms
            lines = response.strip().split('\n')
            keywords_text = ""
            for line in reversed(lines):
                if ',' in line and len(line.split(',')) >= 2:
                    keywords_text = line.strip()
                    break
            
            # Strategy 3: Use the entire response
            if not keywords_text:
                keywords_text = response.strip()
        
        # Clean and split keywords
        queries = []
        if keywords_text:
            raw_queries = keywords_text.split(',')
            for query in raw_queries:
                cleaned = self._clean_keyword_query(query)
                if cleaned and len(cleaned.split()) <= 8:
                    queries.append(cleaned)
        
        return queries[:target_num]
    
    def _clean_keyword_query(self, query: str) -> Optional[str]:
        """Clean and normalize a single keyword query."""
        query = query.strip()
    
        # Remove common artifacts
        artifacts = ["[Response_Start]", "[Response_End]", "Search queries:", "Query:"]
        for artifact in artifacts:
            query = query.replace(artifact, "")
    
        # Remove leading numbers and bullets
        query = re.sub(r'^\d+\.\s*', '', query)
        query = re.sub(r'^[-•*]\s*', '', query)
    
        # Normalize whitespace
        query = ' '.join(query.split())
    
        return query if query else None

    def _extract_fallback_keywords(self, question: str, num_queries: int = 4, model_name: str = 'gpt-4o') -> List[str]:
        """Intelligent fallback keyword extraction using LLM-powered analysis."""
        if self.verbose:
            print("[KEYWORD_GEN] Using intelligent fallback extraction")
        
        try:
            # First attempt: Simple LLM call
            fallback_prompt = f"""
Extract {num_queries} search queries from this question for academic literature search.
Focus on key concepts, entities, and terms that would be most effective for finding relevant papers.

Question: {question}

Return {num_queries} search queries, one per line, without numbering or bullets:
"""
            
            response = chat_completion(fallback_prompt, model=model_name)
            
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            queries = []
            
            for line in lines[:num_queries]:
                cleaned = line
                cleaned = re.sub(r'^\d+\.?\s*', '', cleaned)  # Remove numbering
                cleaned = re.sub(r'^[-•*]\s*', '', cleaned)    # Remove bullets
                cleaned = cleaned.strip('"\'')                  # Remove quotes
                
                if cleaned and len(cleaned.split()) >= 2:
                    queries.append(cleaned)
            
            if len(queries) >= 2:
                if self.verbose:
                    print(f"[KEYWORD_GEN] LLM fallback generated {len(queries)} queries")
                return queries[:num_queries]
                
        except Exception as e:
            if self.verbose:
                print(f"[KEYWORD_GEN] ERROR: LLM fallback failed: {e}")
        
        # Rule-based fallback
        return self._rule_based_keyword_extraction(question, num_queries, model_name)
    
    def _rule_based_keyword_extraction(self, question: str, num_queries: int = 4, model_name: str = 'gpt-4o') -> List[str]:
        """
        LLM-powered keyword extraction as ultimate fallback.
        Uses intelligent analysis without any hardcoded patterns.
        """
        if self.verbose:
            print("[KEYWORD_GEN] Using LLM-powered rule-based extraction")
        
        try:
            # Use LLM to intelligently analyze and extract keywords
            analysis_prompt = f"""
Analyze this research question and extract {num_queries} distinct search queries for academic literature databases.

Research Question: {question}

Instructions:
1. Identify the most important concepts, entities, and technical terms
2. Generate {num_queries} different search queries that would find relevant academic papers
3. Each query should focus on different aspects: specific terms, broader concepts, methodological approaches, related fields
4. Avoid common stop words and focus on meaningful academic terminology
5. Keep queries concise (2-6 words each)

Return exactly {num_queries} search queries, one per line, without numbering or formatting:
"""
            
            response = chat_completion(analysis_prompt, model=model_name)
            
            # Parse and clean the response
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            queries = []
            
            for line in lines[:num_queries]:
                # Clean formatting artifacts
                cleaned = line
                cleaned = re.sub(r'^\d+\.?\s*', '', cleaned)  # Remove numbering
                cleaned = re.sub(r'^[-•*]\s*', '', cleaned)    # Remove bullets
                cleaned = cleaned.strip('"\'()[]{}')           # Remove wrapper chars
                cleaned = ' '.join(cleaned.split())            # Normalize whitespace
                
                if cleaned and len(cleaned.split()) >= 2 and len(cleaned) >= 8:
                    queries.append(cleaned)
            
            if len(queries) >= 2:
                if self.verbose:
                    print(f"[KEYWORD_GEN] LLM rule-based generated {len(queries)} queries")
                return queries[:num_queries]
                
        except Exception as e:
            if self.verbose:
                print(f"[KEYWORD_GEN] ERROR: LLM rule-based extraction failed: {e}")
        
        # Text-based fallback only if LLM completely fails
        return self._text_analysis_fallback(question, num_queries)
    
    def _text_analysis_fallback(self, question: str, num_queries: int = 4) -> List[str]:
        """
        Pure text analysis fallback using statistical methods only.
        No hardcoded patterns - purely data-driven.
        """
        if self.verbose:
            print("[KEYWORD_GEN] Using statistical text analysis fallback")
        
        # Tokenize
        words = re.findall(r'\b\w+\b', question.lower())
        word_freq = Counter(words)
        total_words = len(words)
        
        if not words:
            return [question[:50]]
        
        # Statistical analysis for stop word detection
        # Words that are very short or very frequent are likely stop words
        stop_words = set()
        for word, freq in word_freq.items():
            # Statistical thresholds based on distribution
            if (len(word) <= 2 or 
                freq / total_words > 0.25 or  # Very frequent
                freq > total_words * 0.3):    # Appears too often
                stop_words.add(word)
        
        # Extract meaningful terms using statistical measures
        meaningful_terms = []
        
        # Terms with intermediate frequency (not too rare, not too common)
        for word, freq in word_freq.items():
            if (word not in stop_words and 
                len(word) > 3 and
                0.1 <= freq / total_words <= 0.8):  # Sweet spot frequency
                meaningful_terms.append(word)
        
        # Add unique terms (appear only once but are long enough)
        unique_terms = [word for word, freq in word_freq.items() 
                       if freq == 1 and len(word) > 4 and word not in stop_words]
        meaningful_terms.extend(unique_terms)
        
        # Add capitalized words from original question (likely entities)
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]+\b', question)
        meaningful_terms.extend([term.lower() for term in capitalized])
        
        # Remove duplicates and sort by relevance
        meaningful_terms = list(set(meaningful_terms))
        
        # Create combinations
        queries = []
        if len(meaningful_terms) >= 2:
            # Strategy 1: First few meaningful terms
            queries.append(' '.join(meaningful_terms[:3]))
            
            # Strategy 2: Mix different parts
            if len(meaningful_terms) >= 4:
                queries.append(' '.join(meaningful_terms[1:4]))
            
            # Strategy 3: Longer combinations for specificity
            if len(meaningful_terms) >= 6:
                queries.append(' '.join(meaningful_terms[2:5]))
            
            # Strategy 4: Alternative combination
            if len(meaningful_terms) >= 5:
                queries.append(' '.join([meaningful_terms[0], meaningful_terms[-1], meaningful_terms[len(meaningful_terms)//2]]))
        
        # Clean and validate
        final_queries = []
        seen = set()
        
        for query in queries:
            cleaned = ' '.join(query.split())
            if (cleaned and cleaned not in seen and 
                len(cleaned.split()) >= 2 and len(cleaned) >= 8):
                final_queries.append(cleaned)
                seen.add(cleaned)
        
        # If we still don't have enough, use sentence chunks
        if len(final_queries) < num_queries:
            # Split by punctuation and use meaningful chunks
            chunks = re.split(r'[.!?;,]', question)
            for chunk in chunks:
                chunk_words = [w for w in chunk.lower().split() if len(w) > 3]
                if len(chunk_words) >= 2:
                    chunk_query = ' '.join(chunk_words[:4])
                    if chunk_query not in seen and len(chunk_query) >= 8:
                        final_queries.append(chunk_query)
                        seen.add(chunk_query)
                        if len(final_queries) >= num_queries:
                            break
        
        # Ultimate fallback
        if not final_queries:
            final_queries = [question[:50]]
        
        result = final_queries[:num_queries]
        if self.verbose:
            print(f"[KEYWORD_GEN] Statistical analysis generated {len(result)} queries")
        return result
    



# =============================================================================
# PAPER RELEVANCE ASSESSMENT
# =============================================================================

class LLMPaperJudge:
    """LLM-powered paper relevance assessment with detailed explanations."""
    
    def __init__(self):
        pass
    
    def judge_paper(
        self,
        query: str,
        paper_entity: Dict[str, Any],
        model_name: str,
        max_retries: int = 3,
        backoff_sec: float = 1.0,
        verbose: bool = True,
    ) -> Tuple[bool, str]:
        """
        Assess paper relevance and return decision with explanation.
    
    Args:
        query: The user's research question
            paper_entity: Dictionary containing paper metadata
            model_name: Name of the LLM model to use
            max_retries: Maximum number of retry attempts
            backoff_sec: Initial backoff time for exponential backoff
            
        Returns:
            Tuple of (relevance_decision, explanation_100_chars)
        """
        # Validate inputs
        if not self._validate_inputs(query, paper_entity, verbose):
            return False, "Invalid input data provided"
        
        # Prepare clean paper data
        try:
            clean_paper, paper_json_str = self._prepare_paper_data(paper_entity)
        except Exception as e:
            if verbose:
                print(f"[JUDGE_PAPER] ERROR: Error preparing paper data: {e}")
            return False, f"Data preparation error: {str(e)[:50]}"
        
        # Create enhanced prompt
        enhanced_prompt = self._create_enhanced_prompt(query, paper_json_str)
        
        # Retry logic with exponential backoff
        for attempt in range(1, max_retries + 1):
            try:
                # print(f"[JUDGE_PAPER] Attempt {attempt}/{max_retries}")
                
                response = chat_completion(enhanced_prompt, model=model_name).strip()
                decision, explanation = self._parse_judge_response(response)
                
                if decision is not None:
                    if verbose:
                        paper_title = clean_paper.get('title', 'Unknown')[:50]
                        print(f"[JUDGE_PAPER] Decision: {decision}")
                        print(f"[JUDGE_PAPER] Explanation: {explanation}")
                        print(f"[JUDGE_PAPER] Paper: '{paper_title}...'")
                    return decision, explanation
                
                if verbose:
                    print(f"[JUDGE_PAPER] WARNING: Could not parse response on attempt {attempt}")
                
                if attempt < max_retries:
                    wait_time = backoff_sec * (2 ** (attempt - 1))
                    time.sleep(wait_time)
                    
            except Exception as e:
                if verbose:
                    print(f"[JUDGE_PAPER] ERROR: Attempt {attempt} failed: {e}")
                
                if attempt < max_retries:
                    wait_time = backoff_sec * (2 ** (attempt - 1))
                    time.sleep(wait_time)
        
        # All retries failed
        paper_title = clean_paper.get('title', 'Unknown paper')[:40]
        default_explanation = f"Failed assessment after {max_retries} attempts - defaulting to reject"
        
        if verbose:
            print(f"[JUDGE_PAPER] WARNING: All {max_retries} attempts failed, defaulting to False")
        return False, default_explanation
    
    def _validate_inputs(self, query: str, paper_entity: Dict[str, Any], verbose: bool = True) -> bool:
        """Validate input parameters."""
        if not query or not isinstance(query, str):
            if verbose:
                print(f"[JUDGE_PAPER] ERROR: Invalid query: {query}")
            return False
        
        if not paper_entity or not isinstance(paper_entity, dict):
            if verbose:
                print(f"[JUDGE_PAPER] ERROR: Invalid paper entity: {paper_entity}")
            return False
        
        return True
    
    def _prepare_paper_data(self, paper_entity: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        Prepare clean paper data for the prompt, including main text if available.
        
        Includes: title, abstract, main text/passages, year, authors, url
        """
        clean_paper = {
            "title": (paper_entity.get("title") or "").strip(),
            "abstract": (paper_entity.get("abstract") or "").strip(),
            "year": paper_entity.get("year", ""),
            "authors": paper_entity.get("authors", []),
            "url": paper_entity.get("url", "")
        }
        
        # Include main text/passages if available (for more comprehensive assessment)
        # Check various possible fields for text content
        text_fields = ['text', 'full_text', 'passages', 'content', 'body']
        for field in text_fields:
            if field in paper_entity and paper_entity[field]:
                content = paper_entity[field]
                
                # Handle list of passages vs single text
                if isinstance(content, list):
                    # Take first few passages (intro) and last few (conclusion)
                    if len(content) > 10:
                        # Take first 5 and last 5 paragraphs
                        selected = content[:5] + content[-5:]
                        clean_paper['key_sections'] = ' '.join(
                            p if isinstance(p, str) else p.get('text', '') 
                            for p in selected
                        )[:3000]  # Limit to 3000 chars
                    else:
                        clean_paper['main_text'] = ' '.join(
                            p if isinstance(p, str) else p.get('text', '')
                            for p in content
                        )[:3000]  # Limit to 3000 chars
                elif isinstance(content, str):
                    # Truncate long text but keep beginning and end
                    if len(content) > 3000:
                        clean_paper['main_text'] = content[:1500] + "\n...\n" + content[-1500:]
                    else:
                        clean_paper['main_text'] = content
                break  # Only use first available text field
        
        # Remove empty fields
        clean_paper = {k: v for k, v in clean_paper.items() if v}
        paper_json_str = json.dumps(clean_paper, indent=2)
        
        return clean_paper, paper_json_str
    
    def _create_enhanced_prompt(self, query: str, paper_json_str: str) -> str:
        """Create enhanced prompt for detailed feedback."""
        return f"""
{instructions.relevance_judge_prompt.format_map({
            "query": query.strip(),
            "paper_json": paper_json_str
})}

Additionally, provide a brief explanation (approximately 100 characters) for your decision.

Response format:
Decision: [True/False]
Explanation: [Your ~100 character explanation]
"""
    
    def _parse_judge_response(self, response: str) -> Tuple[Optional[bool], str]:
        """Parse the LLM judge response to extract decision and explanation."""
        # Strategy 1: Structured format
        decision_match = re.search(r'Decision:\s*(True|False)', response, re.IGNORECASE)
        explanation_match = re.search(r'Explanation:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        
        if decision_match and explanation_match:
            decision = decision_match.group(1).lower() == 'true'
            explanation = explanation_match.group(1).strip()[:100]
            return decision, explanation
        
        # Strategy 2: Simple True/False with explanation
        simple_match = re.search(r'(True|False)\s*[:\-]?\s*(.+)', response, re.IGNORECASE)
        if simple_match:
            decision = simple_match.group(1).lower() == 'true'
            explanation = simple_match.group(2).strip()[:100]
            return decision, explanation
        
        # Strategy 3: Just True/False
        if response.strip() in {"True", "False"}:
            decision = response.strip() == "True"
            explanation = "Accept paper as relevant" if decision else "Reject paper as not relevant"
            return decision, explanation
        
        # Strategy 4: Contains True or False
        if 'True' in response and 'False' not in response:
            explanation = response.strip()[:100] if response.strip() else "Accept paper as relevant"
            return True, explanation
        elif 'False' in response and 'True' not in response:
            explanation = response.strip()[:100] if response.strip() else "Reject paper as not relevant"
            return False, explanation
        
        return None, f"Could not parse response: {response[:50]}..."


class PaperQueryAligner:
    """Filter papers using parallel LLM relevance assessment."""
    
    def __init__(self, judge: LLMPaperJudge = None):
        self.judge = judge or LLMPaperJudge()
    
    def filter_papers(
        self,
        query: str,
        paper_list: Iterable[Dict[str, Any]],
        model_name: str = "o1-mini-2025-01",
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
            Filter papers in parallel using LLM relevance assessment.
            
            Args:
                query: The user query
                paper_list: Papers to evaluate
                model_name: LLM model identifier
                max_workers: Degree of parallelism
                
            Returns:
                Subset of paper_list deemed relevant
            """
        results = [None] * len(paper_list)

        def _wrapper(idx: int, paper: Dict[str, Any]):
            is_relevant, explanation = self.judge.judge_paper(query, paper, model_name)
            
            # Log the assessment with 100-character explanation
            paper_title = paper.get('title', 'Unknown')[:30]
            status = "ACCEPT" if is_relevant else "REJECT"
            print(f"[PAPER_JUDGE] {status}: {paper_title}... | {explanation}")
            
            return idx, is_relevant

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_wrapper, i, p) for i, p in enumerate(paper_list)]
            for fut in as_completed(futures):
                idx, is_relevant = fut.result()
                results[idx] = is_relevant

        return [paper for paper, flag in zip(paper_list, results) if flag]
    

# =============================================================================
# SEARCH FUNCTIONALITY
# =============================================================================

class SemanticScholarSearch:
    """Search interface for Semantic Scholar API."""
    
    def __init__(self, extractor: KeywordExtractor = None, verbose: bool = True):
        self.extractor = extractor or KeywordExtractor()
        self.verbose = verbose
    
    def search(
        self,
        question: str,
        model_name: str,
        max_paper_num: int = 2,
        attempt: int = 3,
        min_citation_count: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for relevant papers.
        
        Args:
            question: Research question
            model_name: LLM model for keyword generation
            max_paper_num: Maximum papers to retrieve
            attempt: Number of search attempts
            min_citation_count: Minimum citation count filter
            
        Returns:
            List of processed paper dictionaries
        """
        new_keywords = self.extractor.extract_keywords(
            question, 
            model_name=model_name,
            platform='semantic_scholar'
        )
        
        if self.verbose:
            print(f"[SEMANTIC_SCHOLAR] Using keywords: {new_keywords}")
        
        paper_list = {}
        successful_queries = 0
        
        for i, keyword in enumerate(new_keywords):
            if self.verbose:
                print(f"[SEMANTIC_SCHOLAR] Processing keyword {i+1}/{len(new_keywords)}: '{keyword}'")
            
            try:
                top_papers = search_paper_via_query(
                    query=keyword,
                    max_paper_num=max_paper_num,
                    attempt=attempt,
                    minCitationCount=min_citation_count,
                )
                
                if top_papers is None:
                    if self.verbose:
                        print(f"[SEMANTIC_SCHOLAR] WARNING: No papers found for keyword: '{keyword}'")
                    continue
                
                successful_queries += 1
                if self.verbose:
                    print(f"[SEMANTIC_SCHOLAR] Found {len(top_papers)} papers for keyword: '{keyword}'")
                
                for paper in top_papers:
                    if paper["paperId"] not in paper_list:
                        paper["text"] = paper["abstract"]
                        paper["citation_counts"] = paper["citationCount"]
                        paper_list[paper["paperId"]] = paper
            
            except Exception as e:
                if self.verbose:
                    print(f"[SEMANTIC_SCHOLAR] ERROR: Search failed for '{keyword}': {e}")
        
        if successful_queries == 0:
            if self.verbose:
                print("[SEMANTIC_SCHOLAR] ERROR: No successful queries found")
            return []
        
        if self.verbose:
            print(f"[SEMANTIC_SCHOLAR] {successful_queries}/{len(new_keywords)} queries successful, {len(paper_list)} unique papers found")
        
        # Process papers into final format
        final_paper_list = []
        for paper_id in paper_list:
            paper = paper_list[paper_id]
            
            # Add abstract entry
            final_paper_list.append({
                "semantic_scholar_id": paper_id,
                "type": "ss_abstract",
                "year": paper["year"],
                "authors": paper["authors"],
                "title": paper["title"],
                "text": paper["text"],
                "url": paper["url"],
                "citation_counts": paper["citationCount"],
                "abstract": paper["abstract"]
            })
            
            # Add ArXiv passages if available
            if (paper.get("externalIds") is not None and 
                "ArXiv" in paper["externalIds"]):
                try:
                    passages = ArxivPaperProcessor.retrieve_passages(paper["externalIds"]["ArXiv"])
                    for passage in passages:
                        final_paper_list.append({
                            "semantic_scholar_id": paper_id,
                            "type": "ss_passage",
                            "year": paper["year"],
                            "authors": paper["authors"],
                            "title": paper["title"],
                            "text": passage,
                            "url": paper["url"],
                            "citation_counts": paper["citationCount"],
                            "abstract": paper["abstract"]
                        })
                except Exception as e:
                    if self.verbose:
                        print(f"[SEMANTIC_SCHOLAR] WARNING: Failed to retrieve ArXiv passages for {paper_id}: {e}")
        
        return final_paper_list


class ArxivPaperProcessor:
    """Process papers from ArXiv for text extraction."""
    
    @staticmethod
    def retrieve_passages(arxiv_id: str) -> List[str]:
        """Retrieve passages from a single ArXiv paper."""
        ar5iv_link = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        return ArxivPaperProcessor._parse_paragraphs(ar5iv_link)
    
    @staticmethod
    def _parse_paragraphs(link: str) -> List[str]:
        """Parse paragraphs from ArXiv HTML."""
        response = requests.get(link, verify=False)
        time.sleep(0.1)
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract abstract
        raw_abstract = soup.find_all("div", "ltx_abstract")
        try:
            abstract = ''.join(raw_abstract[0].text.split("\n")[2:])
        except:
            abstract = ""
        
        # Extract sections
        subsections = soup.find_all(class_='ltx_para', id=re.compile(r"^S\d+\.+(p|S)"))
        
        paragraphs = []
        for subsection in subsections:
            paragraphs.append(re.sub(r"\n", "", subsection.text))
        
        return paragraphs


# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

class TextProcessor:
    """Utilities for text processing and cleaning."""
    
    @staticmethod
    def remove_citations(text: str) -> str:
        """Remove citation markers from text."""
        text = re.sub(r"\[\d+", "", text)
        text = re.sub(r" \[\d+", "", text)
        return text.replace(" |", "").replace("]", "")
    
    @staticmethod
    def process_paragraph(text: str) -> str:
        """Clean and process paragraph text."""
        text = text.replace("<cit.>", "")
        return TextProcessor.remove_citations(text)

    @staticmethod
    def process_input_data(data: List[Dict], use_contexts: bool = False) -> List[Dict]:
        """Process input data for OpenScholar reasoning."""
        processed_data = []
        
        for item in data:
            # Ensure required fields
            if "answer" not in item:
                item["answer"] = ""
            if "input" not in item:
                if "question" in item:
                    item["input"] = item["question"]
                elif "query" in item:
                    item["input"] = item["query"]

            if use_contexts:
                item = TextProcessor._process_contexts(item)
            
            processed_data.append(item)
        
        return processed_data
    
    @staticmethod
    def _process_contexts(item: Dict) -> Dict:
        """Process context data for an item."""
        new_ctxs = []
        
        # Normalize context format
        for ctx in item.get("ctxs", []):
            if isinstance(ctx, list):
                new_ctxs.extend([c for c in ctx if isinstance(c, dict)])
            elif isinstance(ctx, dict):
                new_ctxs.append(ctx)
        
        item["ctxs"] = new_ctxs

        # Remove duplicated contexts
        processed_paras = []
        
        with tqdm(total=len(item["ctxs"])) as pbar:
            for ctx in item["ctxs"]:
                pbar.update(1)
                
                if "retrieval text" in ctx:
                    ctx["text"] = ctx["retrieval text"]
                
                if not ctx.get("text"):
                    continue
                    
                if not isinstance(ctx["text"], str):
                    ctx["text"] = " ".join(ctx["text"]["contexts"])
                
                ctx["text"] = TextProcessor.process_paragraph(ctx["text"])
                
                if "title" not in ctx:
                    ctx["title"] = ""
                
                processed_paras.append(ctx)

        # Deduplicate by text + title
        processed_paras_dict = {
            paper["text"][:100] + paper["title"]: paper 
            for paper in processed_paras
        }
        
        item["ctxs"] = list(processed_paras_dict.values())
        item["original_ctxs"] = item["ctxs"]
        
        return item


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

class OpenScholarReasoning:
    """End-to-end literature analysis pipeline."""
    
    def __init__(self):
        self.extractor = KeywordExtractor()
        self.judge = LLMPaperJudge()
        self.aligner = PaperQueryAligner(self.judge)
        self.searcher = SemanticScholarSearch(self.extractor)
    
    def reason(
        self,
        query: str,
        model_name: str = "gpt-4o",
        default_reranker: str = "OpenSciLM/OpenScholar_Reranker",
        **kwargs
    ) -> Tuple[str, Optional[Any]]:
        """
        Perform end-to-end literature reasoning.
        
        Args:
            query: Research question
            model_name: LLM model name
            default_reranker: Reranker model name
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (answer, metadata)
        """
        # Search for papers
        ss_retrieved_passages = self.searcher.search(
            question=query, 
            model_name=model_name,
            max_paper_num=kwargs.get('max_paper_num', 5),
            min_citation_count=kwargs.get('min_citation', 0)
        )

        # Filter relevant papers with adaptive approach
        relevant_papers = self.aligner.filter_papers(
            query=query,
            paper_list=ss_retrieved_passages,
            model_name=model_name,
        )

        # Adaptive fallback: if no papers found, try more relaxed approach for complex queries
        if not relevant_papers and ss_retrieved_passages:
            print(f"[ADAPTIVE_FILTER] No papers passed initial filter. Trying exploratory approach for {len(ss_retrieved_passages)} papers...")
            
            # Create a more exploratory prompt for complex/novel queries
            exploratory_aligner = PaperQueryAligner(judge=LLMPaperJudge())
            
            # Use a special flag to indicate exploratory mode 
            exploratory_query = f"[EXPLORATORY_MODE] {query}"
            relevant_papers = exploratory_aligner.filter_papers(
                query=exploratory_query,
                paper_list=ss_retrieved_passages,
                model_name=model_name,
            )
            
            if relevant_papers:
                print(f"[ADAPTIVE_FILTER] Exploratory approach found {len(relevant_papers)} potentially relevant papers")
            else:
                print(f"[ADAPTIVE_FILTER] Even exploratory approach found no relevant papers")

        if not relevant_papers:
            return "ABSTAIN — no relevant scientific literature available for this query", None
        
        # Process data
        reason_dict = [{'input': query, 'ctxs': relevant_papers}]
        
        if not relevant_papers:
            kwargs['ranking_ce'] = False
            kwargs['use_contexts'] = False
            kwargs['max_per_paper'] = None
        
        # Initialize reranker and OpenScholar
        print(f"[OpenScholarReasoning] Initializing reranker: {default_reranker}", flush=True)
        print(f"[OpenScholarReasoning] First-time use may download model files (~1GB)...", flush=True)
        
        reranker = FlagReranker(
            default_reranker, 
            use_fp16=True
        )
        
        print(f"[OpenScholarReasoning] Reranker initialized successfully", flush=True)
        
        open_scholar = OpenScholar(
            model=kwargs.get('model'),
            tokenizer=kwargs.get('tokenizer'),
            client_llm=model_name, 
            reranker=reranker,
            use_reranker=True,
            query_type="citation_qa",
        )
        
        # Process and generate response
        final_results = open_scholar.generate(
            reason_dict,
            top_n=kwargs.get('top_n', 4),
            min_citation=kwargs.get('min_citation', 4),
            norm_cite=kwargs.get('norm_cite', True),
            ss_retriever=kwargs.get('ss_retriever', True),
            use_contexts=kwargs.get('use_contexts', True),
            ranking_ce=kwargs.get('ranking_ce', True),
            feedback=kwargs.get('feedback', True),
            skip_generation=kwargs.get('skip_generation', False),
            posthoc_at=kwargs.get('posthoc_at', True),
            zero_shot=kwargs.get('zero_shot', True),
            use_abstract=kwargs.get('use_abstract', True),
            max_per_paper=kwargs.get('max_per_paper', 3),
            max_tokens=kwargs.get('max_tokens', 2000),
            task_name=kwargs.get('task_name', "default"),
        )
        
        return final_results[0], None


# =============================================================================
# LEGACY FUNCTION COMPATIBILITY
# =============================================================================

# Initialize global instances for backward compatibility
_extractor = KeywordExtractor()
_judge = LLMPaperJudge()
_aligner = PaperQueryAligner(_judge)
_searcher = SemanticScholarSearch(_extractor)
_reasoning = OpenScholarReasoning()

# Legacy function wrappers
def retrieve_keywords(question, model_name='gpt-4o', query_num=4, platform='semantic_scholar', max_retries=3, verbose=True):
    """Legacy wrapper for keyword extraction."""
    # Create a temporary extractor with the specified verbose setting
    extractor = KeywordExtractor(verbose=verbose)
    return extractor.extract_keywords(question, model_name, query_num, platform, max_retries)

def search_semantic_scholar(question, model_name, max_paper_num=2, attempt=3, minCitationCount=5):
    """Legacy wrapper for Semantic Scholar search."""
    papers = _searcher.search(question, model_name, max_paper_num, attempt, minCitationCount)
    keywords = _extractor.extract_keywords(question, model_name, platform='semantic_scholar')
    return papers, keywords

def paper_query_aligner(query, paper_list, model_name="o1-mini-2025-01", max_workers=4):
    """Legacy wrapper for paper filtering."""
    return _aligner.filter_papers(query, paper_list, model_name, max_workers)

def retrieve_passages_single_paper(arxiv_id):
    """Legacy wrapper for ArXiv paper processing."""
    return ArxivPaperProcessor.retrieve_passages(arxiv_id)

def parsing_paragraph(link):
    """Legacy wrapper for paragraph parsing."""
    return ArxivPaperProcessor._parse_paragraphs(link)

def remove_citations(sent):
    """Legacy wrapper for citation removal."""
    return TextProcessor.remove_citations(sent)

def process_paragraph(text):
    """Legacy wrapper for paragraph processing."""
    return TextProcessor.process_paragraph(text)

def process_input_data(data, use_contexts=False):
    """Legacy wrapper for input data processing."""
    return TextProcessor.process_input_data(data, use_contexts)

def openscholar_reasoning(query, **kwargs):
    """Legacy wrapper for OpenScholar reasoning."""
    return _reasoning.reason(query, **kwargs)