import requests
import urllib.parse
from typing import Dict, List, Optional, Union, ClassVar, Any

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OpenAlexTool:
    """
    Tool to retrieve literature from OpenAlex based on search keywords.
    
    This class provides methods to search for academic papers using the OpenAlex API
    with title and abstract search functionality.
    """
    
    BASE_URL: ClassVar[str] = "https://api.openalex.org/works"
    DEFAULT_EMAIL: ClassVar[str] = "research@example.com"
    MAX_PER_PAGE: ClassVar[int] = 200
    DEFAULT_MAX_RESULTS: ClassVar[int] = 5
    
    def __init__(self, email: str = DEFAULT_EMAIL):
        """
        Initialize the OpenAlex tool.
        
        Args:
            email: Email address for polite pool access (optional)
        """
        self.base_url = self.BASE_URL
        self.email = email
        
    def run(self, arguments: Dict[str, Any]) -> Union[List[Dict[str, Any]], str]:
        """
        Main entry point for the tool.
        
        Args:
            arguments: Dictionary containing search parameters
            
        Returns:
            List of paper dictionaries or error message
        """
        search_keywords = arguments.get("search_keywords")
        if not search_keywords:
            return "Error: search_keywords parameter is required"
            
        max_results = arguments.get("max_results", self.DEFAULT_MAX_RESULTS)
        year_from = arguments.get("year_from")
        year_to = arguments.get("year_to")
        open_access = arguments.get("open_access")
        
        return self.search_literature(search_keywords, max_results, year_from, year_to, open_access)
    
    def search_literature(
        self, 
        search_keywords: str, 
        max_results: int = DEFAULT_MAX_RESULTS, 
        year_from: Optional[int] = None, 
        year_to: Optional[int] = None, 
        open_access: Optional[bool] = None,
        verbose: bool = True
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Search for literature using OpenAlex API with title and abstract search.
        
        Args:
            search_keywords: Keywords to search for in title and abstract
            max_results: Maximum number of results to return (default: 10)
            year_from: Start year for publication date filter (optional)
            year_to: End year for publication date filter (optional)
            open_access: Filter for open access papers only (optional)
            
        Returns:
            List of dictionaries containing paper information or error message
        """
        if not search_keywords or not search_keywords.strip():
            return "Error: search_keywords cannot be empty"
            
        try:
            params = self._build_search_params(search_keywords, max_results, year_from, year_to, open_access)
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response with error handling
            data = response.json()
            if not data or not isinstance(data, dict):
                return "Error: Invalid response format from OpenAlex API"
            
            results = data.get('results', [])
            if not results:
                return f"No results found for search keywords: '{search_keywords}'"
            
            # Extract paper info with error handling
            papers = []
            for work in results:
                if work and isinstance(work, dict):  # Ensure work is not None and is a dict
                    try:
                        paper_info = self._extract_paper_info(work)
                        if paper_info:  # Only add if extraction was successful
                            papers.append(paper_info)
                    except Exception as e:
                        print(f"[OpenAlex] Warning: Failed to extract paper info: {e}")
                        continue
            
            if not papers:
                return f"Error: No valid papers could be extracted from search results"
            
            if verbose:
                print(f"[OpenAlex] Retrieved {len(papers)} papers for keywords: '{search_keywords}'")
            return papers
            
        except requests.exceptions.Timeout:
            return "Error: Request timeout - OpenAlex API did not respond in time"
        except requests.exceptions.RequestException as e:
            return f"Error retrieving data from OpenAlex: {e}"
        except Exception as e:
            return f"Unexpected error during search: {e}"
    
    def _build_search_params(
        self, 
        search_keywords: str, 
        max_results: int, 
        year_from: Optional[int], 
        year_to: Optional[int], 
        open_access: Optional[bool]
    ) -> Dict[str, str]:
        """Build query parameters for OpenAlex API request with title and abstract search."""
        # Clean and prepare keywords
        keywords = search_keywords.strip()
        
        # Build search query - use the filter approach for title_and_abstract.search
        params = {
            'filter': f'title_and_abstract.search:{keywords}',
            'per-page': str(min(max_results, self.MAX_PER_PAGE)),
            'sort': 'relevance_score:desc',  # Sort by OpenAlex relevance score
            'mailto': self.email  # Required for OpenAlex API access
        }
        
        # Build additional filters
        filters = [f'title_and_abstract.search:{keywords}']
        
        # Add year filters
        if year_from is not None and year_to is not None:
            filters.append(f"publication_year:{year_from}-{year_to}")
        elif year_from is not None:
            filters.append(f"publication_year:>{year_from-1}")
        elif year_to is not None:
            filters.append(f"publication_year:<{year_to+1}")
            
        # Add open access filter
        if open_access is True:
            filters.append("is_oa:true")
        elif open_access is False:
            filters.append("is_oa:false")
            
        # Combine all filters
        params['filter'] = ','.join(filters)
            
        return params
    
    def _extract_paper_info(self, work: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant information from a work object returned by OpenAlex API.
        
        Args:
            work: Work object from OpenAlex API response
            
        Returns:
            Formatted paper information dictionary matching semantic scholar format or None if authors are empty
        """
        if not work or not isinstance(work, dict):
            return None
            
        try:
            # Extract basic information with safe defaults
            title = work.get('title', 'No title available') or 'No title available'
            publication_year = work.get('publication_year', 'Year not available') or 'Year not available'
            
            # Extract and reconstruct abstract
            abstract = self._reconstruct_abstract(work.get('abstract_inverted_index'))
            
            # Extract authors in semantic scholar format
            authors = self._extract_authors_semantic_scholar_format(work.get('authorships', []) or [])
            
            # Filter: Skip papers with no authors
            if not authors or len(authors) == 0:
                return None
            
            # Extract publication details
            primary_location = work.get('primary_location') or {}
            venue = 'Unknown venue'
            if primary_location and isinstance(primary_location, dict):
                source = primary_location.get('source') or {}
                if isinstance(source, dict):
                    venue = source.get('display_name', 'Unknown venue') or 'Unknown venue'
            
            doi = work.get('doi', 'No DOI') or 'No DOI'
            citation_count = work.get('cited_by_count', 0) or 0
            
            # Extract open access information
            open_access_info = work.get('open_access') or {}
            is_open_access = False
            pdf_url = None
            if isinstance(open_access_info, dict):
                is_open_access = open_access_info.get('is_oa', False) or False
                pdf_url = open_access_info.get('oa_url')
            
            # Build URL - prefer DOI, fallback to OpenAlex ID
            url = doi if doi != 'No DOI' else work.get('id', '') or ''
            openalex_id = work.get('id', '') or ''
            
            # Get OpenAlex relevance score if available
            relevance_score = work.get('relevance_score', 0.0) or 0.0
            
            # Return in semantic scholar compatible format
            return {
                "semantic_scholar_id": openalex_id,  # Use OpenAlex ID as identifier
                "type": "openalex_abstract", 
                "year": publication_year, 
                "authors": authors, 
                "title": title, 
                "text": abstract,  # Abstract as text field
                "url": url, 
                "citation_counts": citation_count,  # Match semantic scholar field name
                "abstract": abstract,  # Also keep separate abstract field
                # Additional OpenAlex-specific fields
                "venue": venue,
                "doi": doi,
                "open_access": is_open_access,
                "pdf_url": pdf_url,
                "openalex_id": openalex_id,
                "relevance_score": relevance_score
            }
        except Exception as e:
            print(f"[OpenAlex] Error extracting paper info: {e}")
            return None
    
    def _reconstruct_abstract(self, abstract_inverted_index: Optional[Dict[str, List[int]]]) -> str:
        """
        Reconstruct abstract from inverted index.
        
        Args:
            abstract_inverted_index: Dictionary mapping words to their positions
            
        Returns:
            Reconstructed abstract text
        """
        if not abstract_inverted_index:
            return "Abstract not available"
            
        try:
            # Find the maximum position to determine array size
            max_position = 0
            for positions in abstract_inverted_index.values():
                if positions:
                    max_position = max(max_position, max(positions))
            
            # Create array with appropriate size
            abstract_words = [''] * (max_position + 1)
            
            # Fill in the words at their positions
            for word, positions in abstract_inverted_index.items():
                for pos in positions:
                    if 0 <= pos < len(abstract_words):
                        abstract_words[pos] = word
            
            # Join words and clean up
            abstract = ' '.join(word for word in abstract_words if word).strip()
            return abstract if abstract else "Abstract not available"
            
        except Exception:
            return "Abstract reconstruction failed"
    
    def _extract_authors_semantic_scholar_format(self, authorships: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract author names from authorships in semantic scholar format."""
        authors = []
        for authorship in authorships:
            author = authorship.get('author', {})
            author_name = author.get('display_name', 'Unknown Author')
            # Match semantic scholar format with authorId and name
            authors.append({
                "authorId": author.get('id', '') or '',  # OpenAlex author ID
                "name": author_name
            })
        return authors
    
    def _extract_authors(self, authorships: List[Dict[str, Any]]) -> List[str]:
        """Extract author names from authorships (legacy method)."""
        authors = []
        for authorship in authorships:
            author = authorship.get('author', {})
            author_name = author.get('display_name', 'Unknown Author')
            authors.append(author_name)
        return authors
    
    def _extract_organizations(self, authorships: List[Dict[str, Any]]) -> List[str]:
        """Extract organization names from authorships."""
        organizations = set()
        for authorship in authorships:
            for institution in authorship.get('institutions', []):
                org_name = institution.get('display_name')
                if org_name:
                    organizations.add(org_name)
        return list(organizations)


def paper_search_from_openalex(
    search_keywords: str, 
    max_results: int = 10, 
    year_from: Optional[int] = None, 
    year_to: Optional[int] = None, 
    open_access: Optional[bool] = None
) -> Union[List[Dict[str, Any]], str]:
    """
    Search for literature using OpenAlex API with title and abstract search.
    
    Args:
        search_keywords: Keywords to search for
        max_results: Maximum number of results to return
        year_from: Start year for publication date filter
        year_to: End year for publication date filter
        open_access: Filter for open access papers only
        
    Returns:
        List of paper dictionaries in semantic scholar format or error message
    """
    openalex = OpenAlexTool()
    return openalex.search_literature(search_keywords, max_results, year_from, year_to, open_access)


def search_openalex_papers(
    question: str, 
    max_results: int = 5, 
    year_from: Optional[int] = None, 
    year_to: Optional[int] = None, 
    open_access: Optional[bool] = None,
    model_name: str = 'gpt-4o',
    verbose: bool = True
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Search for literature using OpenAlex API with intelligent keyword generation.
    
    Args:
        question: Search question/keywords
        max_results: Maximum number of results to return
        year_from: Start year for publication date filter
        year_to: End year for publication date filter
        open_access: Filter for open access papers only
        model_name: Model for keyword generation
        
    Returns:
        Tuple of (paper_list, keywords_used) matching semantic scholar format
    """
    # Import here to avoid circular imports
    try:
        from tool_space.search_api import retrieve_keywords
        
        # Generate optimized keywords for OpenAlex
        if verbose:
            print(f"[SEARCH_OPENALEX] INFO: Generating keywords for question: '{question}'")
        optimized_keywords = retrieve_keywords(
            question, 
            model_name=model_name, 
            platform='openalex',
            query_num=4,
            verbose=verbose
        )
        if verbose:
            print(f"[SEARCH_OPENALEX] INFO: Using keywords: {optimized_keywords}")
        
    except Exception as e:
        if verbose:
            print(f"[SEARCH_OPENALEX] WARNING: Keyword generation failed ({e}), using question directly")
        optimized_keywords = [question.strip()]
    
    openalex = OpenAlexTool()
    all_papers = {}
    successful_queries = 0
    
    # Search with each keyword
    for i, keyword in enumerate(optimized_keywords):
        if verbose:
            print(f"[SEARCH_OPENALEX] DEBUG: Processing keyword {i+1}/{len(optimized_keywords)}: '{keyword}'")
        
        results = openalex.search_literature(keyword, max_results, year_from, year_to, open_access, verbose)
        
        # If error occurred, try next keyword
        if isinstance(results, str):
            if verbose:
                print(f"[SEARCH_OPENALEX] WARNING: No papers found for keyword: '{keyword}' - {results}")
            continue
        
        successful_queries += 1
        if verbose:
            print(f"[SEARCH_OPENALEX] SUCCESS: Found {len(results)} papers for keyword: '{keyword}'")
        
        # Deduplicate by OpenAlex ID
        for paper in results:
            paper_id = paper.get('openalex_id', paper.get('semantic_scholar_id', ''))
            if paper_id and paper_id not in all_papers:
                all_papers[paper_id] = paper
    
    if successful_queries == 0:
        if verbose:
            print(f"[SEARCH_OPENALEX] ERROR: No successful queries found")
        return [], []
    
    # Convert to list and sort by relevance score if available
    final_papers = list(all_papers.values())
    try:
        final_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    except:
        pass  # Sort failed, continue with unsorted list
    
    # Limit to max_results
    final_papers = final_papers[:max_results]
    
    if verbose:
        print(f"[SEARCH_OPENALEX] SUMMARY: {successful_queries}/{len(optimized_keywords)} queries successful, {len(final_papers)} unique papers found")
    
    return final_papers, optimized_keywords


if __name__ == "__main__":
    # import the root path
    import sys
    import dotenv
    dotenv.load_dotenv('.env')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tool_space.open_alex import search_openalex_papers, paper_search_from_openalex
    
    # Test new semantic scholar compatible function
    print("=== Testing search_openalex_papers (semantic scholar format) ===")
    papers, keywords = search_openalex_papers("MCF7 Breast Cancer", max_results=3)
    print(f"Keywords used: {keywords}")
    print(f"Number of papers: {len(papers)}")
    if papers:
        print(f"First paper format:")
        for key, value in papers[0].items():
            print(f"  {key}: {value}")
    
    print("\n=== Testing paper_search_from_openalex (legacy format) ===")
    result = paper_search_from_openalex("MCF7 Breast Cancer", max_results=2)
    if isinstance(result, list) and result:
        print(f"Number of papers: {len(result)}")
        print(f"First paper keys: {list(result[0].keys())}")
    else:
        print(f"Result: {result}")