import json
import requests
import urllib.parse
import networkx as nx
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter


@dataclass
class InteractionResult:
    """Data class to store genetic interaction prediction results."""
    gene_pair: Tuple[str, str]
    interaction_score: float
    confidence_level: str
    evidence_pathways: List[str]
    predicted_interaction_type: str
    supporting_data: Dict[str, any]


class BaseEnrichrInteractionTool(ABC):
    """Base class for individual Enrichr library interaction tools."""
    
    ENRICHR_URL = "https://maayanlab.cloud/Enrichr/addList"
    ENRICHMENT_URL = "https://maayanlab.cloud/Enrichr/enrich"
    
    def __init__(self, library_name: str, tool_name: str, top_results: int = 5):
        self.library_name = library_name
        self.tool_name = tool_name
        self.top_results = top_results
        self.gene_cache = {}
        self.request_delay = 1.0
        self.max_retries = 3
        
    def _log(self, message: str, level: str = "INSIGHT"):
        """Log insights for reasoning analysis."""
        prefix = f"[{self.tool_name}]"
        print(f"{prefix} {level}: {message}", flush=True)
        
    def _rate_limited_request(self, request_func, *args, **kwargs):
        """Execute request with rate limiting and exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                response = request_func(*args, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = (2 ** attempt) * self.request_delay
                    if attempt == self.max_retries - 1:
                        self._log(f"API rate limit exceeded - analysis may be incomplete", "WARNING")
                        raise Exception(f"Rate limit exceeded after {self.max_retries} attempts")
                    time.sleep(wait_time)
                else:
                    raise e
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Request failed after {self.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
    
    def get_official_gene_name(self, gene_name: str) -> str:
        """Get official gene symbol with insight logging."""
        if gene_name in self.gene_cache:
            return self.gene_cache[gene_name]
            
        encoded_gene_name = urllib.parse.quote(gene_name)
        url = f"https://mygene.info/v3/query?q={encoded_gene_name}&fields=symbol,alias&species=human"
        
        try:
            response = self._rate_limited_request(requests.get, url, timeout=10)
            data = response.json()
            hits = data.get("hits", [])
            
            if not hits:
                self._log(f"Gene '{gene_name}' not found in database - using original name", "WARNING")
                self.gene_cache[gene_name] = gene_name
                return gene_name

            # Find exact match
            for hit in hits:
                symbol = hit.get("symbol", "")
                if symbol.upper() == gene_name.upper():
                    if symbol != gene_name:
                        self._log(f"Gene name standardized: {gene_name} → {symbol}")
                    self.gene_cache[gene_name] = symbol
                    return symbol
                aliases = hit.get("alias", [])
                if any(gene_name.upper() == alias.upper() for alias in aliases):
                    self._log(f"Gene alias resolved: {gene_name} → {symbol}")
                    self.gene_cache[gene_name] = symbol
                    return symbol

            # Use top hit
            top_hit = hits[0]
            symbol = top_hit.get("symbol", gene_name)
            if symbol != gene_name:
                self._log(f"Gene name corrected: {gene_name} → {symbol}")
            self.gene_cache[gene_name] = symbol
            return symbol
            
        except Exception as e:
            self._log(f"Gene name resolution failed for '{gene_name}' - proceeding with original", "WARNING")
            self.gene_cache[gene_name] = gene_name
            return gene_name

    def submit_gene_list(self, genes: List[str]) -> str:
        """Submit gene list to Enrichr."""
        gene_list_str = "\n".join(genes)
        payload = {
            "list": (None, gene_list_str),
            "description": (None, f"Gene pair analysis: {' + '.join(genes)}")
        }
        
        try:
            response = self._rate_limited_request(requests.post, self.ENRICHR_URL, files=payload, timeout=30)
            user_list_id = response.json()["userListId"]
            return user_list_id
        except Exception as e:
            self._log(f"Failed to submit genes to Enrichr: {e}", "ERROR")
            raise Exception(f"Error submitting gene list to Enrichr: {e}")

    def get_enrichment_results(self, user_list_id: str) -> List:
        """Fetch enrichment results."""
        query_string = f"?userListId={user_list_id}&backgroundType={self.library_name}"
        
        try:
            response = self._rate_limited_request(requests.get, self.ENRICHMENT_URL + query_string, timeout=30)
            results = response.json()
            library_results = results.get(self.library_name, [])[:self.top_results]
            return library_results
        except Exception as e:
            self._log(f"Failed to retrieve enrichment data: {e}", "ERROR")
            raise Exception(f"Error fetching enrichment results: {e}")

    @abstractmethod
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair interaction and return standardized 4-element output.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        pass

    def _extract_shared_terms(self, gene1: str, gene2: str, enrichment_results: List) -> Tuple[List[str], float]:
        """Extract shared terms and calculate interaction score with insights."""
        shared_terms = []
        interaction_score = 0.0
        high_significance_terms = []
        
        for result in enrichment_results:
            if len(result) < 6:
                continue
                
            term_name = result[1]
            p_value = result[2]
            associated_genes = result[5]
            
            if gene1 in associated_genes and gene2 in associated_genes:
                shared_terms.append(term_name)
                score_contribution = -np.log10(max(p_value, 1e-10))
                interaction_score += score_contribution
                
                # Log highly significant findings
                if p_value < 0.001:
                    high_significance_terms.append((term_name, p_value))
        
        if high_significance_terms:
            top_term, top_p = high_significance_terms[0]
            self._log(f"Strong evidence found: '{top_term}' (p={top_p:.2e})")
        
        if shared_terms:
            self._log(f"Interaction detected: {len(shared_terms)} shared terms, score={interaction_score:.2f}")
        else:
            self._log("No shared terms detected between gene pair")
        
        return shared_terms, interaction_score

    def _calculate_confidence(self, shared_count: int, interaction_score: float) -> str:
        """Calculate confidence level with reasoning insight."""
        if shared_count == 0:
            return "none"
        
        # Normalized confidence score
        confidence_score = min(1.0, (shared_count * interaction_score) / 50.0)
        
        if confidence_score >= 0.6:
            confidence = "high"
        elif confidence_score >= 0.3:
            confidence = "medium"
        elif confidence_score >= 0.1:
            confidence = "low"
        else:
            confidence = "very_low"
        
        # Log reasoning for confidence level
        factors = []
        if shared_count >= 3:
            factors.append(f"multiple shared terms ({shared_count})")
        if interaction_score >= 10:
            factors.append("high statistical significance")
        if shared_count == 1:
            factors.append("single shared term")
        
        if factors:
            self._log(f"Confidence '{confidence}' based on: {', '.join(factors)}")
        
        return confidence


class WikiPathwaysInteractionTool(BaseEnrichrInteractionTool):
    """Individual tool for WikiPathways-based genetic interaction analysis."""
    
    def __init__(self):
        super().__init__("WikiPathways_2024_Human", "PATHWAY", top_results=5)
        
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair using WikiPathways data.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        self._log(f"Analyzing pathway interactions for {gene1} + {gene2}")
        
        # Resolve gene names
        official_gene1 = self.get_official_gene_name(gene1)
        official_gene2 = self.get_official_gene_name(gene2)
        
        try:
            # Submit genes and get enrichment
            user_list_id = self.submit_gene_list([official_gene1, official_gene2])
            enrichment_results = self.get_enrichment_results(user_list_id)
            
            if not enrichment_results:
                self._log("No enriched pathways found in WikiPathways database")
                return ("No pathway interactions found", "none", [], [])
            
            # Extract shared pathways
            shared_pathways, interaction_score = self._extract_shared_terms(official_gene1, official_gene2, enrichment_results)
            
            if not shared_pathways:
                self._log("Genes do not co-occur in same pathways")
                return ("Genes do not share significant pathways", "none", [], [])
            
            # Analyze pathway types for biological insights
            pathway_text = " ".join(shared_pathways).lower()
            mechanisms = []
            pathway_insights = []
            
            if any(term in pathway_text for term in ["signaling", "signal", "cascade"]):
                mechanisms.append("signaling_pathway")
                pathway_insights.append("signal transduction")
            if any(term in pathway_text for term in ["metabolic", "metabolism", "synthesis"]):
                mechanisms.append("metabolic_pathway")
                pathway_insights.append("metabolic regulation")
            if any(term in pathway_text for term in ["dna repair", "cell cycle", "apoptosis"]):
                mechanisms.append("regulatory_pathway")
                pathway_insights.append("cell cycle control")
            if any(term in pathway_text for term in ["transport", "trafficking"]):
                mechanisms.append("transport_pathway")
                pathway_insights.append("cellular transport")
            
            if not mechanisms:
                mechanisms = ["general_pathway"]
                pathway_insights.append("general pathway interaction")
            
            # Log biological insights
            if pathway_insights:
                self._log(f"Biological mechanisms identified: {', '.join(pathway_insights)}")
            
            # Calculate confidence
            confidence = self._calculate_confidence(len(shared_pathways), interaction_score)
            
            # Create summary with biological context
            summary = f"Genes interact through {len(shared_pathways)} shared pathway(s) with {confidence} confidence"
            
            # Limit pathways to top 5
            key_pathways = shared_pathways[:5]
            
            self._log(f"Pathway analysis complete: {confidence} confidence, {len(mechanisms)} mechanism types")
            
            return (summary, confidence, mechanisms, key_pathways)
            
        except Exception as e:
            self._log(f"Pathway analysis failed: {str(e)}", "ERROR")
            return ("Pathway analysis failed", "none", [], [])


class ReactomeInteractionTool(BaseEnrichrInteractionTool):
    """Individual tool for Reactome-based genetic interaction analysis."""
    
    def __init__(self):
        super().__init__("Reactome_Pathways_2024", "REACTOME", top_results=5)
        
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair using Reactome data.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        self._log(f"Analyzing biochemical interactions for {gene1} + {gene2}")
        
        official_gene1 = self.get_official_gene_name(gene1)
        official_gene2 = self.get_official_gene_name(gene2)
        
        try:
            user_list_id = self.submit_gene_list([official_gene1, official_gene2])
            enrichment_results = self.get_enrichment_results(user_list_id)
            
            if not enrichment_results:
                self._log("No biochemical reactions found in Reactome database")
                return ("No biochemical interactions found", "none", [], [])
            
            shared_reactions, interaction_score = self._extract_shared_terms(official_gene1, official_gene2, enrichment_results)
            
            if not shared_reactions:
                self._log("Genes do not participate in same biochemical reactions")
                return ("Genes do not share biochemical reactions", "none", [], [])
            
            # Analyze reaction types for molecular insights
            mechanisms = []
            molecular_insights = []
            reaction_text = " ".join(shared_reactions).lower()
            
            if "phosphorylation" in reaction_text:
                mechanisms.append("phosphorylation_regulation")
                molecular_insights.append("phosphorylation signaling")
            if "complex" in reaction_text:
                mechanisms.append("protein_complex_formation")
                molecular_insights.append("protein complex assembly")
            if "binding" in reaction_text:
                mechanisms.append("direct_protein_binding")
                molecular_insights.append("direct molecular binding")
            if "transport" in reaction_text:
                mechanisms.append("transport_mechanism")
                molecular_insights.append("molecular transport")
            if "degradation" in reaction_text:
                mechanisms.append("protein_degradation")
                molecular_insights.append("protein degradation")
            
            if not mechanisms:
                mechanisms = ["biochemical_interaction"]
                molecular_insights.append("general biochemical interaction")
            
            # Log molecular insights
            if molecular_insights:
                self._log(f"Molecular mechanisms identified: {', '.join(molecular_insights)}")
            
            confidence = self._calculate_confidence(len(shared_reactions), interaction_score)
            summary = f"Genes interact through {len(shared_reactions)} shared biochemical reaction(s) with {confidence} confidence"
            key_pathways = shared_reactions[:5]
            
            self._log(f"Reactome analysis complete: {confidence} confidence, molecular interaction detected")
            
            return (summary, confidence, mechanisms, key_pathways)
            
        except Exception as e:
            self._log(f"Reactome analysis failed: {str(e)}", "ERROR")
            return ("Reactome analysis failed", "none", [], [])


class MSigDBHallmarkInteractionTool(BaseEnrichrInteractionTool):
    """Individual tool for MSigDB Hallmark-based genetic interaction analysis."""
    
    def __init__(self):
        super().__init__("MSigDB_Hallmark_2020", "HALLMARK", top_results=5)
        
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair using MSigDB Hallmarks.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        self._log(f"Analyzing cancer hallmark processes for {gene1} + {gene2}")
        
        official_gene1 = self.get_official_gene_name(gene1)
        official_gene2 = self.get_official_gene_name(gene2)
        
        try:
            user_list_id = self.submit_gene_list([official_gene1, official_gene2])
            enrichment_results = self.get_enrichment_results(user_list_id)
            
            if not enrichment_results:
                self._log("No cancer hallmark processes identified")
                return ("No hallmark processes found", "none", [], [])
            
            shared_hallmarks, interaction_score = self._extract_shared_terms(official_gene1, official_gene2, enrichment_results)
            
            if not shared_hallmarks:
                self._log("Genes do not co-regulate cancer hallmark processes")
                return ("Genes do not share cancer hallmark processes", "none", [], [])
            
            # Analyze hallmark types for cancer biology insights
            mechanisms = []
            cancer_insights = []
            hallmark_text = " ".join(shared_hallmarks).lower()
            
            if "apoptosis" in hallmark_text:
                mechanisms.append("apoptotic_regulation")
                cancer_insights.append("cell death regulation")
            if any(term in hallmark_text for term in ["proliferation", "cell_cycle"]):
                mechanisms.append("proliferative_control")
                cancer_insights.append("growth control")
            if "metabolism" in hallmark_text:
                mechanisms.append("metabolic_reprogramming")
                cancer_insights.append("metabolic rewiring")
            if any(term in hallmark_text for term in ["immune", "inflammatory"]):
                mechanisms.append("immune_response")
                cancer_insights.append("immune system interaction")
            if "dna_repair" in hallmark_text:
                mechanisms.append("dna_repair_mechanism")
                cancer_insights.append("genome stability")
            if "angiogenesis" in hallmark_text:
                mechanisms.append("angiogenesis_regulation")
                cancer_insights.append("blood vessel formation")
            
            if not mechanisms:
                mechanisms = ["hallmark_process"]
                cancer_insights.append("general cancer process")
            
            # Log cancer biology insights
            if cancer_insights:
                self._log(f"Cancer biology roles identified: {', '.join(cancer_insights)}")
            
            # Hallmarks get higher confidence weighting due to curated nature
            confidence = self._calculate_confidence(len(shared_hallmarks), interaction_score * 1.5)
            summary = f"Genes co-regulate {len(shared_hallmarks)} cancer hallmark process(es) with {confidence} confidence"
            key_pathways = shared_hallmarks[:5]
            
            self._log(f"Hallmark analysis complete: {confidence} confidence, cancer relevance established")
            
            return (summary, confidence, mechanisms, key_pathways)
            
        except Exception as e:
            self._log(f"Hallmark analysis failed: {str(e)}", "ERROR")
            return ("Hallmark analysis failed", "none", [], [])


class GOFunctionInteractionTool(BaseEnrichrInteractionTool):
    """Individual tool for GO Molecular Function-based genetic interaction analysis."""
    
    def __init__(self):
        super().__init__("GO_Molecular_Function_2023", "GO_FUNCTION", top_results=5)
        
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair using GO Molecular Functions.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        self._log(f"Analyzing molecular functions for {gene1} + {gene2}")
        
        official_gene1 = self.get_official_gene_name(gene1)
        official_gene2 = self.get_official_gene_name(gene2)
        
        try:
            user_list_id = self.submit_gene_list([official_gene1, official_gene2])
            enrichment_results = self.get_enrichment_results(user_list_id)
            
            if not enrichment_results:
                self._log("No shared molecular functions identified")
                return ("No shared molecular functions found", "none", [], [])
            
            shared_functions, interaction_score = self._extract_shared_terms(official_gene1, official_gene2, enrichment_results)
            
            if not shared_functions:
                self._log("Genes have distinct molecular function profiles")
                return ("Genes do not share molecular functions", "none", [], [])
            
            # Analyze function types for biochemical insights
            mechanisms = []
            functional_insights = []
            function_text = " ".join(shared_functions).lower()
            
            if "kinase" in function_text:
                mechanisms.append("kinase_activity")
                functional_insights.append("protein phosphorylation")
            if "phosphatase" in function_text:
                mechanisms.append("phosphatase_activity")
                functional_insights.append("protein dephosphorylation")
            if "binding" in function_text:
                mechanisms.append("protein_binding")
                functional_insights.append("molecular binding")
            if "transcription" in function_text:
                mechanisms.append("transcriptional_regulation")
                functional_insights.append("gene expression control")
            if "enzyme" in function_text:
                mechanisms.append("enzymatic_activity")
                functional_insights.append("catalytic activity")
            if "receptor" in function_text:
                mechanisms.append("receptor_activity")
                functional_insights.append("signal reception")
            
            if not mechanisms:
                mechanisms = ["molecular_function"]
                functional_insights.append("shared molecular activity")
            
            # Log functional insights
            if functional_insights:
                self._log(f"Functional activities identified: {', '.join(functional_insights)}")
            
            confidence = self._calculate_confidence(len(shared_functions), interaction_score)
            summary = f"Genes share {len(shared_functions)} molecular function(s) with {confidence} confidence"
            key_pathways = shared_functions[:5]
            
            self._log(f"Function analysis complete: {confidence} confidence, functional similarity detected")
            
            return (summary, confidence, mechanisms, key_pathways)
            
        except Exception as e:
            self._log(f"Function analysis failed: {str(e)}", "ERROR")
            return ("Function analysis failed", "none", [], [])


class GOProcessInteractionTool(BaseEnrichrInteractionTool):
    """Individual tool for GO Biological Process-based genetic interaction analysis."""
    
    def __init__(self):
        super().__init__("GO_Biological_Process_2023", "GO_PROCESS", top_results=5)
        
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair using GO Biological Processes.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        self._log(f"Analyzing biological processes for {gene1} + {gene2}")
        
        official_gene1 = self.get_official_gene_name(gene1)
        official_gene2 = self.get_official_gene_name(gene2)
        
        try:
            user_list_id = self.submit_gene_list([official_gene1, official_gene2])
            enrichment_results = self.get_enrichment_results(user_list_id)
            
            if not enrichment_results:
                self._log("No shared biological processes identified")
                return ("No shared biological processes found", "none", [], [])
            
            shared_processes, interaction_score = self._extract_shared_terms(official_gene1, official_gene2, enrichment_results)
            
            if not shared_processes:
                self._log("Genes participate in distinct biological processes")
                return ("Genes do not share biological processes", "none", [], [])
            
            # Analyze process types for biological insights
            mechanisms = []
            process_insights = []
            process_text = " ".join(shared_processes).lower()
            
            if any(term in process_text for term in ["cell cycle", "mitosis", "division"]):
                mechanisms.append("cell_cycle_regulation")
                process_insights.append("cell division control")
            if any(term in process_text for term in ["apoptosis", "cell death"]):
                mechanisms.append("apoptotic_process")
                process_insights.append("programmed cell death")
            if "signal transduction" in process_text:
                mechanisms.append("signal_transduction")
                process_insights.append("cellular signaling")
            if any(term in process_text for term in ["dna repair", "dna damage"]):
                mechanisms.append("dna_repair_process")
                process_insights.append("genome maintenance")
            if "metabolic" in process_text:
                mechanisms.append("metabolic_process")
                process_insights.append("metabolic regulation")
            if "development" in process_text:
                mechanisms.append("developmental_process")
                process_insights.append("developmental biology")
            
            if not mechanisms:
                mechanisms = ["biological_process"]
                process_insights.append("shared cellular process")
            
            # Log process insights
            if process_insights:
                self._log(f"Biological processes identified: {', '.join(process_insights)}")
            
            confidence = self._calculate_confidence(len(shared_processes), interaction_score)
            summary = f"Genes participate in {len(shared_processes)} shared biological process(es) with {confidence} confidence"
            key_pathways = shared_processes[:5]
            
            self._log(f"Process analysis complete: {confidence} confidence, process co-participation detected")
            
            return (summary, confidence, mechanisms, key_pathways)
            
        except Exception as e:
            self._log(f"Process analysis failed: {str(e)}", "ERROR")
            return ("Process analysis failed", "none", [], [])


# Convenience functions for easy tool access
def analyze_pathway_interaction(gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
    """Analyze gene pair using WikiPathways."""
    tool = WikiPathwaysInteractionTool()
    return tool.analyze_gene_pair(gene1, gene2)


def analyze_reactome_interaction(gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
    """Analyze gene pair using Reactome."""
    tool = ReactomeInteractionTool()
    return tool.analyze_gene_pair(gene1, gene2)


def analyze_hallmark_interaction(gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
    """Analyze gene pair using MSigDB Hallmarks."""
    tool = MSigDBHallmarkInteractionTool()
    return tool.analyze_gene_pair(gene1, gene2)


def analyze_function_interaction(gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
    """Analyze gene pair using GO Molecular Functions."""
    tool = GOFunctionInteractionTool()
    return tool.analyze_gene_pair(gene1, gene2)


def analyze_process_interaction(gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
    """Analyze gene pair using GO Biological Processes."""
    tool = GOProcessInteractionTool()
    return tool.analyze_gene_pair(gene1, gene2)


def analyze_comprehensive_interaction(gene1: str, gene2: str) -> Dict[str, Tuple[str, str, List[str], List[str]]]:
    """
    Run all interaction analysis tools on a gene pair.
    Returns a dictionary with results from each tool.
    """
    print(f"\n=== COMPREHENSIVE INTERACTION ANALYSIS ===")
    print(f"Analyzing gene pair: ({gene1}, {gene2})")
    print("=" * 50)
    
    tools = {
        'pathway': WikiPathwaysInteractionTool(),
        'reactome': ReactomeInteractionTool(),
        'hallmark': MSigDBHallmarkInteractionTool(),
        'function': GOFunctionInteractionTool(),
        'process': GOProcessInteractionTool()
    }
    
    results = {}
    
    for tool_name, tool in tools.items():
        try:
            print(f"\n--- {tool_name.upper()} ANALYSIS ---")
            result = tool.analyze_gene_pair(gene1, gene2)
            results[tool_name] = result
            
            summary, confidence, mechanisms, pathways = result
            print(f"Summary: {summary}")
            print(f"Confidence: {confidence}")
            print(f"Mechanisms: {mechanisms}")
            print(f"Key Evidence: {pathways[:3] if pathways else 'None'}")
            
        except Exception as e:
            print(f"ERROR in {tool_name}: {e}")
            results[tool_name] = ("Analysis failed", "none", [], [])
    
    return results


# Legacy compatibility function
def enrichr_api(genes: list, libs: list = None) -> dict:
    """
    Legacy function for backward compatibility.
    """
    if len(genes) < 2:
        raise ValueError("At least 2 genes required for gene pair analysis")
    
    gene1, gene2 = genes[0], genes[1]
    
    if libs is None:
        libs = ["WikiPathways_2024_Human", "Reactome_Pathways_2024", "MSigDB_Hallmark_2020"]
    
    tool_mapping = {
        "WikiPathways_2024_Human": analyze_pathway_interaction,
        "Reactome_Pathways_2024": analyze_reactome_interaction,
        "MSigDB_Hallmark_2020": analyze_hallmark_interaction,
        "GO_Molecular_Function_2023": analyze_function_interaction,
        "GO_Biological_Process_2023": analyze_process_interaction
    }
    
    results = {}
    
    for lib in libs:
        if lib in tool_mapping:
            try:
                summary, confidence, mechanisms, pathways = tool_mapping[lib](gene1, gene2)
                results[lib] = {
                    "summary": summary,
                    "confidence": confidence,
                    "mechanisms": mechanisms,
                    "pathways": pathways
                }
            except Exception as e:
                print(f"Error with {lib}: {e}")
                results[lib] = {
                    "summary": "Analysis failed",
                    "confidence": "none",
                    "mechanisms": [],
                    "pathways": []
                }
    
    return results


if __name__ == "__main__":
    # Example usage of individual tools
    print("=== INDIVIDUAL TOOL TESTING ===")
    
    # Test individual tools
    gene1, gene2 = 'TP53', 'MDM2'
    
    print(f"\n1. PATHWAY ANALYSIS:")
    summary, confidence, mechanisms, pathways = analyze_pathway_interaction(gene1, gene2)
    print(f"   Summary: {summary}")
    print(f"   Confidence: {confidence}")
    print(f"   Mechanisms: {mechanisms}")
    print(f"   Key Pathways: {pathways[:3]}")
    
    print(f"\n2. REACTOME ANALYSIS:")
    summary, confidence, mechanisms, pathways = analyze_reactome_interaction(gene1, gene2)
    print(f"   Summary: {summary}")
    print(f"   Confidence: {confidence}")
    print(f"   Mechanisms: {mechanisms}")
    print(f"   Key Reactions: {pathways[:3]}")
    
    print(f"\n3. HALLMARK ANALYSIS:")
    summary, confidence, mechanisms, pathways = analyze_hallmark_interaction(gene1, gene2)
    print(f"   Summary: {summary}")
    print(f"   Confidence: {confidence}")
    print(f"   Mechanisms: {mechanisms}")
    print(f"   Key Hallmarks: {pathways[:3]}")
    
    # Test comprehensive analysis
    print(f"\n=== COMPREHENSIVE ANALYSIS ===")
    all_results = analyze_comprehensive_interaction(gene1, gene2)
    
    print(f"\n=== SUMMARY OF ALL TOOLS ===")
    for tool_name, (summary, confidence, mechanisms, pathways) in all_results.items():
        print(f"{tool_name.upper()}: {confidence} confidence - {summary}")
        if mechanisms:
            print(f"  Key mechanisms: {', '.join(mechanisms[:2])}")
        if pathways:
            print(f"  Top evidence: {pathways[0]}")