import requests
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod

# Handle imports for both direct execution and module import
try:
    from tool_space.enrichr import get_official_gene_name
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from enrichr import get_official_gene_name
    except ImportError:
        # Fallback function if enrichr module not available
        def get_official_gene_name(gene: str) -> str:
            """Fallback function when enrichr module is not available"""
            return gene.upper()

@dataclass
class HumanBaseResult:
    """Standardized result structure for HumanBase analysis"""
    tissue: str
    genes: List[str]
    interaction_summary: str
    network_strength: str  # high/medium/low/minimal
    key_interactions: List[Dict[str, Any]]
    biological_processes: List[str]
    tissue_specificity: str  # high/medium/low
    clinical_relevance: str

class BaseHumanBaseTool(ABC):
    """Abstract base class for HumanBase tissue-specific analysis tools"""
    
    def __init__(self):
        self.base_url = "https://hb.flatironinstitute.org/api"
        self.max_retries = 3
        self.retry_delay = 1.0
        self._gene_cache = {}
        # Valid tissue types in HumanBase
        self._valid_tissues = {
            'adipose-tissue', 'adrenal-gland', 'blood', 'bone', 'brain', 'breast', 
            'colon', 'endothelial-cell', 'esophagus', 'heart', 'kidney', 'liver', 
            'lung', 'muscle', 'ovary', 'pancreas', 'prostate', 'skin', 'stomach', 
            'testis', 'thyroid', 'uterus', 'artery-endothelial-cell', 
            'gut-endothelial-cell', 'vein-endothelial-cell'
        }
    
    @abstractmethod
    def get_interaction_type(self) -> str:
        """Return the specific interaction type for this tool"""
        pass
    
    @abstractmethod
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        """Interpret the biological significance of interactions"""
        pass
    
    def _normalize_tissue(self, tissue: str) -> str:
        """Normalize tissue name for API"""
        return tissue.replace(" ", "-").replace("_", "-").lower()
    
    def _validate_tissue(self, tissue: str) -> Tuple[bool, str, List[str]]:
        """
        Validate tissue name and suggest alternatives if invalid
        Returns: (is_valid, normalized_tissue, suggestions)
        """
        normalized = self._normalize_tissue(tissue)
        
        if normalized in self._valid_tissues:
            return True, normalized, []
        
        # Find close matches for suggestions
        suggestions = []
        if 'endothelial' in tissue.lower():
            suggestions = ['endothelial-cell', 'artery-endothelial-cell', 'gut-endothelial-cell', 'vein-endothelial-cell']
        else:
            # Find tissues that contain part of the input
            suggestions = [t for t in self._valid_tissues if any(part in t for part in normalized.split('-'))]
            suggestions = suggestions[:3]  # Limit to 3 suggestions
        
        return False, normalized, suggestions
    
    def _get_entrez_ids(self, gene_names: List[str]) -> List[str]:
        """Convert gene names to Entrez IDs with caching"""
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        entrez_ids = []
        
        # Resolve gene names to official symbols first
        resolved_genes = []
        for gene in gene_names:
            if gene in self._gene_cache:
                resolved_genes.append(self._gene_cache[gene])
            else:
                official_name = get_official_gene_name(gene)
                self._gene_cache[gene] = official_name
                resolved_genes.append(official_name)
                print(f"[HUMANBASE] INFO: Resolved {gene} → {official_name}", flush=True)
        
        for gene in resolved_genes:
            cache_key = f"entrez_{gene}"
            if cache_key in self._gene_cache:
                entrez_ids.append(self._gene_cache[cache_key])
                continue
                
            params = {
                'db': 'gene',
                'term': f"{gene}[gene] AND Homo sapiens[orgn]",
                'retmode': 'xml',
                'retmax': '1'
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                xml_data = response.text
                
                start_idx = xml_data.find("<Id>")
                end_idx = xml_data.find("</Id>")
                
                if start_idx != -1 and end_idx != -1:
                    entrez_id = xml_data[start_idx + 4:end_idx]
                    entrez_ids.append(entrez_id)
                    self._gene_cache[cache_key] = entrez_id
                else:
                    entrez_ids.append(None)
                    print(f"[HUMANBASE] WARNING: No Entrez ID found for {gene}", flush=True)
                    
            except Exception as e:
                print(f"[HUMANBASE] ERROR: Failed to get Entrez ID for {gene}: {e}", flush=True)
                entrez_ids.append(None)
                
            time.sleep(0.1)  # Rate limiting
        
        return [eid for eid in entrez_ids if eid is not None]
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"[HUMANBASE] WARNING: Request failed, retrying in {wait_time}s: {e}", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"[HUMANBASE] ERROR: Request failed after {self.max_retries} attempts: {e}", flush=True)
                    return None
    
    def _calculate_network_strength(self, interactions: List[Dict]) -> str:
        """Calculate overall network strength based on meaningful interactions"""
        if not interactions:
            return "minimal"
        
        # Only consider interactions with meaningful evidence
        meaningful_interactions = [
            edge for edge in interactions 
            if edge.get('weight', 0) >= 0.3 and self._has_meaningful_evidence(edge.get('evidence', {}))
        ]
        
        if not meaningful_interactions:
            return "minimal"
        
        weights = [edge.get('weight', 0) for edge in meaningful_interactions]
        avg_weight = sum(weights) / len(weights) if weights else 0
        
        if avg_weight >= 0.8:
            return "high"
        elif avg_weight >= 0.6:
            return "medium"
        elif avg_weight >= 0.4:
            return "low"
        else:
            return "minimal"
    
    def _has_meaningful_evidence(self, evidence: Dict) -> bool:
        """Check if interaction has meaningful evidence (not all zeros)"""
        if not evidence:
            return False
        
        # Check if any evidence type has a meaningful score
        meaningful_evidence = [
            score for score in evidence.values() 
            if isinstance(score, (int, float)) and score > 0.1
        ]
        
        return len(meaningful_evidence) > 0
    
    def _analyze_evidence_types(self, interactions: List[Dict]) -> str:
        """Analyze and summarize the types of evidence supporting interactions"""
        if not interactions:
            return ""
        
        evidence_counts = {}
        for interaction in interactions:
            evidence = interaction.get('evidence', {})
            for evidence_type, score in evidence.items():
                if isinstance(score, (int, float)) and score > 0.1:
                    evidence_counts[evidence_type] = evidence_counts.get(evidence_type, 0) + 1
        
        if not evidence_counts:
            return ""
        
        # Sort by frequency and take top 3
        top_evidence = sorted(evidence_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        evidence_names = [ev[0].lower().replace(' ', '_') for ev, count in top_evidence]
        
        if len(evidence_names) == 1:
            return evidence_names[0]
        elif len(evidence_names) == 2:
            return f"{evidence_names[0]} and {evidence_names[1]}"
        else:
            return f"{', '.join(evidence_names[:-1])}, and {evidence_names[-1]}"
    
    def _assess_tissue_specificity(self, tissue: str, interactions: List[Dict]) -> str:
        """Assess tissue specificity of interactions"""
        # Simplified heuristic based on interaction strength and type
        if not interactions:
            return "low"
        
        # Tissues with high specificity
        high_specificity_tissues = ['brain', 'heart', 'liver', 'kidney', 'muscle']
        medium_specificity_tissues = ['blood', 'lung', 'skin', 'bone']
        
        tissue_norm = self._normalize_tissue(tissue)
        
        if any(ht in tissue_norm for ht in high_specificity_tissues):
            return "high"
        elif any(mt in tissue_norm for mt in medium_specificity_tissues):
            return "medium"
        else:
            return "low"
    
    def analyze_tissue_network(self, genes: List[str], tissue: str, max_interactions: int = 10) -> HumanBaseResult:
        """Analyze tissue-specific gene interactions"""
        print(f"[HUMANBASE] INFO: Analyzing {self.get_interaction_type()} interactions for {len(genes)} genes in {tissue}", flush=True)
        
        # Validate tissue name first
        is_valid, tissue_norm, suggestions = self._validate_tissue(tissue)
        if not is_valid:
            error_msg = f"Invalid tissue '{tissue}' for HumanBase analysis."
            if suggestions:
                error_msg += f" Available options: {', '.join(suggestions)}"
            print(f"[HUMANBASE] ERROR: {error_msg}", flush=True)
            return HumanBaseResult(
                tissue=tissue,
                genes=genes,
                interaction_summary=f"Invalid tissue name: {tissue}. {error_msg}",
                network_strength="minimal",
                key_interactions=[],
                biological_processes=[],
                tissue_specificity="low",
                clinical_relevance=f"Unable to assess - {error_msg}"
            )
        
        # Get Entrez IDs
        entrez_ids = self._get_entrez_ids(genes)
        if not entrez_ids:
            print(f"[HUMANBASE] ERROR: No valid Entrez IDs found for genes: {genes}", flush=True)
            return HumanBaseResult(
                tissue=tissue,
                genes=genes,
                interaction_summary="No valid gene identifiers found",
                network_strength="minimal",
                key_interactions=[],
                biological_processes=[],
                tissue_specificity="low",
                clinical_relevance="Unable to assess - gene resolution failed"
            )
        
        interaction_type = self.get_interaction_type()
        
        # Get network data
        network_url = f"{self.base_url}/integrations/{tissue_norm}/network/"
        network_params = {
            'datatypes': interaction_type,
            'entrez': entrez_ids,  # Pass as list, requests will handle the formatting
            'node_size': max_interactions + 5  # Get a few extra for filtering
        }
        
        network_data = self._make_request(network_url, network_params)
        if not network_data:
            return HumanBaseResult(
                tissue=tissue,
                genes=genes,
                interaction_summary="Network data unavailable",
                network_strength="minimal",
                key_interactions=[],
                biological_processes=[],
                tissue_specificity="low",
                clinical_relevance="Unable to assess - network data unavailable"
            )
        
        # Process interactions with quality filtering
        interactions = []
        gene_map = {g['entrez']: g['standard_name'] for g in network_data.get('genes', [])}
        
        # First pass: collect all interactions and sort by weight
        all_edges = sorted(
            network_data.get('edges', []), 
            key=lambda x: x.get('weight', 0), 
            reverse=True
        )
        
        for edge in all_edges:
            # Skip very low weight interactions
            if edge.get('weight', 0) < 0.2:
                continue
                
            source_entrez = network_data['genes'][edge['source']]['entrez']
            target_entrez = network_data['genes'][edge['target']]['entrez']
            source_name = gene_map.get(source_entrez, f"Gene_{source_entrez}")
            target_name = gene_map.get(target_entrez, f"Gene_{target_entrez}")
            
            # Get detailed interaction evidence
            evidence_url = f"{self.base_url}/integrations/{tissue_norm}/evidence/"
            evidence_params = {
                'limit': 5,
                'source': source_entrez,
                'target': target_entrez
            }
            
            evidence_data = self._make_request(evidence_url, evidence_params)
            evidence_types = {}
            if evidence_data:
                evidence_types = {t['title']: round(t['weight'], 3) for t in evidence_data.get('datatypes', [])}
            
            # Only include interactions with meaningful evidence
            if self._has_meaningful_evidence(evidence_types):
                # Filter evidence to only include meaningful scores
                filtered_evidence = {
                    evidence_type: score 
                    for evidence_type, score in evidence_types.items()
                    if isinstance(score, (int, float)) and score > 0.1
                }
                
                interactions.append({
                    'source': source_name,
                    'target': target_name,
                    'weight': round(edge['weight'], 3),
                    'evidence': filtered_evidence
                })
                
                # Stop when we have enough high-quality interactions
                if len(interactions) >= max_interactions:
                    break
        
        # Get biological processes
        bp_url = f"{self.base_url}/terms/annotated/"
        bp_params = {
            'database': 'gene-ontology-bp',
            'entrez': entrez_ids,  # Pass as list, requests will handle the formatting
            'max_term_size': 15
        }
        
        bp_data = self._make_request(bp_url, bp_params)
        biological_processes = []
        if bp_data:
            biological_processes = [bp['title'] for bp in bp_data[:10]]  # Top 10 processes
        
        # Calculate metrics
        network_strength = self._calculate_network_strength(interactions)
        tissue_specificity = self._assess_tissue_specificity(tissue, interactions)
        
        # Generate summary and clinical relevance
        interaction_summary = self.interpret_interactions(interactions, tissue)
        clinical_relevance = self._assess_clinical_relevance(interactions, tissue, biological_processes)
        
        print(f"[HUMANBASE] INFO: Found {len(interactions)} {interaction_type} interactions with {network_strength} strength", flush=True)
        if biological_processes:
            print(f"[HUMANBASE] INFO: Identified {len(biological_processes)} relevant biological processes", flush=True)
        
        return HumanBaseResult(
            tissue=tissue,
            genes=genes,
            interaction_summary=interaction_summary,
            network_strength=network_strength,
            key_interactions=interactions,
            biological_processes=biological_processes,
            tissue_specificity=tissue_specificity,
            clinical_relevance=clinical_relevance
        )
    
    def _assess_clinical_relevance(self, interactions: List[Dict], tissue: str, processes: List[str]) -> str:
        """Assess clinical relevance of findings"""
        if not interactions:
            return "Limited clinical relevance - no significant interactions detected"
        
        # High-impact tissues for clinical relevance
        clinical_tissues = {
            'brain': 'neurological disorders',
            'heart': 'cardiovascular disease',
            'liver': 'metabolic disorders',
            'kidney': 'renal disease',
            'blood': 'hematological conditions',
            'lung': 'respiratory disorders',
            'muscle': 'muscular dystrophies'
        }
        
        tissue_norm = self._normalize_tissue(tissue)
        clinical_context = "general health"
        
        for clinical_tissue, context in clinical_tissues.items():
            if clinical_tissue in tissue_norm:
                clinical_context = context
                break
        
        strength = self._calculate_network_strength(interactions)
        
        if strength == "high":
            return f"High clinical relevance - strong interactions suggest therapeutic targets for {clinical_context}"
        elif strength == "medium":
            return f"Moderate clinical relevance - interactions may inform {clinical_context} mechanisms"
        elif strength == "low":
            return f"Low clinical relevance - weak interactions require validation for {clinical_context}"
        else:
            return f"Minimal clinical relevance - insufficient interaction strength for {clinical_context}"

class CoExpressionAnalyzer(BaseHumanBaseTool):
    """Analyzes tissue-specific gene co-expression patterns"""
    
    def get_interaction_type(self) -> str:
        return "co-expression"
    
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        if not interactions:
            return f"No significant co-expression patterns detected in {tissue}"
        
        strong_coexp = [i for i in interactions if i['weight'] >= 0.7]
        moderate_coexp = [i for i in interactions if 0.4 <= i['weight'] < 0.7]
        
        # Analyze evidence types
        evidence_summary = self._analyze_evidence_types(interactions)
        
        summary = f"Identified {len(interactions)} high-quality co-expression relationships in {tissue}. "
        
        if strong_coexp:
            summary += f"{len(strong_coexp)} show strong co-expression (≥0.7), suggesting coordinated regulation. "
        if moderate_coexp:
            summary += f"{len(moderate_coexp)} show moderate co-expression (0.4-0.7), indicating functional relationships. "
        
        if evidence_summary:
            summary += f"Primary evidence: {evidence_summary}."
        
        return summary

class ProteinInteractionAnalyzer(BaseHumanBaseTool):
    """Analyzes tissue-specific protein-protein interactions"""
    
    def get_interaction_type(self) -> str:
        return "interaction"
    
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        if not interactions:
            return f"No significant protein interactions detected in {tissue}"
        
        direct_interactions = [i for i in interactions if i['weight'] >= 0.6]
        evidence_summary = self._analyze_evidence_types(interactions)
        
        summary = f"Detected {len(interactions)} high-quality protein interactions in {tissue}. "
        
        if direct_interactions:
            summary += f"{len(direct_interactions)} represent high-confidence direct interactions (≥0.6), "
            summary += "indicating physical protein complexes or direct binding partners. "
        else:
            summary += "Interactions suggest pathway-level associations and functional relationships. "
        
        if evidence_summary:
            summary += f"Supported by {evidence_summary}."
        
        return summary

class TranscriptionFactorAnalyzer(BaseHumanBaseTool):
    """Analyzes tissue-specific transcription factor binding patterns"""
    
    def get_interaction_type(self) -> str:
        return "tf-binding"
    
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        if not interactions:
            return f"No significant transcription factor binding detected in {tissue}"
        
        tf_targets = {}
        for interaction in interactions:
            # Assume source is TF, target is gene (simplified)
            tf = interaction['source']
            if tf not in tf_targets:
                tf_targets[tf] = []
            tf_targets[tf].append(interaction['target'])
        
        evidence_summary = self._analyze_evidence_types(interactions)
        summary = f"Identified {len(interactions)} high-quality transcription factor-gene relationships in {tissue}. "
        
        if tf_targets:
            hub_tfs = [tf for tf, targets in tf_targets.items() if len(targets) >= 2]
            if hub_tfs:
                summary += f"Key regulatory hubs: {', '.join(hub_tfs[:3])} control multiple targets, "
                summary += "suggesting master regulatory roles in tissue-specific expression. "
        
        if evidence_summary:
            summary += f"Evidence includes {evidence_summary}."
        
        return summary

class MicroRNATargetAnalyzer(BaseHumanBaseTool):
    """Analyzes tissue-specific microRNA-target interactions"""
    
    def get_interaction_type(self) -> str:
        return "gsea-microrna-targets"
    
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        if not interactions:
            return f"No significant microRNA-target relationships detected in {tissue}"
        
        evidence_summary = self._analyze_evidence_types(interactions)
        summary = f"Found {len(interactions)} high-quality microRNA-target interactions in {tissue}. "
        summary += "These represent post-transcriptional regulatory mechanisms that fine-tune gene expression "
        summary += f"in {tissue}-specific contexts, controlling cellular responses and adaptation. "
        
        if evidence_summary:
            summary += f"Supported by {evidence_summary}."
        
        return summary

class PerturbationAnalyzer(BaseHumanBaseTool):
    """Analyzes tissue-specific gene perturbation outcomes"""
    
    def get_interaction_type(self) -> str:
        return "gsea-perturbations"
    
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        if not interactions:
            return f"No significant perturbation responses detected in {tissue}"
        
        high_impact = [i for i in interactions if i['weight'] >= 0.6]
        evidence_summary = self._analyze_evidence_types(interactions)
        
        summary = f"Identified {len(interactions)} high-quality perturbation-response relationships in {tissue}. "
        
        if high_impact:
            summary += f"{len(high_impact)} show high-impact responses (≥0.6), indicating genes with strong "
            summary += f"functional consequences when perturbed in {tissue}. These represent potential "
            summary += "therapeutic targets or biomarkers for tissue-specific interventions. "
        else:
            summary += f"Perturbations show moderate but significant effects in {tissue}, suggesting "
            summary += "important regulatory roles that merit further investigation. "
        
        if evidence_summary:
            summary += f"Evidence based on {evidence_summary}."
        
        return summary

# Convenience functions for agent-friendly access
def humanbase_analyze_tissue_coexpression(genes: List[str], tissue: str, max_interactions: int = 10) -> Tuple[str, str, List[Dict], List[str]]:
    """
    Analyze tissue-specific gene co-expression patterns
    Returns: (summary, confidence, top_interactions, biological_processes)
    """
    analyzer = CoExpressionAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
    
    confidence = result.network_strength
    summary = result.interaction_summary
    top_interactions = result.key_interactions
    biological_processes = result.biological_processes
    
    return summary, confidence, top_interactions, biological_processes

def humanbase_analyze_tissue_protein_interactions(genes: List[str], tissue: str, max_interactions: int = 10) -> Tuple[str, str, List[Dict], List[str]]:
    """
    Analyze tissue-specific protein-protein interactions
    Returns: (summary, confidence, top_interactions, biological_processes)
    """
    analyzer = ProteinInteractionAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
    
    confidence = result.network_strength
    summary = result.interaction_summary
    top_interactions = result.key_interactions
    biological_processes = result.biological_processes
    
    return summary, confidence, top_interactions, biological_processes

def humanbase_analyze_tissue_transcription_regulation(genes: List[str], tissue: str, max_interactions: int = 10) -> Tuple[str, str, List[Dict], List[str]]:
    """
    Analyze tissue-specific transcription factor regulation
    Returns: (summary, confidence, top_interactions, biological_processes)
    """
    analyzer = TranscriptionFactorAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
    
    confidence = result.network_strength
    summary = result.interaction_summary
    top_interactions = result.key_interactions
    biological_processes = result.biological_processes
    
    return summary, confidence, top_interactions, biological_processes

def humanbase_analyze_tissue_microrna_regulation(genes: List[str], tissue: str, max_interactions: int = 10) -> Tuple[str, str, List[Dict], List[str]]:
    """
    Analyze tissue-specific microRNA-target regulation
    Returns: (summary, confidence, top_interactions, biological_processes)
    """
    analyzer = MicroRNATargetAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
    
    confidence = result.network_strength
    summary = result.interaction_summary
    top_interactions = result.key_interactions
    biological_processes = result.biological_processes
    
    return summary, confidence, top_interactions, biological_processes

def humanbase_analyze_tissue_perturbation_outcomes(genes: List[str], tissue: str, max_interactions: int = 10) -> Tuple[str, str, List[Dict], List[str]]:
    """
    Analyze tissue-specific gene perturbation outcomes
    Returns: (summary, confidence, top_interactions, biological_processes)
    """
    analyzer = PerturbationAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
    
    confidence = result.network_strength
    summary = result.interaction_summary
    top_interactions = result.key_interactions
    biological_processes = result.biological_processes
    
    return summary, confidence, top_interactions, biological_processes

def humanbase_analyze_comprehensive_tissue_network(genes: List[str], tissue: str, max_interactions: int = 8) -> Dict[str, Tuple[str, str, List[Dict], List[str]]]:
    """
    Comprehensive tissue-specific network analysis across all interaction types
    Returns: Dict with keys: coexpression, protein_interactions, tf_regulation, microrna_regulation, perturbation_outcomes
    """
    print(f"[HUMANBASE] INFO: Starting comprehensive tissue network analysis for {len(genes)} genes in {tissue}", flush=True)
    
    results = {}
    
    # Run all analyses
    analyzers = {
        'coexpression': CoExpressionAnalyzer(),
        'protein_interactions': ProteinInteractionAnalyzer(),
        'tf_regulation': TranscriptionFactorAnalyzer(),
        'microrna_regulation': MicroRNATargetAnalyzer(),
        'perturbation_outcomes': PerturbationAnalyzer()
    }
    
    for analysis_type, analyzer in analyzers.items():
        try:
            result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
            results[analysis_type] = (
                result.interaction_summary,
                result.network_strength,
                result.key_interactions,
                result.biological_processes
            )
        except Exception as e:
            print(f"[HUMANBASE] ERROR: {analysis_type} analysis failed: {e}", flush=True)
            results[analysis_type] = (
                f"Analysis failed: {str(e)}",
                "minimal",
                [],
                []
            )
    
    print(f"[HUMANBASE] INFO: Comprehensive analysis completed for {tissue}", flush=True)
    return results

# Legacy function for backward compatibility
def humanbase_ppi_retrieve(genes: list, tissue: str, max_node=10, interaction=None):
    """Legacy function - use class-based analyzers instead"""
    print("[HUMANBASE] WARNING: Using deprecated function. Consider using class-based analyzers.", flush=True)
    
    analyzer = ProteinInteractionAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_node)
    
    # Return simplified format for compatibility
    return result.key_interactions, result.biological_processes

def get_entrez_ids(gene_names):
    """Legacy function - use class methods instead"""
    analyzer = BaseHumanBaseTool()
    return analyzer._get_entrez_ids(gene_names)

if __name__ == "__main__":
    # Test the new system
    test_genes = ["ATP6AP1", "METAP2"]
    test_tissue = "blood"
    
    print("Testing Co-expression Analysis:")
    summary, confidence, interactions, processes = humanbase_analyze_tissue_protein_interactions(test_genes, test_tissue)
    print(f"Summary: {summary}")
    print(f"Confidence: {confidence}")
    print(f"Interactions: {interactions}")
    print(f"Processes: {processes}")