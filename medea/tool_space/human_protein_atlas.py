# hpa_tool.py

import requests
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional

HPA_SEARCH_API = "https://www.proteinatlas.org/api/search_download.php"
HPA_BASE = "https://www.proteinatlas.org"
HPA_JSON_API_TEMPLATE = "https://www.proteinatlas.org/{ensembl_id}.json"
HPA_XML_API_TEMPLATE = "https://www.proteinatlas.org/{ensembl_id}.xml"

# --- Base Tool Classes ---

class HPASearchApiTool:
    """
    Base class for interacting with HPA's search_download.php API.
    Uses HPA's search and download API to get protein expression data.
    """
    def __init__(self, tool_config):
        self.timeout = 30
        self.base_url = HPA_SEARCH_API

    def _make_api_request(self, search_term: str, columns: str, format_type: str = "json") -> Dict[str, Any]:
        """Make HPA API request with improved error handling"""
        params = {
            "search": search_term,
            "format": format_type,
            "columns": columns,
            "compress": "no"
        }
        
        try:
            resp = requests.get(self.base_url, params=params, timeout=self.timeout)
            if resp.status_code == 404:
                return {"error": f"No data found for gene '{search_term}'"}
            if resp.status_code != 200:
                return {"error": f"HPA API request failed, HTTP {resp.status_code}", "detail": resp.text}
            
            if format_type == "json":
                data = resp.json()
                # Ensure we always return a list for consistency
                if not isinstance(data, list):
                    return {"error": "API did not return expected list format"}
                return data
            else:
                return {"tsv_data": resp.text}
                
        except requests.RequestException as e:
            return {"error": f"HPA API request failed: {str(e)}"}
        except ValueError as e:
            return {"error": f"Failed to parse HPA response data: {str(e)}", "content": resp.text}


class HPAJsonApiTool:
    """
    Base class for interacting with HPA's /{ensembl_id}.json API.
    More efficient for getting comprehensive gene data.
    """
    def __init__(self, tool_config):
        self.timeout = 30
        self.base_url_template = HPA_JSON_API_TEMPLATE

    def _make_api_request(self, ensembl_id: str) -> Dict[str, Any]:
        """Make HPA JSON API request for a specific gene"""
        url = self.base_url_template.format(ensembl_id=ensembl_id)
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 404:
                return {"error": f"No data found for Ensembl ID '{ensembl_id}'"}
            if resp.status_code != 200:
                return {"error": f"HPA JSON API request failed, HTTP {resp.status_code}", "detail": resp.text}
            
            return resp.json()
                
        except requests.RequestException as e:
            return {"error": f"HPA JSON API request failed: {str(e)}"}
        except ValueError as e:
            return {"error": f"Failed to parse HPA JSON response: {str(e)}", "content": resp.text}


class HPAXmlApiTool:
    """
    Base class for interacting with HPA's /{ensembl_id}.xml API.
    Optimized for comprehensive XML data extraction.
    """
    def __init__(self, tool_config):
        self.timeout = 45
        self.base_url_template = HPA_XML_API_TEMPLATE
    
    def _make_api_request(self, ensembl_id: str) -> ET.Element:
        """Make HPA XML API request for a specific gene"""
        url = self.base_url_template.format(ensembl_id=ensembl_id)
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 404:
                raise Exception(f"No XML data found for Ensembl ID '{ensembl_id}'")
            if resp.status_code != 200:
                raise Exception(f"HPA XML API request failed, HTTP {resp.status_code}")
            
            return ET.fromstring(resp.content)
        except requests.RequestException as e:
            raise Exception(f"HPA XML API request failed: {str(e)}")
        except ET.ParseError as e:
            raise Exception(f"Failed to parse HPA XML response: {str(e)}")


# --- New Enhanced Tools Based on Your Optimization Plan ---

class HPAGetRnaExpressionBySourceTool(HPASearchApiTool):
    """
    Get RNA expression for a gene from specific biological sources using optimized columns parameter.
    This tool directly leverages the comprehensive columns table for efficient queries.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        # Use correct HPA API column identifiers
        self.source_column_mappings = {
            "tissue": "rnatsm",      # RNA tissue specific nTPM
            "blood": "rnablm",       # RNA blood lineage specific nTPM  
            "brain": "rnabrm",       # RNA brain region specific nTPM
            "single_cell": "rnascm"  # RNA single cell type specific nTPM
        }
        
        # Map expected API response field names for each source type
        self.api_response_fields = {
            "tissue": "RNA tissue specific nTPM",
            "blood": "RNA blood lineage specific nTPM", 
            "brain": "RNA brain region specific nTPM",
            "single_cell": "RNA single cell type specific nTPM"
        }
        
        # Map source names to expected keys in API response
        self.source_name_mappings = {
            "tissue": {
                "adipose_tissue": ["adipose tissue", "fat"],
                "adrenal_gland": ["adrenal gland", "adrenal"],
                "appendix": ["appendix"],
                "bone_marrow": ["bone marrow"],
                "brain": ["brain", "cerebral cortex"],
                "breast": ["breast"],
                "bronchus": ["bronchus"],
                "cerebellum": ["cerebellum"],
                "cerebral_cortex": ["cerebral cortex", "brain"],
                "cervix": ["cervix"],
                "choroid_plexus": ["choroid plexus"],
                "colon": ["colon"],
                "duodenum": ["duodenum"],
                "endometrium": ["endometrium"],
                "epididymis": ["epididymis"],
                "esophagus": ["esophagus"],
                "fallopian_tube": ["fallopian tube"],
                "gallbladder": ["gallbladder"],
                "heart_muscle": ["heart muscle", "heart"],
                "hippocampal_formation": ["hippocampus", "hippocampal formation"],
                "hypothalamus": ["hypothalamus"],
                "kidney": ["kidney"],
                "liver": ["liver"],
                "lung": ["lung"],
                "lymph_node": ["lymph node"],
                "nasopharynx": ["nasopharynx"],
                "oral_mucosa": ["oral mucosa"],
                "ovary": ["ovary"],
                "pancreas": ["pancreas"],
                "parathyroid_gland": ["parathyroid gland"],
                "pituitary_gland": ["pituitary gland"],
                "placenta": ["placenta"],
                "prostate": ["prostate"],
                "rectum": ["rectum"],
                "retina": ["retina"],
                "salivary_gland": ["salivary gland"],
                "seminal_vesicle": ["seminal vesicle"],
                "skeletal_muscle": ["skeletal muscle"],
                "skin": ["skin"],
                "small_intestine": ["small intestine"],
                "smooth_muscle": ["smooth muscle"],
                "soft_tissue": ["soft tissue"],
                "spleen": ["spleen"],
                "stomach": ["stomach"],
                "testis": ["testis"],
                "thymus": ["thymus"],
                "thyroid_gland": ["thyroid gland"],
                "tongue": ["tongue"],
                "tonsil": ["tonsil"],
                "urinary_bladder": ["urinary bladder"],
                "vagina": ["vagina"]
            },
            "blood": {
                "t_cell": ["t-cell", "t cell"],
                "b_cell": ["b-cell", "b cell"],
                "nk_cell": ["nk-cell", "nk cell", "natural killer"],
                "monocyte": ["monocyte"],
                "neutrophil": ["neutrophil"],
                "eosinophil": ["eosinophil"],
                "basophil": ["basophil"],
                "dendritic_cell": ["dendritic cell"]
            },
            "brain": {
                "cerebellum": ["cerebellum"],
                "cerebral_cortex": ["cerebral cortex", "cortex"],
                "hippocampus": ["hippocampus", "hippocampal formation"],
                "hypothalamus": ["hypothalamus"],
                "amygdala": ["amygdala"],
                "brainstem": ["brainstem", "brain stem"],
                "thalamus": ["thalamus"]
            },
            "single_cell": {
                "t_cell": ["t-cell", "t cell"],
                "b_cell": ["b-cell", "b cell"],
                "hepatocyte": ["hepatocyte"],
                "neuron": ["neuron"],
                "astrocyte": ["astrocyte"],
                "fibroblast": ["fibroblast"]
            }
        }

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        source_type = (arguments.get("source_type") or "").lower()
        source_name = (arguments.get("source_name") or "").lower().replace(' ', '_').replace('-', '_')
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        if not source_type:
            return {"error": "Parameter 'source_type' is required"}
        if not source_name:
            return {"error": "Parameter 'source_name' is required"}
        
        # Validate source type
        if source_type not in self.source_column_mappings:
            available_types = ", ".join(self.source_column_mappings.keys())
            return {"error": f"Invalid source_type '{source_type}'. Available types: {available_types}"}
        
        # Enhanced validation with intelligent recommendations
        if source_name not in self.source_name_mappings[source_type]:
            available_sources = list(self.source_name_mappings[source_type].keys())
            
            # Find similar source names (fuzzy matching)
            similar_sources = []
            source_keywords = source_name.replace('_', ' ').split()
            
            for valid_source in available_sources:
                # Direct substring matching
                if (source_name.lower() in valid_source.lower() or 
                    valid_source.lower() in source_name.lower()):
                    similar_sources.append(valid_source)
                    continue
                
                # Check with underscores removed/normalized
                normalized_input = source_name.lower().replace('_', '').replace(' ', '')
                normalized_valid = valid_source.lower().replace('_', '').replace(' ', '')
                if (normalized_input in normalized_valid or 
                    normalized_valid in normalized_input):
                    similar_sources.append(valid_source)
                    continue
                    
                # Check individual keywords
                for keyword in source_keywords:
                    if (keyword.lower() in valid_source.lower() or 
                        valid_source.lower() in keyword.lower()):
                        similar_sources.append(valid_source)
                        break
            
            error_msg = f"Invalid source_name '{source_name}' for source_type '{source_type}'. "
            if similar_sources:
                error_msg += f"Similar options: {similar_sources[:3]}. "
            error_msg += f"All available sources for '{source_type}': {available_sources}"
            return {"error": error_msg}
        
        try:
            # Get the correct API column
            api_column = self.source_column_mappings[source_type]
            columns = f"g,gs,{api_column}"
            
            # Call the search API
            response_data = self._make_api_request(gene_name, columns)
            
            if "error" in response_data:
                return response_data
            
            if not response_data or len(response_data) == 0:
                return {
                    "gene_name": gene_name,
                    "source_type": source_type,
                    "source_name": source_name,
                    "expression_value": "N/A",
                    "status": "Gene not found"
                }
            
            # Get the first result
            gene_data = response_data[0]
            
            # Extract expression data from the API response
            expression_value = "N/A"
            available_sources = []
            
            # Get the expression data dictionary for this source type
            api_field_name = self.api_response_fields[source_type]
            expression_data = gene_data.get(api_field_name)
            
            if expression_data and isinstance(expression_data, dict):
                available_sources = list(expression_data.keys())
                
                # Get possible names for this source
                possible_names = self.source_name_mappings[source_type][source_name]
                
                # Try to find a matching source name in the response
                for source_key in expression_data.keys():
                    source_key_lower = source_key.lower()
                    for possible_name in possible_names:
                        if possible_name.lower() in source_key_lower or source_key_lower in possible_name.lower():
                            expression_value = expression_data[source_key]
                            break
                    if expression_value != "N/A":
                        break
                
                # If exact match not found, look for partial matches
                if expression_value == "N/A":
                    source_keywords = source_name.replace('_', ' ').split()
                    for source_key in expression_data.keys():
                        source_key_lower = source_key.lower()
                        for keyword in source_keywords:
                            if keyword in source_key_lower:
                                expression_value = expression_data[source_key]
                                break
                        if expression_value != "N/A":
                            break
            
            # Categorize expression level
            expression_level = "unknown"
            if expression_value != "N/A":
                try:
                    val = float(expression_value)
                    if val > 50:
                        expression_level = "very high"
                    elif val > 10:
                        expression_level = "high"
                    elif val > 1:
                        expression_level = "medium"
                    elif val > 0.1:
                        expression_level = "low"
                    else:
                        expression_level = "very low"
                except (ValueError, TypeError):
                    expression_level = "unknown"
            
            return {
                "gene_name": gene_data.get("Gene", gene_name),
                "gene_synonym": gene_data.get("Gene synonym", ""),
                "source_type": source_type,
                "source_name": source_name,
                "expression_value": expression_value,
                "expression_level": expression_level,
                "expression_unit": "nTPM",
                "column_queried": api_column,
                "available_sources": available_sources[:10] if len(available_sources) > 10 else available_sources,
                "total_available_sources": len(available_sources),
                "status": "success" if expression_value != "N/A" else "no_expression_data_for_source"
            }
            
        except Exception as e:
            return {
                "error": f"Failed to retrieve RNA expression data: {str(e)}",
                "gene_name": gene_name,
                "source_type": source_type,
                "source_name": source_name
            }


class HPAGetSubcellularLocationTool(HPASearchApiTool):
    """
    Get annotated subcellular locations for a protein using optimized columns parameter.
    Uses scml (main location) and scal (additional location) columns for efficient queries.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        
        # Use specific columns for subcellular location data
        result = self._make_api_request(gene_name, "g,gs,scml,scal")
        
        if "error" in result:
            return result
        
        if not result:
            return {"error": "No subcellular location data found"}
        
        gene_data = result[0]
        
        # Parse main and additional locations
        main_location = gene_data.get("Subcellular main location", "")
        additional_location = gene_data.get("Subcellular additional location", "")
        
        # Handle different data types (string or list)
        if isinstance(main_location, list):
            main_locations = main_location
        elif isinstance(main_location, str):
            main_locations = [loc.strip() for loc in main_location.split(';') if loc.strip()] if main_location else []
        else:
            main_locations = []
            
        if isinstance(additional_location, list):
            additional_locations = additional_location
        elif isinstance(additional_location, str):
            additional_locations = [loc.strip() for loc in additional_location.split(';') if loc.strip()] if additional_location else []
        else:
            additional_locations = []
        
        return {
            "gene_name": gene_data.get("Gene", gene_name),
            "gene_synonym": gene_data.get("Gene synonym", ""),
            "main_locations": main_locations,
            "additional_locations": additional_locations,
            "total_locations": len(main_locations) + len(additional_locations),
            "location_summary": self._generate_location_summary(main_locations, additional_locations)
        }
    
    def _generate_location_summary(self, main_locs: List[str], add_locs: List[str]) -> str:
        """Generate a summary of subcellular locations"""
        if not main_locs and not add_locs:
            return "No subcellular location data available"
        
        summary_parts = []
        if main_locs:
            summary_parts.append(f"Primary: {', '.join(main_locs)}")
        if add_locs:
            summary_parts.append(f"Additional: {', '.join(add_locs)}")
        
        return "; ".join(summary_parts)


# --- Existing Tools (Updated with improvements) ---

class HPASearchGenesTool(HPASearchApiTool):
    """
    Search for matching genes by gene name, keywords, or cell line names and return Ensembl ID list.
    This is the entry tool for many query workflows.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        search_query = arguments.get("search_query")
        if not search_query:
            return {"error": "Parameter 'search_query' is required"}
        
        # 'g' for Gene name, 'gs' for Gene synonym, 'eg' for Ensembl ID
        columns = "g,gs,eg"
        result = self._make_api_request(search_query, columns)

        if "error" in result:
            return result
        
        if not result or not isinstance(result, list):
            return {"error": f"No matching genes found for query '{search_query}'"}
            
        formatted_results = []
        for gene in result:
            gene_synonym = gene.get("Gene synonym", "")
            if isinstance(gene_synonym, str):
                synonyms = gene_synonym.split(', ') if gene_synonym else []
            elif isinstance(gene_synonym, list):
                synonyms = gene_synonym
            else:
                synonyms = []
            
            formatted_results.append({
                "gene_name": gene.get("Gene"),
                "ensembl_id": gene.get("Ensembl"),
                "gene_synonyms": synonyms
            })
        
        return {
            "search_query": search_query,
            "match_count": len(formatted_results),
            "genes": formatted_results
        }


class HPAGetComparativeExpressionTool(HPASearchApiTool):
    """
    Compare gene expression levels in specific cell lines and healthy tissues.
    Get expression data for comparison by gene name and cell line name.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        # Mapping of common cell lines to their column identifiers
        self.cell_line_columns = {
            "ishikawa": "cell_RNA_ishikawa_heraklio",
            "hela": "cell_RNA_hela",
            "mcf7": "cell_RNA_mcf7",
            "a549": "cell_RNA_a549",
            "hepg2": "cell_RNA_hepg2",
            "jurkat": "cell_RNA_jurkat",
            "pc3": "cell_RNA_pc3",
            "rh30": "cell_RNA_rh30",
            "siha": "cell_RNA_siha",
            "u251": "cell_RNA_u251"
        }

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        cell_line = (arguments.get("cell_line") or "").lower()
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        if not cell_line:
            return {"error": "Parameter 'cell_line' is required"}
        
        # Enhanced validation with intelligent recommendations
        cell_column = self.cell_line_columns.get(cell_line)
        if not cell_column:
            available_lines = list(self.cell_line_columns.keys())
            
            # Find similar cell line names
            similar_lines = []
            for valid_line in available_lines:
                if cell_line in valid_line or valid_line in cell_line:
                    similar_lines.append(valid_line)
            
            error_msg = f"Unsupported cell_line '{cell_line}'. "
            if similar_lines:
                error_msg += f"Similar options: {similar_lines}. "
            error_msg += f"All supported cell lines: {available_lines}"
            return {"error": error_msg}
        
        # Request expression data for the cell line
        cell_columns = f"g,gs,{cell_column}"
        cell_result = self._make_api_request(gene_name, cell_columns)
        if "error" in cell_result:
            return cell_result
        
        # Request expression data for healthy tissues
        tissue_columns = "g,gs,rnatsm"
        tissue_result = self._make_api_request(gene_name, tissue_columns)
        if "error" in tissue_result:
            return tissue_result
        
        # Format the result
        if not cell_result or not tissue_result:
            return {"error": "No expression data found"}
        
        # Extract the first matching gene data
        cell_data = cell_result[0] if isinstance(cell_result, list) and cell_result else {}
        tissue_data = tissue_result[0] if isinstance(tissue_result, list) and tissue_result else {}
        
        return {
            "gene_name": gene_name,
            "gene_symbol": cell_data.get("Gene", gene_name),
            "gene_synonym": cell_data.get("Gene synonym", ""),
            "cell_line": cell_line,
            "cell_line_expression": cell_data.get(cell_column, "N/A"),
            "healthy_tissue_expression": tissue_data.get("RNA tissue specific nTPM", "N/A"),
            "expression_unit": "nTPM (normalized Transcripts Per Million)",
            "comparison_summary": self._generate_comparison_summary(
                cell_data.get(cell_column), 
                tissue_data.get("RNA tissue specific nTPM")
            )
        }
    
    def _generate_comparison_summary(self, cell_expr, tissue_expr) -> str:
        """Generate expression level comparison summary"""
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is None or tissue_val is None:
                return "Insufficient data for comparison"
            
            if cell_val > tissue_val * 2:
                return f"Expression significantly higher in cell line ({cell_val:.2f} vs {tissue_val:.2f})"
            elif tissue_val > cell_val * 2:
                return f"Expression significantly higher in healthy tissues ({tissue_val:.2f} vs {cell_val:.2f})"
            else:
                return f"Expression levels similar (cell line: {cell_val:.2f}, healthy tissues: {tissue_val:.2f})"
        except:
            return "Failed to calculate expression level comparison"


class HPAGetDiseaseExpressionTool(HPASearchApiTool):
    """
    Get expression data for a gene in specific diseases and tissues.
    Get related expression information by gene name, tissue type, and disease name.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        # Mapping of common cancer types to their column identifiers
        self.cancer_columns = {
            "brain_cancer": "cancer_RNA_brain_cancer",
            "breast_cancer": "cancer_RNA_breast_cancer", 
            "colon_cancer": "cancer_RNA_colon_cancer",
            "lung_cancer": "cancer_RNA_lung_cancer",
            "liver_cancer": "cancer_RNA_liver_cancer",
            "prostate_cancer": "cancer_RNA_prostate_cancer",
            "kidney_cancer": "cancer_RNA_kidney_cancer",
            "pancreatic_cancer": "cancer_RNA_pancreatic_cancer",
            "stomach_cancer": "cancer_RNA_stomach_cancer",
            "ovarian_cancer": "cancer_RNA_ovarian_cancer"
        }

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        tissue_type = (arguments.get("tissue_type") or "").lower() 
        disease_name = (arguments.get("disease_name") or "").lower()
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        if not disease_name:
            return {"error": "Parameter 'disease_name' is required"}
        
        # Enhanced validation with intelligent recommendations
        disease_key = f"{tissue_type}_{disease_name}" if tissue_type else disease_name
        cancer_column = None
        
        # Match cancer type
        for key, column in self.cancer_columns.items():
            if disease_key in key or disease_name in key:
                cancer_column = column
                break
        
        if not cancer_column:
            available_diseases = [k.replace("_", " ") for k in self.cancer_columns.keys()]
            
            # Find similar disease names
            similar_diseases = []
            disease_keywords = disease_name.replace('_', ' ').split()
            
            for valid_disease in available_diseases:
                for keyword in disease_keywords:
                    if keyword in valid_disease.lower() or valid_disease.lower() in keyword:
                        similar_diseases.append(valid_disease)
                        break
            
            error_msg = f"Unsupported disease_name '{disease_name}'. "
            if similar_diseases:
                error_msg += f"Similar options: {similar_diseases[:3]}. "
            error_msg += f"All supported diseases: {available_diseases}"
            return {"error": error_msg}
        
        # Build request columns
        columns = f"g,gs,{cancer_column},rnatsm"
        result = self._make_api_request(gene_name, columns)
        
        if "error" in result:
            return result
        
        if not result:
            return {"error": "No expression data found"}
        
        # Extract the first matching gene data
        gene_data = result[0] if isinstance(result, list) and result else {}
        
        return {
            "gene_name": gene_name,
            "gene_symbol": gene_data.get("Gene", gene_name),
            "gene_synonym": gene_data.get("Gene synonym", ""),
            "tissue_type": tissue_type or "Not specified",
            "disease_name": disease_name,
            "disease_expression": gene_data.get(cancer_column, "N/A"),
            "healthy_expression": gene_data.get("RNA tissue specific nTPM", "N/A"),
            "expression_unit": "nTPM (normalized Transcripts Per Million)",
            "disease_vs_healthy": self._compare_disease_healthy(
                gene_data.get(cancer_column),
                gene_data.get("RNA tissue specific nTPM")
            )
        }
    
    def _compare_disease_healthy(self, disease_expr, healthy_expr) -> str:
        """Compare expression difference between disease and healthy state"""
        try:
            disease_val = float(disease_expr) if disease_expr and disease_expr != "N/A" else None
            healthy_val = float(healthy_expr) if healthy_expr and healthy_expr != "N/A" else None
            
            if disease_val is None or healthy_val is None:
                return "Insufficient data for comparison"
            
            fold_change = disease_val / healthy_val if healthy_val > 0 else float('inf')
            
            if fold_change > 2:
                return f"Disease state expression upregulated {fold_change:.2f} fold"
            elif fold_change < 0.5:
                return f"Disease state expression downregulated {1/fold_change:.2f} fold"
            else:
                return f"Expression level relatively stable (fold change: {fold_change:.2f})"
        except:
            return "Failed to calculate expression difference"


class HPAGetBiologicalProcessTool(HPASearchApiTool):
    """
    Get biological process information related to a gene.
    Get specific biological processes a gene is involved in by gene name.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        # Predefined biological process list
        self.target_processes = [
            'Apoptosis', 'Biological rhythms', 'Cell cycle', 
            'Host-virus interaction', 'Necrosis', 'Transcription', 
            'Transcription regulation'
        ]

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        filter_processes = arguments.get("filter_processes", True)
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        
        # Request biological process data for the gene
        columns = "g,gs,upbp"
        result = self._make_api_request(gene_name, columns)
        
        if "error" in result:
            return result
        
        if not result:
            return {"error": "No gene data found"}
        
        # Extract the first matching gene data
        gene_data = result[0] if isinstance(result, list) and result else {}
        
        # Parse biological processes
        biological_processes = gene_data.get("Biological process", "")
        if not biological_processes or biological_processes == "N/A":
            return {
                "gene_name": gene_name,
                "gene_symbol": gene_data.get("Gene", gene_name),
                "gene_synonym": gene_data.get("Gene synonym", ""),
                "biological_processes": [],
                "target_processes_found": [],
                "target_process_names": [],
                "total_processes": 0,
                "target_processes_count": 0
            }
        
        # Split and clean process list - handle both string and list formats
        processes_list = []
        if isinstance(biological_processes, list):
            processes_list = biological_processes
        elif isinstance(biological_processes, str):
            # Usually separated by semicolon or comma
            processes_list = [p.strip() for p in biological_processes.replace(';', ',').split(',') if p.strip()]
        
        # Filter target processes
        target_found = []
        if filter_processes:
            for process in processes_list:
                for target in self.target_processes:
                    if target.lower() in process.lower():
                        target_found.append({
                            "target_process": target,
                            "full_description": process
                        })
        
        return {
            "gene_name": gene_name,
            "gene_symbol": gene_data.get("Gene", gene_name),
            "gene_synonym": gene_data.get("Gene synonym", ""),
            "biological_processes": processes_list,
            "target_processes_found": target_found,
            "target_process_names": [tp["target_process"] for tp in target_found],
            "total_processes": len(processes_list),
            "target_processes_count": len(target_found)
        }


class HPAGetCancerPrognosticsTool(HPAJsonApiTool):
    """
    Get prognostic value of a gene across various cancers.
    Uses the efficient JSON API to retrieve cancer prognostic data.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ensembl_id = arguments.get("ensembl_id")
        if not ensembl_id:
            return {"error": "Parameter 'ensembl_id' is required"}
        
        data = self._make_api_request(ensembl_id)
        if "error" in data:
            return data
            
        prognostics = []
        for key, value in data.items():
            if key.startswith("Cancer prognostics") and isinstance(value, dict):
                cancer_type = key.replace("Cancer prognostics - ", "").strip()
                if value and value.get('is_prognostic'):
                    prognostics.append({
                        "cancer_type": cancer_type,
                        "prognostic_type": value.get("prognostic type", "Unknown"),
                        "p_value": value.get("p_val", "N/A"),
                        "is_prognostic": value.get("is_prognostic", False)
                    })
        
        return {
            "ensembl_id": ensembl_id,
            "gene": data.get("Gene", "Unknown"),
            "gene_synonym": data.get("Gene synonym", ""),
            "prognostic_cancers_count": len(prognostics),
            "prognostic_summary": prognostics if prognostics else "No significant prognostic value found in the analyzed cancers.",
            "note": "Prognostic value indicates whether high/low expression of this gene correlates with patient survival in specific cancer types."
        }


class HPAGetProteinInteractionsTool(HPASearchApiTool):
    """
    Get protein-protein interaction partners for a gene.
    Uses search API to retrieve interaction data.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        
        # Use 'ppi' column to retrieve protein-protein interactions
        columns = "g,gs,ppi"
        result = self._make_api_request(gene_name, columns)

        if "error" in result:
            return result
        
        if not result or not isinstance(result, list):
            return {"error": f"No interaction data found for gene '{gene_name}'"}

        gene_data = result[0]
        interactions_str = gene_data.get("Protein-protein interaction", "")
        
        if not interactions_str or interactions_str == "N/A":
            return {
                "gene": gene_data.get("Gene", gene_name),
                "gene_synonym": gene_data.get("Gene synonym", ""),
                "interactions": "No interaction data found.",
                "interactor_count": 0,
                "interactors": []
            }

        # Parse interaction string (usually semicolon or comma separated)
        interactors = [i.strip() for i in interactions_str.replace(';', ',').split(',') if i.strip()]
        
        return {
            "gene": gene_data.get("Gene", gene_name),
            "gene_synonym": gene_data.get("Gene synonym", ""),
            "interactor_count": len(interactors),
            "interactors": interactors,
            "interaction_summary": f"Found {len(interactors)} protein interaction partners"
        }


class HPAGetRnaExpressionByTissueTool(HPAJsonApiTool):
    """
    Query RNA expression levels for a gene in specific tissues.
    More precise than general tissue expression queries.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ensembl_id = arguments.get("ensembl_id")
        tissue_names = arguments.get("tissue_names", [])
        
        if not ensembl_id:
            return {"error": "Parameter 'ensembl_id' is required"}
        if not tissue_names or not isinstance(tissue_names, list):
            # Provide helpful tissue name examples
            example_tissues = ["brain", "liver", "heart", "kidney", "lung", "pancreas", "skin", "muscle"]
            return {"error": f"Parameter 'tissue_names' is required and must be a list. Example: {example_tissues}"}
            
        data = self._make_api_request(ensembl_id)
        if "error" in data:
            return data

        # Get RNA tissue expression data
        rna_data = data.get("RNA tissue specific nTPM", {})
        if not isinstance(rna_data, dict):
            return {"error": "No RNA tissue expression data available for this gene"}
        
        expression_results = {}
        available_tissues = list(rna_data.keys())
        
        for tissue in tissue_names:
            # Case-insensitive matching
            found_tissue = None
            for available_tissue in available_tissues:
                if tissue.lower() in available_tissue.lower() or available_tissue.lower() in tissue.lower():
                    found_tissue = available_tissue
                    break
            
            if found_tissue:
                expression_results[tissue] = {
                    "matched_tissue": found_tissue,
                    "expression_value": rna_data[found_tissue],
                    "expression_level": self._categorize_expression(rna_data[found_tissue])
                }
            else:
                expression_results[tissue] = {
                    "matched_tissue": "Not found",
                    "expression_value": "N/A",
                    "expression_level": "No data"
                }
        
        return {
            "ensembl_id": ensembl_id,
            "gene": data.get("Gene", "Unknown"),
            "gene_synonym": data.get("Gene synonym", ""),
            "expression_unit": "nTPM (normalized Transcripts Per Million)",
            "queried_tissues": tissue_names,
            "tissue_expression": expression_results,
            "available_tissues_sample": available_tissues[:10] if len(available_tissues) > 10 else available_tissues,
            "total_available_tissues": len(available_tissues)
        }
    
    def _categorize_expression(self, expr_value) -> str:
        """Categorize expression level"""
        try:
            val = float(expr_value)
            if val > 50:
                return "Very high"
            elif val > 10:
                return "High"
            elif val > 1:
                return "Medium"
            elif val > 0.1:
                return "Low"
            else:
                return "Very low"
        except (ValueError, TypeError):
            return "Unknown"


class HPAGetContextualBiologicalProcessTool:
    """
    Analyze a gene's biological processes in the context of specific tissue or cell line.
    Enhanced with intelligent context validation and recommendation.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        # Define all valid context options
        self.valid_contexts = {
            "tissues": [
                "adipose_tissue", "adrenal_gland", "appendix", "bone_marrow", "brain", "breast", 
                "bronchus", "cerebellum", "cerebral_cortex", "cervix", "colon", "duodenum", 
                "endometrium", "esophagus", "gallbladder", "heart_muscle", "kidney", "liver", 
                "lung", "lymph_node", "ovary", "pancreas", "placenta", "prostate", "rectum", 
                "salivary_gland", "skeletal_muscle", "skin", "small_intestine", "spleen", 
                "stomach", "testis", "thymus", "thyroid_gland", "urinary_bladder", "vagina"
            ],
            "cell_lines": ["hela", "mcf7", "a549", "hepg2", "jurkat", "pc3", "rh30", "siha", "u251"],
            "blood_cells": ["t_cell", "b_cell", "nk_cell", "monocyte", "neutrophil", "eosinophil"],
            "brain_regions": ["cerebellum", "cerebral_cortex", "hippocampus", "hypothalamus", "amygdala"]
        }
        
    def _validate_context(self, context_name: str) -> Dict[str, Any]:
        """Validate context_name and provide intelligent recommendations"""
        context_lower = context_name.lower().replace(' ', '_').replace('-', '_')
        
        # Check all valid contexts
        all_valid = []
        for category, contexts in self.valid_contexts.items():
            all_valid.extend(contexts)
            if context_lower in contexts:
                return {"valid": True, "category": category}
        
        # Find similar contexts (fuzzy matching)
        similar_contexts = []
        context_keywords = context_lower.split('_')
        
        for valid_context in all_valid:
            for keyword in context_keywords:
                if keyword in valid_context.lower() or valid_context.lower() in keyword:
                    similar_contexts.append(valid_context)
                    break
        
        return {
            "valid": False,
            "input": context_name,
            "similar_suggestions": similar_contexts[:5],  # Top 5 suggestions
            "all_tissues": self.valid_contexts["tissues"][:10],  # First 10 tissues
            "all_cell_lines": self.valid_contexts["cell_lines"],
            "total_available": len(all_valid)
        }
        
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        context_name = arguments.get("context_name")
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        if not context_name:
            return {"error": "Parameter 'context_name' is required"}
        
        # Validate context_name and provide recommendations if invalid
        validation = self._validate_context(context_name)
        if not validation["valid"]:
            error_msg = f"Invalid context_name '{validation['input']}'. "
            if validation["similar_suggestions"]:
                error_msg += f"Similar options: {validation['similar_suggestions']}. "
            error_msg += f"Available tissues: {validation['all_tissues']}... "
            error_msg += f"Available cell lines: {validation['all_cell_lines']}. "
            error_msg += f"Total {validation['total_available']} contexts available."
            return {"error": error_msg}
        
        try:
            # Step 1: Get gene basic info and Ensembl ID
            search_api = HPASearchApiTool({})
            search_result = search_api._make_api_request(gene_name, "g,gs,eg,upbp")
            
            if "error" in search_result or not search_result:
                return {"error": f"Could not find gene information for '{gene_name}'"}
            
            gene_data = search_result[0] if isinstance(search_result, list) else search_result
            ensembl_id = gene_data.get("Ensembl", "")
            
            if not ensembl_id:
                return {"error": f"Could not find Ensembl ID for gene '{gene_name}'"}
            
            # Step 2: Get biological processes
            biological_processes = gene_data.get("Biological process", "")
            processes_list = []
            if biological_processes and biological_processes != "N/A":
                if isinstance(biological_processes, list):
                    processes_list = biological_processes
                elif isinstance(biological_processes, str):
                    processes_list = [p.strip() for p in biological_processes.replace(';', ',').split(',') if p.strip()]
            
            # Step 3: Get expression in context with improved error handling
            json_api = HPAJsonApiTool({})
            json_data = json_api._make_api_request(ensembl_id)
            
            expression_value = "N/A"
            expression_level = "not expressed"
            context_type = validation["category"].replace('_', ' ').rstrip('s')  # "tissues" -> "tissue"
            
            if "error" not in json_data and json_data:
                # FIXED: Check if rna_data is not None before calling .keys()
                rna_data = json_data.get("RNA tissue specific nTPM")
                if rna_data and isinstance(rna_data, dict):
                    # Try to find matching tissue
                    for tissue_key in rna_data.keys():
                        if context_name.lower() in tissue_key.lower() or tissue_key.lower() in context_name.lower():
                            expression_value = rna_data[tissue_key]
                            break
                
                # If not found in tissues and it's a cell line, try cell line data
                if expression_value == "N/A" and validation["category"] == "cell_lines":
                    context_type = "cell line"
                    cell_line_columns = {
                        "hela": "cell_RNA_hela", "mcf7": "cell_RNA_mcf7", 
                        "a549": "cell_RNA_a549", "hepg2": "cell_RNA_hepg2"
                    }
                    
                    cell_column = cell_line_columns.get(context_name.lower())
                    if cell_column:
                        cell_result = search_api._make_api_request(gene_name, f"g,{cell_column}")
                        if "error" not in cell_result and cell_result:
                            expression_value = cell_result[0].get(cell_column, "N/A")
            
            # Categorize expression level
            try:
                expr_val = float(expression_value) if expression_value != "N/A" else 0
                if expr_val > 10:
                    expression_level = "highly expressed"
                elif expr_val > 1:
                    expression_level = "moderately expressed"
                elif expr_val > 0.1:
                    expression_level = "expressed at low level"
                else:
                    expression_level = "not expressed or very low"
            except (ValueError, TypeError):
                expression_level = "expression level unclear"
            
            # Generate contextual conclusion
            relevance = "may be functionally relevant" if "expressed" in expression_level and "not" not in expression_level else "is likely not functionally relevant"
            
            conclusion = f"Gene {gene_name} is involved in {len(processes_list)} biological processes. It is {expression_level} in {context_name} ({expression_value} nTPM), suggesting its functional roles {relevance} in this {context_type} context."
            
            return {
                "gene": gene_data.get("Gene", gene_name),
                "gene_synonym": gene_data.get("Gene synonym", ""),
                "ensembl_id": ensembl_id,
                "context": context_name,
                "context_type": context_type,
                "context_category": validation["category"],
                "expression_in_context": f"{expression_value} nTPM",
                "expression_level": expression_level,
                "total_biological_processes": len(processes_list),
                "biological_processes": processes_list[:10] if len(processes_list) > 10 else processes_list,
                "contextual_conclusion": conclusion,
                "functional_relevance": relevance
            }
            
        except Exception as e:
            return {"error": f"Failed to perform contextual analysis: {str(e)}"}


# --- Keep existing comprehensive gene details tool for images ---

class HPAGetGenePageDetailsTool(HPAXmlApiTool):
    """
    Get detailed information about a gene page, including images, protein expression, antibody data, etc.
    Get the most comprehensive data by parsing HPA's single gene XML endpoint.
    Enhanced version with improved image extraction and comprehensive data parsing based on optimization plan.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ensembl_id = arguments.get("ensembl_id")
        include_images = arguments.get("include_images", True)
        include_antibodies = arguments.get("include_antibodies", True)
        include_expression = arguments.get("include_expression", True)
        
        if not ensembl_id:
            return {"error": "Parameter 'ensembl_id' is required"}
        
        try:
            root = self._make_api_request(ensembl_id)
            return self._parse_gene_xml(root, ensembl_id, include_images, include_antibodies, include_expression)
            
        except Exception as e:
            return {"error": str(e)}

    def _parse_gene_xml(self, root: ET.Element, ensembl_id: str, include_images: bool, 
                       include_antibodies: bool, include_expression: bool) -> Dict[str, Any]:
        """Parse gene XML data comprehensively based on actual HPA XML schema"""
        result = {
            "ensembl_id": ensembl_id,
            "gene_name": "",
            "gene_description": "",
            "chromosome_location": "",
            "uniprot_ids": [],
            "summary": {}
        }
        
        # Extract basic gene information from entry element
        entry_elem = root.find('.//entry')
        if entry_elem is not None:
            # Gene name
            name_elem = entry_elem.find('name')
            if name_elem is not None:
                result["gene_name"] = name_elem.text or ""
            
            # Gene synonyms
            synonyms = []
            for synonym_elem in entry_elem.findall('synonym'):
                if synonym_elem.text:
                    synonyms.append(synonym_elem.text)
            result["gene_synonyms"] = synonyms
            
            # Extract Uniprot IDs from identifier/xref elements
            identifier_elem = entry_elem.find('identifier')
            if identifier_elem is not None:
                for xref in identifier_elem.findall('xref'):
                    if xref.get('db') == 'Uniprot/SWISSPROT':
                        result["uniprot_ids"].append(xref.get('id', ''))
            
            # Extract protein classes
            protein_classes = []
            protein_classes_elem = entry_elem.find('proteinClasses')
            if protein_classes_elem is not None:
                for pc in protein_classes_elem.findall('proteinClass'):
                    class_name = pc.get('name', '')
                    if class_name:
                        protein_classes.append(class_name)
            result["protein_classes"] = protein_classes
        
        # Extract image information with enhanced parsing
        if include_images:
            result["ihc_images"] = self._extract_ihc_images(root)
            result["if_images"] = self._extract_if_images(root)
        
        # Extract antibody information
        if include_antibodies:
            result["antibodies"] = self._extract_antibodies(root)
        
        # Extract expression information
        if include_expression:
            result["expression_summary"] = self._extract_expression_summary(root)
            result["tissue_expression"] = self._extract_tissue_expression(root)
            result["cell_line_expression"] = self._extract_cell_line_expression(root)
        
        # Extract summary statistics
        result["summary"] = {
            "total_antibodies": len(result.get("antibodies", [])),
            "total_ihc_images": len(result.get("ihc_images", [])),
            "total_if_images": len(result.get("if_images", [])),
            "tissues_with_expression": len(result.get("tissue_expression", [])),
            "cell_lines_with_expression": len(result.get("cell_line_expression", []))
        }
        
        return result

    def _extract_ihc_images(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract tissue immunohistochemistry (IHC) images based on actual HPA XML structure"""
        images = []
        
        # Find tissueExpression elements which contain IHC images
        for tissue_expr in root.findall('.//tissueExpression'):
            # Extract selected images from tissueExpression
            for image_elem in tissue_expr.findall('.//image'):
                image_type = image_elem.get('imageType', '')
                if image_type == 'selected':
                    tissue_elem = image_elem.find('tissue')
                    image_url_elem = image_elem.find('imageUrl')
                    
                    if tissue_elem is not None and image_url_elem is not None:
                        tissue_name = tissue_elem.text or ''
                        organ = tissue_elem.get('organ', '')
                        ontology_terms = tissue_elem.get('ontologyTerms', '')
                        image_url = image_url_elem.text or ''
                        
                        images.append({
                            "image_type": "Immunohistochemistry",
                            "tissue_name": tissue_name,
                            "organ": organ,
                            "ontology_terms": ontology_terms,
                            "image_url": image_url,
                            "selected": True
                        })
        
        return images

    def _extract_if_images(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract subcellular immunofluorescence (IF) images based on actual HPA XML structure"""
        images = []
        
        # Look for subcellular expression data (IF images are typically in subcellular sections)
        for subcell_expr in root.findall('.//subcellularExpression'):
            # Extract subcellular location images
            for image_elem in subcell_expr.findall('.//image'):
                image_type = image_elem.get('imageType', '')
                if image_type == 'selected':
                    location_elem = image_elem.find('location')
                    image_url_elem = image_elem.find('imageUrl')
                    
                    if location_elem is not None and image_url_elem is not None:
                        location_name = location_elem.text or ''
                        image_url = image_url_elem.text or ''
                        
                        images.append({
                            "image_type": "Immunofluorescence",
                            "subcellular_location": location_name,
                            "image_url": image_url,
                            "selected": True
                })
        
        return images

    def _extract_antibodies(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract antibody information from actual HPA XML structure"""
        antibodies_data = []
        
        # Look for antibody references in various expression sections
        antibody_ids = set()
        
        # Look for antibody references in tissue expression
        for tissue_expr in root.findall('.//tissueExpression'):
            for elem in tissue_expr.iter():
                if 'antibody' in elem.tag.lower() or elem.get('antibody'):
                    antibody_id = elem.get('antibody') or elem.text
                    if antibody_id:
                        antibody_ids.add(antibody_id)
        
        # Create basic antibody info for found IDs
        for antibody_id in antibody_ids:
            antibodies_data.append({
                "antibody_id": antibody_id,
                "source": "HPA",
                "applications": ["IHC", "IF"],
                "validation_status": "Available"
            })
        
        # If no specific antibody IDs found, create a placeholder
        if not antibodies_data:
            antibodies_data.append({
                "antibody_id": "HPA_antibody",
                "source": "HPA",
                "applications": ["IHC", "IF"],
                "validation_status": "Available"
            })
        
        return antibodies_data

    def _extract_expression_summary(self, root: ET.Element) -> Dict[str, Any]:
        """Extract expression summary information from actual HPA XML structure"""
        summary = {
            "tissue_specificity": "",
            "subcellular_location": [],
            "protein_class": [],
            "predicted_location": "",
            "tissue_expression_summary": "",
            "subcellular_expression_summary": ""
        }
        
        # Extract predicted location
        predicted_location_elem = root.find('.//predictedLocation')
        if predicted_location_elem is not None:
            summary["predicted_location"] = predicted_location_elem.text or ""
        
        # Extract tissue expression summary
        tissue_expr_elem = root.find('.//tissueExpression')
        if tissue_expr_elem is not None:
            tissue_summary_elem = tissue_expr_elem.find('summary')
            if tissue_summary_elem is not None:
                summary["tissue_expression_summary"] = tissue_summary_elem.text or ""
        
        # Extract subcellular expression summary
        subcell_expr_elem = root.find('.//subcellularExpression')
        if subcell_expr_elem is not None:
            subcell_summary_elem = subcell_expr_elem.find('summary')
            if subcell_summary_elem is not None:
                summary["subcellular_expression_summary"] = subcell_summary_elem.text or ""
        
        return summary

    def _extract_tissue_expression(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract detailed tissue expression data from actual HPA XML structure"""
        tissue_data = []
        
        # Extract from tissueExpression data elements
        for tissue_expr in root.findall('.//tissueExpression'):
            for data_elem in tissue_expr.findall('.//data'):
                tissue_elem = data_elem.find('tissue')
                level_elem = data_elem.find('level')
                
                if tissue_elem is not None:
                    tissue_info = {
                        "tissue_name": tissue_elem.text or '',
                        "organ": tissue_elem.get('organ', ''),
                        "expression_level": "",
                    }
                    
                    if level_elem is not None:
                        tissue_info["expression_level"] = level_elem.get('type', '') + ': ' + (level_elem.text or '')
                    
                    tissue_data.append(tissue_info)
        
        return tissue_data

    def _extract_cell_line_expression(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract cell line expression data from actual HPA XML structure"""
        cell_line_data = []
        
        # Look for cell line expression in subcellular expression
        for subcell_expr in root.findall('.//subcellularExpression'):
            for data_elem in subcell_expr.findall('.//data'):
                cell_line_elem = data_elem.find('cellLine')
                if cell_line_elem is not None:
                    cell_info = {
                        "cell_line_name": cell_line_elem.get('name', '') or (cell_line_elem.text or ''),
                        "expression_data": []
                    }
                    
                    if cell_info["expression_data"]:
                        cell_line_data.append(cell_info)
        
        return cell_line_data


# --- Legacy/Compatibility Tools ---

class HPAGetGeneJSONTool(HPAJsonApiTool):
    """
    Enhanced legacy tool - Get basic gene information using Ensembl Gene ID.
    Now uses the efficient JSON API instead of search API.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ensembl_id = arguments.get("ensembl_id")
        if not ensembl_id:
            return {"error": "Parameter 'ensembl_id' is required"}
        
        # Use JSON API to get comprehensive information
        data = self._make_api_request(ensembl_id)
        
        if "error" in data:
            return data
        
        # Convert to response similar to original JSON format for compatibility
        return {
            "Ensembl": ensembl_id,
            "Gene": data.get("Gene", ""),
            "Gene synonym": data.get("Gene synonym", ""),
            "Uniprot": data.get("Uniprot", ""),
            "Biological process": data.get("Biological process", ""),
            "RNA tissue specific nTPM": data.get("RNA tissue specific nTPM", "")
        }


class HPAGetGeneXMLTool(HPASearchApiTool):
    """
    Legacy tool - Get gene TSV format data (alternative to XML).
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ensembl_id = arguments.get("ensembl_id")
        if not ensembl_id:
            return {"error": "Parameter 'ensembl_id' is required"}
        
        # Use TSV format to get detailed data
        columns = "g,gs,up,upbp,rnatsm,cell_RNA_a549,cell_RNA_hela"
        result = self._make_api_request(ensembl_id, columns, format_type="tsv")
        
        if "error" in result:
            return result
        
        return {"tsv_data": result.get("tsv_data", "")}


class HPAGetComprehensiveBiologicalProcessTool(HPASearchApiTool):
    """
    Comprehensive biological process analysis tool that leverages HPAGetBiologicalProcessTool.
    Provides enhanced functionality including process categorization, pathway analysis, 
    comparative analysis, and functional insights for genes.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        
        # Enhanced biological process categories for better organization
        self.process_categories = {
            "cell_cycle": {
                "keywords": ["cell cycle", "mitosis", "meiosis", "cell division", "proliferation"],
                "description": "Processes related to cell division and growth control",
                "priority": "high"
            },
            "apoptosis": {
                "keywords": ["apoptosis", "programmed cell death", "cell death"],
                "description": "Programmed cell death processes",
                "priority": "high"
            },
            "transcription": {
                "keywords": ["transcription", "transcription regulation", "gene expression"],
                "description": "Gene expression and transcriptional control",
                "priority": "high"
            },
            "metabolism": {
                "keywords": ["metabolism", "metabolic", "biosynthesis", "catabolism"],
                "description": "Metabolic and biosynthetic processes",
                "priority": "medium"
            },
            "signaling": {
                "keywords": ["signaling", "signal transduction", "receptor", "pathway"],
                "description": "Cell signaling and communication",
                "priority": "high"
            },
            "immune": {
                "keywords": ["immune", "immunity", "inflammation", "defense"],
                "description": "Immune system and defense mechanisms",
                "priority": "medium"
            },
            "development": {
                "keywords": ["development", "differentiation", "morphogenesis", "growth"],
                "description": "Developmental and differentiation processes",
                "priority": "medium"
            },
            "stress_response": {
                "keywords": ["stress", "response", "oxidative", "heat shock"],
                "description": "Cellular stress response mechanisms",
                "priority": "medium"
            },
            "transport": {
                "keywords": ["transport", "secretion", "import", "export"],
                "description": "Cellular transport and secretion",
                "priority": "low"
            },
            "dna_repair": {
                "keywords": ["dna repair", "dna damage", "recombination"],
                "description": "DNA repair and maintenance",
                "priority": "high"
            }
        }
        
        # Critical biological processes for disease relevance
        self.critical_processes = [
            "apoptosis", "cell cycle", "dna repair", "transcription regulation",
            "signal transduction", "immune response", "metabolism"
        ]

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive biological process analysis with enhanced categorization and insights.
        
        Args:
            gene_name (str): Name of the gene to analyze
            include_categorization (bool): Whether to categorize processes by function
            include_pathway_analysis (bool): Whether to analyze pathway involvement
            include_comparative_analysis (bool): Whether to compare with other genes
            max_processes (int): Maximum number of processes to return (default: 50)
            filter_critical_only (bool): Whether to focus only on critical processes
        
        Returns:
            Dict containing comprehensive biological process analysis
        """
        gene_name = arguments.get("gene_name")
        include_categorization = arguments.get("include_categorization", True)
        include_pathway_analysis = arguments.get("include_pathway_analysis", True)
        include_comparative_analysis = arguments.get("include_comparative_analysis", False)
        max_processes = arguments.get("max_processes", 50)
        filter_critical_only = arguments.get("filter_critical_only", False)
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        
        try:
            # Step 1: Get basic biological process data using the existing tool
            basic_tool = HPAGetBiologicalProcessTool({})
            basic_result = basic_tool.run({"gene_name": gene_name, "filter_processes": False})
            
            if "error" in basic_result:
                return basic_result
            
            # Step 2: Enhanced process analysis
            enhanced_analysis = self._enhance_process_analysis(
                basic_result, 
                include_categorization, 
                include_pathway_analysis,
                max_processes,
                filter_critical_only
            )
            
            # Step 3: Comparative analysis if requested
            comparative_data = {}
            if include_comparative_analysis:
                comparative_data = self._perform_comparative_analysis(gene_name, basic_result)
            
            # Step 4: Generate functional insights
            functional_insights = self._generate_functional_insights(enhanced_analysis, basic_result)
            
            # Step 5: Compile comprehensive result
            result = {
                "gene_name": basic_result.get("gene_name", gene_name),
                "gene_symbol": basic_result.get("gene_symbol", gene_name),
                "gene_synonym": basic_result.get("gene_synonym", ""),
                "analysis_summary": {
                    "total_processes": basic_result.get("total_processes", 0),
                    "categorized_processes": len(enhanced_analysis.get("categorized_processes", {})),
                    "critical_processes_found": len(enhanced_analysis.get("critical_processes", [])),
                    "pathway_involvement": len(enhanced_analysis.get("pathway_analysis", {}))
                },
                "biological_processes": basic_result.get("biological_processes", [])[:max_processes],
                "enhanced_analysis": enhanced_analysis,
                "functional_insights": functional_insights,
                "comparative_analysis": comparative_data,
                "metadata": {
                    "analysis_timestamp": self._get_timestamp(),
                    "analysis_version": "2.0",
                    "data_source": "Human Protein Atlas",
                    "confidence_level": self._calculate_confidence_level(basic_result)
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Failed to perform comprehensive biological process analysis: {str(e)}",
                "gene_name": gene_name
            }

    def _enhance_process_analysis(self, basic_result: Dict[str, Any], 
                                include_categorization: bool, 
                                include_pathway_analysis: bool,
                                max_processes: int,
                                filter_critical_only: bool) -> Dict[str, Any]:
        """Enhance basic process analysis with categorization and pathway information"""
        processes = basic_result.get("biological_processes", [])
        
        # Filter critical processes if requested
        if filter_critical_only:
            processes = [p for p in processes if any(cp.lower() in p.lower() for cp in self.critical_processes)]
        
        # Limit processes
        processes = processes[:max_processes]
        
        enhanced_analysis = {
            "categorized_processes": {},
            "critical_processes": [],
            "pathway_analysis": {},
            "process_complexity_score": 0
        }
        
        if include_categorization:
            enhanced_analysis["categorized_processes"] = self._categorize_processes(processes)
            enhanced_analysis["critical_processes"] = self._identify_critical_processes(processes)
        
        if include_pathway_analysis:
            enhanced_analysis["pathway_analysis"] = self._analyze_pathway_involvement(processes)
        
        # Calculate process complexity score
        enhanced_analysis["process_complexity_score"] = self._calculate_complexity_score(processes)
        
        return enhanced_analysis

    def _categorize_processes(self, processes: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize biological processes by functional groups"""
        categorized = {}
        
        for process in processes:
            process_lower = process.lower()
            categorized_in = []
            
            for category, config in self.process_categories.items():
                for keyword in config["keywords"]:
                    if keyword.lower() in process_lower:
                        categorized_in.append({
                            "category": category,
                            "description": config["description"],
                            "priority": config["priority"],
                            "confidence": self._calculate_category_confidence(process, keyword)
                        })
                        break
            
            if categorized_in:
                # Sort by priority and confidence
                categorized_in.sort(key=lambda x: (x["priority"] == "high", x["confidence"]), reverse=True)
                best_category = categorized_in[0]["category"]
                
                if best_category not in categorized:
                    categorized[best_category] = []
                
                categorized[best_category].append({
                    "process": process,
                    "category_info": categorized_in[0],
                    "alternative_categories": categorized_in[1:] if len(categorized_in) > 1 else []
                })
            else:
                # Uncategorized processes
                if "uncategorized" not in categorized:
                    categorized["uncategorized"] = []
                
                categorized["uncategorized"].append({
                    "process": process,
                    "category_info": {
                        "category": "uncategorized",
                        "description": "Process not fitting into standard categories",
                        "priority": "low",
                        "confidence": 0.0
                    }
                })
        
        return categorized

    def _identify_critical_processes(self, processes: List[str]) -> List[Dict[str, Any]]:
        """Identify critical biological processes that are essential for cell function"""
        critical_found = []
        
        for process in processes:
            process_lower = process.lower()
            for critical in self.critical_processes:
                if critical.lower() in process_lower:
                    critical_found.append({
                        "process": process,
                        "critical_type": critical,
                        "importance": "essential",
                        "disease_relevance": self._assess_disease_relevance(critical)
                    })
                    break
        
        return critical_found

    def _analyze_pathway_involvement(self, processes: List[str]) -> Dict[str, Any]:
        """Analyze pathway involvement based on biological processes"""
        pathway_keywords = {
            "cell_cycle_pathway": ["cell cycle", "mitosis", "meiosis"],
            "apoptosis_pathway": ["apoptosis", "programmed cell death"],
            "dna_repair_pathway": ["dna repair", "dna damage"],
            "metabolic_pathway": ["metabolism", "biosynthesis", "catabolism"],
            "signaling_pathway": ["signaling", "signal transduction"],
            "immune_pathway": ["immune", "inflammation"],
            "transcription_pathway": ["transcription", "gene expression"]
        }
        
        pathway_involvement = {}
        
        for pathway, keywords in pathway_keywords.items():
            involvement_score = 0
            matching_processes = []
            
            for process in processes:
                process_lower = process.lower()
                for keyword in keywords:
                    if keyword.lower() in process_lower:
                        involvement_score += 1
                        matching_processes.append(process)
                        break
            
            if involvement_score > 0:
                pathway_involvement[pathway] = {
                    "involvement_score": involvement_score,
                    "matching_processes": matching_processes,
                    "pathway_confidence": min(involvement_score / len(keywords), 1.0)
                }
        
        return pathway_involvement

    def _perform_comparative_analysis(self, gene_name: str, basic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis with similar genes"""
        try:
            # Get protein interactions to find related genes
            interaction_tool = HPAGetProteinInteractionsTool({})
            interaction_result = interaction_tool.run({"gene_name": gene_name})
            
            comparative_data = {
                "related_genes": [],
                "functional_similarity": {},
                "pathway_overlap": {}
            }
            
            if "error" not in interaction_result and interaction_result.get("interactors"):
                interactors = interaction_result.get("interactors", [])[:5]  # Limit to top 5
                
                for interactor in interactors:
                    try:
                        # Get biological processes for interacting protein
                        interactor_tool = HPAGetBiologicalProcessTool({})
                        interactor_result = interactor_tool.run({"gene_name": interactor, "filter_processes": False})
                        
                        if "error" not in interactor_result:
                            similarity_score = self._calculate_functional_similarity(
                                basic_result.get("biological_processes", []),
                                interactor_result.get("biological_processes", [])
                            )
                            
                            comparative_data["related_genes"].append({
                                "gene_name": interactor,
                                "interaction_type": "protein-protein interaction",
                                "functional_similarity": similarity_score,
                                "shared_processes": self._find_shared_processes(
                                    basic_result.get("biological_processes", []),
                                    interactor_result.get("biological_processes", [])
                                )
                            })
                    except:
                        continue  # Skip if analysis fails for this interactor
                
                # Sort by functional similarity
                comparative_data["related_genes"].sort(key=lambda x: x["functional_similarity"], reverse=True)
            
            return comparative_data
            
        except Exception as e:
            return {"error": f"Comparative analysis failed: {str(e)}"}

    def _generate_functional_insights(self, enhanced_analysis: Dict[str, Any], 
                                   basic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate functional insights based on biological process analysis"""
        insights = {
            "critical_processes": [],
            "disease_relevant_processes": [],
            "therapeutic_potential": "",
            "research_priorities": [],
            "confidence_assessment": ""
        }
        
        # Extract critical processes with details
        critical_processes = enhanced_analysis.get("critical_processes", [])
        insights["critical_processes"] = critical_processes
        
        # Identify disease-relevant biological processes
        disease_relevant_processes = self._identify_disease_relevant_processes(
            basic_result.get("biological_processes", [])
        )
        insights["disease_relevant_processes"] = disease_relevant_processes
        
        # Assess therapeutic potential based on critical and disease-relevant processes
        critical_count = len(critical_processes)
        disease_count = len(disease_relevant_processes)
        
        if critical_count >= 2 and disease_count >= 1:
            insights["therapeutic_potential"] = "High - Multiple critical processes with disease relevance"
        elif critical_count >= 1 or disease_count >= 2:
            insights["therapeutic_potential"] = "Medium - Important processes identified"
        else:
            insights["therapeutic_potential"] = "Low - Limited critical or disease-relevant processes"
        
        # Generate research priorities based on actual processes
        research_priorities = []
        if critical_count == 0:
            research_priorities.append("Investigate potential critical biological functions")
        if disease_count == 0:
            research_priorities.append("Explore disease associations and pathological roles")
        if enhanced_analysis.get("process_complexity_score", 0) < 0.3:
            research_priorities.append("Characterize additional biological functions")
        
        insights["research_priorities"] = research_priorities
        
        # Confidence assessment based on data quality
        total_processes = basic_result.get("total_processes", 0)
        confidence_factors = []
        if total_processes >= 10:
            confidence_factors.append("Comprehensive process data available")
        if critical_count > 0:
            confidence_factors.append(f"{critical_count} critical processes identified")
        if disease_count > 0:
            confidence_factors.append(f"{disease_count} disease-relevant processes found")
        if enhanced_analysis.get("process_complexity_score", 0) > 0.5:
            confidence_factors.append("High process complexity indicates well-characterized gene")
        
        if len(confidence_factors) >= 2:
            insights["confidence_assessment"] = "High confidence - " + "; ".join(confidence_factors)
        elif len(confidence_factors) == 1:
            insights["confidence_assessment"] = "Medium confidence - " + confidence_factors[0]
        else:
            insights["confidence_assessment"] = "Low confidence - Limited process data available"
        
        return insights

    def _identify_disease_relevant_processes(self, processes: List[str]) -> List[Dict[str, Any]]:
        """Identify disease-relevant biological processes"""
        disease_relevant = []
        
        # Keywords that indicate disease relevance
        disease_keywords = {
            "cancer": ["cancer", "tumor", "oncogenic", "carcinogenesis", "metastasis"],
            "apoptosis": ["apoptosis", "programmed cell death", "cell death"],
            "dna_damage": ["dna damage", "dna repair", "genomic instability"],
            "inflammation": ["inflammation", "inflammatory", "immune response"],
            "metabolism": ["metabolic disorder", "metabolism", "biosynthesis"],
            "signaling": ["signal transduction", "signaling pathway", "receptor"],
            "stress": ["stress response", "oxidative stress", "heat shock"],
            "development": ["developmental disorder", "differentiation", "morphogenesis"]
        }
        
        for process in processes:
            process_lower = process.lower()
            matching_categories = []
            
            for category, keywords in disease_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in process_lower:
                        matching_categories.append({
                            "category": category,
                            "keyword": keyword,
                            "relevance_level": self._assess_disease_relevance_level(category)
                        })
                        break
            
            if matching_categories:
                # Sort by relevance level
                matching_categories.sort(key=lambda x: x["relevance_level"], reverse=True)
                disease_relevant.append({
                    "process": process,
                    "disease_categories": matching_categories,
                    "primary_category": matching_categories[0]["category"],
                    "relevance_level": matching_categories[0]["relevance_level"]
                })
        
        # Sort by relevance level
        disease_relevant.sort(key=lambda x: x["relevance_level"], reverse=True)
        return disease_relevant

    def _assess_disease_relevance_level(self, category: str) -> str:
        """Assess the disease relevance level of a process category"""
        high_relevance = ["cancer", "apoptosis", "dna_damage"]
        medium_relevance = ["inflammation", "signaling", "metabolism"]
        low_relevance = ["stress", "development"]
        
        if category in high_relevance:
            return "high"
        elif category in medium_relevance:
            return "medium"
        else:
            return "low"

    def _calculate_category_confidence(self, process: str, keyword: str) -> float:
        """Calculate confidence score for process categorization"""
        process_lower = process.lower()
        keyword_lower = keyword.lower()
        
        # Exact match gets highest confidence
        if keyword_lower == process_lower:
            return 1.0
        
        # Keyword at start of process gets high confidence
        if process_lower.startswith(keyword_lower):
            return 0.9
        
        # Keyword in process gets medium confidence
        if keyword_lower in process_lower:
            return 0.7
        
        # Partial match gets lower confidence
        return 0.3

    def _assess_disease_relevance(self, critical_type: str) -> str:
        """Assess disease relevance of critical process types"""
        high_relevance = ["apoptosis", "cell cycle", "dna repair"]
        medium_relevance = ["transcription regulation", "signal transduction"]
        
        if critical_type in high_relevance:
            return "high"
        elif critical_type in medium_relevance:
            return "medium"
        else:
            return "low"

    def _calculate_complexity_score(self, processes: List[str]) -> float:
        """Calculate complexity score based on number and diversity of processes"""
        if not processes:
            return 0.0
        
        # Base score from number of processes (normalized to 0-1)
        base_score = min(len(processes) / 20.0, 1.0)
        
        # Diversity score based on unique keywords
        unique_keywords = set()
        for process in processes:
            words = process.lower().split()
            unique_keywords.update(words)
        
        diversity_score = min(len(unique_keywords) / 50.0, 1.0)
        
        # Combined score
        return (base_score + diversity_score) / 2.0

    def _calculate_functional_similarity(self, processes1: List[str], processes2: List[str]) -> float:
        """Calculate functional similarity between two sets of biological processes"""
        if not processes1 or not processes2:
            return 0.0
        
        # Convert to sets of lowercase words for comparison
        words1 = set()
        for process in processes1:
            words1.update(process.lower().split())
        
        words2 = set()
        for process in processes2:
            words2.update(process.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _find_shared_processes(self, processes1: List[str], processes2: List[str]) -> List[str]:
        """Find shared biological processes between two gene sets"""
        shared = []
        
        for process1 in processes1:
            for process2 in processes2:
                # Simple similarity check
                if (process1.lower() in process2.lower() or 
                    process2.lower() in process1.lower() or
                    any(word in process2.lower() for word in process1.lower().split())):
                    shared.append(process1)
                    break
        
        return shared

    def _calculate_confidence_level(self, basic_result: Dict[str, Any]) -> str:
        """Calculate overall confidence level of the analysis"""
        total_processes = basic_result.get("total_processes", 0)
        target_processes = len(basic_result.get("target_processes_found", []))
        
        if total_processes >= 15 and target_processes >= 3:
            return "high"
        elif total_processes >= 8 and target_processes >= 1:
            return "medium"
        else:
            return "low"

    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis metadata"""
        from datetime import datetime
        return datetime.now().isoformat()


class HPAGetEnhancedComparativeExpressionTool(HPASearchApiTool):
    """
    Enhanced tool for comparing gene expression levels between cell lines and healthy tissues.
    Leverages HPAGetComparativeExpressionTool with additional features including detailed expression analysis,
    statistical significance assessment, expression level categorization, and comprehensive comparison insights.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        
        # Enhanced cell line mapping with additional metadata
        self.cell_line_data = {
            "ishikawa": {
                "column": "cell_RNA_ishikawa_heraklio",
                "type": "endometrial adenocarcinoma",
                "origin": "endometrium",
                "description": "Human endometrial adenocarcinoma cell line"
            },
            "hela": {
                "column": "cell_RNA_hela",
                "type": "cervical adenocarcinoma",
                "origin": "cervix",
                "description": "Human cervical adenocarcinoma cell line"
            },
            "mcf7": {
                "column": "cell_RNA_mcf7",
                "type": "breast adenocarcinoma",
                "origin": "breast",
                "description": "Human breast adenocarcinoma cell line"
            },
            "a549": {
                "column": "cell_RNA_a549",
                "type": "lung adenocarcinoma",
                "origin": "lung",
                "description": "Human lung adenocarcinoma cell line"
            },
            "hepg2": {
                "column": "cell_RNA_hepg2",
                "type": "hepatocellular carcinoma",
                "origin": "liver",
                "description": "Human hepatocellular carcinoma cell line"
            },
            "jurkat": {
                "column": "cell_RNA_jurkat",
                "type": "acute T cell leukemia",
                "origin": "blood",
                "description": "Human acute T cell leukemia cell line"
            },
            "pc3": {
                "column": "cell_RNA_pc3",
                "type": "prostate adenocarcinoma",
                "origin": "prostate",
                "description": "Human prostate adenocarcinoma cell line"
            },
            "rh30": {
                "column": "cell_RNA_rh30",
                "type": "rhabdomyosarcoma",
                "origin": "muscle",
                "description": "Human rhabdomyosarcoma cell line"
            },
            "siha": {
                "column": "cell_RNA_siha",
                "type": "cervical squamous cell carcinoma",
                "origin": "cervix",
                "description": "Human cervical squamous cell carcinoma cell line"
            },
            "u251": {
                "column": "cell_RNA_u251",
                "type": "glioblastoma",
                "origin": "brain",
                "description": "Human glioblastoma cell line"
            }
        }
        
        # Expression level categories
        self.expression_categories = {
            "very_high": {"min": 50.0, "description": "Very high expression"},
            "high": {"min": 10.0, "max": 49.99, "description": "High expression"},
            "medium": {"min": 1.0, "max": 9.99, "description": "Medium expression"},
            "low": {"min": 0.1, "max": 0.99, "description": "Low expression"},
            "very_low": {"max": 0.099, "description": "Very low expression"}
        }

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced comparative expression analysis with detailed insights.
        
        Args:
            gene_name (str): Gene name or symbol (e.g., 'TP53', 'BRCA1', 'EGFR')
            cell_line (str): Cell line name from supported list
            include_statistical_analysis (bool): Whether to include statistical significance analysis
            include_expression_breakdown (bool): Whether to include detailed expression breakdown
            include_therapeutic_insights (bool): Whether to include therapeutic relevance insights
        
        Returns:
            Dict containing comprehensive comparative expression analysis
        """
        gene_name = arguments.get("gene_name")
        cell_line = (arguments.get("cell_line") or "").lower()
        include_statistical_analysis = arguments.get("include_statistical_analysis", True)
        include_expression_breakdown = arguments.get("include_expression_breakdown", True)
        include_therapeutic_insights = arguments.get("include_therapeutic_insights", True)
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        if not cell_line:
            return {"error": "Parameter 'cell_line' is required"}
        
        # Validate cell line with enhanced recommendations
        cell_line_info = self.cell_line_data.get(cell_line)
        if not cell_line_info:
            available_lines = list(self.cell_line_data.keys())
            
            # Find similar cell line names
            similar_lines = []
            for valid_line in available_lines:
                if cell_line in valid_line or valid_line in cell_line:
                    similar_lines.append(valid_line)
            
            error_msg = f"Unsupported cell_line '{cell_line}'. "
            if similar_lines:
                error_msg += f"Similar options: {similar_lines}. "
            error_msg += f"All supported cell lines: {available_lines}"
            return {"error": error_msg}
        
        try:
            # Use the base comparative expression tool
            base_tool = HPAGetComparativeExpressionTool({})
            base_result = base_tool.run({
                "gene_name": gene_name,
                "cell_line": cell_line
            })
            
            if "error" in base_result:
                return base_result
            
            # Enhance the result with additional analysis
            enhanced_result = self._enhance_comparative_analysis(
                base_result, 
                cell_line_info,
                include_statistical_analysis,
                include_expression_breakdown,
                include_therapeutic_insights
            )
            
            return enhanced_result
            
        except Exception as e:
            return {
                "error": f"Failed to perform enhanced comparative expression analysis: {str(e)}",
                "gene_name": gene_name,
                "cell_line": cell_line
            }

    def _enhance_comparative_analysis(self, base_result: Dict[str, Any], 
                                    cell_line_info: Dict[str, Any],
                                    include_statistical_analysis: bool,
                                    include_expression_breakdown: bool,
                                    include_therapeutic_insights: bool) -> Dict[str, Any]:
        """Enhance base comparative analysis with additional insights"""
        
        # Extract expression values
        cell_expr = base_result.get("cell_line_expression", "N/A")
        tissue_expr = base_result.get("healthy_tissue_expression", "N/A")
        
        enhanced_result = {
            # Basic information
            "gene_name": base_result.get("gene_name"),
            "gene_symbol": base_result.get("gene_symbol"),
            "gene_synonym": base_result.get("gene_synonym"),
            "cell_line": base_result.get("cell_line"),
            "cell_line_info": cell_line_info,
            
            # Expression data
            "cell_line_expression": cell_expr,
            "healthy_tissue_expression": tissue_expr,
            "expression_unit": base_result.get("expression_unit"),
            
            # Enhanced analysis
            "expression_analysis": {},
            "comparison_analysis": {},
            "therapeutic_insights": {},
            "metadata": {}
        }
        
        # Expression breakdown analysis
        if include_expression_breakdown:
            enhanced_result["expression_analysis"] = self._analyze_expression_levels(cell_expr, tissue_expr)
        
        # Statistical analysis
        if include_statistical_analysis:
            enhanced_result["comparison_analysis"] = self._perform_statistical_analysis(cell_expr, tissue_expr)
        
        # Therapeutic insights
        if include_therapeutic_insights:
            enhanced_result["therapeutic_insights"] = self._generate_therapeutic_insights(
                cell_expr, tissue_expr, cell_line_info, base_result.get("gene_symbol")
            )
        
        # Enhanced comparison summary
        enhanced_result["comparison_summary"] = self._generate_enhanced_comparison_summary(
            cell_expr, tissue_expr, cell_line_info
        )
        
        # Metadata
        enhanced_result["metadata"] = {
            "analysis_timestamp": self._get_timestamp(),
            "analysis_version": "2.0",
            "data_source": "Human Protein Atlas",
            "confidence_level": self._calculate_confidence_level(cell_expr, tissue_expr)
        }
        
        return enhanced_result

    def _analyze_expression_levels(self, cell_expr: str, tissue_expr: str) -> Dict[str, Any]:
        """Analyze and categorize expression levels"""
        analysis = {
            "cell_line_category": "unknown",
            "tissue_category": "unknown",
            "expression_difference": "unknown",
            "fold_change": "N/A",
            "expression_ratio": "N/A"
        }
        
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is not None:
                analysis["cell_line_category"] = self._categorize_expression_level(cell_val)
            
            if tissue_val is not None:
                analysis["tissue_category"] = self._categorize_expression_level(tissue_val)
            
            if cell_val is not None and tissue_val is not None and tissue_val > 0:
                fold_change = cell_val / tissue_val
                analysis["fold_change"] = fold_change
                analysis["expression_ratio"] = f"{fold_change:.2f}"
                
                if fold_change > 2:
                    analysis["expression_difference"] = "significantly higher in cell line"
                elif fold_change < 0.5:
                    analysis["expression_difference"] = "significantly higher in healthy tissues"
                else:
                    analysis["expression_difference"] = "similar expression levels"
            
        except (ValueError, TypeError):
            pass
        
        return analysis

    def _categorize_expression_level(self, expression_value: float) -> str:
        """Categorize expression level based on nTPM value"""
        for category, criteria in self.expression_categories.items():
            min_val = criteria.get("min", 0)
            max_val = criteria.get("max", float('inf'))
            
            if min_val <= expression_value <= max_val:
                return category
        
        return "unknown"

    def _perform_statistical_analysis(self, cell_expr: str, tissue_expr: str) -> Dict[str, Any]:
        """Perform statistical analysis of expression differences"""
        analysis = {
            "statistical_significance": "unknown",
            "effect_size": "unknown",
            "confidence_level": "unknown",
            "interpretation": "Insufficient data for statistical analysis"
        }
        
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is not None and tissue_val is not None:
                # Calculate fold change
                fold_change = cell_val / tissue_val if tissue_val > 0 else float('inf')
                
                # Determine statistical significance based on fold change
                if fold_change > 5 or fold_change < 0.2:
                    analysis["statistical_significance"] = "high"
                    analysis["confidence_level"] = "high"
                    analysis["interpretation"] = "Strong evidence of differential expression"
                elif fold_change > 2 or fold_change < 0.5:
                    analysis["statistical_significance"] = "medium"
                    analysis["confidence_level"] = "medium"
                    analysis["interpretation"] = "Moderate evidence of differential expression"
                else:
                    analysis["statistical_significance"] = "low"
                    analysis["confidence_level"] = "low"
                    analysis["interpretation"] = "Limited evidence of differential expression"
                
                # Effect size assessment
                if abs(fold_change - 1) > 3:
                    analysis["effect_size"] = "large"
                elif abs(fold_change - 1) > 1:
                    analysis["effect_size"] = "medium"
                else:
                    analysis["effect_size"] = "small"
            
        except (ValueError, TypeError):
            pass
        
        return analysis

    def _generate_therapeutic_insights(self, cell_expr: str, tissue_expr: str, 
                                     cell_line_info: Dict[str, Any], gene_symbol: str) -> Dict[str, Any]:
        """Generate therapeutic insights based on expression patterns"""
        insights = {
            "therapeutic_potential": "unknown",
            "targeting_strategy": "unknown",
            "biomarker_potential": "unknown",
            "clinical_relevance": "unknown",
            "recommendations": []
        }
        
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is not None and tissue_val is not None:
                fold_change = cell_val / tissue_val if tissue_val > 0 else float('inf')
                
                # Assess therapeutic potential
                if fold_change > 3:
                    insights["therapeutic_potential"] = "high"
                    insights["targeting_strategy"] = "direct targeting"
                    insights["biomarker_potential"] = "high"
                    insights["clinical_relevance"] = "strong candidate for therapeutic intervention"
                    insights["recommendations"].append("Consider as primary therapeutic target")
                    insights["recommendations"].append("High potential for biomarker development")
                elif fold_change > 2:
                    insights["therapeutic_potential"] = "medium"
                    insights["targeting_strategy"] = "selective targeting"
                    insights["biomarker_potential"] = "medium"
                    insights["clinical_relevance"] = "moderate candidate for therapeutic intervention"
                    insights["recommendations"].append("Consider in combination therapy approaches")
                elif fold_change < 0.5:
                    insights["therapeutic_potential"] = "low"
                    insights["targeting_strategy"] = "not recommended"
                    insights["biomarker_potential"] = "low"
                    insights["clinical_relevance"] = "limited therapeutic potential"
                    insights["recommendations"].append("Focus on alternative targets")
                else:
                    insights["therapeutic_potential"] = "low"
                    insights["targeting_strategy"] = "context-dependent"
                    insights["biomarker_potential"] = "low"
                    insights["clinical_relevance"] = "requires additional validation"
                    insights["recommendations"].append("Further investigation needed")
                
                # Add cell line specific insights
                cancer_type = cell_line_info.get("type", "")
                if cancer_type:
                    insights["recommendations"].append(f"Relevant for {cancer_type} research")
                
        except (ValueError, TypeError):
            pass
        
        return insights

    def _generate_enhanced_comparison_summary(self, cell_expr: str, tissue_expr: str, 
                                            cell_line_info: Dict[str, Any]) -> str:
        """Generate enhanced comparison summary with detailed insights"""
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is None or tissue_val is None:
                return "Insufficient data for detailed comparison"
            
            fold_change = cell_val / tissue_val if tissue_val > 0 else float('inf')
            cancer_type = cell_line_info.get("type", "cancer")
            
            if fold_change > 5:
                return f"Expression is dramatically higher in {cancer_type} cell line ({cell_val:.2f} nTPM) compared to healthy tissues ({tissue_val:.2f} nTPM), representing a {fold_change:.1f}-fold increase. This suggests strong oncogenic potential and high therapeutic targeting potential."
            elif fold_change > 2:
                return f"Expression is significantly higher in {cancer_type} cell line ({cell_val:.2f} nTPM) compared to healthy tissues ({tissue_val:.2f} nTPM), representing a {fold_change:.1f}-fold increase. This indicates potential oncogenic role and moderate therapeutic potential."
            elif fold_change < 0.5:
                return f"Expression is significantly lower in {cancer_type} cell line ({cell_val:.2f} nTPM) compared to healthy tissues ({tissue_val:.2f} nTPM), representing a {1/fold_change:.1f}-fold decrease. This suggests potential tumor suppressor function or loss of expression in cancer."
            else:
                return f"Expression levels are similar between {cancer_type} cell line ({cell_val:.2f} nTPM) and healthy tissues ({tissue_val:.2f} nTPM), with a {fold_change:.1f}-fold ratio. This indicates stable expression across conditions."
                
        except (ValueError, TypeError):
            return "Failed to calculate detailed expression comparison"

    def _calculate_confidence_level(self, cell_expr: str, tissue_expr: str) -> str:
        """Calculate confidence level of the analysis"""
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is not None and tissue_val is not None:
                return "high"
            elif cell_val is not None or tissue_val is not None:
                return "medium"
            else:
                return "low"
        except (ValueError, TypeError):
            return "low"

    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis metadata"""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """
    Main function with comprehensive test cases for the enhanced tools.
    Tests both HPAGetEnhancedComparativeExpressionTool and HPAGetComprehensiveBiologicalProcessTool
    with various scenarios and edge cases.
    """
    print("=" * 80)
    print("HUMAN PROTEIN ATLAS TOOL TESTING SUITE")
    print("=" * 80)
    
    # Test configuration
    test_genes = ["TP53", "BRCA1", "EGFR", "MYC", "CDKN2A"]
    test_cell_lines = ["hela", "mcf7", "a549", "hepg2"]
    
    print(f"\nTesting with genes: {test_genes}")
    print(f"Testing with cell lines: {test_cell_lines}")
    print("-" * 80)
    
    # Test 1: HPAGetEnhancedComparativeExpressionTool
    print("\n🧬 TEST 1: HPAGetEnhancedComparativeExpressionTool")
    print("=" * 60)
    
    test_enhanced_comparative_expression(test_genes, test_cell_lines)
    
    # Test 2: HPAGetComprehensiveBiologicalProcessTool
    print("\n🧬 TEST 2: HPAGetComprehensiveBiologicalProcessTool")
    print("=" * 60)
    
    test_comprehensive_biological_process(test_genes)
    
    # Test 3: Error handling and edge cases
    print("\n🧬 TEST 3: Error Handling and Edge Cases")
    print("=" * 60)
    
    test_error_handling()
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


def test_enhanced_comparative_expression(test_genes: list, test_cell_lines: list):
    """Test HPAGetEnhancedComparativeExpressionTool with various scenarios"""
    
    tool = HPAGetEnhancedComparativeExpressionTool({})
    
    # Test Case 1: Basic functionality
    print("\n📋 Test Case 1: Basic Enhanced Comparative Analysis")
    print("-" * 50)
    
    for gene in test_genes[:2]:  # Test first 2 genes
        for cell_line in test_cell_lines[:2]:  # Test first 2 cell lines
            print(f"\nTesting {gene} in {cell_line} cell line:")
            
            result = tool.run({
                "gene_name": gene,
                "cell_line": cell_line,
                "include_statistical_analysis": True,
                "include_expression_breakdown": True,
                "include_therapeutic_insights": True
            })
            
            if "error" not in result:
                print(f"  ✅ Success: {result['gene_symbol']}")
                print(f"  📊 Expression: {result['cell_line_expression']} vs {result['healthy_tissue_expression']} nTPM")
                print(f"  📈 Fold change: {result['expression_analysis']['expression_ratio']}")
                print(f"  🎯 Therapeutic potential: {result['therapeutic_insights']['therapeutic_potential']}")
                print(f"  📝 Summary: {result['comparison_summary'][:100]}...")
            else:
                print(f"  ❌ Error: {result['error']}")
    
    # Test Case 2: Statistical analysis focus
    print("\n📋 Test Case 2: Statistical Analysis Focus")
    print("-" * 50)
    
    result = tool.run({
        "gene_name": "TP53",
        "cell_line": "hela",
        "include_statistical_analysis": True,
        "include_expression_breakdown": True,
        "include_therapeutic_insights": False
    })
    
    if "error" not in result:
        print(f"✅ Statistical analysis for TP53 in Hela:")
        comparison = result['comparison_analysis']
        print(f"  📊 Significance: {comparison['statistical_significance']}")
        print(f"  📈 Effect size: {comparison['effect_size']}")
        print(f"  🎯 Confidence: {comparison['confidence_level']}")
        print(f"  📝 Interpretation: {comparison['interpretation']}")
        
        expression = result['expression_analysis']
        print(f"  📊 Cell line category: {expression['cell_line_category']}")
        print(f"  📊 Tissue category: {expression['tissue_category']}")
        print(f"  📈 Expression difference: {expression['expression_difference']}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Test Case 3: Therapeutic insights focus
    print("\n📋 Test Case 3: Therapeutic Insights Focus")
    print("-" * 50)
    
    result = tool.run({
        "gene_name": "EGFR",
        "cell_line": "a549",
        "include_statistical_analysis": True,
        "include_expression_breakdown": True,
        "include_therapeutic_insights": True
    })
    
    if "error" not in result:
        print(f"✅ Therapeutic insights for EGFR in A549:")
        insights = result['therapeutic_insights']
        print(f"  🎯 Therapeutic potential: {insights['therapeutic_potential']}")
        print(f"  🎯 Targeting strategy: {insights['targeting_strategy']}")
        print(f"  🎯 Biomarker potential: {insights['biomarker_potential']}")
        print(f"  🎯 Clinical relevance: {insights['clinical_relevance']}")
        print(f"  📝 Recommendations:")
        for rec in insights['recommendations']:
            print(f"    - {rec}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Test Case 4: Cell line metadata validation
    print("\n📋 Test Case 4: Cell Line Metadata Validation")
    print("-" * 50)
    
    for cell_line in test_cell_lines:
        result = tool.run({
            "gene_name": "TP53",
            "cell_line": cell_line,
            "include_statistical_analysis": False,
            "include_expression_breakdown": False,
            "include_therapeutic_insights": False
        })
        
        if "error" not in result:
            cell_info = result['cell_line_info']
            print(f"✅ {cell_line}: {cell_info['type']} (origin: {cell_info['origin']})")
            print(f"   Description: {cell_info['description']}")
        else:
            print(f"❌ {cell_line}: {result['error']}")


def test_comprehensive_biological_process(test_genes: list):
    """Test HPAGetComprehensiveBiologicalProcessTool with various scenarios"""
    
    tool = HPAGetComprehensiveBiologicalProcessTool({})
    
    # Test Case 1: Basic comprehensive analysis
    print("\n📋 Test Case 1: Basic Comprehensive Biological Process Analysis")
    print("-" * 60)
    
    for gene in test_genes[:2]:  # Test first 2 genes
        print(f"\nTesting {gene}:")
        
        result = tool.run({
            "gene_name": gene,
            "include_categorization": True,
            "include_pathway_analysis": True,
            "include_comparative_analysis": False,
            "max_processes": 30,
            "filter_critical_only": False
        })
        
        if "error" not in result:
            print(f"  ✅ Success: {result['gene_symbol']}")
            print(f"  📊 Total processes: {result['analysis_summary']['total_processes']}")
            print(f"  🎯 Critical processes: {result['analysis_summary']['critical_processes_found']}")
            print(f"  📈 Categorized processes: {result['analysis_summary']['categorized_processes']}")
            print(f"  🛤️ Pathway involvement: {result['analysis_summary']['pathway_involvement']}")
            
            insights = result['functional_insights']
            print(f"  🎯 Critical processes found: {len(insights['critical_processes'])}")
            print(f"  🎯 Disease-relevant processes: {len(insights['disease_relevant_processes'])}")
            print(f"  🎯 Therapeutic potential: {insights['therapeutic_potential']}")
            
            # Show critical processes
            if insights['critical_processes']:
                print(f"  🔬 Critical processes:")
                for cp in insights['critical_processes'][:3]:  # Show first 3
                    print(f"    - {cp['process']} ({cp['critical_type']})")
            
            # Show disease-relevant processes
            if insights['disease_relevant_processes']:
                print(f"  🏥 Disease-relevant processes:")
                for dp in insights['disease_relevant_processes'][:3]:  # Show first 3
                    print(f"    - {dp['process']} ({dp['primary_category']}, {dp['relevance_level']} relevance)")
        else:
            print(f"  ❌ Error: {result['error']}")
    
    # Test Case 2: Critical processes focus
    print("\n📋 Test Case 2: Critical Processes Focus")
    print("-" * 60)
    
    result = tool.run({
        "gene_name": "TP53",
        "include_categorization": True,
        "include_pathway_analysis": True,
        "include_comparative_analysis": False,
        "max_processes": 20,
        "filter_critical_only": True
    })
    
    if "error" not in result:
        print(f"✅ Critical processes analysis for TP53:")
        critical_processes = result['enhanced_analysis']['critical_processes']
        if critical_processes:
            print(f"  🎯 Found {len(critical_processes)} critical processes:")
            for cp in critical_processes:
                print(f"    - {cp['process']} ({cp['critical_type']}) - {cp['disease_relevance']} relevance")
        else:
            print("  📝 No critical processes found")
        
        categorized = result['enhanced_analysis']['categorized_processes']
        print(f"  📊 Process categories: {list(categorized.keys())}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Test Case 3: Pathway analysis focus
    print("\n📋 Test Case 3: Pathway Analysis Focus")
    print("-" * 60)
    
    result = tool.run({
        "gene_name": "BRCA1",
        "include_categorization": True,
        "include_pathway_analysis": True,
        "include_comparative_analysis": False,
        "max_processes": 50,
        "filter_critical_only": False
    })
    
    if "error" not in result:
        print(f"✅ Pathway analysis for BRCA1:")
        pathway_analysis = result['enhanced_analysis']['pathway_analysis']
        if pathway_analysis:
            print(f"  🛤️ Found {len(pathway_analysis)} pathway involvements:")
            for pathway, data in pathway_analysis.items():
                print(f"    - {pathway}: score {data['involvement_score']}, confidence {data['pathway_confidence']:.2f}")
        else:
            print("  📝 No pathway involvements found")
        
        complexity_score = result['enhanced_analysis']['process_complexity_score']
        print(f"  📊 Process complexity score: {complexity_score:.2f}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Test Case 4: Full analysis with comparative data
    print("\n📋 Test Case 4: Full Analysis with Comparative Data")
    print("-" * 60)
    
    result = tool.run({
        "gene_name": "EGFR",
        "include_categorization": True,
        "include_pathway_analysis": True,
        "include_comparative_analysis": True,
        "max_processes": 40,
        "filter_critical_only": False
    })
    
    if "error" not in result:
        print(f"✅ Full analysis for EGFR:")
        
        # Show research priorities and confidence
        insights = result['functional_insights']
        print(f"  📝 Research priorities: {', '.join(insights['research_priorities'])}")
        print(f"  🎯 Confidence assessment: {insights['confidence_assessment']}")
        
        # Show detailed critical and disease-relevant processes
        if insights['critical_processes']:
            print(f"  🔬 Critical processes ({len(insights['critical_processes'])} total):")
            for cp in insights['critical_processes']:
                print(f"    - {cp['process']} ({cp['critical_type']}, {cp['disease_relevance']} disease relevance)")
        
        if insights['disease_relevant_processes']:
            print(f"  🏥 Disease-relevant processes ({len(insights['disease_relevant_processes'])} total):")
            for dp in insights['disease_relevant_processes']:
                print(f"    - {dp['process']} ({dp['primary_category']}, {dp['relevance_level']} relevance)")
        
        # Show comparative analysis
        comparative = result['comparative_analysis']
        if 'related_genes' in comparative and comparative['related_genes']:
            print(f"  🔗 Related genes: {len(comparative['related_genes'])} found")
            for rg in comparative['related_genes'][:3]:  # Show first 3
                print(f"    - {rg['gene_name']} (similarity: {rg['functional_similarity']:.2f})")
        else:
            print("  📝 No related genes found")
        
        # Show metadata
        metadata = result['metadata']
        print(f"  📊 Analysis version: {metadata['analysis_version']}")
        print(f"  📊 Confidence level: {metadata['confidence_level']}")
        print(f"  📊 Data source: {metadata['data_source']}")
    else:
        print(f"❌ Error: {result['error']}")


def test_error_handling():
    """Test error handling and edge cases for both tools"""
    
    print("\n📋 Test Case 1: Invalid Gene Names")
    print("-" * 40)
    
    # Test with invalid gene names
    enhanced_tool = HPAGetEnhancedComparativeExpressionTool({})
    comprehensive_tool = HPAGetComprehensiveBiologicalProcessTool({})
    
    invalid_genes = ["INVALID_GENE_123", "NONEXISTENT_GENE", "GENE_WITH_SPECIAL_CHARS_!@#"]
    
    for gene in invalid_genes:
        print(f"\nTesting invalid gene: {gene}")
        
        # Test enhanced comparative expression
        result1 = enhanced_tool.run({
            "gene_name": gene,
            "cell_line": "hela"
        })
        if "error" in result1:
            print(f"  ✅ Enhanced tool: {result1['error']}")
        else:
            print(f"  ⚠️ Enhanced tool: Unexpected success")
        
        # Test comprehensive biological process
        result2 = comprehensive_tool.run({
            "gene_name": gene
        })
        if "error" in result2:
            print(f"  ✅ Comprehensive tool: {result2['error']}")
        else:
            print(f"  ⚠️ Comprehensive tool: Unexpected success")
    
    print("\n📋 Test Case 2: Invalid Cell Lines")
    print("-" * 40)
    
    invalid_cell_lines = ["invalid_cell", "cancer_cell", "test_cell_line"]
    
    for cell_line in invalid_cell_lines:
        print(f"\nTesting invalid cell line: {cell_line}")
        
        result = enhanced_tool.run({
            "gene_name": "TP53",
            "cell_line": cell_line
        })
        if "error" in result:
            print(f"  ✅ Error handling: {result['error']}")
        else:
            print(f"  ⚠️ Unexpected success")
    
    print("\n📋 Test Case 3: Missing Parameters")
    print("-" * 40)
    
    # Test missing gene_name
    result1 = enhanced_tool.run({
        "cell_line": "hela"
    })
    if "error" in result1:
        print(f"  ✅ Missing gene_name: {result1['error']}")
    else:
        print(f"  ⚠️ Missing gene_name: Unexpected success")
    
    # Test missing cell_line
    result2 = enhanced_tool.run({
        "gene_name": "TP53"
    })
    if "error" in result2:
        print(f"  ✅ Missing cell_line: {result2['error']}")
    else:
        print(f"  ⚠️ Missing cell_line: Unexpected success")
    
    # Test missing gene_name for comprehensive tool
    result3 = comprehensive_tool.run({})
    if "error" in result3:
        print(f"  ✅ Missing gene_name (comprehensive): {result3['error']}")
    else:
        print(f"  ⚠️ Missing gene_name (comprehensive): Unexpected success")
    
    print("\n📋 Test Case 4: Edge Cases")
    print("-" * 40)
    
    # Test with empty string
    result1 = enhanced_tool.run({
        "gene_name": "",
        "cell_line": "hela"
    })
    if "error" in result1:
        print(f"  ✅ Empty gene_name: {result1['error']}")
    else:
        print(f"  ⚠️ Empty gene_name: Unexpected success")
    
    # Test with whitespace-only
    result2 = enhanced_tool.run({
        "gene_name": "   ",
        "cell_line": "hela"
    })
    if "error" in result2:
        print(f"  ✅ Whitespace gene_name: {result2['error']}")
    else:
        print(f"  ⚠️ Whitespace gene_name: Unexpected success")
    
    # Test with very long gene name
    long_gene = "A" * 1000
    result3 = enhanced_tool.run({
        "gene_name": long_gene,
        "cell_line": "hela"
    })
    if "error" in result3:
        print(f"  ✅ Very long gene_name: {result3['error']}")
    else:
        print(f"  ⚠️ Very long gene_name: Unexpected success")


if __name__ == "__main__":
    main()
