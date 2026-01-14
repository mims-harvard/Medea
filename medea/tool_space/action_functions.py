import pandas as pd
from .gpt_utils import chat_completion
from .env_utils import get_medeadb_path as _get_medeadb_path
import unicodedata
import requests
import difflib
import torch 
import json 
import os
import copy



base_url = "https://api.platform.opentargets.org/api/v4/graphql"

open_target_query_string = """
    query target($diseaseName: String!){
        search(queryString: $diseaseName){
        hits{
            id
            entity
            category
            name
            description
        }
    }
}
"""

QUERY_DISEASEID = """Please provide the disease ID of {disease_name} from OpenTargets using the format: 'DiseaseID of <disease>: <diseaseID>'.
---
Here are related disease entites from OpenTargets:
{disease_entities}
"""


def build_pinnacle_ppi(ppi_embed_path: str, labels_path: str):
    """Load the PINNACLE celltype-specific PPI net with embeddings for each gene on different cell types.

    Args:
        ppi_embed_path (str): path to the PINNACLE PPI embeddings
        labels_path (str): path to the labels file containing the cell type and activated genes for each cell type
    
    Return:
        ppi_embed_dict (dict): a dictionary containing the cell type as key, and the value is a celltype-specific PPI dict with gene embeddings. For celltype-specific PPI dict, the key is the gene name and the value is the gene embedding.
        
    """
    embed = torch.load(ppi_embed_path)
    with open(labels_path, "r") as f:
        labels_dict = f.read()
    labels_dict = labels_dict.replace("\'", "\"")
    labels_dict = json.loads(labels_dict)
    celltypes = [c for c in labels_dict["Cell Type"] if c.startswith("CCI")]
    celltype_dict = {ct.split("CCI_")[1]: i for i, ct in enumerate(celltypes)}
    assert len(celltype_dict) > 0
    
    protein_names = []
    protein_celltypes = []
    for c, p in zip(labels_dict["Cell Type"], labels_dict["Name"]):
        if c.startswith("BTO") or c.startswith("CCI") or c.startswith("Sanity"): continue
        protein_names.append(p) 
        protein_celltypes.append(c) 

    proteins = pd.DataFrame.from_dict({"target": protein_names, "cell type": protein_celltypes})
    celltype_protein_dict = proteins.pivot_table(values="target", index="cell type", aggfunc={"target": list}).to_dict()["target"]
    assert len(celltype_protein_dict) > 0
    
    ppi_embed_dict = {}
    for celltype, index in celltype_dict.items():
        cell_embed_dict = {}
        cell_embed = embed[index]
        for i, gene in enumerate(celltype_protein_dict[celltype]):
            gene_embed = cell_embed[i, :]
            cell_embed_dict[gene] = gene_embed
            # print(f"[pinnacle]: {celltype} - {gene} - {gene_embed.shape}")
        celltype = celltype.replace(" ", "_")
        ppi_embed_dict[celltype] = cell_embed_dict
    return ppi_embed_dict


def load_pinnacle_ppi(
    cell_type: str,
    embed_path: str = None, 
    weights_only=False
):
    """
    Load the PINNACLE PPI embeddings for a specific cell type from a specified path.
    
    Uses the same normalization and matching logic as celltype_avaliability_checker() to ensure consistency.
    Accepts multiple cell type formats and normalizes them automatically.
    
    Args:
        cell_type (str): The specific cell type to load embeddings for. Accepts multiple formats:
                        - Standardized: 'cd4_positive_alpha_beta_memory_t_cell'
                        - Raw: 'cd4-positive,_alpha-beta_memory_t_cell'
                        - User-friendly: 'CD4+ memory T cell'
                        All formats are normalized to the same internal representation.
        embed_path (str): Path to the PPI embeddings file. If None, uses MEDEADB_PATH environment variable.
        weights_only (bool): Whether to load weights only or the full state.
    
    Returns:
        dict: A dictionary of PPI embeddings for the specified cell type with gene name as key 
            and cell type-specific gene embedding as value (torch.Tensor). Returns empty dict if cell type not found.
    """
    # Set default embed_path if not provided
    if embed_path is None:
        embed_path = os.path.join(_get_medeadb_path(), 'pinnacle_embeds/ppi_embed_dict.pth')
    
    # Load the full PPI dictionary from the specified path
    ppi_dict = torch.load(embed_path, weights_only=weights_only)
    
    # Normalize cell type using same function as checker
    def _normalize(s):
        return s.replace(",", "").replace("-", "_").replace(" ", "_").replace("+", "_positive").replace("α", "alpha").replace("β", "beta").lower()
    
    def _format_display(s):
        """Format for clean, consistent display."""
        formatted = s.replace(",", "").replace("-", "_").replace(" ", "_")
        while "__" in formatted:
            formatted = formatted.replace("__", "_")
        return formatted.lower()
    
    formalized_cell_type = _normalize(cell_type)
    
    # First pass: exact match
    for cell_key in ppi_dict.keys():
        formalized_key = _normalize(cell_key)
        if formalized_key == formalized_cell_type:
            formatted_name = _format_display(cell_key)
            print(f"[load_pinnacle_ppi] ✓ Loaded {len(ppi_dict[cell_key])} genes for cell type: '{formatted_name}'", flush=True)
            return ppi_dict[cell_key]
        
    # Second pass: fuzzy matching with same logic as checker
    from thefuzz import fuzz
    best_match = None
    best_score = 0
    
    for cell_key in ppi_dict.keys():
        candidate_norm = _normalize(cell_key)
        
        # Compute similarity score using multiple strategies
        token_sort = fuzz.token_sort_ratio(formalized_cell_type, candidate_norm)
        token_set = fuzz.token_set_ratio(formalized_cell_type, candidate_norm)
        
        # Combined score
        score = token_sort * 0.7 + token_set * 0.3
        
        # Token overlap boost
        query_tokens = set(formalized_cell_type.split('_'))
        candidate_tokens = set(candidate_norm.split('_'))
        if query_tokens and candidate_tokens:
            intersection = len(query_tokens.intersection(candidate_tokens))
            union = len(query_tokens.union(candidate_tokens))
            jaccard = intersection / union if union > 0 else 0
            score += jaccard * 15
        
        if score > best_score:
            best_score = score
            best_match = cell_key
    
    # Return best match if score is reasonable
    if best_match and best_score >= 60:
        formatted_name = _format_display(best_match)
        print(f"[load_pinnacle_ppi] ⚠ Cell type '{cell_type}' matched to '{formatted_name}' (score: {best_score:.0f})", flush=True)
        print(f"[load_pinnacle_ppi] ✓ Loaded {len(ppi_dict[best_match])} genes for matched cell type", flush=True)
        return ppi_dict[best_match]
    
    # If no good match found, return empty dict
    print(f"[load_pinnacle_ppi] ✗ ERROR: Cell type '{cell_type}' not found in PINNACLE embeddings (best match score: {best_score:.0f}).", flush=True)
    print(f"[load_pinnacle_ppi] → Please use the celltype_avaliability_checker to find valid cell types.", flush=True)
    return {}


def read_labels_from_evidence(
        positive_protein_prefix, 
        negative_protein_prefix, 
        raw_data_prefix, 
        positive_proteins={}, 
        negative_proteins={}, 
        all_relevant_proteins={}
    ):
    try:
        with open(positive_protein_prefix + '.json', 'r') as f:
            temp = json.load(f)
            positive_proteins = temp
        with open(negative_protein_prefix + '.json', 'r') as f:
            temp = json.load(f)
            negative_proteins = temp
        
        if raw_data_prefix != None:
            with open(raw_data_prefix + '.json', 'r') as f:
                temp = json.load(f)
                all_relevant_proteins = temp
        else: all_relevant_proteins = {}

        return positive_proteins, negative_proteins, all_relevant_proteins
    except:
        print("Files not found")
        return {}, {}, {}


def search_disease_open_target(disease_name):
    # Set variables object of arguments to be passed to endpoint
    relavent_entities = None
    variables = {"diseaseName": disease_name}
    for i in range(5):
        try:
            # Perform POST request and check status code of response
            r = requests.post(base_url, json={"query": open_target_query_string, "variables": variables})
            # Transform API response from JSON into Python dictionary and print in console
            api_response = json.loads(r.text)
            relavent_entities = api_response['data']['search']['hits']
        except Exception as e:
            print(f"[OpenTarget] Bad OpenTarget API response, retrying...")
            continue
        if relavent_entities is not None: break

    if relavent_entities is None:
        raise ValueError(f"[OpenTarget] No relevant entities found for {disease_name}")
    # Check the top 5 entities
    filtered_entities = [ e for e in relavent_entities if e['entity'] == 'disease']
    return filtered_entities


import requests
import re

def normalize_string(s):
    """
    Normalize a string by removing punctuation and converting to lowercase.
    """
    return re.sub(r'[^\w\s]', '', s).lower().strip()



def standardize_disease_name(disease_name):
    # Normalize unicode characters to remove accents
    disease_name = unicodedata.normalize('NFKD', disease_name).encode('ASCII', 'ignore').decode('utf-8')
    
    # Remove possessive apostrophes (e.g., "'s")
    disease_name = re.sub(r"'s\b", "", disease_name)
    
    # Remove any remaining punctuation
    disease_name = re.sub(r"[^\w\s]", "", disease_name)
    
    # Normalize whitespace
    disease_name = re.sub(r"\s+", " ", disease_name).strip()
    
    # Capitalize each word (optional, depending on your target style)
    disease_name = disease_name.title()
    
    return disease_name


def get_efo_id(disease_name):
    """
    Retrieve the EFO identifier for a given disease name using the EMBL-EBI OLS API.

    Args:
        disease_name (str): The name of the disease.

    Returns:
        str: The EFO identifier if found, else None.
    """
    # Define the API endpoint
    api_url = "https://www.ebi.ac.uk/ols/api/search"
    disease_name = standardize_disease_name(disease_name)

    # First attempt: exact match search
    params = {
        'q': disease_name,
        'ontology': 'efo',
        'exact': 'true'
    }
    response = requests.get(api_url, params=params)
    
    if response.status_code == 200:
        results = response.json()
        if results['response']['numFound'] > 0:
            return results['response']['docs'][0]['obo_id']
    else:
        print(f"Failed to connect to OLS API. Status code: {response.status_code}")
        return None

    # If exact match not found, try fuzzy search with normalized labels.
    print(f"Exact match not found for '{disease_name}'. Trying fuzzy search.")
    params['exact'] = 'false'
    response = requests.get(api_url, params=params)
    
    if response.status_code == 200:
        results = response.json()
        normalized_target = normalize_string(disease_name)
        for doc in results['response']['docs']:
            label = doc.get('label', '')
            if normalize_string(label) == normalized_target:
                return doc.get('obo_id')
        print(f"No matching EFO ID found for '{disease_name}' in fuzzy search.")
        return None
    else:
        print(f"Failed to connect to OLS API on fuzzy search. Status code: {response.status_code}")
        return None

def search_disease_efo(disease_name, attempts=5):
    """
    Searches for the EFO ID of a given disease in the EFO ontology.
    
    Args:
        disease_name (str): The name of the disease to search for.
        
    Returns:
        str: The EFO or MONDO ID of the disease if found, otherwise None.
    """
    
    # Attempt to query up to 5 times to handle potential issues
    for attempt in range(attempts):
        try:
            # Search for disease entities using the Open Targets platform
            disease_name = disease_name.replace("_", " ")
            disease_efo = get_efo_id(disease_name)
            disease_id = disease_efo.replace(":", "_")
            
            # Check if the response contains a valid EFO or MONDO ID
            if 'EFO' in disease_id or 'MONDO' in disease_id:
                return disease_id
        except Exception as e:
            print(f"[Attempt {attempt + 1}/5] Error: {e}")
            print("[search_disease_efo]: Bad request format or other issue encountered. Retrying...")
    
    # Return None if a valid ID was not found after 5 attempts
    print(f"[search_disease_efo]: Failed to find EFO or MONDO ID for '{disease_name}' after {attempts} attempts.")
    return None


def compare_strings(str1, str2):
    matcher = difflib.SequenceMatcher(None, str1, str2)
    return matcher.ratio()


def load_disease_targets(disease_name, data_dir=None, attributes=["otGeneticsPortal", "chembl"], use_api=True, max_retries=5):
    """
    Load the disease-associated targets from OpenTargets API or local JSON file.

    Parameters:
        disease_name (str): The name of the disease.
        data_dir (str): The directory containing disease target data. Only used if use_api=False.
        attributes (list): List of attributes to filter targets. 
                          For API: filters by datasource types (e.g., "ot_genetics_portal", "chembl")
                          For local: filters by JSON field names (e.g., "otGeneticsPortal", "chembl")
        use_api (bool): If True, retrieve data from OpenTargets API. If False, use local JSON files.
        max_retries (int): Maximum number of retry attempts for API calls.

    Returns:
        set: A set of gene symbols associated with the disease.
    """
    if use_api:
        return _load_disease_targets_from_api(disease_name, attributes, max_retries)
    else:
        return _load_disease_targets_from_local(disease_name, data_dir, attributes)


def _load_disease_targets_from_api(disease_name, attributes=["otGeneticsPortal", "chembl"], max_retries=5):
    """
    Load disease-associated targets from OpenTargets GraphQL API.
    
    Parameters:
        disease_name (str): The name of the disease.
        attributes (list): List of datasource types to filter targets.
        max_retries (int): Maximum number of retry attempts.
    
    Returns:
        set: A set of gene symbols associated with the disease.
    """
    # Get the EFO/MONDO ID for the disease
    disease_efo = search_disease_efo(disease_name)
    if disease_efo is None:
        raise ValueError(f"[load_disease_targets] Could not find EFO/MONDO ID for disease: {disease_name}")
    
    print(f"[load_disease_targets] Querying OpenTargets API for disease: {disease_name} ({disease_efo})")
    
    # Convert attribute names to match OpenTargets datasource format
    # Map common attribute names to OpenTargets datasource IDs
    # Actual API datasource IDs: genetic_association, genetic_literature, known_drug, 
    # literature, rna_expression, animal_model, somatic_mutation, etc.
    datasource_mapping = {
        "otGeneticsPortal": "genetic_association",
        "chembl": "known_drug",
        "europepmc": "literature",
        "expression_atlas": "rna_expression",
        "intogen": "somatic_mutation",
        "literature": "literature",
        "genetic_association": "genetic_association",
        "genetic_literature": "genetic_literature",
        "known_drug": "known_drug",
        "animal_model": "animal_model",
        "rna_expression": "rna_expression",
        "somatic_mutation": "somatic_mutation"
    }
    
    # Normalize attributes
    normalized_attributes = []
    for attr in attributes:
        if attr in datasource_mapping:
            normalized_attributes.append(datasource_mapping[attr])
        else:
            normalized_attributes.append(attr.lower())
    
    # GraphQL query to get disease-target associations
    query = """
    query diseaseTargets($efoId: String!, $size: Int!, $index: Int!) {
        disease(efoId: $efoId) {
            id
            name
            associatedTargets(page: {size: $size, index: $index}) {
                count
                rows {
                    target {
                        id
                        approvedSymbol
                    }
                    score
                    datatypeScores {
                        id
                        score
                    }
                }
            }
        }
    }
    """
    
    # Implement pagination (API max page size is 3000)
    page_size = 3000
    page_index = 0
    target_set = set()
    total_count = None
    
    while True:
        # Use the EFO ID as-is (with underscores, not colons)
        variables = {
            "efoId": disease_efo,
            "size": page_size,
            "index": page_index
        }
        
        # Retry logic for API calls
        success = False
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    base_url,
                    json={"query": query, "variables": variables},
                    timeout=30
                )
                
                # Check response before raising for status to see error details
                if response.status_code != 200:
                    try:
                        error_detail = response.json()
                        print(f"[load_disease_targets] API Error Detail: {json.dumps(error_detail, indent=2)}")
                    except:
                        print(f"[load_disease_targets] API Error: {response.text}")
                
                response.raise_for_status()
                
                api_response = response.json()
                
                # Check for errors in the GraphQL response
                if "errors" in api_response:
                    print(f"[load_disease_targets] GraphQL errors: {api_response['errors']}")
                    raise ValueError(f"GraphQL query returned errors: {api_response['errors']}")
                
                # Extract disease data
                disease_data = api_response.get("data", {}).get("disease")
                if not disease_data:
                    raise ValueError(f"[load_disease_targets] No disease data found for {disease_name} ({disease_efo})")
                
                associated_targets = disease_data.get("associatedTargets", {})
                rows = associated_targets.get("rows", [])
                total_count = associated_targets.get("count", 0)
                
                if page_index == 0:
                    print(f"[load_disease_targets] Found {total_count} total target associations from API")
                
                # Filter targets based on attributes if specified
                if attributes and len(attributes) > 0:
                    for row in rows:
                        datatype_scores = row.get("datatypeScores", [])
                        # Check if any of the specified datasources have a score
                        has_relevant_datasource = False
                        for ds in datatype_scores:
                            datasource_id = ds.get("id", "").lower()
                            score = ds.get("score", 0)
                            if any(attr in datasource_id for attr in normalized_attributes) and score > 0:
                                has_relevant_datasource = True
                                break
                        
                        if has_relevant_datasource:
                            symbol = row.get("target", {}).get("approvedSymbol")
                            if symbol:
                                target_set.add(symbol)
                else:
                    # If no attributes specified, return all targets with any association
                    for row in rows:
                        symbol = row.get("target", {}).get("approvedSymbol")
                        if symbol and row.get("score", 0) > 0:
                            target_set.add(symbol)
                
                success = True
                break
                
            except requests.exceptions.RequestException as e:
                print(f"[load_disease_targets] Attempt {attempt + 1}/{max_retries} - Network error: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"[load_disease_targets] Failed to retrieve data from OpenTargets API after {max_retries} attempts: {e}")
            except Exception as e:
                print(f"[load_disease_targets] Attempt {attempt + 1}/{max_retries} - Error: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"[load_disease_targets] Failed to process API response after {max_retries} attempts: {e}")
        
        if not success:
            break
        
        # Check if we've fetched all pages
        if len(rows) < page_size or (page_index + 1) * page_size >= total_count:
            break
        
        page_index += 1
        print(f"[load_disease_targets] Fetching page {page_index + 1} (retrieved {len(target_set)} targets so far)...")
    
    print(f"[load_disease_targets] Retrieved {len(target_set)} targets from OpenTargets API after filtering")
    return target_set


def _load_disease_targets_from_local(disease_name, data_dir=None, attributes=["otGeneticsPortal", "chembl"]):
    """
    Load disease-associated targets from local JSON file (legacy method).
    
    Parameters:
        disease_name (str): The name of the disease.
        data_dir (str): The directory containing disease target data.
        attributes (list): List of attributes to filter targets.
    
    Returns:
        set: A set of gene symbols associated with the disease.
    """
    # Set default data_dir if not provided
    if data_dir is None:
        data_dir = os.path.join(_get_medeadb_path(), "targetID/disease_target")
    
    critera_flag, target_set = False, set()
    disease_efo = search_disease_efo(disease_name)
    file_path = os.path.join(data_dir, disease_efo + '.json')
    if not os.path.exists(file_path):
        avaliable_diseases = []
        for _, dirnames, _ in os.walk(data_dir):
            for dir_name in dirnames:
                avaliable_diseases.append(dir_name)
        raise ValueError(f"[load_disease_targets] Disease: {disease_name} ({disease_efo}) is not avaliable locally, the avaliable disease options are: {avaliable_diseases}.")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        for entry in data:
            for a in attributes:
                if entry.get(a, "No data") != "No data":
                    critera_flag = True
            if critera_flag:
                target_set.add(entry['symbol'])
    return target_set




def get_gene_synonyms(gene_name, species="Homo sapiens"):
    """
    Fetch synonyms for a given gene using the NCBI Entrez API.

    Args:
        gene_name (str): The name of the gene to search for.
        species (str): The species to filter results. Default is 'Homo sapiens'.

    Returns:
        list: A list of synonyms for the gene, or an error message if not found.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    # Step 1: Search for the gene in the NCBI Gene database
    search_params = {
        "db": "gene",
        "term": f"{gene_name}[Gene] AND {species}[Organism]",
        "retmode": "json"
    }
    while True:
        response = requests.get(base_url, params=search_params)
        if response.status_code == 200:
            break
        # print(f"Error: Unable to fetch data from NCBI. Status code: {response.status_code}")
        
    search_data = response.json()
    if "esearchresult" not in search_data or not search_data["esearchresult"].get("idlist"):
        return None

    gene_id = search_data["esearchresult"]["idlist"][0]  # Get the first gene ID

    # Step 2: Fetch gene details using the gene ID
    summary_params = {
        "db": "gene",
        "id": gene_id,
        "retmode": "json"
    }
    while True:
        summary_response = requests.get(summary_url, params=summary_params)
        if summary_response.status_code == 200:
            break
        # print( f"Error: Unable to fetch gene details. Status code: {summary_response.status_code}")

    summary_data = summary_response.json()
    gene_summary = summary_data.get("result", {}).get(gene_id, {})
    synonyms = gene_summary.get("otheraliases", "")

    # Return the synonyms as a list
    if synonyms:
        return synonyms.strip("'").split(", ")
    else:
        return None
