import ast
import json
import os
import re
from typing import List, Tuple

# NOTE: Import HUMANBASE_CONTEXT_CHECKER locally in functions to avoid circular import
from thefuzz import process
from .action_functions import *
from .enrichr import WikiPathwaysInteractionTool
from .env_utils import get_medeadb_path as _get_medeadb_path
from .gpt_utils import chat_completion
from .humanbase import *
from .transcriptformer import TranscriptformerEmbeddingTool

# Constants
MAX_RECOMMENDATIONS = 5  # Limit recommendation list size
MAX_RETRY_ATTEMPTS = 3


def disease_name_checker(disease_name: str) -> Tuple[bool, str]:
    """
    Check if a disease name is valid for loading targets.
    
    Args:
        disease_name: Disease name to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        load_disease_targets(disease_name)
        msg = f'[disease_name_checker] {disease_name} is available for tool load_disease_targets().'
        print(msg, flush=True)
        return True, msg
    except Exception as e:
        msg = f'[disease_name_checker] Error: {str(e)}'
        print(msg, flush=True)
        return False, msg


def _normalize_celltype(cell_type: str) -> str:
    """Normalize cell type name for matching."""
    return cell_type.replace(",", "").replace("-", "_").replace(" ", "_").replace("+", "_positive").replace("α", "alpha").replace("β", "beta").lower()


def _format_celltype_display(cell_type: str) -> str:
    """
    Format cell type name for clean, consistent display.
    
    Converts internal representation to standardized format:
    'cd4-positive,_alpha-beta_memory_t_cell' → 'cd4_positive_alpha_beta_memory_t_cell'
    """
    # Simply normalize to consistent underscore format, removing commas and hyphens
    formatted = cell_type.replace(",", "").replace("-", "_").replace(" ", "_")
    
    # Remove any double underscores that might have been created
    while "__" in formatted:
        formatted = formatted.replace("__", "_")
    
    # Keep it lowercase for consistency
    return formatted.lower()


def _compute_celltype_similarity(query: str, candidate: str) -> float:
    """
    Compute similarity between cell type names using multiple fuzzy matching strategies.
    Returns a score between 0 and 100.
    
    This approach is generalizable and doesn't rely on hardcoded keywords.
    It uses multiple fuzzy matching algorithms and token-based similarity.
    """
    from thefuzz import fuzz
    
    # Normalize both strings
    query_norm = _normalize_celltype(query)
    candidate_norm = _normalize_celltype(candidate)
    
    # Exact match gets highest score
    if query_norm == candidate_norm:
        return 100.0
    
    # Use multiple fuzzy matching strategies and combine them
    # 1. Token sort ratio: good for reordered words
    token_sort_score = fuzz.token_sort_ratio(query_norm, candidate_norm)
    
    # 2. Token set ratio: good for subset matches
    token_set_score = fuzz.token_set_ratio(query_norm, candidate_norm)
    
    # 3. Partial ratio: good for substring matches
    partial_score = fuzz.partial_ratio(query_norm, candidate_norm)
    
    # 4. Simple ratio: baseline similarity
    simple_score = fuzz.ratio(query_norm, candidate_norm)
    
    # Weighted combination of different strategies
    # Token-based methods are most important for scientific terms
    combined_score = (
        token_sort_score * 0.4 +      # Reordered words (most important)
        token_set_score * 0.3 +        # Subset matching
        partial_score * 0.2 +          # Substring matching
        simple_score * 0.1             # Direct similarity
    )
    
    # Token overlap boost (data-driven, not hardcoded)
    query_tokens = set(query_norm.split('_'))
    candidate_tokens = set(candidate_norm.split('_'))
    
    # Calculate Jaccard similarity for token overlap
    if query_tokens and candidate_tokens:
        intersection = len(query_tokens.intersection(candidate_tokens))
        union = len(query_tokens.union(candidate_tokens))
        jaccard_similarity = intersection / union if union > 0 else 0
        
        # Boost score based on token overlap percentage
        # More shared tokens = higher confidence
        overlap_boost = jaccard_similarity * 15  # Up to 15 point boost
        combined_score += overlap_boost
    
    # Penalize large length differences (different specificity levels)
    len_diff = abs(len(query_norm) - len(candidate_norm))
    max_len = max(len(query_norm), len(candidate_norm))
    if max_len > 0:
        len_ratio = len_diff / max_len
        if len_ratio > 0.5:  # Significantly different lengths
            combined_score *= 0.85  # Small penalty for very different lengths
    
    return min(combined_score, 100.0)


def celltype_avaliability_checker(disease_name:str, cell_type:str, model_name:str="pinnacle"):
    model_name = model_name.lower()
    if model_name == "pinnacle":
        # Load full embedding dictionary to get all available cell types
        import torch
        embed_path = os.path.join(_get_medeadb_path(), 'pinnacle_embeds/ppi_embed_dict.pth')
        full_embedding_dict = torch.load(embed_path, weights_only=False)
        function_name = "load_pinnacle_ppi"
    else:
        return False, f"[celltype_avaliability_checker] Invalid model_name: {model_name}. Please use 'pinnacle'."
    
    # Check if cell type exists (with normalization)
    formalized_cell_type = _normalize_celltype(cell_type)
    
    # First pass: exact match
    for cell_key in full_embedding_dict.keys():
        formalized_key = _normalize_celltype(cell_key)
        if formalized_key == formalized_cell_type:
            formatted_cell_type = _format_celltype_display(cell_key)
            msg = f'[{function_name}] Cell type \'{formatted_cell_type}\' is available in {model_name}.'
            print(msg, flush=True)
            return True, msg
    
    # Second pass: find similar cell types using improved fuzzy matching
    similarity_scores = []
    for cell_key in full_embedding_dict.keys():
        score = _compute_celltype_similarity(cell_type, cell_key)
        similarity_scores.append((cell_key, score))
    
    # Sort by score
    sorted_matches = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Filter to only include reasonable matches (score >= 40)
    good_matches = [(ct, score) for ct, score in sorted_matches if score >= 40]
    
    if not good_matches:
        # If no good matches, show top 5 anyway
        good_matches = sorted_matches[:MAX_RECOMMENDATIONS]
    else:
    # Limit to top MAX_RECOMMENDATIONS
        good_matches = good_matches[:MAX_RECOMMENDATIONS]
    
    # Format message with best match highlighted (using clean formatted names)
    best_match_raw, best_score = good_matches[0]
    best_match_formatted = _format_celltype_display(best_match_raw)
    other_matches_formatted = [_format_celltype_display(ct) for ct, _ in good_matches[1:]]
        
    msg = (
        f"[{function_name}] Cell type \"{cell_type}\" not found in {model_name}.\n"
        f"  → BEST MATCH (score: {best_score:.0f}): '{best_match_formatted}'\n"
    )
    
    if other_matches_formatted:
        msg += f"  → Other options: {other_matches_formatted}\n"
    
    msg += f"  → RECOMMENDATION: Use '{best_match_formatted}' in your proposal and explicitly state you are using this alternative cell type."
    
    print(msg, flush=True)
    return False, msg


def context_avalibility_checker(disease_name:str, cell_type:str, gene_list:list=None, model_name="pinnacle"):
    model_name = model_name.lower()
    if cell_type is None:
        msg = f"[context_avalibility_checker] To leverage {model_name} embedding, cell_type can not be 'None', but received {cell_type}."
        print(msg, flush=True)
        return False, msg
    
    # Only call celltype_avaliability_checker for pinnacle model
    if model_name == "pinnacle":
        checker, msg = celltype_avaliability_checker(disease_name=disease_name, cell_type=cell_type, model_name=model_name)
        if not checker:
            return False, msg
    else:
        msg = f"[context_avalibility_checker] Invalid model_name: {model_name}."
        print(msg, flush=True)
        return False, msg

    # If gene_list is empty/None, skip gene-level validation (discovery mode)
    if gene_list is None or len(gene_list) == 0:
        msg = f"[context_avalibility_checker] Cell type '{cell_type}' is valid for {model_name}. Gene validation skipped (discovery mode - genes will come from disease targets or other sources)."
        print(msg, flush=True)
        return True, msg

    if model_name == "pinnacle":
        # Load full embedding dictionary and find the correct cell type key
        import torch
        embed_path = os.path.join(_get_medeadb_path(), 'pinnacle_embeds/ppi_embed_dict.pth')
        full_embedding_dict = torch.load(embed_path, weights_only=False)
        function_name = "load_pinnacle_ppi"
        
        # Find the actual cell type key (with normalization using same function as checker)
        formalized_cell_type = _normalize_celltype(cell_type)
        actual_cell_type_key = None
        
        for cell_key in full_embedding_dict.keys():
            formalized_key = _normalize_celltype(cell_key)
            if formalized_key == formalized_cell_type:
                actual_cell_type_key = cell_key
                break
                
        if actual_cell_type_key is None:
            return False, f"[context_avalibility_checker] Cell type '{cell_type}' not found in embeddings."
            
        cell_type_embedding = full_embedding_dict[actual_cell_type_key]
    else:
        return False, f"[context_avalibility_checker] Invalid model_name: {model_name}. Only 'pinnacle' is supported."

    avaliability = [gene in cell_type_embedding for gene in gene_list]
    if all(avaliability):
        msg = f"[context_avalibility_checker] {gene_list} all have {cell_type}-specific gene embeddings in [{function_name}]."
        print(msg, flush=True)
        return True, msg
    filtered_genes = [gene for index, gene in enumerate(gene_list) if not avaliability[index]]
    if any(avaliability):  # Some genes are available
        msg = f"[context_avalibility_checker] Warning: {filtered_genes} don't have {cell_type}-specific gene embedding in [{function_name}]."
        print(msg, flush=True)
        return True, msg
    else:  # No genes are available
        msg = f"[context_availability_checker] Warning: None of the candidate genes are available in the {function_name} embeddings for the specified cell type '{cell_type}'. Please consider using a similar cell type or an alternative model."
        print(msg, flush=True)
        return True, msg
    

def humanbase_context_checker(gene_list:list[str]=None, tissue:str=None, top_n=5):
    # Local import to avoid circular dependency
    try:
        from ..agents.prompt_template import HUMANBASE_CONTEXT_CHECKER
    except (ImportError, ValueError):
        # Fallback for when relative import fails
        from medea.agents.prompt_template import HUMANBASE_CONTEXT_CHECKER
    
    # If gene_list is empty/None, skip gene validation (discovery mode)
    if gene_list is None or len(gene_list) == 0:
        # Still validate tissue context
        if tissue is None:
            msg = f"[humanbase_context_checker] tissue parameter is required but was None."
            print(msg, flush=True)
            return False, msg
        
        # Validate tissue only
        available_slug = []
        tissue = tissue.replace(" ", "-").lower()
        base_url = "https://hb.flatironinstitute.org/api/integrations/"
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            data = response.json()
            for e in data:
                context = e.get("context")
                if not isinstance(context, dict):
                    continue
                term = context.get("term")
                if not isinstance(term, dict):
                    continue
                identifier = term.get("identifier")
                if identifier and "BTO:" in identifier:
                    slug = e.get("slug")
                    if slug:
                        available_slug.append(slug)
        except requests.exceptions.HTTPError as http_err:
            print("[humanbase_context_checker] Error 404: The requested resource was not found", flush=True)
        
        if tissue in available_slug:
            msg = f"[humanbase_context_checker] Tissue '{tissue}' is valid for HumanBase. Gene validation skipped (discovery mode - genes will come from disease targets or other sources)."
            print(msg, flush=True)
            return True, msg
        else:
            # Find similar tissues for invalid tissue
            try:
                prompt = HUMANBASE_CONTEXT_CHECKER.format(
                    context=tissue,
                    tissues=available_slug,
                    max_items=MAX_RECOMMENDATIONS
                )
                relevant_tissues_raw = chat_completion(messages=prompt, temperature=0.3)
                relevant_tissues = _parse_and_validate_list(
                    relevant_tissues_raw,
                    valid_options=available_slug,
                    max_items=MAX_RECOMMENDATIONS
                )
                if not relevant_tissues:
                    relevant_tissues = _fuzzy_match_tissues(tissue, available_slug, MAX_RECOMMENDATIONS)
            except Exception as e:
                print(f"[humanbase_context_checker] GPT recommendation failed: {e}. Using fuzzy matching.", flush=True)
                relevant_tissues = _fuzzy_match_tissues(tissue, available_slug, MAX_RECOMMENDATIONS)
            
            msg = (
                f"[humanbase_context_checker] Tissue \"{tissue}\" not found in HumanBase. "
                f"Top {len(relevant_tissues)} similar options: {relevant_tissues}. "
                f"If using an alternative, state this clearly in your proposal."
            )
            print(msg, flush=True)
            return False, msg
    
    # Qualify gene name & retrieve entrez id 
    try: 
        gene_ids = get_entrez_ids(gene_list)
    except Exception as e:
        return enrichr_gene_name_checker(gene_list)
    
    if gene_ids is None:
        avaliable_genes = {}
        syno_dict = {gene: get_gene_synonyms(gene) for gene in gene_list}
        for k, v in syno_dict.items():
            for syn in v:
                if get_entrez_ids(syn) != None:
                    avaliable_genes[k] = syn
                    break
        
        msg = f"[humanbase_context_checker] Invalid gene name (can't retivel corresponding Entrez ID): {gene_list}. However, Entrez ID is avaliable for synonyms: {avaliable_genes}"
        print(msg, flush=True)
        return False, msg
    
    available_slug = []
    tissue = tissue.replace(" ", "-").lower()
    base_url = "https://hb.flatironinstitute.org/api/integrations/"
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        for e in data:
            context = e.get("context")
            if not isinstance(context, dict):  # Ensure `context` is a dictionary
                continue

            term = context.get("term")
            if not isinstance(term, dict):  # Ensure `term` is a dictionary
                continue

            identifier = term.get("identifier")
            if identifier and "BTO:" in identifier:
                slug = e.get("slug")
                if slug:
                    available_slug.append(slug)
                    # print(f"Adding {slug}")

    except requests.exceptions.HTTPError as http_err:
        print("[humanbase_context_checker] Error 404: The requested resource was not found", flush=True)
    if tissue not in available_slug:
        # Find functionally similar tissues using GPT with validation
        prompt = HUMANBASE_CONTEXT_CHECKER.format(
            context=tissue,
            tissues=available_slug,
            max_items=MAX_RECOMMENDATIONS
        )
        
        try:
            relevant_tissues_raw = chat_completion(messages=prompt, temperature=0.3)  # Lower temp for more focused results
            
            # Validate and parse the GPT response
            relevant_tissues = _parse_and_validate_list(
                relevant_tissues_raw,
                valid_options=available_slug,
                max_items=MAX_RECOMMENDATIONS
            )
            
            if not relevant_tissues:
                # Fallback to fuzzy matching
                relevant_tissues = _fuzzy_match_tissues(tissue, available_slug, MAX_RECOMMENDATIONS)
            
        except Exception as e:
            print(f"[humanbase_context_checker] GPT recommendation failed: {e}. Using fuzzy matching.", flush=True)
            relevant_tissues = _fuzzy_match_tissues(tissue, available_slug, MAX_RECOMMENDATIONS)
        
        msg = (
            f"[humanbase_context_checker] Tissue \"{tissue}\" not found in HumanBase. "
            f"Top {len(relevant_tissues)} similar options: {relevant_tissues}. "
            f"If using an alternative, state this clearly in your proposal."
        )
        print(msg, flush=True)
        return False, msg

    gene_url = "https://hb.flatironinstitute.org/api/genes/{gene_id}/"
    for gene in gene_ids:
        try:
            response = requests.get(gene_url.format(gene_id=gene))
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            msg = f"[humanbase_context_checker] Invalid gene Entrez id: {gene}."
            print(msg, flush=True)
            return False, msg
    msg = f"[HumanBase Context Checker] The genes {gene_list} and the tissue {tissue} are valid for querying the HumanBase API."
    print(msg, flush=True)
    return True, msg

def enrichr_gene_name_checker(gene_list: list=None):
    # If gene_list is empty/None, skip validation (discovery mode)
    if gene_list is None or len(gene_list) == 0:
        msg = "[enrichr_gene_name_checker] Gene validation skipped (discovery mode - genes will come from disease targets or other sources)."
        print(msg, flush=True)
        return True, msg
    
    unavailable_genes = []
    official_genes = {}
    
    # Create an instance to call the instance method
    tool_instance = WikiPathwaysInteractionTool()  # Any subclass will work since we just need the base method
    
    for gene in gene_list:
        try:
            official_gene = tool_instance.get_official_gene_name(gene)
            if official_gene != gene:
                official_genes[gene] = official_gene
        except Exception as e:
            print(f"[enrichr_gene_name_checker] Error checking gene {gene}: {e}", flush=True)
            unavailable_genes.append(gene)
    
    if not unavailable_genes and not official_genes:
        msg = f"[enrichr_gene_name_checker] All provided genes {gene_list} are valid for the enrichr_api function."
        return True, msg
    
    messages = []
    if official_genes:
        corrections = ', '.join([f"{g} → {og}" for g, og in official_genes.items()])
        messages.append(f"[enrichr_gene_name_checker] Please use official gene symbols: {corrections}. Clearly state the reason for this adjustment in your proposal, specifying that the original gene names provided by the user were not standardized. This clarification will inform other agents about the necessity of using standardized gene nomenclature.")
        
    if unavailable_genes:
        messages.append(f"[enrichr_gene_name_checker] Official symbols not found for: {unavailable_genes}. Please verify these gene names.")
    
    final_message = ' '.join(messages)
    return False, final_message
    

def concept_name_checker(user_query: str, concept_names: List[str], attempt: int = MAX_RETRY_ATTEMPTS) -> Tuple[bool, str]:
    """
    Check if concept names are valid for Compass predictions.
    
    Args:
        user_query: User's original query for context
        concept_names: List of concept names to validate
        attempt: Number of retry attempts
        
    Returns:
        Tuple of (is_valid, message)
    """
    predefined_concepts = [
        'Memory_Bcell', 'Naive_Tcell', 'Apoptosis_pathway', 'pDC', 'Memory_Tcell', 'Genome_integrity', 
        'Adipocyte', 'Erythrocyte', 'Pancreatic', 'Innate_lymphoid_cell', 'cDC', 'CD8_Tcell', 
        'Hepatocyte', 'Platelet', 'Treg', 'CD4_Tcell', 'Stroma', 'Endothelial', 'Myeloid', 
        'Exhausted_Tcell', 'Immune_checkpoint', 'Tcell_general', 'Cell_proliferation', 'IFNg_pathway', 
        'Mesothelial', 'TLS', 'Cytokine', 'Monocyte', 'Cytotoxic_Tcell', 'Plasma_cell', 'Fibroblast', 
        'TGFb_pathway', 'Pneumocyte', 'Granulocyte', 'Macrophage', 'NKcell', 'Stem', 'Epithelial', 
        'Mast', 'Pericyte', 'Bcell_general', 'Naive_Bcell'
    ]

    # Check if all concepts are valid
    invalid_concepts = [c for c in concept_names if c not in predefined_concepts]
    
    if not invalid_concepts:
        msg = f"[concept_name_checker] All concepts {concept_names} are valid."
        return True, msg
    
    # Find alternatives using GPT with validation
    concept_checker_prompt = """
Given the User Query and invalid concepts, identify up to {max_items} relevant alternatives from the Available Concepts list.

Available Concepts:
{predefined_concepts}

User Query:
{user_query}

Invalid Concepts: {invalid_concepts}

Return ONLY a Python list of up to {max_items} concept names from the available list. Example: ['CD4_Tcell', 'Memory_Tcell']
"""

    for retry in range(attempt):
        try:
            checker_prompt = concept_checker_prompt.format(
                user_query=user_query, 
                predefined_concepts=predefined_concepts,
                invalid_concepts=invalid_concepts,
                max_items=MAX_RECOMMENDATIONS
            )
            
            ava_concept_raw = chat_completion(checker_prompt, temperature=0.3)
            
            # Parse and validate the response
            parsed_list = _parse_and_validate_list(
                ava_concept_raw,
                valid_options=predefined_concepts,
                max_items=MAX_RECOMMENDATIONS
            )
            
            if not parsed_list:
                msg = f"[concept_name_checker] {invalid_concepts} have no relevant alternatives in Compass."
                return False, msg
            
            msg = (
                f"[concept_name_checker] Invalid concepts: {invalid_concepts}. "
                f"Suggested alternatives: {parsed_list}. "
                f"If using alternatives, document this in your proposal."
            )
            return False, msg

        except Exception as parse_error:
            print(f"[concept_name_checker] Attempt {retry + 1} failed: {parse_error}", flush=True)
            if retry == attempt - 1:
                # Final fallback: use fuzzy matching
                parsed_list = _fuzzy_match_concepts(invalid_concepts[0], predefined_concepts, MAX_RECOMMENDATIONS)
                msg = f"[concept_name_checker] GPT failed. Fuzzy-matched alternatives: {parsed_list}"
                return False, msg
    
    msg = f"[concept_name_checker] All concepts are valid."
    return True, msg


def transcriptformer_context_checker(user_query: str, state: str, cell_type: str, gene_names: list[str] = None, disease: str = None):
    try:
        tool = TranscriptformerEmbeddingTool()
        
        # Step 1: Validate Disease
        if disease is None:
            msg = f"[transcriptformer_context_checker] Disease parameter is required but was None."
            print(msg, flush=True)
            return False, msg
            
        if disease not in tool.available_diseases:
            prompt = f"Based on the user query '{user_query}', which of the following diseases is the best fit? Return ONLY the disease name from this list: {tool.available_diseases}"
            suggested_disease = chat_completion(prompt).strip()
            
            # Clean the LLM response - remove quotes, extra words, and normalize
            suggested_disease = suggested_disease.strip('"\'`').strip()
            
            # Try to find the best match from available diseases
            if suggested_disease not in tool.available_diseases:
                # Use fuzzy matching to find the closest disease name
                best_match = process.extractOne(suggested_disease, tool.available_diseases)
                if best_match and best_match[1] >= 80:  # 80% similarity threshold
                    suggested_disease = best_match[0]
                    print(f"[transcriptformer_context_checker] LLM suggested '{suggested_disease}' (fuzzy matched from '{chat_completion(prompt).strip()}')", flush=True)
                else:
                    msg = f"[transcriptformer_context_checker] The suggested disease '{suggested_disease}' is not valid. Available diseases: {tool.available_diseases}"
                    print(msg, flush=True)
                    return False, msg
            
            disease = suggested_disease

        # Step 2: Load metadata and validate the rest of the context
        try:
            metadata = tool._load_metadata(disease)
        except Exception as e:
            msg = f"[transcriptformer_context_checker] Failed to load metadata for disease '{disease}': {e}"
            print(msg, flush=True)
            return False, msg
        
        errors = {}
        # Normalize cell_type for comparison
        normalized_cell_type = cell_type.lower().replace(" ", "_")
        if normalized_cell_type not in metadata['available_cell_types']:
            errors['cell_type'] = metadata['available_cell_types']

        if state not in metadata['available_states']:
            errors['state'] = metadata['available_states']

        # Validate gene names directly against available symbols (case-insensitive)
        if gene_names is not None:
            invalid_genes = []
            for gene_name in gene_names:
                # Check if gene exists in available symbols (case-insensitive)
                gene_upper = gene_name.upper()
                if gene_upper not in [s.upper() for s in metadata['available_symbols']]:
                    invalid_genes.append(gene_name)
            
            if invalid_genes:
                # Fuzzy matching for invalid genes - limit to reasonable number
                all_candidate_genes = []
                try:
                    for gene_name in invalid_genes:
                        top_matches = process.extract(gene_name, metadata['available_symbols'], limit=3)  # Limit per gene
                        all_candidate_genes.extend([match[0] for match in top_matches])
                    
                    # Deduplicate and limit total recommendations
                    unique_candidate_genes = sorted(list(set(all_candidate_genes)))[:MAX_RECOMMENDATIONS * 2]
                    errors['gene_names'] = unique_candidate_genes
                except Exception as e:
                    print(f"[transcriptformer_context_checker] Fuzzy matching failed: {e}", flush=True)
                    errors['gene_names'] = invalid_genes[:MAX_RECOMMENDATIONS]

        if not errors:
            return True, "[transcriptformer_context_checker] Context is valid."

        # Build error message with limited recommendations
        error_messages = []
        if 'cell_type' in errors:
            limited_cell_types = errors['cell_type'][:MAX_RECOMMENDATIONS]
            error_messages.append(f"Invalid cell_type. Top {len(limited_cell_types)} options: {limited_cell_types}")
        if 'state' in errors:
            limited_states = errors['state'][:MAX_RECOMMENDATIONS]
            error_messages.append(f"Invalid state. Available options: {limited_states}")
        if 'gene_names' in errors:
            limited_genes = errors['gene_names'][:MAX_RECOMMENDATIONS * 2]
            error_messages.append(f"Invalid gene_names. Top {len(limited_genes)} candidates: {limited_genes}")
        
        # Create concise prompt for GPT
        prompt = f"""
User query: "{user_query}"

TranscriptformerEmbeddingTool context errors:
{chr(10).join(f'- {err}' for err in error_messages)}

Return a JSON object with corrected values. Select from the provided options above.
Format: {{"disease": "{disease}", "cell_type": "...", "state": "...", "gene_names": [...]}}
Keep gene_names to max {MAX_RECOMMENDATIONS} items.
"""
        
        try:
            recommendation_raw = chat_completion(prompt, temperature=0.3)
            
            # Validate and parse the recommendation
            recommendation = _parse_and_validate_dict(recommendation_raw)
            
            if recommendation:
                msg = (
                    f"[transcriptformer_context_checker] Context invalid. "
                    f"Suggested correction: {recommendation}. "
                    f"Please refine with recommended values."
                )
            else:
                msg = (
                    f"[transcriptformer_context_checker] Context invalid. "
                    f"Issues: {'; '.join(error_messages)}. "
                    f"Please refine manually."
                )
            
        except Exception as e:
            print(f"[transcriptformer_context_checker] GPT recommendation failed: {e}", flush=True)
            msg = (
                f"[transcriptformer_context_checker] Context invalid. "
                f"Issues: {'; '.join(error_messages)}."
            )
        
        print(msg, flush=True)
        return False, msg

    except Exception as e:
        msg = f"[transcriptformer_context_checker] An unexpected error occurred: {e}"
        print(msg, flush=True)
        return False, msg


# ============================================================================
# HELPER FUNCTIONS FOR GPT RESPONSE VALIDATION
# ============================================================================

def _parse_and_validate_list(
    response: str, 
    valid_options: List[str] = None, 
    max_items: int = MAX_RECOMMENDATIONS
) -> List[str]:
    """
    Parse and validate a list response from GPT.
    
    Args:
        response: Raw GPT response
        valid_options: List of valid values (filters output if provided)
        max_items: Maximum number of items to return
        
    Returns:
        Validated and limited list of items
    """
    # Clean response - remove code blocks and extra text
    cleaned = response.strip()
    
    # Remove markdown code blocks
    if '```python' in cleaned:
        cleaned = cleaned.split('```python')[1].split('```')[0]
    elif '```json' in cleaned:
        cleaned = cleaned.split('```json')[1].split('```')[0]
    elif '```' in cleaned:
        cleaned = cleaned.split('```')[1].split('```')[0]
    
    cleaned = cleaned.strip()
    
    # Try to parse as list
    try:
        parsed = ast.literal_eval(cleaned)
        if not isinstance(parsed, list):
            # If it's a string, try to extract list-like content
            list_match = re.search(r'\[.*?\]', cleaned)
            if list_match:
                parsed = ast.literal_eval(list_match.group(0))
            else:
                return []
    except (ValueError, SyntaxError):
        # Try JSON parsing
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                return []
        except json.JSONDecodeError:
            return []
    
    # Filter by valid options if provided
    if valid_options is not None:
        valid_set = set(valid_options)
        parsed = [item for item in parsed if item in valid_set]
    
    # Limit to max_items
    return parsed[:max_items]


def _parse_and_validate_dict(response: str) -> dict:
    """
    Parse and validate a dictionary response from GPT.
    
    Args:
        response: Raw GPT response
        
    Returns:
        Parsed dictionary or None if invalid
    """
    # Clean response
    cleaned = response.strip()
    
    # Remove markdown code blocks
    if '```json' in cleaned:
        cleaned = cleaned.split('```json')[1].split('```')[0]
    elif '```python' in cleaned:
        cleaned = cleaned.split('```python')[1].split('```')[0]
    elif '```' in cleaned:
        cleaned = cleaned.split('```')[1].split('```')[0]
    
    cleaned = cleaned.strip()
    
    # Try to parse as dict
    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, dict):
            # Limit gene_names if present
            if 'gene_names' in parsed and isinstance(parsed['gene_names'], list):
                parsed['gene_names'] = parsed['gene_names'][:MAX_RECOMMENDATIONS]
            return parsed
    except (ValueError, SyntaxError):
        pass
    
    # Try JSON parsing
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            if 'gene_names' in parsed and isinstance(parsed['gene_names'], list):
                parsed['gene_names'] = parsed['gene_names'][:MAX_RECOMMENDATIONS]
            return parsed
    except json.JSONDecodeError:
        pass
    
    return None


def _fuzzy_match_tissues(tissue: str, available_tissues: List[str], max_items: int = MAX_RECOMMENDATIONS) -> List[str]:
    """
    Find similar tissues using fuzzy string matching.
    
    Args:
        tissue: Tissue name to match
        available_tissues: List of valid tissue names
        max_items: Maximum number of matches to return
        
    Returns:
        List of similar tissue names
    """
    try:
        matches = process.extract(tissue, available_tissues, limit=max_items)
        return [match[0] for match in matches]
    except Exception:
        return available_tissues[:max_items]


def _fuzzy_match_concepts(concept: str, available_concepts: List[str], max_items: int = MAX_RECOMMENDATIONS) -> List[str]:
    """
    Find similar concepts using fuzzy string matching.
    
    Args:
        concept: Concept name to match
        available_concepts: List of valid concept names
        max_items: Maximum number of matches to return
        
    Returns:
        List of similar concept names
    """
    try:
        matches = process.extract(concept, available_concepts, limit=max_items)
        return [match[0] for match in matches]
    except Exception:
        return available_concepts[:max_items]