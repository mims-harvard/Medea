#!/usr/bin/env python3
"""
High-performance tool to load cell-type and disease-state specific gene embeddings.

This script provides a class-based tool that queries a pre-generated,
high-performance embedding store. It supports both Ensembl IDs and official
gene symbols and features near-instant initialization and on-demand data loading.
"""

import os
import sys
import json
import gzip
import numpy as np
from typing import Union

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .gpt_utils import chat_completion
from .env_utils import get_env_with_error
from thefuzz import process

class TranscriptformerEmbeddingTool:
    """
    A tool for fast, on-demand querying of gene embeddings from a prepared store.
    """
    def __init__(self):
        """
        Initializes the tool by identifying available disease-specific embedding stores.
        """
        medeadb_path = get_env_with_error(
            "MEDEADB_PATH",
            default="/root/MedeaDB",
            required=False,
            description="accessing Transcriptformer embeddings"
        )
        self.base_dir = os.path.join(medeadb_path, "transcriptformer_embedding", "embedding_store")
        if not os.path.exists(self.base_dir):
            error_msg = (
                f"\n\n❌ Transcriptformer embedding store not found!\n\n"
                f"Expected location: {self.base_dir}\n\n"
                f"To fix this issue:\n"
                f"1. Verify your MEDEADB_PATH is set correctly\n"
                f"2. Ensure the transcriptformer_embedding/embedding_store directory exists\n"
                f"3. Download or generate the required embedding data\n\n"
                f"Current MEDEADB_PATH: {medeadb_path}\n"
            )
            raise FileNotFoundError(error_msg)
        self.available_diseases = [d.lower().replace(" ", "_") for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        self.metadata_cache = {}

    def _load_metadata(self, disease: str):
        """
        Loads metadata for a given disease and caches it.
        """
        if disease in self.metadata_cache:
            return self.metadata_cache[disease]

        if disease not in self.available_diseases:
            raise FileNotFoundError(f"Disease '{disease}' is not available/invalid. Please choose from the following avaliable diseases: {self.available_diseases}")

        store_path = os.path.join(self.base_dir, disease.replace(" ", "_"))
        metadata_path = os.path.join(store_path, "metadata.json.gz")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}. Please run prepare_embedding_store.py first.")

        print(f"🛠️  Initializing tool from embedding store: {os.path.basename(store_path)}...")
        with gzip.open(metadata_path, 'rt', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.metadata_cache[disease] = {
            'store_path': store_path,
            'ensembl_ids_ordered': metadata['ensembl_ids_ordered'],
            'gene_to_idx': {gene: i for i, gene in enumerate(metadata['ensembl_ids_ordered'])},
            'symbol_to_ensembl': metadata['gene_map_symbol_to_ensembl'],
            'available_symbols': sorted(list(metadata['gene_map_symbol_to_ensembl'].keys())),
            'groups_meta': {k.lower().replace(" ", "_"): v for k, v in metadata['groups'].items()},
            'available_cell_types': sorted(list(set(details['cell_type'].lower().replace(" ", "_") for details in metadata['groups'].values()))),
            'available_states': sorted(list(set(details['disease_state'].lower().replace(" ", "_") for details in metadata['groups'].values()))),
        }
        print("✅ Tool initialized successfully (metadata loaded).")
        return self.metadata_cache[disease]

    def get_embedding_for_context(self, state: str, cell_type: str, gene_names: Union[list[str], None], disease: str):
        """
        Retrieves gene embeddings for a list of genes, accepting either Ensembl IDs or gene symbols.
        Loads the required embedding matrix from its .npy file on demand.
        
        Args:
            state (str): The disease state context, e.g., 'control' or 'disease'.
            cell_type (str): The specific cell type context, e.g., 'B cell'.
            gene_names (list[str] | None): A list of gene identifiers (symbols or Ensembl IDs). 
                                         If None, returns embeddings for all genes in the context.
            disease (str): The name of the disease to load embeddings for.
        """
        disease = disease.lower().replace(" ", "_")
        state = state.lower().replace(" ", "_")
        cell_type = cell_type.lower().replace(" ", "_")
        metadata = self._load_metadata(disease)
        
        context_info = []
        embeddings = {}
        invalid_genes = []

        # 1. Validate state
        if state not in metadata['available_states']:
            context_info.append(f"Invalid state '{state}'. Available states: {metadata['available_states']}")
            return None, context_info

        # 2. Validate cell_type
        if cell_type not in metadata['available_cell_types']:
            context_info.append(f"Invalid cell_type '{cell_type}'. Available cell types: {metadata['available_cell_types']}")
            return None, context_info

        # 3. Handle gene_names parameter
        if gene_names is None:
            # Return all genes for this context
            print(f"🔄 Loading all gene embeddings for context: {disease} - {state} - {cell_type}")
            # Create a mapping of all genes using their Ensembl IDs as keys
            for ensembl_id in metadata['ensembl_ids_ordered']:
                # Find the gene symbol for this Ensembl ID
                gene_symbol = None
                for symbol, ens_id in metadata['symbol_to_ensembl'].items():
                    if ens_id == ensembl_id:
                        gene_symbol = symbol
                        break
                
                if gene_symbol:
                    embeddings[gene_symbol] = ensembl_id
                else:
                    # If no symbol found, use Ensembl ID as key
                    embeddings[ensembl_id] = ensembl_id
        else:
            # Validate specific gene names and get Ensembl IDs
            for gene_name in gene_names:
                ensembl_id = None
                if gene_name.upper().startswith('ENSG'):
                    ensembl_id = gene_name.upper()
                    if ensembl_id not in metadata['gene_to_idx']:
                        invalid_genes.append(gene_name)
                else:
                    ensembl_id = metadata['symbol_to_ensembl'].get(gene_name.upper())
                    if not ensembl_id:
                        invalid_genes.append(gene_name)
                
                if ensembl_id:
                    embeddings[gene_name] = ensembl_id

            if invalid_genes:
                # Use fuzzy matching to find the top 20 most similar gene names for each invalid gene
                all_candidate_genes = []
                for gene_name in invalid_genes:
                    top_matches = process.extract(gene_name, metadata['available_symbols'], limit=20)
                    all_candidate_genes.extend([match[0] for match in top_matches])
                
                unique_candidate_genes = sorted(list(set(all_candidate_genes)))

                prompt = f"""
                The following genes are not valid or available in our database for the disease '{disease}': {invalid_genes}.
                Please analyze the invalid gene names and suggest the most likely correct gene symbols from the following list of candidate symbols.
                For each invalid gene, provide a list of the top 5 most relevant suggestions.

                Candidate gene symbols:
                {unique_candidate_genes}

                Return a JSON object where keys are the invalid gene names and values are lists of suggested gene symbols.
                """
                recommended_genes = chat_completion(prompt)
                context_info.append(f"Invalid genes: {invalid_genes}. Recommended alternatives: {recommended_genes}")

        if not embeddings:
            return None, context_info

        # --- Retrieve Embeddings ---
        group_key = f"{cell_type}_{state}".replace(' ', '_').replace('(', '').replace(')', '')
        
        if group_key not in metadata['groups_meta']:
            available_keys = list(metadata['groups_meta'].keys())
            return None, [f"Could not find data for combination: state='{state}', cell_type='{cell_type}'. Available groups: {available_keys}"]

        # On-demand loading of the specific .npy file
        npy_path = os.path.join(metadata['store_path'], f"{group_key}.npy")
        if not os.path.exists(npy_path):
            return None, [f"Embedding file not found for group '{group_key}' at {npy_path}"]
            
        embedding_matrix = np.load(npy_path)
        
        final_embeddings = {}
        for gene_name, ensembl_id in embeddings.items():
            gene_idx = metadata['gene_to_idx'].get(ensembl_id)
            if gene_idx is not None:
                embedding_vector = embedding_matrix[gene_idx].astype(np.float32) # De-quantize
                final_embeddings[gene_name] = embedding_vector

        return final_embeddings, context_info


if __name__ == '__main__':
    print("\n--- Running High-Performance Embedding Tool Examples ---")
    
    try:
        # --- Test Case 1: Successful Initialization ---
        print("\n--- Test Case 1: Successful Initialization ---")
        tool = TranscriptformerEmbeddingTool()
        print(f"   ✅ Available diseases: {tool.available_diseases}")

        # --- Test Case 2: Valid Query ---
        if tool.available_diseases:
            print("\n--- Test Case 2: Valid Query ---")
            valid_disease = "sjogren syndrome"
            valid_state = 'normal'
            valid_cell_type = 'acinar_cell_of_salivary_gland'
            valid_symbols = None
            embeddings, context = tool.get_embedding_for_context(valid_state, valid_cell_type, valid_symbols, disease=valid_disease)
            assert embeddings, f"Failed valid query: {context}"
            print(f"   ✅ Success! Retrieved embeddings for {len(embeddings)} genes.")

            # --- Test Case 3: Invalid Gene Name ---
            print("\n--- Test Case 3: Invalid Gene Name ---")
            embeddings, context = tool.get_embedding_for_context(valid_state, valid_cell_type, ['INVALID_GENE', 'CD79A'], disease=valid_disease)
            assert context and 'INVALID_GENE' in context[0], "Failed to catch invalid gene"
            print(f"   ✅ Success! Context returned: {context[0]}")

            # --- Test Case 4: Invalid Cell Type ---
            print("\n--- Test Case 4: Invalid Cell Type ---")
            embedding, context = tool.get_embedding_for_context(valid_state, 'INVALID_CELL_TYPE', valid_symbols, disease=valid_disease)
            assert context is not None, "Failed to catch invalid cell type"
            print(f"   ✅ Success! Context returned: {context[0]}")

            # --- Test Case 5: Invalid State ---
            print("\n--- Test Case 5: Invalid State ---")
            embedding, context = tool.get_embedding_for_context('INVALID_STATE', valid_cell_type, valid_symbols, disease=valid_disease)
            assert context is not None, "Failed to catch invalid state"
            print(f"   ✅ Success! Context returned: {context[0]}")

            # --- Test Case 6: All Genes Query ---
            print("\n--- Test Case 6: All Genes Query ---")
            embeddings, context = tool.get_embedding_for_context(valid_state, valid_cell_type, None, disease=valid_disease)
            assert embeddings is not None and len(embeddings) > 0, "Failed to retrieve all genes"
            print(f"   ✅ Success! Retrieved embeddings for all {len(embeddings)} genes in the context.")

        # --- Test Case 7: Invalid Disease Query ---
        print("\n--- Test Case 7: Invalid Disease Query ---")
        embedding, context = tool.get_embedding_for_context('control', 'b_cell', ['CD79A'], disease='INVALID_DISEASE')
        assert context is not None, "Failed to catch invalid disease"
        print(f"   ✅ Success! Context returned: {context[0]}")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not run examples. {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
        
    from tool_space.id_checkers import transcriptformer_context_checker
    print(transcriptformer_context_checker("I want to know the expression of CD79A in a b cell from a healthy patient", "healthy", "b cell", ["CD79A"], "disease"))
    
    # Test with None gene_names
    print(transcriptformer_context_checker("Extract all gene embeddings for salivary gland acinar cells", "control", "acinar_cell", None, "sjogren_syndrome"))