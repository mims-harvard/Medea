# from collections import Counter
from typing import Dict, List
import numpy as np
import pandas as pd
import json, os

# from tool_space.read_data import load_PPI_data, read_labels_from_evidence
# from tool_space.extract_txdata_utils import *

from read_data import load_PPI_data, read_labels_from_evidence
from extract_txdata_utils import *

MAX_RETRY = 10  # To mitigate the effect of random state, we will redo data splitting for MAX_RETRY times if the number of positive samples in test set is less than TEST_CELLTYPE_POS_NUM_MIN
TEST_CELLTYPE_POS_NUM_MIN = 5 # For each cell type, the number of positive samples in test set must be greater than 5, or else the disease won't be evlauated


def get_labels_from_evidence(celltype_protein_dict: Dict[str, List[str]], disease: str, evidence_dir: str, all_drug_targets_path: str, curated_disease_dir: str, chembl2db_path: str, 
                             positive_protein_prefix: str, negative_protein_prefix: str, raw_data_prefix: str, 
                             overwrite: bool, disease_drug_evidence_prefix = "", wandb = None):
    """
    Get positive and negative targets associated with each disease and descendants.
    """
    
    # Read in CHEMBL data
    chembl_db_df = pd.read_table(chembl2db_path) 
    chembl_db_df.columns = ['chembl', 'db']
    chembl2db = chembl_db_df.set_index('chembl').to_dict()['db']

    # Read in all approved drug-target data
    dti_tab = pd.read_csv(all_drug_targets_path, index_col=0)  # approved drug-target table
    assert dti_tab['Drug IDs'].isna().sum()==0
    dti_tab = dti_tab[dti_tab.Species=='Humans']
    druggable_targets = dti_tab[['Gene Name', 'GenAtlas ID']]
    druggable_targets = set(druggable_targets.values.flatten())
    druggable_targets.remove(np.nan)  # all approved drugs' targets

    evidence_files = os.listdir(evidence_dir)
    
    positive_proteins = {}
    negative_proteins = {}
    all_relevant_proteins = {}

    if not overwrite:
        positive_proteins, negative_proteins, all_relevant_proteins = read_labels_from_evidence(positive_protein_prefix, negative_protein_prefix, raw_data_prefix, positive_proteins, negative_proteins, all_relevant_proteins)

    if len(positive_proteins) == 0:

        # Get all disease descendants (we include indirect evidence)
        all_disease = get_disease_descendants(disease, source='ot', curated_disease_dir=curated_disease_dir)
        if wandb is not None:
            wandb.log({f'number of disease descendants':len(all_disease)})
        
        # Look for clinically relevant evidence on targets related to each of the diseases.
        disease_drug_evidence_data = get_all_drug_evidence(evidence_files, evidence_dir, all_disease, chembl2db)

        # Get all associated targets of disease
        all_associated_targets, ensg2otgenename = get_all_associated_targets(disease)

        # Convert clinically relevant targets to gene names
        disease_drug_targets = evidence2genename(disease_drug_evidence_data, ensg2otgenename)
        
        # Saving disease/drug-target evidence
        if disease_drug_evidence_prefix != "":
            disease_drug_evidence_data["targetId_genename"] = disease_drug_evidence_data["targetId"].map(ensg2otgenename)
            # print(disease_drug_evidence_data)
            # print(Counter("Phase " + str(a) + "," + str(b) for a, b in zip(disease_drug_evidence_data["clinicalPhase"].tolist(), disease_drug_evidence_data["clinicalStatus"].tolist())))
            # print(disease_drug_evidence_data["drugId"].unique())
            disease_drug_evidence_data.to_csv(disease_drug_evidence_prefix + disease + ".csv", index = False, sep = "\t")

        # Get positive and negative labels for proteins
        positive_proteins = {ct: list(disease_drug_targets.intersection(ppi_proteins)) for ct, ppi_proteins in celltype_protein_dict.items()}  # PPI proteins associated with the disease with drug or clinical candidate > II's evidence
        negative_proteins = {ct: list(set(ppi_proteins).difference(all_associated_targets).intersection(druggable_targets)) for ct, ppi_proteins in celltype_protein_dict.items()}  # PPI proteins that are not associated with the disease except for text mining, but are still druggable

        # Collect all targets (for diseases, not considering the intersection with PPI).
        all_relevant_proteins = list(disease_drug_targets)

        # Save data
        with open(positive_protein_prefix + disease + '.json', 'w') as f:
            json.dump(positive_proteins, f)
        with open(negative_protein_prefix + disease + '.json', 'w') as f:
            json.dump(negative_proteins, f)
        with open(raw_data_prefix + disease + '.json', 'w') as f:
            json.dump(all_relevant_proteins, f)
        
        # Plot protein counts
        tmp_positive_proteins = {disease: positive_proteins}
        # print(tmp_positive_proteins)
        positive_protein_counts_celltype = pd.DataFrame(tmp_positive_proteins).rename(
            index={ind:ind[:-2] for ind in tmp_positive_proteins.keys()}
        ).reset_index()
        # print(positive_protein_counts_celltype.head(5))
        # Use melt, but immediately rename 'variable' if it's a potential duplicate column name.
        positive_protein_counts_celltype = positive_protein_counts_celltype.melt(id_vars = ['index'])
        # print(positive_protein_counts_celltype.head(5))
        
        # Group by the new 'index' and 'variable_new', then aggregate.
        positive_protein_counts_celltype = positive_protein_counts_celltype.groupby(by=['index', 'variable']).aggregate(list)
        # print(positive_protein_counts_celltype.head(5))
        # Apply a function to calculate the unique counts.
        positive_protein_counts_celltype = positive_protein_counts_celltype.applymap(lambda x: len(np.unique(sum(x, start=[]))))
        # print(positive_protein_counts_celltype.head(5))
        # Reset index without causing a column name conflict.
        positive_protein_counts_celltype = positive_protein_counts_celltype.reset_index()
        # print(positive_protein_counts_celltype.head(5))
        # groupby(by=['index', 'variable']).aggregate(list).applymap(lambda x: len(np.unique(sum(x, start = [])))).reset_index()
        # sns.barplot(x='variable', y='value', data=positive_protein_counts_celltype, hue='index')

    return positive_proteins, negative_proteins, all_relevant_proteins


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--celltype_ppi", type=str, help="Filename (prefix) of cell type PPI.")
    parser.add_argument('--disease', type=str)
    parser.add_argument('--evidence_dir', type=str) 
    parser.add_argument('--all_drug_targets_path', type=str, default="../MedeaDB/therapeutic_target_task/all_approved_oct2022.csv")
    parser.add_argument('--curated_disease_dir', type=str) 
    parser.add_argument('--chembl2db_path', type=str)  # Download mapping from ChEMBL id to DrugBank id from https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src2.txt (version: 13-Apr-2022)
    parser.add_argument('--disease_drug_evidence_prefix', type=str, default="../MedeaDB/therapeutic_target_task/disease_drug_evidence_")
    parser.add_argument('--positive_proteins_prefix', type=str, default="../MedeaDB/therapeutic_target_task/positive_proteins_")
    parser.add_argument('--negative_proteins_prefix', type=str, default="../MedeaDB/therapeutic_target_task/negative_proteins_")
    parser.add_argument('--raw_data_prefix', type=str, default="../MedeaDB/therapeutic_target_task/raw_targets_")
    args = parser.parse_args()
    # celltype_ppi = ppi_edgelist/
    celltype_protein_dict = load_PPI_data(args.celltype_ppi)

    positive_proteins, negative_proteins, all_relevant_proteins = get_labels_from_evidence(celltype_protein_dict, args.disease, args.evidence_dir, args.all_drug_targets_path, args.curated_disease_dir, args.chembl2db_path, args.positive_proteins_prefix, args.negative_proteins_prefix, args.raw_data_prefix, overwrite = True, disease_drug_evidence_prefix = args.disease_drug_evidence_prefix)
    
    for c, v in positive_proteins.items():
        assert len(v) == len(set(v).intersection(set(all_relevant_proteins)))


if __name__ == '__main__':
    main()
    
# Example usage
# python prepare_txdata.py \
#     --celltype_ppi ../MedeaDB/networks/ppi_edgelists/ \
#     --disease "EFO_0000616" \
#     --evidence_dir ../MedeaDB/sourceId_chembl_evidence/ \
#     --curated_disease_dir ../MedeaDB/curated_diseases/ \
#     --chembl2db_path ../MedeaDB/therapeutic_target_task/chembl2db.txt