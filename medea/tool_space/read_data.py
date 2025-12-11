import glob
import json
import networkx as nx
import numpy as np
import pandas as pd
import torch


def load_PPI_data(ppi_dir):
    ppi_layers = dict()
    for f in glob.glob(ppi_dir + "*"): # Expected format of filename: <PPI_DIR>/<CONTEXT>.<suffix>
        context = f.split(ppi_dir)[1].split(".")[0]
        # print(f"read: {f}")
        ppi = nx.read_edgelist(f)
        ppi_layers[context] = ppi
    return ppi_layers


def read_labels_from_evidence(positive_protein_prefix, negative_protein_prefix, raw_data_prefix, positive_proteins={}, negative_proteins={}, all_relevant_proteins={}):
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


def load_data(embed_path: str, labels_path: str, positive_proteins_prefix: str, negative_proteins_prefix: str, raw_data_prefix: str):
    
    embed = torch.load(embed_path)
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

    positive_proteins, negative_proteins, all_relevant_proteins = read_labels_from_evidence(positive_proteins_prefix, negative_proteins_prefix, raw_data_prefix)
    assert len(positive_proteins) > 0

    return embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, all_relevant_proteins

