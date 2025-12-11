import sys
import os
from sklearn.preprocessing import StandardScaler
from .env_utils import get_medeadb_path as _get_medeadb_path

sys.path.insert(0, os.path.join(_get_medeadb_path(), 'immune-compass/COMPASS'))
from compass.tokenizer import CANCER_CODE
from compass import loadcompass
import torch


def get_top_columns_per_row(df, top_n=44, exclude=['CANCER', 'Reference']):
    """
    Given a DataFrame, returns a list where each sublist contains the column names
    sorted by value in descending order for each row, limited to the top_n columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame with numerical values.
    top_n (int): Number of top columns to select per row.

    Returns:
    list: A list of lists, where each sublist contains the sorted column names for a row.
    """
    sorted_conecepts = [list(row.sort_values(ascending=False).index[:top_n]) for _, row in df.iterrows()]
    return [[(col, row[col]) for col in sorted_conecepts[i] if col not in exclude] for i, (_, row) in enumerate(df.iterrows())]  # Updated to return a 2D list of column names with values


def compass_predict(
        df_tpm, 
        root_path=None, 
        ckp_path="pft_leave_IMVigor210.pt", 
        device="cuda", 
        threshold=0.5, 
        batch_size=128
    ):
    # Set default root_path if not provided
    if root_path is None:
        root_path = os.path.join(_get_medeadb_path(), "immune-compass/checkpoint")
    
    responder = False
    tpm_path = os.path.join(root_path, ckp_path)
    
    if not os.path.exists(tpm_path):
        raise FileNotFoundError(f"[compass_predict] The file does not exist or the file path is invalid: {tpm_path}")
    finetuner = loadcompass(tpm_path, weights_only=False, map_location=torch.device(device))
    finetuner.count_parameters()
    df_tpm.index.name = 'Index'
    dfcx = df_tpm.copy()
    
    if device == 'cpu': finetuner.device = 'cpu'
    _, _, dfct = finetuner.extract(dfcx, batch_size=batch_size, with_gene_level=True)
    _, dfpred = finetuner.predict(dfcx)
    sorted_cell_concept = get_top_columns_per_row(dfct)
    
    # Generalized logic to determine responder status based on maximum predicted value
    if dfpred.iloc[:, 1].max() >= threshold: 
        responder = True

    return responder, sorted_cell_concept[0] # return the responsiveness prediction + cell concept scores


if __name__ == "__main__":
    import pandas as pd
    patient_file = os.path.join(_get_medeadb_path(), "immune-compass/patients/IMVigor210-0257bb-ar-0257bbb.pkl")
    df_tpm = pd.read_pickle(patient_file)
    dfpred, sorted_cell_concept = compass_predict(df_tpm, device='cuda')
    print(sorted_cell_concept)
    print(dfpred)
    