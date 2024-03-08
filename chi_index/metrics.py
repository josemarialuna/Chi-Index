import os
import pandas as pd
from scipy.stats import chi2_contingency

def chi_index_score(cluster_labels, class_labels, k, verbose=False, save_results=False):
          
    df_cluster = pd.crosstab(cluster_labels, class_labels, normalize='index')    
    df_features = pd.crosstab(class_labels, cluster_labels, normalize='index').T

    try:
        if save_results:
            if save_results:
                os.makedirs('results', exist_ok=True)
            df_cluster.to_csv(f'results/{k}_df_cluster_result.csv', sep='\t')
            df_features.to_csv(f'results/{k}_df_features_result.csv', sep='\t')
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")

    df_cluster *= 100
    df_features *= 100

    chi_row = chi2_contingency(df_cluster)[0]
    chi_col = chi2_contingency(df_features)[0]

    r = df_cluster.shape[0]
    c = df_cluster.shape[1]

    if r <= c:
        chi_row_max = 100 * r * (r - 1)
        chi_col_max = 100 * c * (r - 1)
    else:
        chi_row_max = 100 * r * (c - 1)
        chi_col_max = 100 * c * (c - 1)
    if verbose:
        print(f'r={r},c={c},chi_row={chi_row},chi_col={chi_col},chi_row_max={chi_row_max},chi_col_max={chi_col_max}')

    chi_row_norm = chi_row / chi_row_max
    chi_col_norm = chi_col / chi_col_max

    chi_index_value = chi_row_norm + chi_col_norm - abs(chi_row_norm - chi_col_norm)

    return chi_index_value