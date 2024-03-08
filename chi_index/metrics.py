import pandas as pd
from scipy.stats import chi2_contingency

def chi_index_score(X: pd.DataFrame, k, class_name = "Class", clusters_col_name = "cluster"):
    if class_name is not None:
        X.rename(columns={class_name: 'Class'}, inplace=True)
    if clusters_col_name is not None:
        X.rename(columns={clusters_col_name: 'cluster'}, inplace=True)
             
    df_cluster = pd.crosstab(X.cluster, X.Class, normalize='index')
    df_features = pd.crosstab(X.Class, X.cluster, normalize='index').T

    df_cluster.to_csv(f'results/{k}_df_cluster_result.csv', sep='\t')
    df_features.to_csv(f'results/{k}_df_features_result.csv', sep='\t')

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

    print(f'r={r},c={c},chi_row={chi_row},chi_col={chi_col},chi_row_max={chi_row_max},chi_col_max={chi_col_max}')

    chi_row_norm = chi_row / chi_row_max
    chi_col_norm = chi_col / chi_col_max

    chi_index_value = chi_row_norm + chi_col_norm - abs(chi_row_norm - chi_col_norm)
    #return (k, r,c, chi_row, chi_row_max, chi_col, chi_col_max, chi_index_value)
    return chi_index_value