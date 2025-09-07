#!/usr/bin/env python3
# compare_essentiality_tnseq.py
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# --- CONFIG: filenames ---
MODEL_GENES = "model_genes.csv"    # column: model_gene_id, (optional: gene_name)
TNS_EQ = "tnseq_calls.csv"         # columns: tnseq_gene_id, call_glucose, call_methanol
MODEL_CALLS = "model_calls.csv"    # columns: model_gene_id, call_glucose, call_methanol (from FBA sims)

# --- READ INPUTS ---
model_df = pd.read_csv(MODEL_GENES, dtype=str)
tn_df = pd.read_csv(TNS_EQ, dtype=str)
model_calls = pd.read_csv(MODEL_CALLS, dtype=str)

# --- SIMPLE JOIN: by gene name or provide a mapping table if IDs differ ---
# assume model_df has 'model_gene_id' and optional 'gene_name' that matches tnseq_gene_id
# merge on 'gene_name' if available, else require a mapping file
if 'gene_name' in model_df.columns:
    merged = model_df.merge(model_calls, on='model_gene_id', how='left').merge(
        tn_df, left_on='gene_name', right_on='tnseq_gene_id', how='left')
else:
    # try direct merge on ids
    merged = model_df.merge(model_calls, on='model_gene_id', how='left').merge(
        tn_df, left_on='model_gene_id', right_on='tnseq_gene_id', how='left')

# fill NaN calls with 'unknown'
for col in ['call_glucose_x','call_methanol_x','call_glucose_y','call_methanol_y']:
    if col in merged.columns:
        merged[col] = merged[col].fillna('unknown')

# rename columns for clarity
merged.rename(columns={
    'call_glucose_x':'model_call_glucose','call_methanol_x':'model_call_methanol',
    'call_glucose_y':'tn_call_glucose','call_methanol_y':'tn_call_methanol',
    'tnseq_gene_id':'tnseq_gene_id'
}, inplace=True)

# Save detailed mapping table
merged.to_csv("Supplementary_Table_S6_gene_mapping_and_calls.csv", index=False)

# --- Function to compute confusion metrics ---
def compute_metrics(model_calls, tn_calls, condition_name):
    # consider only genes where tn_calls is 'essential' or 'nonessential'
    valid = (tn_calls!='unknown') & (model_calls!='unknown')
    m = model_calls[valid].values
    t = tn_calls[valid].values
    # map to boolean: essential=True
    def to_bool(arr):
        return np.array([1 if x.lower()=='essential' else 0 for x in arr])
    mb = to_bool(m); tb = to_bool(t)
    TP = int(np.sum((mb==1)&(tb==1)))
    FP = int(np.sum((mb==1)&(tb==0)))
    FN = int(np.sum((mb==0)&(tb==1)))
    TN = int(np.sum((mb==0)&(tb==0)))
    precision = TP/(TP+FP) if (TP+FP)>0 else np.nan
    recall = TP/(TP+FN) if (TP+FN)>0 else np.nan
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else np.nan
    acc = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else np.nan
    return {'condition':condition_name,'TP':TP,'FP':FP,'FN':FN,'TN':TN,
            'precision':precision,'recall':recall,'f1':f1,'accuracy':acc}

# compute for glucose
metrics_glucose = compute_metrics(merged['model_call_glucose'], merged['tn_call_glucose'],'glucose')
metrics_methanol = compute_metrics(merged['model_call_methanol'], merged['tn_call_methanol'],'methanol')
metrics_df = pd.DataFrame([metrics_glucose, metrics_methanol])
metrics_df.to_csv("Supplementary_Table_S7_confusion_metrics.csv", index=False)

# --- Venn and confusion visualizations (methanol example) ---
# genes predicted essential by model vs tnseq (methanol)
set_model_ess = set(merged.loc[merged['model_call_methanol'].str.lower()=='essential','model_gene_id'])
set_tn_ess = set(merged.loc[merged['tn_call_methanol'].str.lower()=='essential','model_gene_id'])

plt.figure(figsize=(5,5))
venn2([set_model_ess, set_tn_ess], set_labels=('Model essential (methanol)','Tn-seq essential (methanol)'))
plt.title('Overlap of essential genes (methanol)')
plt.savefig("Fig_Sx_venn_methanol.png", dpi=300)
plt.close()

# confusion matrix heatmap
import seaborn as sns
from sklearn.metrics import confusion_matrix
valid = merged[merged['tn_call_methanol'].isin(['essential','nonessential']) & merged['model_call_methanol'].isin(['essential','nonessential'])]
y_true = [1 if x=='essential' else 0 for x in valid['tn_call_methanol']]
y_pred = [1 if x=='essential' else 0 for x in valid['model_call_methanol']]
cm = confusion_matrix(y_true,y_pred)
cm_df = pd.DataFrame(cm, index=['Tn_non','Tn_ess'], columns=['Model_non','Model_ess'])
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix (methanol)')
plt.savefig('Fig_Sx_confusion_methanol.png', dpi=300)
plt.close()

print("Done. Outputs:\n - Supplementary_Table_S6_gene_mapping_and_calls.csv\n - Supplementary_Table_S7_confusion_metrics.csv\n - Fig_Sx_venn_methanol.png\n - Fig_Sx_confusion_methanol.png")
