# double_gene_analysis.py
import cobra, pandas as pd, itertools, multiprocessing as mp, os
from cobra.flux_analysis import single_gene_deletion, double_gene_deletion, flux_variability_analysis
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_FILE = "iLD1283.xml"  # change as needed
PRIORITY_SUBSYSTEMS = ["Glycolysis","TCA cycle","Pentose phosphate pathway","Oxidative phosphorylation","Amino acid metabolism","Transport"]

# load model
model = cobra.io.read_sbml_model(MODEL_FILE)

# 1) single gene deletion to filter essentials
print("Running single gene deletion...")
sgd = single_gene_deletion(model, processes=8)  # returns DataFrame with 'growth'
sgd['growth_fraction'] = sgd['growth'] / sgd['growth'].max()
nonessential_genes = sgd[sgd['growth_fraction'] >= 0.01].index.tolist()  # >=1% WT taken as nonessential

# 2) build prioritized gene pool by subsystem annotation if available
def gene_in_priority(g):
    # check reactions associated with gene for subsystem annotation
    for r in model.genes.get_by_id(g).reactions:
        subs = getattr(r,'subsystem',None)
        if subs:
            for p in PRIORITY_SUBSYSTEMS:
                if p.lower() in subs.lower():
                    return True
    return False

priority_genes = [g for g in nonessential_genes if gene_in_priority(g)]
print(f"Priority gene count: {len(priority_genes)}")

# 3) Exhaustive double-gene deletion within priority pool
pairs = list(itertools.combinations(priority_genes,2))
print(f"Testing {len(pairs)} pairs")
# function to test pair
def test_pair(pair):
    g1,g2 = pair
    res = double_gene_deletion(model, [g1,g2], processes=1)  # returns dict or DataFrame depending on cobra version
    # extract growth
    # handle different output types
    if isinstance(res, dict):
        growth = list(res.values())[0]
    else:
        growth = res['growth'].iloc[0]
    return (g1,g2,growth)

# parallel map
pool = mp.Pool(processes=8)
results = list(tqdm(pool.imap(test_pair, pairs), total=len(pairs)))
pool.close()

# aggregate
df = pd.DataFrame(results, columns=['geneA','geneB','growth'])
df['growth_frac'] = df['growth'] / model.slim_optimize()  # normalize by WT growth
# classify
def classify(gf):
    if gf < 0.01: return 'SL'
    elif gf < 0.5: return 'Strong'
    elif gf < 0.9: return 'Moderate'
    else: return 'Neutral'
df['class'] = df['growth_frac'].apply(classify)
df.to_csv("Supplementary_Table_S8_double_gene_results.csv", index=False)

# 4) heuristics: flux coupling & neighbor pairs
from cobra.flux_analysis import find_essential_reactions # or use flux_coupling in cobrapy versions
# ... (omitted for brevity) ...
# 5) frequency analysis
freq = pd.concat([df['geneA'],df['geneB']]).value_counts().rename_axis('gene').reset_index(name='freq')
freq.to_csv("gene_freq_in_SLpairs.csv", index=False)

# 6) chokepoint and FVA
fva = flux_variability_analysis(model, fraction_of_optimum=0.99)
fva.to_csv("FVA_results.csv")
# chokepoint: identify reactions that uniquely produce or consume a metabolite
# (implement simple algorithm: count producers/consumers per metabolite)
# 7) visualizations: network of SL edges
G = nx.Graph()
for _,r in df[df['class']=='SL'].iterrows():
    G.add_edge(r['geneA'], r['geneB'])
sizes = [freq.set_index('gene').loc[n,'freq']*50 for n in G.nodes()]
pos = nx.spring_layout(G, k=0.5)
plt.figure(figsize=(10,10))
nx.draw_networkx(G, pos, node_size=sizes, with_labels=True)
plt.savefig("Fig4c_SL_network.png", dpi=300)
