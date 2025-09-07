# compare_models.py
import cobra
import pandas as pd
import os
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
import seaborn as sns

# CONFIG: list model filenames and labels
models = {
    "iLD1283": "iLD1283.xml",
    "iRY1243": "iRY1243.xml",
    "iMT1026": "iMT1026.xml",
    # add paths for other models
}

# helper: normalize metabolite id (remove compartment suffix)
def norm_met_id(mid):
    # common suffixes: _c _m _e _p etc.
    for suf in ["_c","_m","_e","_p","_x"]:
        if mid.endswith(suf):
            return mid[:-len(suf)]
    return mid

def reaction_signature(rxn):
    # create canonical stoichiometry string sorted by metabolite name and coefficient
    terms = []
    for met, coeff in rxn.metabolites.items():
        mid = norm_met_id(met.id)
        terms.append(f"{coeff:.6g} {mid}")
    terms_sorted = sorted(terms)
    # reversible flag
    rev = "<=>" if rxn.reversibility else "=>"
    return rev + " " + " + ".join(terms_sorted)

# load models and extract sets
genes_sets = {}
rxn_sets = {}
met_sets = {}
rxn_info = defaultdict(dict)

for name, fname in models.items():
    print("Loading", name, fname)
    model = cobra.io.read_sbml_model(fname) if fname.endswith(".xml") else cobra.io.load_json_model(fname)
    # genes
    genes = set([g.id for g in model.genes])
    genes_sets[name] = genes
    # metabolites
    mets = set([norm_met_id(m.id) for m in model.metabolites])
    met_sets[name] = mets
    # reactions: use signature for comparison, also keep original id->signature map
    sigs = set()
    for r in model.reactions:
        sig = reaction_signature(r)
        sigs.add(sig)
        rxn_info[name][sig] = {"id": r.id, "name": r.name, "ec_number": getattr(r, "ec_number", "")}
    rxn_sets[name] = sigs

# pairwise and global intersections
def pairwise_counts(sets_dict):
    names = list(sets_dict.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = names[i]; b = names[j]
            inter = len(sets_dict[a].intersection(sets_dict[b]))
            rows.append({"model_a":a,"model_b":b,"shared":inter,"a_total":len(sets_dict[a]),"b_total":len(sets_dict[b])})
    return pd.DataFrame(rows)

genes_pair = pairwise_counts(genes_sets)
mets_pair = pairwise_counts(met_sets)
rxn_pair = pairwise_counts(rxn_sets)

genes_pair.to_csv("comparison_genes_pairwise.csv", index=False)
mets_pair.to_csv("comparison_mets_pairwise.csv", index=False)
rxn_pair.to_csv("comparison_rxns_pairwise.csv", index=False)

# global shared sets
from functools import reduce
all_names = list(models.keys())
genes_shared_all = set.intersection(*[genes_sets[n] for n in all_names])
mets_shared_all = set.intersection(*[met_sets[n] for n in all_names])
rxn_shared_all = set.intersection(*[rxn_sets[n] for n in all_names])

pd.Series(list(genes_shared_all)).to_csv("genes_shared_all.csv", index=False, header=["gene_id"])
pd.Series(list(mets_shared_all)).to_csv("mets_shared_all.csv", index=False, header=["met_id"])
# for reactions, also export mapping to original model ids if available
with open("rxns_shared_all.csv","w") as f:
    f.write("signature," + ",".join([n+"_rxn_id" for n in all_names]) + "\n")
    for sig in rxn_shared_all:
        row = [sig]
        for n in all_names:
            row.append(rxn_info[n].get(sig, {}).get("id",""))
        f.write(",".join(row) + "\n")

# also compute unique items per model
for n in all_names:
    unique_genes = genes_sets[n] - set.union(*[genes_sets[m] for m in all_names if m!=n])
    pd.Series(list(unique_genes)).to_csv(f"genes_unique_{n}.csv", index=False, header=["gene_id"])
    unique_mets = met_sets[n] - set.union(*[met_sets[m] for m in all_names if m!=n])
    pd.Series(list(unique_mets)).to_csv(f"mets_unique_{n}.csv", index=False, header=["met_id"])
    unique_rxns = rxn_sets[n] - set.union(*[rxn_sets[m] for m in all_names if m!=n])
    # map to ids
    with open(f"rxns_unique_{n}.csv","w") as f:
        f.write("signature,"+n+"_rxn_id\n")
        for sig in unique_rxns:
            f.write(sig.replace(",",";") + "," + rxn_info[n].get(sig,{}).get("id","") + "\n")

# simple venn for first three models (if >=3)
if len(all_names) >= 3:
    a,b,c = all_names[:3]
    plt.figure(figsize=(6,6))
    venn3([genes_sets[a], genes_sets[b], genes_sets[c]], (a,b,c))
    plt.title("Gene overlap (Venn) for first 3 models")
    plt.savefig("venn_genes_3models.png", dpi=300)
    plt.close()

# pathway-level overlap heatmap (requires reaction->subsystem mapping; best if models have 'subsystem' annotations)
# build subsystem overlap matrix if present
subsystems = set()
sub_map = {n:defaultdict(set) for n in all_names}
for n,fname in models.items():
    model = cobra.io.read_sbml_model(fname) if fname.endswith(".xml") else cobra.io.load_json_model(fname)
    for r in model.reactions:
        subs = getattr(r, "subsystem", None) or getattr(r, "subSystems", None) or ""
        if isinstance(subs, list):
            s = subs[0] if subs else ""
        else:
            s = subs
        if s:
            subs = [x.strip() for x in s.split(";")] if ";" in s else [s]
            for ss in subs:
                sub_map[n][ss].add(reaction_signature(r))
                subsystems.add(ss)

sub_list = sorted(list(subsystems))
mat = pd.DataFrame(index=sub_list, columns=all_names)
for ss in sub_list:
    for n in all_names:
        mat.loc[ss,n] = len(sub_map[n].get(ss, set()))
mat = mat.fillna(0).astype(int)
# convert to fraction relative to max or common baseline if desired
sns.clustermap(mat, standard_scale=1, figsize=(8,10))
plt.savefig("subsystem_overlap_heatmap.png", dpi=300)
