# phpp_analysis.py
import cobra, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from cobra.flux_analysis import flux_variability_analysis
from cobra.util.solver import linear_reaction_coefficients
import os

MODEL_FILE = "iLD1283.xml"  # replace with your model file
model = cobra.io.read_sbml_model(MODEL_FILE)

# IDs you may need to change to match your model naming
SAM_exchange_id = 'EX_sam_e'    # exchange or sink reaction for SAM
BIOMASS_rxn = 'BIOMASS_REACTION' # replace
ATP_synth_id = 'ATPSynthase'    # ATP synthase reaction id
O2_ex_id = 'EX_o2_e'            # oxygen exchange rxn (uptake negative)
CIT_in_ex = 'EX_triCi_e'        # citrate exchange (example id)
# define other reaction ids as needed (IDH, ACS, FA_synth etc.)

# function to set citrate uptake bound and compute metrics
def evaluate_at_citrate(uptake):
    m = model.copy()
    # set citrate uptake (negative for uptake in cobrapy)
    m.reactions.get_by_id(CIT_in_ex).lower_bound = -uptake
    # optionally set other feed constraints to measured values etc.
    # 1) maximize SAM
    # if you have an explicit SAM secretion or demand reaction:
    if SAM_exchange_id in [r.id for r in m.reactions]:
        m.objective = SAM_exchange_id
    else:
        # alternative: set SAM synthetic reaction flux if internal reaction id known
        pass
    sol_sam = m.optimize()
    sam_flux = sol_sam.objective_value
    # record biomass when maximizing biomass:
    m2 = m.copy()
    m2.objective = BIOMASS_rxn
    sol_bio = m2.optimize()
    bio_flux_when_opt_bio = sol_bio.objective_value
    # now compute ATP synthase flux under SAM optimization (if exists)
    atp_flux = 0
    try:
        atp_flux = sol_sam.fluxes.get(ATP_synth_id, 0.0)
    except:
        atp_flux = 0.0
    # total NADH producing flux: sum of fluxes of NADH-producing reactions (you need list)
    # For general approach, you can define list of NADH rxn ids
    NADH_rxns = ['IDH', 'MDH', 'GAPDH']  # replace with actual ids
    nadhtot = sum([sol_sam.fluxes.get(r,0.0) for r in NADH_rxns])
    # citrate secretion / accumulation
    cit_ex = sol_sam.fluxes.get(CIT_in_ex, 0.0)  # negative if uptake
    # run FVA for a small set of reactions of interest
    ROI = [SAM_exchange_id, BIOMASS_rxn, ATP_synth_id] + NADH_rxns
    fva = flux_variability_analysis(m, ROI, fraction_of_optimum=0.99)
    return {'citrate_uptake':uptake, 'sam_flux':sam_flux, 'bio_flux_biomax':bio_flux_when_opt_bio,
            'atp_flux':atp_flux, 'nadhtot':nadhtot, 'cit_exchange': cit_ex, 'fva': fva}

# grid scan
uptakes = np.linspace(0,0.5,26)  # adjust bounds and resolution as needed
rows = []
fva_dict = {}
for u in uptakes:
    print("Evaluating uptake", u)
    res = evaluate_at_citrate(u)
    rows.append({'citrate_uptake':res['citrate_uptake'],
                 'sam_flux':res['sam_flux'],
                 'bio_flux_biomax':res['bio_flux_biomax'],
                 'atp_flux':res['atp_flux'],
                 'nadhtot':res['nadhtot'],
                 'cit_exchange':res['cit_exchange']})
    fva_dict[u] = res['fva']

df = pd.DataFrame(rows)
df.to_csv("phpp_scan_results.csv", index=False)

# plotting
plt.figure(figsize=(8,6))
plt.plot(df['citrate_uptake'], df['sam_flux'], label='SAM flux')
plt.plot(df['citrate_uptake'], df['bio_flux_biomax']/df['bio_flux_biomax'].max(), label='Biomass (norm)')
plt.plot(df['citrate_uptake'], df['atp_flux'], label='ATP synthase flux')
plt.xlabel('Citrate uptake (mmol gDW^-1 h^-1)')
plt.ylabel('Flux (units)')
plt.legend()
plt.title('PhPP decomposition: SAM, biomass, ATP vs citrate uptake')
plt.savefig('phpp_decompose.png', dpi=300)

# O2 sensitivity: choose three citrate levels and sweep O2 bounds
o2_bounds = [-5,-10,-20,-40,-80]
citrate_levels = [0.05, 0.15, 0.3]  # low, optimal, excessive example
heat = []
for c in citrate_levels:
    for o in o2_bounds:
        m = model.copy()
        m.reactions.get_by_id(CIT_in_ex).lower_bound = -c
        m.reactions.get_by_id(O2_ex_id).lower_bound = o
        # maximize SAM
        if SAM_exchange_id in [r.id for r in m.reactions]:
            m.objective = SAM_exchange_id
        sol = m.optimize()
        heat.append({'citrate':c,'o2':o,'sam':sol.objective_value})
heat_df = pd.DataFrame(heat)
heat_pivot = heat_df.pivot(index='o2', columns='citrate', values='sam')
sns.heatmap(heat_pivot, annot=True)
plt.title('SAM vs O2 and citrate')
plt.savefig('phpp_o2_sensitivity.png', dpi=300)

# Shadow prices (requires solver producing duals)
# maximize SAM and inspect solution
m = model.copy()
m.reactions.get_by_id(CIT_in_ex).lower_bound = -0.3  # example high citrate
m.objective = SAM_exchange_id
sol = m.optimize()
# some solvers expose shadow_prices in solution
try:
    shadows = sol.shadow_prices
    pd.Series(shadows).sort_values(ascending=False).head(20).to_csv('shadow_prices_top20.csv')
except Exception as e:
    print("Shadow price extraction failed:",e)
