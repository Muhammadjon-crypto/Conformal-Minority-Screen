"""
Selective prediction + redesigned utility function.
Cost structure per dataset per Reviewer 2's directionality point:
 - BACE: class 1 = active inhibitor = the GOAL. Missing it (false negative) is the costly error.
 - Tox21 SR-ARE, ClinTox: class 1 = toxic = the thing to AVOID advancing. Missing it (false
   negative -> toxic compound waved through as safe) is the costly error.
 - BBBP: goal-dependent, report symmetric as before with a flag.
In all cases "class 1 you can't afford to miss" -> asymmetric costs C_fn > C_fp on THAT class.
"""
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.filterwarnings('ignore')

def dedupe(df):
    g = df.groupby('clean_smiles')['label'].nunique()
    conflict = g[g>1].index
    df = df[~df['clean_smiles'].isin(conflict)]
    return df.drop_duplicates('clean_smiles').reset_index(drop=True)

def fp(smiles_list):
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    X = np.zeros((len(smiles_list),2048), dtype=np.uint8)
    for i,s in enumerate(smiles_list):
        X[i] = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
    return X

def lac_q(scores, alpha):
    n=len(scores); return np.quantile(scores, min(np.ceil((n+1)*(1-alpha))/n,1.0), method='higher')

def utility(n_sent, n_tp, n_fp, n_fn, Btp=1, Cfp=5, Cfn=5, Clab=0.5):
    return Btp*n_tp - Cfp*n_fp - Cfn*n_fn - Clab*n_sent

configs = {
    'BACE': ('data/bace_clean.csv', 6, 200, 'find_class1'),
    'BBBP': ('data/bbbp_clean.csv', 6, 200, 'symmetric'),
    'Tox21_SRARE': ('data/tox21_clean.csv', 4, 100, 'avoid_class1'),
    'ClinTox': ('data/clintox_clean.csv', 6, 200, 'avoid_class1'),
}
alpha=0.10
rows=[]
for name,(path,seeds,trees,goal) in configs.items():
    df = dedupe(pd.read_csv(path))
    X = fp(df['clean_smiles'].tolist()); y = df['label'].values.astype(int)
    for seed in range(seeds):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(y)); n=len(y); n_tr=int(.5*n); n_cal=int(.25*n)
        tr,cal,te = idx[:n_tr], idx[n_tr:n_tr+n_cal], idx[n_tr+n_cal:]
        clf = RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X); classes=list(clf.classes_)
        y_te = y[te]
        p1 = proba[te, classes.index(1)]; p0 = proba[te, classes.index(0)]
        s_cal = 1 - proba[cal, [classes.index(v) for v in y[cal]]]
        q_marg = lac_q(s_cal, alpha)
        s0 = 1-proba[cal[y[cal]==0]][:,classes.index(0)]; s1 = 1-proba[cal[y[cal]==1]][:,classes.index(1)]
        q0,q1 = lac_q(s0,alpha), lac_q(s1,alpha)
        # identify the MINORITY label in this test split -- utility is scored on
        # decisions about ground-truth-minority compounds ONLY, matching the
        # original paper's "1000 minority decisions" framing (Sec 3.5)
        crit = 1 if (y_te==1).sum() < (y_te==0).sum() else 0
        mask_min = (y_te == crit)
        n_min = mask_min.sum()
        for scheme, in0, in1 in [
            ('marginal', p0>=(1-q_marg), p1>=(1-q_marg)),
            ('mondrian', p0>=(1-q0), p1>=(1-q1)),
        ]:
            in_crit = (in1 if crit==1 else in0)
            in_other = (in0 if crit==1 else in1)
            singleton_crit = in_crit & (~in_other)
            singleton_other = in_other & (~in_crit)
            abstain = in0 & in1
            # restrict to minority ground-truth compounds
            n_sent = (abstain & mask_min).sum()
            n_tp = (singleton_crit & mask_min).sum()    # correctly confident on a minority compound
            n_fn = (singleton_other & mask_min).sum()   # confidently wrong -> missed a minority compound
            n_fp = 0  # not defined within minority-only decisions; kept 0 per original framing
            # scale to 1000 minority decisions
            scale = 1000.0 / max(n_min, 1)
            n_sent_s, n_tp_s, n_fn_s = n_sent*scale, n_tp*scale, n_fn*scale
            if goal in ('find_class1','avoid_class1'):
                U = utility(n_sent_s, n_tp_s, n_fp, n_fn_s, Btp=1, Cfp=5, Cfn=15, Clab=0.5)  # Cfn >> Cfp
                U_sym = utility(n_sent_s, n_tp_s, n_fp, n_fn_s, Btp=1, Cfp=5, Cfn=5, Clab=0.5)
            else:
                U = utility(n_sent_s, n_tp_s, n_fp, n_fn_s, Btp=1, Cfp=5, Cfn=5, Clab=0.5)
                U_sym = U
            rows.append(dict(dataset=name, seed=seed, scheme=scheme, goal=goal,
                              n_min=n_min, n_sent=n_sent, tp=n_tp, fn=n_fn,
                              accept_rate=1-n_sent/max(n_min,1), U_asymmetric=U, U_symmetric=U_sym))
    print(name,'done')

res = pd.DataFrame(rows)
res.to_csv('results/utility_selective.csv', index=False)
summ = res.groupby(['dataset','scheme','goal'])[['accept_rate','U_asymmetric','U_symmetric']].mean().reset_index()
print(summ.round(2).to_string())
summ.to_csv('results/utility_summary.csv', index=False)
