import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score
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

def aps_scores(proba, y, classes):
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    true_idx = np.array([classes.index(v) for v in y])
    rank = np.array([np.where(order[i]==true_idx[i])[0][0] for i in range(len(y))])
    return np.array([cum[i, rank[i]] for i in range(len(y))])

def aps_set(proba, q, classes):
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    include = cum <= q
    include[:,0] = True
    inset = np.zeros_like(proba, dtype=bool)
    for i in range(proba.shape[0]):
        for r in range(proba.shape[1]):
            if include[i,r]:
                inset[i, order[i,r]] = True
    return inset

df = dedupe(pd.read_csv('data/tp53_pubchem_raw.csv'))
print('TP53 PubChem deduped n=', len(df), df['label'].value_counts().to_dict())
df.to_csv('data/tp53_pubchem_clean.csv', index=False)

X = fp(df['clean_smiles'].tolist()); y = df['label'].values.astype(int)
alpha=0.10
out=[]
for seed in range(8):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y)); n=len(y); n_tr=int(.5*n); n_cal=int(.25*n)
    tr,cal,te = idx[:n_tr], idx[n_tr:n_tr+n_cal], idx[n_tr+n_cal:]
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    proba_all = clf.predict_proba(X); classes=list(clf.classes_)
    y_te = y[te]
    maj = 0 if (y_te==0).sum()>(y_te==1).sum() else 1
    minl = 1-maj
    s_cal = 1 - proba_all[cal, [classes.index(v) for v in y[cal]]]
    q_marg = lac_q(s_cal, alpha)
    p1 = proba_all[te, classes.index(1)]; p0 = proba_all[te, classes.index(0)]
    cov_marg = np.where(y_te==1, p1>=(1-q_marg), p0>=(1-q_marg))
    setsz_marg = (p0>=(1-q_marg)).astype(int)+(p1>=(1-q_marg)).astype(int)
    s0 = 1-proba_all[cal[y[cal]==0]][:,classes.index(0)]
    s1 = 1-proba_all[cal[y[cal]==1]][:,classes.index(1)]
    q0,q1 = lac_q(s0,alpha), lac_q(s1,alpha)
    cov_mond = np.where(y_te==1, p1>=(1-q1), p0>=(1-q0))
    setsz_mond = (p0>=(1-q0)).astype(int)+(p1>=(1-q1)).astype(int)
    s_cal_aps = aps_scores(proba_all[cal], y[cal], classes)
    q_aps = lac_q(s_cal_aps, alpha)
    inset_aps = aps_set(proba_all[te], q_aps, classes)
    cov_aps = np.array([inset_aps[i, classes.index(y_te[i])] for i in range(len(y_te))])
    setsz_aps = inset_aps.sum(axis=1)
    pred = clf.predict(X[te])
    out.append(dict(seed=seed,
        mcc=matthews_corrcoef(y_te,pred), auprc=average_precision_score(y_te,p1), auroc=roc_auc_score(y_te,p1),
        cov_min_marg_lac=cov_marg[y_te==minl].mean(), cov_maj_marg_lac=cov_marg[y_te==maj].mean(),
        cov_min_mond_lac=cov_mond[y_te==minl].mean(), cov_maj_mond_lac=cov_mond[y_te==maj].mean(),
        setsz_marg_lac=setsz_marg.mean(), setsz_mond_lac=setsz_mond.mean(),
        cov_min_marg_aps=cov_aps[y_te==minl].mean(), setsz_marg_aps=setsz_aps.mean(),
        minority_rate=(y_te==minl).mean()))
r = pd.DataFrame(out)
r.to_csv('results/TP53_PubChem_full.csv', index=False)
print(r.mean(numeric_only=True).round(4))
