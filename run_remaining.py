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
    df = df.drop_duplicates('clean_smiles')
    return df.reset_index(drop=True)

def fp(smiles_list):
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    X = np.zeros((len(smiles_list),2048), dtype=np.uint8)
    for i,s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        X[i] = gen.GetFingerprintAsNumPy(m)
    return X

def lac_threshold(scores, alpha):
    n = len(scores)
    q = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    return np.quantile(scores, q, method='higher')

def run_dataset(df, alpha=0.10, n_seeds=3, class_weight=None, n_est=100):
    X = fp(df['clean_smiles'].tolist())
    y = df['label'].values.astype(int)
    results = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(y))
        n = len(y); n_tr=int(0.5*n); n_cal=int(0.25*n)
        tr, cal, te = idx[:n_tr], idx[n_tr:n_tr+n_cal], idx[n_tr+n_cal:]
        clf = RandomForestClassifier(n_estimators=n_est, random_state=seed, n_jobs=-1, class_weight=class_weight)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X)
        classes = list(clf.classes_)
        p_true_cal = proba[cal, [classes.index(v) for v in y[cal]]]
        s_cal = 1 - p_true_cal
        q_marg = lac_threshold(s_cal, alpha)
        p_te = proba[te]; p1 = p_te[:, classes.index(1)]; p0 = p_te[:, classes.index(0)]
        in_set1_marg = p1 >= (1-q_marg); in_set0_marg = p0 >= (1-q_marg)
        setsize_marg = in_set0_marg.astype(int)+in_set1_marg.astype(int)
        y_te = y[te]
        cov_marg = np.where(y_te==1, in_set1_marg, in_set0_marg)
        s_cal0 = 1 - proba[cal[y[cal]==0]][:, classes.index(0)]
        s_cal1 = 1 - proba[cal[y[cal]==1]][:, classes.index(1)]
        q0 = lac_threshold(s_cal0, alpha) if len(s_cal0)>0 else q_marg
        q1 = lac_threshold(s_cal1, alpha) if len(s_cal1)>0 else q_marg
        in_set0_mond = p0 >= (1-q0); in_set1_mond = p1 >= (1-q1)
        setsize_mond = in_set0_mond.astype(int)+in_set1_mond.astype(int)
        cov_mond = np.where(y_te==1, in_set1_mond, in_set0_mond)
        pred = clf.predict(X[te])
        mcc = matthews_corrcoef(y_te, pred); auprc = average_precision_score(y_te, p1); auroc = roc_auc_score(y_te, p1)
        maj_label = 0 if (y_te==0).sum() > (y_te==1).sum() else 1
        min_label = 1-maj_label
        results.append(dict(seed=seed, mcc=mcc, auprc=auprc, auroc=auroc,
            cov_min_marg=cov_marg[y_te==min_label].mean(), cov_maj_marg=cov_marg[y_te==maj_label].mean(),
            cov_min_mond=cov_mond[y_te==min_label].mean(), cov_maj_mond=cov_mond[y_te==maj_label].mean(),
            setsize_marg=setsize_marg.mean(), setsize_mond=setsize_mond.mean()))
    return pd.DataFrame(results)

datasets = {'Tox21_SRARE':'data/tox21_clean.csv', 'ClinTox':'data/clintox_clean.csv'}
for name, path in datasets.items():
    df = dedupe(pd.read_csv(path))
    print(name, 'n=', len(df))
    ru = run_dataset(df, n_seeds=3, class_weight=None, n_est=100)
    rw = run_dataset(df, n_seeds=3, class_weight='balanced', n_est=100)
    ru.to_csv(f'results/{name}_unweighted.csv', index=False)
    rw.to_csv(f'results/{name}_weighted.csv', index=False)
    print(ru.mean(numeric_only=True).to_dict())
    print(rw.mean(numeric_only=True).to_dict())
print("DONE")
