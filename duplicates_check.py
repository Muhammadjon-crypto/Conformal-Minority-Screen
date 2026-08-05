import pandas as pd
for name, path in [('BBBP','data/bbbp_clean.csv'),('BACE','data/bace_clean.csv'),
                    ('Tox21 SR-ARE','data/tox21_clean.csv'),('ClinTox','data/clintox_clean.csv')]:
    df = pd.read_csv(path)
    n = len(df)
    dup_smiles = df['clean_smiles'].duplicated().sum()
    # check label consistency among duplicates
    dupe_groups = df[df.duplicated('clean_smiles', keep=False)].groupby('clean_smiles')['label'].nunique()
    conflicting = (dupe_groups > 1).sum()
    print(f"{name}: n={n}, duplicated canonical SMILES={dup_smiles} ({dup_smiles/n:.1%}), label-conflicting duplicate groups={conflicting}")
