import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def clean_smiles(smi):
    if not isinstance(smi, str) or smi.strip()=='':
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

# BBBP
bbbp = pd.read_csv('data/BBBP.csv')
bbbp = bbbp[['smiles','p_np']].rename(columns={'p_np':'label'})
bbbp['clean_smiles'] = bbbp['smiles'].apply(clean_smiles)
bbbp = bbbp.dropna(subset=['clean_smiles'])
print('BBBP', len(bbbp), bbbp['label'].value_counts().to_dict())

# BACE - has Class column, and smiles
bace = pd.read_csv('data/bace.csv', low_memory=False)
bace = bace[['smiles','Class']].rename(columns={'Class':'label'})
bace['clean_smiles'] = bace['smiles'].apply(clean_smiles)
bace = bace.dropna(subset=['clean_smiles'])
print('BACE', len(bace), bace['label'].value_counts().to_dict())

# Tox21 SR-ARE
tox21 = pd.read_csv('data/tox21.csv')
tox21 = tox21[['smiles','SR-ARE']].rename(columns={'SR-ARE':'label'})
tox21 = tox21.dropna(subset=['label'])
tox21['clean_smiles'] = tox21['smiles'].apply(clean_smiles)
tox21 = tox21.dropna(subset=['clean_smiles'])
print('Tox21 SR-ARE', len(tox21), tox21['label'].value_counts().to_dict())

# ClinTox CT_TOX
clintox = pd.read_csv('data/clintox.csv')
clintox = clintox[['smiles','CT_TOX']].rename(columns={'CT_TOX':'label'})
clintox['clean_smiles'] = clintox['smiles'].apply(clean_smiles)
clintox = clintox.dropna(subset=['clean_smiles'])
print('ClinTox CT_TOX', len(clintox), clintox['label'].value_counts().to_dict())

bbbp.to_csv('data/bbbp_clean.csv', index=False)
bace.to_csv('data/bace_clean.csv', index=False)
tox21.to_csv('data/tox21_clean.csv', index=False)
clintox.to_csv('data/clintox_clean.csv', index=False)
