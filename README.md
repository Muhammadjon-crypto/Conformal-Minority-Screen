# Conformal Minority Screen

Code and data supporting "A Quiet Failure in Calibrated Virtual Screening: Marginal Conformal
Prediction Under-Covers the Minority Class, and a Class-Conditional Fix Recovers It."

## What this shows

Standard (marginal) conformal prediction on imbalanced molecular classification datasets meets its
global coverage target while leaving the minority class badly under-covered. Class-conditional
(Mondrian) calibration fixes it. We validate this across five datasets, three model architectures,
and show it survives class-weighted training and holds up against adaptive prediction sets (APS)
as an alternative conformal score.

## Repository structure

- `data/` — cleaned, deduplicated datasets (BACE, BBBP, Tox21 SR-ARE, ClinTox, and a PubChem
  bioactivity screen, AID 651631/TP53). Raw sources: MoleculeNet (BACE/BBBP/Tox21/ClinTox) and
  PubChem BioAssay (AID 651631). Not redistributed here in raw form — see Data Sources below.
- `src/` — all analysis scripts:
  - `prep_data.py` — cleans and canonicalizes the four MoleculeNet datasets
  - `duplicates_check.py` — duplicate/label-conflict audit
  - `run_core.py`, `full_pipeline.py` — core LAC marginal/Mondrian conformal experiments
  - `run_remaining.py` — Tox21/ClinTox reduced-cost variant
  - `run_tp53.py` — PubChem AID 651631 pipeline
  - `coverage_gap_diagnostic.py` — the pre-flight coverage-gap diagnostic and its validation
  - `utility_and_selective.py` — selective prediction and the direction-aware utility model
- `results/` — per-seed and summary CSVs for every table in the paper.

## Reproducing

```bash
pip install rdkit scikit-learn pandas numpy
python src/prep_data.py            # clean MoleculeNet data
python src/duplicates_check.py     # duplicate audit
python src/full_pipeline.py        # core LAC + Mondrian + APS experiments
python src/coverage_gap_diagnostic.py   # pre-flight diagnostic validation
python src/utility_and_selective.py     # direction-aware utility model
```

## Data sources

- MoleculeNet (BACE, BBBP, Tox21, ClinTox): https://moleculenet.org/
- PubChem BioAssay AID 651631: https://pubchem.ncbi.nlm.nih.gov/bioassay/651631

We do not redistribute raw upstream data beyond the cleaned/deduplicated CSVs needed to reproduce
our specific experiments; please cite the original sources above if you use this data.

## Citation

If you use this code or these results, please cite:

Tursunbadalov, M. and Tursunbadalov, M. "A Quiet Failure in Calibrated Virtual Screening: Marginal
Conformal Prediction Under-Covers the Minority Class, and a Class-Conditional Fix Recovers It."

## License

MIT — see LICENSE.
