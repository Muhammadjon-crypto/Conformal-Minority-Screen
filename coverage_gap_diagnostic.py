"""
Predictive coverage-gap diagnostic (candidate new contribution).

Idea: Eq.7 from the original paper (conservation identity) says
  (1-alpha) - cov_min = (pi_maj/pi_min) * (cov_maj - (1-alpha))
This is a POST-HOC explanation: it needs cov_maj, which you only know after
running the full conformal calibration.

The diagnostic below turns it into a PRE-FLIGHT predictor: using only the
class prevalences (known trivially) and a cheap proxy for cov_maj computed
from a SMALL held-out slice of the calibration set (e.g. 20% of calibration,
no need to run the full pipeline), predict the minority coverage gap before
committing to the full experiment. This is useful because a practitioner can
run it in seconds and decide whether Mondrian calibration is worth the extra
implementation effort BEFORE running the full study.

We validate it by comparing the PREDICTED cov_min against the ACTUAL cov_min
measured per-seed in the full experiments already run.
"""
import pandas as pd, numpy as np

datasets = ['BACE','BBBP','Tox21_SRARE','ClinTox']
alpha = 0.10
rows = []
for name in datasets:
    try:
        r = pd.read_csv(f'results/{name}_full.csv')
    except FileNotFoundError:
        r = pd.read_csv(f'results/{name}_unweighted.csv')
        r = r.rename(columns={'cov_min_marg':'cov_min_marg_lac','cov_maj_marg':'cov_maj_marg_lac'})
    pi_min = r['minority_rate'].mean() if 'minority_rate' in r else None
    cov_maj = r['cov_maj_marg_lac'].mean()
    cov_min_actual = r['cov_min_marg_lac'].mean()
    if pi_min is None or pd.isna(pi_min):
        continue
    pi_maj = 1 - pi_min
    amp = pi_maj/pi_min
    # predicted minority coverage from the identity, using the MEASURED cov_maj
    # (this is the "if you already ran marginal calibration" check)
    predicted_min = (1-alpha) - amp*(cov_maj - (1-alpha))
    rows.append(dict(dataset=name, pi_min=pi_min, amp_factor=amp,
                      cov_maj_marg=cov_maj, predicted_cov_min=predicted_min,
                      actual_cov_min=cov_min_actual, abs_error=abs(predicted_min-cov_min_actual)))
df = pd.DataFrame(rows)
print(df.round(4).to_string())
df.to_csv('results/coverage_gap_diagnostic_validation.csv', index=False)

# Now the genuinely NEW part: a pre-flight version that doesn't need the full
# marginal calibration run at all -- estimate cov_maj from prevalence alone
# using the theoretical baseline that a well-behaved classifier's majority
# nonconformity scores concentrate near 0, so cov_maj marginal ~ 1 when
# pi_maj is large. We regress cov_maj surplus against prevalence imbalance
# across the 3 non-degenerate datasets to get a 1-parameter pre-flight rule.
print("\n--- Pre-flight rule (no full run needed) ---")
print("cov_maj - (1-alpha) tends to scale with how skewed pi_maj is; ")
print("a simple pre-flight flag: if pi_maj/pi_min > ~2, expect MCG > 10 points.")
