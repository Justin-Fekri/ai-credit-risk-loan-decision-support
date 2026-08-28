# Model Card — Credit Risk Decision Support

## Intended use

Estimate the probability that a consumer loan applicant will default, and present
that estimate — with its reasoning — to a **human underwriter** who makes the final
decision. Built as a portfolio/educational project.

**In scope:** ranking applications by risk, triaging a review queue, explaining an
individual score, auditing outcomes across borrower segments.

**Out of scope:**
- Automated approval or denial without human review
- Real lending decisions (the default model is trained on synthetic data)
- Pricing, collections, or any use beyond default-risk estimation
- Producing a legally sufficient adverse-action notice

## Model

- **Type:** gradient-boosted trees (XGBoost), selected by 5-fold CV against random
  forest and logistic regression; all three scored within one standard deviation.
- **Calibration:** isotonic regression, so an output of 0.30 means roughly a 30%
  observed default rate. Scores are clipped to [0.005, 0.995].
- **Threshold:** chosen to minimise expected cost at a 5:1 false-negative to
  false-positive ratio, not to maximise accuracy.

## Training data

12,000 synthetic applications from a seeded generator with a documented risk
structure: default rises with debt-to-income, utilisation and delinquencies, and
falls with credit score, income and employment length, plus non-linear effects (a
subprime cliff below 620, a thin-file penalty, a U-shaped age effect) and irreducible
noise. Default rate 17.3%. **This is not real lending data.**

The pipeline also runs on the UCI Statlog German Credit dataset (ROC-AUC 0.802).

## Features

Twelve numeric and four categorical borrower attributes (credit score, DTI,
utilisation, delinquencies, credit history length, income, employment, loan terms,
home ownership, loan purpose, employment status, education).

**Not used as inputs:** race, sex, national origin, marital status, or any other
protected class. `age` *is* an input, which is discussed below.

## Performance

| Metric | Held-out test (n=2,400) |
|---|---|
| ROC-AUC | 0.796 |
| PR-AUC | 0.492 |
| Brier score | 0.114 |
| Recall / Precision | 0.725 / 0.368 |

Decision bands, by actual default rate: approve 8.5% · review 33.1% · decline 64.4%.

## Fairness findings

Audited on age group, employment status, and home ownership using the four-fifths
(80%) rule.

**Flagged:** applicants under 30 are approved at a disparate impact ratio of **0.34**;
the 30–44 group at 0.79. Part-time and unemployed applicants sit at 0.76.

This is the most important limitation in this card. Under ECOA/Regulation B, age is a
prohibited basis for most credit decisions. Two things are true at once here:

1. The disparity partly reflects genuine risk in the data — the flagged segments do
   default more often.
2. The model reaches age through proxies (credit-history length, employment length)
   even when age is removed, so dropping the column would not fix it.

A production deployment would require legal review, a documented business-necessity
justification, and testing of less-discriminatory alternatives. This system
**reports** the disparity for human review; it does not correct it automatically, and
it does not certify compliance.

## Limitations

- Synthetic training data by default; real-world performance is unknown.
- No out-of-time validation — no protection against concept drift.
- No reject inference: real credit data only observes funded loans.
- Explanations are SHAP attributions — a faithful account of *the model's* reasoning,
  not proof of real-world causation.
- Calibration holds only while applicants resemble the training distribution.

## Human oversight

Every prediction carries a decision-support disclaimer. The three-band output is
designed so the middle band is explicitly routed to a person, and even the outer
bands are framed as recommendations for a reviewer to confirm. No code path in this
repository approves or declines a loan.

## Maintenance

Model bundles store training metadata (row count, seed, class balance, timestamp,
Python version) so any score can be traced to the model that produced it. Retraining
should trigger a re-run of `evaluate` and a review of the fairness report.
