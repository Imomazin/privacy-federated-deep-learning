# RAPSA-FL Research Continuity File Index

Read these files before freezing the new paper's architecture or novelty claim.

1. `README.md`
   - research-programme map
   - novelty boundary
   - execution datasets
   - integrity rules

2. `Paper-I_part-01.md`
3. `Paper-I_part-02.md`
   - Paper I reading copy
   - optimiser comparison, non-IID benchmark, DP, membership inference, gradient reconstruction and temporal privacy-risk findings

4. `Paper-II_Research-Continuity.md`
   - Paper II definitive protocol, experiment matrix, main numerical results, limitations and follow-up boundary

5. `Paper-III_Privacy-Resilient-Federated-Credit-Risk.md`
   - Paper III reading copy
   - record/example-level DP, privacy accounting, membership inference, clipping-only baseline and front-loaded privacy schedule

6. `Survey_Research-Continuity.md`
   - technical survey continuity copy
   - five-dimensional taxonomy, threat model, DP/secure-aggregation distinction, non-IID taxonomy and open gaps

7. `acquire-datasets.py`
   - acquisition and integrity checks for Taiwan Credit Default, MNIST and Fashion-MNIST

The repository root contains the existing federated-learning source code and prior experimental assets.

## Required reading order

`README.md` -> Paper I -> Paper II -> Paper III -> Survey -> repository code.

## Novelty warning

Paper III already tests a front-loaded privacy schedule. Paper I already establishes higher early-round privacy exposure. Do not position front-loaded DP alone as a new contribution.

The new work must be evaluated as a systems architecture, including secure aggregation, privacy-risk-driven orchestration and resilience compatible with hidden client updates.
