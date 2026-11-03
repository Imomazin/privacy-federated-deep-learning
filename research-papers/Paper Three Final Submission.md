# Privacy-Resilient Federated Credit-Risk Forecasting: Differential Privacy, Membership-Inference Risk and Utility Loss under Operational Stress

Imo J. Enang; Joffrey L. Leevy; Preston Billion-Polak; Taghi M. Khoshgoftaar

**Status:** Final author-supplied submission version, compiled 28 October 2026. Not yet verified as accepted or published. Do not cite the IEEE Access typesetting placeholder DOI as a real DOI.

**Corresponding author:** Preston Billion-Polak  
**Venue indicated on submitted manuscript:** IEEE Access  
**Article source:** Author's final submitted PDF and complete LaTeX archive supplied 3 November 2026. The source files have not been uploaded into this repository by this Markdown record.

## Abstract

Federated learning trains a shared default model without pooling raw borrower data, yet its updates can still leak information about individual records. We measure what formal privacy costs and protects for a federated credit-scoring model under heterogeneity, label noise and partial participation. Using differentially private stochastic gradient descent, we cross six stress conditions with six privacy treatments, including a clipping-only baseline that separates bounded gradients from added noise. The data are the Taiwan Credit Card Default set (30,000 borrowers, 23 features, default rate about 22 per cent), modelled by logistic regression across five simulated clients over 100 seeds. At delta = 1e-5 a differentially private treatment at a moderate budget has a worst-case epsilon = 2.41 with moderate stress, rising to 4.04 when severe class imbalance is introduced. The added noise costs about 0.0016 in the Area Under the ROC Curve (AUC) on clean data, or 0.16 percentage points, and lowers the calibration slope from 0.92 to 0.47. In membership-inference tests, attack advantage remains near chance under clean conditions and becomes positive under severe class imbalance, reaching 0.043; combined with label noise, clean-label and label-agnostic attacks also recover a positive signal. We find that differentially private training yields a small utility benefit under stress, which the clipping-only baseline attributes to gradient clipping rather than added noise, and a front-loaded schedule gives a small gain at a matched budget. These findings are specific to the linear tabular setting and need not hold for higher-capacity models.

## Final submitted design

- Public benchmark: Taiwan Credit Card Default, 30,000 borrowers, 23 features.
- Model: federated logistic regression across five simulated clients.
- Evaluation: six operational stress conditions; six privacy treatments; 100 main seeds.
- Privacy: example-level DP-SGD, Poisson-subsampled RDP accountant, delta = 1e-5; worst-case epsilon reporting without claimed participation amplification.
- Comparison includes a non-private/unclipped baseline, a clipping-only baseline without a formal guarantee, uniform noise treatments, and an epsilon-matched front-loaded noise schedule.
- Adversaries: final-model membership inference and a selected-round checkpoint diagnostic; the attack scores are explicitly lower bounds.
- Evaluated outcomes: ROC AUC, calibration, membership-inference advantage, privacy budget and sensitivity to stress.

## Main results and interpretation

1. Under moderate conditions, the medium DP treatment reports epsilon 2.41 with an AUC change of -0.0016 against the non-private baseline.
2. Calibration slope on clean data falls from approximately 0.92 to 0.47 under the medium privacy treatment. Ranking and probability calibration should not be conflated.
3. A positive empirical membership-inference advantage of approximately 0.043 appears in the severe class-imbalance condition. Negative observed-label advantage under corrupted labels is not evidence of no leakage: clean-label and label-agnostic attacks recover positive signals under combined severe imbalance and noise.
4. Clipping alone explains most of the observed stability improvement under operational stress. This must not be attributed to Gaussian privacy noise.
5. Front-loaded privacy scheduling yields only a small matched-budget utility gain; no detectable improvement in final-model membership advantage is established.
6. The empirical scope is linear tabular credit scoring with one public dataset and simulated clients. These outcomes cannot be generalized to higher-capacity models or real cross-silo deployments.

## Follow-on research identified in the final manuscript

- Additional real-world credit datasets and genuine institutional cross-silo deployment.
- Stronger membership-inference attacks, including likelihood-ratio, shadow-model and label-only variants.
- Higher-capacity models such as gradient-boosted trees and shallow networks.
- Recalibration under DP with accounting for any use of private calibration records.
- More precise privacy amplification analysis where the sampling model supports it.
- Per-round gradients and client-update transcript adversaries, where technically observable.
- Extension of selected 30-seed diagnostics to the full experimental seed count.

## Relationship to Papers One and Two

**Paper One:** Benchmarking Optimiser Choice and Privacy-Risk Dynamics in Non-IID Federated Learning (IEEE BigDataService 2026; DOI 10.1109/BIGDATASERVICE70481.2026.00011). Establishes optimiser effects and temporal exposure in heterogeneous training.

**Paper Two:** Resilience of Federated Learning for Credit Risk Forecasting Under Heterogeneity, Label Noise and Partial Participation (IEEE Access 2026; DOI 10.1109/ACCESS.2026.3743719). Establishes empirical resilience of credit scoring under non-IID conditions, corrupted labels and partial participation.

**Paper Three:** Extends to example-level privacy accounting, empirical membership risk, calibration, clipping and noise decomposition.

## Source integrity

This is the current final-submission research record. The earlier `Paper-III_Privacy-Resilient-Federated-Credit-Risk.md` is an older June 2026 working copy and is not definitive for the final protocol, author list, abstract or findings. Do not present the final submission as a published paper. Publication details and any DOI must be obtained from the publisher's acceptance or publication notice.
