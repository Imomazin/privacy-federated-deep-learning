# When Is Federated Learning Both Private and Reliable?

## A Critical Survey of Optimisation, Operational Stress and Time-Dependent Information Leakage

### Central thesis
A credible federated learning system must be evaluated across its statistical regime, optimiser, operational environment, adversarial observability, formal privacy mechanism and decision-level utility. Keeping raw records local is not a sufficient privacy guarantee; a competitive final-model AUC is not sufficient evidence of calibrated or reliable performance.

### Original synthesis
The manuscript links six dimensions which are frequently reported separately:

1. **Statistical heterogeneity:** label, feature, quantity, concept and temporal heterogeneity; partition construction and severity.
2. **Optimisation and training phase:** FedAvg, FedProx, SCAFFOLD, FedAdam, client drift, early versus late updates.
3. **Operational stress:** label corruption, class imbalance, intermittent clients and combined disturbances.
4. **Threat model:** membership, gradient inversion, property inference; released model versus checkpoints versus individual client updates.
5. **Formal and cryptographic protection:** record-level versus client-level DP; noise versus clipping; privacy accounting; secure aggregation.
6. **Decision-level evidence:** discrimination, calibration, empirical leakage, formal epsilon/delta, fairness, overhead and reproducibility.

### Review questions
- RQ1: How do the source and degree of heterogeneity affect optimiser choice, time-dependent attack exposure and predictive stability?
- RQ2: How do corrupted labels, imbalance, intermittent participation and compound stress change privacy and utility?
- RQ3: How do formal privacy bounds relate to empirical attack risk under different attacker-access assumptions?
- RQ4: What is the minimum reporting and experimental standard for comparable FL results?
- RQ5: What future work remains after the newest optimiser, credit-risk resilience and differential privacy experiments?

### Evidence-synthesis protocol
Adopt PRISMA 2020 reporting principles where appropriate (Page et al., BMJ 2021, doi:10.1136/bmj.n71). Search IEEE Xplore, ACM Digital Library, Scopus and Web of Science, supplemented by SpringerLink, ScienceDirect and citation chaining. Maintain a complete dated log with queries, inclusion/exclusion reasons, exact deduplication counts and the final extraction dataset. No PRISMA flow numbers are asserted without a recorded screening exercise.

A candidate query:
```
("federated learning" OR "federated optimization")
AND ("non-IID" OR heterogeneity OR "client drift" OR "partial participation"
     OR "label noise" OR robustness OR resilience)
AND (privacy OR "differential privacy" OR "membership inference"
     OR "gradient inversion" OR "secure aggregation" OR calibration)
```

Minimum extraction fields: source, year, venue, task, dataset, model, number of clients, heterogeneity mechanism and severity, client scheduling, algorithm, privacy mechanism and protected unit, epsilon/delta and accountant assumptions, attacker access, membership/reconstruction metric, utility and calibration metrics, repeated seeds, uncertainty intervals, security guarantees, overhead, code availability and limitations.

### Research-programme evidence map

**Paper One, published.** Benchmarking Optimiser Choice and Privacy-Risk Dynamics in Non-IID Federated Learning. IEEE BigDataService 2026, pp. 9–23. DOI: 10.1109/BIGDATASERVICE70481.2026.00011. Investigates optimiser effects and temporal privacy exposure in MNIST/CIFAR-10 benchmarks.

**Paper Two, published.** Resilience of Federated Learning for Credit Risk Forecasting Under Heterogeneity, Label Noise and Partial Participation. IEEE Access 2026. DOI: 10.1109/ACCESS.2026.3743719. Uses simulated credit-risk federation to examine operational reliability under heterogeneity, corruption and dropout, without formal DP or secure aggregation.

**Paper Three, final submitted manuscript.** Privacy-Resilient Federated Credit-Risk Forecasting: Differential Privacy, Membership-Inference Risk and Utility Loss under Operational Stress. Uses five simulated clients, one public credit dataset, six stress conditions, six privacy treatments and 100 main seeds. Formal example-level DP, empirical membership inference, calibration and clipping/noise attribution. See `Paper Three Final Submission.md`. Publisher acceptance and DOI are not confirmed.

The survey's future-research agenda must distinguish the achievements of Papers One and Two and the submitted findings of Paper Three from work that remains open. Do not describe their existing contributions as novel untested future research. Cite them with their actual status and preserve their bounded claims.

### Literature that any competitive survey must engage

- Kairouz et al., *Advances and Open Problems in Federated Learning*, 2021.
- Li et al., *Federated Learning: Challenges, Methods and Future Directions*, 2020.
- Uddin et al., *A Systematic Literature Review of Robust Federated Learning*, ACM Computing Surveys 2025, DOI 10.1145/3727643.
- Bai et al., *Membership Inference Attacks and Defenses in Federated Learning: A Survey*, ACM Computing Surveys 2025, DOI 10.1145/3704633.
- Zhao et al., *The Federation Strikes Back*, ACM Computing Surveys 2025, DOI 10.1145/3724113.
- Sen et al., *Recent Advances in Privacy-Preserving Federated Learning for Financial Applications*, Discover Artificial Intelligence 2026, DOI 10.1007/s44163-026-01853-5.
- Rafi et al., *Fairness and Privacy Preserving in Federated Learning*, Information Fusion 2024, DOI 10.1016/j.inffus.2023.102198.

These existing surveys preclude unsupported claims that no broad review of FL privacy, robustness or financial applications exists. This survey's distinctive value must be defended at the interaction level and substantiated by the evidence matrix.

### Minimum comparative reporting matrix
| Evidence layer | Required information |
| --- | --- |
| Predictive utility | Absolute ROC-AUC, paired change, confidence interval |
| Reliability | Calibration slope, intercept, Brier/ECE and subgroup diagnostics where relevant |
| Formal privacy | Unit protected, epsilon/delta, accountant, sampling, schedule |
| Empirical privacy | Attacker type/access, metric, member-set integrity, attack strength |
| Robustness | Clean, single-stressor and compound-stressor comparisons |
| Systems | Secure aggregation and dropout properties, executed overhead and latency |
| Reproducibility | Datasets, splits, seeds, configs, code, raw outputs and figure traceability |

### Future research directions
1. Higher-capacity models and multiple externally independent datasets.
2. Likelihood-ratio and other stronger membership attacks and realistic update-transcript adversaries.
3. Differentially private recalibration, with explicit accounting for any protected validation data.
4. Dynamic privacy governance under fixed total composition, without claiming front-loaded noise scheduling itself is novel.
5. Secure-aggregation-compatible robustness and malicious-client defence.
6. Cross-silo trials with real institutional data holders and independently measured operational overhead.
7. Standardised reporting of attacker access, training-phase risk, calibration, subgroup impacts and uncertainty.

### Manuscript control
The detailed initial manuscript is maintained as a 12-page Word document titled *Federated Learning Survey Manuscript*. It contains the academic abstract, thematic literature synthesis, cross-layer taxonomy, empirical study mapping, protocol, comparison tables, technical research agenda and 27 seeded scholarly references. Its literature corpus must be systematically screened and checked before a systematic-review claim is made. The authorship list and target journal remain to be agreed by the research team.
