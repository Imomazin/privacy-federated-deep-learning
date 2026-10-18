# Research Papers for RAPSA-FL

This folder is the research continuity pack for the new paper:

**Privacy-Preserving Federated Intelligence Ecosystems: A Differential Privacy and Secure Aggregation Architecture for Cross-Organisational Machine Learning**

Use the files in this folder together with the repository code before fixing the novelty claim, threat model or experiment matrix.

## Prior programme documents

1. **Paper I**: Communication-Efficient Learning with Differential Privacy: Empirical Privacy-Utility Analysis Under Non-IID Federated Conditions
   - FedAvg, FedProx, FedAdam, DP-FedAvg
   - MNIST and CIFAR-10
   - non-IID learning, membership inference and gradient reconstruction
   - key result for the new paper: privacy exposure is temporally concentrated and early rounds are higher risk

2. **Paper II**: Resilience of Federated Learning for Credit Risk Forecasting Under Heterogeneity, Label Noise and Partial Participation
   - Taiwan Credit Card Default dataset
   - horizontal FL, five simulated institutional clients
   - 100 seeds and 50 communication rounds
   - heterogeneity, label noise, partial participation and compound stress
   - does not implement secure aggregation or formal DP

3. **Paper III**: Privacy-Resilient Federated Credit-Risk Forecasting: Differential Privacy, Membership-Inference Risk and Utility Loss under Operational Stress
   - same credit-risk setting
   - record/example-level differential privacy
   - membership-inference testing
   - includes a front-loaded privacy schedule
   - medium setting reports worst-case epsilon 2.41 at delta 1e-5
   - important boundary: the new paper must not claim that front-loaded DP by itself is novel

4. **Technical Survey**: Privacy-Preserving Federated Learning Image Classification under Non-IID Data: A Technical Survey of Threats, Defences, Optimisation and Evaluation and Their Interaction
   - five-dimensional taxonomy covering threat model, privacy mechanism, heterogeneity, optimiser and evaluation
   - 41 primary studies in the scored synthesis
   - identifies the separation between privacy and heterogeneity literatures and the lack of temporal privacy analysis

## New-paper novelty boundary

Do not position the contribution as simply "DP + secure aggregation" or simply "adaptive/front-loaded DP".

The defensible contribution should be tested around the integrated architecture and its orchestration, including:

- participant/client-level protection where technically appropriate
- secure aggregation
- privacy-risk-driven scheduling based on information available during training
- robustness under hidden client updates
- explicit analysis of the tension between secure aggregation and Byzantine/robust aggregation
- matched evaluation of privacy, predictive utility, resilience and systems overhead

Paper III already establishes record-level DP and tests a front-loaded schedule. Paper I already establishes that early rounds exhibit higher privacy exposure.

## Execution datasets

Use the verified and computationally executable set:

- Taiwan Credit Card Default
- MNIST
- Fashion-MNIST

Do not force CIFAR-10 into the definitive matrix if this reduces the principal comparisons to an underpowered seed count.

## Code

The repository root contains the existing experimental codebase. Audit and extend it. Do not relabel prior outputs as RAPSA-FL results.

## Integrity rule

The chain for every new empirical claim is:

**dataset -> configuration -> executed run -> raw output -> statistics -> table/figure -> manuscript claim**

No placeholder results. No estimated results reported as measured. No prior-paper result presented as a new RAPSA-FL result.

## Reading copies

The manuscript reading copies in this folder are provided to make the full research programme accessible in one repository for novelty and configuration auditing. Where a reading copy was generated from a PDF or tracked-change Word file, use it for technical comparison and trace the source document named in its header when exact layout or tracked-change history matters.


## IEEE research (2026)

- [Benchmarking Optimiser Choice and Privacy-Risk Dynamics in Non-IID Federated Learning](./Paper%20I%20-%20Published%20Citation.md). IEEE BigDataService 2026, pp. 9-23. DOI: [10.1109/BigDataService70481.2026.00011](https://doi.org/10.1109/BigDataService70481.2026.00011). 15-page author manuscript archived in the research Google Drive; access currently restricted to owner.
- [Resilience of Federated Learning for Credit Risk Forecasting Under Heterogeneity, Label Noise and Partial Participation](./Paper%20II%20-%20IEEE%20Access%20Citation.md). IEEE Access, accepted 2026. DOI: [10.1109/ACCESS.2026.3743719](https://doi.org/10.1109/ACCESS.2026.3743719). 14-page accepted author manuscript archived in the research Google Drive; access currently restricted to owner.

[Export citations in BibTeX](./IEEE%202026%20bibliography.bib)
