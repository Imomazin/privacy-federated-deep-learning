# Paper II Research Continuity Copy

## Resilience of Federated Learning for Credit Risk Forecasting Under Heterogeneity, Label Noise and Partial Participation

This reading copy records the contribution, definitive protocol, results and limitations that matter for the RAPSA-FL novelty audit. It is based on the final manuscript in the research library.

## Contribution and scope

Paper II is a controlled empirical resilience study of horizontal Federated Averaging for credit-risk forecasting. It does not propose a new optimiser. It evaluates whether federated credit-risk learning preserves predictive performance when the consortium experiences realistic statistical and operational stress.

The study uses the Taiwan Credit Card Default dataset and a ten-experiment stress programme spanning:

- IID and non-IID baselines
- moderate and severe feature-distribution heterogeneity
- 10% and 20% symmetric label noise
- 20% asymmetric label noise
- 60% and 40% client participation
- two compound-stress conditions

The principal comparison is centralised training versus isolated local training versus federated training.

## Dataset and federation

Dataset: Taiwan Credit Card Default.

- 30,000 borrowers
- 23 predictor features
- binary next-month default target
- 77.88% non-default and 22.12% default
- 80/20 stratified train/test split
- training-set z-score normalisation
- class_weight='balanced'
- no resampling

The federation contains K = 5 simulated lending institutions.

Moderate non-IID partitioning groups borrowers by credit-limit quintile using `pd.qcut`. This creates covariate-based institutional heterogeneity, with lower- and higher-credit-limit borrowers concentrated in different clients. Severe non-IID further amplifies this divergence by subsampling selected class/client combinations.

## Model and training protocol

Base model: logistic regression.

Federated mechanism: sample-size-weighted FedAvg.

Principal protocol:

- 50 communication rounds
- one local epoch per round
- 100 independent random seeds
- full participation, 60% participation or 40% participation depending on condition
- client sampling repeated independently by round under the run seed
- `LogisticRegression` with `max_iter=5000`, `solver='lbfgs'`, `class_weight='balanced'`

Seeds are generated from:

```python
np.random.RandomState(2024).choice(range(1, 200000), size=100, replace=False)
```

Each seed controls the train/test split, partition draw, label-noise pattern and per-round client sample. Paradigms within one experiment use the same seed, enabling paired comparisons.

## Experiment roadmap

| ID | Stress condition | Paradigms |
|---|---|---|
| E1 | IID baseline | Central / Local / FedAvg |
| E2 | Moderate non-IID baseline | Central / Local / FedAvg |
| E3 | Severe non-IID baseline | Central / Local / FedAvg |
| E4 | Moderate non-IID + 10% symmetric label noise | Central / Local / FedAvg |
| E5 | Moderate non-IID + 20% symmetric label noise | Central / Local / FedAvg |
| E6 | Moderate non-IID + 20% asymmetric label noise | Central / Local / FedAvg |
| E7 | Moderate non-IID + 60% client participation | Central / Local / FedAvg |
| E8 | Moderate non-IID + 40% client participation | Central / Local / FedAvg |
| E9 | 20% symmetric noise + 60% participation | Central / Local / FedAvg |
| E10 | 20% asymmetric noise + 40% participation | Central / Local / FedAvg |

## Evaluation and statistics

Primary metric: ROC-AUC.

Secondary credit-risk measures include Kolmogorov-Smirnov and Gini where relevant.

Each experiment reports:

- mean AUC
- standard deviation
- 95% confidence interval
- paired t-test on the 100 matched seeds
- Cohen's d on paired differences

The paper also defines degradation coefficients for heterogeneity, symmetric noise, asymmetric noise, participation and compound stress.

## Definitive AUC results

| Experiment | Centralised | Local | Federated |
|---|---:|---:|---:|
| E1 IID | 0.7232 | 0.7204 | 0.7242 |
| E2 moderate non-IID | 0.7232 | 0.7025 | 0.7213 |
| E3 severe non-IID | 0.7072 | 0.7009 | 0.7208 |
| E4 +10% symmetric noise | 0.7195 | 0.6959 | 0.7192 |
| E5 +20% symmetric noise | 0.7163 | 0.6848 | 0.7155 |
| E6 +20% asymmetric noise | 0.7172 | 0.6917 | 0.7166 |
| E7 +60% participation | 0.7232 | 0.7025 | 0.7194 |
| E8 +40% participation | 0.7232 | 0.7025 | 0.7171 |
| E9 compound symmetric | 0.7163 | 0.6848 | 0.7110 |
| E10 compound asymmetric | 0.7172 | 0.6917 | 0.7087 |

These are Paper II results. They must not be presented as new RAPSA-FL results.

## Main findings

Federated learning outperforms local training in every tested condition. Paired effects against local training are large, with Cohen's d between 1.80 and 6.54 and p < 0.001 across the programme.

The federated heterogeneity decline across E1 to E3 is 0.0034 AUC, compared with 0.0195 for local learning. The paper reports a heterogeneity degradation coefficient of 0.0017 AUC per heterogeneity level for federated learning and 0.0098 for local learning, a 5.8-fold difference.

Under 20% symmetric label noise, federated AUC declines by 0.0058 from the moderate non-IID baseline. Under 20% asymmetric noise it declines by 0.0047. Local models degrade more strongly.

At 40% client participation, final federated AUC is 0.7171, a 0.0042 decline from full participation. Peak AUC within the 50-round horizon remains near the centralised reference, indicating slower convergence and greater variance under partial participation rather than catastrophic failure.

The severest compound condition, 20% asymmetric noise with 40% participation, produces federated AUC 0.7087. This remains above the clean local baseline of 0.7025.

## Critical boundary for RAPSA-FL

Paper II does not implement:

- cryptographic secure aggregation
- formal differential privacy
- transport-layer communication security
- real cross-institutional networking

Its federation is simulated on one machine with disjoint client partitions and parameter aggregation. The paper explicitly identifies secure-aggregation overhead and the accuracy cost of formal differential privacy as follow-up work.

That makes Paper II a direct empirical predecessor for RAPSA-FL, but its results cannot be re-labelled as RAPSA-FL results.

## Limitations relevant to the new paper

- single credit dataset
- five simulated clients
- logistic-regression base model
- simulated single-machine federation
- no secure aggregation
- no formal DP
- no direct communication/latency measurement
- horizontal FL only

## Research-continuity rule

RAPSA-FL may reuse the dataset, partitioning logic, seed discipline and stress definitions where scientifically appropriate. Any claim about the new privacy/secure-aggregation architecture requires a new execution under the new configuration.
