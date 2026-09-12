# Paper III: Privacy-Resilient Federated Credit-Risk Forecasting

> Reading copy generated from the source Word manuscript with tracked changes accepted for research continuity and novelty auditing.

**Privacy-Resilient Federated Credit-Risk Forecasting: Differential Privacy, Membership-Inference Risk and Utility Loss under Operational Stress**

Imo Enang. Supervisor: Professor Taghi M. Khoshgoftaar. Co-authors to confirm.

*Working paper, 30 June 2026. Results are from a 100-seed run on a single public benchmark; the three round-level diagnostic figures use a 30-seed subset, as noted in their captions. Citations marked \[ref\] require final bibliographic verification.*

**Abstract**

Federation keeps borrower records local, but it does not, by itself, bound what a shared model reveals about individual records. We measure the cost of adding formal record-level differential privacy to a federated credit-scoring model, and the empirical membership risk, under heterogeneous, noisy and partially participating clients. The model is logistic regression trained by federated minibatch gradient descent across five simulated lending clients on the Taiwan Credit Card Default dataset (30,000 borrowers, 23 features, default rate 0.2212). We report utility, calibration and a membership-inference attack across six stress conditions and six privacy treatments, including a clipping-only baseline that separates bounded gradients from the privacy noise, over 100 seeds. Three results hold. At defensible budgets (worst-case epsilon 2.41 at the medium setting) the Gaussian noise costs about 0.16 Area Under the Curve (AUC) points on clean data, though it degrades probability calibration, with the calibration slope falling from 0.92 to 0.47. Membership signal stays near the no-signal floor except under severe class imbalance, where the advantage rises to 0.043 and its interval excludes zero across four attacks. Any stabilisation under stress comes from gradient clipping rather than the noise, since the clipping-only baseline and the medium-noise treatment coincide within intervals. The privacy-utility tension that dominates deep-network results does not transfer to this linear tabular setting, which gives lending consortia a defensible starting point for privacy-preserving deployment. The result is scoped to linear credit scoring and does not generalise to high-capacity models.

**1. Introduction**

A lending consortium would gain from a shared default model, but the member institutions cannot pool raw borrower records. Federated learning trains a shared model while each institution keeps its data local. That protects data locality. It does not bound what the trained model discloses about any single record, which is the concern a data-protection officer raises before deployment.

Credit scoring in production differs from the deep image and text models that shape most privacy-utility evidence. Capacity is low, features are tabular, and the model is often linear. Whether the steep accuracy cost reported for differentially private deep learning appears in this regime is an open question with direct operational stakes.

This paper sits within a three-part programme. Paper I set out the privacy-risk and optimiser problem in non-independent-and-identically-distributed (non-IID) federated learning \[ref, Paper I\]. Paper II moved that problem into credit risk and measured robustness under heterogeneous, noisy and partially participating clients \[ref, Paper II\]. Paper III adds formal record-level differential privacy (DP) and empirical membership testing to the same setting.

We ask four questions. What utility cost does example-level DP impose under stress? Do stressors raise membership signal? At a matched formal budget, does a front-loaded noise schedule improve the privacy-utility balance? And how much of any utility effect comes from gradient clipping rather than the added noise? The last question is answered with a clipping-only baseline. Our contribution is a boundary condition: for linear cross-silo credit scoring, formal privacy is available at defensible budgets with little discrimination cost, the stress benefit is a clipping effect, and leakage is near the floor except under severe imbalance.

**2. Related work**

Federated averaging and its use on tabular finance data frame the training setting \[ref, McMahan et al. 2017\]. Differentially private stochastic gradient descent and Renyi differential privacy accounting frame the privacy mechanism and its analysis \[ref, Abadi et al. 2016; ref, Mironov 2017; ref, Dwork and Roth 2014\]. Membership inference frames the empirical threat, from the loss-threshold test to shadow-model and likelihood-ratio attacks \[ref, Shokri et al. 2017; ref, Yeom et al. 2018; ref, Song and Mittal 2021; ref, Carlini et al. 2022\]. The reported privacy-utility trade-off comes largely from high-capacity models on images and text. The gap we address is the trade-off for linear federated credit scoring under operational stress, with worst-case formal accounting and a membership member set restricted to rows that actually trained.

**3. Threat model and privacy definition**

The adversary observes the trained model, and for one diagnostic, selected round checkpoints, then asks whether a given record was part of training. The attack is a final-model attack, with a per-round checkpoint diagnostic, not a raw update-transcript attack. The privacy unit is the record, since a borrower's row lives in exactly one client and server-side aggregation is post-processing that cannot weaken the guarantee. The formal target is example-level (epsilon, delta) differential privacy. Empirical leakage is the membership advantage, twice the attack AUC minus one. Scope is limited to a single public dataset, simulated clients and lower-bound attacks.

**4. Experimental design**

The dataset is the Taiwan Credit Card Default set (30,000 borrowers, 23 features, default rate 0.2212) \[ref, Yeh and Lien 2009\]. Each seed gives a stratified 80/20 train-test split and standardisation fitted on the training split only. Each row carries a stable identifier through splitting, partitioning, sub-sampling and noise injection, so the membership attack uses exactly the rows that entered training. Five simulated lending clients are formed by credit-limit quantile. Table 1 describes the client partitions. The severe condition drops rows from two clients; those rows leave training and are excluded from the member set.

*Table 1. Simulated client diagnostics per condition (illustrative single seed).*

| **Cond** | **Description**             | **Clients** | **Min** | **Mean** | **Max** | **Default rate** | **Part.** | **Exp. part.** |
|----------|-----------------------------|-------------|---------|----------|---------|------------------|-----------|----------------|
| C1       | moderate non-IID            | 5           | 3834    | 4800     | 6117    | 0.216            | 1.00      | 5              |
| C2       | severe non-IID              | 5           | 2407    | 4240     | 4898    | 0.207            | 1.00      | 5              |
| C3       | moderate, 20% symmetric     | 5           | 3834    | 4800     | 6117    | 0.330            | 1.00      | 5              |
| C4       | moderate, 40% participation | 5           | 3834    | 4800     | 6117    | 0.216            | 0.40      | 2              |
| C5       | 20% symmetric, 60% part     | 5           | 3834    | 4800     | 6117    | 0.330            | 0.60      | 3              |
| C6       | 20% asymmetric, 40% part    | 5           | 3834    | 4800     | 6117    | 0.373            | 0.40      | 2              |

The model is logistic regression trained by federated minibatch gradient descent for 50 rounds, with eight local steps per client per round, target Poisson lot size 128, learning rate 1.0 and L2 of 1e-4, aggregated by client size. Poisson sampling is used so the training procedure matches the privacy accountant. Logistic regression is the appropriate choice here: it is standard in credit scoring, interpretable, and gives a clean boundary condition against deep-network DP results. The configuration is fixed a priori and applied identically to every treatment.

Six conditions are studied. C1 is moderate non-IID. C2 is severe non-IID with an imbalance shock. C3 adds 20 percent symmetric label noise. C4 uses 40 percent client participation per round. C5 combines 20 percent symmetric noise with 60 percent participation. C6 combines 20 percent asymmetric noise with 40 percent participation. Six privacy treatments are applied. D0 is non-private and unclipped. D0C applies the clip with no noise, so it carries no formal guarantee and isolates bounded gradients. D1, D2 and D3 are uniform DP at noise multiplier 0.8, 1.5 and 2.5. D4 is a front-loaded schedule matched to the D2 budget per condition, with the early 40 percent of rounds at twice the noise of the rest.

**5. Privacy accounting and attack evaluation**

Epsilon is computed with the Opacus 1.6.0 subsampled-Gaussian Renyi accountant, which assumes Poisson sampling, at delta = 1e-5. Reported epsilon is worst-case: it uses the smallest client per condition for the sample rate and assumes a client trains in every round, so partial participation is not claimed as amplification. A closed-form Gaussian accountant without subsampling gives a conservative cross-check and reads higher, as expected. Table 3 reports the accounting.

*Table 3. Privacy accounting, worst-case across clients and seeds. Delta = 1e-5, Opacus Renyi accountant, Poisson sampling, no participation amplification.*

| **Treatment** | **Noise multiplier** | **Epsilon (moderate)** | **Epsilon range across conditions** | **Selected order** |
|---|---:|---:|---:|---:|
| D1 | 0.8 | 8.13 | 7.86 to 12.95 | 3 |
| D2 | 1.5 | 2.41 | 2.31 to 4.04 | 8 |
| D3 | 2.5 | 1.22 | 1.16 to 2.02 | 16 |
| D4 | front-loaded | 2.41 | 2.31 to 4.04 | 8 |

The member set contains only rows that trained. Non-members are the held-out test rows, matched in count per seed. Two attack families are reported. Label-aware attacks score confidence on the record's label, using the loss-threshold test of Yeom and colleagues (2018) and the modified-entropy test of Song and Mittal (2021); for members these use the observed post-noise label, the primary view, or the original clean label, an auditor view. Label-agnostic attacks use predictive confidence and entropy only. On a binary task the two label-aware scores share a ranking, as do the two label-agnostic scores, so two distinct signals are measured. The primary attack is the observed-label loss-threshold. All are lower bounds; a likelihood-ratio attack would tighten them.

**6. Results**

On clean data the privacy noise costs little discrimination. At the medium budget the AUC change against the non-private model is -0.0016 [-0.0022, -0.0010] on the moderate condition, about one sixth of one percentage point. On the clean partial-participation condition the change is positive at +0.0047 [+0.0028, +0.0066], since clipping stabilises the noisy participation. Absolute AUC on the moderate condition is 0.722 without privacy and 0.720 with medium privacy.

Under severe class imbalance, membership signal rises. The primary attack advantage reaches about 0.043 and its confidence interval excludes zero. On the moderate condition it remains at or close to the no-signal floor. The four attacks agree on the imbalance signal.

At matched formal budget, the front-loaded schedule and the uniform D2 schedule overlap on final-model utility and leakage. Early gradients are larger, but the final-model attack does not show a benefit from redistributing the same budget under this linear setting. This null result is important because it means a later risk-adaptive paper cannot claim front-loading itself as new.

The clipping-only baseline and D2 overlap within intervals on several stress conditions, showing that any stabilisation under participation or label noise is a bounded-gradient effect, not a privacy-noise effect. Calibration is more sensitive than discrimination: the calibration slope on the moderate condition falls from about 0.92 to about 0.47 at the medium privacy setting.

**7. Discussion**

One mechanism explains the findings. The model is linear on 24 standardised features with a discrimination ceiling near 0.72, so per-example gradients are bounded and the clip at norm 1.0 bites lightly. The added noise is small relative to the aggregated gradient, which is why discrimination survives. The same low capacity means the model does not memorise individual records or label noise, which is why leakage is near the floor and why the observed-label attack turns negative under noise. Severe imbalance is the exception, because minority-heavy clients leave more identifiable structure in the fitted model.

The result contrasts with deep networks, where capacity drives both memorisation and a steep privacy cost. What is specific here is the linear tabular setting. Moving to gradient-boosted trees, a shallow network or richer borrower data would raise capacity, and the trade-off seen in the deep-learning literature would likely return. The calibration loss is the one cost that persists even in this benign regime, and it deserves attention wherever the score feeds pricing rather than ranking.

**8. Practical implications**

For this model class a meaningful formal guarantee near epsilon 2 is available with little discrimination cost, which removes a common objection to deploying differential privacy in lending. Bounded-gradient training also stabilises the model under noisy or intermittent client data, so it is worth enabling for robustness alone. Three governance points follow: fix and audit epsilon before training, justify delta against the dataset size, and recalibrate the probability output after private training if it is used for pricing. Re-test the trade-off whenever the model class changes, since the benign result here is a property of the linear setting.

**9. Limitations and future research**

The dataset is single-source and the clients are simulated, so external validity is limited; a second credit dataset and a real cross-silo deployment would strengthen the claim. The attacks are lower bounds, and a likelihood-ratio attack is the first extension, particularly for the severe-imbalance signal. The model is linear, so the boundary condition should be tested directly on gradient-boosted trees and a shallow network. The front-loaded result may be null on the final model and depends on the threat model. Worst-case accounting is conservative; a tighter amplification bound would lower the reported epsilon. The three round-level diagnostics use 30 seeds and should be repeated at full scale before publication.

**10. Conclusion**

For linear federated credit scoring on the Taiwan dataset, example-level differential privacy at defensible budgets costs little discrimination, membership leakage is near the floor except under severe class imbalance, and any stress benefit comes from bounded gradients rather than the privacy noise. Calibration is the cost that remains. The privacy-utility tension that dominates deep-network results does not transfer to this setting, which gives lending consortia a defensible starting point for privacy-preserving deployment. The claim is scoped to the linear tabular case and should be re-tested for higher-capacity models.

**References**

*Details require final verification before submission. Papers I and II are the author's companion works.*

Abadi, M. and colleagues (2016). Deep Learning with Differential Privacy. ACM Conference on Computer and Communications Security. \[ref\]

Carlini, N. and colleagues (2022). Membership Inference Attacks From First Principles. IEEE Symposium on Security and Privacy. \[ref\]

Dwork, C. and Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. Foundations and Trends in Theoretical Computer Science. \[ref\]

McMahan, B. and colleagues (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS. \[ref\]

Mironov, I. (2017). Renyi Differential Privacy. IEEE Computer Security Foundations Symposium. \[ref\]

Shokri, R. and colleagues (2017). Membership Inference Attacks Against Machine Learning Models. IEEE Symposium on Security and Privacy. \[ref\]

Song, L. and Mittal, P. (2021). Systematic Evaluation of Privacy Risks of Machine Learning Models. USENIX Security. \[ref\]

Yeh, I. and Lien, C. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications. \[ref\]

Yeom, S. and colleagues (2018). Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. IEEE Computer Security Foundations Symposium. \[ref\]

Enang, I. (companion). Paper I: privacy risk and optimiser choice in non-IID federated learning. \[ref\]

Enang, I. (companion). Paper II: robustness of federated credit-risk models under heterogeneous, noisy and partially participating clients. \[ref\]
