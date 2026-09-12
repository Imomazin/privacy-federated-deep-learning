risk. The transition corresponds to the model moving from random initialisation, where all
gradient directions carry substantial per-sample information, to a region where gradients
become smaller and more aligned with dominant curva- ture. Yin et al. (2021) showed
similar effects; our contribution is to quantify them under non-IID federated conditions
with Dirichlet partitioning. DP-SGD provides additional protection, reducing round-5 PSNR
from 12.8 to 7.1 dB. The combined evidence suggests that privacy mechanisms and model
release policies should be designed with training dynamics in mind. Protecting early rounds
more aggressively (through higher noise, delayed release or secure aggregation)
would yield greater privacy benefit than uniform protection across all rounds.
7.3 Privacy-Utility Tradeoff
The DP baseline with σ = 1.0 incurs approximately 3 percentage points of accuracy
20expected participation (20% per round), ε
degradation for ε = 6.24 (worst-case). Under
tightens to 3.40. Extended training to 100 rounds substantially increases expendi- ture
(ε = 21.01), illustrating the cumulative cost of composition. This creates a ten- sion:

more rounds improve accuracy but erode privacy guarantees. The early-round focus
partially sidesteps this by operating at lower cumulative ε. The interaction between
privacy expenditure and optimisation stability deserves care- ful attention. Our results
indicate that privacy risk is not static across training but phase-dependent. The early
optimisation regime, characterised by large gradient norms and rapid parameter
movement, presents systematically higher attack fidelity than later stages. This suggests
that privacy mechanisms should be designed with training dynamics in mind rather
than evaluated solely at convergence. A practical implication: adaptive noise schedules
that inject more noise in early rounds and less in later rounds could achieve better
privacy-utility tradeoffs than the uniform σ = 1.0 we evaluate here. Translating ε values
to intuitive privacy semantics remains an open challenge (Cum- mings et al., 2021). Our
reported ε ≈ 6 at T = 10 represents a moderate privacy regime. Practitioners should
interpret these values in conjunction with the empiri- cal attack results (membership
inference AUC declining from 0.61 to 0.53 under DP) rather than relying on ε alone.
7.4 Practical Implications for Federated System Design
Three practical recommendations emerge from these results. First, server-side adap- tive
optimisation should be the default for federated deployments under severe heterogeneity. The additional computational cost at the server is negligible (maintaining two
additional state vectors per parameter), and the performance gains under class- restricted
conditions are substantial and statistically significant. Second, for applications where
variance control matters more than peak accuracy (e.g. safety-critical systems, regulated
environments), FedProx at appropriately tuned μ provides measurably more predictable
convergence. The 80% variance reduction at μ = 0.1 means fewer rounds where
accuracy unexpectedly regresses, a property valued in production monitoring. Third,
privacy protection should be front-loaded. Given that both gradient inversion and
membership inference success peak in early rounds, deploying stronger protec- tions
during the first 10–20 rounds (whether through higher noise, secure aggrega- tion or
delayed model release) would yield greater privacy benefit per unit of accuracy cost than
uniform protection across all rounds.
7.5 Limitations
Seed count. Twenty seeds for the main FedAvg and FedAdam evaluations provide
reasonably tight confidence intervals (95% CI width of 1.16 pp on CIFAR-10 FedAvg; 3.57
pp on MNIST FedAdam). The 3-seed FedProx and DP-FedAvg comparisons on CIFAR-10
are directionally informative but would benefit from expansion to 20 seeds in future work.
The 3-seed design was adopted for these conditions as exploratory comparisons rather
than primary baselines. The Levene’s test result (p < 0.01) and bootstrap CI provide
statistical support, but cautious interpretation is warranted.
Attack scope. Our gradient inversion uses a simplified DLG variant with single- sample
gradients. Production federated systems aggregate over batches and multiple clients,
reducing inversion fidelity. The membership inference attack uses confidence thresholds,
among the simplest available; more sophisticated
methods (Carlini et al., 2022) may yield
21
higher AUC.
Architecture and scale. The CNN (470K parameters) omits batch normalisation and

dropout. The 50-client (CIFAR-10) and 10-client (MNIST) populations represent
moderate scale; production deployments involve orders of magnitude more clients.
Alternative normalisation strategies (Li et al., 2021), adaptive aggregation (Reddi et al.,
2021) and drift correction beyond FedProx (Karimireddy et al., 2020; Acar et al., 2021)
would likely improve accuracy and may alter privacy characteristics.
Benchmark scope. CIFAR-10 and MNIST are controlled benchmarks. The LEAF benchmark
suite (Caldas et al., 2019) offers more realistic federated datasets with natural user-based
partitioning. Extending to larger-scale models and datasets re- mains necessary for
stronger generalisability claims.
8. Conclusion
We established reproducible baselines for FedAvg, FedProx and FedAdam on CIFAR- 10
and MNIST under two non-IID regimes, evaluated across up to 20 independent seeds per
algorithm (20 seeds for main baselines; 3 seeds for exploratory compar- isons). Four
principal findings emerge, each supported by statistical evidence. Adaptive server-side
optimisation significantly improves convergence under severe heterogeneity.
FedAdam achieves 81.93% peak accuracy on MNIST non-IID, com- pared with 65.21%
for FedAvg (t = 3.69, p = 0.002) and 69.46% for FedProx (t = 5.29, p < 0.001). The
advantage arises from server-side momentum smoothing con- flicting client updates
across rounds. Across 20 seeds, FedAdam achieves mean peak accuracy of 74.10% (std
= 4.07%, 95% CI [72.31%, 75.88%]), confirming robustness. Proximal regularisation
reduces optimisation variance by a statistically significant margin under Dirichlet
heterogeneity. On CIFAR-10, FedProx at μ = 0.1 reduces cross-seed variance by 80%
(Levene's test W = 12.41, p = 0.008) without improving mean accuracy. The bootstrap
95% CI for the variance ratio ([0.04, 0.52]) excludes 1.0. This reframes FedProx's
contribution in early-training regimes as a stability mechanism rather than a convergence
accelerator. Weak proximal regularisation (μ = 0.01) actually increases variance, indicating
a non-monotonic relationship between regularisation strength and stability. Early-round
gradients exhibit empirically higher reconstruction vulnerability. Gradi- ent norms decline
3.2× between rounds 1 and 50. Inversion PSNR drops 30% and membership inference AUC
decreases from 0.63 to 0.55 over the same interval. The first 10–20 rounds represent the
highest-risk window for both gradient inversion and membership inference. The steepest
decline in both gradient norms and attack suc- cess occurs in the first 15 rounds, after
which metrics stabilise.
Differential privacy reduces attack fidelity at measurable accuracy cost. DP-SGD with σ
= 1.0 reduces inversion PSNR by 44% (from 12.8 to 7.1 dB) and membership infer- ence
AUC from 0.61 to 0.53, at approximately 3 percentage points of accuracy and ε = 6.24
(worst-case, δ = 10⁻⁵). Under expected participation, ε tightens to 3.40. Robustness
experiments on MNIST confirm that FedAvg tolerates substantial vari- ation in Dirichlet
concentration (α = 0.1 to 1.0), client participation rate (25% to 100%) and
communication budget (10 to 50 rounds), with peak accuracy differences below 0.3
percentage points across all conditions given
22 sufficient rounds. These ex- periments
establish baselines against which future work on heterogeneity-aware al- gorithms can
be measured. These findings connect optimisation behaviour to privacy risk through
calibrated em- pirical evidence. The key practical insight is that privacy risk is phase-

dependent rather than static: the early-round regime warrants stronger protection, and
the choice of optimiser (FedAvg, FedProx, FedAdam) affects not only convergence but
the predictability and vulnerability of the gradient landscape. The baselines, attack
experiments, trained models and per-round metrics are publicly available to support
systematic comparison.
Reproducibility Statement
All experimental artefacts are available at https://github.com/imomazin/privacyfederated-deep- learning. The repository contains training notebooks for FedAvg (20seed), FedProx, FedAdam, DP-FedAvg, Dirichlet heterogeneity and client partic- ipation
experiments, compatible with Google Colaboratory. Environment: Python 3.11, PyTorch
2.0, Opacus 1.4, NumPy 1.24, scikit-learn 1.5, CPU runtime.
References
Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K. and Zhang,
L. (2016). Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC
Conference on Computer and Communications Security, pp. 308–318. ACM.
Acar, D. A. E., Zhao, Y., Navarro, R. M., Mattina, M., Whatmough, P. N. and Saber, V.
(2021). Federated learning based on dynamic regularization. In ICLR.
Anderson, M. (2015). Technology device ownership: 2015. Pew Research Center Report.
Andreux, M., du Terrail, J. O., Beguier, C. and Tramel, E. W. (2020). Siloed federated
learning for multi-centric histopathology datasets. In MICCAI Workshop, pp. 129– 139.
Apple (2017). Learning with privacy at scale. Apple Machine Learning Journal.
Balle, B., Barthe, G., Gavin, M. and Stronati, M. (2020). Hypothesis testing interpretations and Rényi differential privacy. In AISTATS, pp. 2496–2506.
Bassily, R., Smith, A. and Thakurta, A. (2014). Private empirical risk minimization. In
FOCS, pp. 464–473.
Bell, J. H., Bonawitz, K. A., Gascón, A., Lepoint, T. and Raykova, M. (2020). Secure
single-server aggregation with (poly)logarithmic overhead. In CCS, pp. 1253–1269.
Boenisch, F., Dziedzic, A., Schuster, R., Shamsabadi, A. S., Shumailov, I. and Papernot,
N. (2023). When the curious abandon honesty: Federated learning is not private. In
IEEE European Symposium on Security and Privacy, pp. 175–199.
Bonawitz, K. et al. (2017). Practical secure aggregation for privacy-preserving ma- chine
learning. In CCS, pp. 1175–1191.
Bun, M. and Steinke, T. (2016). Concentrated differential privacy. In TCC, pp. 635– 658.
Caldas, S. et al. (2019). LEAF: A benchmark for federated settings. arXiv:1812.01097.
Carlini, N. et al. (2022). Membership inference attacks from first principles. In IEEE S&P,
pp. 1897–1914.
23
Cheng, Y., Liu, Y., Chen, T. and Yang, Q. (2022).
Federated learning for
privacy- preserving AI. Communications of the ACM, 65(7):90–100.

Cummings, R., Kaptchuk, G. and Nissim, K. (2021). I need a better description: An
investigation into user expectations for differential privacy. In CCS, pp. 3037–3052.
Draxler, F., Veschgini, K., Salmhofer, M. and Hamprecht, F. (2018). Essentially no
barriers in neural network energy landscape. In ICML, pp. 1309–1318.
Dwork, C., McSherry, F., Nissim, K. and Smith, A. (2006). Calibrating noise to sensi- tivity
in private data analysis. In TCC, pp. 265–284.
Dwork, C. and Roth, A. (2014). The algorithmic foundations of differential privacy.
Foundations and Trends in Theoretical Computer Science, 9(3–4):211–407.
Fowl, L. et al. (2022). Decepticons: Corrupted transformers breach privacy in feder- ated
learning for language models. In ICLR.
GDPR (2016). Regulation (EU) 2016/679. Ofiicial Journal of the European Union.
Geiping, J., Bauermeister, H., Dröge, H. and Moeller, M. (2020). Inverting gradients: How
easy is it to break privacy in federated learning? In NeurIPS, pp. 16937–16947.
Geyer, R. C., Klein, T. and Nabi, M. (2017). Differentially private federated learning: A
client level perspective. arXiv:1712.07557.
Goodfellow, I. J., Vinyals, O. and Saxe, A. M. (2015). Qualitatively characterizing
neural network optimization problems. In ICLR.
Hard, A. et al. (2018).
arXiv:1811.03604.

Federated learning for mobile keyboard prediction.

Hsieh, K., Phanishayee, A., Mutlu, O. and Gibbons, P. (2020). The non-IID data quagmire of decentralized machine learning. In ICML, pp. 4387–4398.
Hsu, T. M. H., Qi, H. and Brown, M. (2019). Measuring the effects of non-identical
data distribution for federated visual classification. arXiv:1909.06335.
Huang, Y. et al. (2021). Evaluating gradient inversion attacks and defenses in feder- ated
learning. In NeurIPS, pp. 7232–7241.
Humphries, T., Chipperfield, S., Mayberry, J. and Cummings, R. (2020). Differentially
private learning does not bound membership inference. arXiv:2010.12112.
Kairouz, P., McMahan, H. B. et al. (2021). Advances and open problems in federated
learning. Foundations and Trends in Machine Learning, 14(1–2):1–210.
Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S. and Suresh, A. T. (2020).
SCAFFOLD: Stochastic controlled averaging for federated learning.
In
ICML,
pp. 5132–5143.
Khaled, A., Mishchenko, K. and Richtárik, P. (2020). Tighter theory for local SGD on
identical and heterogeneous data. In AISTATS, pp. 4519–4529.
Konečný, J., McMahan, H. B., Yu, F. X., 24
Richtárik, P., Suresh, A. T. and Bacon, D.
(2016). Federated learning: Strategies for improving communication efficiency.
arXiv:1610.05492.
Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Techni- cal

Report, University of Toronto.
LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P. (1998). Gradient-based learning applied
to document recognition. Proceedings of the IEEE, 86(11):2278–2324.
Li, Q., Diao, Y., Chen, Q. and He, B. (2022). Federated learning on non-IID data silos: An
experimental study. In ICDE, pp. 965–978.
Li, T., Sahu, A. K., Talwalkar, A. and Smith, V. (2020a). Federated learning: Chal- lenges,
methods and future directions. IEEE Signal Processing Magazine, 37(3):50– 60.
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A. and Smith, V. (2020b). Federated optimization in heterogeneous networks. In MLSys, pp. 429–450.
Li, X. et al. (2021). FedBN: Federated learning on non-IID features via local batch
normalization. In ICLR.
Long, Y. et al. (2020). Pragmatic adversarial privacy for membership inference attack. In
IEEE EuroS&P, pp. 497–512.
McMahan, H. B., Moore, E., Ramage, D., Hampson, S. and Arcas, B. A. y. (2017).
Communication-efficient learning of deep networks from decentralized data. In AISTATS, pp. 1273–1282.
McMahan, H. B., Ramage, D., Talwar, K. and Zhang, L. (2018). Learning differentially
private recurrent language models. In ICLR.
Melis, L., Song, C., De Cristofaro, E. and Shmatikov, V. (2019). Exploiting unintended
feature leakage in collaborative learning. In IEEE S&P, pp. 691–706.
Mironov, I. (2017). Rényi differential privacy. In IEEE Computer Security Foundations
Symposium, pp. 263–275.
Nasr, M., Shokri, R. and Houmansadr, A. (2019). Comprehensive privacy analysis of
deep learning. In IEEE S&P, pp. 739–753.
Nissim, K., Raskhodnikova, S. and Smith, A. (2007). Smooth sensitivity and sampling in
private data analysis. In STOC, pp. 75–84.
Noble, M., Bellet, A. and Dieuleveut, A. (2022). Differentially private federated learning on heterogeneous data. In AISTATS, pp. 10110–10132.
Papernot, N. et al. (2021). Tempered sigmoid activations for deep learning with differential privacy. In AAAI, pp. 9312–9321.
Poushter, J. (2016). Smartphone ownership and internet usage continues to climb in
emerging economies. Pew Research Center.
Ramaswamy, S., Mathews, R., Rao, K. and Beaufays, F. (2019). Federated learning for
emoji prediction in a mobile keyboard. arXiv:1906.04329.
Reddi, S. J. et al. (2021). Adaptive federated optimization. In ICLR.
Salem, A. et al. (2019). ML-Leaks: Model and
25data independent membership infer- ence
attacks and defenses. In NDSS.
Scheliga, D., Mäder, P. and Seeland, M. (2022). PRECODE: A generic model extension to

prevent deep gradient leakage. In WACV, pp. 3605–3614.
Shokri, R., Stronati, M., Song, C. and Shmatikov, V. (2017). Membership inference
attacks against machine learning models. In IEEE S&P, pp. 3–18.
So, J., Güler, B. and Avestimehr, A. S. (2022). Turbo-aggregate: Breaking the quadratic
aggregation barrier in secure federated learning. IEEE JSAIT, 3(1):48–66.
Song, S., Chaudhuri, K. and Sarwate, A. D. (2013). Stochastic gradient descent with
differentially private updates. In GlobalSIP, pp. 245–248.
Sotthiwat, E., Zhu, L., Zhang, C. and Li, Z. (2021). Partially encrypted multi-party
computation for federated learning. In IWQOS, pp. 1–10.
Sun, J. et al. (2021). Soteria: Provable defense against privacy leakage in federated
learning. In CVPR, pp. 9311–9319.
Tramèr, F. and Boneh, D. (2021). Differentially private learning needs better features (or
much more data). In ICLR.
Wang, H., Kaplan, Z., Niu, D. and Li, B. (2021a). Optimizing federated learning on nonIID data with reinforcement learning. In INFOCOM, pp. 1698–1707.
Wang, J., Liu, Q., Liang, H., Joshi, G. and Poor, H. V. (2020). Tackling the objective inconsistency problem in heterogeneous federated optimization. In NeurIPS, pp. 7611–
7623.
Wei, K. et al. (2020). Federated learning with differential privacy: Algorithms and
performance analysis. IEEE TIFS, 15:3454–3469.
Yeom, S., Giacomelli, I., Fredrikson, M. and Jha, S. (2018). Privacy risk in machine
learning: Analyzing the connection to overfitting. In CSF, pp. 268–282.
Yin, H. et al. (2021). See through gradients: Image batch recovery via GradInversion. In
CVPR, pp. 16337–16346.
Yousefpour, A. et al. (2021). Opacus: User-friendly differential privacy library in Py- Torch.
arXiv:2109.12298.
Yurochkin, M. et al. (2019). Bayesian nonparametric federated learning of neural
networks. In ICML, pp. 7252–7261.
Zhao, B., Mopuri, K. R. and Bilen, H. (2020). iDLG: Improved deep leakage from
gradients. arXiv:2001.02610.
Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D. and Chandra, V. (2018). Federated learning with
non-IID data. arXiv:1806.00582.
Zhu, L., Liu, Z. and Han, S. (2019).
pp. 14774–14784.

Deep leakage from gradients.

26

In NeurIPS,

Figure 1. FedAvg 20-seed accuracy trajectories on CIFAR-10 (Dirichlet  = 0.5), showing mean ± 1 SD.

Figure 2. CIFAR-10 IID convergence comparison between FedAvg and FedProx ( = 0.01) over 20 seeds.

27

Figure 6. Cross-seed standard deviation comparison on CIFAR-10 non-IID.

Figure 3. MNIST non-IID convergence trajectories for FedAvg, FedProx and FedAdam (2 classes per
client).
28

Figure 4. Convergence under varying Dirichlet concentration  on MNIST.

29
Figure 5. Convergence under varying client participation rates on MNIST.

Figure 9. Accuracy vs communication round budget on MNIST.

Figure 7. Mean gradient L■ norm vs communication round, showing rapid decline in the first 15 rounds.
30

Figure 8. Both attack metrics decline with training progress; the steepest drop occurs in the first 20 rounds.

31


```
