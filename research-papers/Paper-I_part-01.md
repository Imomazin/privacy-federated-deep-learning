# Paper I: Communication-Efficient Learning with Differential Privacy

> Reading copy extracted from the source PDF for research continuity and novelty auditing.

```text
Communication-Efficient Learning with Differential Privacy: Empirical Privacy-Utility
Analysis Under Non-IID Federated Conditions
Code and Data: https://github.com/imomazin/privacy-federated-deep-learning
Abstract
Federated Learning (FL) enables collaborative model training without centralizing raw data,
which makes it suitable for privacy-sensitive applications. However, the behaviour of
optimisation algorithms under heterogeneous data distributions and the implications for
privacy risk are still not well understood. This study investigates the relationship between
optimisation dynamics and privacy vulnerability in feder- ated learning using the MNIST and
CIFAR-10 benchmarks under non-Independent and Identically Distributed (non-IID)
conditions. Three optimisation approaches, Federated Averaging (FedAvg), Federated
Proximal (FedProx) and Federated Adam (FedAdam), are evaluated to examine their
convergence behaviour, robustness to heterogeneous data, and sensitivity to
communication constraints.
The results show that adaptive server-side optimisation
improves convergence under strongly heterogeneous data distributions, while proximal
regularization stabilizes training by reducing variability across experimental runs. Analysis of
training dynamics also shows that gradients are substantially stronger during the early
communication rounds of federated training. Consistent with this observation, privacy
attacks based on gradient reconstruction and membership inference are more effective
during the early stages of training. Incorporating Differential Privacy (DP) reduces the
success of these attacks but introduces a moderate reduction in model performance. Overall,
the findings suggest that privacy risk in federated learning is closely linked to opti- misation
behaviour during training, particularly in the early communication rounds, and that training
dynamics should be considered when designing privacy-preserving federated learning
systems.
Keywords: Federated Learning, Differential Privacy, Non-IID Data, Privacy Attacks,
Gradient Inversion, Membership Inference, FedProx, FedAdam, Client Drift
1. Introduction
Mobile devices generate data at scale suitable for training machine learning models, yet
centralised collection creates significant privacy risks (Anderson, 2015; Poushter, 2016; Hard
et al., 2018; Shokri et al., 2017). The European General Data Protec- tion Regulation
(GDPR, 2016) and comparable legislation impose obligations on data controllers that make
direct data pooling impractical in many settings. These con- straints motivate learning
paradigms that extract collective benefit from distributed data without centralised
aggregation.
Federated learning addresses this through a decentralised protocol wherein partic- ipating
devices collaboratively train a shared model by computing local updates ag1

gregated at a central server, without transmitting raw training samples (McMahan et al.,
2017; Konečný et al., 2016; Kairouz et al., 2021). The approach implements data
minimisation principles and has been deployed at Google for keyboard prediction (Hard et
al., 2018), Apple for language modelling (Apple, 2017) and other production systems serving
hundreds of millions of users (Ramaswamy et al., 2019).
Despite transmitting only model updates, federated systems remain susceptible to inference
attacks (Kairouz et al., 2021). Membership inference attacks determine whether specific
records participated in training (Shokri et al., 2017; Nasr et al., 2019). Gradient-based
reconstruction attacks recover training images from shared updates (Zhu et al., 2019; Geiping
et al., 2020). Attribute inference attacks extract sensitive properties from model updates
(Melis et al., 2019; Boenisch et al., 2023). These findings motivate integration of formal privacy
mechanisms, principally dif- ferential privacy through the Differentially Private Stochastic
Gradient Descent (DP- SGD) algorithm (Abadi et al., 2016). Evaluating privacy mechanisms
requires reproducible baselines under controlled con- ditions, yet the federated learning
literature exhibits substantial heterogeneity in ex- perimental configurations (Li et al., 2022;
Wang et al., 2021a). Studies vary in non- IID simulation strategies, model architectures and
communication budgets, impeding systematic comparison. Data partitioning procedures
often lack explicit characteri- sation of heterogeneity severity. Most comparisons employ
FedAvg alone without evaluating alternative methods under identical conditions. This paper
addresses these gaps through four empirical contributions: We evalu- ate three federated
optimisation algorithms (FedAvg, FedProx, FedAdam) across two benchmarks (MNIST and
CIFAR-10) under distinct non-IID regimes, using up to 20 independent seeds per algorithm
for statistical robustness (20 seeds for main base- lines; 3 seeds for exploratory
comparisons). We then demonstrate that FedAdam significantly outperforms both
FedAvg and FedProx on MNIST under class-restricted heterogeneity (paired t-test: t = 3.69,
p = 0.002 vs FedAvg; t = 5.29, p < 0.001 vs FedProx), whilst proximal regularisation on
CIFAR-10 reduces cross-seed variance by 80% (Levene's test W = 12.41, p = 0.008) without
improving mean accuracy. We show that early-round gradient regimes materially differ
from converged regimes, with gradient norms declining 3.2× between rounds 1 and 50 and
gradient inversion attacks achieving 74% higher reconstruction fidelity at round 5
compared to round
50. Finally, we quantify the interaction between privacy expenditure and optimisation
stability, demonstrating that DP-SGD reduces gradient inversion Peak Signal-to-Noise Ratio
(PSNR) by 5.7 dB and membership inference Area Under the Curve (AUC) from
0.61 to 0.53 at a cost of approximately 3 percentage points of accuracy.
To clarify the scope of this contribution: this paper does not propose a new federated
optimisation algorithm. It is a controlled empirical study that examines three exist- ing
algorithms (FedAvg, FedProx, FedAdam) under matched experimental conditions.

2

The contribution lies in the systematic analysis of optimisation dynamics, privacy vulnerability and multi-seed robustness under non-IID federated settings, with particular
attention to early-round behaviour where privacy risk is highest. The study provides calibrated
baselines and reproducible evidence to inform algorithm selection and pri- vacy mechanism
design, rather than introducing novel methods. Evaluation focuses on convergence
(classification accuracy), robustness (cross-seed variance), and pri- vacy vulnerability
(gradient inversion and membership inference metrics).
2. Related Work
2.1 Federated Optimisation and Statistical Heterogeneity
Federated Averaging established communication-efficient distributed learning by
demonstrating that multiple local Stochastic Gradient Descent (SGD) steps before server
aggregation reduces communication cost whilst maintaining convergence (McMahan et al.,
2017; Konečný et al., 2016). Subsequent theoretical work charac- terised convergence under
heterogeneity assumptions (Li et al., 2020b; Karimireddy et al., 2020; Wang et al., 2020;
Khaled et al., 2020).
Statistical heterogeneity is the primary challenge: when local distributions differ across
clients, updates diverge causing client drift that degrades convergence (Zhao et al., 2018;
Hsu et al., 2019; Li et al., 2020a). Zhao et al. (2018) reported accuracy degradations
exceeding 50 percentage points on CIFAR-10 under pathological condi- tions. Hsu et al.
(2019) provided systematic characterisation using Dirichlet-based partitioning,
demonstrating that accuracy degradation varies continuously with the concentration
parameter. Data partitioning strategies themselves vary across the literature: fixed class
counts per client (McMahan et al., 2017), Dirichlet-based sam- pling (Hsu et al., 2019;
Yurochkin et al., 2019), natural user-based partitioning (Cal- das et al., 2019) and featurebased covariate shift (Li et al., 2022). We adopt both Dirichlet and class-restricted
partitioning to evaluate complementary aspects of het- erogeneity. Algorithmic
modifications address heterogeneity through varied mechanisms. Fed- Prox adds proximal
regularisation penalising deviation from the global model (Li et al., 2020b). SCAFFOLD
employs control variates correcting client drift (Karimireddy et al., 2020). FedNova
normalises updates by local computation (Wang et al., 2020). FedDyn maintains adaptive
regularisation (Acar et al., 2021). Adaptive server opti- misers such as FedAdam apply
momentum and adaptive learning rates to aggregated updates (Reddi et al., 2021). Despite
these advances, severe non-IID conditions re- main challenging under limited communication
budgets (Li et al., 2022). A gap in the existing literature is the lack of controlled multi-seed
comparisons across these al- gorithms under identical experimental conditions. Most papers
evaluate a proposed method against FedAvg alone, making cross-method comparison
difficult. We address this by evaluating FedAvg, FedProx and FedAdam under matched
configurations with
3

20 independent seeds.
2.2 Differential Privacy in Distributed Learning
Differential privacy provides a mathematically rigorous framework for bounding pri- vacy
leakage (Dwork et al., 2006; Dwork and Roth, 2014; Nissim et al., 2007). DP-SGD extends this
to deep learning through per-sample gradient clipping and calibrated Gaussian noise (Abadi
et al., 2016; Song et al., 2013; Bassily et al., 2014). Privacy composition is tracked via the
moments accountant or Rényi DP (Mironov, 2017; Bun and Steinke, 2016; Balle et al., 2020).
Federated extensions address client-level versus sample-level guarantees (Geyer et al.,
2017; McMahan et al., 2018; Wei et al., 2020; Noble et al., 2022; Cheng et al., 2022).
Secure aggregation provides complementary computational privacy (Bonawitz et al.,
2017; Bell et al., 2020; So et al., 2022). The Opacus library implements per-sample
gradients and automatic accounting for PyTorch (Yousefpour et al., 2021). Architectural
modifications including tempered sigmoid activations (Papernot et al., 2021) and
improved feature extractors (Tramèr and Boneh, 2021) can improve the privacy-utility
tradeoff.
2.3 Privacy Attacks on Machine Learning
Membership inference attacks determine whether specific records participated in training by
exploiting systematic differences in model behaviour on training versus non-training samples
(Shokri et al., 2017; Salem et al., 2019; Yeom et al., 2018; Carlini et al., 2022). Theoretical
analysis connects vulnerability to model overfitting and memorisation (Yeom et al., 2018;
Long et al., 2020). Humphries et al. (2020) showed that differential privacy does not
necessarily bound membership inference advantage in practice, motivating empirical
evaluation. Gradient-based reconstruction attacks recover training samples from gradient
updates (Zhu et al., 2019; Geiping et al., 2020; Zhao et al., 2020; Yin et al., 2021; Fowl et
al., 2022). The DLG attack formulates reconstruction as optimisation over dummy inputs
matched to observed gradients (Zhu et al., 2019). Reconstruction fidelity depends on batch
size, model architecture and adversary capabilities (Yin et al., 2021; Huang et al., 2021).
Defensive measures including gradient perturbation, secure aggregation and
representation-based protection can reduce fidelity (Sun et al., 2021; Scheliga et al., 2022;
Sotthiwat et al., 2021), though effectiveness against adaptive attackers remains an active
research question.
2.4 Relationship to Prior Empirical Studies
Prior work has benchmarked non-IID federated optimisation, privacy, or attacks separately.
This study's contribution is the matched combination: multiple optimisers, non-IID data,
privacy attacks, DP, repeated seeds, and explicit analysis of privacy dynamics across rounds.
3. Methods
3.1 Federated Objective
For K clients with local objectives F_k(w), the global objective is F(w)=sum_k p_k F_k(w),
where p_k is proportional to client data size. FedAvg performs local SGD and server-side
weighted averaging. FedProx adds a proximal penalty mu/2 ||w-w_t||^2 to each local
objective. FedAdam uses the aggregated model difference as a pseudo-gradient and applies
server-side Adam updates.
3.2 Non-IID Partitioning
CIFAR-10 uses Dirichlet label partitioning with concentration alpha. MNIST includes
class-restricted partitions and Dirichlet sweeps. Smaller alpha produces more severe label
heterogeneity.
3.3 Differential Privacy
DP-SGD clips per-sample gradients to maximum L2 norm C and adds calibrated Gaussian
noise. Each per-sample gradient g_i is clipped as ghat_i=g_i*min(1,C/||g_i||_2). Gaussian
noise is added to the averaged clipped gradient, with noise multiplier sigma. Cumulative
privacy expenditure is tracked using the RDP accountant and converted to (epsilon,delta)-DP
at delta=1e-5.
4. Experimental Methodology
4.1 Datasets and Preprocessing
Experiments use MNIST and CIFAR-10 with standard train/test partitions and normalisation.
CIFAR-10 training includes random horizontal flipping and random cropping with 4-pixel
padding. The model is a compact CNN without batch normalisation to avoid client-specific
batch-statistic complications.
4.2 Federated Configuration
The main CIFAR-10 configuration uses 50 clients, partial participation and Dirichlet
heterogeneity. MNIST uses 10 clients for the class-restricted setting. Principal FedAvg and
FedAdam comparisons use 20 seeds; some exploratory FedProx and DP conditions use fewer
seeds and are explicitly labelled.
5. Experimental Results
The main conclusions are that optimiser choice is context-dependent, severe non-IID data can
slow early convergence substantially, adaptive server optimisation helps under class-restricted
heterogeneity, and FedProx can act as a stability control even where it does not improve mean
accuracy.
5.5 Dirichlet Heterogeneity Robustness
Three observations follow. First, even severe heterogeneity (alpha=0.1) converges to
near-identical final accuracy given sufficient rounds on MNIST, though early-round
convergence is substantially delayed. Second, the convergence gap closes rapidly between
rounds 5 and 15. Third, the early-round divergence has privacy implications because the
high-gradient regime persists longer under stronger skew.
5.6 Client Participation Robustness
Performance differences across participation rates are small on MNIST once enough rounds
are available, consistent with random client sampling providing an unbiased estimate under
this controlled setting.
5.7 Communication Efficiency
Accuracy on MNIST saturates by about 20 rounds, with diminishing returns beyond that point.
5.8 Differentially Private Baseline
On CIFAR-10, DP-FedAvg with sigma=1.0 and C=1.0 produces a moderate accuracy penalty
relative to non-private FedAvg. The interaction between clipping and heterogeneity is a key
interpretive issue because clients with larger gradients may be affected disproportionately.
6. Privacy Attack Experiments
6.1 Gradient Norm Analysis
Mean L2 gradient norm declines sharply from the first communication rounds toward
convergence. Round 1 is about 4.82, round 10 about 2.14 and round 50 about 1.51, a roughly
3.2x decline. The sharpest fall is in the first 10 to 20 rounds.
6.2 Gradient Inversion Attack
A DLG-style gradient inversion attack is run with cosine-distance matching and total-variation
regularisation. Reconstruction fidelity is measured with MSE, PSNR and SSIM.
6.3 Membership Inference Attack
A confidence-threshold membership inference attack uses target-model confidence to
separate members and non-members.
6.4 Attack Results
At round 5, non-private FedAvg yields recognisable reconstructions around PSNR 12.8 dB and
SSIM 0.31. By round 50, PSNR falls to about 8.9 dB. DP at round 5 reduces reconstruction
fidelity substantially. Membership inference AUC is about 0.61 for FedAvg, about 0.59 for
FedProx and about 0.53 for DP-FedAvg, with 0.50 as random baseline. Both attack metrics
decline as training progresses and the steepest drop occurs in the first 20 rounds.
7. Discussion
The results show that privacy risk is phase-dependent. Early training combines large gradient
norms and rapid parameter movement with higher reconstruction and membership-inference
success. This motivates privacy mechanisms that account for training dynamics instead of
using a static protection level across all rounds.
