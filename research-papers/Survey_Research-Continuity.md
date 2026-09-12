# Survey Research Continuity Copy

## Privacy-Preserving Federated Learning Image Classification under Non-IID Data: A Technical Survey of Threats, Defences, Optimisation and Evaluation and Their Interaction

This reading copy records the technical taxonomy, review protocol and gaps that matter for the RAPSA-FL novelty audit. The source is the tracked-change Survey V6 manuscript.

## Survey scope

The survey examines privacy-preserving horizontal federated learning for image classification under non-IID client data. Its core premise is that two issues are usually treated separately:

1. statistical heterogeneity across clients
2. privacy leakage through shared model updates

The survey treats privacy-preserving federated learning as federated learning plus an explicit privacy mechanism, such as differential privacy or secure aggregation, that limits what an adversary can infer from shared updates.

The image-classification setting is used because reconstructed gradients can reveal the sensitive training image itself, while label-skew simulations on MNIST and CIFAR-style benchmarks dominate the heterogeneity literature.

## Review protocol

Candidate studies were compiled from IEEE Xplore, ACM Digital Library, Scopus, Web of Science, SpringerLink and ScienceDirect, with reference chaining from foundational federated learning, differential privacy and privacy-attack studies.

The protocol screened:

- 104 identified records
- 96 records after duplicate removal
- 96 title/abstract screens
- 58 full-text records
- 41 primary studies included in the scored synthesis

Twenty-one additional works were cited for context, including regulation, privacy-accounting theory, dataset descriptions, tooling and prior surveys.

## Five-dimensional taxonomy

Each primary study is analysed across five linked dimensions:

1. **Threat model**
2. **Privacy mechanism**
3. **Data heterogeneity**
4. **Federated optimisation**
5. **Evaluation**

The taxonomy is important for RAPSA-FL because the new paper must occupy a genuinely under-examined joint region of the design space, not merely combine mechanisms already studied individually.

## Threat model taxonomy

The survey distinguishes adversaries by both position and goal.

Relevant adversaries include:

- honest-but-curious server
- malicious or curious client
- external observer on the update channel
- colluding participants
- active server that deliberately changes the protocol to amplify leakage

Privacy attacks include:

- membership inference
- gradient inversion and reconstruction
- model inversion across observed model states
- property or attribute inference
- malicious-server amplification

The survey explicitly separates confidentiality/privacy attacks from integrity/security attacks. Differential privacy and secure aggregation protect confidentiality. Robust aggregation protects integrity. They do not substitute for one another.

This distinction is central to RAPSA-FL.

## Differential privacy

The survey distinguishes privacy by the unit protected:

- **record/sample-level DP** protects an individual training record
- **client/participant-level DP** protects the participation of an entire client

DP-SGD uses per-sample clipping plus calibrated Gaussian noise, with cumulative expenditure tracked through privacy accounting such as Renyi Differential Privacy.

Smaller epsilon means stronger formal protection, with utility cost depending on the setting.

Paper III in the research programme already uses record/example-level DP. RAPSA-FL therefore cannot claim record-level DP itself as a contribution.

## Secure aggregation

Secure aggregation is a cryptographic mechanism. Clients mask individual contributions so that the server can recover the aggregate but cannot inspect a single plaintext client update.

The survey stresses three points:

1. secure aggregation does not perturb the update in the way DP does
2. it prevents the server from reading individual client updates
3. it does not stop inference from the released aggregate/global model, so it is complementary to DP

This creates an important systems tension for the new paper. Many Byzantine or anomaly-resistant aggregation rules require access to individual updates. Secure aggregation deliberately hides those updates.

A valid RAPSA-FL robustness design must therefore remain compatible with the secure-aggregation threat model. A method that decrypts all individual updates before applying a robust aggregator cannot be described as preserving the same secure-aggregation guarantee.

## Other privacy mechanisms covered

The survey also discusses:

- local DP
- central DP
- homomorphic encryption
- secure multiparty computation
- clipping and noise injection
- representation-level defences such as Soteria and PRECODE

The manuscript treats cryptographic privacy and perturbation privacy as complementary mechanism families with different compute, communication and utility costs.

## Non-IID taxonomy

The survey distinguishes:

- label-distribution skew
- feature-distribution skew
- quantity skew
- concept shift
- participation imbalance
- temporal drift

Common simulation strategies include:

- Dirichlet partitioning with concentration parameter alpha
- class-restricted or shard partitioning
- quantity-based partitioning
- naturally partitioned user-level datasets

Low Dirichlet alpha gives more severe concentration/skew. Results at one alpha or one partition scheme do not automatically transfer to another.

The survey also identifies a measurement issue: label-skew partitioning can manufacture severe local class imbalance while studies still report only global top-1 accuracy. The new study should therefore include metrics that remain informative under class imbalance where applicable.

## Optimisation families

The survey covers the role of optimiser choice under heterogeneous data, including:

- FedAvg
- FedProx
- FedAdam / adaptive server optimisation
- SCAFFOLD and drift-correction methods

The key programme-level point is that optimiser behaviour, heterogeneity and privacy exposure interact. Paper I then provides empirical evidence that the privacy exposure itself also changes over training time.

## Main gaps identified by the survey

The scored corpus supports several gaps relevant to the new paper:

- studies that measure privacy empirically often use IID data
- studies that model non-IID heterogeneity carefully often do not apply an explicit privacy mechanism
- matched comparisons of several optimisers under identical non-IID conditions with repeated seeds and joint privacy/utility reporting are scarce
- privacy is usually reported as a static final budget rather than as exposure that changes over training
- evaluation often under-reports class-sensitive performance under locally imbalanced partitions
- privacy accounting and empirical attack testing are often not reported together

The survey therefore motivates controlled studies that jointly report formal privacy, empirical attack exposure, predictive utility and heterogeneous training behaviour.

## Direct boundary for RAPSA-FL

The new paper must not claim as novel:

- federated learning itself
- DP itself
- secure aggregation itself
- FedAvg/FedProx/FedAdam comparisons
- non-IID partitioning
- membership inference
- gradient inversion
- the observation that secure aggregation and DP are complementary
- a simple front-loaded DP schedule

The strongest defensible architecture question is how to orchestrate privacy protection, secure aggregation and resilience when privacy risk changes during training and the server cannot inspect individual plaintext client updates.

## Evaluation implication

The new architecture should report, under matched conditions:

- predictive utility
- formal epsilon/delta accounting
- empirical membership-inference risk
- reconstruction risk only under an attacker observation model that actually exposes the required update
- secure-aggregation compute and communication overhead
- client dropout tolerance
- malicious-client resilience
- non-IID sensitivity
- repeated-seed uncertainty and effect sizes

Formal privacy accounting and empirical attack measurement should be treated as different evidence. One does not replace the other.

## Integrity rule

Where the tracked-change survey contains unresolved editorial alternatives, RAPSA-FL should use the technical conclusion only after checking the surrounding context. Do not treat deleted or superseded tracked-change wording as a definitive published claim.
