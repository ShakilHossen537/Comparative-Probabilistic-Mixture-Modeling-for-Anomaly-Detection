# Comparative-Probabilistic-Mixture-Modeling-for-Anomaly-Detection

**Gaussian vs Student-t mixture models for anomaly detection in high-dimensional data**

---

# Project Overview

This project is a controlled, reproducible comparison of Gaussian and non-Gaussian (Student-t) mixture models for unsupervised anomaly detection in high-dimensional data.

Mixture models detect anomalies by learning the probability density of normal data and flagging observations that have unusually low likelihood. Most existing work keeps Gaussian components, reports a narrow set of metrics, and works in low dimension. This project asks a more specific question:

> Does the choice of mixture-component distribution change anomaly detection reliability when the normal data are Gaussian, heavy-tailed or skewed, and how do dimension and training contamination affect that?

The study compares a full-covariance Gaussian mixture (GMM-F), a diagonal Gaussian mixture (GMM-D) and a Student-t mixture model (TMM, fitted by an ECM algorithm written from scratch) against Isolation Forest, Local Outlier Factor and One-Class SVM. It uses threshold-free and threshold-based metrics, runtime, and paired statistical tests over 10 random seeds. The study is unsupervised, needs no labels at training time, and is reproducible from one script that runs on a laptop or in Google Colab.

**Important:** This is a research prototype evaluated on **synthetic data only**. It is not a production security, fraud or safety system, and its results have not been validated on real-world datasets.

---

# Why This Research Matters

| Why it matters | Explanation |
|---|---|
| Anomalies are costly | Missed anomalies mean undetected cyberattacks, fraud, equipment failure or safety incidents |
| Labels are rare | Anomalies are scarce and expensive to label, so unsupervised density-based detection is practical |
| The Gaussian assumption is rarely checked | Most mixture detectors keep Gaussian components without testing whether the data justify it |
| Heavy tails cause false alarms | Legitimate extreme observations can get a very low probability and be flagged as anomalies |
| Contamination distorts the model | Hidden anomalies in the training data shift Gaussian mean and covariance estimates |
| High dimension is hard | Covariance estimation, sparsity and diluted signals degrade detection as features grow |
| Accuracy is misleading | With rare anomalies, a detector that flags nothing still scores high accuracy, so AUPRC, F1 and false-positive rate are reported |
| Honest evidence is needed | The project states what the experiments do and do not support |

---

# Research Objective

- Compare Gaussian and Student-t mixture models for anomaly detection under controlled conditions.
- Measure the effect of the normal-data distribution (Gaussian, heavy-tailed, skewed).
- Measure the effect of dimension (2 to 100) and training contamination (0% to 10%).
- Compare density models with Isolation Forest, LOF and One-Class SVM.
- Evaluate with AUROC, AUPRC, precision, recall, F1, false-positive rate and runtime.
- Select the detection threshold without test labels.
- Test differences with paired Wilcoxon tests across 10 seeds.
- Implement the Student-t mixture (ECM) from scratch with a verified monotone likelihood.



# Business / Real-World Objective

The approach is motivated by settings where anomalies are rare, labels are unavailable, and normal data may be heavy-tailed or contaminated, including:

- Cybersecurity and network intrusion detection
- Fraud detection in financial transactions
- Industrial sensor monitoring and predictive maintenance
- Structural health monitoring
- Healthcare data screening
- Manufacturing quality control
- IoT and cyber-physical systems

The goal is to help choose the right density model, covariance structure and evaluation protocol. Suitability for any real application would require validation on that application's own data.

---

# Project Workflow

# 1. Data Generation (Synthetic) and Optional Real Data

The experiments use a controlled synthetic generator. Normal data follow a three-component mixture in $d$ dimensions. Component means are drawn from $N(0, 1.5^2 I)$ and covariances are $\Sigma_k = B_k B_k^{\top} + I$, where $B_k$ is a $d \times \min(3, d)$ matrix with $N(0, 0.5^2)$ entries.

Within each component, the noise is one of:

| Condition | Within-component noise |
|---|---|
| Gaussian | standard normal |
| Heavy-tailed | multivariate Student-t, 3 degrees of freedom |
| Skewed | standardized Gamma (shape 2, skewness about 1.41) |

**Anomalies** are generated from the same mixture and shifted by 3 within-component standard deviations in 5 randomly chosen coordinates, with random signs.

| Setting | Value |
|---|---|
| Training set size | 2,000 points |
| Training contamination | 0%, 1%, 5%, 10% |
| Test set | 1,000 normal + 100 anomalies (9.1% anomalous) |
| Validation set (threshold) | 500 unlabeled points from the training distribution |
| Dimensions | 2, 5, 10, 20, 50, 100 |
| Random seeds | 10 (each seed draws new parameters and data) |

A separate runner accepts any labeled CSV for optional real-dataset experiments. **No real-dataset results are reported in the paper.** If you use public datasets, obtain them from their official sources and follow their terms of use. Do not upload restricted data to GitHub.

## 2. Preprocessing

- Standardize each feature, $z_j = (x_j - \mu_j) / \sigma_j$, using statistics from the **training data only**.
- Apply the same transformation unchanged to validation and test data.
- For real data: one-hot encode categorical columns, impute missing values, drop constant columns, and optionally reduce dimension with PCA fitted on training data.

### 3. Model Fitting

Six detectors are compared under identical data, preprocessing, scoring and thresholding:

| Model | Description |
|---|---|
| **GMM-F** | Full-covariance Gaussian mixture (scikit-learn, K = 3) |
| **GMM-D** | Diagonal-covariance Gaussian mixture |
| **TMM** | Student-t mixture, full covariance, ECM written from scratch (K = 3, ν = 4) |
| **IF** | Isolation Forest (200 trees) |
| **LOF** | Local Outlier Factor (k = 20, novelty mode) |
| **OCSVM** | One-Class SVM (RBF kernel, gamma = scale, ν = 0.05) |

### 4. Anomaly Scoring and Thresholding

- **Score:** negative log-likelihood, $S(x) = -\log p(x)$ (larger means more anomalous). Isolation Forest, LOF and One-Class SVM use their native scores, negated.
- **Threshold:** the 95th percentile of validation scores. No test labels are used.

$$
\hat{y} = \begin{cases} 1 & S(x) > \tau \\ 0 & S(x) \le \tau \end{cases}
$$

### 5. Statistical Testing

For each condition, the per-seed AUPRC differences between TMM and GMM-F are compared with a paired Wilcoxon signed-rank test. With $n = 10$ seeds:

$$
p_{\min} = \frac{2}{2^{n}} \approx 0.002, \qquad \alpha_{\text{Bonferroni}} = \frac{0.05}{27} \approx 0.0019
$$

The smallest attainable p-value is above the Bonferroni threshold, so **no single comparison can pass a multiple-comparison correction with 10 seeds**. Reported p-values are suggestive, not confirmatory.

---

## Mixture Modeling Framework

### Mixture Density

$$
p(x) = \sum_{k=1}^{K} \pi_k \, f(x \mid \theta_k), \qquad \pi_k \ge 0, \quad \sum_{k=1}^{K} \pi_k = 1
$$

### Gaussian Mixture (EM)

E-step (responsibilities):

$$
\gamma_{ik} = \frac{\pi_k \, N(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, N(x_i \mid \mu_j, \Sigma_j)}
$$

M-step:

$$
N_k = \sum_i \gamma_{ik}, \qquad \mu_k = \frac{1}{N_k} \sum_i \gamma_{ik} x_i, \qquad \pi_k = \frac{N_k}{N}
$$

$$
\Sigma_k = \frac{1}{N_k} \sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^{\top} + \epsilon I
$$

### Student-t Mixture (ECM)

The multivariate Student-t density has heavier tails than the Gaussian:

$$
t_{\nu}(x \mid \mu, \Sigma) = \frac{\Gamma\left(\frac{\nu+d}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) (\nu \pi)^{d/2} |\Sigma|^{1/2}} \left(1 + \frac{\delta}{\nu}\right)^{-\frac{\nu+d}{2}}, \qquad \delta = (x-\mu)^{\top} \Sigma^{-1} (x-\mu)
$$

A t component is a Gaussian scale mixture, so each point receives a latent weight that shrinks as it moves away from the component:

$$
u_{ik} = \frac{\nu + d}{\nu + \delta_{ik}}, \qquad \delta_{ik} = (x_i - \mu_k)^{\top} \Sigma_k^{-1} (x_i - \mu_k)
$$

The M-step uses the effective weights $\gamma_{ik} u_{ik}$:

$$
\mu_k = \frac{\sum_i \gamma_{ik} u_{ik} x_i}{\sum_i \gamma_{ik} u_{ik}}, \qquad
\Sigma_k = \frac{1}{N_k} \sum_i \gamma_{ik} u_{ik} (x_i - \mu_k)(x_i - \mu_k)^{\top} + \epsilon I
$$

Points far from a component receive a small $u_{ik}$, which is the mechanism intended to resist contaminated training data. As $\nu \to \infty$ the component approaches a Gaussian.

### Model Complexity

Free parameters of a K-component mixture in d dimensions:

$$
\text{Full covariance:} \quad K \left[ \frac{d(d+1)}{2} + d \right] + K - 1, \qquad \text{Diagonal covariance:} \quad 2Kd + K - 1
$$

For K = 3 and d = 100, the full-covariance model has 15,452 parameters estimated from 2,000 training points, against 602 for the diagonal model.

**Implementation note:** The GMMs use scikit-learn. The Student-t mixture is a from-scratch NumPy implementation of the ECM algorithm above, with fixed ν = 4 and K = 3. Selection of K by BIC is implemented in the real-data runner only; the synthetic experiments in the paper fix K = 3. Mixtures of probabilistic PCA or factor analyzers, noise-component mixtures, random-subspace ensembles, stray and DAGMM are **not implemented**.

---

## System Architecture

The conceptual workflow of the study is:

```
      Synthetic data generator               Real labeled CSV (optional)
 (Gaussian / heavy-tailed / skewed;
  d = 2 to 100; contamination 0-10%)
                  |
                  v
      Split: train / validation / test
                  |
                  v
    Standardize (fit on training data only)
                  |
      +-----------+-----------+------------------+
      |           |           |                  |
      v           v           v                  v
    GMM-F       GMM-D        TMM          IF / LOF / OCSVM
  (sklearn)    (sklearn)  (ECM, NumPy)     (sklearn baselines)
      |           |           |                  |
      +-----------+-----------+------------------+
                  |
                  v
   Anomaly score S(x) = -log p(x)
   (native score for IF / LOF / OCSVM)
                  |
         +--------+--------+
         |                 |
         v                 v
   Ranking metrics    Threshold tau
   AUROC / AUPRC      (95th percentile of validation scores)
         |                 |
         |                 v
         |        Precision / Recall / F1 / FPR
         +--------+--------+
                  |
                  v
    Paired Wilcoxon tests across 10 seeds
                  |
                  v
     Tables (.csv .md .tex) and figures
```

<p align="center"><img src="assets/figures/fig1_framework.png" width="55%" alt="Framework"></p>
<p align="center"><img src="assets/figures/fig2_backbone.png" width="100%" alt="Backbone architecture"></p>

The diagram is a conceptual representation of the study. Update it if the final implementation uses a different execution path.

---

## Evaluation Metrics

### Detection Metrics

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \qquad \text{FPR} = \frac{FP}{FP + TN}
$$

- **AUROC:** the probability that a random anomaly scores higher than a random normal point.
- **AUPRC:** the area under the precision-recall curve. A random scorer achieves the anomaly rate, here $100/1100 \approx 0.091$, so AUPRC is the main metric for rare anomalies.
- **Accuracy** is not used as a primary metric because anomaly detection is strongly imbalanced.
- **Realized false-positive rate** is reported because the validation threshold targets a nominal 5% rate.

### Efficiency Metrics

- **Runtime:** wall-clock time to fit the model and score the test and validation sets.
- **Parameter count** of full and diagonal covariance models (formula above).

Runtime values should be accompanied by the machine and software versions, since they vary with hardware.

### Statistical Metrics

- Mean and standard deviation over 10 seeds
- Paired Wilcoxon signed-rank p-values and win counts, TMM versus GMM-F

---

## Results & Analysis

All values are means over 10 seeds from this repository's code. Best value per row in **bold**. The full tables are generated in `results/`.

### AUPRC versus training contamination (d = 20)

| Data | Contam. | GMM-F | GMM-D | TMM | IF | LOF | OCSVM |
|---|---|---|---|---|---|---|---|
| Gaussian | 10% | 0.998 | 0.947 | **0.999** | 0.789 | 0.997 | 0.930 |
| Heavy-tailed | 10% | 0.276 | 0.226 | 0.284 | 0.215 | **0.295** | 0.241 |
| Skewed | 0% | 0.973 | 0.801 | 0.970 | 0.591 | **0.975** | 0.907 |
| Skewed | 10% | 0.942 | 0.791 | **0.972** | 0.686 | 0.967 | 0.817 |

<p align="center"><img src="assets/figures/fig3_contamination.png" width="100%" alt="AUPRC vs contamination"></p>

### AUPRC versus dimension (5% contamination)

| Data | d | GMM-F | GMM-D | TMM | IF | LOF | OCSVM |
|---|---|---|---|---|---|---|---|
| Gaussian | 2 | 0.842 | 0.850 | **0.852** | 0.845 | 0.718 | 0.607 |
| Gaussian | 100 | 0.976 | 0.323 | **0.977** | 0.188 | 0.879 | 0.389 |
| Heavy-tailed | 100 | 0.135 | 0.115 | **0.136** | 0.116 | **0.136** | 0.122 |
| Skewed | 100 | 0.798 | 0.262 | **0.799** | 0.172 | 0.666 | 0.330 |

<p align="center"><img src="assets/figures/fig4_dimension.png" width="100%" alt="AUPRC vs dimension"></p>

### Threshold-based performance (d = 20, 5% contamination, Gaussian data)

| Model | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| GMM-F | 0.981 | 0.990 | **0.986** | 0.002 |
| TMM | 0.979 | 0.992 | 0.985 | 0.002 |
| LOF | 0.976 | 0.966 | 0.970 | 0.003 |
| OCSVM | 0.900 | 0.867 | 0.882 | 0.010 |
| IF | 0.771 | 0.629 | 0.687 | 0.019 |

<p align="center"><img src="assets/figures/fig5_f1_dimension.png" width="100%" alt="F1 vs dimension"></p>

### Paired comparison, TMM minus GMM-F (AUPRC)

| Condition | Mean diff. | Seeds TMM wins | p (Wilcoxon) |
|---|---|---|---|
| Skewed, d = 20, 10% contam. | +0.030 | 10/10 | 0.002 |
| Heavy, d = 10, 5% contam. | +0.029 | 10/10 | 0.002 |
| Heavy, d = 5, 5% contam. | +0.013 | 9/10 | 0.004 |
| Skewed, d = 20, 0% contam. | -0.003 | 1/10 | 0.010 |
| Gaussian, d = 20, any contam. | at most +0.001 | 2 to 6/10 | 0.055 to 0.625 |

<p align="center"><img src="assets/figures/fig6_paired_difference.png" width="100%" alt="Paired difference"></p>

### Runtime (seconds, fit plus scoring, 5% contamination)

| d | GMM-F | GMM-D | TMM | IF | LOF | OCSVM |
|---|---|---|---|---|---|---|
| 2 | 0.052 | 0.032 | 0.126 | 0.242 | 0.016 | 0.021 |
| 100 | 0.123 | 0.037 | 0.770 | 0.233 | 0.065 | 0.039 |

These runtime values are from one machine and should not be treated as independently reproduced. They depend on hardware and implementation: the TMM is NumPy code, compared with scikit-learn's compiled routines.

### Important Interpretation

- The full-covariance GMM and the TMM performed **nearly identically** in most conditions.
- The clearest Student-t advantage is on skewed data with 10% training contamination (AUPRC 0.972 vs 0.942). On heavy-tailed data the TMM gain is small (0.003 to 0.008 at d = 20) and does not grow with contamination.
- No single paired test survives a multiple-comparison correction with 10 seeds, so conclusions are preliminary.
- All models struggle on heavy-tailed data at high dimension (AUPRC about 0.135 at d = 100, only slightly above the 0.091 random level).
- The explanations offered for these trends are plausible but were not tested directly.
- These results come from synthetic data and do not establish performance on any real application.

---

## Key Findings

### Gaussian vs Student-t Mixtures
The two performed nearly identically in most conditions. The Student-t mixture helped mainly under skewed data with heavy training contamination, and was slightly worse on clean skewed data (-0.003), a small cost for robustness.

### Covariance Structure Matters More Than Component Family
Diagonal GMM, Isolation Forest and One-Class SVM deteriorated sharply as dimension grew (Gaussian d = 100: AUPRC 0.323, 0.188 and 0.389), while full-covariance mixtures stayed high (0.976 and 0.977) and LOF declined more gently (0.879).

### LOF Is a Strong Baseline
LOF matched or slightly beat the mixtures at low to moderate dimension on heavy-tailed and skewed data, but fell behind at d = 50 and 100.

### Thresholds Change What Users See
Realized false-positive rates were mostly below the nominal 5% (mean 0.026 to 0.035) because the validation set itself contains contamination. On heavy-tailed data, where anomalies are hard to separate, rates were close to 5%.

### Practical Guidance

| Situation | Suggested choice |
|---|---|
| Clean, approximately Gaussian normal data | Full-covariance GMM (efficient and interpretable) |
| Skewed normal data, or training data possibly contaminated by several percent | Student-t mixture (robustness at small cost) |
| High-dimensional data | Model feature correlations (full or structured covariance); avoid diagonal covariance and default Isolation Forest or One-Class SVM settings |
| Any problem | Include LOF as a baseline; report AUPRC with threshold-based metrics |

---

## Tools & Technologies

- **Python:** main programming language
- **NumPy / SciPy:** from-scratch Student-t ECM, log-sum-exp, Wilcoxon tests
- **scikit-learn:** Gaussian mixtures, Isolation Forest, LOF, One-Class SVM, preprocessing, metrics
- **pandas:** numerical processing and result tables
- **Matplotlib:** figures
- **Expectation-Maximization / ECM:** parameter estimation
- **Negative log-likelihood:** anomaly score
- **Wilcoxon signed-rank test:** paired statistical comparison
- **Google Colab / Jupyter:** notebook execution
- **GitHub:** source code and documentation

Package versions are not pinned in `requirements.txt`; record the versions you use when reporting results.

---

## Key Deliverables

Implemented in this repository:

- From-scratch Student-t mixture model (ECM) with a verified monotone likelihood
- Controlled synthetic data generator (Gaussian, heavy-tailed, skewed)
- Six-detector comparison pipeline (GMM-F, GMM-D, TMM, IF, LOF, OCSVM)
- Threshold selection without test labels
- AUROC, AUPRC, precision, recall, F1, FPR and runtime evaluation
- Paired statistical testing across seeds
- Automatic generation of all tables (.csv, .md, .tex) and figures
- Real-dataset runner for any labeled CSV (including BIC selection of K)
- Google Colab notebook
- Research paper and documentation

Not implemented (listed under Future Scope): mixtures of PPCA or factor analyzers, noise-component mixtures, random-subspace ensembles, stray, DAGMM, alternative scores, and real-dataset results.

---

## Project Structure

```
Comparative-Probabilistic-Mixture-Modeling-for-Anomaly-Detection/
├── README.md
├── requirements.txt
├── mixture_anomaly_full.py        models, experiments, tables, figures, real-data runner
├── notebooks/
│   └── Mixture_Anomaly_Colab.ipynb
├── assets/
│   └── figures/                   figures shown in this README
├── paper/
│   └── Comparative_Probabilistic_Mixture_Modeling.pdf
└── results/                       created when you run the script
    ├── results.csv                every run (model, seed, metrics)
    ├── table_*.csv / .md / .tex
    └── fig1 ... fig6 (.png)
```

Avoid committing large datasets or restricted data. The `results/` folder is generated and may be added to `.gitignore`.

---

## Installation & Usage

### Install

```bash
pip install -r requirements.txt
```

### Run everything (experiments, tables and figures; about 4 to 6 minutes)

```bash
python mixture_anomaly_full.py all --seeds 10
```

Output is saved in the `results/` folder.

### Other commands

```bash
python mixture_anomaly_full.py synthetic --seeds 10      # experiments only
python mixture_anomaly_full.py report                    # tables and figures from results/results.csv
python mixture_anomaly_full.py real --demo breast_cancer # quick test on a built-in dataset
python mixture_anomaly_full.py real --csv data.csv --label-col label --anomaly-value 1 --protocol semi
```

For `real`: `--protocol semi` trains on normal rows only, `--protocol unsup` trains on all rows; `--pca 20` reduces dimension.

### Run in Google Colab

Upload `notebooks/Mixture_Anomaly_Colab.ipynb` and choose **Runtime, Run all**. Inside a notebook, call `main(['all', '--seeds', '10'])` after running the script cell.

### Reproducibility

- Randomness is controlled by seeds; each seed draws new model parameters and data.
- `--seeds 10` regenerates every table and figure of the paper.
- Preprocessing is fitted on training data only; baselines use fixed settings with no label-based tuning.
- Document the device, software versions and run date when reporting runtime.

---

## Limitations

- **Synthetic data only:** no real-dataset results are reported.
- **Fixed settings:** K fixed at the true value (3) and Student-t degrees of freedom fixed at ν = 4, which favors the TMM on heavy-tailed data generated with ν = 3.
- **One anomaly type and one sample size:** shifts in five coordinates; 2,000 training points.
- **Untested components:** structured-covariance mixtures, noise-component mixtures, random-subspace ensembles, stray, DAGMM, alternative scores and other threshold strategies.
- **Statistical power:** with 10 seeds, no single paired test can pass a multiple-comparison correction.
- **Mechanisms not tested:** explanations for the observed trends are plausible but unverified.
- **Runtime comparisons** mix NumPy code with optimized library code and depend on the machine.
- **Literature coverage:** some related papers could not be read in full text, so the review is not exhaustive.

---

## Future Scope

- Mixtures of probabilistic PCA and factor analyzers to test structured covariance (H2).
- Noise-component mixtures and random-subspace ensembles; stray and DAGMM as additional comparisons.
- Estimate ν and select K by BIC, AIC or ICL, including K = 1 and over-specified K (H4).
- Alternative scores (nearest-component Mahalanobis, posterior-based) and threshold strategies (H3).
- More anomaly types (local-density, between-cluster, group), other anomaly ratios and tail indices, and more seeds for confirmatory statistics.
- Validate on real data: cybersecurity (UNSW-NB15, NSL-KDD), fraud (credit card), industrial sensing (SKAB) and ADBench.
- Skew-normal, skew-t and generalized hyperbolic components; adaptive choice of the component distribution.
- Online EM for streaming data and concept drift.
- Feature-level explanations of anomaly scores.
- Document checkpoints, random seeds, preprocessing and software versions for all reported results.

---

## Research Paper

**Title:** Comparative Probabilistic Mixture Modeling for Anomaly Detection in High-Dimensional Data: Gaussian and Non-Gaussian Approaches

**Research Area:** Machine Learning, Probabilistic Modeling, Anomaly Detection, Statistical Learning, Unsupervised Learning



---

## Authors

Md Shakil Hossen
Department of Computer Science and Engineering (ME CSE, Data Science), Chandigarh University, Mohali, Punjab, India

---

## Keywords

Anomaly Detection, Gaussian Mixture Model, Student-t Mixture Model, Expectation-Maximization, Heavy-Tailed Distribution, High-Dimensional Data, Probabilistic Modeling, Unsupervised Learning, Outlier Detection, AUPRC, Isolation Forest, Local Outlier Factor, One-Class SVM

---

## Responsible Use

This repository is intended for research and educational use. It must not be used as the sole basis for security, fraud, safety or other consequential decisions. Detectors can produce false alarms and miss real anomalies, and a low density score does not prove that an observation is malicious or faulty. Any real deployment requires validation on the target data, human review of alerts, and appropriate oversight.

---

## Citation

If you use this project in academic work, cite the paper. Add the complete citation here once the publication details are confirmed. Clearly distinguish the paper's reported results from results reproduced with your own code, data splits and hardware.

```
Md Shakil Hossen, "Comparative Probabilistic Mixture Modeling for Anomaly Detection in
High-Dimensional Data: Gaussian and Non-Gaussian Approaches," Chandigarh University, 2026.
```

**License:** add a `LICENSE` file of your choice (for example MIT) in the repository root.
