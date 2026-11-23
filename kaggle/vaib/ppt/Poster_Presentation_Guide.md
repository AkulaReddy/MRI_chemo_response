---
title: "Poster presentation guide: Predicting Chemotherapy Response from Pre-treatment Breast MRI"
subtitle: "What each section of the poster says, and what to tell visitors"
author: "Chinta Phani Rama Vaibhav (2025201067) · Rohit Rao (2025201016)"
geometry: margin=2cm
fontsize: 10pt
mainfont: Helvetica Neue
colorlinks: true
---

# The 30-second pitch (say this first)

> "Breast cancer patients often get chemotherapy before surgery, but only about 30% respond completely, and we
> only find out after months. We asked: can the MRI taken *before* treatment, together with routine clinical data,
> predict who will respond? We used 1,452 patients from three hospitals. The MRI alone is weak, below what the
> tumour's receptor subtype already tells us. But combined with clinical data, our self-supervised 3D MRI model
> reached an AUC of 0.71, up from 0.65, with the biggest gain in HER2-positive tumours. Cross-validation says the
> real gain is small, so we report it honestly."

**Walk the poster in this order:** title $\rightarrow$ Problem (left, top) $\rightarrow$ Our Approach (left, bottom) $\rightarrow$ Our Solution
(centre, top, the diagram) $\rightarrow$ How It Works $\rightarrow$ What We Learned (results chart) $\rightarrow$ Impact (right, with the subtype
chart).

**Three numbers to remember:** subtype lookup **0.662** · clinical model **0.652** · clinical + 3D MRI **0.713**.

---

# 1. Title bar

**What it says.** *Predicting Chemotherapy Response from Pre-treatment Breast MRI*, plus a one-line description:
can an MRI taken before chemotherapy, with routine clinical data, predict a pathologic complete response?

**Explain.**

- **pCR (pathologic complete response):** no invasive cancer left in the tissue removed at surgery. It is the best
  outcome and is linked to better long-term survival.
- **Pre-treatment:** we use only the scan taken *before* chemotherapy starts, not later scans.
- **DCE-MRI:** an MRI taken before and after injecting a contrast agent; tumours light up because their blood
  vessels are leaky.

---

# 2. The Problem & Context (left, top)

**What it says.** About 30% of patients reach pCR; response is known only after months; receptor subtype is coarse;
MRI models built at one hospital fail at another.

**Explain.**

- Patients who will not respond still go through months of toxic treatment. An early estimate could support
  treatment discussions and closer monitoring.
- Today, oncologists mainly use **receptor subtype**: HR (hormone receptor) and HER2 status. It is useful but coarse:
  two HER2+ patients can respond very differently.
- **The gap:** most published imaging models are trained and tested at one hospital. Different scanners produce
  different images, so these models often fail elsewhere. That is why we use **three cohorts** (I-SPY1, I-SPY2,
  Duke).

**If asked "why not just use the subtype?"** That is exactly our baseline. The question is whether the MRI adds
anything *beyond* subtype.

---

# 3. Our Approach (left, bottom): how we arrived at the solution

**What it says.** An honest baseline first, then simple to complex: understanding the data, challenges, the
alternatives we tried, why we chose 3D self-supervised learning, and how we tested.

**Explain, point by point.**

1. **Baseline to beat: the subtype lookup.** Each patient simply gets the pCR rate of their subtype in the training
   data (HR+/HER2− 14%, HER2+ 43%, triple-negative 38%). This "3-row table" scores **AUC 0.662**. Any model must beat
   it to be useful.
2. **Understanding the data.** Our first image model did not learn at all (AUC 0.53). Analysing the inputs, we found
   three problems: the field of view ranged from 70 to 315 mm (same tumour at very different sizes), the bright
   enhancing tumour was clipped (saturated), and many slices contained no tumour. We rebuilt the pipeline at one
   physical scale, 0.7 mm per pixel.
3. **Challenges.** Only 1,452 labelled patients; three sites with different scanners; only the pre-treatment scan;
   no grade, Ki-67 or chemotherapy regimen in the data.
4. **Alternatives we built and compared** (all on the same patients):
   - a **clinical model** (age, race, menopause, HR, HER2);
   - **radiomics:** hand-crafted features (enhancement, 2D/3D shape and texture, rings of tissue around the tumour,
     tumour sub-regions);
   - **ComBat harmonisation** (removes scanner differences from features) and **per-slice normalisation** (the
     professor's suggestions);
   - **2D deep learning:** ViT, DINOv2, ResNet-18, MAE, more than 30 runs, and a slice transformer;
   - a **3D network pretrained without labels (SimCLR)**.
5. **Why 3D self-supervised.** The 2D networks memorised the training patients and were at chance on Duke.
   Self-supervision lets us learn from all 1,798 scans, including 618 without outcome labels, and it clearly beat the
   same network with random weights.
6. **Honest testing.** We rank models by repeated cross-validation on 1,098 training patients, confirm on 177
   validation patients with paired statistical tests (DeLong), and keep the 176 test patients untouched.

**Key terms if asked.**

- **AUC:** the chance that the model ranks a responder above a non-responder. 0.5 = coin flip, 1.0 = perfect. We
  use it instead of accuracy because only 30% respond (always saying "no response" would already be 70% accurate).
- **Cross-validation:** split the training patients into 5 parts, train on 4, test on the 5th, rotate. Every
  patient is tested by a model that never saw them.

---

# 4. Our Solution (centre, top) and the pipeline diagram

**What it says.** A model that estimates each patient's probability of pCR from the pre-treatment MRI and routine
clinical data. Users: oncologists and radiologists, as decision support (not a diagnosis).

**Walk the diagram left to right.**

1. **Data:** BreastDCEDL, three cohorts, 1,452 labelled patients plus 618 unlabelled scans; three MRI phases and
   clinical data.
2. **Pre-processing:** same 0.7 mm scale for everyone; tumour region found from contrast enhancement; kinetic maps;
   a 3D crop scaled to the tumour.
3. **Self-supervised 3D:** a 3D network learns from the scans with no labels.
4. **Add clinical data:** image features + age, menopause, HR, HER2.
5. **Output and testing:** a probability of pCR, checked with cross-validation and validation.

**Bottom strip of the diagram:** everything else we built and compared, and the reference to beat (0.662).

**Explain "addresses the problem":** trained on three sites at once; kinetic maps that are largely independent of
the scanner; learns from scans that have no outcome label.

---

# 5. How It Works (centre, bottom left)

**Explain the five steps in plain words.**

1. **Input:** the three MRI phases (before contrast, early and late after contrast), the tumour location, and the
   clinical data.
2. **Pre-processing and kinetic maps:** instead of raw brightness (which depends on the scanner), we compute
   ratios for every voxel:
   - **percent enhancement** = how strongly the tissue takes up contrast;
   - **washout** = whether the signal drops, stays flat or keeps rising afterwards (dropping is typical of
     aggressive tumours).

   Ratios cancel out the scanner's brightness setting. The tumour is cropped in 3D and scaled so small and large
   tumours fill the input equally.
3. **Self-supervised pretraining (SimCLR):** the network sees two randomly rotated or zoomed versions of the same
   tumour and learns to give them similar descriptions, and different tumours different ones. No pCR labels are
   needed, so all scans can be used.
4. **Prediction:** the image features and clinical data go into a logistic regression, or into a fine-tuned network
   where the clinical data adjusts the image features (**FiLM**). Output: a probability of pCR.
5. **Evaluation:** 5 × 5-fold cross-validation on 1,098 patients, 177 validation patients, paired DeLong tests
   against the subtype lookup.

**Technology:** Python, PyTorch, scikit-learn, SimpleITK, DINOv2/ViT, SimCLR, ComBat; Kaggle GPUs.

---

# 6. What We Learned (centre, bottom right) and the results chart

**How to read the chart.** Each row is a model. The **dot** is the validation AUC (177 patients) with its 95%
confidence interval (the line); the **diamond** is the cross-validation AUC (1,098 patients). The **dashed line**
is the subtype lookup (0.662). A model is useful if it sits clearly to the right of the dashed line.

**What worked.**

- **Clinical + 3D MRI: validation AUC 0.713** vs 0.652 for clinical alone (p = 0.047, statistically significant on
  validation).
- **Largest gain in HER2+ tumours** (0.61 $\rightarrow$ 0.74).
- **Self-supervised pretraining helped:** MRI-only 0.62 with pretraining vs 0.55 with random weights.

**What did not work.**

- **MRI alone** stays below the subtype lookup (about 0.55–0.62).
- **2D networks overfit:** they learned the training patients and failed on new ones, especially Duke.
- **Per-slice normalisation** (professor's suggestion) made 3D features worse: it removes real differences between
  slices of the same tumour.
- **ComBat** removed the scanner signal (the hospital could be recognised from the features with AUC 0.93–0.99
  before, about 0.5 after) but did not improve pCR prediction.

**The limitation (say it yourself before anyone asks).** Cross-validation did not confirm the 0.713: there, clinical
alone scored 0.70 and clinical + MRI 0.68. So the MRI's real contribution is small and uncertain, somewhere between
0 and about 0.06. The final test set has not been used yet.

---

# 7. Impact and What's Next (right column) and the subtype chart

**The subtype chart.** Validation AUC *within* each subtype, clinical model vs clinical + 3D MRI. Within a subtype,
receptor status cannot help, so any gain comes from the image: HER2+ 0.61 $\rightarrow$ **0.74**, HR+/HER2− 0.56 $\rightarrow$ 0.61,
triple-negative 0.48 $\rightarrow$ 0.60. These groups are small (49–65 patients), so the intervals are wide.

**Explain the impact.**

- **For patients and clinicians:** a pre-treatment estimate of response could support treatment discussions and
  closer monitoring of likely non-responders.
- **Broader:** hospitals hold many scans without outcome labels; self-supervised learning puts them to use.

**Before real-world use:** one final test-set evaluation, testing on a hospital the model has never seen
(leave-one-site-out), calibration (are the probabilities honest?), external validation on local hospital data, and
clinician review.

**Next:** full-precision images; adding Ki-67, grade and regimen; early-treatment MRI; subtype-specific models,
starting with HER2+.

---

# 8. Acknowledgment and References (right, bottom)

- **Thanks:** Prof. Bapi Raju and Prof. Vinod; the BreastDCEDL authors and the I-SPY1, I-SPY2 and Duke study teams
  (The Cancer Imaging Archive); Kaggle for GPU compute.
- **References, one line each if asked:** BreastDCEDL (the dataset); SimCLR (our self-supervised method); DINOv2
  (a pretrained ViT we compared); ComBat (scanner harmonisation); Braman 2017 (tissue around the tumour predicts
  pCR); FiLM (how clinical data adjusts image features); N4 (MRI brightness correction); Medical Slice Transformer
  (combining slices).

---

# Likely questions and short answers

| question | answer |
|---|---|
| **Did you beat the baseline?** | On validation yes: 0.713 vs 0.662 for subtype (not significant, p 0.16) and vs 0.652 clinical (p 0.047). Cross-validation does not confirm it, so the MRI's gain is small and uncertain. |
| **The dataset paper reports AUC 0.94. Why is yours lower?** | Their 0.94 is one subgroup of 40 patients with only 5 responders, using extra data (genomic score, drug arm) that is not public. On their whole test set they report 0.72, the same level as ours. |
| **Why AUC and not accuracy?** | Only 30% respond; predicting "no response" for everyone already gives 70% accuracy. AUC measures ranking, independent of a threshold. |
| **Did you use the test set?** | No. All results are cross-validation and validation; the test set is kept for one final run. |
| **What is ComBat?** | A statistical method that removes each scanner's shift and scale from the features, fitted on training patients only. |
| **Why does it fail on Duke?** | Different scanners and protocols, smaller tumours, boxes instead of exact tumour outlines. Image-only models are still near chance there. |
| **Is this ready for clinics?** | No. It is a research result; it needs the test-set evaluation, external validation and clinician review. |
| **What would improve it most?** | Mid-treatment MRI (tumour shrinkage), and pathology variables like Ki-67 and grade. |
