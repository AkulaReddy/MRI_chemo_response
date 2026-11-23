# Topics in AI for Healthcare: poster content

Figures in this folder: `workflow.png` (main visual), `results.png` (results chart), `within_subtype.png` (subtype chart).
To redraw them: `python3 make_poster_figs.py`.

---

## Poster title
**Can a Pre-treatment Breast MRI Predict Response to Chemotherapy? Self-supervised 3D Learning on Breast DCE-MRI**

**Team members:** Chinta Phani Rama Vaibhav (2025201067), Rohit Rao (2025201016)

**One-line project description:**
Predicting whether a breast cancer patient will have a pathologic complete response to neoadjuvant chemotherapy from the MRI taken before treatment, combined with routine clinical data, and testing whether the image adds anything beyond receptor subtype.

---

## 1. THE PROBLEM & CONTEXT

- **The problem:** many breast cancer patients receive chemotherapy before surgery (neoadjuvant chemotherapy). Only about 30% reach **pathologic complete response (pCR)**, meaning no invasive cancer is left at surgery. Today, response is known only after months of treatment.
- **Who is affected:** patients who go through months of toxic treatment that may not work, and oncologists who must choose a treatment before knowing whether it will work.
- **Why it matters:** an early estimate of response could support treatment planning, and could help target closer monitoring or trials of alternative therapies at patients unlikely to respond.
- **Gaps in current approaches:**
  - Receptor subtype (HR/HER2) is the main predictor in use, but it is coarse: patients with the same subtype respond very differently.
  - Many published imaging models are trained on one hospital and tested on that same hospital.
  - MRI scanners differ between sites, so models often fail on data from a new hospital.

## 2. OUR APPROACH

- **Proposed solution:** a prediction model that uses the **pre-treatment DCE-MRI** (three contrast phases) together with **routine clinical data** (age, menopause, HR, HER2). It outputs the probability of pCR.
- **Intended users:** oncologists and radiologists, as a research tool for decision support, not as a stand-alone diagnosis.
- **What it enables:** a response estimate before treatment starts, and evidence of whether the MRI adds information beyond receptor subtype.
- **How it addresses the problem:**
  - It learns from three cohorts at once, so it sees scanner variation.
  - It uses **kinetic maps** (enhancement ratios) that are largely independent of scanner gain.
  - It uses **self-supervised learning** to make use of 618 extra scans that have no outcome label.
- **Visual:** `workflow.png`

## 3. OUR SOLUTION

- **Understanding the problem:**
  - We set a strong, honest reference first: the **subtype lookup**, where each patient gets the pCR rate of their HR/HER2 subtype (validation AUC 0.662).
  - Exploratory analysis of the image pipeline found three problems: the field of view varied from 70 to 315 mm, the late phase was saturated, and many slices contained little or no tumour.
- **Assumptions and challenges:**
  - Only 1,452 labelled patients, from 3 sites with different scanners.
  - Only pre-treatment imaging.
  - No grade, Ki-67 or treatment regimen in the data.
  - Results were compared only with paired statistical tests, because one validation AUC carries about ±0.08 uncertainty.
- **Alternatives explored:**
  1. Clinical-only models.
  2. Hand-crafted radiomics: kinetic, shape and texture features, peritumoral rings 0–12 mm, and kinetic habitats.
  3. Scanner harmonisation (ComBat) and per-slice intensity normalisation.
  4. 2D deep learning with ViT, DINOv2, ResNet and MAE (30+ runs), and a Medical Slice Transformer over all slices.
  5. A **3D CNN pretrained without labels (SimCLR)** on the tumour's kinetic maps.
- **Why this approach:**
  - The 2D networks overfit and were near chance on the Duke cohort.
  - Self-supervised 3D pretraining uses all scans, including unlabelled ones, and clearly beat the same network with random weights.
- **Main visual:** `workflow.png`

## 4. HOW IT WORKS

1. **Input:** pre, early and late DCE-MRI phases, the tumour bounding box, and the clinical variables.
2. **Pre-processing:**
   - Resample every scan to 0.7 mm pixels.
   - Define the tumour region from its enhancement.
   - Compute kinetic maps: log pre-contrast, percent enhancement and washout.
   - Crop a 32 × 64 × 64 volume around the tumour, scaled to the tumour's size.
3. **Self-supervised pretraining:** a 3D CNN learns from 1,798 scans that two random augmentations of the same tumour should look alike. No outcome labels are used.
4. **Prediction:** the image features are combined with the clinical data. We tried two ways: a regularised logistic regression, and fine-tuning in which the clinical data modulates the image features (FiLM).
5. **Evaluation:**
   - 5 × repeated 5-fold cross-validation on 1,098 training patients.
   - An independent check on 177 validation patients.
   - Paired DeLong tests against the subtype lookup.
   - 176 test patients kept aside for one final run.
- **Technologies:** Python, PyTorch, scikit-learn, SimpleITK (N4 correction), DINOv2 / ViT, SimCLR, ComBat, Kaggle GPUs (T4).

## 5. WHAT WE LEARNED

**Results (AUC):**

| Model | Cross-validation (1,098 train) | Validation (177) |
|---|---|---|
| Subtype lookup (reference) | 0.63 | 0.66 |
| Clinical model | 0.70 | 0.65 |
| MRI only, best (self-supervised 3D) | 0.58 | 0.62 |
| **Clinical + 3D MRI** | 0.68 | **0.71** |
| Clinical + tumour volume + 3D MRI | **0.71** | 0.67 |

Figures: `results.png` and `within_subtype.png`.

- **What worked:**
  - **Self-supervised 3D pretraining** beat the same network with random weights (validation 0.62 vs 0.55).
  - **Adding the 3D MRI score to clinical data** raised validation AUC from 0.65 to 0.71 (p = 0.047). Combined with tumour volume, it added +0.016 in cross-validation (p = 0.03).
  - The biggest gain was in **HER2+ patients**: within-subtype validation AUC rose from 0.61 to 0.74.
  - **Kinetic maps and tumour-scale crops** made the image model work across sites (validation 0.69–0.73 in each cohort).
- **What did not work:**
  - **MRI alone** stayed below receptor subtype.
  - **2D deep networks** overfit, with the best epoch usually 1–7, and were near chance on Duke.
  - **Per-slice intensity normalisation** made 3D features worse.
  - **ComBat** removed the scanner signal (site recognisable at AUC 0.93–0.99 before, about 0.5 after) but did not improve prediction.
- **Limitations:**
  - The validation gain of 0.71 was **not confirmed by cross-validation** (clinical alone 0.70 vs clinical + MRI 0.68). The real image gain is small and uncertain, roughly 0–0.06.
  - Only pre-treatment imaging is used. Grade, Ki-67 and the treatment regimen are missing.
  - The current runs use 8-bit images, in which part of the brightest enhancement is clipped.
  - The test set has not been evaluated yet.

## 6. IMPACT AND WHAT'S NEXT

- **Potential impact:**
  - A pre-treatment estimate of response could help oncologists discuss expected benefit with patients and plan closer monitoring.
  - Self-supervised learning on unlabelled scans is a practical way to use the large volumes of imaging that hospitals hold without outcome labels.
- **Before real-world use:**
  - A single evaluation on the held-out test set.
  - Leave-one-site-out testing (train on two hospitals, test on the third).
  - Calibration of the predicted probabilities.
  - External validation on local hospital data.
  - Clinician review of the outputs.
- **Next steps:**
  - Use full-precision images.
  - Add Ki-67, grade and treatment regimen.
  - Use early-treatment MRI, where response is known to be easier to predict.
  - Build subtype-specific models, starting with HER2+.

## 7. ACKNOWLEDGMENT

- Prof. Bapi Raju and Prof. Vinod, course instructors and project guides.
- Dr. Mrunalini, MNJ Institute of Oncology, for clinical guidance. *(Include only if she has agreed to help.)*
- The BreastDCEDL dataset authors and the I-SPY1, I-SPY2 and Duke breast MRI study teams (via The Cancer Imaging Archive).
- Kaggle, for GPU compute.
- *Logos: institute logo, and MNJ if applicable.*

## 8. REFERENCES

*Check each reference against the original paper before printing.*

1. BreastDCEDL: a curated breast DCE-MRI dataset for deep learning (I-SPY1, I-SPY2, Duke), Zenodo.
2. Chen T. et al. A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). ICML 2020.
3. Oquab M. et al. DINOv2: Learning Robust Visual Features without Supervision. 2023.
4. Johnson W.E. et al. Adjusting batch effects in microarray expression data using empirical Bayes methods (ComBat). Biostatistics 2007.
5. Braman N. et al. Intratumoral and peritumoral radiomics for the pretreatment prediction of pathological complete response to neoadjuvant chemotherapy based on breast DCE-MRI. Breast Cancer Research 2017.
6. Perez E. et al. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018.
7. Tustison N. et al. N4ITK: Improved N3 Bias Correction. IEEE TMI 2010.
8. Müller-Franzes G. et al. Medical Slice Transformer for 3D medical image analysis. Scientific Reports 2025.
