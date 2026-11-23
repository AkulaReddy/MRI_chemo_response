"""Fill the course poster template with the project content -> TIAH_Poster_pCR_MRI.pptx (same layout and styling)."""
import os
import re
from copy import deepcopy

from pptx import Presentation
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.text.text import _Paragraph
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = f"{HERE}/Topics_in_AI_for_Healthcare_Project_showcase_Nov20th_Event_Poster_Template.pptx"
OUT = f"{HERE}/TIAH_Poster_pCR_MRI.pptx"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

prs = Presentation(TEMPLATE)
slide = prs.slides[0]
S = {sh.name: sh for sh in slide.shapes}


def runs_of(text):
    """'**bold** rest' -> [(text, bold)]"""
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    return [(p[2:-2], True) if p.startswith("**") else (p, False) for p in parts if p]


def fill(name, items, size, height=None, width=None, space=5):
    """items: list of (kind, text); kind = intro (italic), plain, bullet, sub (indented plain)."""
    shape = S[name]
    tf = shape.text_frame
    paras = tf.paragraphs
    bullet_tpl = next((p for p in paras if p._p.pPr is not None and p._p.pPr.find(A + "buChar") is not None), None)
    plain_tpl = paras[0]
    body = tf._txBody
    for p in body.findall(A + "p"):
        body.remove(p)
    for kind, text in items:
        tpl = bullet_tpl if kind in ("bullet", "sub") and bullet_tpl is not None else plain_tpl
        p = deepcopy(tpl._p)
        for child in p.findall(A + "r") + p.findall(A + "br") + p.findall(A + "fld"):
            p.remove(child)
        end = p.find(A + "endParaRPr")
        if end is not None:
            end.set("sz", str(int(size * 100)))
        if kind == "sub":
            pPr = p.get_or_add_pPr()
            for b in pPr.findall(A + "buChar") + pPr.findall(A + "buFont") + pPr.findall(A + "buAutoNum"):
                pPr.remove(b)
            pPr.set("marL", str(int(Inches(0.75))))
            pPr.set("indent", str(int(Inches(-0.3))))
        body.append(p)
        para = _Paragraph(p, tf)
        para.space_after = Pt(space)
        for t, bold in runs_of(text):
            r = para.add_run()
            r.text = t
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = kind == "intro"
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.word_wrap = True
    if height is not None:
        shape.height = Inches(height)
    if width is not None:
        shape.width = Inches(width)


def set_text(name, text, size, bold=None):
    tf = S[name].text_frame
    p = tf.paragraphs[0]
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    for extra in tf.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)
    r = p.runs[0]
    r.text = text
    r.font.size = Pt(size)
    if bold is not None:
        r.font.bold = bold


# ------------------------------------------------------------------ header and footer
set_text("TextBox 7", "Predicting Chemotherapy Response from Pre-treatment Breast MRI", 54)
set_text("TextBox 4", "Can a breast MRI taken before chemotherapy, combined with routine clinical data, predict a "
                      "pathologic complete response? We combine self-supervised 3D deep learning with clinical data "
                      "and test it against receptor subtype.", 24)
S["TextBox 4"].width = Inches(22.0)
set_text("TextBox 2", "Team member names:  Chinta Phani Rama Vaibhav (2025201067)  ·  Rohit Rao (2025201016)", 24)

# ------------------------------------------------------------------ 1. the problem & context
fill("TextBox 17", [
    ("bullet", "Many breast cancer patients get **chemotherapy before surgery** (neoadjuvant). Only **~30%** reach a "
               "**pathologic complete response (pCR)**: no invasive cancer left at surgery."),
    ("bullet", "Response is known only **after months of treatment**: patients who will not respond still bear the "
               "toxicity and the delay."),
    ("bullet", "Oncologists rely on **receptor subtype (HR/HER2)**, which is coarse: patients of the same subtype "
               "respond very differently."),
    ("bullet", "**Gap:** most MRI models are built and tested at one hospital; scanner differences make them fail "
               "at a new site."),
], size=22, height=5.1, space=8)

# ------------------------------------------------------------------ 2. our approach (how we arrived at it)
fill("TextBox 32", [
    ("intro", "We set an honest baseline first, then went from simple to complex."),
    ("bullet", "**Baseline to beat:** subtype lookup: each patient gets the training pCR rate of their HR/HER2 "
               "subtype (HR+/HER2- 14%, HER2+ 43%, triple-negative 38%). **Validation AUC 0.662.**"),
    ("bullet", "**Understanding the data:** our exploratory analysis found the field of view varied 70-315 mm, the "
               "late phase was saturated and many slices had no tumour, so we rebuilt the image pipeline at one "
               "physical scale (0.7 mm)."),
    ("bullet", "**Challenges:** 1,452 labelled patients from 3 sites with different scanners; pre-treatment scan "
               "only; no grade, Ki-67 or chemotherapy regimen in the data."),
    ("bullet", "**Alternatives we built and compared:**"),
    ("sub", "1. Clinical-only model (age, race, menopause, HR, HER2)"),
    ("sub", "2. Radiomics: kinetic, 2D / 3D shape and texture, peritumoral rings 0-12 mm, kinetic habitats"),
    ("sub", "3. Scanner harmonisation (ComBat), per-slice normalisation"),
    ("sub", "4. 2D deep learning: ViT, DINOv2, ResNet-18, MAE (30+ runs), Medical Slice Transformer"),
    ("sub", "5. **3D CNN pretrained without labels (SimCLR)** on kinetic maps"),
    ("bullet", "**Why 3D self-supervised:** the 2D networks overfit and were at chance on the Duke cohort. "
               "Self-supervision learns from all 1,798 scans, including 618 without outcome labels, and clearly "
               "beat the same network with random weights."),
    ("bullet", "**Honest testing:** settings ranked by repeated cross-validation, confirmed on 177 validation "
               "patients with paired DeLong tests; the 176 test patients are kept untouched."),
], size=18, height=9.9, space=6)

# ------------------------------------------------------------------ 3. our solution (what we built) + main visual
fill("TextBox 24", [
    ("intro", "A model that estimates each patient's probability of pCR from the pre-treatment DCE-MRI and routine "
              "clinical data (age, menopause, HR, HER2)."),
    ("bullet", "**Users:** oncologists and radiologists, as research decision support, not a stand-alone diagnosis."),
    ("bullet", "**Enables:** a response estimate before chemotherapy starts, and a test of whether the MRI adds "
               "anything beyond receptor subtype."),
    ("bullet", "**Addresses the problem:** trained on 3 sites at once, uses scanner-robust kinetic maps and learns "
               "from unlabelled scans."),
], size=18, height=2.2, space=2)
ph = S["Flowchart: Process 51"]
ph._element.getparent().remove(ph._element)
slide.shapes.add_picture(f"{HERE}/workflow_poster.png", Inches(10.47), Inches(7.72), width=Inches(14.1))

# ------------------------------------------------------------------ 4. how it works
S["TextBox 9"].width = Inches(6.8)
fill("TextBox 10", [
    ("intro", "From scan to prediction in five steps (figure above)."),
    ("bullet", "**1. Input:** pre, early and late DCE-MRI phases, the tumour box, and the clinical data."),
    ("bullet", "**2. Pre-processing:** resample to 0.7 mm; tumour region from contrast enhancement; kinetic maps "
               "(log pre-contrast, percent enhancement, washout: ratios, so scanner gain cancels); tumour-scale "
               "32x64x64 3D crop."),
    ("bullet", "**3. Self-supervised pretraining:** a 3D CNN learns, with no labels, that two random augmentations "
               "of the same tumour look alike (SimCLR, 300 epochs)."),
    ("bullet", "**4. Prediction:** image features + clinical data -> logistic regression, or fine-tuning where the "
               "clinical data modulates the image features (FiLM) -> probability of pCR."),
    ("bullet", "**5. Evaluation:** 5 x 5-fold cross-validation on 1,098 patients, 177 validation patients, paired "
               "DeLong tests against the subtype lookup."),
    ("plain", "**Technology:** Python, PyTorch, scikit-learn, SimpleITK (N4), DINOv2 / ViT, SimCLR, ComBat; "
              "Kaggle T4 GPUs."),
], size=16.5, height=7.15, space=6)

# ------------------------------------------------------------------ 5. what we learned (+ results chart)
S["TextBox 85"].width = Inches(6.8)
slide.shapes.add_picture(f"{HERE}/results_poster.png", Inches(17.87), Inches(14.35), width=Inches(6.9))
fill("TextBox 86", [
    ("bullet", "**Worked:** clinical + 3D MRI reached **validation AUC 0.713** vs 0.652 for clinical alone "
               "(p = 0.047); largest gain in **HER2+** (0.61 -> 0.74). Self-supervised pretraining beat random "
               "weights (0.62 vs 0.55, MRI only)."),
    ("bullet", "**Did not work:** MRI alone stays below subtype; 2D networks overfit; per-slice normalisation hurt "
               "3D features; ComBat removed the site signal (site recognisable at AUC 0.93-0.99) but did not "
               "improve prediction."),
    ("bullet", "**Limitation:** cross-validation did not confirm the 0.713 (clinical 0.70 vs clinical + MRI 0.68): "
               "the MRI gain is small and uncertain. Test set still to be evaluated."),
], size=14, height=3.2, space=3)
S["TextBox 86"].top = Inches(18.72)

# ------------------------------------------------------------------ 6. impact and what's next
fill("TextBox 59", [
    ("bullet", "**For patients and clinicians:** a pre-treatment estimate of response could support treatment "
               "discussions and closer monitoring of patients unlikely to respond."),
    ("bullet", "**Broader impact:** self-supervised learning puts unlabelled hospital scans to use where outcome "
               "labels are scarce."),
    ("bullet", "**Before real-world use:** one final test-set evaluation, leave-one-site-out testing, calibration, "
               "external validation on local hospital data, and clinician review."),
    ("bullet", "**Next:** full-precision images; add Ki-67, grade and regimen; early-treatment MRI; "
               "subtype-specific models, starting with HER2+."),
], size=18, height=4.2, space=7)
slide.shapes.add_picture(f"{HERE}/within_subtype_poster.png", Inches(25.65), Inches(11.85), width=Inches(6.4))

# ------------------------------------------------------------------ 7. acknowledgment, 8. references
fill("TextBox 43", [
    ("plain", "Prof. Bapi Raju and Prof. Vinod for guidance; the BreastDCEDL authors and the I-SPY1, I-SPY2 and Duke "
              "study teams (The Cancer Imaging Archive); Kaggle for GPU compute."),
], size=15, height=1.5)
fill("TextBox 44", [
    ("plain", "1. BreastDCEDL: curated breast DCE-MRI dataset (I-SPY1, I-SPY2, Duke), Zenodo."),
    ("plain", "2. Chen et al. SimCLR: contrastive learning of visual representations. ICML 2020."),
    ("plain", "3. Oquab et al. DINOv2: robust visual features without supervision. 2023."),
    ("plain", "4. Johnson et al. Batch effects and empirical Bayes (ComBat). Biostatistics 2007."),
    ("plain", "5. Braman et al. Intra- and peritumoral radiomics for pCR prediction on DCE-MRI. Breast Cancer Res 2017."),
    ("plain", "6. Perez et al. FiLM: feature-wise linear modulation. AAAI 2018."),
    ("plain", "7. Tustison et al. N4ITK bias-field correction. IEEE TMI 2010."),
    ("plain", "8. Mueller-Franzes et al. Medical Slice Transformer. Sci Rep 2025."),
], size=12.5, height=3.0, space=2)

prs.save(OUT)
print("saved", OUT)
