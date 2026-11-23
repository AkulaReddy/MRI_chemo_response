import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
INK, MUTED = "#1f2a30", "#55606a"
C = {"data": ("#E8F0FB", "#2F6FDF"), "prep": ("#EEF6EE", "#2A9D5C"), "model": ("#FFF4E6", "#D9822B"),
     "ssl": ("#FCEBEA", "#B3261E"), "fuse": ("#F1ECFA", "#6E4BB8"), "eval": ("#EEF1F3", "#46545E")}


def box(ax, x, y, w, h, kind, title, lines, fs=9.2, wrap=40, title_fs=10.6, bold_lines=()):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=1.8))
    ax.text(x + 0.16, y - 0.24, title, fontsize=title_fs, fontweight="bold", color=ec, va="center")
    yy = y - 0.56
    for ln in lines:
        for k, piece in enumerate(textwrap.wrap(ln, wrap)):
            ax.text(x + 0.16 + (0 if k == 0 else 0.14), yy, ("• " if k == 0 else "") + piece, fontsize=fs,
                    color=INK, va="center", fontweight="bold" if ln in bold_lines else "normal")
            yy -= 0.27
        yy -= 0.04
    return y - h


def arrow(ax, a, b, color="#46545E", style="-|>", lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=16, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def workflow():
    W, H = 22, 13.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.4, H - 0.45, "Predicting pCR from pre-treatment breast DCE-MRI: end-to-end workflow", fontsize=21,
            fontweight="bold", color=INK, va="center")
    ax.text(0.4, H - 0.95, "Question: does the pre-treatment MRI add anything to what HR/HER2 subtype and routine clinical "
            "data already tell us about response to neoadjuvant chemotherapy?", fontsize=12, color=MUTED, va="center")

    top = H - 1.45
    x1, w1 = 0.4, 3.9
    box(ax, x1, top, w1, 4.35, "data", "1. DATA: BreastDCEDL (public)", [
        "3 cohorts: I-SPY1 172, I-SPY2 982, Duke 298",
        "1,452 patients with a pCR label (29.5% pCR) + 618 unlabelled Duke scans",
        "DCE-MRI: pre, early and late post-contrast phases; tumour box (I-SPY masks)",
        "Clinical: age, race, menopause, HR, HER2",
        "Fixed split: train 1,099 / validation 177 / test 176 (test held out)",
    ], wrap=36)
    box(ax, x1, top - 4.6, w1, 2.75, "eval", "Reference to beat", [
        "Subtype lookup: each patient gets the train pCR rate of their HR/HER2 subtype",
        "(HR+/HER2- 14%, HER2+ 43%, TN 38%)",
        "Validation AUC 0.662",
    ], wrap=36, bold_lines=("Validation AUC 0.662",))

    x2, w2 = 4.75, 3.9
    box(ax, x2, top, w2, 7.35, "prep", "2. PRE-PROCESSING", [
        "Resample to 0.7 mm pixels / 2 mm slices: same physical scale for every patient",
        "Tumour-centred crops; tumour region = enhancement (PE >= 0.7) inside the box, cleaned",
        "Kinetic maps: log pre-contrast, percent enhancement (PE), washout: ratios, so scanner gain cancels",
        "Intensity fixes: N4 bias-field correction + Nyul histogram standardisation",
        "ComBat harmonisation of features (fitted on train only)",
        "EDA found 3 pipeline problems: field of view 70-315 mm, late-phase saturation, empty slices",
    ], wrap=36)

    x3, w3 = 9.1, 4.25
    y = top
    c1 = y - 0.55; y = box(ax, x3, y, w3, 1.1, "model", "3a. Clinical model", ["Logistic regression on age, race, menopause, HR, HER2"], wrap=40) - 0.2
    c2 = y - 0.85; y = box(ax, x3, y, w3, 1.75, "model", "3b. Radiomics (hand-crafted)", [
        "Kinetic (PE, SER, wash-in/out), 2D / 3D shape + texture, per-slice curves",
        "Peritumoral rings 0-12 mm, kinetic habitats"], wrap=40) - 0.2
    c3 = y - 0.85; y = box(ax, x3, y, w3, 1.75, "model", "3c. 2D deep learning", [
        "ViT, DINOv2, ResNet-18, MAE on tumour slices (30+ runs)",
        "Medical Slice Transformer / attention over all slices"], wrap=40) - 0.2
    c4 = y - 1.2; box(ax, x3, y, w3, 2.45, "ssl", "3d. 3D self-supervised (best image model)", [
        "3D CNN pretrained with SimCLR on 1,798 scans, no labels (incl. unlabelled Duke)",
        "Input: tumour-scale 32x64x64 crop of the kinetic maps",
        "Then: linear probe, or fine-tuned with clinical data via FiLM"], wrap=40)

    x4, w4 = 13.8, 3.6
    box(ax, x4, top - 1.6, w4, 3.25, "fuse", "4. FUSION", [
        "Clinical + tumour volume + image score",
        "Regularised logistic regression (stacking)",
        "Also tested per subtype",
    ], wrap=33)

    x5, w5 = 17.85, 3.75
    box(ax, x5, top, w5, 4.6, "eval", "5. EVALUATION", [
        "5 x repeated 5-fold CV on 1,098 train patients (out-of-fold AUC: stable)",
        "Validation 177 patients: independent check (+/-0.08)",
        "Paired DeLong / bootstrap tests vs the subtype lookup",
        "Test 176: one final run only",
    ], wrap=34)

    for xa, xb, yy in ((x1 + w1, x2, top - 2.0), (x2 + w2, x3, top - 2.0)):
        arrow(ax, (xa + 0.03, yy), (xb - 0.03, yy))
    for yy in (c1, c2, c3, c4):
        arrow(ax, (x3 + w3 + 0.03, yy), (x4 - 0.03, top - 3.2), rad=0.0, lw=1.2)
    arrow(ax, (x4 + w4 + 0.03, top - 3.2), (x5 - 0.03, top - 2.3))

    # results strip
    ry = 2.95
    ax.add_patch(FancyBboxPatch((0.4, 0.35), W - 0.8, ry, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#FBFBFC", ec="#C9D0D5", lw=1.2))
    ax.text(0.65, ry + 0.05, "6. KEY RESULTS (AUC; CV = out-of-fold on 1,098 train patients, Val = 177 validation patients)",
            fontsize=12.5, fontweight="bold", color=INK, va="center")
    res = [("Subtype lookup", "CV 0.63 | Val 0.66", "#46545E"),
           ("Clinical model", "CV 0.70 | Val 0.65", "#D9822B"),
           ("MRI only (best)", "CV 0.58 | Val 0.62", "#2F6FDF"),
           ("Clinical + 3D MRI", "CV 0.68 | Val 0.71", "#B3261E"),
           ("Clin + vol + 3D MRI", "CV 0.71 | Val 0.67", "#6E4BB8")]
    cw = (W - 1.3) / len(res)
    for i, (t, v, col) in enumerate(res):
        cx = 0.65 + i * cw
        ax.add_patch(FancyBboxPatch((cx, 1.85), cw - 0.25, 0.9, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="white", ec=col, lw=1.8))
        ax.text(cx + (cw - 0.25) / 2, 2.5, t, ha="center", fontsize=11, fontweight="bold", color=col)
        ax.text(cx + (cw - 0.25) / 2, 2.1, v, ha="center", fontsize=11, color=INK)
    findings = ("MRI alone is weak (below subtype).   Adding the self-supervised 3D MRI score to clinical data raised "
                "validation AUC 0.652 -> 0.713 (p = 0.047),\nbut cross-validation did not confirm it (0.695 -> 0.676).   "
                "With tumour volume, the MRI gain was +0.016 in CV (p = 0.03).   Scanner/site effects are large "
                "(site recognisable at AUC 0.93-0.99).\nSelf-supervised pretraining beat the same network with random weights.   "
                "Test set (176) still held out for one final evaluation.")
    ax.text(0.65, 1.1, findings, fontsize=10.6, color=INK, va="center", linespacing=1.5)

    fig.savefig(f"{OUT}/workflow.png", dpi=220, facecolor="white")
    plt.close(fig)


def results():
    rows = [  # label, cv(oof), val, lo, hi, colour
        ("Subtype lookup (reference)", 0.629, 0.662, 0.578, 0.740, "#46545E"),
        ("Clinical model (LR)", 0.695, 0.652, 0.566, 0.737, "#D9822B"),
        ("Clinical + volume + 3D MRI (stack)", 0.712, 0.670, 0.584, 0.754, "#6E4BB8"),
        ("Clinical + 3D MRI (FiLM fine-tuned)", 0.670, 0.688, 0.599, 0.766, "#B3261E"),
        ("Clinical + 3D MRI (SSL probe)", 0.676, 0.713, 0.632, 0.791, "#B3261E"),
        ("3D MRI only, self-supervised", 0.575, 0.621, 0.531, 0.714, "#2F6FDF"),
        ("3D MRI only, random weights", 0.538, 0.546, 0.451, 0.640, "#8FB3F5"),
        ("2D slice transformer (DINOv2)", 0.596, 0.582, 0.491, 0.670, "#8FB3F5"),
    ]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.2))
    n = len(rows)
    for i, (lab, cv, val, lo, hi, col) in enumerate(rows):
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color=col, lw=2, alpha=0.55)
        ax.plot(val, y, "o", color=col, ms=8, mec="white", mew=0.8, zorder=3)
        ax.plot(cv, y, "D", color=col, ms=6, mfc="white", mew=1.6, zorder=3)
        ax.text(0.815, y, f"Val {val:.3f} | CV {cv:.3f}", va="center", fontsize=9, color=col)
    ax.axvline(0.662, color="#46545E", ls="--", lw=1)
    ax.axvline(0.5, color="#bbbbbb", ls=":", lw=1)
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0.42, 0.95); ax.set_xlabel("AUC")
    ax.plot([], [], "o", color="#444", label="validation (177), with 95% CI")
    ax.plot([], [], "D", color="#444", mfc="white", label="cross-validation (1,098 train, out-of-fold)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=8.5, frameon=False)
    ax.set_title("Every model against the subtype lookup (dashed line)", fontsize=12, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/results.png", dpi=220); plt.close(fig)


def subtype():
    sub = ["HER2+\n(n=49)", "HR+/HER2-\n(n=65)", "Triple neg.\n(n=63)"]
    clin, img = [0.610, 0.561, 0.483], [0.735, 0.609, 0.598]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    import numpy as np
    x = np.arange(3)
    ax.bar(x - 0.2, clin, 0.38, color="#D9822B", label="clinical model")
    ax.bar(x + 0.2, img, 0.38, color="#B3261E", label="clinical + 3D MRI")
    for i in range(3):
        ax.text(i - 0.2, clin[i] + 0.01, f"{clin[i]:.2f}", ha="center", fontsize=9)
        ax.text(i + 0.2, img[i] + 0.01, f"{img[i]:.2f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="#999", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(sub); ax.set_ylim(0.4, 0.8); ax.set_ylabel("validation AUC within subtype")
    ax.set_title("Within each subtype (validation)", fontsize=11, loc="left"); ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/within_subtype.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    workflow(); results(); subtype()
    print(sorted(f for f in os.listdir(OUT) if f.endswith(".png")))


def workflow_poster():
    """Compact pipeline for the poster's 'solution visual' area (14.3 x 5.5 in, printed at full size)."""
    W, H = 14.3, 5.5
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    stages = [
        ("data", "1. Data", ["BreastDCEDL: I-SPY1,", "I-SPY2, Duke", "1,452 labelled", "+ 618 unlabelled scans",
                             "DCE-MRI: pre / early /", "late phase + clinical"]),
        ("prep", "2. Pre-processing", ["Resample to 0.7 mm", "Tumour region from", "contrast enhancement",
                                       "Kinetic maps: log pre,", "% enhancement, washout", "Tumour-scale 3D crop"]),
        ("ssl", "3. Self-supervised 3D", ["3D CNN learns from", "1,798 scans with NO", "labels (SimCLR): two",
                                         "views of one tumour", "should look alike"]),
        ("fuse", "4. Add clinical data", ["Image features +", "age, menopause,", "HR, HER2", "Logistic regression",
                                         "or FiLM fine-tuning"]),
        ("eval", "5. Output + testing", ["Probability of pCR", "5x5-fold CV on 1,098", "Validation on 177",
                                        "Paired tests vs subtype", "Test 176 held out"]),
    ]
    bw, gap, top, bh = 2.5, 0.42, H - 0.12, 3.45
    x_off = (W - 5 * bw - 4 * gap) / 2
    for i, (k, t, lines) in enumerate(stages):
        x = x_off + i * (bw + gap)
        fc, ec = C[k]
        ax.add_patch(FancyBboxPatch((x, top - bh), bw, bh, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=2.2))
        ax.text(x + bw / 2, top - 0.34, t, ha="center", va="center", fontsize=14, fontweight="bold", color=ec)
        for j, ln in enumerate(lines):
            ax.text(x + bw / 2, top - 0.86 - j * 0.42, ln, ha="center", va="center", fontsize=12.2, color=INK)
        if i < len(stages) - 1:
            arrow(ax, (x + bw + 0.04, top - bh / 2), (x + bw + gap - 0.04, top - bh / 2), lw=2.4)
    y0 = top - bh - 0.18
    ax.add_patch(FancyBboxPatch((0.12, 0.1), W - 0.24, y0 - 0.1, boxstyle="round,pad=0,rounding_size=0.1",
                                fc="#FBFBFC", ec="#C9D0D5", lw=1.2))
    ax.text(0.35, y0 - 0.32, "Also built and compared:", fontsize=12.5, fontweight="bold", color=INK, va="center")
    ax.text(0.35, y0 - 0.88, "clinical-only model  ·  radiomics (kinetic, 2D/3D shape + texture, peritumoral rings, habitats)  ·  "
            "ComBat harmonisation\n2D ViT / DINOv2 / ResNet-18 / MAE (30+ runs)  ·  Medical Slice Transformer  ·  per-slice normalisation",
            fontsize=11.5, linespacing=1.5, color=INK, va="center")
    ax.text(0.35, y0 - 1.48, "Reference to beat: subtype lookup (each patient gets the pCR rate of their HR/HER2 subtype), validation AUC 0.662",
            fontsize=11.5, color="#B3261E", fontweight="bold", va="center")
    fig.savefig(f"{OUT}/workflow_poster.png", dpi=250, facecolor="white"); plt.close(fig)


workflow_poster()


def poster_charts():
    """Results + subtype charts drawn at their printed size on the poster, so the text is readable."""
    import numpy as np
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    rows = [("Subtype lookup", 0.629, 0.662, 0.578, 0.740, "#46545E"),
            ("Clinical model", 0.695, 0.652, 0.566, 0.737, "#D9822B"),
            ("Clinical + 3D MRI", 0.676, 0.713, 0.632, 0.791, "#B3261E"),
            ("Clin. + 3D MRI (FiLM)", 0.670, 0.688, 0.599, 0.766, "#B3261E"),
            ("Clin. + vol. + 3D MRI", 0.712, 0.670, 0.584, 0.754, "#6E4BB8"),
            ("3D MRI only (SSL)", 0.575, 0.621, 0.531, 0.714, "#2F6FDF"),
            ("3D MRI, random wts", 0.538, 0.546, 0.451, 0.640, "#8FB3F5"),
            ("2D slice transformer", 0.596, 0.582, 0.491, 0.670, "#8FB3F5")]
    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    n = len(rows)
    for i, (lab, cv, val, lo, hi, col) in enumerate(rows):
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color=col, lw=2.2, alpha=0.5)
        ax.plot(val, y, "o", color=col, ms=8, mec="white", mew=0.8, zorder=3)
        ax.plot(cv, y, "D", color=col, ms=6.5, mfc="white", mew=1.7, zorder=3)
        ax.text(0.81, y, f"{val:.3f}", va="center", fontsize=10.5, color=col, fontweight="bold")
    ax.text(0.81, n - 0.35, "Val", fontsize=10, color=MUTED, va="bottom")
    ax.axvline(0.662, color="#46545E", ls="--", lw=1.1)
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10.5)
    ax.set_xlim(0.44, 0.85); ax.set_xticks([0.5, 0.6, 0.7, 0.8]); ax.set_xlabel("AUC (dashed: subtype lookup 0.662)")
    ax.plot([], [], "o", color="#444", label="validation, 177 (95% CI)")
    ax.plot([], [], "D", color="#444", mfc="white", label="cross-validation, 1,098")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, fontsize=10, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/results_poster.png", dpi=250); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 3.7))
    x = np.arange(3)
    clin, img = [0.610, 0.561, 0.483], [0.735, 0.609, 0.598]
    ax.bar(x - 0.2, clin, 0.38, color="#D9822B", label="clinical model")
    ax.bar(x + 0.2, img, 0.38, color="#B3261E", label="clinical + 3D MRI")
    for i in range(3):
        ax.text(i - 0.2, clin[i] + 0.008, f"{clin[i]:.2f}", ha="center", fontsize=11)
        ax.text(i + 0.2, img[i] + 0.008, f"{img[i]:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="#999", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(["HER2+ (n=49)", "HR+/HER2- (n=65)", "Triple neg. (n=63)"], fontsize=10.5)
    ax.set_ylim(0.4, 0.8); ax.set_ylabel("validation AUC")
    ax.set_title("Where the MRI helps most: within each subtype", fontsize=12, loc="left", fontweight="bold")
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/within_subtype_poster.png", dpi=250); plt.close(fig)


poster_charts()
