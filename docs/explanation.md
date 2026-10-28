# How the pipeline works: explanations

Reference notes on how the data is prepared and what the model actually receives.
Code: [src/data/resample.py](../src/data/resample.py), [scripts/build_cache.py](../scripts/build_cache.py),
[src/models/fusion.py](../src/models/fusion.py).

1. [The model input: from MRI slices to ViT patches](#1-the-model-input-from-mri-slices-to-vit-patches)
2. [Interpolation: how the cache is resampled](#2-interpolation-how-the-cache-is-resampled)
3. [How slice thickness is normalised to 2 mm](#3-how-slice-thickness-is-normalised-to-2-mm)

---

## 1. The model input: from MRI slices to ViT patches

The whole path from one patient's MRI to one pCR probability, with the array shape at each step.

### Step 1: what's stored per patient

Each patient's MRI is a 3D volume: a stack of 2D slices, like pages in a book. For each slice we have
**three scans taken at different times**:

- **pre**: before the contrast agent is injected;
- **early**: about 1–2 minutes after injection, when the tumour lights up;
- **late**: several minutes after, when some tumours "wash out" (get darker again).

The cache stores, per patient, an array of shape **(S, 240, 240, 3)**:
- `S` slices (about 15 tumour slices plus margins);
- each slice is 240×240 pixels;
- each pixel has **3 numbers**: its brightness in pre, early and late.

That last axis of 3 is the "RGB". **It has nothing to do with colour.** A normal photo stores red,
green and blue per pixel, and we reused those three slots for the three time points: R = pre,
G = early, B = late.

### Step 2: pick 8 slices (the "bag")

From the patient's tumour slices we pick 8 (random ones in training, 16 evenly spaced ones in
evaluation):

```
one patient:  (8, 240, 240, 3)      8 slices, each a 240×240 image with 3 time points per pixel
crop to 224:  (8, 3, 224, 224)      random crop + flips in training, centre crop in evaluation
```

### Step 3: the ViT sees one slice at a time, not 8 at once

**The 8 slices are not fed into the ViT together.** They're like 8 separate photos. In training a
batch has several patients, say 4:

```
4 patients × 8 slices  →  (4, 8, 3, 224, 224)
flatten               →  (32, 3, 224, 224)     just 32 independent images
```

The ViT processes each of the 32 images **on its own**. It has no idea that slices 1–8 belong to the
same patient. They're only regrouped by patient at the end (step 5).

### Step 4: inside the ViT, how one slice becomes "words"

Take **one** slice, (3, 224, 224):

**4a. Cut it into patches.** The image is cut into a grid of 16×16-pixel squares, like cutting a photo
into tiles:
```
224 / 16 = 14 tiles across, 14 tiles down  →  14 × 14 = 196 patches
```
(DINOv2 uses 14×14-pixel patches: 224 / 14 = 16 → 16 × 16 = 256 patches.)

**4b. Turn each patch into a list of numbers.** A patch is 16 × 16 pixels × 3 values per pixel
(pre, early, late):
```
16 × 16 × 3 = 768 numbers
```
These are laid out in a line. So one patch carries **the same small square of tissue at all three
time points**, and the model can see "this spot was dark before contrast and bright after".

**4c. Embed each patch.** A learned linear layer (the "patch embedding") turns those 768 raw numbers
into a 768-number **token**. That's the "word" from the paper's title, *"An image is worth 16×16
words"*: each 16×16 patch becomes one word.

**4d. Add position and the CLS token.**
- Each token gets a position code, so the model knows where in the image the patch was.
- One extra learned token, **CLS**, is added at the front. It works as a summary slot.
```
1 CLS + 196 patch tokens = 197 tokens, each 768 numbers
```

**4e. Transformer blocks.** 12 layers of self-attention, where every token looks at every other token:
"this bright patch is next to that dark patch…". At the end the **CLS token holds a 768-number
summary of the whole slice**.

### Step 5: from slices back to the patient

```
each slice's CLS (768)  →  small linear head  →  1 number (slice score)
8 slice scores          →  average            →  1 patient score
                        →  sigmoid            →  pCR probability, e.g. 0.37
```

The loss compares that **one patient probability** with the patient's true pCR label. The gradient
then flows back through all 8 slices.

The whole pipeline in one picture:

```
patient ─┬─ slice 1 (3×224×224) ─► 196 patches ─► ViT ─► CLS ─► score 1 ─┐
         ├─ slice 2              ─► 196 patches ─► ViT ─► CLS ─► score 2 ─┤
         │   ...                        (same ViT, run separately)        ├─► average ─► pCR probability
         └─ slice 8              ─► 196 patches ─► ViT ─► CLS ─► score 8 ─┘
```

### Step 6: why 3 channels if we use early only?

**In principle a ViT can take a 1-channel image.** Two things to know.

**1. Channels don't change the number of patches.**
- With 1 channel, each patch is 16 × 16 × 1 = 256 numbers. With 3 channels, it's 768.
- Either way the patch embedding turns it into one token, and there are 196 tokens.
- So the ViT's work is the same. Only the length of each raw patch vector differs, before the
  embedding.

**2. The pretrained weights expect 3 channels.**
- Our ViT was pretrained on millions of colour photos. Its patch-embedding layer has weights of shape
  (768 outputs × **3** channels × 16 × 16), one set per channel.
- Feed it 1 channel and the shapes don't match.
- We could train a new 1-channel ViT from scratch, but with ~1,000 patients it would learn far less
  than the pretrained one. That's why we use pretrained models at all.

So with `input_phases: early` we **copy the early image into all three slots**: (early, early, early).
This is not a waste:
- it's **mathematically identical to a 1-channel ViT** whose patch weights are the sum of the 3
  pretrained colour weights;
- the copies just let us keep the pretrained weights unchanged;
- same number of tokens, same compute.

In other words, **with `input_phases: early` the model already gets only the early image.** The 3
copies are a technicality.

**Then why ever use pre/early/late?** Because the change over time is information that early alone
lacks:
- **Pre → early** shows how strongly the tumour takes up contrast.
- **Early → late** shows whether it washes out, which is linked to tumour aggressiveness.

With all three in each patch the model can see that change directly. In our runs, early-only and
pre/early/late came out about the same (differences within noise), so neither has clearly won yet.

### Summary

Each slice is cut into 196 small squares, and each square, with its 3 time points, becomes one "word".
The ViT reads each slice's words separately and summarises the slice in one vector. A score is made
per slice, and the 8 slice scores are averaged into one pCR prediction per patient.

---

## 2. Interpolation: how the cache is resampled

We use **trilinear interpolation** with a **Gaussian anti-aliasing filter** before any downsampling.
The code is in [src/data/resample.py](../src/data/resample.py), built on SciPy.

### 2.1 Interpolation: trilinear (linear in 3D)

```python
ndimage.affine_transform(sub, scales, offset=..., order=1, mode="constant", cval=0.0, prefilter=False)
```
- **`order=1`** means linear interpolation along each axis: slices (z), rows (y) and columns (x).
  Together that's *trilinear*.
- **Weighted average.** Each new voxel's value is a weighted average of the 8 nearest original voxels,
  with weights set by distance. For example, a new pixel 30% of the way between two old pixels gets
  0.7 × the first + 0.3 × the second.
- **`mode="constant", cval=0.0`:** anywhere outside the original scan becomes 0. That's the black
  padding when the 168-mm window (240 px × 0.7 mm) is bigger than the scan.
- **One step for all three axes.** z (to 2.0 mm) and in-plane (to 0.7 mm) are resampled together,
  separately for each DCE phase.

### 2.2 Anti-aliasing: a Gaussian blur, only when shrinking

```python
sigma = (scale - 1) / 2   # per axis, only where scale > 1 (downsampling)
ndimage.gaussian_filter(sub, sigma, ...)
```
- **When it applies.** Downsampling happens when the scan is finer than the target, e.g.
  0.31 mm → 0.7 mm in-plane, or 1.2 mm → 2.0 mm slices. Each new voxel then covers several old ones.
- **Why it's needed.** Plain linear interpolation would just *pick* nearby values and skip the rest.
  That causes aliasing: jagged edges and noise patterns.
- **What it does.** We blur slightly first, then sample. Each new voxel then effectively **averages
  all the tissue it covers**. That's how thin slices get averaged, and it also reduces noise.
- **How much.** σ = (factor − 1)/2 voxels, the same rule scikit-image uses. Example: 0.31 → 0.7 mm is a
  factor of 2.26, so σ ≈ 0.63 pixels.
- **No blur when upsampling** (scan coarser than the target, e.g. 1.41 → 0.7 mm, or 3.4 mm → 2 mm
  slices): just linear interpolation. It **can't create detail** the scanner didn't capture; those
  images just look smoother.

### 2.3 The tumour mask

- **Resampled linearly** with no blur, then **thresholded at 0.5**: a new voxel counts as tumour if it's
  at least half inside the original mask.
- This keeps the mask binary and aligned with the image.

### 2.4 Order of steps

1. Compute the clip range (`lo`, `hi`) on the **original** volumes.
2. Resample each phase (blur if shrinking, then trilinear).
3. Convert to uint8 RGB with that clip range.

### 2.5 Why linear and not something fancier

| method | pros | cons |
|---|---|---|
| nearest neighbour | fast, keeps original values | blocky, jagged edges; bad for images |
| **linear (ours)** | smooth, no overshoot, fast, standard for MRI intensities | slightly blurs fine detail |
| cubic / B-spline (`order=3`) | sharper | **overshoot/ringing** at sharp edges (e.g. a bright tumour next to fat), values below 0 or above the original range |
| Lanczos / sinc | sharpest | ringing, slower |

**Why this matters for us:**
- Linear never produces values outside the range of its neighbours.
- We use intensity *ratios between phases* (enhancement), and ringing would distort them.
- Linear is the default for intensity images in common medical-imaging tools (e.g. SimpleITK's
  `sitkLinear`, nnU-Net's resampling of low-anisotropy data).

---

## 3. How slice thickness is normalised to 2 mm

Slice thickness is normalised by **resampling along the slice axis (z)**, in the same step as the
in-plane resampling to 0.7 mm (section 2). The functions are `tumour_z_grid` and `resample` in
[src/data/resample.py](../src/data/resample.py), called from
[scripts/build_cache.py](../scripts/build_cache.py).

### Step 1: place new slices every 2.0 mm across the tumour

From the tumour's slice range `[z0, z1)` in the original scan and the original spacing `dz` (CSV
`slice_thick`):

```
tumour depth (mm)      = (z1 − z0) × dz
tumour slices (new)    = round(depth / 2.0)          e.g. 30 mm → 15 slices
new slice centres      = every 2.0 mm, centred on the tumour's middle
+ 2 margin slices on each side (dropped if they'd fall outside the scan)
```

Every patient's tumour is sampled at the same physical depth step. `t0:t1` in the cache marks the new
tumour slices.

### Step 2: compute each new slice from the original slices

Each new slice sits at some *fractional* position between original slices.

**Thin slices (`dz` < 2 mm, e.g. 1.2 mm): downsampling**
- One new slice covers 2.0 / 1.2 = 1.67 original slices.
- First a **Gaussian blur along z** with σ = (1.67 − 1)/2 ≈ 0.33 slices (anti-aliasing), then
  **linear interpolation** between the two nearest original slices.
- In effect each new slice is a **weighted average of the 1–2 thin slices it covers**. Thin slices are
  noisier, so this also reduces noise.

**Thick slices (`dz` > 2 mm, e.g. 3.4 mm): upsampling**
- One new slice covers 2.0 / 3.4 = 0.59 original slices, so new slices fall *between* the originals.
- **Linear interpolation** only: a new slice 40% of the way between two originals =
  0.6 × the first + 0.4 × the second.
- This **creates no new detail**. The scanner never measured between those slices, so they're smooth
  blends of their neighbours.

**`dz` = 2 mm (the median):** scale 1.0, so the new slices land on the original ones and the values are
unchanged.

### Example: the same 30-mm tumour on three scans

| scan slice spacing | original tumour slices | after v2 |
|---|---|---|
| 1.2 mm | 25 | 15 (each ≈ average of 1.67 originals) |
| 2.0 mm | 15 | 15 (unchanged) |
| 3.4 mm | 9 | 15 (6 of them interpolated between originals) |

Before v2, the bag sampler picked from 25, 15 or 9 slices for the same tumour. Thin-slice scans had
many near-duplicate slices and thick-slice scans few. Now it's 15 for everyone.

### Caveats

- **Thickness vs spacing.** We use the CSV `slice_thick` as the distance between slice centres. On some
  MRI protocols slices overlap or have gaps, so the true spacing differs from the thickness. The
  archive's NIfTI headers say 1 mm (wrong), so the CSV is the only source we have.
- **Not isotropic, on purpose.** The voxels are 0.7 × 0.7 × 2.0 mm. Our models work on 2-D slices, so
  we only need consistent in-plane detail and a consistent slice step. Going to 0.7 mm in z would
  triple the slice count without adding real information.
- **Under review.** Whether to resample z at all is still open: for the 2D models it may be unnecessary.
  An ablation with native slices (cache v3) is planned; see [ideas.md](ideas.md), item 13.
