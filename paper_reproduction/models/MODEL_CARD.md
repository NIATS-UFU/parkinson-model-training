---
# Model license is OpenRAIL-M with an added no-clinical-use restriction (see "License"
# below and the LICENSE file in this folder). Pending NIATS-UFU steward sign-off before publishing.
license: other
license_name: openrail-m
license_link: LICENSE
library_name: skops
tags:
  - parkinsons-disease
  - spiral-drawing
  - dinov2
  - linear-probe
  - medical-imaging
  - scikit-learn
pipeline_tag: image-classification
---

# DINOv2 + Hemato linear probe — spiral Parkinson's classifier (reference checkpoint)

A logistic-regression **linear probe** on top of a **frozen, public DINOv2
ViT-B/14** backbone, fit on 100% of the **Hemato** clinical spiral-drawing cohort.
This is the trained artifact released with *"Cross-Dataset Robustness of Spiral
Classifiers for Parkinson's Disease Diagnosis."* It is the paper's H4 novelty: a
competitive in-domain classifier that needs **no fine-tuning** and only three
configuration choices.

**What is and isn't in this repo.** The backbone is *not* included — it is the
public `facebookresearch/dinov2` `dinov2_vitb14`, downloaded at run time and never
trained by us. The only artifact here is the ~KB scikit-learn probe
(`parkinson_spiral_dinov2_hemato.skops`) plus its `config.json`.

## Intended use

This is a **reference / exploration checkpoint** for researchers, released so the
frozen-DINOv2 direction can be explored in future work — **not** a screening or
diagnostic tool. The companion paper's central finding (H2) is that spiral
classifiers, including this one, **do not transfer across datasets**: a Hemato-fit
probe degrades on other cohorts and off-domain probes collapse to near-chance on
Hemato. Do **not** apply this probe to non-Hemato images and expect calibrated
results, and do **not** use it for any clinical decision.

## How to use

```python
# pip install -e ".[train,hf]"  (from the paper_reproduction/ code release)
python models/predict.py --model-dir <this folder> --images path/to/spirals/
```

Or directly:

```python
import json, torch, skops.io as sio
from PIL import Image
from parkinson_spiral.recipes import get_recipe

cfg = json.load(open("config.json"))
probe = sio.load(
    "parkinson_spiral_dinov2_hemato.skops", trusted=sio.get_untrusted_types(file="parkinson_spiral_dinov2_hemato.skops")
)
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
tf = get_recipe("AUG-None").eval_transform

x = tf(Image.open("spiral.png").convert("RGB")).unsqueeze(0)
with torch.inference_mode():
    feat = backbone(x).numpy()
p_pd = probe.predict_proba(feat)[0, 1]  # P(Parkinson's)
```

Inputs are Archimedean-spiral drawings; preprocessing is AUG-None (resize 224,
ImageNet normalization, RGB). Output column 1 is P(Parkinson's). CPU is sufficient.

## Training data

100% of the Hemato spiral images (no holdout). Hemato is an IRB-restricted clinical
cohort collected at NIATS-UFU and is **not** publicly distributed; it is available
on request (see the code release `docs/DATA.md`). Labels are subject-level
Parkinson's-vs-control.

## Evaluation

Pooled out-of-fold 5-fold participant-grouped CV on Hemato (the value this
all-data probe is calibrated to reproduce): **subject accuracy 0.739**, AUC 0.752.
Cross-dataset transfer is poor by design of the finding — see the paper's Table
"cross-dataset" and Fig. "matrix".

## Limitations

- Spiral drawings only; one clinical cohort; modest size.
- Does **not** generalize across datasets (the paper's main result).
- Not validated for clinical or screening use.

## License

**OpenRAIL-M with an added no-clinical-use restriction** — see the [`LICENSE`](LICENSE)
file in this folder for the full terms. It is a permissive Responsible AI License:
broad rights to use and build on the probe, subject to use restrictions that must be
passed on to downstream users. The key added restriction (Attachment A.1) **forbids
using the model for clinical, diagnostic, or screening purposes** unless the
deployment has been independently validated and cleared by the relevant regulatory
authority — which makes the "reference checkpoint, not a diagnostic tool" statement
above a binding term rather than a disclaimer.

This choice reflects the model's clinical provenance (it is derived from the
IRB-restricted Hemato cohort). **Pending sign-off:** the NIATS-UFU data stewards must
confirm these terms — and that the IRB/consent permits redistributing a derived model
at all — before this model is published. The repository code is licensed separately
under MIT.

## Citation

See the paper *"Cross-Dataset Robustness of Spiral Classifiers for Parkinson's
Disease Diagnosis"* (NIATS-UFU). Code: <https://github.com/NIATS-UFU/parkinson-model-training>
(`paper_reproduction/`).
