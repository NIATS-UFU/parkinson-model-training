# OpenRAIL-M License — DINOv2 + Hemato spiral Parkinson's probe

**Open Responsible AI License (Model variant).** This license applies to the
**trained model artifact** released with the paper *"Cross-Dataset Robustness of
Spiral Classifiers for Parkinson's Disease Diagnosis"* (NIATS-UFU): the serialized
scikit-learn probe `parkinson_spiral_dinov2_hemato.skops` and its `config.json`.
It does **not** apply to the source code in this repository, which is licensed
separately under the MIT License (see the top-level `LICENSE`), nor to the public
DINOv2 backbone, which is governed by its own upstream license
(`facebookresearch/dinov2`).

> **Status: pending data-steward approval.** Because this model is derived from
> IRB-restricted clinical data (the Hemato cohort), this license is the *proposed*
> licensing terms and must be confirmed by the NIATS-UFU data stewards before the
> model is published. Do not redistribute the model until that confirmation is in
> place.

This is an [Open RAIL-M](https://www.licenses.ai/) license: it grants broad,
royalty-free permission to use and build on the model, subject to a set of
**use-based restrictions** (Attachment A) that must be passed on to every
downstream user. The structure follows the BigScience OpenRAIL-M template; the
canonical base text is available at <https://www.licenses.ai/>.

## Section I — Grant of rights

Subject to the use restrictions in Attachment A, the Licensor grants you a
perpetual, worldwide, non-exclusive, royalty-free, irrevocable copyright and
patent license to reproduce, use, publicly display, distribute, modify, and
create derivatives of the Model and its Derivatives (e.g. re-fitted probes,
distillations, or other artifacts incorporating the Model's parameters or
outputs).

## Section II — Conditions of use

1. **Attribution.** You must give appropriate credit by citing the paper above and
   retaining this license and the accompanying model card with any copy or
   Derivative of the Model.
2. **Use restrictions propagate.** You must include the use restrictions in
   Attachment A (at minimum, identical in substance) in any license you apply to a
   Derivative of the Model, and you must not operate the Model or its Derivatives
   in a way that removes or weakens those restrictions for downstream users. The
   rest of your license terms may differ.
3. **As-is.** The Model is provided "AS IS", without warranties or conditions of
   any kind. The Licensor is not liable for any damages arising from its use. The
   Model is a research checkpoint and has **not** been validated, cleared, or
   approved for any clinical, diagnostic, or screening purpose by any regulatory
   authority.

## Attachment A — Use restrictions

You agree not to use the Model or any Derivative of the Model:

1. **For clinical or diagnostic purposes.** To provide medical or health advice,
   to diagnose, screen for, triage, or make or materially inform any clinical
   decision about Parkinson's disease or any other condition for an individual,
   or to otherwise substitute for the judgment of a qualified clinician — unless
   and until the deployment has been independently validated and has obtained the
   regulatory clearance or approval required in the relevant jurisdiction.
2. In any way that violates applicable national, federal, state, local, or
   international law or regulation, including data-protection and human-subjects
   research law.
3. To exploit, harm, or attempt to exploit or harm minors or any vulnerable group
   in any way.
4. To generate or disseminate information presented as a fact about a person
   (including a health status or diagnosis) without that person's express,
   informed consent and an explicit disclosure that the information was produced
   by an automated model.
5. To discriminate against, or to deny, restrict, or impose adverse terms on
   individuals or groups on the basis of legally protected characteristics, or
   on the basis of an automated health inference.
6. To attempt to re-identify, or to infer additional protected attributes about,
   any individual in the training data or any other person.

The above restrictions are the operative behavioral terms; they sit alongside the
standard OpenRAIL-M restrictions in the canonical base text referenced above, which
are incorporated by reference where not in conflict with the above.

---

*Copyright (c) 2026 NIATS-UFU. Licensed under OpenRAIL-M as described above. The
no-clinical-use restriction (Attachment A.1) is added by the Licensor and is
specific to this medical model.*
