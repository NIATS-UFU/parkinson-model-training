# Local Data Layout

Large datasets are intentionally ignored by git. Keep downloaded or uploaded data under this directory and commit only this README plus code that knows how to read the layouts.

## Normalized Manifest

Training uses a CSV manifest with these required columns:

```csv
image_path,subject_id,label,dataset,drawing_type,source_label,label_source
data/hemato/V001/V001_PARKINSON_RAW_DATA_drawings_1/5_final/test3_spi_centered.png,V001,PD,hemato,spiral,PD,filename
data/hemato/V041/V041_CONTROL_RAW_DATA_drawings_1/5_final/test3_spi_centered.png,V041,HC,hemato,spiral,HC,filename
```

Required fields:

- `image_path`: image path consumed by the PyTorch dataset.
- `subject_id`: participant/form identifier used for subject-level splitting.
- `label`: normalized class, `PD` for Parkinson and `HC` for healthy/control.

Common metadata fields:

- `dataset`: source dataset name.
- `drawing_type`: `spiral`, `meander`, `circle`, `sentence`, or `unknown`.
- `source_label`: normalized label inferred from the source path or file name.
- `label_source`: currently `filename` for supported loaders.

## Supported Loaders

Create manifests with:

```sh
main-cli create-manifest hemato data/hemato data/manifests/hemato.csv
main-cli create-manifest niats-folador data/folador/HOG-IA-desenho-manuscrito/pyTremorDetect/dataset-junto data/manifests/niats_folador.csv
main-cli create-manifest image-tree data/my-labelled-images data/manifests/image_tree.csv
```

The `hemato` loader reads the **Moura** cohort — the NIATS-UFU clinical dataset, whose frozen key in
code is `hemato`. It reads only `**/5_final/*` image crops. It uses the top folder under the dataset
root as `subject_id` and preserves the final-crop parent folder as `drawing_session` metadata.
Non-final processing folders should not be kept here.
Example shape:

```text
data/hemato/V13/V13_PARKINSON_RAW_DATA_drawings_9/5_final/test1_spi_centered.png
```

This loader supports the `spiral-cropper` output tree only. The de-identified
`moura-neto-2023-2025` release ships a `V<n>/{spiral,wave}/` tree with no `5_final/` directories, so
pointing this loader at it raises `No image records found under: <root>`. That release layout is read
by the separate `paper_reproduction` loader — see
[`../paper_reproduction/docs/DATA.md`](../paper_reproduction/docs/DATA.md).

The `niats-folador` loader reads either Folador layout:

```text
dataset-junto/espiral/treinamento/parkinson/1_2_PE3.jpg
dataset-separado/dia2/ondas/teste/higido/2_5_HO1.jpg
```

It derives `subject_id` from the drawing code plus patient number, such as `PE-2` or `HO-5`, so repeated collections from different days stay grouped by participant rather than by crop.
The `image-tree` loader is a generic fallback. It scans recursively, infers labels from path text, and uses the image parent folder as `subject_id`.

## Cropper Metadata

Older Moura distributions ship a single `all_metadata.parquet` alongside the crops, carrying crop
paths and QC fields. The `paper_reproduction` loader joins it to `subject_id` and `class_label`
automatically when it is present; see
[`../paper_reproduction/docs/DATA.md`](../paper_reproduction/docs/DATA.md) for where to place it.

The root `main-cli` loaders above do not read that parquet — they infer everything from the directory
layout.
