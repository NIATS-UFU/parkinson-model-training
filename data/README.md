# Local Data Layout

Large datasets are intentionally ignored by git. Keep downloaded or uploaded data under this directory and commit only this README plus code that knows how to read the layouts.

## Normalized Manifest

Training uses a CSV manifest with these required columns:

```csv
image_path,subject_id,label,dataset,drawing_type,source_label,label_source
data/handpd/SpiralPatients/0001-1.jpg,0001,PD,handpd,spiral,PD,filename
data/handpd/SpiralControl/0059-1.jpg,0059,HC,handpd,spiral,HC,filename
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
main-cli create-manifest handpd data/handpd data/manifests/handpd.csv
main-cli create-manifest newhandpd data/newhandpd data/manifests/newhandpd.csv
main-cli create-manifest hemato data/hemato data/manifests/hemato.csv
main-cli create-manifest kaggle-drawings data/kaggle/parkinsons-drawings data/manifests/kaggle_drawings.csv
main-cli create-manifest kaggle-spiral data/kaggle_2 data/manifests/kaggle_2.csv
main-cli create-manifest roboflow data/roboflow data/manifests/roboflow.csv
main-cli create-manifest folador data/folador data/manifests/folador.csv
```

The `hemato` loader reads only `**/5_final/*` image crops. Non-final processing folders should not be kept here.
The `kaggle-drawings` loader targets Kaggle `kmader/parkinsons-drawings`, including nested spiral/wave and train/test class folders. It uses image-level pseudo-subject IDs because the Kaggle files do not provide reliable participant identifiers.
The `roboflow` loader preserves the exported `train`/`valid`/`test` folder as metadata, but training still makes its own subject-level split from the original filename stem before `.rf.`.
The `folador` loader reads `dataset-junto` image folders and uses `HE...`, `HO...`, `PE...`, or `PO...` filename tokens as subject IDs.

## NIATS Cropper Metadata

The Obsidian UFU postdoc notes identify NIATS crops as the primary benchmark. The cropper metadata is available at:

```sh
/data/git/ufu/crop_spiral_tests/output/all_metadata.parquet
```

That file has crop paths and QC fields, but it still needs a clinical join that adds `subject_id` and `label` before `parkinson-model-train` can produce meaningful NIATS metrics. Until that join exists, use the external manifests above for reader smoke tests and comparison baselines.

## Downloaded References

- HandPD and NewHandPD: `https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/`
- Kaggle Parkinson's Drawings: `https://www.kaggle.com/datasets/kmader/parkinsons-drawings`
- PaHaW GitHub mirror: `https://github.com/musaru/PD_PaHaW`
- Roboflow export: `object detection parkinson`, CC BY 4.0, exported March 2, 2026.
- Folador HOG data/code: local `HOG-IA-desenho-manuscrito` archive.

PaHaW currently downloads as extracted CSV feature files in the referenced repository, not as drawing images for this image-classifier path.
