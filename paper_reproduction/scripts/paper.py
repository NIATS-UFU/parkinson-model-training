"""Consolidated CLI for paper reproduction tasks.

Usage:
    python scripts/paper.py build-manifest --data-root <path> --out results/manifest.parquet
    python scripts/paper.py reproduce --manifest results/manifest.parquet --out results
    python scripts/paper.py regenerate --results results --out paper_artifacts
    python scripts/paper.py verify-catalog --results results
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parkinson_spiral.analysis as A  # noqa: E402
from parkinson_spiral.catalog import EXPECTED_N_CELLS, all_cells  # noqa: E402
from parkinson_spiral.pipelines import get_pipeline_class  # noqa: E402
from parkinson_spiral.runner import RunPaths, contiguous_shard, run_cell  # noqa: E402

SOURCES = ("kaggle", "folador", "hemato")
SHORT = {"kaggle": "K", "folador": "F", "hemato": "H"}

# (pipeline_name, recipe, pipeline_kwargs-marker) per native configuration.
NATIVE = [
    ("dinov2_probe", "AUG-None", "DINOv2 (AUG-None)"),
    ("folador_hog", "AUG-None", "Folador HOG (AUG-None)"),
    ("huang_vgg19", "AUG-Huang", "Huang VGG19 (AUG-Huang)"),
    ("dogan_triple", "AUG-Dogan", "Dogan triple (AUG-Dogan)"),
]

# H1 "Published" column = each source paper's own reported accuracy (external literature
# constants, NOT computed here). Kept verbatim from conference_v2.tex with its footnotes.
PUBLISHED_LITERATURE = {
    "huang_vgg19": ("0.87", "a", "spiral, single split"),
    "dogan_triple": ("0.994", "b", "spiral+wave combined, single split (upper bound)"),
    "folador_hog": ("0.696", "c", "spiral, 5-fold CV (sinusoidal best 0.831)"),
}

# Short training-regime descriptions for the H4 cost table (conference_v2.tex Table h4-cost).
TRAINING_REGIME = {
    "dinov2_probe": "frozen backbone; single-pass features + 1 logistic fit",
    "folador_hog": "deterministic HOG + classifier fit (no epochs)",
    "dogan_triple": "3 frozen public backbones + NCA + linear SVM",
    "huang_vgg19": "end-to-end fine-tune, 50 epochs",
}
GPU_PIPELINES = {"huang_vgg19", "dogan_triple"}
PREDICTION_COLUMNS = {
    "cell_id",
    "image_id",
    "subject_id",
    "session_id",
    "dataset",
    "drawing_type",
    "y_true",
    "y_proba_pos",
}


def build_manifest_command(args: argparse.Namespace) -> int:
    """Build the unified manifest and assign subject-level CV folds."""
    from parkinson_spiral.data import assign_subject_folds, build_unified_manifest

    manifest = build_unified_manifest(args.data_root)
    if manifest.empty:
        raise SystemExit(f"no images found under {args.data_root}. Check the layout in docs/DATA.md.")
    manifest = assign_subject_folds(manifest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.out, index=False)

    by_ds = manifest.groupby("dataset").size().to_dict()
    print(f"wrote {len(manifest)} rows -> {args.out}")
    print(f"datasets: {by_ds}")
    print(f"spiral rows: {int((manifest['drawing_type'] == 'spiral').sum())}")
    return 0


def reproduce_command(args: argparse.Namespace) -> int:
    """Run the selected experiment cells from the catalog."""
    cells = all_cells()
    if args.pipeline:
        keep = set(args.pipeline)
        cells = [c for c in cells if c.pipeline_name in keep]
    if args.cell_id:
        requested = set(args.cell_id)
        available = {cell.cell_id for cell in cells}
        unknown = sorted(requested - available)
        if unknown:
            raise SystemExit(f"unknown or filtered-out cell ids: {unknown}")
        cells = [cell for cell in cells if cell.cell_id in requested]
    cells = contiguous_shard(cells, shard_count=args.shard_count, shard_index=args.shard_index)

    shard = f" (shard {args.shard_index + 1}/{args.shard_count})" if args.shard_count > 1 else ""
    print(f"{len(cells)} cells selected{shard}")
    if args.dry_run:
        for c in cells:
            gpu = " [GPU]" if c.pipeline_name in GPU_PIPELINES else ""
            fold = f" fold={c.cv_fold}" if c.cv_fold is not None else ""
            print(
                f"  {c.cell_id}  {c.pipeline_name:13s} {c.recipe_name:9s} "
                f"{c.train_source}->{c.test_source} [{c.split_source}]{fold}{gpu}"
            )
        return 0

    if not args.manifest.is_file():
        raise SystemExit(
            f"manifest not found: {args.manifest}. Build one with `python scripts/paper.py build-manifest` "
            "(Layer-2 re-training needs the datasets on disk; see docs/REPRODUCE.md)."
        )

    manifest = pd.read_parquet(args.manifest, columns=["dataset"])
    available = set(manifest["dataset"].dropna().astype(str).unique())
    runnable = [c for c in cells if c.train_source in available and c.test_source in available]
    skipped = len(cells) - len(runnable)
    if skipped:
        missing = sorted({source for c in cells for source in (c.train_source, c.test_source)} - available)
        print(f"skipping {skipped} cells because these datasets are absent from the manifest: {missing}")
    cells = runnable
    if not cells:
        raise SystemExit("no selected cells can run with the datasets present in the manifest")

    paths = RunPaths(manifest=args.manifest, outputs_dir=args.out)
    for i, spec in enumerate(cells, 1):
        row = run_cell(spec, paths, force=args.force)
        acc = row.get("subject_accuracy")
        acc_s = f"{acc:.3f}" if isinstance(acc, (int, float)) else "n/a"
        print(
            f"[{i}/{len(cells)}] {spec.pipeline_name} {spec.recipe_name} "
            f"{spec.train_source}->{spec.test_source} -> subj_acc={acc_s}"
        )

    print(f"done -> {args.out}")
    return 0


def _fmt_ci(res: dict, nd: int = 2) -> str:
    return f"{res['accuracy']:.3f} [{res['ci_lo']:.{nd}f}-{res['ci_hi']:.{nd}f}]"


def _n(x: float) -> str:
    """Round a coordinate to 3 dp for clean TikZ output."""
    return f"{x:.3f}"


# ---------------------------------------------------------------- tables


def table_b1_fidelity(cells, preds) -> str:
    rows = []
    rels = {
        "huang_vgg19": ("AUG-Huang", "kaggle", "Huang VGG19 (Kaggle)"),
        "dogan_triple": ("AUG-Dogan", "kaggle", "Dogan triple (Kaggle)"),
        "folador_hog": ("AUG-None", "folador", "Folador HOG-SVM (Folador)"),
    }
    for pipe, (recipe, source, label) in rels.items():
        pub, fn, _ = PUBLISHED_LITERATURE[pipe]
        released = A.subject_accuracy(
            cells,
            preds,
            bootstrap=False,
            pipeline_name=pipe,
            recipe_name=recipe,
            train_source=source,
            split_source="published",
        )
        ceiling = A.diagonal_ceiling(cells, preds, pipe, recipe, source, bootstrap=False)
        rows.append({
            "Pipeline (dataset)": label,
            "Published": f"{pub} ({fn})",
            "Released-split": f"{released['point']:.3f}",
            "Grouped-CV ceiling": f"{ceiling['point']:.3f}",
        })
    md = pd.DataFrame(rows).to_markdown(index=False)
    notes = "  ".join(f"({fn}) {note}" for _, (_, fn, note) in PUBLISHED_LITERATURE.items())
    return "### Table: H1 reproduction fidelity (`b1-fidelity`)\n\n" + md + f"\n\n_Published footnotes:_ {notes}\n"


def table_cv_ceiling(cells, preds) -> str:
    rows = []
    for pipe, recipe, label in NATIVE:
        for source in SOURCES:
            res = A.subject_suite(
                cells,
                preds,
                bootstrap=True,
                pipeline_name=pipe,
                recipe_name=recipe,
                train_source=source,
                test_source=source,
                split_source="cv-fold",
            )
            rows.append({
                "Pipeline (recipe)": label,
                "Dataset": source.capitalize(),
                "Subj. acc. [95% CI]": _fmt_ci(res, nd=3),
                "AUC": f"{res['auc']:.3f}",
            })
    md = pd.DataFrame(rows).to_markdown(index=False)
    return "### Table: in-domain CV ceilings (`cv-ceiling`)\n\n" + md + "\n"


def table_cross_dataset(cells, preds) -> str:
    blocks = []
    for pipe, recipe, _ in NATIVE:
        label = {
            "dinov2_probe": "DINOv2 AUG-None",
            "huang_vgg19": "Huang AUG-Huang",
            "dogan_triple": "Dogan AUG-Dogan",
            "folador_hog": "Folador HOG AUG-None",
        }[pipe]
        rows = []
        accs = []
        for train in SOURCES:
            for test in SOURCES:
                if train == test:
                    continue
                res = A.subject_suite(
                    cells,
                    preds,
                    bootstrap=True,
                    pipeline_name=pipe,
                    recipe_name=recipe,
                    train_source=train,
                    test_source=test,
                    split_source="cross-dataset",
                )
                accs.append(res["accuracy"])
                rows.append({
                    "Pipeline": label,
                    "Dir": f"{SHORT[train]}->{SHORT[test]}",
                    "Acc [95% CI]": _fmt_ci(res, nd=2),
                    "AUC": f"{res['auc']:.3f}",
                    "Sens": f"{res['sensitivity']:.2f}",
                    "Spec": f"{res['specificity']:.2f}",
                    "Prec": f"{res['precision']:.2f}",
                    "F1": f"{res['f1']:.2f}",
                })
        rows.append({
            "Pipeline": label,
            "Dir": "mean",
            "Acc [95% CI]": f"{np.mean(accs):.3f}",
            "AUC": "",
            "Sens": "",
            "Spec": "",
            "Prec": "",
            "F1": "",
        })
        blocks.append(pd.DataFrame(rows).to_markdown(index=False))
    return "### Table: cross-dataset off-diagonals (`cross-dataset`)\n\n" + "\n\n".join(blocks) + "\n"


def _hemato_fold_sigma(cells, pipe, recipe) -> float:
    sub = cells[
        (cells.pipeline_name == pipe)
        & (cells.recipe_name == recipe)
        & (cells.split_source == "cv-fold")
        & (cells.train_source == "hemato")
    ]
    return float(np.std(sub["subject_accuracy"].to_numpy(), ddof=1))


def table_h4_cost(cells, preds) -> str:
    rows = []
    order = ["dinov2_probe", "folador_hog", "dogan_triple", "huang_vgg19"]
    recipe_of = {p: r for p, r, _ in NATIVE}
    for pipe in order:
        recipe = recipe_of[pipe]
        n_cfg = len(get_pipeline_class(pipe).FREE_HYPERPARAMETERS)
        ceiling = A.diagonal_ceiling(cells, preds, pipe, recipe, "hemato", bootstrap=False)
        sigma = _hemato_fold_sigma(cells, pipe, recipe)
        label = {
            "dinov2_probe": "DINOv2 (frozen) AUG-None",
            "folador_hog": "Folador HOG AUG-None",
            "dogan_triple": "Dogan triple AUG-Dogan",
            "huang_vgg19": "Huang VGG19 AUG-Huang",
        }[pipe]
        rows.append({
            "Pipeline (recipe)": label,
            "# cfg": n_cfg,
            "Training regime": TRAINING_REGIME[pipe],
            "Hemato CV": f"{ceiling['point']:.3f} (sigma {sigma:.3f})",
        })
    md = pd.DataFrame(rows).to_markdown(index=False)
    # The augmentation-robustness swing (strong form of H4).
    swing = [
        A.diagonal_ceiling(cells, preds, "dinov2_probe", r, "hemato", bootstrap=False)["point"]
        for r in ("AUG-None", "AUG-Huang", "AUG-Dogan")
    ]
    note = (
        f"\n\n_DINOv2 Hemato aug swing:_ AUG-None {swing[0]:.3f} -> AUG-Huang {swing[1]:.3f} "
        f"-> AUG-Dogan {swing[2]:.3f}\n"
    )
    return "### Table: reproducibility cost (`h4-cost`)\n\n" + md + note


# ---------------------------------------------------------------- figures (TikZ)


def fig1_matrices(cells, preds) -> str:
    """2x2 grid of 3x3 cross-dataset accuracy matrices, one per pipeline."""
    lines = [
        "% Auto-generated by make_paper_tables_figures.py — cross-dataset matrices (fig1)",
        "\\begin{tikzpicture}[font=\\scriptsize]",
    ]
    titles = {
        "dinov2_probe": "DINOv2 AUG-None",
        "huang_vgg19": "Huang AUG-Huang",
        "dogan_triple": "Dogan AUG-Dogan",
        "folador_hog": "Folador AUG-None",
    }
    positions = {"dinov2_probe": (0, 0), "huang_vgg19": (5.2, 0), "dogan_triple": (0, -4.6), "folador_hog": (5.2, -4.6)}
    cell = 1.2
    for pipe, recipe, _ in NATIVE:
        ox, oy = positions[pipe]
        m = A.cross_matrix(cells, preds, pipe, recipe, bootstrap=False)
        lines.append(f"  % {titles[pipe]}")
        lines.append(
            f"  \\node[anchor=south west, font=\\scriptsize\\bfseries] at ({ox},{oy + 0.15}) {{{titles[pipe]}}};"
        )
        for i, train in enumerate(SOURCES):
            for j, test in enumerate(SOURCES):
                val = m.iloc[i][test]
                x = ox + j * cell
                y = oy - i * cell
                # colour ramp: chance (0.5) white -> 1.0 strong.
                t = max(0.0, min(1.0, (float(val) - 0.45) / 0.55))
                fill = f"green!{int(t * 60)}!white"
                lines.append(
                    f"  \\node[draw, minimum width={cell}cm, minimum height={cell}cm, "
                    f"fill={fill}, anchor=north west] at ({_n(x)},{_n(y)}) {{{val:.3f}}};"
                )
            # row label (train)
            lines.append(
                f"  \\node[anchor=east] at ({_n(ox - 0.05)},{_n(oy - i * cell - cell / 2)}) {{{SHORT[train]}}};"
            )
        for j, test in enumerate(SOURCES):  # column labels (test)
            lines.append(
                f"  \\node[anchor=south] at ({_n(ox + j * cell + cell / 2)},{_n(oy - 3 * cell)}) {{{SHORT[test]}}};"
            )
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def fig2_asymmetry(cells, preds) -> str:
    """Per-pipeline asymmetry index: mean(H->{K,F}) - mean({K,F}->H)."""
    lines = [
        "% Auto-generated by make_paper_tables_figures.py — asymmetry index (fig2)",
        "\\begin{tikzpicture}[font=\\scriptsize]",
    ]
    labels = {"dinov2_probe": "DINOv2", "huang_vgg19": "Huang", "dogan_triple": "Dogan", "folador_hog": "Folador"}
    bar_w = 0.8
    gap = 1.4
    scale = 6.0  # units per accuracy point on the x-ish axis
    lines.append("  \\draw[->] (0,0) -- (0,5) node[above] {pipeline};")
    lines.append("  \\draw (0,0) -- (0,0);")
    for k, (pipe, recipe, _) in enumerate(NATIVE):
        into = np.mean([
            A.cross_value(cells, preds, pipe, recipe, "hemato", t, bootstrap=False)["point"]
            for t in ("kaggle", "folador")
        ])
        outof = np.mean([
            A.cross_value(cells, preds, pipe, recipe, s, "hemato", bootstrap=False)["point"]
            for s in ("kaggle", "folador")
        ])
        idx = into - outof
        y = 4.0 - k * gap
        length = idx * scale
        colour = "blue!60" if idx >= 0 else "red!60"
        label_x = length + (0.1 if idx >= 0 else -0.1)
        anchor = "west" if idx >= 0 else "east"
        lines.append(f"  \\node[anchor=east] at (-0.1,{_n(y)}) {{{labels[pipe]}}};")
        lines.append(f"  \\fill[{colour}] (0,{_n(y - bar_w / 2)}) rectangle ({_n(length)},{_n(y + bar_w / 2)});")
        lines.append(f"  \\node[anchor={anchor}] at ({_n(label_x)},{_n(y)}) {{{idx:+.3f}}};")
    lines.append(
        "  \\node[anchor=north] at (1.5,-0.4) {asymmetry index = mean(H$\\to\\{$K,F$\\}$) $-$ mean($\\{$K,F$\\}\\to$H)};"
    )
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- commands


def regenerate_command(args: argparse.Namespace) -> int:
    """Regenerate paper tables and TikZ figures from local result parquet files."""
    cells, preds = A.load_results(args.results)
    out = args.out
    (out / "figures" / "tikz").mkdir(parents=True, exist_ok=True)

    tables = "\n\n".join([
        "# Paper tables (regenerated from results/)\n",
        table_b1_fidelity(cells, preds),
        table_cv_ceiling(cells, preds),
        table_cross_dataset(cells, preds),
        table_h4_cost(cells, preds),
    ])
    (out / "results_tables.md").write_text(tables, encoding="utf-8")
    (out / "figures" / "tikz" / "fig1_crossdataset_matrices.tex").write_text(
        fig1_matrices(cells, preds), encoding="utf-8"
    )
    (out / "figures" / "tikz" / "fig2_asymmetry.tex").write_text(fig2_asymmetry(cells, preds), encoding="utf-8")

    print(f"wrote {out / 'results_tables.md'}")
    print(f"wrote {out / 'figures' / 'tikz' / 'fig1_crossdataset_matrices.tex'}")
    print(f"wrote {out / 'figures' / 'tikz' / 'fig2_asymmetry.tex'}")
    return 0


def verify_catalog_command(args: argparse.Namespace) -> int:
    """Validate that cells.parquet and prediction files match the declared catalog."""
    cells_path = args.results / "cells.parquet"
    if not cells_path.is_file():
        print(f"ERROR: {cells_path} not found", file=sys.stderr)
        return 1

    catalog = all_cells()
    catalog_ids = {c.cell_id for c in catalog}
    result_ids = set(pd.read_parquet(cells_path)["cell_id"])

    ok = True
    if len(catalog) != EXPECTED_N_CELLS:
        print(f"FAIL: catalog has {len(catalog)} cells, expected {EXPECTED_N_CELLS}")
        ok = False
    missing = catalog_ids - result_ids
    extra = result_ids - catalog_ids
    if missing:
        print(f"FAIL: {len(missing)} catalog cells missing from results: {sorted(missing)[:5]}")
        ok = False
    if extra:
        print(f"FAIL: {len(extra)} result cells not in catalog: {sorted(extra)[:5]}")
        ok = False

    predictions_dir = args.results / "predictions"
    if not predictions_dir.is_dir():
        print(f"FAIL: predictions directory not found: {predictions_dir}")
        ok = False
    else:
        for cell_id in sorted(catalog_ids & result_ids):
            pred_path = predictions_dir / f"{cell_id}.parquet"
            if not pred_path.is_file():
                print(f"FAIL: prediction parquet missing for cell {cell_id}: {pred_path}")
                ok = False
                continue
            pred = pd.read_parquet(pred_path)
            missing_columns = PREDICTION_COLUMNS - set(pred.columns)
            if missing_columns:
                print(f"FAIL: {pred_path} missing columns: {sorted(missing_columns)}")
                ok = False
            if pred.empty:
                print(f"FAIL: {pred_path} has no prediction rows")
                ok = False
            pred_ids = set(pred["cell_id"].astype(str).unique()) if "cell_id" in pred.columns else set()
            if pred_ids != {cell_id}:
                print(f"FAIL: {pred_path} contains cell_ids {sorted(pred_ids)}, expected only {cell_id}")
                ok = False

    if ok:
        print(f"PASS: catalog and results agree on all {len(catalog_ids)} cell_ids.")
        return 0
    return 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def _git_value(repo: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _command_output(*args: str) -> str | None:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _hardware_profile() -> dict[str, str]:
    raw = _command_output("system_profiler", "SPHardwareDataType")
    if raw is None:
        return {}
    allowed = {"Model Name", "Model Identifier", "Chip", "Total Number of Cores", "Memory"}
    profile = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in allowed:
            profile[key] = value
    return profile


def freeze_evidence_command(args: argparse.Namespace) -> int:
    """Freeze manifest, folds, catalog, environment, and artifact checksums."""
    repo = Path(__file__).resolve().parents[2]
    manifest = pd.read_parquet(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)

    subject_folds = (
        manifest[["dataset", "subject_id", "class_label", "cv_fold"]]
        .drop_duplicates()
        .sort_values(["dataset", "subject_id"])
    )
    conflicting = subject_folds.groupby(["dataset", "subject_id"]).size()
    if (conflicting > 1).any():
        raise ValueError("manifest assigns conflicting labels or CV folds to at least one subject")
    subject_folds.to_csv(args.out / "subject_folds.csv", index=False)
    if args.historical_manifest.is_file():
        historical = pd.read_parquet(args.historical_manifest)
        historical["dataset"] = historical["dataset"].replace({"niats": "folador"})
        historical["subject_id"] = (
            historical["subject_id"]
            .astype(str)
            .str.replace(
                "niats:",
                "folador:",
                regex=False,
            )
        )
        historical_folds = (
            historical[["dataset", "subject_id", "class_label", "cv_fold"]]
            .drop_duplicates()
            .sort_values(["dataset", "subject_id"])
        )
        historical_folds.to_csv(args.out / "historical_subject_folds.csv", index=False)
        comparison = historical_folds.merge(
            subject_folds,
            on=["dataset", "subject_id", "class_label"],
            suffixes=("_historical", "_current"),
            validate="one_to_one",
        )
        comparison["changed"] = comparison["cv_fold_historical"] != comparison["cv_fold_current"]
        fold_comparison = {
            "historical_manifest": _display_path(args.historical_manifest, repo),
            "historical_manifest_sha256": _sha256(args.historical_manifest),
            "subjects_compared": len(comparison),
            "subjects_changed": int(comparison["changed"].sum()),
            "by_dataset": {
                dataset: {
                    "subjects": len(group),
                    "subjects_changed": int(group["changed"].sum()),
                }
                for dataset, group in comparison.groupby("dataset")
            },
        }
        (args.out / "fold_comparison.json").write_text(
            json.dumps(fold_comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    catalog = [{"cell_id": spec.cell_id, **asdict(spec)} for spec in all_cells()]
    (args.out / "catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_configurations = {
        name: {
            "free_hyperparameters": get_pipeline_class(name).FREE_HYPERPARAMETERS,
            "methods_notes": get_pipeline_class(name).METHODS_SECTION_NOTES,
        }
        for name in sorted({spec.pipeline_name for spec in all_cells()})
    }
    (args.out / "model_configurations.json").write_text(
        json.dumps(model_configurations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    packages = {}
    for name in [
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scikit-image",
        "pillow",
        "torch",
        "torchvision",
        "timm",
        "tensorflow",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    accelerator: dict[str, bool | str | None] = {"torch_available": packages["torch"] is not None}
    if packages["torch"] is not None:
        import torch

        accelerator.update({
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        })

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "accelerator": accelerator,
        "git_commit": _git_value(repo, "rev-parse", "HEAD"),
        "git_branch": _git_value(repo, "branch", "--show-current"),
        "git_status": _git_value(repo, "status", "--short"),
        "manifest_rows": len(manifest),
        "manifest_sha256": _sha256(args.manifest),
        "catalog_cells": len(catalog),
        "hardware_profile": _hardware_profile(),
        "run_device_policy": {
            "dinov2_probe": "PyTorch MPS",
            "huang_vgg19": "PyTorch MPS, full precision",
            "dogan_triple_torch": "PyTorch MPS",
            "dogan_triple_nasnetmobile": "TensorFlow CPU",
            "folador_hog": "CPU",
        },
        "tensorflow_devices": _command_output(
            str(Path(sys.executable)),
            "-c",
            "import tensorflow as tf; print(tf.config.list_physical_devices())",
        )
        if packages["tensorflow"] is not None
        else None,
    }
    (args.out / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    repository_patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    (args.out / "repository.patch").write_text(repository_patch, encoding="utf-8")

    if args.manuscript.is_file():
        shutil.copy2(args.manuscript, args.out / args.manuscript.name)

    checksum_paths = [args.manifest, repo / "uv.lock", repo / "paper_reproduction" / "uv.lock"]
    checksum_paths.extend([
        args.out / "catalog.json",
        args.out / "model_configurations.json",
        args.out / "subject_folds.csv",
        args.out / "environment.json",
        args.out / "repository.patch",
    ])
    if args.historical_manifest.is_file():
        checksum_paths.extend([
            args.historical_manifest,
            args.out / "historical_subject_folds.csv",
            args.out / "fold_comparison.json",
        ])
    if args.manuscript.is_file():
        checksum_paths.append(args.out / args.manuscript.name)
    if args.results.is_dir():
        checksum_paths.extend(sorted(path for path in args.results.rglob("*") if path.is_file()))
    unique_checksum_paths = dict.fromkeys(path.resolve() for path in checksum_paths if path.is_file())
    checksum_lines = [
        f"{_sha256(path)}  {_display_path(path, repo)}"
        for path in unique_checksum_paths
        if path != (args.out / "checksums.sha256").resolve()
    ]
    (args.out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    readme = (
        "# Frozen reproduction evidence\n\n"
        "Generated by `scripts/paper.py freeze-evidence`.\n\n"
        "- `environment.json`: runtime, package, accelerator, git, and manifest provenance.\n"
        "- `subject_folds.csv`: immutable subject label and CV-fold assignment.\n"
        "- `historical_subject_folds.csv` and `fold_comparison.json`: prior manuscript fold map and drift audit, when supplied.\n"
        "- `catalog.json`: canonical 97-cell experiment specification.\n"
        "- `model_configurations.json`: declared hyperparameters and methods notes.\n"
        "- `repository.patch`: exact uncommitted code state relative to the recorded commit.\n"
        "- manuscript PDF: frozen submitted manuscript snapshot when supplied.\n"
        "- `checksums.sha256`: hashes for lockfiles, manifest, and generated result artifacts.\n"
    )
    (args.out / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote frozen evidence -> {args.out}")
    return 0


def dataset_fingerprint_command(args: argparse.Namespace) -> int:
    """Hash every image in the manifest and write a per-image fingerprint CSV.

    The fingerprint records image_id, metadata, cv_fold, and the SHA-256 of the
    image file bytes. A collaborator with the same raw data can re-run
    verify-fingerprint to confirm their copy matches byte-for-byte.
    """
    manifest_path = args.manifest
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = pd.read_parquet(manifest_path)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    n = len(manifest)
    for i, (_, row) in enumerate(manifest.iterrows(), 1):
        if i % 500 == 0 or i == n:
            print(f"\r  hashing {i}/{n}...", end="", flush=True)
        fp = Path(str(row["file_path"]))
        sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        rows.append({
            "image_id": row["image_id"],
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "class_label": row["class_label"],
            "drawing_type": row["drawing_type"],
            "cv_fold": row["cv_fold"],
            "sha256": sha,
        })
    print()

    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    out_csv = args.out / "dataset_fingerprint.csv"
    df.to_csv(out_csv, index=False)

    fingerprint_hash = hashlib.sha256(out_csv.read_bytes()).hexdigest()

    summary: dict = {
        "fingerprint_sha256": fingerprint_hash,
        "total_images": int(len(df)),
        "datasets": {},
    }
    # session_id is present for hemato; for others derive a session count from the
    # parent directory of the logical image path (second-to-last path component).
    manifest_indexed = manifest.set_index("image_id")
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        m_sub = manifest_indexed[manifest_indexed["dataset"] == ds]
        session_col = "session_id" if "session_id" in m_sub.columns else None
        if session_col and not m_sub[session_col].isna().all():
            n_sessions = int(m_sub[session_col].nunique())
        else:
            n_sessions = None
        summary["datasets"][ds] = {
            "images": int(len(sub)),
            "subjects": int(sub["subject_id"].nunique()),
            "sessions": n_sessions,
            "class_counts": {k: int(v) for k, v in sub.groupby("class_label")["subject_id"].nunique().items()},
            "fold_distribution": {
                str(int(k)): int(v)
                for k, v in sub.dropna(subset=["cv_fold"]).groupby("cv_fold")["subject_id"].nunique().items()
            },
        }

    out_summary = args.out / "dataset_fingerprint_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {out_csv}  ({len(df)} rows)")
    print(f"wrote {out_summary}")
    print(f"fingerprint SHA-256: {fingerprint_hash}")
    return 0


def verify_fingerprint_command(args: argparse.Namespace) -> int:
    """Verify the local dataset matches the committed fingerprint.

    Build your manifest first:
        python scripts/paper.py build-manifest --data-root <your-root>

    Then run:
        python scripts/paper.py verify-fingerprint

    Every image in the committed fingerprint is looked up by image_id in the
    local manifest, hashed, and compared. Reports: matched, missing, sha256
    mismatch, fold mismatch, extra images in local manifest not in fingerprint.
    """
    fp_path = args.evidence / "dataset_fingerprint.csv"
    summary_path = args.evidence / "dataset_fingerprint_summary.json"
    if not fp_path.is_file():
        print(f"ERROR: fingerprint not found: {fp_path}", file=sys.stderr)
        print("       run: python scripts/paper.py dataset-fingerprint", file=sys.stderr)
        return 1

    manifest_path = args.manifest
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        print("       run: python scripts/paper.py build-manifest --data-root <path>", file=sys.stderr)
        return 1

    expected = pd.read_csv(fp_path)
    manifest = pd.read_parquet(manifest_path)

    local_by_id = manifest.set_index("image_id")

    matched = sha_mismatch = fold_mismatch = missing = extra = 0
    failures: list[str] = []

    n = len(expected)
    for i, (_, row) in enumerate(expected.iterrows(), 1):
        if i % 500 == 0 or i == n:
            print(f"\r  checking {i}/{n}...", end="", flush=True)

        iid = row["image_id"]
        if iid not in local_by_id.index:
            missing += 1
            failures.append(f"MISSING  {iid}")
            continue

        local = local_by_id.loc[iid]
        fp = Path(str(local["file_path"]))
        if not fp.is_file():
            missing += 1
            failures.append(f"FILE_NOT_FOUND  {iid}  ({fp})")
            continue

        local_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        if local_sha != row["sha256"]:
            sha_mismatch += 1
            failures.append(f"SHA256_MISMATCH  {iid}")
            continue

        # check fold assignment for in-domain (cv_fold is set)
        expected_fold = row["cv_fold"]
        local_fold = local["cv_fold"]
        if not (pd.isna(expected_fold) and pd.isna(local_fold)):
            if str(expected_fold) != str(local_fold):
                fold_mismatch += 1
                failures.append(f"FOLD_MISMATCH  {iid}  expected={expected_fold} got={local_fold}")
                continue

        matched += 1

    print()

    expected_ids = set(expected["image_id"])
    local_ids = set(manifest["image_id"])
    extra = len(local_ids - expected_ids)

    print("\nDataset fingerprint verification")
    print(f"  fingerprint: {fp_path}")
    print(f"  manifest:    {manifest_path}")
    print(f"  expected images:  {n}")
    print(f"  local images:     {len(manifest)}")
    print()
    print(f"  matched:          {matched}")
    print(f"  missing:          {missing}")
    print(f"  sha256 mismatch:  {sha_mismatch}")
    print(f"  fold mismatch:    {fold_mismatch}")
    print(f"  extra (local only): {extra}")

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        print(f"\n  committed fingerprint SHA-256: {summary['fingerprint_sha256']}")

    if failures:
        print("\nFirst failures (up to 20):")
        for f in failures[:20]:
            print(f"  {f}")

    all_ok = missing == 0 and sha_mismatch == 0 and fold_mismatch == 0
    if all_ok and extra == 0:
        print("\nPASS: dataset is byte-identical to the committed fingerprint.")
    elif all_ok and extra > 0:
        print(f"\nPASS (with extras): all committed images match; {extra} additional images in local manifest.")
    else:
        print(f"\nFAIL: {missing + sha_mismatch + fold_mismatch} discrepancies.")
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-manifest", help="build unified manifest from datasets on disk")
    build.add_argument("--data-root", type=Path, required=True, help="directory holding kaggle/ folador/ hemato/")
    build.add_argument("--out", type=Path, default=repo / "results" / "manifest.parquet")
    build.set_defaults(func=build_manifest_command)

    reproduce = sub.add_parser("reproduce", help="run catalog cells from scratch")
    reproduce.add_argument("--manifest", type=Path, default=repo / "results" / "manifest.parquet")
    reproduce.add_argument("--out", type=Path, default=repo / "runs" / "reproduce")
    reproduce.add_argument("--pipeline", action="append", default=None, help="restrict to these pipeline names")
    reproduce.add_argument("--cell-id", action="append", default=None, help="restrict to exact catalog cell ids")
    reproduce.add_argument("--shard-count", type=int, default=1, help="split selected cells into contiguous shards")
    reproduce.add_argument("--shard-index", type=int, default=0, help="zero-based shard to run")
    reproduce.add_argument("--dry-run", action="store_true", help="list cells and exit without training")
    reproduce.add_argument("--force", action="store_true", help="recompute even if cached")
    reproduce.set_defaults(func=reproduce_command)

    regenerate = sub.add_parser("regenerate", help="regenerate paper tables and figures from local results")
    regenerate.add_argument("--results", type=Path, default=repo / "results")
    regenerate.add_argument("--out", type=Path, default=repo / "paper_artifacts")
    regenerate.set_defaults(func=regenerate_command)

    verify = sub.add_parser("verify-catalog", help="verify catalog cells and prediction parquet files")
    verify.add_argument("--results", type=Path, default=repo / "results")
    verify.set_defaults(func=verify_catalog_command)

    freeze = sub.add_parser("freeze-evidence", help="freeze manifest, folds, environment, and artifact checksums")
    freeze.add_argument("--manifest", type=Path, default=repo / "results" / "manifest.parquet")
    freeze.add_argument("--results", type=Path, default=repo / "results")
    freeze.add_argument("--out", type=Path, default=repo / "evidence")
    freeze.add_argument("--manuscript", type=Path, default=Path(""))
    freeze.add_argument("--historical-manifest", type=Path, default=Path(""))
    freeze.set_defaults(func=freeze_evidence_command)

    fingerprint = sub.add_parser(
        "dataset-fingerprint",
        help="hash every image in the manifest and write evidence/dataset_fingerprint.csv",
    )
    fingerprint.add_argument("--manifest", type=Path, default=repo / "results" / "manifest.parquet")
    fingerprint.add_argument("--out", type=Path, default=repo / "evidence")
    fingerprint.set_defaults(func=dataset_fingerprint_command)

    verify_fp = sub.add_parser(
        "verify-fingerprint",
        help="verify local dataset matches the committed fingerprint (build-manifest first)",
    )
    verify_fp.add_argument("--manifest", type=Path, default=repo / "results" / "manifest.parquet")
    verify_fp.add_argument("--evidence", type=Path, default=repo / "evidence")
    verify_fp.set_defaults(func=verify_fingerprint_command)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
