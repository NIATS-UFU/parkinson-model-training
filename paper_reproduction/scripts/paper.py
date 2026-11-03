"""Consolidated CLI for paper reproduction tasks.

Usage:
    python scripts/paper.py build-manifest --data-root <path> --out results/manifest.parquet
    python scripts/paper.py reproduce --manifest results/manifest.parquet --out results
    python scripts/paper.py reproduce --manifest results/manifest.parquet --out results --save-models models
    python scripts/paper.py verify-models --models models --manifest results/manifest.parquet --results results
    python scripts/paper.py statistics --results results --out results/statistics
    python scripts/paper.py distances --embeddings results/embeddings.parquet --out results/distances --figure
    python scripts/paper.py regenerate --results results --out paper_artifacts
    python scripts/paper.py verify-catalog --results results
    python scripts/paper.py export-embeddings --manifest results/manifest.parquet \
        --out results/embeddings.parquet
    python scripts/paper.py cost-profile --manifest results/manifest.parquet \
        --out results/cost_profile.json --compare results
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
from parkinson_spiral.protocol import CellSpec  # noqa: E402
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


def _cuda_requirement_error(
    *,
    gpu_pipelines: list[str],
    cuda_available: bool,
    torch_cuda_version: str | None,
) -> str | None:
    """Pure decision behind `_require_cuda`: the abort message, or None to proceed.

    Kept separate from the host reads so every branch is testable on a machine
    without a GPU. The two causes need different fixes, so they are reported apart:
    a CPU-only wheel is a reinstall, a missing device is a driver problem.
    """
    if not gpu_pipelines or cuda_available:
        return None
    if torch_cuda_version is None:
        cause = (
            "the installed torch is a CPU-only build (torch.version.cuda is None). Reinstall the "
            "CUDA build with `python -m pip install -r requirements-lock.txt`."
        )
    else:
        cause = (
            f"torch is built against CUDA {torch_cuda_version} but no device is visible. "
            "Check that a GPU is attached and the driver is installed (`nvidia-smi`)."
        )
    return (
        f"CUDA is required to run: {', '.join(gpu_pipelines)}. {cause}\n"
        "These pipelines train on the GPU and take days on CPU, so they abort instead of falling "
        "back silently. To run only the CPU pipelines, pass "
        "`--pipeline dinov2_probe --pipeline folador_hog`."
    )


def _require_cuda(cells: list[CellSpec]) -> None:
    """Abort before any training if a selected cell needs a GPU and CUDA is absent."""
    gpu_pipelines = sorted({cell.pipeline_name for cell in cells} & GPU_PIPELINES)
    if not gpu_pipelines:
        return

    # Imported here, not at module scope, so Layer-1 commands keep working without
    # the training extras installed. `parkinson_spiral` is already imported above,
    # so the determinism environment variables are set before CUDA initialises.
    try:
        import torch
    except ImportError:
        raise SystemExit(
            f"CUDA is required to run: {', '.join(gpu_pipelines)}, but torch is not installed. "
            'Install the training extra with `python -m pip install -e ".[train]"`.'
        ) from None

    message = _cuda_requirement_error(
        gpu_pipelines=gpu_pipelines,
        cuda_available=torch.cuda.is_available(),
        torch_cuda_version=torch.version.cuda,
    )
    if message is not None:
        raise SystemExit(message)


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
    _require_cuda(cells)
    on_fitted, saved = _model_saver(cells, args.save_models)

    paths = RunPaths(manifest=args.manifest, outputs_dir=args.out)
    for i, spec in enumerate(cells, 1):
        row = run_cell(spec, paths, force=args.force, on_fitted=on_fitted)
        acc = row.get("subject_accuracy")
        acc_s = f"{acc:.3f}" if isinstance(acc, (int, float)) else "n/a"
        print(
            f"[{i}/{len(cells)}] {spec.pipeline_name} {spec.recipe_name} "
            f"{spec.train_source}->{spec.test_source} -> subj_acc={acc_s}"
        )

    if args.save_models:
        missing = sorted({released_model_name(cell) for cell in cells} - {None} - saved)
        if missing:
            raise SystemExit(
                f"no model saved for {missing}: their cells were served from cache, so nothing was trained. "
                "Re-run with --force or an empty --out."
            )
        print(f"saved {len(saved)} models -> {args.save_models}")
    print(f"done -> {args.out}")
    return 0


def _model_saver(cells: list[CellSpec], models_dir: Path | None):
    """Return (on_fitted callback or None, set of saved model names) for `reproduce --save-models`.

    Everything that would make the save fail is checked here, before any training starts. A model
    is saved from the first cell that trains it; the second off-diagonal cell from the same
    training dataset reuses or refits the same model, and `verify-models` checks both.
    """
    saved: set[str] = set()
    if models_dir is None:
        return None, saved
    names = {released_model_name(cell) for cell in cells} - {None}
    if not names:
        raise SystemExit("--save-models given, but no selected cell trains a released (off-diagonal) model")
    for cell in cells:
        if released_model_name(cell) is not None:
            check_model_saveable(cell)
    existing = sorted(name for name in names if (models_dir / name).exists())
    if existing:
        raise SystemExit(f"{models_dir} already holds {existing}; refusing to overwrite saved models")
    importlib.import_module("skops.io")

    def on_fitted(spec: CellSpec, pipeline) -> None:
        name = released_model_name(spec)
        if name is None or name in saved:
            return
        save_released_model(spec, pipeline, models_dir)
        saved.add(name)
        print(f"  saved model {name}")

    return on_fitted, saved


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
    # Optional environment probe. Some callers run tools that exist only on
    # certain platforms (e.g. macOS-only `system_profiler`); off that platform
    # the executable is simply absent, which is a normal condition rather than a
    # failure. Treat a missing binary like a non-zero exit and return None so the
    # evidence capture degrades gracefully instead of crashing on Linux/Windows.
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None
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


EMBEDDING_METADATA_COLUMNS = (
    "image_id",
    "dataset",
    "subject_id",
    "session_id",
    "drawing_type",
    "class_label",
    "cv_fold",
)


def class_label_to_int() -> dict[str, int]:
    """The canonical class-label mapping, obtained without requiring torch.

    `CLASS_LABEL_TO_INT` is defined in `parkinson_spiral.data.dataset`, which
    imports torch at module level because it defines a `torch.utils.data.Dataset`.
    torch lives in the `[train]` extra, so a Layer-1 install cannot import that
    module at all, and `build_embedding_frame` is on the Layer-1 path.

    The literal is duplicated here rather than moved to a torch-free module
    because moving it would edit `src/parkinson_spiral/`, which changes
    `_source_sha256()` and invalidates every cached cell. The duplication is
    held in check by a test that forces this fallback and asserts it still
    equals the canonical mapping.
    """
    try:
        from parkinson_spiral.data.dataset import CLASS_LABEL_TO_INT
    except ImportError:
        return {"healthy": 0, "parkinson": 1}
    return dict(CLASS_LABEL_TO_INT)


def build_embedding_frame(
    manifest: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Join extracted feature vectors onto their manifest rows, one row per image.

    Pure, so the two ways this can silently go wrong are testable without loading
    a backbone. The extractor's DataLoader runs unshuffled, so row i of `features`
    must be manifest row i; comparing the labels it emitted against the manifest's
    own is what checks that the correspondence actually held.
    """
    CLASS_LABEL_TO_INT = class_label_to_int()

    if len(features) != len(manifest):
        raise ValueError(f"extractor returned {len(features)} vectors for {len(manifest)} images")
    expected = manifest["class_label"].map(CLASS_LABEL_TO_INT).to_numpy()
    if not np.array_equal(labels, expected):
        raise ValueError("extracted labels do not follow the manifest order; the embeddings are unusable")

    return pd.concat(
        [
            manifest[list(EMBEDDING_METADATA_COLUMNS)].reset_index(drop=True),
            pd.DataFrame(features, columns=[f"f{i:03d}" for i in range(features.shape[1])]),
        ],
        axis=1,
    )


def export_embeddings_command(args: argparse.Namespace) -> int:
    """Export the frozen DINOv2 CLS embedding of every manifest image.

    Runs the released `DinoV2Probe` feature extractor unchanged, so the vectors
    written here are exactly the ones the published probe is fitted on. That
    identity is the point: features from a reimplementation would describe a
    different model than the paper reports.
    """
    from parkinson_spiral.data.dataset import PreprocessedSpiralDataset
    from parkinson_spiral.pipelines.dinov2_probe import DinoV2Probe
    from parkinson_spiral.recipes import get_recipe

    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    manifest = pd.read_parquet(args.manifest)
    if args.task_filter != "both":
        manifest = manifest[manifest["drawing_type"] == args.task_filter]
    manifest = manifest.reset_index(drop=True)
    if manifest.empty:
        raise SystemExit(f"no manifest rows left after task filter {args.task_filter!r}")

    recipe = get_recipe(args.recipe)
    dataset = PreprocessedSpiralDataset(manifest, recipe, is_train=False)
    probe = DinoV2Probe(vit_size=args.vit_size, batch_size=args.batch_size)
    print(f"extracting {len(manifest)} embeddings with dinov2_{args.vit_size} under {args.recipe}...")
    features, labels = probe._extract(dataset)  # noqa: SLF001 -- the released path is the point
    frame = build_embedding_frame(manifest, features, labels)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    sidecar = {
        "backbone": f"dinov2_{args.vit_size}",
        "backbone_weight_checksums": getattr(probe, "_model_weight_checksums", {}),
        "recipe": args.recipe,
        "image_size": recipe.meta.image_size,
        "task_filter": args.task_filter,
        "embedding_dim": int(features.shape[1]),
        "n_images": int(len(frame)),
        "n_subjects": int(frame["subject_id"].nunique()),
        "images_per_dataset": frame.groupby("dataset").size().to_dict(),
        "manifest_sha256": _sha256(args.manifest),
    }
    sidecar_path = args.out.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {len(frame)} x {features.shape[1]}-d embeddings -> {args.out}")
    print(f"metadata -> {sidecar_path}")
    print(f"datasets: {sidecar['images_per_dataset']}, subjects: {sidecar['n_subjects']}")
    return 0


# ---- cost-profile ----

# The cost profile runs one representative training configuration per pipeline: its native
# recipe, trained on the largest dataset (Moura) and evaluated off-diagonal on Zham. Each is an
# EXISTING catalog cell rather than a bespoke config, so the cost figures describe a run the
# paper actually reports and the predictions can be checked against it.
COST_PROFILE_TRAIN = "hemato"
COST_PROFILE_TEST = "kaggle"
COST_PROFILE_SPLIT = "cross-dataset"

COST_PROFILE_FIELDS = (
    "cell_id",
    "pipeline_name",
    "recipe_name",
    "train_source",
    "test_source",
    "split_source",
    "n_train_images",
    "n_test_images",
    "n_train_subjects",
    "n_test_subjects",
    "fit_seconds",
    "fit_was_cached",
    "predict_seconds",
    "infer_latency_ms_per_img",
    "accelerator_device",
    "gpu_baseline_mb",
    "gpu_peak_mb",
    "gpu_peak_delta_mb",
    "gpu_peak_reserved_mb",
    "host_rss_peak_mb",
    "subject_accuracy",
    "subject_auc",
    "python_version",
    "platform",
)


def _plain(value):
    """Coerce numpy scalars to JSON-serialisable Python types."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def cost_profile_specs() -> list[CellSpec]:
    """The four catalog cells the cost profile runs, one per pipeline in its native recipe."""
    native = {(pipeline, recipe) for pipeline, recipe, _ in NATIVE}
    chosen: dict[str, CellSpec] = {}
    for cell in all_cells():
        if (cell.pipeline_name, cell.recipe_name) not in native:
            continue
        if (cell.train_source, cell.test_source, cell.split_source) != (
            COST_PROFILE_TRAIN,
            COST_PROFILE_TEST,
            COST_PROFILE_SPLIT,
        ):
            continue
        if cell.pipeline_name in chosen:
            raise ValueError(f"catalog holds two {cell.pipeline_name} cells matching the cost-profile config")
        chosen[cell.pipeline_name] = cell
    missing = sorted({pipeline for pipeline, _, _ in NATIVE} - set(chosen))
    if missing:
        raise ValueError(f"no catalog cell matches the cost-profile config for: {missing}")
    return [chosen[pipeline] for pipeline, _, _ in NATIVE]


def build_cost_profile_entry(row: dict, pipeline_class) -> dict:
    """Reduce a `run_cell` row to the cost fields, plus what the H4 cost argument counts.

    `fit_was_cached` must be False. The profile exists to measure a real fit in a process that
    ran nothing else, so a reused fit means the isolation this command promises did not happen
    and the numbers would describe a cache lookup.
    """
    if row.get("fit_was_cached"):
        raise ValueError(f"cell {row.get('cell_id')} reused a cached fit; a cost profile must measure a real one")
    entry = {field: _plain(row.get(field)) for field in COST_PROFILE_FIELDS}
    entry["free_hyperparameters"] = list(pipeline_class.FREE_HYPERPARAMETERS)
    entry["n_free_hyperparameters"] = len(pipeline_class.FREE_HYPERPARAMETERS)
    entry["training_regime"] = TRAINING_REGIME.get(row.get("pipeline_name"))
    return entry


def compare_cost_profile_predictions(entries: list[dict], results_dir: Path, work_dir: Path) -> list[dict]:
    """Compare each profiled cell against the same cell in a completed run.

    Nearly free evidence, because the profile retrains real catalog cells in a process that ran
    nothing else beforehand. Any difference from the same cell inside a full run is what
    execution context alone is worth.
    """
    comparison = []
    for entry in entries:
        cell_id = entry["cell_id"]
        reference = results_dir / "predictions" / f"{cell_id}.parquet"
        profiled = work_dir / cell_id / "predictions" / f"{cell_id}.parquet"
        record = {"cell_id": cell_id, "pipeline_name": entry["pipeline_name"]}
        if not reference.is_file() or not profiled.is_file():
            comparison.append({**record, "compared": False, "reason": "predictions missing on one side"})
            continue
        ref = pd.read_parquet(reference).set_index("image_id").sort_index()
        new = pd.read_parquet(profiled).set_index("image_id").sort_index()
        same_images = bool(ref.index.equals(new.index))
        delta = float((new["y_proba_pos"] - ref["y_proba_pos"]).abs().max()) if same_images else None
        comparison.append({
            **record,
            "compared": True,
            "same_test_images": same_images,
            "max_abs_proba_delta": delta,
            "identical_predictions": bool(same_images and delta == 0.0),
        })
    return comparison


def _cost_profile_worker(args: argparse.Namespace) -> int:
    """Profile exactly one pipeline. Runs as a child process so the measurement is isolated."""
    from parkinson_spiral.runner import RunPaths, run_cell

    spec = next((cell for cell in cost_profile_specs() if cell.cell_id == args.worker_cell), None)
    if spec is None:
        raise SystemExit(f"{args.worker_cell} is not one of the cost-profile cells")

    paths = RunPaths(manifest=args.manifest, outputs_dir=args.work_dir / spec.cell_id)
    row = run_cell(spec, paths, force=True)
    entry = build_cost_profile_entry(row, get_pipeline_class(spec.pipeline_name))

    args.worker_out.parent.mkdir(parents=True, exist_ok=True)
    args.worker_out.write_text(json.dumps(entry, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def cost_profile_command(args: argparse.Namespace) -> int:
    """Measure each pipeline's training and inference cost in a FRESH PROCESS.

    Isolation is the whole point. In the 97-cell run all four pipelines share one interpreter
    and keep their backbones cached between cells, so a pipeline's peak memory includes whatever
    its predecessors left allocated -- `folador_hog` reports over a gigabyte of CUDA memory while
    running entirely on CPU. Here each pipeline gets its own interpreter, so `host_rss_peak_mb`
    (a process-lifetime high-water mark) and `gpu_peak_reserved_mb` (what the caching allocator
    took from the card) finally describe one pipeline instead of a shared process.
    """
    if args.worker_cell:
        return _cost_profile_worker(args)

    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")

    specs = cost_profile_specs()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for index, spec in enumerate(specs, start=1):
        fragment = args.work_dir / f"{spec.pipeline_name}.json"
        print(
            f"[{index}/{len(specs)}] {spec.pipeline_name} ({spec.recipe_name}) "
            f"{spec.train_source}->{spec.test_source} in a fresh process...",
            flush=True,
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "cost-profile",
                "--manifest",
                str(args.manifest),
                "--work-dir",
                str(args.work_dir),
                "--worker-cell",
                spec.cell_id,
                "--worker-out",
                str(fragment),
            ],
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(f"cost profile for {spec.pipeline_name} failed (rc={completed.returncode})")
        entries.append(json.loads(fragment.read_text(encoding="utf-8")))

    comparison = None
    if args.compare:
        comparison = compare_cost_profile_predictions(entries, args.compare, args.work_dir)

    repo = Path(__file__).resolve().parents[1]
    profile = {
        "schema": 1,
        "isolation": "one fresh interpreter per pipeline",
        "config": {
            "train_source": COST_PROFILE_TRAIN,
            "test_source": COST_PROFILE_TEST,
            "split_source": COST_PROFILE_SPLIT,
            "note": "each pipeline in its native recipe; every cell is a real catalog cell",
        },
        "git_commit": _git_value(repo.parent, "rev-parse", "HEAD"),
        "manifest_sha256": _sha256(args.manifest),
        "pipelines": entries,
        "comparison_vs_results": comparison,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")

    header = (
        f"{'pipeline':14s} {'device':7s} {'fit s':>9s} {'GPU dMB':>9s} "
        f"{'GPU resMB':>10s} {'RSS MB':>9s} {'ms/img':>8s} {'params':>7s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for entry in entries:
        device = str(entry["accelerator_device"] or "cpu")
        print(
            f"{entry['pipeline_name']:14s} {device:7s} "
            f"{entry['fit_seconds'] or 0.0:9.1f} {entry['gpu_peak_delta_mb'] or 0.0:9.1f} "
            f"{entry['gpu_peak_reserved_mb'] or 0.0:10.1f} {entry['host_rss_peak_mb'] or 0.0:9.1f} "
            f"{entry['infer_latency_ms_per_img'] or 0.0:8.2f} {entry['n_free_hyperparameters']:7d}"
        )
    if comparison:
        print("\nvs the completed run (same cells, full process context):")
        for record in comparison:
            if not record.get("compared"):
                print(f"  {record['pipeline_name']:14s} not compared ({record.get('reason')})")
                continue
            verdict = "identical" if record["identical_predictions"] else "DIFFERS"
            print(f"  {record['pipeline_name']:14s} {verdict}  max |proba delta| = {record['max_abs_proba_delta']:.3e}")
    print(f"\nwrote {args.out}")
    return 0


# ---- released models ----

# The released models are the 12 fits on ALL of one dataset behind the off-diagonal cells: one per
# pipeline and training dataset. Each pipeline's fitted state is the files below, keyed to the
# attribute that holds it. Its frozen ImageNet or DINOv2 backbones are not saved: they are public,
# and the pipeline downloads them itself.
MODEL_ARTIFACTS = {
    "dinov2_probe": {"probe.skops": "_probe"},
    "folador_hog": {"classifier.skops": "_model"},
    "dogan_triple": {"nca.skops": "_nca", "svm.skops": "_svm"},
    "huang_vgg19": {"vgg19_state_dict.pt": "_model"},
}
MODEL_CARD = "model.json"


def released_model_name(spec: CellSpec) -> str | None:
    """Name of the model a cell trains on all of its training dataset, or None.

    Only off-diagonal cells train on a whole dataset; fold and released-split cells hold data out,
    so their models are not released.
    """
    if spec.split_source != "cross-dataset":
        return None
    return f"{spec.pipeline_name}-{spec.train_source}"


def check_model_saveable(spec: CellSpec) -> None:
    """Raise if a cell's fitted state is not fully captured by `MODEL_ARTIFACTS`."""
    if spec.pipeline_name == "dogan_triple" and spec.pipeline_kwargs.get("fine_tune", False):
        raise ValueError(
            "saving dogan_triple with fine_tune=True is not supported: the fine-tuned backbones are not saved"
        )
    if spec.pipeline_name == "folador_hog" and spec.pipeline_kwargs.get("classifier") == "CNN1D":
        raise ValueError("saving folador_hog with the CNN1D classifier is not supported")


def _dump_artifact(fitted, path: Path) -> None:
    if path.suffix == ".skops":
        import skops.io as sio

        sio.dump(fitted, path)
        return
    import torch

    torch.save({key: value.detach().cpu() for key, value in fitted.state_dict().items()}, path)


def _load_artifact(path: Path):
    """Load a saved artifact without executing code from it.

    skops refuses any type outside its trusted defaults, and `weights_only` restricts torch to
    tensors, so a tampered file fails to load instead of running.
    """
    if path.suffix == ".skops":
        import skops.io as sio

        return sio.load(path)
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def _same_value(a, b) -> bool:
    """Exact equality through the containers a fitted scikit-learn estimator holds."""
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype:
            return False
        return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, np.random.RandomState):
        return isinstance(b, np.random.RandomState) and _same_value(list(a.get_state()), list(b.get_state()))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same_value(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(_same_value, a, b))
    return type(a) is type(b) and a == b


def same_fitted_state(fitted, reloaded) -> bool:
    """Whether a reloaded artifact holds exactly the fitted state that was saved."""
    if isinstance(reloaded, dict):
        import torch

        state = fitted.state_dict()
        return state.keys() == reloaded.keys() and all(
            torch.equal(state[key].detach().cpu(), reloaded[key]) for key in state
        )
    return type(fitted) is type(reloaded) and _same_value(vars(fitted), vars(reloaded))


def save_released_model(spec: CellSpec, pipeline, models_dir: Path) -> Path:
    """Save the fitted pipeline of an off-diagonal cell to `models_dir/<name>/`.

    Every file is reloaded and compared with the in-memory model before the card is written, so
    a directory with a card holds a model that reloads exactly. The card lists the cells the
    model serves and the code that trained it.
    """
    from parkinson_spiral.runner import _package_versions, _source_sha256

    name = released_model_name(spec)
    if name is None:
        raise ValueError(f"cell {spec.cell_id} is not an off-diagonal cell; its model is not released")
    check_model_saveable(spec)
    directory = models_dir / name
    if directory.exists():
        raise FileExistsError(f"{directory} already exists; refusing to overwrite a saved model")
    directory.mkdir(parents=True)

    files = {}
    for filename, attribute in MODEL_ARTIFACTS[spec.pipeline_name].items():
        fitted = getattr(pipeline, attribute)
        path = directory / filename
        _dump_artifact(fitted, path)
        if not same_fitted_state(fitted, _load_artifact(path)):
            raise RuntimeError(f"{path} does not reload to the fitted state that was saved")
        files[filename] = _sha256(path)

    served = [cell for cell in all_cells() if released_model_name(cell) == name]
    repo = Path(__file__).resolve().parents[1]
    card = {
        "schema": 1,
        "name": name,
        "pipeline_name": spec.pipeline_name,
        "recipe_name": spec.recipe_name,
        "pipeline_kwargs": spec.pipeline_kwargs,
        "train_source": spec.train_source,
        "task_filter": spec.task_filter,
        "seed": spec.seed,
        "cells": {cell.cell_id: cell.test_source for cell in served},
        "files": files,
        "public_backbones": get_pipeline_class(spec.pipeline_name).MODEL_WEIGHT_IDENTIFIERS,
        "public_backbone_checksums": getattr(pipeline, "_model_weight_checksums", {}),
        "git_commit": _git_value(repo.parent, "rev-parse", "HEAD"),
        "source_sha256": _source_sha256(),
        "packages": {**_package_versions(), "skops": importlib.metadata.version("skops")},
    }
    (directory / MODEL_CARD).write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return directory


def load_released_model(directory: Path):
    """Rebuild a saved model as a pipeline ready for `predict_proba`. Returns (pipeline, card).

    File checksums are checked against the card first. The public backbones are downloaded on
    first use, as in training.
    """
    from parkinson_spiral.pipelines import accelerator_device, get_pipeline

    card = json.loads((directory / MODEL_CARD).read_text(encoding="utf-8"))
    for filename, expected in card["files"].items():
        if _sha256(directory / filename) != expected:
            raise ValueError(f"{directory / filename} does not match the checksum in its {MODEL_CARD}")

    name = card["pipeline_name"]
    pipeline = get_pipeline(name, **card["pipeline_kwargs"])
    if name == "huang_vgg19":
        # Rebuild the architecture, then replace every weight with the saved ones.
        model = pipeline._build()
        model.load_state_dict(_load_artifact(directory / "vgg19_state_dict.pt"))
        pipeline._device = accelerator_device()
        pipeline._model = model.to(pipeline._device)
        return pipeline, card
    for filename, attribute in MODEL_ARTIFACTS[name].items():
        setattr(pipeline, attribute, _load_artifact(directory / filename))
    if name == "dogan_triple":
        pipeline._build_backbones()
    return pipeline, card


def _same_metrics(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.keys() == b.keys() and all(a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key])) for key in a)


def compare_cell_predictions(reference: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Compare one cell's predictions from a saved model with the same cell in a completed run.

    Participant-level metrics are the criterion, since the paper reports at that level. The largest
    per-image probability difference is reported alongside, since image probabilities can move without
    changing any participant's decision.
    """
    from parkinson_spiral.metrics import metrics_at_all_levels

    ref = reference.set_index("image_id").sort_index()
    out = new.set_index("image_id").sort_index()
    if not ref.index.equals(out.index):
        return {"same_test_images": False, "max_abs_proba_delta": None, "same_subject_metrics": False}
    reference_subject = metrics_at_all_levels(reference)["subject"]
    new_subject = metrics_at_all_levels(new)["subject"]
    return {
        "same_test_images": True,
        "max_abs_proba_delta": float((out["y_proba_pos"] - ref["y_proba_pos"]).abs().max()),
        "same_subject_metrics": _same_metrics(reference_subject, new_subject),
        "subject_metrics": new_subject,
    }


def verify_models_command(args: argparse.Namespace) -> int:
    """Reload every saved model, re-predict the cells it serves, and compare with a completed run."""
    from parkinson_spiral.data.dataset import PreprocessedSpiralDataset
    from parkinson_spiral.recipes import get_recipe
    from parkinson_spiral.runner import _configure_reproducibility, _recipe_kwargs_for_spec, resolve_split

    directories = sorted(card.parent for card in args.models.glob(f"*/{MODEL_CARD}"))
    if not directories:
        raise SystemExit(f"no saved models (*/{MODEL_CARD}) under {args.models}")
    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")
    manifest = pd.read_parquet(args.manifest)
    available = set(manifest["dataset"].dropna().astype(str).unique())
    catalog = {cell.cell_id: cell for cell in all_cells()}

    records = []
    for directory in directories:
        pipeline, card = load_released_model(directory)
        for cell_id in card["cells"]:
            spec = catalog[cell_id]
            if spec.test_source not in available:
                print(f"  {card['name']:22s} -> {spec.test_source:8s} skipped: dataset absent from the manifest")
                continue
            reference_path = args.results / "predictions" / f"{cell_id}.parquet"
            if not reference_path.is_file():
                raise SystemExit(f"{reference_path} not found; --results must hold the run that trained these models")
            _configure_reproducibility(spec.seed)
            _, test_idx = resolve_split(manifest, spec)
            test_manifest = manifest.loc[test_idx].reset_index(drop=True)
            recipe = get_recipe(spec.recipe_name, **_recipe_kwargs_for_spec(spec))
            test_ds = PreprocessedSpiralDataset(test_manifest, recipe, is_train=False, target_column=spec.target_column)
            proba = pipeline.predict_proba(test_ds)
            new = pd.DataFrame({
                "image_id": test_manifest["image_id"].astype(str).to_numpy(),
                "subject_id": test_manifest["subject_id"].astype(str).to_numpy(),
                "session_id": test_manifest["session_id"].astype("string").to_numpy(),
                "y_true": (test_manifest["class_label"] == "parkinson").astype(int).to_numpy(),
                "y_proba_pos": proba[:, 1],
            })
            record = {"model": card["name"], "cell_id": cell_id, "test_source": spec.test_source}
            record.update(compare_cell_predictions(pd.read_parquet(reference_path), new))
            records.append(record)
            verdict = "same" if record["same_subject_metrics"] else "DIFFERENT"
            delta = record["max_abs_proba_delta"]
            delta_s = "n/a" if delta is None else f"{delta:.3e}"
            print(
                f"  {card['name']:22s} -> {spec.test_source:8s} participant metrics {verdict}; max |proba delta| {delta_s}"
            )

    if not records:
        raise SystemExit("no served cell has its test dataset in the manifest; nothing was verified")
    report = {
        "schema": 1,
        "models_dir": str(args.models),
        "results_dir": str(args.results),
        "manifest_sha256": _sha256(args.manifest),
        "cells": records,
        "all_same_subject_metrics": all(record["same_subject_metrics"] for record in records),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n{sum(r['same_subject_metrics'] for r in records)}/{len(records)} cells reproduce; wrote {args.out}")
    return 0 if report["all_same_subject_metrics"] else 1


# ---- statistics and distances ----


def statistics_command(args: argparse.Namespace) -> int:
    """Participant-level intervals, paired comparisons and majority baselines of a completed run."""
    from parkinson_spiral import evaluation

    if not (args.results / "cells.parquet").is_file():
        raise SystemExit(f"{args.results / 'cells.parquet'} not found")
    intervals = evaluation.intervals(args.results)
    paired = evaluation.paired(args.results)
    baselines = evaluation.baselines(args.results)

    args.out.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(args.out / "intervals.csv", index=False)
    paired.to_csv(args.out / "paired.csv", index=False)
    baselines.to_csv(args.out / "baselines.csv", index=False)
    summary = {
        "schema": 1,
        "cells_sha256": _sha256(args.results / "cells.parquet"),
        "n_cells": len(intervals),
        "one_class_resamples": int(intervals.n_one_class_resamples.sum()),
        "points_equal_reported_metrics": bool(intervals.points_equal_binary_metrics.all()),
        "n_comparisons": len(paired),
        "n_rejections_after_holm": int(paired.rejects_after_holm.sum()),
        "n_holm_intervals_excluding_zero": int(paired.holm_interval_excludes_zero.sum()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"{summary['n_cells']} cells with intervals; {summary['one_class_resamples']} one-class resamples")
    print(
        f"{summary['n_comparisons']} paired comparisons against {evaluation.REFERENCE}: "
        f"{summary['n_rejections_after_holm']} reject after Holm, "
        f"{summary['n_holm_intervals_excluding_zero']} adjusted intervals exclude zero"
    )
    print(f"wrote {args.out}")
    return 0


def distances_command(args: argparse.Namespace) -> int:
    """Sliced Wasserstein distances between the datasets, with floor and null, plus the t-SNE map."""
    from parkinson_spiral import distances

    if not args.embeddings.is_file():
        raise SystemExit(f"embeddings not found: {args.embeddings} (write them with export-embeddings)")
    if args.figure:
        importlib.import_module("matplotlib")  # fail before the slow steps, not after
    embeddings = pd.read_parquet(args.embeddings)
    table = distances.dataset_distances(distances.participant_clouds(embeddings))
    coordinates = distances.tsne_coordinates(embeddings)

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "distances.csv", index=False)
    coordinates.to_csv(args.out / "tsne.csv", index=False)
    if args.figure:
        plot_tsne(coordinates, args.out / "fig_tsne.png")

    between = table[table.kind == "between"]
    for row in between.itertuples():
        print(
            f"  {PAPER_NAMES[row.a]:7s} vs {PAPER_NAMES[row.b]:7s} distance {row.distance:.3f}  "
            f"above floor: {row.exceeds_floor_p95}  above null: {row.exceeds_null_p95}"
        )
    print(f"wrote {args.out}")
    return 0


PAPER_NAMES = {"kaggle": "Zham", "folador": "Folador", "hemato": "Moura"}


def plot_tsne(coordinates: pd.DataFrame, out: Path) -> None:
    """The t-SNE figure: colour by dataset, marker by class. Needs the figures extra (matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    color = {"hemato": "#2a78d6", "folador": "#eb6834", "kaggle": "#1baf7a"}
    surface, ink, ink_2, muted, baseline = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    marker = {"healthy": "o", "parkinson": "^"}
    # Zham and Folador overlap, so their labels are pulled apart; Moura sits apart already.
    label_offset = {"hemato": (0, 0), "folador": (28, 26), "kaggle": (-26, -26)}
    order = ["kaggle", "folador", "hemato"]

    fig, ax = plt.subplots(figsize=(3.5, 3.1), facecolor=surface)
    ax.set_facecolor(surface)
    for dataset in order[::-1]:  # the large set first, so the small ones stay visible
        for label, shape in marker.items():
            m = (coordinates.dataset == dataset) & (coordinates.class_label == label)
            ax.scatter(
                coordinates.x[m], coordinates.y[m], s=8, marker=shape, c=color[dataset], alpha=0.45, linewidths=0
            )
        m = coordinates.dataset == dataset
        centre = (float(np.median(coordinates.x[m])), float(np.median(coordinates.y[m])))
        dx, dy = label_offset[dataset]
        ax.annotate(
            PAPER_NAMES[dataset],
            centre,
            textcoords="offset points",
            xytext=(dx, dy),
            color=ink,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": surface, "ec": color[dataset], "alpha": 0.9},
            arrowprops={"arrowstyle": "-", "color": color[dataset], "lw": 0.8} if (dx or dy) else None,
        )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(baseline)
    ax.tick_params(colors=muted, labelsize=7)
    ax.set_xlabel("t-SNE 1", color=ink_2, fontsize=8)
    ax.set_ylabel("t-SNE 2", color=ink_2, fontsize=8)
    handles = [
        Line2D([], [], marker="s", ls="", color=color[d], label=PAPER_NAMES[d], markersize=6) for d in order[::-1]
    ] + [Line2D([], [], marker=marker[c], ls="", color=ink_2, label=c, markersize=6) for c in ("healthy", "parkinson")]
    ax.legend(
        handles=handles,
        loc="best",
        ncol=2,
        frameon=False,
        fontsize=6.5,
        labelcolor=ink_2,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=300, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


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
    reproduce.add_argument(
        "--save-models",
        type=Path,
        default=None,
        help="save each model trained on all of a dataset (off-diagonal cells) here; needs the hf extra",
    )
    reproduce.set_defaults(func=reproduce_command)

    verify_models = sub.add_parser(
        "verify-models",
        help="reload saved models, re-predict their cells, and compare with the run that trained them",
    )
    verify_models.add_argument("--models", type=Path, required=True, help="directory written by --save-models")
    verify_models.add_argument("--manifest", type=Path, default=repo / "results" / "manifest.parquet")
    verify_models.add_argument("--results", type=Path, default=repo / "results", help="results of that run")
    verify_models.add_argument("--out", type=Path, default=repo / "results" / "verify_models.json")
    verify_models.set_defaults(func=verify_models_command)

    statistics = sub.add_parser(
        "statistics",
        help="participant-level bootstrap intervals, paired McNemar comparisons and majority baselines",
    )
    statistics.add_argument("--results", type=Path, default=repo / "results")
    statistics.add_argument("--out", type=Path, default=repo / "results" / "statistics")
    statistics.set_defaults(func=statistics_command)

    dist = sub.add_parser(
        "distances",
        help="sliced Wasserstein distances between the datasets in DINOv2 space, and the t-SNE map",
    )
    dist.add_argument("--embeddings", type=Path, default=repo / "results" / "embeddings.parquet")
    dist.add_argument("--out", type=Path, default=repo / "results" / "distances")
    dist.add_argument("--figure", action="store_true", help="also draw fig_tsne.png (needs the figures extra)")
    dist.set_defaults(func=distances_command)

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

    export_emb = sub.add_parser(
        "export-embeddings",
        help="export frozen DINOv2 CLS embeddings for every manifest image",
    )
    export_emb.add_argument("--manifest", type=Path, default=repo / "results" / "manifest.parquet")
    export_emb.add_argument("--out", type=Path, default=repo / "results" / "embeddings.parquet")
    export_emb.add_argument(
        "--task-filter",
        default="spiral",
        choices=["spiral", "wave", "both"],
        help="which drawing type to embed (default: spiral, the benchmark's task)",
    )
    export_emb.add_argument(
        "--recipe",
        default="AUG-None",
        choices=["AUG-None", "AUG-Huang", "AUG-Dogan"],
        help="preprocessing recipe; AUG-None is the deterministic resize+normalize the probe uses",
    )
    export_emb.add_argument("--vit-size", default="vitb14", choices=["vits14", "vitb14", "vitl14", "vitg14"])
    export_emb.add_argument("--batch-size", type=int, default=32)
    export_emb.set_defaults(func=export_embeddings_command)

    cost = sub.add_parser(
        "cost-profile",
        help="measure per-pipeline training/inference cost, each pipeline in a fresh process",
    )
    cost.add_argument("--manifest", type=Path, default=repo / "results" / "manifest.parquet")
    cost.add_argument("--out", type=Path, default=repo / "results" / "cost_profile.json")
    cost.add_argument(
        "--work-dir",
        type=Path,
        default=repo / "results" / "cost_profile_work",
        help="scratch directory for the per-pipeline runs; kept, so the fits stay inspectable",
    )
    cost.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="results directory of a completed run; compares the profiled cells against it",
    )
    cost.add_argument("--worker-cell", default=None, help=argparse.SUPPRESS)
    cost.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    cost.set_defaults(func=cost_profile_command)

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
