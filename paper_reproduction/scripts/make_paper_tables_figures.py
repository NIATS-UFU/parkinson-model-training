"""Regenerate the paper's four tables + two TikZ figures from the shipped results.

No training, no GPU — reads only `results/` and writes:
  <out>/results_tables.md                          (b1-fidelity, cv-ceiling, cross-dataset, h4-cost)
  <out>/figures/tikz/fig1_crossdataset_matrices.tex
  <out>/figures/tikz/fig2_asymmetry.tex

The numbers it prints are the numbers in conference_v2.tex. `scripts/regenerate.py`
is a thin wrapper around this for the headline reproduce-from-results workflow.

Usage:  python scripts/make_paper_tables_figures.py [--out paper_artifacts]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parkinson_spiral.analysis as A  # noqa: E402
from parkinson_spiral.pipelines import get_pipeline_class  # noqa: E402

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
    "dogan_triple": "3 backbones fine-tuned + NCA + linear SVM",
    "huang_vgg19": "end-to-end fine-tune, 50 epochs",
}


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
            cells, preds, bootstrap=False,
            pipeline_name=pipe, recipe_name=recipe, train_source=source, split_source="published",
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
                cells, preds, bootstrap=True,
                pipeline_name=pipe, recipe_name=recipe,
                train_source=source, test_source=source, split_source="cv-fold",
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
        label = {"dinov2_probe": "DINOv2 AUG-None", "huang_vgg19": "Huang AUG-Huang",
                 "dogan_triple": "Dogan AUG-Dogan", "folador_hog": "Folador HOG AUG-None"}[pipe]
        rows = []
        accs = []
        for train in SOURCES:
            for test in SOURCES:
                if train == test:
                    continue
                res = A.subject_suite(
                    cells, preds, bootstrap=True,
                    pipeline_name=pipe, recipe_name=recipe,
                    train_source=train, test_source=test, split_source="cross-dataset",
                )
                accs.append(res["accuracy"])
                rows.append({
                    "Pipeline": label,
                    "Dir": f"{SHORT[train]}->{SHORT[test]}",
                    "Acc [95% CI]": _fmt_ci(res, nd=2),
                    "AUC": f"{res['auc']:.3f}", "Sens": f"{res['sensitivity']:.2f}",
                    "Spec": f"{res['specificity']:.2f}", "Prec": f"{res['precision']:.2f}",
                    "F1": f"{res['f1']:.2f}",
                })
        rows.append({"Pipeline": label, "Dir": "mean", "Acc [95% CI]": f"{np.mean(accs):.3f}",
                     "AUC": "", "Sens": "", "Spec": "", "Prec": "", "F1": ""})
        blocks.append(pd.DataFrame(rows).to_markdown(index=False))
    return "### Table: cross-dataset off-diagonals (`cross-dataset`)\n\n" + "\n\n".join(blocks) + "\n"


def _hemato_fold_sigma(cells, pipe, recipe) -> float:
    sub = cells[(cells.pipeline_name == pipe) & (cells.recipe_name == recipe)
                & (cells.split_source == "cv-fold") & (cells.train_source == "hemato")]
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
        label = {"dinov2_probe": "DINOv2 (frozen) AUG-None", "folador_hog": "Folador HOG AUG-None",
                 "dogan_triple": "Dogan triple AUG-Dogan", "huang_vgg19": "Huang VGG19 AUG-Huang"}[pipe]
        rows.append({
            "Pipeline (recipe)": label,
            "# cfg": n_cfg,
            "Training regime": TRAINING_REGIME[pipe],
            "Hemato CV": f"{ceiling['point']:.3f} (sigma {sigma:.3f})",
        })
    md = pd.DataFrame(rows).to_markdown(index=False)
    # The augmentation-robustness swing (strong form of H4).
    swing = [A.diagonal_ceiling(cells, preds, "dinov2_probe", r, "hemato", bootstrap=False)["point"]
             for r in ("AUG-None", "AUG-Huang", "AUG-Dogan")]
    note = (f"\n\n_DINOv2 Hemato aug swing:_ AUG-None {swing[0]:.3f} -> AUG-Huang {swing[1]:.3f} "
            f"-> AUG-Dogan {swing[2]:.3f}\n")
    return "### Table: reproducibility cost (`h4-cost`)\n\n" + md + note


# ---------------------------------------------------------------- figures (TikZ)

def fig1_matrices(cells, preds) -> str:
    """2x2 grid of 3x3 cross-dataset accuracy matrices, one per pipeline."""
    lines = ["% Auto-generated by make_paper_tables_figures.py — cross-dataset matrices (fig1)",
             "\\begin{tikzpicture}[font=\\scriptsize]"]
    titles = {"dinov2_probe": "DINOv2 AUG-None", "huang_vgg19": "Huang AUG-Huang",
              "dogan_triple": "Dogan AUG-Dogan", "folador_hog": "Folador AUG-None"}
    positions = {"dinov2_probe": (0, 0), "huang_vgg19": (5.2, 0),
                 "dogan_triple": (0, -4.6), "folador_hog": (5.2, -4.6)}
    cell = 1.2
    for pipe, recipe, _ in NATIVE:
        ox, oy = positions[pipe]
        m = A.cross_matrix(cells, preds, pipe, recipe, bootstrap=False)
        lines.append(f"  % {titles[pipe]}")
        lines.append(f"  \\node[anchor=south west, font=\\scriptsize\\bfseries] at ({ox},{oy+0.15}) {{{titles[pipe]}}};")
        for i, train in enumerate(SOURCES):
            for j, test in enumerate(SOURCES):
                val = m.iloc[i][test]
                x = ox + j * cell
                y = oy - i * cell
                # colour ramp: chance (0.5) white -> 1.0 strong.
                t = max(0.0, min(1.0, (float(val) - 0.45) / 0.55))
                fill = f"green!{int(t*60)}!white"
                lines.append(
                    f"  \\node[draw, minimum width={cell}cm, minimum height={cell}cm, "
                    f"fill={fill}, anchor=north west] at ({_n(x)},{_n(y)}) {{{val:.3f}}};"
                )
            # row label (train)
            lines.append(f"  \\node[anchor=east] at ({_n(ox-0.05)},{_n(oy - i*cell - cell/2)}) {{{SHORT[train]}}};")
        for j, test in enumerate(SOURCES):  # column labels (test)
            lines.append(f"  \\node[anchor=south] at ({_n(ox + j*cell + cell/2)},{_n(oy - 3*cell)}) {{{SHORT[test]}}};")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def fig2_asymmetry(cells, preds) -> str:
    """Per-pipeline asymmetry index: mean(H->{K,F}) - mean({K,F}->H)."""
    lines = ["% Auto-generated by make_paper_tables_figures.py — asymmetry index (fig2)",
             "\\begin{tikzpicture}[font=\\scriptsize]"]
    labels = {"dinov2_probe": "DINOv2", "huang_vgg19": "Huang",
              "dogan_triple": "Dogan", "folador_hog": "Folador"}
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
        lines.append(f"  \\fill[{colour}] (0,{_n(y - bar_w/2)}) rectangle ({_n(length)},{_n(y + bar_w/2)});")
        lines.append(f"  \\node[anchor={anchor}] at ({_n(label_x)},{_n(y)}) {{{idx:+.3f}}};")
    lines.append("  \\node[anchor=north] at (1.5,-0.4) {asymmetry index = mean(H$\\to\\{$K,F$\\}$) $-$ mean($\\{$K,F$\\}\\to$H)};")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main

def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=repo / "results")
    ap.add_argument("--out", type=Path, default=repo / "paper_artifacts")
    args = ap.parse_args()

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
    (out / "figures" / "tikz" / "fig1_crossdataset_matrices.tex").write_text(fig1_matrices(cells, preds), encoding="utf-8")
    (out / "figures" / "tikz" / "fig2_asymmetry.tex").write_text(fig2_asymmetry(cells, preds), encoding="utf-8")

    print(f"wrote {out / 'results_tables.md'}")
    print(f"wrote {out / 'figures' / 'tikz' / 'fig1_crossdataset_matrices.tex'}")
    print(f"wrote {out / 'figures' / 'tikz' / 'fig2_asymmetry.tex'}")


if __name__ == "__main__":
    main()
