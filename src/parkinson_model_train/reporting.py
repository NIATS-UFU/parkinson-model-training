"""Dependency-free SVG plots for training summaries."""

from html import escape
from pathlib import Path

SVG_WIDTH = 760
PLOT_LEFT = 180
PLOT_WIDTH = 480


def write_metric_plots(output_dir: Path, summary: dict) -> list[Path]:
    """Write standalone SVG plots from a training run summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "test_metrics.svg",
        output_dir / "split_counts.svg",
        output_dir / "validation_history.svg",
    ]
    _write_text(paths[0], _metric_bars(summary.get("test_metrics", {}), "Test Metrics"))
    _write_text(paths[1], _split_count_bars(summary.get("split_counts", {}), summary.get("subject_counts", {})))
    _write_text(paths[2], _validation_history(summary.get("history", [])))
    return paths


def _metric_bars(metrics: dict, title: str) -> str:
    values = {
        name: value
        for name, value in metrics.items()
        if isinstance(value, int | float) and 0 <= value <= 1 and name != "loss"
    }
    if not values:
        return _empty_svg(title, "No bounded metrics available.")

    height = 74 + 34 * len(values)
    bars = []
    for index, (name, value) in enumerate(sorted(values.items())):
        y = 54 + index * 34
        width = round(PLOT_WIDTH * value, 1)
        bars.append(
            f'<text class="label" x="24" y="{y + 15}">{escape(name)}</text>'
            f'<rect class="track" x="{PLOT_LEFT}" y="{y}" width="{PLOT_WIDTH}" height="20" rx="3" />'
            f'<rect class="bar" x="{PLOT_LEFT}" y="{y}" width="{width}" height="20" rx="3" />'
            f'<text class="value" x="{PLOT_LEFT + PLOT_WIDTH + 14}" y="{y + 15}">{value:.3f}</text>'
        )
    return _svg(title, "".join(bars), height)


def _split_count_bars(split_counts: dict, subject_counts: dict) -> str:
    split_names = sorted(set(split_counts) | set(subject_counts))
    if not split_names:
        return _empty_svg("Split Counts", "No split counts available.")

    max_count = max([1, *[int(split_counts.get(name, 0)) for name in split_names]])
    max_count = max(max_count, *[int(subject_counts.get(name, 0)) for name in split_names])

    height = 84 + 46 * len(split_names)
    bars = []
    for index, name in enumerate(split_names):
        y = 58 + index * 46
        image_count = int(split_counts.get(name, 0))
        subject_count = int(subject_counts.get(name, 0))
        image_width = round(PLOT_WIDTH * image_count / max_count, 1)
        subject_width = round(PLOT_WIDTH * subject_count / max_count, 1)
        bars.append(
            f'<text class="label" x="24" y="{y + 22}">{escape(name)}</text>'
            f'<rect class="bar" x="{PLOT_LEFT}" y="{y}" width="{image_width}" height="16" rx="3" />'
            f'<rect class="bar-alt" x="{PLOT_LEFT}" y="{y + 20}" width="{subject_width}" height="16" rx="3" />'
            f'<text class="value" x="{PLOT_LEFT + PLOT_WIDTH + 14}" y="{y + 13}">{image_count} images</text>'
            f'<text class="value muted" x="{PLOT_LEFT + PLOT_WIDTH + 14}" y="{y + 33}">{subject_count} subjects</text>'
        )
    legend = (
        f'<rect class="bar" x="{PLOT_LEFT}" y="30" width="14" height="14" rx="2" />'
        f'<text class="value" x="{PLOT_LEFT + 20}" y="42">images</text>'
        f'<rect class="bar-alt" x="{PLOT_LEFT + 92}" y="30" width="14" height="14" rx="2" />'
        f'<text class="value" x="{PLOT_LEFT + 112}" y="42">subjects</text>'
    )
    return _svg("Split Counts", legend + "".join(bars), height)


def _validation_history(history: list[dict]) -> str:
    points = []
    for item in history:
        val_metrics = item.get("val_metrics", {})
        metric_name = "balanced_accuracy" if "balanced_accuracy" in val_metrics else "accuracy"
        metric_value = val_metrics.get(metric_name)
        if isinstance(metric_value, int | float):
            points.append((int(item.get("epoch", len(points) + 1)), float(metric_value), metric_name))

    if not points:
        return _empty_svg("Validation History", "No validation history available.")

    height = 260
    top = 52
    plot_height = 150
    left = 70
    width = SVG_WIDTH - 120
    max_epoch = max(epoch for epoch, _, _ in points)
    min_epoch = min(epoch for epoch, _, _ in points)
    span = max(1, max_epoch - min_epoch)

    coords = []
    for epoch, value, _metric_name in points:
        x = left + width * (epoch - min_epoch) / span
        y = top + plot_height * (1 - max(0.0, min(1.0, value)))
        coords.append((x, y, epoch, value))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in coords)
    circles = "".join(
        f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="4" />'
        f'<text class="value" x="{x - 8:.1f}" y="{y - 10:.1f}">{value:.2f}</text>'
        for x, y, _epoch, value in coords
    )
    axis = (
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + width}" y2="{top + plot_height}" />'
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" />'
        f'<text class="value" x="{left}" y="{top + plot_height + 28}">epoch {min_epoch}</text>'
        f'<text class="value" x="{left + width - 70}" y="{top + plot_height + 28}">epoch {max_epoch}</text>'
        f'<text class="value" x="24" y="{top + 5}">1.0</text>'
        f'<text class="value" x="24" y="{top + plot_height + 5}">0.0</text>'
    )
    metric_name = points[-1][2]
    body = f'<text class="subtitle" x="{left}" y="42">{escape(metric_name)}</text>{axis}<polyline class="line" points="{polyline}" />{circles}'
    return _svg("Validation History", body, height)


def _svg(title: str, body: str, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{height}" viewBox="0 0 {SVG_WIDTH} {height}" role="img" aria-label="{escape(title)}">
  <style>
    text {{ font-family: ui-sans-serif, system-ui, sans-serif; fill: #172033; }}
    .title {{ font-size: 20px; font-weight: 700; }}
    .subtitle, .label, .value {{ font-size: 13px; }}
    .label {{ fill: #334155; }}
    .muted {{ fill: #64748b; }}
    .track {{ fill: #e8eef7; }}
    .bar {{ fill: #2563eb; }}
    .bar-alt {{ fill: #14b8a6; }}
    .axis {{ stroke: #94a3b8; stroke-width: 1; }}
    .line {{ fill: none; stroke: #2563eb; stroke-width: 3; }}
    .point {{ fill: #ffffff; stroke: #2563eb; stroke-width: 2; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text class="title" x="24" y="28">{escape(title)}</text>
  {body}
</svg>
"""


def _empty_svg(title: str, message: str) -> str:
    body = f'<text class="label" x="24" y="64">{escape(message)}</text>'
    return _svg(title, body, 100)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
