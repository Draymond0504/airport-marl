"""Add journal-style functional labels to a harbor-apron CAD reference image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


LABELS = [
    # label, target (x, y), label position (x, y), horizontal alignment
    ("Stand area", (0.28, 0.47), (0.06, 0.32), "left"),
    ("Stand", (0.83, 0.46), (0.94, 0.34), "right"),
    ("Pushback area", (0.78, 0.58), (0.94, 0.66), "right"),
    ("Shared corridor", (0.53, 0.72), (0.52, 0.17), "center"),
    ("Main taxiway", (0.53, 0.84), (0.08, 0.94), "left"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a harbor-apron CAD layout.")
    parser.add_argument("--input", required=True, help="Source CAD PNG/JPG image.")
    parser.add_argument("--output", default=None, help="Annotated PNG output path.")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output) if args.output else source.with_name(f"{source.stem}_annotated.png")
    image = mpimg.imread(source)
    height, width = image.shape[:2]
    fig_width = 12.8
    fig_height = fig_width * height / width
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=240)
    ax.imshow(image, extent=(0, 1, 1, 0))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.axis("off")

    for label, target, text_pos, alignment in LABELS:
        ax.annotate(
            label,
            xy=target,
            xytext=text_pos,
            ha=alignment,
            va="center",
            fontsize=11,
            color="#263238",
            fontfamily="DejaVu Sans",
            arrowprops={"arrowstyle": "-", "color": "#4B5563", "lw": 0.85, "shrinkA": 3, "shrinkB": 3},
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
            zorder=5,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Wrote annotated layout to {output}")


if __name__ == "__main__":
    main()
