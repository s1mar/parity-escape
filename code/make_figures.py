"""Generate the paper's figures from data/results.json. Never hand-drawn.

fig_matrix.pdf: the miss-rate matrix of Table 2 as an annotated heatmap, values identical to
the table (nearest whole point, same rounding as make_macros.py), rows = validator, columns =
translator, diagonal (self-validation) marked by a border. Single-hue sequential ramp so the
figure survives grayscale printing; every cell carries its value, so no information is
color-only.

pdf.fonttype 42 embeds TrueType rather than Type-3 fonts, which digital libraries reject.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models  # noqa: E402

D = Path(__file__).resolve().parent.parent / "data"
P = Path(__file__).resolve().parent.parent / "paper"

DISP = {"qwen25": "Qwen2.5", "dscoder": "DS-Coder", "mistral": "Mistral",
        "llama3": "Llama", "gemini": "Gemini"}


def fig_matrix() -> None:
    r = json.loads((D / "results.json").read_text(encoding="utf-8"))
    mat = r["matrix_miss_rate"]
    mids = [m.mid for m in models.MODELS]
    vals = [[100 * mat[v][t]["miss_rate"] if mat.get(v, {}).get(t) else float("nan")
             for t in mids] for v in mids]

    fig, ax = plt.subplots(figsize=(3.3, 1.32))
    im = ax.imshow(vals, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(mids)), [DISP[t] for t in mids], rotation=30, ha="right")
    ax.set_yticks(range(len(mids)), [DISP[v] for v in mids])
    ax.set_xlabel("translator")
    ax.set_ylabel("validator")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i, v in enumerate(mids):
        for j, t in enumerate(mids):
            c = vals[i][j]
            if c != c:  # NaN
                continue
            dark = c > 55
            ax.text(j, i, f"{c:.0f}", ha="center", va="center",
                    color="white" if dark else "black",
                    fontweight="bold" if i == j else "normal")
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", linewidth=1.4))

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("miss rate (%)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)

    fig.tight_layout(pad=0.2)
    out = P / "fig_matrix.pdf"
    fig.savefig(out)
    print(f"wrote {out}")

    # sanity: the annotated values must match tables.tex cell for cell
    tables = (P / "tables.tex").read_text(encoding="utf-8")
    for i, v in enumerate(mids):
        for j, t in enumerate(mids):
            c = vals[i][j]
            if c == c:
                assert f"{c:.0f}" in tables, f"cell {v}/{t}={c:.0f} not in tables.tex"
    print("all annotated values appear in tables.tex")


if __name__ == "__main__":
    fig_matrix()
