"""Generate the figures for the menstrual health and hygiene paper.

Every value plotted here comes from a published source cited in the paper
(NFHS-4, NFHS-5, WHO/UNICEF JMP 2023 and 2024) except Fig. 7, which is computed from
Eq. (1)-(2) and the stated assumptions in Table II.

Run:  python make_figures.py      (writes PNGs into ./figures)
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Validated palette (dataviz reference palette, checked against a white page).
BLUE = "#2a78d6"
BLUE_DARK = "#1c5cab"
BLUE_DEEP = "#0d366b"
BLUE_LIGHT = "#86b6ef"
NEUTRAL = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

COL_W = 3.4  # inches; one IEEE column is 3.5 in.

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "axes.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def tidy(ax, grid_axis="y"):
    """Recessive chrome: hairline grid behind the data, no top/right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


# Fig. 1 - Global household hygiene service ladder, 2022 (WHO/UNICEF JMP 2023).
# 2.0 billion lacked basic hygiene, of whom 653 million had no facility at all.
def fig_hygiene_ladder():
    labels = ["Basic service", "Limited service", "No facility"]
    values = [75, 17, 8]
    colors = [BLUE_DEEP, BLUE, BLUE_LIGHT]
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    wedges, _ = ax.pie(
        values, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for w, v, c in zip(wedges, values, colors):
        ang = (w.theta2 + w.theta1) / 2
        r = 0.62 if v > 12 else 0.78
        x, y = r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))
        ax.text(x, y, f"{v}%", ha="center", va="center", fontsize=8,
                color="white" if c != BLUE_LIGHT else INK, fontweight="bold")
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, handlelength=1.0, handleheight=1.0)
    ax.set_aspect("equal")
    save(fig, "fig1_hygiene_ladder_pie.png")


# Fig. 2 - WASH and menstrual health provision in schools, 2023
# (WHO/UNICEF JMP 2024 schools report; MH indicators from countries with data).
ORANGE = "#eb6834"


def fig_school_wash():
    labels = ["Drinking water", "Sanitation", "Hygiene",
              "Menstrual health education", "Bins for menstrual waste"]
    values = [77, 78, 67, 39, 31]
    colors = [BLUE] * 3 + [ORANGE] * 2
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Schools providing the service (%)")
    for b, v in zip(bars, values):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v}%",
                va="center", ha="left", fontsize=7.5, color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Basic WASH service"),
                       Patch(color=ORANGE, label="Menstrual health provision")],
              loc="upper center", bbox_to_anchor=(0.38, 1.2), ncol=2,
              frameon=False, handlelength=1.0, handleheight=1.0)
    tidy(ax, "x")
    save(fig, "fig2_school_wash_bar.png")


# Fig. 3 - Women aged 15-24 using hygienic methods, NFHS-4 vs NFHS-5 (India).
def fig_nfhs_trend():
    groups = ["Urban", "Rural", "Total"]
    nfhs4 = [77.5, 48.2, 57.6]
    nfhs5 = [89.4, 72.6, 77.3]
    x = range(len(groups))
    w = 0.34
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    b4 = ax.bar([i - w / 2 - 0.01 for i in x], nfhs4, w, color=BLUE_LIGHT,
                label="NFHS-4 (2015–16)")
    b5 = ax.bar([i + w / 2 + 0.01 for i in x], nfhs5, w, color=BLUE_DARK,
                label="NFHS-5 (2019–21)")
    for bars in (b4, b5):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{b.get_height():.1f}", ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Women using hygienic methods (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2,
              frameon=False, handlelength=1.0, handleheight=1.0)
    tidy(ax, "y")
    save(fig, "fig3_nfhs_trend_bar.png")


# Fig. 4 - Urban vs rural share using hygienic methods, NFHS-5 (two pies).
def fig_urban_rural_pies():
    data = [("Urban", 89.4), ("Rural", 72.6)]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.6))
    handles = None
    for ax, (name, hyg) in zip(axes, data):
        wedges, _ = ax.pie(
            [hyg, 100 - hyg], colors=[BLUE_DARK, NEUTRAL], startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
        )
        handles = wedges
        ax.text(0, 0.07, f"{hyg:.1f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color=INK)
        ax.text(0, -0.2, name, ha="center", va="center", fontsize=7.5,
                color=INK_2)
        ax.set_aspect("equal")
    fig.legend(handles, ["Hygienic method", "Non-hygienic method"],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.0, handleheight=1.0)
    fig.subplots_adjust(bottom=0.14, top=1.0, wspace=0.05)
    save(fig, "fig4_urban_rural_pie.png")


# Fig. 5 - Methods used by women aged 15-24, NFHS-5 (multiple responses).
def fig_methods():
    labels = ["Sanitary napkins", "Cloth", "Locally prepared napkins",
              "Tampons", "Menstrual cup"]
    values = [64.4, 49.6, 15.0, 1.7, 0.3]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    bars = ax.barh(labels, values, color=BLUE, height=0.55)
    ax.invert_yaxis()
    ax.set_xlim(0, 80)
    ax.set_xlabel("Women aged 15–24 reporting use (%)")
    for b, v in zip(bars, values):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                va="center", ha="left", fontsize=7.5, color=INK)
    tidy(ax, "x")
    save(fig, "fig5_methods_bar.png")


# Fig. 6 - Hygienic method use by schooling and wealth, NFHS-5 (approx.).
def fig_determinants():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True)
    panels = [
        ("Schooling", ["None", "12+ yrs"], [44, 90]),
        ("Wealth quintile", ["Lowest", "Highest"], [54, 95]),
    ]
    for ax, (title, cats, vals) in zip(axes, panels):
        bars = ax.bar(cats, vals, color=[BLUE_LIGHT, BLUE_DARK], width=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_title(title, color=INK_2, pad=4)
        ax.set_ylim(0, 108)
        tidy(ax, "y")
    axes[0].set_ylabel("Using hygienic methods (%)")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig6_determinants_bar.png")


# Fig. 7 - Estimated lifetime units per user, from Eq. (1)-(2) and Table II.
def fig_lifetime_units():
    years, cycles, days, per_day = 38, 13, 5, 4
    disposable = cycles * years * days * per_day          # Eq. (1)
    cloth = 12 * -(-years // 2)                            # Eq. (2), k=12, L=2
    underwear = 7 * -(-years // 2)                         # k=7, L=2
    cup = 1 * -(-years // 10)                              # k=1, L=10
    labels = ["Disposable pads\nor tampons", "Reusable cloth pads",
              "Period underwear", "Menstrual cup"]
    values = [disposable, cloth, underwear, cup]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    bars = ax.barh(labels, values, color=BLUE, height=0.55)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1, 60000)
    ax.set_xlabel("Estimated units over a menstruating lifetime (log scale)")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for b, v in zip(bars, values):
        ax.text(v * 1.25, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", ha="left", fontsize=7.5, color=INK)
    tidy(ax, "x")
    save(fig, "fig7_lifetime_units_bar.png")
    return values


if __name__ == "__main__":
    fig_hygiene_ladder()
    fig_school_wash()
    fig_nfhs_trend()
    fig_urban_rural_pies()
    fig_methods()
    fig_determinants()
    print("lifetime units:", fig_lifetime_units())
    print("written to", OUT)
