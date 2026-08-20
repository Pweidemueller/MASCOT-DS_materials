"""
Lab color palette — 8 colors from Winterthur to San Francisco.

Each color is sourced from an actual institution, coat of arms, or landmark:

    Kyburg Gold   #D4A020  L*69  Winterthur coat of arms, heraldic Or
    UCSF Teal     #18A3AC  L*61  UCSF brand identity (identity.ucsf.edu)
    Rain          #888886  L*56  Seattle / PNW overcast grey
    Basel         #3574B8  L*47  FC Basel RotBlau, the Blau half
    Bridge        #C0362C  L*44  Golden Gate International Orange (Pantone 180)
    Hutch         #6B2D90  L*32  Fred Hutch / UW Medicine merged brand purple
    Winterthur    #7A1024  L*25  Kyburg Gules, the rampant lions
    Navy          #052049  L*11  UCSF primary navy

Colorblind notes:
  - Basel and Bridge share an L* lane (Δ3) but blue vs red never merge
    under deuteranopia, protanopia, or tritanopia.
  - Winterthur crimson is 19 L* points below Bridge, keeping the two reds
    separable even when both shift olive-brown under deuteranopia.
  - Full 8-color spread from L*11 to L*69 ensures grayscale safety.

Usage:
    import lab_palette as lp

    # Matplotlib
    plt.plot(x, y, color=lp.BRIDGE)
    ax.bar(categories, values, color=lp.PALETTE[:len(categories)])

    # Seaborn
    sns.set_palette(lp.PALETTE)

    # As a named matplotlib colormap (discrete)
    lp.register()
    plt.scatter(x, y, c=values, cmap="lab")

    # Continuous colormaps
    plt.imshow(data, cmap="lab_teal")   # sequential cool
    plt.imshow(data, cmap="lab_warm")   # sequential warm
    plt.imshow(data, cmap="lab_div")    # diverging red–white–blue
"""

from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgba
import matplotlib.pyplot as plt

# ── Individual colors ────────────────────────────────────────────────────────

KYBURG_GOLD = "#D4A020"
UCSF_TEAL  = "#18A3AC"
RAIN        = "#888886"
BASEL       = "#3574B8"
BRIDGE      = "#C0362C"
HUTCH       = "#6B2D90"
WINTERTHUR  = "#7A1024"
NAVY        = "#052049"

# ── Ordered palette (brightest → darkest) ────────────────────────────────────
# This order maximises visual separation when you use the first N colors.

PALETTE = [
    KYBURG_GOLD,  # L*69
    UCSF_TEAL,   # L*61
    RAIN,         # L*56
    BASEL,        # L*47
    BRIDGE,       # L*44
    HUTCH,        # L*32
    WINTERTHUR,   # L*25
    NAVY,         # L*11
]

# ── Short aliases ────────────────────────────────────────────────────────────

NAMES = [
    "Kyburg Gold",
    "UCSF Teal",
    "Rain",
    "Basel",
    "Bridge",
    "Hutch",
    "Winterthur",
    "Navy",
]

# Name → hex lookup
COLOR = dict(zip(NAMES, PALETTE))

# ── Suggested subsets ────────────────────────────────────────────────────────
# Pre-tested subsets for common category counts.

PALETTE_3 = [KYBURG_GOLD, BRIDGE, NAVY]               # max L* spread
PALETTE_4 = [KYBURG_GOLD, UCSF_TEAL, BRIDGE, NAVY]    # safest 4
PALETTE_5 = [KYBURG_GOLD, UCSF_TEAL, BASEL, HUTCH, NAVY]
PALETTE_6 = [KYBURG_GOLD, UCSF_TEAL, BASEL, BRIDGE, HUTCH, NAVY]

# ── Continuous colormaps ─────────────────────────────────────────────────────
# Three perceptually smooth colormaps built from palette anchor colors.
#
#   lab_teal   — sequential cool:  light → UCSF Teal → Navy
#   lab_warm   — sequential warm:  light → Kyburg Gold → Bridge → Winterthur
#   lab_div    — diverging:        Bridge (red) ← white → Basel (blue)

CMAP_TEAL = LinearSegmentedColormap.from_list(
    "lab_teal", ["#e0f5f5", UCSF_TEAL, NAVY], N=256)
CMAP_TEAL_R = LinearSegmentedColormap.from_list(
    "lab_teal_r", [NAVY, UCSF_TEAL, "#e0f5f5"], N=256)

CMAP_WARM = LinearSegmentedColormap.from_list(
    "lab_warm", ["#fdf0d0", KYBURG_GOLD, BRIDGE, WINTERTHUR], N=256)
CMAP_WARM_R = LinearSegmentedColormap.from_list(
    "lab_warm_r", [WINTERTHUR, BRIDGE, KYBURG_GOLD, "#fdf0d0"], N=256)

CMAP_DIV = LinearSegmentedColormap.from_list(
    "lab_div", [BRIDGE, "#f5c4c0", "#ffffff", "#b8d4f0", BASEL], N=256)
CMAP_DIV_R = LinearSegmentedColormap.from_list(
    "lab_div_r", [BASEL, "#b8d4f0", "#ffffff", "#f5c4c0", BRIDGE], N=256)

# ── Discrete colormap (existing) ────────────────────────────────────────────

CMAP = ListedColormap(PALETTE, name="lab")
CMAP_R = ListedColormap(PALETTE[::-1], name="lab_r")


def register():
    """Register all lab colormaps with matplotlib.

    Registered names: lab, lab_r, lab_teal, lab_teal_r,
    lab_warm, lab_warm_r, lab_div, lab_div_r.
    """
    for cmap in [CMAP, CMAP_R, CMAP_TEAL, CMAP_TEAL_R,
                 CMAP_WARM, CMAP_WARM_R, CMAP_DIV, CMAP_DIV_R]:
        try:
            plt.colormaps.register(cmap)
        except ValueError:
            pass  # already registered


def set_palette():
    """Set the lab palette as the default matplotlib color cycle."""
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTE)


# ── Seaborn helper ───────────────────────────────────────────────────────────

def sns_palette(n=None):
    """Return the palette as a list of RGBA tuples for seaborn.

    Parameters
    ----------
    n : int, optional
        Number of colors to return. Defaults to all 8.
        Uses the pre-tested subsets when n is 3–6.
    """
    subsets = {3: PALETTE_3, 4: PALETTE_4, 5: PALETTE_5, 6: PALETTE_6}
    if n is None or n >= 8:
        colors = PALETTE
    elif n in subsets:
        colors = subsets[n]
    else:
        colors = PALETTE[:n]
    return [to_rgba(c) for c in colors]


# ── Quick swatch display ─────────────────────────────────────────────────────

def show(figsize=(8, 1.2)):
    """Display the palette as a horizontal swatch strip."""
    fig, ax = plt.subplots(figsize=figsize)
    for i, (color, name) in enumerate(zip(PALETTE, NAMES)):
        ax.barh(0, 1, left=i, color=color, edgecolor="white", linewidth=0.5)
        lum = [69, 61, 56, 47, 44, 32, 25, 11][i]
        text_color = "white" if lum < 50 else "black"
        ax.text(i + 0.5, 0, f"{name}\n{color}", ha="center", va="center",
                fontsize=7, color=text_color, fontweight="medium")
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    show()
    plt.show()
