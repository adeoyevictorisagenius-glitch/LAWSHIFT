"""
LAWSHIFT colour system (Dark / Evening Study Edition): one place for the palette,
used by both the app CSS and the charts.

The palette follows the 60-30-10 rule:
  60 % dominant warm dark neutrals - deep espresso/charcoal paper, elevated cards
  30 % secondary sage / forest green - sidebar, headings, tabs, main data colour
  10 % accent terracotta - primary buttons, rank-1 badge, the single thing to look at

The accent is deliberately rare. In charts it marks only the highlight
(held-out points, the optimum, flagged observations); everything else is neutral or sage.
"""
from __future__ import annotations

# ---- 60 %: warm dark neutrals -----------------------------------------------------
CREAM = "#171A17"          # app background (deep forest-tinted charcoal)
PAPER = "#202420"          # cards, chart panels (elevated dark surface)
LINEN = "#313731"          # hairlines, grid lines, subtle borders
INK = "#EAE5DA"            # body text (12.8:1 contrast on CREAM)
MUTED = "#A39E93"          # secondary text (6.2:1 contrast on CREAM)
TAUPE = "#7C766A"          # neutral data colour (third series, baselines)

# ---- 30 %: sage (lifted for dark-surface readability) -----------------------------
SAGE_DEEP = "#C8DACD"      # headings, sidebar text (11.1:1 on CREAM)
SAGE_TEXT = "#98B8A2"      # links, active tab (7.8:1 on CREAM)
SAGE = "#789B84"           # main data colour, borders on hover
SAGE_MID = "#587562"       # second series, dashed borders
SAGE_SOFT = "#222B24"      # sidebar background, soft insight tints

# ---- 10 %: accent -----------------------------------------------------------------
TERRACOTTA = "#D46A42"     # primary button (dark/light readable), highlights
TERRACOTTA_SOFT = "#38221A"# warm low-glow tint for warnings & hidden-box

RANK_COLORS = [SAGE, SAGE_MID, TAUPE, SAGE_DEEP, LINEN]

_FONT_IMPORT = """
@import url('https://googleapis.com');
"""

_CSS = """
<style>
__FONT_IMPORT__

:root {
  --cream: __CREAM__;
  --paper: __PAPER__;
  --linen: __LINEN__;
  --ink: __INK__;
  --muted: __MUTED__;
  --sage: __SAGE__;
  --sage-deep: __SAGE_DEEP__;
  --sage-text: __SAGE_TEXT__;
  --sage-soft: __SAGE_SOFT__;
  --sage-mid: __SAGE_MID__;
  --accent: __TERRACOTTA__;
  --accent-soft: __TERRACOTTA_SOFT__;
  --taupe: __TAUPE__;
}

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span {
  color: var(--ink);
}
.stApp {
  background: var(--cream);
}
.block-container {
  padding-top: 1.6rem;
  max-width: 1200px;
}
header[data-testid="stHeader"] {
  background: transparent;
}
#MainMenu, footer {
  visibility: hidden;
}

/* ---- headings ---- */
h1.page-title {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 600;
  font-size: 2rem;
  color: var(--sage-deep);
  margin: .1rem 0 .15rem 0;
  padding: 0;
  letter-spacing: -.01em;
}
.page-sub {
  color: var(--muted);
  font-size: 1rem;
  margin-bottom: 1.1rem;
  max-width: 62ch;
}
h2, h3 {
  font-family: 'Fraunces', Georgia, serif !important;
  color: var(--sage-deep) !important;
  font-weight: 600 !important;
}
.section-label {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.18rem;
  font-weight: 600;
  color: var(--sage-deep);
  margin: 1.9rem 0 .7rem 0;
  padding-bottom: .35rem;
  border-bottom: 1px solid var(--linen);
}
.small-label {
  font-size: .78rem;
  font-weight: 600;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--sage-text);
  margin: .2rem 0 .3rem 0;
}

/* ---- sidebar (the 30%) ---- */
section[data-testid="stSidebar"] {
  background: var(--sage-soft);
  border-right: 1px solid var(--linen);
}
section[data-testid="stSidebar"] * {
  color: var(--sage-deep);
}
.brand {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 600;
  font-size: 1.45rem;
  color: var(--sage-deep);
  letter-spacing: .02em;
  display: flex;
  align-items: center;
  gap: 10px;
}
.brand-sub {
  color: var(--muted) !important;
  font-size: .82rem;
  margin: 2px 0 10px 0;
}

/* ---- metrics ---- */
div[data-testid="stMetric"] {
  background: var(--paper);
  border: 1px solid var(--linen);
  border-radius: 12px;
  padding: 12px 16px;
}
div[data-testid="stMetricLabel"] p {
  font-size: .8rem !important;
  color: var(--muted);
}
div[data-testid="stMetricValue"] {
  color: var(--sage-deep);
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.5rem;
}

/* ---- buttons: sage outline by default, terracotta only for the primary action ---- */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px;
  border: 1px solid var(--sage);
  background: var(--paper);
  color: var(--sage-deep);
  font-weight: 500;
  transition: background .15s ease, border-color .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--sage-soft);
  border-color: var(--sage-deep);
  color: var(--sage-deep);
}
.stButton > button[kind="primary"] {
  background: var(--accent);
  border: 1px solid var(--accent);
  color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover {
  background: #BD5933;
  border-color: #BD5933;
  color: #FFFFFF;
}

/* ---- tabs, inputs, alerts ---- */
button[data-baseweb="tab"] {
  font-weight: 500;
  font-size: .92rem;
  color: var(--muted);
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--sage-text);
}
div[data-baseweb="tab-highlight"] {
  background-color: var(--sage) !important;
}
div[data-testid="stAlert"] {
  background: var(--paper);
  border: 1px solid var(--linen);
  border-radius: 10px;
}
div[data-testid="stProgress"] div[role="progressbar"] > div {
  background: var(--sage) !important;
}
div[data-testid="stDataFrame"] {
  border: 1px solid var(--linen);
  border-radius: 10px;
}
div[data-testid="stFileUploader"] section {
  background: var(--paper);
  border: 1px dashed var(--sage-mid);
  border-radius: 10px;
}
hr {
  border-color: var(--linen);
}
div[data-testid="stChatMessage"] {
  background: var(--paper);
  border: 1px solid var(--linen);
  border-radius: 14px;
  padding: .9rem 1.1rem;
}
div[data-testid="stChatMessage"] p {
  line-height: 1.6;
}
div[data-testid="stExpander"] {
  border: 1px solid var(--linen);
  border-radius: 10px;
  background: var(--paper);
}

/* ---- notes ---- */
.demo-banner {
  border: 1px solid var(--linen);
  background: var(--paper);
  color: var(--muted);
  padding: 9px 14px;
  border-radius: 10px;
  font-size: .85rem;
  margin: 0 0 1.1rem 0;
}
.demo-banner b {
  color: var(--sage-deep);
}

/* ---- expression cards ---- */
.rank {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.4rem;
  height: 2.4rem;
  border-radius: 50%;
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 600;
  font-size: 1.05rem;
  border: 1.5px solid var(--sage);
  color: var(--sage-deep);
  background: var(--paper);
}
.rank.r1 {
  background: var(--accent);
  border-color: var(--accent);
  color: #FFFFFF;
}
.cand-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: .4rem 0 .2rem 0;
}
.chip {
  border: 1px solid var(--linen);
  background: var(--cream);
  border-radius: 999px;
  padding: 3px 11px;
  font-size: .8rem;
  display: inline-flex;
  gap: 7px;
}
.chip-l {
  color: var(--muted);
}
.chip-v {
  color: var(--sage-deep);
  font-weight: 600;
}
.cand-note {
  margin-top: 6px;
  font-size: .84rem;
  color: var(--muted);
}

/* ---- insights ---- */
.insight-head {
  border-left: 3px solid var(--sage);
  background: var(--sage-soft);
  padding: 10px 14px;
  margin: 6px 0 12px 0;
  font-size: 1rem;
  border-radius: 0 10px 10px 0;
}
.insight {
  border-left: 2px solid var(--sage-mid);
  padding: 5px 12px;
  margin: 6px 0;
  font-size: .93rem;
}
.insight.warn {
  border-left-color: var(--accent);
  background: var(--accent-soft);
  border-radius: 0 8px 8px 0;
}
.hidden-box {
  border: 1px dashed var(--accent);
  background: var(--accent-soft);
  border-radius: 10px;
  padding: 10px 16px;
}
.pill {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: .78rem;
  font-weight: 600;
  background: var(--sage-soft);
  color: var(--sage-deep);
}
.pill.accent {
  background: var(--accent-soft);
  color: var(--accent);
}
</style>
"""


def build_css() -> str:
    """The app stylesheet with palette tokens filled in."""
    tokens = {
        "FONT_IMPORT": _FONT_IMPORT,
        "CREAM": CREAM,
        "PAPER": PAPER,
        "LINEN": LINEN,
        "INK": INK,
        "MUTED": MUTED,
        "SAGE": SAGE,
        "SAGE_DEEP": SAGE_DEEP,
        "SAGE_TEXT": SAGE_TEXT,
        "SAGE_SOFT": SAGE_SOFT,
        "SAGE_MID": SAGE_MID,
        "TERRACOTTA_SOFT": TERRACOTTA_SOFT,
        "TERRACOTTA": TERRACOTTA,
        "TAUPE": TAUPE,
    }
    css = _CSS
    for key in sorted(tokens, key=len, reverse=True):
        css = css.replace(f"__{key}__", tokens[key])
    return css
