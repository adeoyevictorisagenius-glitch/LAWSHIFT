# LAWSHIFT — AI-powered scientific & mathematical discovery engine

> **Status: demo prototype.** Everything the app shows is computed live by the code in this repository.
> Physics data are *simulated* from documented simplified models; business data are *synthetic* unless you upload your own.
> Not a substitute for laboratory measurement or certified engineering/financial analysis.

LAWSHIFT searches for mathematical equations hidden in noisy data using **evolutionary symbolic regression**,
tests the best equations on data they have never seen, explains them in plain English (alongside the full technical
analysis), and can use them to **choose the settings** that push a variable of interest where you want it.

```
Observed data → candidate equations → evolutionary search → best equations → independent validation
              → plain-English read-out → optional optimisation of the settings
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest tests -q         # all tests should pass
streamlit run app.py
```

No paid APIs, cloud services or external AI are used. Everything runs locally.

## Command-line demos (no UI needed)

```bash
python demo_engine.py             # rediscover F = m*a from noisy data
python demo_lab.py --jobs 4       # all three physics experiments with full reports
```

## Use your own data (CSV)

1. **Everyday Data Lab** (any ordinary table) or **Scientific Discovery Lab → Your data (CSV)**, then upload a file (or pick a built-in synthetic sample).
2. **Column headers become the variable names** in the equations (`Length (m)` → `Length_m`).
3. **You choose the variable of interest** (the dependent variable). Every other numeric column is a candidate independent
   variable. The engine cannot infer which variable depends on which, so that choice is yours.
4. Click **Discover expressions**. You get the **top 3 distinct expressions** typeset in LaTeX (the best training fit plus
   alternatives that trade accuracy against simplicity). Each has an **In plain English** read-out and **Technical insights**:
   which variables drive it, in which direction, whether returns diminish, where a sweet spot sits, whether variables
   interact, and the typical error on unseen data.
5. Open the **Optimise** tab to choose settings for a goal (see below).

CSV requirements: header row, at least 30 complete rows, numeric columns, up to 6 independent variables, 50,000 rows, 5 MB.
Rows with missing values are dropped and text columns ignored, and the app tells you what it did.
Sample files live in `sample_data/` (regenerate with `python make_sample_data.py`); they are **synthetic**.

## What is in the app

| Page | What it does |
|---|---|
| Home | Real statistics from stored experiments, accuracy-vs-noise chart, one-click quick suite |
| Scientific Discovery Lab | **Your data (CSV)** with top-3 expressions, or simulated systems (Newtonian, harmonic, projectile) with a hidden law revealed after validation |
| Everyday Data Lab | Any ordinary table: overview, **distributions and normality checks**, relationships, then the pattern finder and optimiser |
| Business Pattern Lab | Demo dataset or CSV upload: top-3 expressions, correlations, regression, prediction errors, unusual observations |
| Experiment History | Every run stored in `runs/experiment_history.jsonl`; inspect, download or delete |
| About LAWSHIFT | Method, assumptions, limitations |

### Everyday Data Lab (ordinary, "normal" data)

For tables that are not physics or business: study habits, shop sales, survey numbers.

- **Overview**: row and column counts, a summary table, a histogram for every column, and a plain-English read of the whole table.
- **Distributions & normality**: for any column, a histogram with the matching normal curve, a Q-Q plot, Shapiro-Wilk, skewness,
  excess kurtosis, the 68-95-99.7 rule check, outlier counts and a plain verdict (normal, nearly normal, skewed, heavy tails...)
  with a suggestion (for example a log transform for a long right tail). Verdicts combine the test with the shape, because a
  test alone is too strict with many rows and too lenient with few. Formulas are shown in LaTeX.
- **Relationships**: correlation heatmap, strongest pairs, any pair as a scatter plot.
- **Pattern finder & optimiser**: the discovery engine, read-out and optimiser described here.

Samples: `coffee_shop.csv` (diminishing returns, a temperature sweet spot, a distractor) and `study_habits.csv` (a sleep sweet spot,
a skewed column, a distractor).

### Choosing optimal values (Optimise tab)

Say how much of the variable of interest you want: **as high as possible**, **as low as possible**, **close to a specific value**
or **inside a range**. Keep some variables fixed (the weather, say) and narrow the range of others. The optimiser searches the
fitted expression with differential evolution plus a local polish, and:

- never leaves the range your data cover (no extrapolation);
- among equally good settings prefers the one closest to typical conditions, and says when a variable barely matters;
- flags settings pinned at the edge of your data, "touchy" variables, and combinations of values far from any real row;
- compares the recommendation with the nearest and the best real observations;
- runs every one of the top expressions and tells you whether they agree.

It describes the fitted equation, not cause and effect.

### Plain-English read-out

Each expression, each column and each optimisation gets a conversational note ("here is what this equation is telling me...")
with the technical terms kept in brackets next to the plain wording. It is **rule-based**: sentences are assembled from computed
quantities (held-out R², permutation importance, partial-effect curves, elasticity, residual diagnostics, optimiser output). It is
not a language model, nothing leaves your machine, the same inputs give the same words, and every claim can be checked in the
technical tabs beside it.

### Look and feel

Equations are typeset with LaTeX (Streamlit's built-in KaTeX); plain text is one click away for copying into code. Colours follow
the **60-30-10 rule** and live in one file, `src/theme.py`, shared by the app and the charts:

| Share | Role | Colours |
|---|---|---|
| 60% | Dominant neutrals: page, cards, chart panels | cream `#F6F1E7`, paper `#FFFCF6`, linen `#E6DECF` |
| 30% | Secondary: sidebar, headings, tabs, main data colour | sage `#5B7A66`, deep sage `#2F4A3C`, soft sage `#E1E8DC` |
| 10% | Accent: primary buttons, rank-1 badge, highlights | terracotta `#B5532D` |

In charts the accent marks only what deserves the eye (held-out points, the recommended setting, flagged observations). Text
colours meet WCAG AA contrast (tested).

## Scientific design (why the results can be trusted)

- **Hidden law.** Data come from a hidden equation. The search receives only training `(X, y)`; the law is displayed after validation. A test intercepts the call to the engine to prove this.
- **Held-out validation.** Train/validation rows never overlap. The best search is chosen by *training* fitness only.
- **Noise ceiling.** The score the true law itself gets on the same noisy held-out data. Matching it means the discovery generalises as well as physically possible.
- **Several independent searches** per experiment; the app reports whether they agree.
- **Complexity penalty** and warnings: a complicated equation with good accuracy may be a coincidence.
- **Rank by training only.** The top 3 are ranked on training fitness; held-out data never selects or ranks anything.
- **Search agreement.** For uploaded data (no ground truth) the app reports how many independent searches reached an equivalent best equation.
- **Insights are computed, not scripted.** Importance = permutation test on held-out rows; direction and shape = partial-effect curves; they describe the fitted equation inside your data range and are *not* proof of cause and effect.
- **Honest optimiser.** Recommendations stay inside the observed range, are reported with the equation's held-out error, and are cross-checked against real rows and against the other top expressions.
- **Exact units.** Columns are rescaled for the search; winners are converted back exactly into your original units.
- **Safe evaluation.** Equations are object trees, not strings; there is no `eval()`. Division by zero, overflow and NaN are neutralised and all values are bounded.
- **Reproducible.** Seeds control data generation, splitting and search.

## Project layout

```
app.py                       Streamlit dashboard
src/theme.py                 the 60-30-10 palette and the app stylesheet (also used by the charts)
src/discovery/               expressions, population, selection, crossover, mutation, fitness,
                             evolution (GA loop + hall of fame), simplify, scaling, candidates (top-3),
                             insights, latex (LaTeX output + formula library), diagnostics (residual checks),
                             optimize (choose settings for a goal), narrative (plain-English read-outs),
                             experiment (simulated cycle), dataset_experiment (your CSV)
src/physics/                 motion.py, oscillation.py, projectile.py (simulated systems + assumptions)
src/data/                    generators.py (noisy observations), preprocessing.py (splits, cleaning),
                             normality.py (column profiles, normality verdicts, Q-Q data)
src/business/analysis.py     correlations, regression, evolved equations, anomaly detection
src/visualization/charts.py  all charts (Matplotlib), driven by real results
src/utils/                   metrics.py, history.py
tests/                       automated tests for maths, engine, physics, experiments, business, charts,
                             LaTeX validity, normality, optimiser, read-outs and the colour system
```

## Limitations (please read)

- Physics systems are **simplified**: one dimension, no air resistance, no damping, Gaussian noise only.
- The operator set is deliberately small; very complex laws may not be found.
- Results vary with the random seed; different-looking equations can fit the same data range equally well.
- Validation covers the sampled input range only (no extrapolation tests yet), so the optimiser is confined to that range.
- Searches use the CPU and take from seconds to minutes depending on the effort preset.
- Business analysis shows **association, not causation**. Unusual-observation flags are statistical, not accusations of error or fraud.
- Uploaded data has no ground truth: held-out accuracy and search agreement are the only evidence, and neither proves the equation is a true law.
- Insights and optimiser recommendations hold only inside the range of your data and describe the equation, not cause and effect.
- Normality verdicts are guidance: the Shapiro-Wilk test uses at most 5,000 values (larger columns are sampled), and a "normal" verdict does not prove the data are normal.
- The plain-English read-out is templated, so it is accurate to the numbers but will not notice context only a person would know.

## Suggested 4-minute demo for judges

*(Fastest strong opener: Everyday Data Lab → sample "Study Habits", variable of interest Exam_score. The engine finds a
saturating effect for study time and a sleep sweet spot, the read-out says so in plain English, and the optimiser recommends
about eight hours of sleep.)*

1. **Home** (20 s): "Every number here is computed from experiments on this machine." Click *Run quick discovery suite* if the history is empty.
2. **Everyday Data Lab** (90 s): load *Study Habits*. On *Distributions & normality* show that screen time is skewed and sleep is normal. Run
   **Discover expressions**, then read the **In plain English** tab against the **Technical insights** tab.
3. **Optimise** (60 s): "as high as possible" for Exam_score, then keep Screen_time fixed and re-run. Point out the edge-of-data warning and
   whether the top expressions agree.
4. **Scientific Discovery Lab → Projectile** (45 s): set noise 2 %, *Quick demo*, run. Show the typeset equation, held-out R² vs the **noise ceiling**,
   then *Post-experiment verification*, where the hidden law is revealed.
5. **Business Pattern Lab** (30 s): demo dataset, *Analyse patterns*. Show the causation warning, the linear vs evolved comparison and the flagged unusual rows.
6. **About** (10 s): limitations, no paid services.
