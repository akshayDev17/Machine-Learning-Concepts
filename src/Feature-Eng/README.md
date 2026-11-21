# Feature Engineering for tabular data — explained from zero

> **Who this is for:** someone who is new to machine learning. You need school algebra (what an average is, what a
> square or a logarithm does to a number) and a little Python. No statistics is assumed: every idea — median,
> skewness, correlation, ROC-AUC, cross-validation — is explained here before it is used.
>
> **The setting:** a table of rows (customers) and columns (age, income, plan, ...), and a target with two classes
> that are about equally common (churned / stayed). That is the easiest case in machine learning, which makes it the
> right place to learn the steps.
>
> **How to read it:** sections 0–2 give the big picture and the few ideas you need. Sections 3–21 are the 19 steps,
> each with the same five questions: *what is it, why does it help, which models care, when does it hurt, what can
> leak?* Section 22 is a cheat sheet by model, section 23 shows how to run all steps as a configurable pipeline.
> Every claim is demonstrated in the notebook in [`lab/`](lab/README.md), on one deliberately messy table.

## Table of contents

0. [Feature engineering in 30 seconds](#0-feature-engineering-in-30-seconds)
1. [The setting: tables, targets, models](#1-the-setting-tables-targets-models)
2. [The toolbox: the only ideas you need](#2-the-toolbox-the-only-ideas-you-need)
3. [Step 1 · Split first](#3-step-1--split-first-stratified-hold-out)
4. [Step 2 · Understand column types](#4-step-2--understand-column-types)
5. [Step 3 · Drop useless columns](#5-step-3--drop-useless-columns-ids-constants-duplicates)
6. [Step 4 · Remove leakage](#6-step-4--remove-leakage)
7. [Step 5 · Fix data types](#7-step-5--fix-data-types)
8. [Step 6 · Missing values](#8-step-6--missing-values-imputation-plus-was-missing-flags)
9. [Step 7 · Outliers](#9-step-7--outliers-cap-winsorise-transform)
10. [Step 8 · Skewed numerics](#10-step-8--skewed-numerics-log-box-cox-yeo-johnson)
11. [Step 9 · Scaling](#11-step-9--scaling-and-why-trees-dont-need-it)
12. [Step 10 · Categorical encoding](#12-step-10--categorical-encoding)
13. [Step 11 · Rare-category grouping](#13-step-11--rare-category-grouping)
14. [Step 12 · Date / time features](#14-step-12--date--time-features)
15. [Step 13 · Domain features](#15-step-13--domain-features-ratios-differences-aggregates)
16. [Step 14 · Interactions and polynomials](#16-step-14--interactions-and-polynomials)
17. [Step 15 · Binning](#17-step-15--binning)
18. [Step 16 · Correlated / redundant features](#18-step-16--correlated--redundant-features-correlation-vif-pca)
19. [Step 17 · Feature selection](#19-step-17--feature-selection-filter-wrapper-embedded)
20. [Step 18 · Dimensionality reduction](#20-step-18--dimensionality-reduction-pca-lda-when-labels-exist)
21. [Step 19 · Validate every step by cross-validation](#21-step-19--validate-every-step-by-cross-validation)
22. [Cheat sheet: which steps matter for which model](#22-cheat-sheet-which-steps-matter-for-which-model)
23. [Running the whole pipeline (scripts + config.yml)](#23-running-the-whole-pipeline-scripts--configyml)
24. [Common misconceptions](#24-common-misconceptions)
25. [The labs](#25-the-labs)
26. [Glossary](#26-glossary)
27. [Further reading](#27-further-reading)
28. [Appendix - Assumptions made](#appendix---assumptions-made)

---

## 0. Feature engineering in 30 seconds

* A model only sees numbers in columns. **Feature engineering** is everything you do to the columns *before* the
  model sees them: fixing them, filling them, reshaping them, combining them, and removing the ones that only add noise
  or cheat.
* It matters because a model can only learn patterns it can *express*. A straight-line model can't see that both
  young and old customers leave unless you give it an `age²` column; no model can use a date written as text.
* The golden rule: **anything a step learns from data (a median, a category's churn rate, which columns to keep) must
  be learned from the training rows only.** Break it and your scores lie (this is called **leakage**).
* The second rule: **check every step by cross-validation.** Many sensible-sounding steps do nothing, and some hurt.
* In the lab, the same customer table goes from a cross-validated ROC-AUC of **0.727** (no feature engineering) to
  **0.895** (all steps) with plain logistic regression — more than any change of model achieves.

---

## 1. The setting: tables, targets, models

**The table.** Each **row** is one example (here, one customer). Each **column** is one **feature** (age, income,
plan, ...). One special column is the **target**: what we want to predict (here `churned`: 1 = left, 0 = stayed).

**Balanced binary target.** *Binary* = two classes. *Balanced* = about as many of each (50 / 50 here). That makes it
the easiest case: accuracy is meaningful (0.5 = coin flip), no class needs special weighting, and every step below
works without extra care. (Rare targets such as fraud need extra techniques, not covered here.)

**Kinds of columns.**

| Kind | Example | What the model needs |
|---|---|---|
| Numeric, continuous | income, age | a number (sometimes reshaped: log, scaled, capped) |
| Numeric, count | number of support calls | a number; often long-tailed |
| Categorical, few levels | plan, payment method | turned into numbers (encoding) |
| Categorical, many levels | city (119 values) | encoded carefully (rare levels, target encoding) |
| Ordinal | basic < standard < premium | numbers that keep the order |
| Date / time | sign-up date | turned into "days since", month, weekday, ... |
| Free text | a support note | turned into numbers (keyword flags, counts, ...) |
| Identifier | customer ID | usually dropped |

**Four families of models** behave very differently towards features. This README keeps coming back to them:

| Family | Examples | How it uses a feature |
|---|---|---|
| **Linear** | logistic regression, linear SVM | adds up *weight × value*; can only draw straight-line effects |
| **Distance-based** | k-nearest neighbours (k-NN), k-means, RBF SVM | compares rows by distance; every column's *scale* matters |
| **Tree-based** | decision tree, random forest, gradient boosting (XGBoost, LightGBM) | asks "is value > threshold?"; only the *order* of values matters |
| **Neural networks** | multilayer perceptron | like linear models stacked; need scaled inputs, learn shapes slowly from small tables |

---

## 2. The toolbox: the only ideas you need

### 2.1 Mean, median, quantiles, standard deviation

* **Mean** = the average. **Median** = the middle value when sorted. With a few huge values (incomes), the mean is
  pulled up and the median is not — that is why the median is the safer "typical value".
* **Quantile** (or percentile): the value below which a given share of the data lies. The 99th percentile of income
  is the income that 99 % of customers are below. **Q1, Q3** are the 25th and 75th percentiles; the **IQR**
  (inter-quartile range) is Q3 − Q1, the width of the middle half.
* **Standard deviation (std)**: the typical distance of values from their mean — how spread out a column is.

### 2.2 Skewness

A column is **right-skewed** when most values are small and a few are very large (income, spend, counts): its
histogram has a long tail to the right. Skewness is a number that measures this: **0 = symmetric**, above 1 = a
strong right tail, negative = a left tail. Income in the lab has skewness 2.45.

### 2.3 Correlation

The **correlation** of two columns, between −1 and +1, says how well a straight line links them. +1: when one goes
up the other goes up by a fixed amount (weekly and monthly logins); 0: no straight-line link; −1: one up, the other
down. It only sees *straight-line* links and only *pairs* of columns.

### 2.4 How we score a model: accuracy and ROC-AUC

A classifier outputs a **churn probability** for each customer.

* **Accuracy**: call everyone above 0.5 a churner and count the share of correct calls. 0.5 = coin flip here.
* **ROC-AUC**: pick one random churner and one random non-churner. ROC-AUC is the chance that the churner got the
  higher probability. **0.5 = coin flip, 1.0 = perfect.** It judges the *ranking* of customers, not a single cut-off,
  so it is the main score in this README and the labs.

### 2.5 Train, validation, test — and cross-validation

* You train on some rows and score on **other** rows. Scoring on rows the model was trained on is meaningless (a
  model can memorise them, Lab 3).
* **Test set**: a part of the rows locked away at the very start and scored **once**, at the very end.
* **Cross-validation (CV)**: to compare choices *without* touching the test set, cut the training rows into k parts
  (folds, here 5). Train on 4, score on the 5th; repeat 5 times so every part is scored once; average the 5 scores.
  The spread of the 5 scores tells you how much a score can wobble by chance (about ±0.01 in the lab).

### 2.6 Fit vs transform — and leakage

Many steps **learn** something from data (**fit**: the median to fill gaps with, a category's churn rate, which
columns to keep) and then **apply** it (**transform**). The rule:

> **Fit on training rows only. Transform training rows, validation rows and test rows with what was learned.**

**Leakage** is any way information from the rows you score on (or from the future) sneaks into training. It makes
scores look better than they will be on real new data. It comes in two flavours, both covered below:

1. **Leaky columns** — a column that is only known *because of* the outcome (Step 4).
2. **Leaky procedures** — fitting a step on rows that are later used for scoring (Steps 1, 10, 17, 19).

In scikit-learn the clean way to obey the rule is a **`Pipeline`**: chain the steps and the model, and
cross-validation then re-fits every step inside every fold automatically. The scripts in [`scripts/`](scripts/README.md)
do the same thing explicitly, stage by stage (section 23).

---

## 3. Step 1 · Split first (stratified hold-out)

**What it is.** Before looking closely at the data, set aside a **test part** (often 20 %) and don't touch it until
the end. **Stratified** splitting keeps the same share of each class in every part (50 / 50 in both).

**Why it helps.** Every later step learns from data. If the test rows take part in that learning, the final test
score is no longer a fair preview of new data. Splitting first makes "fit on training rows only" possible at all.
Stratifying matters most with small data: in Lab 1, random splits of 200 rows give test parts with anywhere from
30 % to 70 % churners; stratified splits always give the same share as the whole.

**Which models.** All of them. **When it hurts:** never — but with very little data, a single hold-out is noisy, so
rely on cross-validation for decisions and keep the test part for one final check.

**Leakage risk.** This step *is* the protection. Two traps: rows that belong together (the same customer twice)
should land in the same part; and with data over time, the test part should be *later* than the training part.

*Lab 1.*

---

## 4. Step 2 · Understand column types

**What it is.** Go through every column and decide what it really holds — number, number stored as text, date,
category (few or many levels), ordered category, free text, identifier, constant — regardless of how the file
stored it.

**Why it helps.** The kind decides every later step: you scale numbers, encode categories, expand dates, drop IDs.
A file reader guesses types from the characters it sees, so `"$49.44"` becomes text and `2024-10-13` becomes text.

**Which models.** All. **When it hurts:** it can't, but automatic guesses can be wrong (a count looks "continuous",
1–5 survey answers look like codes): always confirm by eye.

**Leakage risk.** None by itself. But looking for "useful-looking" columns using the target on *all* rows is the
first step of the selection leak in Step 17 — look at the target only on training rows.

*Lab 2: a column report with type, number of distinct values, share missing and a guessed kind.*

---

## 5. Step 3 · Drop useless columns (IDs, constants, duplicates)

**What it is.** Remove columns that cannot carry information:

* **Identifiers** — a different value on every row (customer ID).
* **Constants** — the same value on every row (`country = US`).
* **Duplicates** — an exact copy of another column (`monthly_fee` = `monthly_charges`).

**Why it helps.** Fewer columns, less noise, and no chance for the model to **memorise** an ID: in Lab 3 a tree on
the ID number alone is 100 % right on its training rows and a coin flip (0.518) on new rows.

**Which models.** All. Trees and k-NN are the most easily fooled by IDs.

**When it hurts.** When the "ID" secretly carries information — e.g. IDs handed out in order encode *when* the
customer joined. Then extract that information on purpose (a sign-up date) instead of keeping the raw ID.

**Leakage risk.** IDs and row numbers can leak: if the data was sorted by outcome before IDs were assigned, the ID
predicts the outcome. Another reason to drop them.

---

## 6. Step 4 · Remove leakage

**What it is.** Drop every column whose value would **not be known at the moment you make the prediction**. In the
lab, `retention_call_made` is a call made *after* a customer asks to cancel.

**Why it helps.** A leaky column makes validation scores wonderful and real predictions useless. In Lab 4 it alone
scores 0.936 (the next best column 0.643) and lifts the model from 0.702 to 0.960 — numbers that would collapse in
production.

**How to find leaks.**

* Ask for every column: *"Would I know this value at prediction time?"*
* Be suspicious of any single column that predicts far better than all others.
* Typical leaks: status or outcome codes, dates after the event, totals that include the future, refunds,
  "reason for cancelling", anything computed using the target.

**Which models.** All. A flexible model finds a leak faster.

**When it hurts.** Dropping a column that *is* available at prediction time loses real signal — that's why the
decision needs domain knowledge, not just a high score.

**Leakage risk.** This step is about leakage. Cross-validation **cannot** detect a leaky column — it rewards it
(Lab 20: switching this step off "improves" CV ROC-AUC from 0.895 to 0.984).

---

## 7. Step 5 · Fix data types

**What it is.** Turn each column into its real type: money text → numbers, `Y`/`N` → 1/0, date text → dates; unify
spellings (`"Credit Card"`, `"credit_card"`, `" credit card "` → `"credit card"`); turn **impossible** values
(age 0, 199, 999) into *missing*.

**Why it helps.** Columns stored as text are invisible to most models, and split spellings divide one category's
rows into several weak ones. In Lab 5, 13 spellings of 4 payment methods become 4, the "mailed check" signal
(churn share 0.63 vs 0.46–0.49) becomes visible, and the score rises from 0.702 to 0.776 — just by letting the
model read the data.

**Which models.** All.

**When it hurts.** When "impossible" is judged wrongly: an age of 105 might be real. Use domain limits, not
statistics, for this step (statistical extremes are Step 7).

**Leakage risk.** None — these are fixed rules, nothing is learned from the data.

---

## 8. Step 6 · Missing values (imputation plus "was missing" flags)

**What it is.** Models (except some tree models) can't handle empty cells. Options:

1. **Drop rows** with gaps — loses data, and a new customer with a gap still needs a prediction.
2. **Impute** — fill the gap with a typical value: the **median** for numbers (robust to extremes), the most frequent
   value or the word `"missing"` for categories.
3. **Impute + flag** — also add a 0/1 column `income_was_missing`.

**Why it helps — and why the flag matters.** Gaps can be:

* *missing at random* (age forgotten by chance) — filling is enough;
* *missing for a reason* (customers about to leave don't bother entering their income) — then the **fact that it is
  missing is information**, and the flag carries it.

In Lab 6 income is missing for 26.2 % of churners but only 6.4 % of the others. Median filling scores 0.745;
adding the flags lifts it to 0.778; the age flag alone adds nothing (age is missing at random).

**Which models.** Linear models, k-NN, SVMs and neural networks need complete numbers. scikit-learn's
`HistGradientBoostingClassifier` (and XGBoost / LightGBM) handle gaps natively by learning which branch a missing
value should go down — they capture the "missing means something" signal by themselves (0.800 in Lab 6).

**When it hurts.** Filling shrinks the column's spread and creates a spike at the median (Lab 6 plot); a flag for a
column missing at random just adds a useless column. Fancier imputation (k-NN, model-based) costs more and rarely
beats median + flag for prediction.

**Leakage risk.** The fill value is learned — compute the median on training rows only (in a pipeline).

---

## 9. Step 7 · Outliers (cap, winsorise, transform)

**What it is.** An **outlier** is a value far from the rest. Two common detection rules:

* **IQR rule**: below Q1 − 1.5·IQR or above Q3 + 1.5·IQR.
* **z-score rule**: more than 3 standard deviations from the mean.

They often disagree (Lab 7: 108 vs 14 rows flagged) — with a long tail, "extreme" is a judgement call. Ways to
handle them:

* **Cap / winsorise**: values beyond a limit (e.g. the 1st / 99th percentile) are set *to* the limit.
* **Transform**: a log squashes big values (Step 8).
* **Remove** rows — only if they are a different kind of row (test accounts, bots).

**First ask why a value is extreme.** A typo → fix it or set it missing (Step 5). A different kind of row → maybe
remove. A genuine rare case → keep it, or transform instead of capping.

**Why it helps.** Linear models fit a line through squared errors, so a few far-away points pull the line towards
themselves. In Lab 7 the automated accounts (40–80 calls) flatten the fitted curve; capping makes the weight for
support calls four times larger (0.053 → 0.214), though the score barely moves (0.745 → 0.748) because only 15 of
3,000 rows are affected.

**Which models.** Linear models, k-NN, neural networks, PCA, and min-max scaling are sensitive. **Trees are not**:
a split "calls > 9" puts 12 and 73 on the same side (0.800 vs 0.803).

**When it hurts.** When the extremes **are** the signal: in a fraud toy where risk keeps rising with payment size,
capping at the 95th percentile drops ROC-AUC from 0.999 to 0.981.

**Leakage risk.** The limits are learned — compute them on training rows only.

---

## 10. Step 8 · Skewed numerics (log, Box-Cox, Yeo-Johnson)

**What it is.** Re-express a long-tailed column so its histogram looks more like a symmetric bell:

* **log** (or **log1p** = log(1 + x), which also works at 0): turns "times 10" into "plus 1".
* **Box-Cox**: a family of power transforms, with a parameter λ chosen to make the column as bell-shaped as
  possible. Needs values > 0.
* **Yeo-Johnson**: like Box-Cox but works for any value, including 0 and negatives.

**Why it helps.** Many real effects work in *ratios*, not differences: earning 2,000 more matters a lot at 1,000 and
little at 20,000. A log turns such effects into straight lines, which is exactly what a linear model can fit. It also
tames outliers. In Lab 8, log1p on four skewed columns lifts the linear model from 0.745 to 0.759.

**Which models.** Linear models, k-NN, neural networks. **Not trees**: a log keeps the order of values, and a tree
only cares about order — in Lab 8 the tree's score is identical to six decimals.

**When it hurts.** When the effect is already straight in the raw scale, or grows *faster* than a straight line: the
log bends it the wrong way (Lab 8 toy: 0.878 → 0.870). Log can also overshoot: `total_spend` goes from skew 1.24 to
−0.83. Check the score instead of transforming by reflex.

**Leakage risk.** log has nothing to learn; Box-Cox and Yeo-Johnson learn λ from data — fit them on training rows.

---

## 11. Step 9 · Scaling (and why trees don't need it)

**What it is.** Put columns on comparable ranges:

* **Standardisation** (`StandardScaler`): subtract the mean, divide by the std → mean 0, std 1.
* **Min-max** (`MinMaxScaler`): squeeze into [0, 1].
* **Robust** (`RobustScaler`): subtract the median, divide by the IQR — outliers barely affect it.

**Why it helps.** Income runs into thousands, survey answers from 1 to 5. A distance between two customers is then
almost entirely an income difference. In Lab 9, k-NN goes from 0.617 unscaled to 0.730 (standard) and 0.745
(robust).

**Which models.**

| Model | Needs scaling? | Why |
|---|---|---|
| k-NN, k-means, RBF SVM | **yes, strongly** | distances mix all columns |
| Logistic / linear regression with regularisation, neural networks | **yes** | the penalty and the optimiser treat all weights alike; unscaled data converges slowly (5 warnings in Lab 9) |
| PCA | **yes** (when units differ) | it chases variance, which depends on units |
| Decision trees, random forests, gradient boosting | **no** | a split "income > 4,100" selects the same customers whatever the units (0.798 for every scaler in Lab 9) |

**When it hurts.** Min-max with outliers: one huge value squashes everyone else into a sliver of [0, 1] (k-NN 0.713
min-max vs 0.730 standard). Scaling 0/1 columns is unnecessary (harmless for most models, but it makes them harder to
read).

**Leakage risk.** The mean/std (or median/IQR) are learned — fit the scaler on training rows only.

---

## 12. Step 10 · Categorical encoding

**What it is.** Turning categories into numbers.

* **One-hot**: one 0/1 column per category (`payment=paypal`, `payment=mailed check`, ...). Safe, no false order.
* **Ordinal**: map ordered categories to 0, 1, 2 (`basic < standard < premium`). Only when there really is an order.
* **Frequency**: each category → its share of the training rows. One column, no labels used.
* **Target encoding**: each category → the average target (churn rate) of that category, **shrunk** towards the
  overall rate for small categories and **cross-fitted** (each row's value computed from the *other* folds).

**Why it helps.** Models need numbers. One-hot is ideal for a few levels; for many levels (119 cities) it creates
many columns, most describing a handful of rows — in Lab 10 one-hot city scores 0.746, *below* leaving city out
(0.753). Target and frequency encoding compress a city into one informative column (0.762 and 0.765).

**Which models.** Linear models, k-NN and neural networks need one-hot or carefully encoded numbers — an arbitrary
code like `city = 57` would be read as "bigger than city 12". Trees can split arbitrary codes into groups, but still
benefit from target encoding with many levels.

**When it hurts.** Ordinal codes for unordered categories invent a false order. One-hot with many rare levels adds
noise (Lab 10). Target encoding done **naively** is a disaster (below).

**Leakage risk — the big one.** Naive target encoding computes each city's churn rate from *all* training rows, so
each row's feature contains its own label. In Lab 10 it scores 0.800 — and on a column of **pure noise** (800 random
"cities") it scores 0.822 where the honest answer is 0.498. Use cross-fitting (scikit-learn's `TargetEncoder` does it
in `fit_transform`) and keep the encoder inside the CV pipeline.

---

## 13. Step 11 · Rare-category grouping

**What it is.** Merge categories that appear in fewer than, say, 1 % of training rows into one "other" category.

**Why it helps.** A category seen 3 times can only be memorised; grouping removes those columns and prepares the
model for categories never seen before (a new city maps to "other" instead of crashing). In Lab 11, grouping cities
under 1 % goes from 119 columns (0.746) to 16 columns (0.768).

**Which models.** Most useful with one-hot encoding (linear, k-NN, neural networks). Helps tree models and target
encoding a little (fewer, more reliable groups).

**When it hurts.** Grouping too aggressively lumps high-churn and low-churn categories together: at a 10 % threshold
only 2 cities keep their own column (0.766). The curve is flat over a wide middle range — pick a threshold in it.

**Leakage risk.** Low (only counts are learned, no labels), but count on training rows only.

---

## 14. Step 12 · Date / time features

**What it is.** Extracting usable numbers from dates:

* **Days since** an event, measured from a fixed reference day (the data's export date): tenure, days since last
  login.
* **Parts** of a date: month, day of week, hour, is-weekend, is-holiday.
* **Cyclical encoding** of parts that wrap around: month → (sin(2π·month/12), cos(2π·month/12)). This puts the months
  on a circle, so December sits next to January.

**Why it helps.** A date as text is useless; a date as one big number hides the patterns. In Lab 12, tenure and days
since last login lift the score from 0.745 to 0.771. Churn is high for December/January sign-ups (promotions): as a
number 1..12 the month adds nothing (December and January are 11 apart), one-hot works (0.797, 12 columns), and
sin/cos does best with 2 columns (0.803).

**Which models.** Linear models, k-NN and neural networks need the circle trick. Trees can cut "month ≥ 11 or ≤ 2"
from the plain number with two splits.

**When it hurts.** Date parts with no real effect add useless columns (day of week in Lab 12). Measuring from
`today()` makes features drift every day you re-run.

**Leakage risk.** Use only dates known at prediction time (not the cancellation date!). With time-ordered data,
split by time (Step 1).

---

## 15. Step 13 · Domain features (ratios, differences, aggregates)

**What it is.** Features built from knowledge of the problem:

* **Ratios**: price / income (can they afford it?), spend / tenure (spend per month).
* **Differences**: days between sign-up and last login.
* **Aggregates** (group statistics): a customer's income relative to the median income of their city.
* **Text flags**: does the support note mention "billing", "crash", "thanks"?

**Why it helps.** It hands the model a quantity it would need many steps (or many splits) to build itself. In Lab 13,
five keyword flags from a free-text note lift the score from 0.822 to 0.842 (trees 0.837 → 0.852). The price/income
ratio clearly drives churn (churn share 0.74 in its top decile), but adds only 0.005 because the model already has
both ingredients.

**Which models.** All. Trees gain most from ratios and differences (they would need a staircase of splits to
approximate one); linear models from anything that turns a curved effect straight.

**When it hurts.** Features without a real effect just add columns (the aggregate and the difference in Lab 13).
Unthinking "feature soup" — every pairwise ratio — can even look great by accident: in Lab 13, 56 ratios score 0.850
because some of them happen to bend the age effect into the real U shape; once the true shape (age²) is added in Lab
14, the soup *lowers* the score (0.879 → 0.874).

**Leakage risk.** Group statistics are learned — compute them on training rows. A group statistic of the **target**
is target encoding and needs cross-fitting (Step 10).

---

## 16. Step 14 · Interactions and polynomials

**What it is.**

* **Polynomial terms**: `age²` (centred: `(age − mean)²`) lets a straight-line model draw a **U**.
* **Interactions**: `premium × calls` lets a model say "calls matter *more* for premium customers" — an effect that
  *depends* on another column.

**Why it helps.** A linear model adds up one effect per column; it can't draw U shapes or "it depends". In Lab 14, the
centred square lifts the score from 0.845 to 0.872, the interaction to 0.854, both to 0.879.

**Which models.** Linear models (and to a lesser degree k-NN and small neural networks). **Trees build interactions
and U shapes themselves** by splitting on one column and then another (Lab 14: 0.854 → 0.855).

**When it hurts.** Adding *all* products blindly explodes the column count: 28 columns → 435, and the score drops to
0.830 because the model fits noise. Add interactions you have a reason for, then check them.

**Leakage risk.** None for fixed formulas; centring for `x²` uses a mean learned on training rows.

---

## 17. Step 15 · Binning

**What it is.** Cut a numeric column into ranges (bins) and give each range its own 0/1 column. Bins can have equal
widths or equal numbers of rows (quantile bins).

**Why it helps.** Each range gets its own weight, so a linear model can follow any shape without you knowing the
formula. In Lab 15, age in 10 bins scores 0.874 against 0.854 for age as a straight line.

**Which models.** Linear models. **Not trees**: a tree *is* a set of learned bins, and fixed bins remove its freedom
to choose the edges (Lab 15: 0.855 → 0.844).

**When it hurts.** Every value inside a bin is treated the same, so detail is lost: for a smooth straight-line
effect, 3 bins drop a toy from 0.871 to 0.856. If you know the shape, a formula wins (age² 0.879 beats 10 bins).
Many bins on few rows give jumpy, noisy steps.

**Leakage risk.** Quantile edges are learned — on training rows only.

---

## 18. Step 16 · Correlated / redundant features (correlation, VIF, PCA)

**What it is.** Find columns that repeat each other and keep one, or compress a group.

* **Correlation** finds pairs (weekly vs monthly logins: +1.00).
* **VIF** (variance inflation factor) = 1 / (1 − R²), where R² measures how well *all the other* columns predict this
  one. VIF 1 = independent; above 10 = worrying; it also catches a column that is a mix of several others
  (`total_spend` ≈ price × tenure, VIF 7.9).
* **PCA** compresses a correlated group into a few uncorrelated columns (Step 18).

**Why it helps.** With near-copies, a linear model can shift credit freely between them: in Lab 16 the weight of
`weekly_logins` over 200 resamples has a spread of 0.345 and even changes sign; with the copy dropped, the spread is
0.069. Stable weights mean trustworthy explanations.

**Which models.** Linear models (unstable weights). Trees split importance between copies, which makes
feature-importance charts misleading.

**When it hurts.** It rarely changes the score (0.879 with or without the copy in Lab 16) — redundancy is a
problem of **stability and interpretation**, not usually of prediction. Moderately correlated columns can each carry
their own information: don't drop the survey questions just because they correlate at 0.4–0.5.

**Leakage risk.** Low (no labels used), but decide on training rows.

---

## 19. Step 17 · Feature selection (filter, wrapper, embedded)

**What it is.** Keep only the columns that help.

* **Filter**: score each column on its own against the target (correlation, mutual information, ANOVA F) and keep
  the best k. Fast, but blind to teamwork between columns.
* **Wrapper**: search over subsets by training the model repeatedly — e.g. **RFE** (recursive feature elimination):
  fit, drop the weakest, repeat. Accurate, slow.
* **Embedded**: the model selects while it trains — **L1 (lasso)** regression pushes useless weights to exactly 0;
  tree importances rank columns.

**Why it helps.** Fewer columns are faster, easier to explain, and remove noise columns. In Lab 17 the lasso keeps
24 of 35 columns (dropping four of the five pure-noise columns) at the same score (0.879); RFE keeps 20 at 0.876.

**Which models.** k-NN and unregularised linear models suffer most from useless columns; regularised models and
trees cope better by themselves.

**When it hurts.** Filters miss columns that only help in combination: in Lab 17 keeping the 5 "best" columns drops
the score to 0.732. Selecting too hard always costs signal.

**Leakage risk — the classic.** Choosing columns using **all** rows, then cross-validating, is one of the most common
mistakes in machine learning. In Lab 17, on 2,000 columns of pure noise with random labels, it produces ROC-AUC 0.760
for a problem with nothing to find; done inside each fold, the honest answer is 0.450 (a coin flip, give or take
chance). Selection must sit **inside** the cross-validation pipeline.

**Checking what the model uses (Lab 19).** After building and selecting, check which columns the fitted model relies
on. **Permutation importance** shuffles one column on held-out rows and measures how much the score drops; about 0
means the column doesn't help. **SHAP values** split each prediction into one contribution per column: they show
*how* the model uses a column (a dependence plot recovers the U shape in age) and explain single customers, but they
also give credit to noise the model happens to split on (10–17 % of the top column's mean |SHAP| for the pure-noise
columns, against about 0 by permutation). Three traps: a leak towers over every real column (a useful alarm);
near-copies hide (permutation) or halve (SHAP) each other's importance; and the built-in "impurity" importance of tree
libraries favours columns with many distinct values (pure noise ranked #13 of 37, above a real effect at #32).

---

## 20. Step 18 · Dimensionality reduction (PCA; LDA when labels exist)

**What it is.** Replace many columns by a few new ones that summarise them.

* **PCA** (principal component analysis) finds the directions in which the data varies most and keeps the top few. It
  ignores the target. Full explanation in [`../PCA/README.md`](../PCA/README.md).
* **LDA** (linear discriminant analysis) finds the direction that best **separates the classes**. It uses the target,
  so it can only be used when labels exist, and gives at most (number of classes − 1) components: **one** for a
  binary target.

**Why it helps.** A group of columns measuring one hidden thing can be summarised by one column. In Lab 18 the 8
survey questions (all measuring satisfaction) are replaced by their first principal component (53 % of their
variance) at no loss: 0.879 with 8 columns, 0.879 with 1. On a scikit-learn toy (breast cancer, 30 columns), one LDA
column (0.993) beats one PCA column (0.971) because LDA aims at the label.

**Which models.** Distance-based models and linear models with many correlated columns benefit. **Trees usually lose**:
every PCA column mixes all originals, and trees split best on the originals (Lab 18: 0.855 → 0.849).

**When it hurts.** When the useful signal lives in a low-variance direction (PCA drops it), when columns are not
correlated (nothing to compress), and when you need to explain the model column by column.

**Leakage risk.** PCA learns directions; LDA also learns from labels — both must be fitted on training rows only.
An LDA fitted on all rows has the same problem as Step 17's selection.

---

## 21. Step 19 · Validate every step by cross-validation

**What it is.** Treat every step as a hypothesis — *"this makes the model better"* — and test it: compare the
cross-validated score **with** and **without** the step, with every step re-fitted inside every fold. Switching
steps off one at a time is called an **ablation**.

**Why it matters.** Sensible-sounding steps often do nothing, and some hurt. In the lab's ablation (Lab 20):

| Stage switched off | Change in CV ROC-AUC |
|---|---|
| make_date_features | −0.023 |
| encode_categoricals | −0.020 |
| impute_missing (with flags) | −0.014 |
| add_interactions, scale_numeric, group_rare_categories | −0.005 to −0.006 |
| most others | within ±0.003 (noise) |
| remove_leakage | **+0.089** (a leak always looks good) |

**How to read such a table.**

* A difference smaller than the spread across folds (about ±0.01 here) is noise.
* "No change" doesn't make a step useless: a later step may be covering for it, and some steps are protection (IDs,
  typos) rather than score.
* Comparing many variants on the same folds slowly fits your choices to them. Score the test set **once**, at the
  end: in the lab, CV 0.895 and test 0.893 agree.
* Accuracy and ROC-AUC usually move together on a balanced target; ROC-AUC is the steadier of the two.

**Leakage risk.** The whole point: if any step is fitted outside the folds, the comparison is meaningless.

---

## 22. Cheat sheet: which steps matter for which model

| Step | Linear / logistic | k-NN, SVM (RBF) | Trees / boosting | Neural network |
|---|---|---|---|---|
| Split first, drop useless, remove leakage, fix types | ✔ | ✔ | ✔ | ✔ |
| Missing values | impute (+ flag) | impute (+ flag) | often native | impute (+ flag) |
| Outliers | ✔ cap / transform | ✔ | ✘ not needed | ✔ |
| Skew (log, Yeo-Johnson) | ✔ | ✔ | ✘ no effect | ✔ |
| Scaling | ✔ (with regularisation) | ✔✔ essential | ✘ no effect | ✔✔ essential |
| One-hot / target encoding | ✔ (never arbitrary codes) | ✔ | target encoding for many levels | ✔ (or embeddings) |
| Rare-category grouping | ✔ | ✔ | small help | ✔ |
| Date features (cyclical) | ✔ sin/cos | ✔ sin/cos | days-since ✔, plain month OK | ✔ sin/cos |
| Domain features | ✔ | ✔ | ✔ (ratios especially) | ✔ |
| Interactions / polynomials | ✔✔ | some | ✘ builds them itself | some |
| Binning | ✔ | some | ✘ hurts | rarely |
| Drop correlated | ✔ (stability) | some | importance charts only | rarely |
| Feature selection | ✔ | ✔✔ | some | some |
| PCA / LDA | some | ✔ | ✘ usually hurts | some |
| CV validation of every step | ✔ | ✔ | ✔ | ✔ |

(✔✔ big effect · ✔ helps · some = depends · ✘ not needed or harmful.) Lab 20 shows the overall effect per model:
with all steps, logistic regression climbs from 0.727 to 0.895 and overtakes the tree models (0.868–0.871), which
gain less because they already did some of this work themselves.

---

## 23. Running the whole pipeline (scripts + config.yml)

The notebook teaches each step in isolation. The folder [`scripts/`](scripts/README.md) runs **all** of them as one
pipeline, one Python file per step, with a builder-style [`scripts/config.yml`](scripts/config.yml) to switch steps on
and off and set their parameters. Full details (what each stage does, every parameter, how to add a stage) are in
[`scripts/README.md`](scripts/README.md). The short version, on a Mac with Apple silicon (M4), from the repository root:

```bash
~/.pyenv/versions/3.12.8/bin/python -m venv ~/venvs/feature-eng
```

```bash
source ~/venvs/feature-eng/bin/activate
```

```bash
pip install -r src/Feature-Eng/lab/requirements.txt
```

```bash
python src/Feature-Eng/scripts/run_pipeline.py --config src/Feature-Eng/scripts/config.yml
```

(Any Python 3.10+ works; the first command just picks a known-good interpreter. Keep the virtual environment outside
the repository.) The run prints every stage with what it learned, then the 5-fold CV score on the training rows and
the test score. It takes about 4 seconds.

**To compare:** set `enabled: false` on a stage in `config.yml` (or pass `--disable impute_missing,scale_numeric`)
and run again; `--ablation` runs once per stage switched off and prints the table from section 21. Required stages
(`load_data`, `split_data`, `train_and_evaluate`) refuse to be disabled and the runner says why.

---

## 24. Common misconceptions

| Myth | Reality |
|---|---|
| "More features = better model." | Useless and blind features add noise: 435 polynomial columns score 0.830 vs 0.879 for two chosen ones (Lab 14). |
| "Always scale your data." | Trees don't care (identical scores, Lab 9). Scale for distance-based, regularised linear models and neural networks. |
| "Always log-transform skewed columns." | Only if the effect is log-shaped; it can hurt (Lab 8 toy) or overshoot (`total_spend`). Trees are unaffected. |
| "Missing values should just be filled." | Often the *fact* that a value is missing is the signal — add a flag (Lab 6: 0.745 → 0.778). |
| "Outliers should be removed." | First ask why: typos → fix; bots → maybe remove; genuine extremes can be the signal (Lab 7 fraud toy). |
| "Target encoding is the best encoding." | Only cross-fitted. Naive target encoding makes pure noise look predictive (0.822, Lab 10). |
| "Select features first, then cross-validate." | That is leakage: pure noise scores 0.760 (Lab 17). Select inside each fold. |
| "Correlated features hurt accuracy." | Usually they hurt *stability and interpretation*, not the score (Lab 16). |
| "A higher CV score always means a better step." | Not if the step leaks (`remove_leakage` off: +0.089) or the gain is within the fold spread (±0.01). |
| "PCA always helps." | It can tie (survey block) or hurt (trees, Lab 18); see the PCA notes. |
| "Feature engineering is unnecessary with gradient boosting." | Boosting needs less of it, but still gains from fixing types, encoding, dates and domain features (0.771 → 0.871, Lab 20). |
| "Do all preprocessing on the full dataset, then split." | Split first; fit every step on training rows only (sections 2.6, 3). |

---

## 25. The labs

Everything above is demonstrated in one Jupyter notebook: **[`lab/fe_lab.ipynb`](lab/fe_lab.ipynb)**, on one messy
customer-churn table ([`data/customer_churn.csv`](data/customer_churn.csv), generated by
[`data/make_customer_churn.py`](data/make_customer_churn.py)) plus a scikit-learn toy dataset. Setup, run instructions
and a map of which lab answers which question are in **[`lab/README.md`](lab/README.md)**.

| Part | Labs | README sections |
|---|---|---|
| A. Hygiene before anything else | 0 meet the data · 1 split · 2 column types · 3 useless columns · 4 leakage · 5 data types | 3–7 |
| B. Cleaning values | 6 missing values · 7 outliers · 8 skew · 9 scaling | 8–11 |
| C. Categories and dates | 10 encoding · 11 rare categories · 12 dates | 12–14 |
| D. Creating features | 13 domain features · 14 interactions · 15 binning | 15–17 |
| E. Fewer, better features | 16 correlated columns · 17 feature selection · 18 PCA / LDA · 19 feature importance (permutation, SHAP) | 18–20 |
| F. Putting it together | 20 the pipeline scripts, ablation, model comparison | 21–23 |

---

## 26. Glossary

| Term | Meaning |
|---|---|
| **Feature / column** | One measured property of each row (age, income, ...). |
| **Target / label** | The column to predict (`churned`). |
| **Balanced binary target** | Two classes, about equally common. |
| **Train / validation / test** | Rows used to fit / to compare choices / to score once at the end. |
| **Cross-validation (CV)** | Rotate which part of the training rows is held out, average the scores. |
| **Stratified** | Each part keeps the same class shares as the whole. |
| **Fit / transform** | Learn something from data / apply what was learned. |
| **Pipeline** | A chain of steps + model that is fitted as one, so CV re-fits every step per fold. |
| **Leakage** | Information from scored rows or from the future reaching training. |
| **Imputation** | Filling missing values. |
| **Missing-value flag** | A 0/1 column saying "this value was missing". |
| **Outlier** | A value far from the others. |
| **Winsorising / capping** | Setting values beyond a limit to the limit. |
| **IQR** | Q3 − Q1, the width of the middle half of the values. |
| **Skewness** | Lopsidedness of a distribution; 0 = symmetric. |
| **log1p** | log(1 + x); a log that works at 0. |
| **Box-Cox / Yeo-Johnson** | Power transforms with a learned parameter λ that make a column more bell-shaped. |
| **Standardisation** | Subtract mean, divide by std. |
| **One-hot encoding** | One 0/1 column per category. |
| **Ordinal encoding** | Ordered categories → 0, 1, 2, ... |
| **Target encoding** | Category → (shrunk, cross-fitted) average target of that category. |
| **Cross-fitting** | Computing each row's learned value from the other folds only. |
| **Cardinality** | Number of distinct categories in a column. |
| **Cyclical encoding** | sin / cos of a wrapping quantity (month, hour), so its ends meet. |
| **Interaction** | A product of two columns, for effects that depend on each other. |
| **Binning** | Cutting a number into ranges. |
| **Correlation** | Straight-line link between two columns, −1 to +1. |
| **VIF** | 1 / (1 − R²): how well the other columns predict this one. |
| **Mutual information** | How much knowing a column reduces uncertainty about the target (any shape of link). |
| **RFE** | Recursive feature elimination: drop the weakest column, refit, repeat. |
| **L1 / lasso** | A penalty that pushes useless weights to exactly 0. |
| **PCA / LDA** | Unsupervised / supervised linear compression into fewer columns. |
| **ROC-AUC** | Chance that a random positive is ranked above a random negative; 0.5 = coin flip. |
| **Permutation importance** | Drop in a held-out score when one column's values are shuffled. |
| **SHAP value** | One column's share of one prediction (a Shapley value); base value + all shares = the model's output. |
| **Impurity importance** | A tree model's built-in importance: how much a column's splits purify the training rows; biased towards many-valued columns. |
| **Ablation** | Removing one step at a time to measure what it is worth. |

---

## 27. Further reading

* Max Kuhn and Kjell Johnson, *Feature Engineering and Selection: A Practical Approach for Predictive Models* — the
  standard book on exactly these steps (free online edition).
* Alice Zheng and Amanda Casari, *Feature Engineering for Machine Learning* (O'Reilly) — short and practical.
* scikit-learn user guide: *Preprocessing data*, *Imputation of missing values*, *Encoding categorical features*
  (incl. `TargetEncoder` and cross-fitting), *Feature selection*, *Pipelines and composite estimators*, and
  *Common pitfalls and recommended practices* (leakage).
* Ambroise and McLachlan (2002), *Selection bias in gene extraction on the basis of microarray gene-expression data*
  (PNAS) — the classic paper on the selection leak shown in Lab 17.
* Kaufman, Rosset, Perlich (2012), *Leakage in data mining: formulation, detection, and avoidance* (ACM TKDD).
* In this repository: [`../PCA/README.md`](../PCA/README.md) (PCA in depth), and the older notebooks
  [`../feature-engineering.ipynb`](../feature-engineering.ipynb) (more encodings and imputation methods, e.g.
  probability-ratio encoding, random-sample and end-of-distribution imputation) and
  [`../handling-outliers.ipynb`](../handling-outliers.ipynb) (Chebyshev's inequality as an outlier bound).

---

## Appendix - Assumptions made

No further input was available, so the following decisions were made on the author's behalf:

1. **Audience.** "Complete beginner" was taken to mean school algebra and basic Python, no statistics. Median,
   quantiles, skewness, correlation, ROC-AUC, cross-validation and leakage are introduced in section 2 before use.
2. **Scope.** Generic feature engineering for tabular data with a balanced binary target. Imbalanced targets,
   regression targets, time-series forecasting, text beyond keyword flags, images and automated feature tools are
   out of scope. The step "put it all in a Pipeline" is realised as the stage scripts in `scripts/` (section 23)
   instead of a lab; the labs use scikit-learn `Pipeline`s wherever fitting inside CV matters.
3. **One dataset.** A synthetic customer-churn table (3,000 rows, 35 columns) was generated so that every step has
   something real to find *and* a known ground truth (listed in Lab 0). It is saved as `data/customer_churn.csv`
   (about 0.55 MB) together with the script that makes it (fixed seed). The only other dataset is scikit-learn's
   bundled breast-cancer table (Lab 18). Nothing needs the internet.
4. **Main score.** 5-fold stratified cross-validated ROC-AUC on the 2,400 training rows; the 600 test rows are scored
   only in Lab 20 / by the pipeline runner. Accuracy is reported by the runner too.
5. **Honest results over neat ones.** Where a step did little or nothing on this data (domain ratio, capping, PCA of
   the survey block, several pipeline stages in the ablation), the README and the labs say so and explain why, rather
   than tuning the data until every step "wins". Small toy datasets are used only where the main table cannot show a
   "when it hurts" case clearly (fraud, accelerating effect, smooth effect, pure noise).
6. **Notebook vs scripts.** The notebook writes each step inline with pandas / scikit-learn so a beginner sees every
   line; the scripts implement the same steps as plain fit/transform functions for the configurable pipeline. The
   two are deliberately independent, so a few rules (e.g. fixing types) appear in both.
7. **Pipeline defaults.** Every optional stage is `enabled: true` by default, as requested, even those that change
   little on this data (the ablation shows which). `load_data`, `split_data` and `train_and_evaluate` are marked
   `required: true`. Default model: logistic regression.
8. **Environment used to execute and verify the notebook and scripts:** Python 3.12.8, NumPy 2.5, pandas 3.1,
   SciPy 1.18, scikit-learn 1.9.1, Matplotlib 3.11, PyYAML. `requirements.txt` asks for scikit-learn ≥ 1.9 because the
   code uses its current APIs (`TargetEncoder` with a CV splitter, `LogisticRegression(l1_ratio=...)`,
   `KBinsDiscretizer(quantile_method=...)`). Exact decimals may differ slightly with other versions.
9. **Notebook is committed with its outputs** (about 0.8 MB, low-resolution figures) so it can be read on GitHub.
10. **Existing files untouched.** The older `src/feature-engineering.ipynb` and `src/handling-outliers.ipynb` were
    read for ideas only and are linked from section 27; they were not edited or moved.
11. **Nothing was committed.** All files are left in the working tree for review.
