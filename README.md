# HR Analytics: predicting employee attrition

<p align="center">
  <a href="#-results">Results</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="#-what-the-data-says-about-who-leaves">What it found</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="#-auditing-the-model-a-second-notebook">Fairness audit</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="#-quickstart">Quickstart</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="#-honest-limitations">Limitations</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="#-where-the-data-comes-from">Data</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"></a>
  <a href="code/"><img src="https://img.shields.io/badge/notebooks-3%2C%20executed-eb6834?style=for-the-badge&logo=jupyter&logoColor=white" alt="Notebooks: 3, executed"></a>
  <a href="data/SOURCE.md"><img src="https://img.shields.io/badge/data-IBM%20sample%2C%20fictional-1baf7a?style=for-the-badge" alt="Data: IBM sample, fictional"></a>
  <a href="https://satsawat.ai"><img src="https://img.shields.io/badge/author-satsawat.ai-e8a112?style=for-the-badge" alt="Author: satsawat.ai"></a>
</p>

Companion code for the writing at [satsawat.ai](https://satsawat.ai).

Companies would rather keep a good employee than hire a new one, so they want to know who
is about to quit. In data terms that event is attrition. This repository shows what that
question looks like as a data exercise, start to finish, on IBM's invented HR sample of
1470 employees. Nobody in the file is real. What you get is one complete "predict who
leaves" workflow, from how well ten models guess to where they fall short. You will see why
the best model's 89% accuracy is a hollow number, and how a do-nothing guess already scores
84%. It also does the part most attrition demos skip: it turns the model score into a ranked
list of the 30 riskiest people, the thing an HR team can actually work down.

### Ten terms, in plain words

- Attrition means an employee leaving. In this file it is one column, `Attrition`, that
  says Yes or No for each person.
- A model is a program that learns from past rows and guesses Yes or No for a row it has
  not seen. This repository tries ten models and calls them by their algorithm names.
- A notebook is a file that mixes code cells with the results those cells printed. All
  three notebooks here are saved with their results, so you can read every number below
  without running anything.
- The model learns from 1,176 employees, called the training rows. It is then scored on
  294 employees it never saw, called the test set, 47 of whom left. Every number in the
  results, decile and audit tables is measured on those 294 people.
- The baseline is the guess that nobody leaves. 247 of the 294 stayed, so that guess is
  right 0.8401 of the time without looking at any data. A model earns its keep only by how
  far it climbs above that line.
- Every score comes from four counts of right and wrong guesses: stayers correctly left
  alone (TN), stayers wrongly flagged (FP), leavers missed (FN), and leavers caught (TP).
  The four together are called a confusion matrix.
- Precision is the share of people the model flagged who really left. Recall is the share
  of real leavers the model flagged. F1 is one number that balances the two, and it is the
  score this repository uses to pick a winner.
- A model gives each person a score between 0 and 1. The cut-off is the score above which
  the model says Yes. 0.5 is the default, and nobody here chose it.
- A decile is a tenth of the test set. Sort the 294 people by the model's score, cut them
  into ten groups of about 30, and count the leavers in the top group.
- A 95% interval is the range the true rate plausibly sits in, given how few people were
  counted. When the interval around a gap includes zero, the data cannot tell that gap
  from noise.

Three notebooks, each saved with its results:

- [`code/HRM_Employee Attrition.ipynb`](code/HRM_Employee%20Attrition.ipynb) is the
  analysis. It loads the CSV, cleans it, trains ten models to predict who leaves, and
  turns the best model's scores into a ranked call list. It is the whole workflow, from
  the first `describe()`, pandas' one-line summary of every column, to the decile table.
  Published May 2019. Repaired in 2026, after
  a library change had stopped it running.
- [`code/fairness_audit.ipynb`](code/fairness_audit.ipynb) audits the winning model. It
  asks who the model reaches and who it misses, group by group, and what that would cost a
  business. Every group rate in it is worked out by hand from the four counts, so you can
  run the same audit on your own model.
- [`code/statistical_rigour.ipynb`](code/statistical_rigour.ipynb) asks whether 294 test
  employees are enough to rank ten models at all. Mostly they are not.

You can open each notebook in the browser through Google Colab, which is the only way to
see the interactive charts (they are drawn with plotly, and GitHub renders them blank):
[main](https://colab.research.google.com/github/netsatsawat/HR-Analytics/blob/master/code/HRM_Employee%20Attrition.ipynb)
· [fairness audit](https://colab.research.google.com/github/netsatsawat/HR-Analytics/blob/master/code/fairness_audit.ipynb)
· [statistical rigour](https://colab.research.google.com/github/netsatsawat/HR-Analytics/blob/master/code/statistical_rigour.ipynb).
Colab loads only the notebook file. The CSV in `data/`, the helper module
`code/myUtilityFunction.py` and `requirements.txt` are not there, so the main notebook
fails at its import cell, and the other two at their first data cell, until you clone the
repository inside Colab and run `pip install -r requirements.txt` there. Nothing here
checks that the Colab links work. Only the local steps in the quickstart have been run
and checked.

![Emergency exit](img/emergency-exit.jpg)

## ⚡ Quickstart

```bash
git clone https://github.com/netsatsawat/HR-Analytics.git
cd HR-Analytics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook "code/HRM_Employee Attrition.ipynb"
```

The filename contains a space, so the quotes are load bearing. The notebook reads `../data/` and
imports the helper module from its own folder, so open it from `code/` as the command
above does. Two system packages sit outside pip on macOS:

```bash
brew install libomp     # xgboost, one of the ten models, cannot load without this
brew install graphviz   # only for the cell that draws the decision tree as a picture
```

The smallest useful check needs nothing installed beyond Python itself:

```bash
python3 scripts/verify_readme_claims.py
```

That script recomputes every number in the results, decile and audit tables below, plus
most of the exploration table, from the CSV and the saved notebook results, then fails if
any of them disagrees with the README. It prints one line per check and ends like this:

```
  ok  notebook image img/xgboost.png exists

every quoted README number matches its artifact
```

That script is the reason you can read the results table without taking my word for any
of it. Two groups of numbers sit outside its reach. The 2019 numbers in the "what moved"
section come from the original 2019 run. That run's results were overwritten in 2026, so
the old numbers now live only in git history, and the script does not check them. Nor
does it check the statistical rigour section, whose numbers are quoted from that
notebook's saved results.

To run the whole thing without opening a browser and produce the numbers for yourself:

```bash
cd code
jupyter nbconvert --to notebook --execute "HRM_Employee Attrition.ipynb" \
  --output /tmp/hr_run.ipynb --ExecutePreprocessor.timeout=1800
jupyter nbconvert --to notebook --execute fairness_audit.ipynb \
  --output /tmp/fairness_run.ipynb --ExecutePreprocessor.timeout=1800
```

The timeout matters. One cell tries 800 XGBoost settings and scores each one five ways,
which is 4000 model fits. The nbconvert default timeout would kill that cell partway
through. The weekly workflow in
[`.github/workflows/scheduled-execute.yml`](.github/workflows/scheduled-execute.yml)
runs the same command into a temporary folder, and that is the form the automated checks
on GitHub (CI) have actually run. Every random choice in the notebook is fixed by the seed
1234 (`random_state=1234` in the code). In my runs the four counts and the metrics come
back identical for all ten models each time, which is what makes the table below checkable
rather than decorative. Nothing in the repository re-runs the notebook and compares the
result against the saved one, so this is my observation. The audit notebook retrains only
the winning model, so it finishes quickly, and before doing anything else it checks that
the four counts match the main notebook's.

Versions are pinned in [`requirements.txt`](requirements.txt), meaning the exact version
of every package is written down, and were resolved on Python 3.11.15.

## 📦 Layout

```
code/HRM_Employee Attrition.ipynb   the analysis, 97 cells, 54 of them code, results saved
code/fairness_audit.ipynb           the fairness and cost audit, 59 cells, 32 of them code
code/statistical_rigour.ipynb       eight tests of whether 294 rows can rank ten models, results saved
code/myUtilityFunction.py           evaluation, plotting and decile helpers
data/                               the CSV and a note on where it came from
img/                                figures used by the notebooks and this README
scripts/verify_readme_claims.py     recomputes the results, exploration, decile and audit numbers
requirements.txt                    pinned versions the saved results came from
```

## 📊 Results

All ten models are scored on the same 294 held-out employees, 47 of whom left. The saved
notebook prints every number in the table directly, with two exceptions. One is the
`ROC-AUC (probs)` column, a 0.5-to-1.0 score for how well a model ranks people by risk,
worked out from a curve each model's cell already draws and explained in full below the
table. The other is the `predict nobody leaves` row, which is worked out from the test
split itself. Both are recomputed by the verifier. The `features` column, where a row
reads all 59, VIF 47 or corr 55, names the set of input columns that model saw, and those
sets are explained just below the table.

| model | features | accuracy | precision | recall | F1 | AUC (labels) | ROC-AUC (probs) |
|---|---|---|---|---|---|---|---|
| predict nobody leaves | n/a | 0.8401 | n/a | 0.0000 | 0.0000 | 0.5000 | n/a |
| Decision tree | all 59 | 0.8401 | 0.5000 | 0.1064 | 0.1754 | 0.5431 | 0.7639 |
| Random forest | all 59 | 0.8401 | n/a | 0.0000 | 0.0000 | 0.5000 | 0.8524 |
| XGBoost | all 59 | 0.8776 | 0.7619 | 0.3404 | 0.4706 | 0.6601 | 0.8170 |
| XGBoost | VIF 47 | 0.8810 | 0.7727 | 0.3617 | 0.4928 | 0.6707 | 0.8337 |
| XGBoost | corr 55 | 0.8741 | 0.7778 | 0.2979 | 0.4308 | 0.6408 | 0.8198 |
| XGBoost, randomized search | all 59 | 0.8503 | 0.5455 | 0.3830 | 0.4500 | 0.6611 | 0.8159 |
| Logistic regression | all 59 | 0.8776 | 0.7037 | 0.4043 | 0.5135 | 0.6859 | 0.8570 |
| **Logistic regression, grid search** | **all 59** | **0.8946** | **0.7105** | **0.5745** | **0.6353** | **0.7650** | **0.8620** |
| Logistic regression, grid search | VIF 47 | 0.8776 | 0.6667 | 0.4681 | 0.5500 | 0.7118 | 0.8462 |
| Logistic regression, grid search | corr 55 | 0.8946 | 0.7222 | 0.5532 | 0.6265 | 0.7564 | 0.8565 |

The `features` column names which input columns the model saw: all 59, or one of two
pruned sets, VIF 47 and corr 55, that drop columns which are near copies of other
columns. Grid search and randomized search are two ways of trying many settings and
keeping the best. Both the pruning and the searches are explained under
[how it works](#-how-it-works).

Read accuracy against 0.8401, not against zero. 247 of the 294 test employees stayed, so a
model that predicts "nobody leaves" scores 0.8401 without looking at the data. The
decision tree and the random forest land on exactly that line. Only the grid-searched
logistic regressions clear it by a margin worth anything.

The column labelled `AUC (labels)` is misnamed. It is not the risk-ranking score you would
expect. The notebook prints `roc_auc_score(y_test, y_pred)` on the predicted labels, meaning
the hard Yes or No answers rather than the scores. Run on labels, that formula is not ROC-AUC
at all. It is balanced accuracy wearing a borrowed name: the average of two recalls, the
recall on leavers and the recall on stayers. That was true in 2019 and I left the computation
alone.

The real risk-ranking score is the last column, `ROC-AUC (probs)`. ROC-AUC measures how well
a model orders people by risk, from 0.5 for a coin toss to 1.0 for a perfect ordering, with no
cut-off involved. It is the area under the curve each cell already plots.

The gap between those two columns for the random forest is the clearest lesson in the table.
It ranks employees respectably (0.8524) yet predicts zero leavers: none of its scores ever
climbs above the 0.5 cut-off, so it never actually says Yes. Two of its settings, explained
below, cause that. Its precision is `0/0`. It flagged nobody, so there is nothing to divide.

The best model is a weighted sum of the 59 columns: a plain logistic regression at `C=100`
with `l1_ratio=1.0`, which is pure l1. Read those two settings as dials for how hard the model
trims useless columns, detailed under [how it works](#-how-it-works). They were picked by trying every combination on the training
rows and keeping the one with the best F1, which is what a grid search does. It flags 38 of the 294
employees, 27 of whom really left, and misses 20. Its four counts, reading TN, FP, FN,
TP: **236, 11, 20, 27**. A simple model, fed the columns as they come, and it beats the
tuned XGBoost by 19 points of F1: 0.6353 against 0.4500, where a point is one hundredth.

### Ranking beats classifying

Using the 0.5 cut-off is the least interesting thing you can do with a score. Sorted by
predicted probability and split into deciles:

| decile | score range | employees | leavers | hit rate | share of all leavers caught |
|---|---|---|---|---|---|
| 1st | 0.59 to 0.97 | 30 | 23 | 0.7667 | 0.4894 |
| top 2 | above 0.29 | 59 | 31 | 0.5254 | 0.6596 |
| top 3 | above 0.15 | 88 | 37 | 0.4205 | 0.7872 |

Call the top thirty names on the 294-person list and 23 of them really left, a hit rate of
0.7667, so roughly three in four are genuine flight risks. Those 23 are 48.9% of everyone
who left. That hit rate is the number an HR partner can plan a week around, and it is a
different conversation from "the model is 89% accurate". The rigour notebook puts a 95%
interval on that hit rate, from 0.5956 to 0.8891, so read it as roughly three in four
rather than as a four-decimal fact.

The two charts below draw the same idea. Cumulative gain shows how many leavers you catch
as you work down the ranked list. Lift shows how much better that is than picking people
at random.

![Cumulative gain](img/Cum_gain.png)

![Lift chart](img/Lift.png)

Further down the notebook, a score band table pushes the same idea harder: of the 8 test
employees scored above 0.80, all 8 left. Eight people is an anecdote, not evidence, and
the band table sits in the notebook as an illustration of the workflow rather than as a
result.

## 🔍 What the data says about who leaves

The notebook's exploratory section asks four questions and answers them with plots. The
rates below are the same relationships in numbers, computed over all 1470 rows, so you
can check them against the CSV in one line of pandas.

| factor | attrition rate | compared with |
|---|---|---|
| Works overtime | 0.3053 (n=416) | 0.1044 for everyone else (n=1054) |
| Travels frequently | 0.2491 (n=277) | 0.1496 for rare travellers (n=1043), 0.0800 for non-travellers (n=150) |
| Single | 0.2553 (n=470) | 0.1248 married (n=673), 0.1009 divorced (n=327) |
| Worst work life balance | 0.3125 (n=80) | 0.1422 at the most common level (n=893) |
| Lowest job satisfaction | 0.2284 (n=289) | 0.1133 at the highest level (n=459) |
| Lives more than 10km away | 0.2095 (n=444) | 0.1404 for everyone closer (n=1026) |

Median monthly income is 3202 for leavers against 5204 for stayers, and attrition falls
steadily with education, from 0.1824 below college to 0.1042 among the 48 employees at
the highest level. Overtime is the strongest single signal in the exploration. And the
fitted model agrees. The rigour notebook shuffles each column in turn and measures how
much the model's ordering of people gets worse, and `OverTime_Yes` ranks second of the 59
features by that test. The exploration and the model agree, but not independently,
because both are reading the same 1470 rows.

Everything in this section describes a dataset IBM invented. See
[the limitations](#-honest-limitations) before carrying any of it into a meeting.

## ⚖ Auditing the model: a second notebook

The one finding that survives is about age. Under 40, the model catches 24 of the 35
leavers, a catch rate of 0.6857. At 40 and over it catches 3 of the 12, a catch rate of
0.2500. The gap is 0.4357. Given the small counts, the true gap could plausibly sit
anywhere from 0.1085 to 0.6419. Zero is not in that range, so the data is enough to call
the gap real.

[`code/fairness_audit.ipynb`](code/fairness_audit.ipynb) asks the question the analysis
above never does. The winning model reads `Gender_Male`, two `MaritalStatus` columns, two
`Department` columns and two `Generation` columns, and `Generation` is just age split into
three groups. Seven of its 59 features describe who somebody is rather than what they do.
The main notebook fits that model, ranks employees by it, and never asks what running it
would mean for the people in the file.

The audit is built from scratch, without the two usual fairness libraries (`fairlearn` and
`aif360`). Its arithmetic is the teaching content. Every group rate in the audit table is
four integers and a division. The four come from the confusion matrix sliced by group. A
protected attribute is a personal trait, such as age, gender or marital status, that a
model should not be allowed to base decisions on. The notebook covers group sizes and base
rates (the share of each group that actually left), selection rate and the demographic
parity ratio, the four-fifths rule, equal opportunity, equalised odds, calibration by
group, a confidence interval on every rate and on every gap, and a ten-line test of
whether deleting a protected column removes the attribute. Each of those terms is
explained inside the notebook where its number appears, and the ones this README quotes
are explained here as they come up. The second half turns the four counts into money: a
cost model whose three parameters the reader sets, a sweep over the cut-off (trying every
cut-off from 0.01 to 0.99 in turn), and the fairness gap priced in the same units.

It is written for someone who has trained a Yes or No model before and has never checked
it for fairness. No legal background is needed, and the one rule with a citation is quoted
where it is used. The notebook holds 59 cells, 32 of them code, saved with their results,
and its charts are plotly and interactive to match the main notebook.

What follows is the same model's four counts, sliced by group. `flagged` is how many
people in the group the model said would leave. `tn`, `fp`, `fn` and `tp` are the four
counts for that group. `selection rate` is flagged divided by n. `TPR` (true positive
rate) is tp divided by leavers: the share of that group's real leavers the model caught,
which this section calls the catch rate. Every rate quoted in this section is derived
from these counts and recomputed by the verifier.

| dimension | group | n | leavers | flagged | tn | fp | fn | tp | selection rate | TPR |
|---|---|---|---|---|---|---|---|---|---|---|
| Gender | Female | 109 | 14 | 10 | 92 | 3 | 7 | 7 | 0.0917 | 0.5000 |
| Gender | Male | 185 | 33 | 28 | 144 | 8 | 13 | 20 | 0.1514 | 0.6061 |
| MaritalStatus | Divorced | 60 | 8 | 4 | 51 | 1 | 5 | 3 | 0.0667 | 0.3750 |
| MaritalStatus | Married | 144 | 18 | 16 | 121 | 5 | 7 | 11 | 0.1111 | 0.6111 |
| MaritalStatus | Single | 90 | 21 | 18 | 64 | 5 | 8 | 13 | 0.2000 | 0.6190 |
| AgeBand | 40 and over | 120 | 12 | 8 | 103 | 5 | 9 | 3 | 0.0667 | 0.2500 |
| AgeBand | Under 40 | 174 | 35 | 30 | 133 | 6 | 11 | 24 | 0.1724 | 0.6857 |
| Department | Human Resources | 11 | 1 | 1 | 10 | 0 | 0 | 1 | 0.0909 | 1.0000 |
| Department | Research & Development | 185 | 24 | 24 | 153 | 8 | 8 | 16 | 0.1297 | 0.6667 |
| Department | Sales | 98 | 22 | 13 | 73 | 3 | 12 | 10 | 0.1327 | 0.4545 |

The age band is cut at 40 because that is the boundary the US Age Discrimination in
Employment Act draws, and because it is the finest cut this test set supports: split four
ways instead and the oldest band holds 2 leavers. Department is audited alongside the
three protected attributes as a contrast, not as a protected class of its own.

The four-fifths rule is a US hiring screen. Divide the lowest selection rate in a
dimension by the highest, and a ratio under 0.8 calls for a closer look. The model fails
the four-fifths screen on every dimension, at 0.6062 for gender, 0.3333 for marital
status, 0.3867 for the age band and 0.6853 for department. Most of that gap is there
because the groups leave at different rates to begin with. In this file, single employees
and employees under 40 leave at roughly twice the rate of their comparison groups, so a
model that flagged both sides equally would be wrong about one of them. A failed
four-fifths screen only tells you to look closer.

Only one of the three gaps holds up. Equal opportunity is the question here: does the
model have the same catch rate in each group? For age, the gap is the 0.4357 quoted at
the top of this section. The gender gap is 0.1061 with an interval of -0.1811 to 0.3808,
and the marital status gap is 0.2440 with an interval of -0.1382 to 0.5387. Both intervals
contain zero, so the data cannot tell those gaps from noise. Two of the three differences
the rate table shows are therefore differences this data cannot support, and the notebook
says so rather than reporting all three. Each single rate carries a Wilson interval and
each gap a Newcombe interval. Those are the two standard formulas for a 95% range on a
rate and on a difference of two rates when the counts are small.

Deleting a protected column does not delete the attribute. Another column can reveal it,
and such a column is called a proxy. The notebook tests for proxies by fitting the
remaining features to the attribute itself and scoring the fit by ROC-AUC, so 1.0 means
the attribute can be read back perfectly and 0.5 means not at all. Marital status comes
back at AUC 0.9505. Every Single employee in this dataset carries `StockOptionLevel` 0, so
a benefits field reads marital status back almost perfectly. The age band
comes back at 0.9339. Drop both `Generation` columns and it still comes back at 0.6908,
because almost everything an HR system records accumulates with time. Gender comes back
at 0.4196, worse than chance. Gender is the exception. Deleting its column really does
delete the information. Three attributes, three different answers, which is why the
notebook tests the claim instead of asserting it.

The 0.5 cut-off is a business decision that nobody made. It is scikit-learn's default,
and nobody here chose otherwise. The notebook defines `cost_of_replacing` (what one
leaver costs to replace), `cost_of_a_retention_conversation` (what one flagged person
costs to talk to) and `conversation_success_rate` (the share of real leavers who stay
after the talk) as parameters the reader sets. It quotes no industry figure for any of
them, because there is no source here for one. Only the ratio of replacement cost to
conversation cost matters. Assume 30% of retention talks work. At the default cut-off the
model flags 38 people, 27 real leavers plus 11 false alarms. About 0.30 x 27 = 8.1 of
those talks work. So the model pays for itself once one replacement is worth more than
38 / 8.1 conversations, a break-even of 4.6914. That break-even is `(27 + 11) / (0.30 x 27)`,
and no other quantity enters it. Change the cut-off and the economics change. At 20
conversations per replacement and that 30% success rate, the cheapest
cut-off is 0.23, which catches 37 leavers for 34 false alarms, against 27 and 11 at the
default 0.5. The notebook then repeats that search across replacement costs of 5 to 80
conversations and success rates of 10% to 50%, and the cheapest cut-off moves anywhere
between 0.97 and 0.01. That cut-off is the largest lever in the deployment, and it is the
one least often discussed.

The fairness gap is also money the company is not spending. At the younger
band's catch rate, 8.23 of the 12 older leavers would have been reached instead of 3. The
8.23 is an expected count, 0.6857 x 12, not a head count. Those are conversations a
company is paying for and not having, with a segment that is older and longer-tenured
than the one it does reach.

The people in this file are fictional, so none of the above is a finding about any real
workforce. It is a demonstration that the method finds one real gap, sizes it, and
correctly declines to call the other two.

![Fairness audit summary](img/fairness_audit_summary.png)

## 📐 Can this test set tell the models apart?

[`code/statistical_rigour.ipynb`](code/statistical_rigour.ipynb) turns the same scepticism
on the results table above. That table ranks ten models. Its best row and its best
XGBoost, the one on the VIF 47 set, are separated by four employees out of 294: the first
gets 263 right and the second 259. The notebook asks whether that ranking is a result or a
reading of noise, and answers with eight tests. Every number in this section is quoted
from that notebook's saved results, and the verifier script does not recompute them.

The answer is not comfortable reading. McNemar's exact test asks whether two models
disagree about the same people more often than chance would allow. It separates no pair
of the ten models from any other. That holds across all 45 pairs. The Holm correction
raises the bar because 45 tests run at once. The smallest adjusted p-value is
0.3323, well above the usual 0.05 line, so every pair of models could easily be tied. A
p-value is the chance of seeing a gap this large if the two models were really equal. If
the four-employee gap between the best row and the best XGBoost were real, a test set
this size would spot it only 8.4% of the time (a power of 0.084), about one in twelve.
Every model's 95% accuracy interval is wider than the gap between the best and worst of
the ten. Re-split the data fifty different ways instead of once and seven of the ten
models change rank. One further test lands harder. After correction, not one of the ten
is distinguishable from predicting that nobody leaves.

What survives is ranking rather than labelling. The scores concentrate risk, with 23 of
the top 30 being leavers. They are also calibrated in aggregate: the average predicted
risk matches the share who really left. Sorting employees by predicted risk works. It is
the Yes or No call at the 0.5 cut-off that 47 leavers are too few to support.

![Statistical rigour summary](img/statistical_rigour_summary.png)

## 🧪 How it works

Cleaning comes first. 35 columns in. Three carry a single value for all 1470 rows
(`EmployeeCount`, `Over18`, `StandardHours`) and are dropped, as is `EmployeeNumber`,
which is an identifier. Seven columns coded as small integers are mapped back to their
labels so the plots read in words.

The split happens before any new feature is built. It is 80/20 with `random_state=1234`,
giving 1,176 training and 294 test rows, 190 and 47 leavers respectively. Feature
engineering means making new columns from old ones. It happens after the split, because
in production the engineered columns will not exist until the row arrives. The new
columns are a generation label from age, a first-job flag and a job-hop index from tenure
and employer count, and a compa-ratio, which is an employee's income divided by the median
income for the same department, role and level.

Encoding turns the text columns into numbers. One-hot encoding makes one 0 or 1 column
per value of each text column. `drop_first` then drops one value per column, because the
others imply it. The result is 59 features. Those 59 are then pruned two ways, and all
three sets are carried through the models: all 59, a VIF-pruned set of 47 at threshold 7,
and a correlation-pruned set of 55 at threshold 0.8, which drops `MonthlyIncome`,
`Department_Sales`, `JobRole_Sales Executive` and `Generation_Millenials`. Both prunings
remove columns that are near copies of other columns. VIF (variance inflation factor)
scores each column by how well the others predict it and drops the worst offender one at
a time. Correlation pruning drops any column whose correlation with another is above the
threshold. The two methods disagree, which is the point of running both.

![Correlation matrix](img/hr_correlation_matrix.png)

The models are three tree-based ones and one linear one. A decision tree is a flowchart of
yes or no questions, a random forest averages many trees, and XGBoost builds trees one
after another with each one correcting the last. Logistic regression is a weighted sum of
the features. Settings the modeller picks by hand, such as how many questions deep a tree
may grow, are called hyperparameters. The runs are: decision tree and random forest at
`max_depth=3` (three questions deep), XGBoost at `learning_rate=0.1, max_depth=3` with 100
trees on each of the three feature sets, a randomized search over 800 parameter
combinations by 5 folds scored on F1, then logistic regression plain and grid-searched
over `C`, `l1_ratio` and class weight on each feature set. A grid search tries every
combination of a few settings and keeps the best. A randomized search samples a much
larger space at random. Each try is scored by splitting the training rows five ways and
training five times, once with each fifth held out. That is the 5 folds, also called
cross-validation. Class weight decides whether a missed leaver counts for more than a
wrongly flagged stayer while the model trains, which pushes the model to say Yes more
often. Helper functions for evaluation and plotting live in
[`code/myUtilityFunction.py`](code/myUtilityFunction.py).

## 🔧 Ported from 2019: what moved and why

I published this in May 2019 and left it alone for seven years. Somewhere in that gap it
stopped running. The helper module, which the second cell imports, asks for
`sklearn.metrics.scorer`, which scikit-learn removed in 0.24, so every clone made after
2020 opened the notebook and got an `ImportError` before a single row of data loaded. Two
`np.bool` references and a dead plotting dependency were waiting behind it.

It runs again, on pandas 3, numpy 2, scikit-learn 1.9 and xgboost 3.2. The analysis is
the one from 2019: same split, same engineered features, same models, same hyperparameter
grids, same narrative. Where a number moved, it moved because a library changed, and this
section says exactly which ones and by how much rather than quietly restating the new
figures as if they were always there.

Four of the ten models reproduce their 2019 outputs exactly, to the individual cell of the
confusion matrix: decision tree, random forest, and the grid-searched logistic regressions
on the all-features and VIF sets. So does the whole pipeline underneath them. Same
1,176/294 split, same 986/190 and 247/47 class counts, same 59 encoded features, the same
47 columns surviving VIF in the same order, the same four columns dropped by correlation,
and a decile table matching to four decimals. The port did not disturb the analysis.

What did move, with the 2019 figure first:

| model | accuracy | F1 | why |
|---|---|---|---|
| XGBoost, all 59 | 0.8707 to 0.8776 | 0.4571 to 0.4706 | xgboost tree-building defaults changed between 0.8x and 3.2 |
| XGBoost, VIF 47 | 0.8776 to 0.8810 | 0.4857 to 0.4928 | same |
| XGBoost, corr 55 | 0.8776 to 0.8741 | 0.4375 to 0.4308 | same, and this one moved down |
| XGBoost, randomized search | 0.8639 to 0.8503 | 0.5000 to 0.4500 | the search picked a different winner, see below |
| Logistic regression, all 59 | 0.8810 to 0.8776 | 0.5570 to 0.5135 | identical hyperparameters, liblinear drift between sklearn 0.20 and 1.9 |
| Logistic regression grid, corr 55 | 0.7857 to 0.8946 | 0.5468 to 0.6265 | the grid flipped on a near tie, see below |

The tuned XGBoost changed winner. The randomized search now settles on 300 estimators at
`learning_rate=0.1, gamma=0.2, subsample=0.8, colsample_bytree=0.6` instead of 2019's 200
at 0.15, 0.3, 0.7 and 0.8. Estimators means trees. The other four are knobs on how each
tree is grown. `max_depth=3` and `min_child_weight=10` are the two settings both searches
agree on. The same seed samples the same 800 combinations from the same grid, so the
search should have drawn the 2019 combination again. Whether it did is not on record: the
notebook prints only the top five by cross-validated F1, and the 2019 winner is not among
them. What changed is the models underneath the search, not the search itself.

The correlation feature set flipped winner on a near tie. This is the ugly one. It is worth
stating plainly, not glossing over. In 2019 the winner was
`C=0.1, l2, class_weight='balanced'`. In 2026 it is `C=100, l1`, unweighted. `C` sets how
hard the model's weights are held back, and a bigger `C` holds back less. l1 and l2 are
two different ways of holding them back. As the grid now reads, l2 is `l1_ratio=0.0` and
l1 is `l1_ratio=1.0`. The two winners are separated by 0.0020 in mean cross-validated F1,
0.5023 against 0.5003, so the grid has no real winner on this feature set. The 2019
winner now ranks 2nd of the 24 candidates. Dropping the balanced class weight is what
swings recall from 0.8085 down to 0.5532 while accuracy climbs from 0.7857 to 0.8946.
Reported either way, the selection rests on a coin flip. The VIF grid tells a milder
version of the same story. It now picks `C=100` where 2019 picked `C=10`, and lands on
identical test metrics.

One judgement call sits in the code. The four logistic regression fits now pin
`solver='liblinear'`. A solver is the fitting routine. The liblinear solver handles both
l1 and l2. In 2019 they ran on scikit-learn's old default, recorded in the saved output as
`solver='warn'`, which was liblinear. In 0.22 the default became lbfgs. The lbfgs solver
cannot fit the l1 candidates the grid searches over. Each of them would raise inside
`GridSearchCV`, score `NaN`, and be dropped with a warning that `filterwarnings('ignore')`
swallows. The notebook would complete with no error while half the stated experiment had
silently stopped running, which is the same class of breakage as the import error rather
than a result changing. With the solver pinned, two of the three grid searches reproduce
2019 exactly.

I retired one deprecated parameter. The three grids search `l1_ratio` where they used to
search `penalty`. scikit-learn deprecated `penalty` on `LogisticRegression` in 1.8 and
removes it in 1.10 (the warning text is quoted in the scheduled workflow file), and
`filterwarnings('ignore')` at the top of the notebook would have swallowed the warning
until the removal simply broke the run. Same trap the missing import set seven years ago.
Under `solver='liblinear'` the two forms are the same model: `l1_ratio=1.0` is the old
`penalty='l1'`, `l1_ratio=0.0` is `penalty='l2'`. The grid still has 24 candidates, the
search still picks the same one on all three feature sets, and the fitted coefficients
come back identical to the last decimal, so every number in the results table above is
unchanged. This one is a rename with proof, not a result moving.

I removed one dependency. `scikitplot` has been dead since scipy 1.12 removed `scipy.interp`.
It drew two charts here, the cumulative gain and the lift curve, and both are now drawn
by plain matplotlib functions in `myUtilityFunction.py` with no new dependency added.

## 🚧 Honest limitations

The data is fictional. IBM's data scientists generated these 1470 employees for a Watson
Analytics demo. Nobody in this file resigned from anything. Every relationship in the
notebook is a property of IBM's generator, so no conclusion here is a finding about any
real workforce, including the ones about overtime and income that sound most like common
sense.

It is small, and the test set is smaller. 1470 rows, 237 leavers. The held-out set is 294
people containing 47 leavers, so one percentage point of accuracy is roughly three people
and one extra caught leaver moves recall by 0.02. Treat gaps of a few points in the
results table as noise. Several of them are.

It is imbalanced, and two models never noticed. Imbalanced means far fewer leavers than
stayers. Only 16% of these employees left, so predicting "nobody leaves" scores 0.8401.
The decision tree and random forest match that and no more. The random forest predicts
zero leavers outright. Both stay in the table because deleting them would make the
notebook look better than the experiment was.

One split, drawn once, not stratified. The data was split into training and test rows
once, at random, without forcing the same share of leavers into each side, which is what
stratified would mean. That one split was never repeated to see how much the numbers
wobble, and the main notebook puts no confidence interval on anything. Every test number
there is conditional on `random_state=1234`. The rigour notebook exists to measure what
that costs.

A feature leaks, mildly. A leak is test information reaching the training side. The
compa-ratio uses a median-income lookup built on all 1470 rows. The lookup is built after
the split, but from the whole dataset, so the test rows help set a number that is later
applied to the test rows. It touches no labels and the effect is small, but the notebook
claims to engineer features after splitting and this one line does not honour that.

Real attrition is a question of when someone leaves, not only whether. This is a static
snapshot: features as of one moment, a Yes or No label, no time dimension, no notion of
when someone leaves or of the people who have not left yet. An organisation that wants to
act needs to know when each person is likely to go, not only who ever left.

The score bands are ranks, not calibrated probabilities. Deciles hold up because they only
require the ordering to be right. Individual bands do not, since the top band holds 8
people.

## 📁 Where the data comes from

`data/WA_Fn-UseC_-HR-Employee-Attrition.csv` is IBM's sample dataset, mirrored on Kaggle,
added to this repository unmodified in May 2019. 1470 rows, 35 columns, 237 leavers, no
missing values, no duplicates.

[`data/SOURCE.md`](data/SOURCE.md) carries the full record of where the file came from:
the checksum, the row and column counts recomputed from the file itself, the three
constant columns, the class balance, a short script to verify all of it yourself, and the
licensing position (MIT covers the code in this repository, not IBM's CSV).

## 📄 License

MIT, see [`LICENSE`](LICENSE). The dataset is IBM's and carries its own terms. See
[`data/SOURCE.md`](data/SOURCE.md).

---

Written by [Satsawat Natakarnkitkul](https://satsawat.ai), a data and AI practitioner in
ASEAN. More writing at [satsawat.ai](https://satsawat.ai). Companion repositories:
[tsfm-bakeoff](https://github.com/netsatsawat/tsfm-bakeoff),
[markov_and_hidden_markov_model](https://github.com/netsatsawat/markov_and_hidden_markov_model),
[agent-failure-lab](https://github.com/netsatsawat/agent-failure-lab). Newsletter:
[AI in Practice](https://satsawat.ai/#newsletter).
