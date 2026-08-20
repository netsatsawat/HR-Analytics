#!/usr/bin/env python3
"""No number in the README without a runnable path behind it.

One assertion, wired into CI: every figure quoted in README.md is
recomputed here from the committed artifacts, and the README must
literally contain the recomputed string.

Recomputed from the artifacts, never restated:
  - the dataset shape, class balance, id range and sha256, read out of
    data/WA_Fn-UseC_-HR-Employee-Attrition.csv
  - the attrition rates in the exploration table, computed over that same
    CSV, since those numbers appear in no notebook output
  - the split and its class counts, which have to add back up to the CSV
  - accuracy, precision, recall, F1 and the notebook's "AUC (labels)" for
    all ten runs, recomputed from the confusion matrix each run plotted
    and cross-checked against the figures that run printed
  - the "ROC-AUC (probs)" column, recovered by integrating the
    probability ROC curve each cell already plotted
  - how many features each run was given, counted off its coefficient
    trace
  - the headline model, picked by recomputed F1 rather than named here
  - the decile table, whose hit rates and captured shares are recomputed
    from the raw Ones and Zeros counts in the same table
  - every figure the fairness audit contributes, rebuilt from the seven
    integer columns of the audit table that notebook prints: selection
    rates, parity ratios, true positive rates, the Wilson and Newcombe
    intervals (using statistics.NormalDist where the notebook uses scipy),
    the proxy AUCs integrated from the ROC curves they plotted, and the
    threshold sweep's cheapest operating point recomputed from the catches
    and false alarms it committed
  - the values the summary infographic was drawn from, checked against
    that same audit, so the PNG cannot drift away from the analysis

Structural checks guard what actually broke this repository. It sat for
seven years with an import no scikit-learn since 0.24 could satisfy, so
the notebook died on its first cell and nothing noticed. Execution counts
must run 1..N with no gaps, every code cell must carry one, and no cell
may hold an error output. A notebook that half ran, or that was re-run
out of order, fails here rather than shipping. Both notebooks are held to
that standard, and the audit's slices have to add back up to the main
notebook's held-out split, so the two cannot describe different models.

A recomputed number may be written three ways: four decimal places
(0.8946), the notebook's own printed form (0.765), or two decimals as a
percentage (89.46%). Counts may carry a thousands separator (1,470).
Nothing looser. Numbers are matched with digit boundaries either side, so
0.05 does not satisfy a claim of 0.0 and 1,470 does not satisfy 470.

Results-table and decile-table claims are checked as whole rows, on one
line. Accuracy alone does not identify a run: three of the ten score
0.8776 and two score 0.8401, so a row could lose its model and stay
green. Prose claims are checked as plain containment, because prose gets
rewrapped and a line break is not drift.

No network, no model, no fitting: it reads a committed notebook, a
committed CSV and two committed Markdown files. Standard library only, so
CI needs nothing installed to run it.
"""

import base64
import csv
import json
import hashlib
import re
import statistics
import struct
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "code" / "HRM_Employee Attrition.ipynb"
AUDIT = REPO / "code" / "fairness_audit.ipynb"
DATASET = REPO / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
SOURCE = REPO / "data" / "SOURCE.md"
README = REPO / "README.md"

# Taxonomy, not measurement: which evaluated cell is which row of the
# README's results table. Every number attached to these keys is
# recomputed. The roster is checked against the notebook below, so a run
# that is added or dropped fails rather than quietly changing the table.
RUNS = [
    ("DecisionTreeClassifier|all|plain", "Decision tree"),
    ("RandomForestClassifier|all|plain", "Random forest"),
    ("XGBClassifier|all|plain", "XGBoost, all features"),
    ("XGBClassifier|vif|plain", "XGBoost, VIF features"),
    ("XGBClassifier|corr|plain", "XGBoost, corr features"),
    ("XGBClassifier|all|search", "XGBoost, randomized search"),
    ("LogisticRegression|all|plain", "Logistic regression, plain"),
    ("LogisticRegression|all|search", "Logistic regression, grid, all"),
    ("LogisticRegression|vif|search", "Logistic regression, grid, VIF"),
    ("LogisticRegression|corr|search", "Logistic regression, grid, corr"),
]

# Its pixels carry the 2019 tuned-XGBoost panel: accuracy 0.8639, F1 0.5,
# precision 0.6061, recall 0.4255. The committed notebook now prints
# 0.8503 / 0.45 / 0.5455 / 0.383, so publishing that screenshot would put
# a contradiction on the front page. Nothing may link to it.
STALE_IMAGES = {"img/xgboost_perf_tuned.png"}

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
                11: "eleven", 12: "twelve"}

# plotly 6 writes numeric arrays as little-endian base64 with a dtype tag.
STRUCT_CODE = {"i1": "b", "i2": "h", "i4": "i", "i8": "q", "u1": "B",
               "u2": "H", "u4": "I", "u8": "Q", "f4": "f", "f8": "d"}

BEFORE = r"(?<![\d.,])"
AFTER = r"(?![\d])"

problems = []


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}" +
          (f" ({detail})" if detail and not condition else ""))
    if not condition:
        problems.append(name)


def decode(value):
    """A plotly array, whether it survived as a list or as packed bytes."""
    if not isinstance(value, dict) or "bdata" not in value:
        return value
    code = STRUCT_CODE[value["dtype"]]
    raw = base64.b64decode(value["bdata"])
    flat = list(struct.unpack(f"<{len(raw) // struct.calcsize(code)}{code}",
                              raw))
    shape = [int(d) for d in str(value.get("shape", "")).split(",")
             if d.strip()]
    if len(shape) == 2:
        return [flat[r * shape[1]:(r + 1) * shape[1]] for r in range(shape[0])]
    return flat


def renderings(value, printed=None):
    """The ways a document may honestly write one recomputed number."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, int):
        return sorted({str(value), f"{value:,}"})
    forms = {f"{value:.4f}", f"{value * 100:.2f}%"}
    if printed is not None:
        forms.add(printed)
    return sorted(forms)


def _prose_only(text):
    """`text` with the badge block and any hex colour removed.

    A bare digit check once matched the `8` inside the shields.io colour `e8a112`,
    so three prose mentions of an 8-person band could all be changed and the check
    stayed green. Badges are markup, not claims: they are not searched.
    """
    text = re.sub(r"<p align=\"center\">.*?</p>", "", text, flags=re.S)
    return re.sub(r"\b[0-9a-fA-F]{6}\b", "", text)


def quotes(text, value, printed=None):
    """True if `text` states `value`, with digit boundaries either side."""
    text = _prose_only(text)
    return any(re.search(BEFORE + re.escape(form) + AFTER, text)
               for form in renderings(value, printed))


def row(text, values):
    """True if one line of `text` carries every one of `values`."""
    return any(all(quotes(line, v) for v in values)
               for line in text.splitlines())


def ratio(text, first, second):
    """True if `text` writes the pair as `first/second`, commas optional."""
    left = "|".join(re.escape(f) for f in renderings(first))
    right = "|".join(re.escape(f) for f in renderings(second))
    return re.search(BEFORE + f"(?:{left})/(?:{right})" + AFTER,
                     text) is not None


def phrase(text, *words):
    """True if `text` runs `words` together, with any whitespace between.

    Prose wraps. A line break between "threshold" and "0.8" is not drift.
    """
    return re.search(r"\s+".join(words), text) is not None


def sequence(text, values):
    """True if `text` lists `values` in order, comma separated.

    Whitespace between them is free, so a comma-separated run of numbers
    still matches when the prose around it wraps onto the next line.
    """
    pattern = r",\s*".join(BEFORE + re.escape(f"{v}") + AFTER for v in values)
    return re.search(pattern, text) is not None


def absent(value, printed=None):
    return f"none of {renderings(value, printed)} in the README"


def streams(cell):
    return "".join("".join(o["text"]) for o in cell.get("outputs", [])
                   if o["output_type"] == "stream")


def text_plain(cell):
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if "text/plain" in data:
            return "".join(data["text/plain"])
    return ""


def plotly_figure(cell):
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if "application/vnd.plotly.v1+json" in data:
            return data["application/vnd.plotly.v1+json"]
    return None


def run_key(cell, figure):
    """`LogisticRegression|vif|search`, read off the cell and its chart."""
    estimator = figure["layout"]["title"]["text"].split("of ")[-1]
    source = "".join(cell["source"])
    features = ("vif" if "X_test_vif" in source else
                "corr" if "X_test_cor" in source else "all")
    tuned = "search" if re.search(r"\.best_(estimator|params)_", source) \
        else "plain"
    return f"{estimator}|{features}|{tuned}"


def trapezoid(xs, ys):
    """Area under a curve given as points, which is what ROC-AUC is."""
    return sum((xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2
               for i in range(len(xs) - 1))


def evaluated_runs(cells):
    """Every cell that printed a model evaluation, keyed and recomputed."""
    found = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        printed = streams(cell)
        figure = plotly_figure(cell)
        if "Accuracy Score:" not in printed or figure is None:
            continue
        matrix = roc = probability_auc = features = None
        for trace in figure["data"]:
            name = str(trace.get("name") or "")
            if name == "Confusion Matrix":
                matrix = decode(trace["z"])
            elif name.startswith("ROC: "):
                roc = float(name[5:])
                probability_auc = trapezoid(decode(trace["x"]),
                                            decode(trace["y"]))
            elif name == "coefficients":
                features = len(decode(trace["y"]))
        (tn, fp), (fn, tp) = matrix
        total = tn + fp + fn + tp
        found[run_key(cell, figure)] = {
            "cell": cell["execution_count"], "features": features,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "rows": total, "positives": fn + tp,
            "accuracy": (tp + tn) / total,
            "precision": tp / (tp + fp) if tp + fp else "n/a",
            "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "auc": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
            "probability_auc": probability_auc,
            "roc_trace": roc,
            "printed": dict(re.findall(
                r"(Accuracy Score|F1 Score|Area Under Curve):\s+([\d.]+)",
                printed)),
        }
    return found


def dataset_rows():
    with DATASET.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def rate(rows, predicate):
    """(count, attrition rate) for the employees matching `predicate`."""
    matching = [r for r in rows if predicate(r)]
    leavers = sum(1 for r in matching if r["Attrition"] == "Yes")
    return len(matching), leavers / len(matching)


def decile_table(cells):
    """Ones, Zeros and the three rates per decile, from the decile cell."""
    for cell in cells:
        if cell["cell_type"] == "code" and "decile_analysis" in \
                "".join(cell["source"]):
            table = {}
            for line in text_plain(cell).splitlines():
                match = re.match(r"^(\d+\.[\[(][^\]\)]*[\]\)])\s+(.*)$",
                                 line.strip())
                if match:
                    table.setdefault(match.group(1), []).extend(
                        float(n) for n in match.group(2).split())
            return [table[k] for k in sorted(table)]
    return []


def score_bands(cells):
    """(lower bound, stayers, leavers) per attrition-score band."""
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "prob_class_1_band" in source \
                and "crosstab" in source:
            return [(float(lo), int(a), int(b)) for lo, a, b in re.findall(
                r"^\d+\.\s+[\[(]([\d.]+) - [\d.]+\]\s+(\d+)\s+(\d+)\s*$",
                text_plain(cell), re.M)]
    return []


def index_size(text, label):
    """How many names a printed pandas Index holds, counted not trusted."""
    body = re.search(rf"{label}.*?Index\(\[(.*?)\],\s*\n?\s*dtype=", text,
                     re.S)
    return len(re.findall(r"'(?:[^'\\]|\\.)*'", body.group(1)))


def referenced_images(text):
    """Every img/ path a document points at, however it spells the link."""
    return sorted(set(re.findall(r"(?:\.\./|/master/|\(|\"|')(img/[^\s\"')]+)",
                                 text)))


def audit_rows(cells):
    """The fairness notebook's printed audit table, one dict per group.

    Seven integers per group, and every rate the audit quotes is derived from
    them here rather than read off the notebook. The table is anchored on its
    own heading so the representation table above it cannot be picked up by
    accident.
    """
    for cell in cells:
        printed = streams(cell)
        if "AUDIT TABLE" not in printed:
            continue
        body = printed.split("AUDIT TABLE", 1)[1]
        rows = []
        for line in body.splitlines():
            match = re.match(r"^\s*(\w+)\s+(\S.*?)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                             r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line)
            if match:
                dimension, group = match.group(1), match.group(2)
                n, leavers, flagged, tn, fp, fn, tp = (
                    int(match.group(i)) for i in range(3, 10))
                rows.append({"dimension": dimension, "group": group, "n": n,
                             "leavers": leavers, "flagged": flagged, "tn": tn,
                             "fp": fp, "fn": fn, "tp": tp,
                             "selection_rate": flagged / n,
                             "base_rate": leavers / n,
                             "tpr": tp / leavers if leavers else None,
                             "fpr": fp / (fp + tn) if fp + tn else None})
        return rows
    return []


def setting(source, name):
    """A named parameter, read out of the notebook's own source."""
    match = re.search(rf"^{name} = ([\d.]+)", source, re.M)
    return float(match.group(1)) if match else None


def normal_quantile(confidence):
    """The z the notebook takes from scipy, from the standard library instead."""
    return statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def wilson_interval(successes, trials, z):
    """Wilson score interval, recomputed rather than parsed."""
    p = successes / trials
    denominator = 1 + z ** 2 / trials
    centre = (p + z ** 2 / (2 * trials)) / denominator
    half = z * (p * (1 - p) / trials
                + z ** 2 / (4 * trials ** 2)) ** 0.5 / denominator
    return centre - half, centre + half


def newcombe_interval(k1, n1, k2, n2, z):
    """Interval for the difference of two proportions, from the two Wilsons."""
    p1, p2 = k1 / n1, k2 / n2
    low1, high1 = wilson_interval(k1, n1, z)
    low2, high2 = wilson_interval(k2, n2, z)
    return ((p1 - p2) - ((p1 - low1) ** 2 + (high2 - p2) ** 2) ** 0.5,
            (p1 - p2) + ((high1 - p1) ** 2 + (p2 - low2) ** 2) ** 0.5)


def parity(rows, dimension, column):
    """(ratio, worst group, best group) on one dimension of the audit table."""
    subset = [r for r in rows if r["dimension"] == dimension]
    worst = min(subset, key=lambda r: r[column])
    best = max(subset, key=lambda r: r[column])
    return worst[column] / best[column], worst, best


def named_traces(cells, title_fragment):
    """Every trace of the figure whose title contains `title_fragment`."""
    for cell in cells:
        figure = plotly_figure(cell)
        if figure is None:
            continue
        title = str(figure["layout"].get("title", {}).get("text", ""))
        if title_fragment in title:
            return {str(t.get("name") or ""): t for t in figure["data"]}
    return {}


def printed_dict(cells, marker):
    """The `key value` block a cell printed, as a dict of strings."""
    for cell in cells:
        printed = streams(cell)
        if marker not in printed:
            continue
        return dict(re.findall(r"^(\w+)\s{2,}(.+?)\s*$", printed, re.M))
    return {}


def audit_checks(readme, test, leavers, headline, sizes):
    """Everything the README quotes from the fairness notebook, recomputed.

    `sizes` is the (cells, code cells) shape of every other notebook in the
    repository, so the layout block's claims can be checked as a set rather
    than one at a time.
    """
    notebook = json.loads(AUDIT.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    # Newline joined, not bare concatenation: a cell's last line carries no
    # trailing newline, so gluing them would hide the first line of the next
    # cell from every line-anchored pattern below.
    source = "\n".join("".join(c["source"]) for c in code)

    print(f"\nnotebook structure, {AUDIT.relative_to(REPO)}:")
    check("audit notebook is nbformat 4", notebook["nbformat"] == 4,
          f"nbformat {notebook['nbformat']}")
    unrun = [i for i, c in enumerate(code) if c.get("execution_count") is None]
    check(f"all {len(code)} audit code cells carry an execution count",
          not unrun, f"cells at index {unrun} never ran")
    counts = [c.get("execution_count") for c in code]
    check(f"audit execution counts run 1 to {len(code)} with no gaps",
          counts == list(range(1, len(code) + 1)), f"got {counts}")
    errored = [c.get("execution_count") for c in code
               if any(o["output_type"] == "error"
                      for o in c.get("outputs", []))]
    check("no audit cell holds an error output", not errored,
          f"error output in cell(s) {errored}")
    tracebacks = [c.get("execution_count") for c in code
                  if "Traceback (most recent call last)" in streams(c)]
    check("no audit cell printed a traceback", not tracebacks,
          f"traceback in cell(s) {tracebacks}")
    # Both notebooks' sizes are written this way, and the audit's is written
    # twice. Bare containment would let one mention go stale while another kept
    # the check green, so every pair the README states has to name a notebook
    # that actually has that shape.
    stated = {(int(a), int(b)) for a, b in
              re.findall(r"(\d+) cells, (\d+) of them code",
                         _prose_only(readme))}
    check(f"every '<n> cells, <m> of them code' claim names a real notebook, "
          f"and the audit's {len(cells)}/{len(code)} is among them",
          (len(cells), len(code)) in stated and stated <= sizes | {
              (len(cells), len(code))},
          f"README states {sorted(stated)}, notebooks are "
          f"{sorted(sizes | {(len(cells), len(code))})}")

    rows = audit_rows(code)
    dimensions = re.search(r"DIMENSIONS = \[(.*?)\]", source, re.S)
    declared = re.findall(r"'([^']+)'", dimensions.group(1)) if dimensions \
        else []
    print(f"the audit table, {len(rows)} groups over {len(declared)} "
          f"dimensions:")
    check("the audit table covers exactly the dimensions the notebook declares",
          declared and sorted({r["dimension"] for r in rows}) ==
          sorted(declared),
          f"table has {sorted({r['dimension'] for r in rows})}, source "
          f"declares {sorted(declared)}")
    inconsistent = [r["group"] for r in rows
                    if r["tn"] + r["fp"] + r["fn"] + r["tp"] != r["n"]
                    or r["fn"] + r["tp"] != r["leavers"]
                    or r["fp"] + r["tp"] != r["flagged"]]
    check("every group's four confusion cells add back to its n, leavers and "
          "flagged", not inconsistent, f"broken for {inconsistent}")
    for dimension in declared:
        subset = [r for r in rows if r["dimension"] == dimension]
        check(f"{dimension} slices the same {test} test rows and {leavers} "
              f"leavers the main notebook held out",
              sum(r["n"] for r in subset) == test and
              sum(r["leavers"] for r in subset) == leavers,
              f"{sum(r['n'] for r in subset)} rows, "
              f"{sum(r['leavers'] for r in subset)} leavers")
    first = [r for r in rows if r["dimension"] == declared[0]]
    rebuilt = tuple(sum(r[cell] for r in first)
                    for cell in ("tn", "fp", "fn", "tp"))
    check(f"the audited model is the main notebook's headline model: "
          f"{list(rebuilt)}",
          rebuilt == (headline["tn"], headline["fp"], headline["fn"],
                      headline["tp"]),
          f"audit sums to {rebuilt}, headline is "
          f"{(headline['tn'], headline['fp'], headline['fn'], headline['tp'])}")

    for r in rows:
        # The two labels ride along with the numbers. Without them a row could
        # keep every figure and lose the group it belongs to, which is the same
        # failure the results table above guards against.
        check(f"README audit row for {r['dimension']} {r['group']}: n="
              f"{r['n']}, {r['leavers']} leavers, {r['flagged']} flagged, "
              f"cells {r['tn']}/{r['fp']}/{r['fn']}/{r['tp']}, selection rate "
              f"{r['selection_rate']:.4f}, TPR {r['tpr']:.4f}",
              row(readme, [r["dimension"], r["group"], r["n"], r["leavers"],
                           r["flagged"], r["tn"], r["fp"], r["fn"], r["tp"],
                           r["selection_rate"], r["tpr"]]),
              "no single line carries both labels and all nine numbers")

    four_fifths = setting(source, "FOUR_FIFTHS")
    print(f"demographic parity, against the {four_fifths:g} screen:")
    for dimension in declared:
        # Not named `ratio`: that is a module-level checker this function calls.
        parity_ratio, worst, best = parity(rows, dimension, "selection_rate")
        check(f"README states the {dimension} parity ratio {parity_ratio:.4f} "
              f"({worst['group']} over {best['group']})",
              quotes(readme, parity_ratio), absent(parity_ratio))
    ratios = {d: parity(rows, d, "selection_rate")[0] for d in declared}
    check(f"all {len(ratios)} dimensions fall below the {four_fifths:g} "
          f"screen, and the README says every one of them does",
          all(v < four_fifths for v in ratios.values()) and
          phrase(readme, "four-fifths", "screen", "on", "every", "dimension"),
          f"ratios {dict((k, round(v, 4)) for k, v in ratios.items())}")

    confidence = setting(source, "CONFIDENCE")
    z = normal_quantile(confidence)
    print(f"equal opportunity, with {confidence:.0%} intervals recomputed:")
    excluding = []
    for dimension in ("Gender", "MaritalStatus", "AgeBand"):
        subset = [r for r in rows if r["dimension"] == dimension]
        high = max(subset, key=lambda r: r["tpr"])
        low = min(subset, key=lambda r: r["tpr"])
        gap = high["tpr"] - low["tpr"]
        lower, upper = newcombe_interval(high["tp"], high["leavers"],
                                         low["tp"], low["leavers"], z)
        if lower > 0 or upper < 0:
            excluding.append(dimension)
        check(f"README states the {dimension} true positive rate gap "
              f"{gap:.4f}, interval {lower:.4f} to {upper:.4f}",
              all(quotes(readme, v) for v in (gap, lower, upper)),
              f"{absent(gap)}, {absent(lower)} or {absent(upper)}")
    check(f"exactly one gap excludes zero, and it is the age band: "
          f"{excluding}", excluding == ["AgeBand"],
          f"intervals exclude zero for {excluding}")

    band = [r for r in rows if r["dimension"] == "AgeBand"]
    age_cut = setting(source, "AGE_CUT")
    check(f"the age band is cut at {age_cut:g}, and the README names both "
          f"sides", age_cut and
          sorted(r["group"] for r in band) ==
          [f"{age_cut:g} and over", f"Under {age_cut:g}"] and
          all(f"{r['group']}" in readme for r in band),
          f"groups are {[r['group'] for r in band]}")
    finer = re.search(r"four bands, for comparison:(.*?)\n\n", streams(next(
        c for c in code if "four bands" in streams(c))), re.S)
    smallest = min(int(n) for n in re.findall(r"leavers=\s*(\d+)",
                                              finer.group(1)))
    check(f"the finest band the notebook prints holds {smallest} leavers, "
          f"and the README says why that is too few",
          quotes(readme, smallest) and phrase(readme, "oldest", "band", "holds",
                                              str(smallest), "leavers"),
          absent(smallest))

    print("the proxy probes, integrated from the ROC curves they plotted:")
    probes = named_traces(code, "Recovering a protected attribute")
    scored = {name: trapezoid(decode(t["x"]), decode(t["y"]))
              for name, t in probes.items() if "AUC" in name}
    check(f"the notebook plotted {len(scored)} proxy ROC curves",
          len(scored) >= 3, f"found {sorted(scored)}")
    for name, area in sorted(scored.items()):
        stated = float(re.search(r"AUC ([\d.]+)", name).group(1))
        check(f"'{name}': the curve integrates to {area:.4f}, and the README "
              f"quotes it",
              abs(area - stated) < 5e-5 and quotes(readme, area),
              f"curve gives {area:.4f}, label says {stated}"
              if abs(area - stated) >= 5e-5 else absent(area))

    print("the threshold sweep, recomputed from the curve it plotted:")
    sweep = named_traces(code, "Every threshold is available")
    thresholds = decode(sweep["catches"]["x"])
    catches = decode(sweep["catches"]["y"])
    alarms = decode(sweep["false alarms"]["y"])
    replace = setting(source, "cost_of_replacing")
    talk = setting(source, "cost_of_a_retention_conversation")
    success = setting(source, "conversation_success_rate")
    def curve(replace_cost, talk_cost, success_rate):
        """Expected cost at every swept threshold, and the cheapest one."""
        costs = [(c + a) * talk_cost
                 + ((leavers - c) + (1 - success_rate) * c) * replace_cost
                 for c, a in zip(catches, alarms)]
        return costs, costs.index(min(costs))

    costs, cheapest = curve(replace, talk, success)
    check("the expected-cost curve is the one those catches and false alarms "
          "imply, at the parameters the notebook sets",
          all(abs(x - y) < 1e-6
              for x, y in zip(costs, decode(sweep["expected cost"]["y"]))),
          "the plotted cost curve disagrees with the recomputed one")
    default = thresholds.index(0.5)
    check(f"at 0.5 the sweep reproduces the headline model's {headline['tp']} "
          f"catches and {headline['fp']} false alarms",
          catches[default] == headline["tp"] and
          alarms[default] == headline["fp"],
          f"sweep gives {catches[default]} and {alarms[default]}")
    check(f"README states the cheapest threshold {thresholds[cheapest]:g}, "
          f"with {catches[cheapest]} catches and {alarms[cheapest]} false "
          f"alarms, against {catches[default]} and {alarms[default]} at 0.5",
          row(readme, [f"{thresholds[cheapest]:g}", catches[cheapest],
                       alarms[cheapest]]) and
          row(readme, [catches[default], alarms[default]]),
          "no single line carries the cheapest operating point")
    # The two corners of the grid the notebook draws: cheap replacement with a
    # conversation that rarely works, against the opposite.
    spread = [thresholds[curve(5.0, 1.0, 0.10)[1]],
              thresholds[curve(80.0, 1.0, 0.50)[1]]]
    check(f"README states the range the cheapest threshold moves over, "
          f"{spread[0]:g} to {spread[1]:g}",
          row(readme, [f"{spread[0]:g}", f"{spread[1]:g}"]),
          f"computed {spread}")
    break_even = (headline["tp"] + headline["fp"]) / (success * headline["tp"])
    check(f"README states the break-even ratio {break_even:.4f}, which is "
          f"({headline['tp']} + {headline['fp']}) / ({success:g} x "
          f"{headline['tp']})", quotes(readme, break_even), absent(break_even))

    older = next(r for r in band if r["tpr"] == min(x["tpr"] for x in band))
    younger = next(r for r in band if r["tpr"] == max(x["tpr"] for x in band))
    reached = younger["tpr"] * older["leavers"]
    check(f"README states that {reached:.2f} of the {older['leavers']} older "
          f"leavers would be reached at the younger band's rate",
          quotes(readme, reached, printed=f"{reached:.2f}"),
          absent(reached, f"{reached:.2f}"))

    print("the summary image, against the values it was drawn from:")
    headline_values = printed_dict(code, "best_false_alarms")
    expected = {
        "test_n": test, "test_leavers": leavers, "tn": headline["tn"],
        "fp": headline["fp"], "fn": headline["fn"], "tp": headline["tp"],
        "tp_old": older["tp"], "leavers_old": older["leavers"],
        "tp_young": younger["tp"], "leavers_young": younger["leavers"],
        "tpr_old": older["tpr"], "tpr_young": younger["tpr"],
        "dp_gender": parity(rows, "Gender", "selection_rate")[0],
        "dp_marital": parity(rows, "MaritalStatus", "selection_rate")[0],
        "dp_age": parity(rows, "AgeBand", "selection_rate")[0],
        "best_threshold": thresholds[cheapest],
        "best_catches": catches[cheapest], "best_false_alarms": alarms[cheapest],
        "half_catches": catches[default], "half_false_alarms": alarms[default],
        "four_fifths": four_fifths,
    }
    drifted = [k for k, v in expected.items()
               if k not in headline_values
               or abs(float(headline_values[k]) - v) > 5e-9]
    check(f"the {len(expected)} values the infographic is drawn from match "
          f"the audit they claim to summarise", not drifted,
          f"drifted or missing: {drifted}")

    audit_markdown = "".join("".join(c["source"]) for c in cells
                             if c["cell_type"] == "markdown")
    for path in referenced_images(audit_markdown):
        check(f"audit notebook image {path} exists", (REPO / path).is_file())
    summary = referenced_images(readme + audit_markdown)
    check("the summary image the audit writes is linked from both the README "
          "and the notebook",
          any("fairness_audit_summary" in p for p in
              referenced_images(readme)) and
          any("fairness_audit_summary" in p for p in
              referenced_images(audit_markdown)),
          f"linked images: {summary}")


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    readme = README.read_text(encoding="utf-8")
    source_md = SOURCE.read_text(encoding="utf-8")

    print(f"notebook structure, {NOTEBOOK.relative_to(REPO)}:")
    check("notebook is nbformat 4", notebook["nbformat"] == 4,
          f"nbformat {notebook['nbformat']}")
    unrun = [i for i, c in enumerate(code) if c.get("execution_count") is None]
    check(f"all {len(code)} code cells carry an execution count", not unrun,
          f"cells at index {unrun} never ran")
    counts = [c.get("execution_count") for c in code]
    check(f"execution counts run 1 to {len(code)} with no gaps",
          counts == list(range(1, len(code) + 1)), f"got {counts}")
    errored = [c.get("execution_count") for c in code
               if any(o["output_type"] == "error"
                      for o in c.get("outputs", []))]
    check("no cell holds an error output", not errored,
          f"error output in cell(s) {errored}")
    tracebacks = [c.get("execution_count") for c in code
                  if "Traceback (most recent call last)" in streams(c)]
    check("no cell printed a traceback", not tracebacks,
          f"traceback in cell(s) {tracebacks}")
    check(f"README's layout block states {len(cells)} cells",
          f"{len(cells)} cells" in readme)

    rows = dataset_rows()
    n_rows, n_cols = len(rows), len(rows[0])
    raw = DATASET.read_bytes()
    leavers_all = sum(1 for r in rows if r["Attrition"] == "Yes")
    positive_rate = leavers_all / n_rows
    ids = [int(r["EmployeeNumber"]) for r in rows]
    ages = [int(r["Age"]) for r in rows]
    print(f"dataset, recomputed from {DATASET.relative_to(REPO)}:")
    check(f"SOURCE.md states {n_rows} rows", f"rows       {n_rows}"
          in source_md)
    check(f"SOURCE.md states {n_cols} columns", f"columns    {n_cols}"
          in source_md)
    check(f"SOURCE.md states {len(raw):,} bytes",
          f"size       {len(raw):,} bytes" in source_md)
    check("SOURCE.md sha256 matches the committed file",
          hashlib.sha256(raw).hexdigest() in source_md,
          hashlib.sha256(raw).hexdigest())
    check(f"SOURCE.md states {leavers_all} Yes, {n_rows - leavers_all} No "
          f"({positive_rate * 100:.1f}% positive)",
          f"{leavers_all} Yes, {n_rows - leavers_all} No  "
          f"({positive_rate * 100:.1f}% positive)" in source_md)
    check(f"SOURCE.md states the all-No model scores "
          f"{(1 - positive_rate) * 100:.1f}%",
          f"scores {(1 - positive_rate) * 100:.1f}%" in source_md)
    check(f"SOURCE.md states {len(set(ids))} unique ids running {min(ids)} "
          f"to {max(ids)}",
          f"{len(set(ids))} unique ids, running {min(ids)} to {max(ids)}"
          in source_md)
    check(f"SOURCE.md states Age {min(ages)} to {max(ages)}",
          f"Age               {min(ages)} to {max(ages)}" in source_md)
    for column in ("EmployeeCount", "Over18", "StandardHours"):
        values = sorted({r[column] for r in rows})
        check(f"{column} is constant at {values[0]}, and SOURCE.md says so",
              len(values) == 1 and re.search(
                  rf"{column}\s+always \"?{values[0]}\"?", source_md),
              f"column holds {values}")
    explorer = next(streams(c) for c in code
                    if "quick_df_explorer" in "".join(c["source"]))
    check(f"notebook read {n_rows} rows and {n_cols} columns",
          f"Number of observations: {n_rows}, Number of columns / "
          f"features: {n_cols}" in explorer)
    check(f"README states the {n_rows:,}-row, {n_cols}-column dataset and "
          f"its {leavers_all} leavers",
          all(quotes(readme, v) for v in (n_rows, n_cols, leavers_all)),
          absent(n_rows))

    print("the exploration table, recomputed from the same CSV:")
    modal_balance = Counter(r["WorkLifeBalance"] for r in rows).most_common(
        1)[0][0]
    top_satisfaction = max(r["JobSatisfaction"] for r in rows)
    factors = [
        ("works overtime", lambda r: r["OverTime"] == "Yes"),
        ("does not", lambda r: r["OverTime"] == "No"),
        ("travels frequently",
         lambda r: r["BusinessTravel"] == "Travel_Frequently"),
        ("does not travel", lambda r: r["BusinessTravel"] == "Non-Travel"),
        ("single", lambda r: r["MaritalStatus"] == "Single"),
        ("married", lambda r: r["MaritalStatus"] == "Married"),
        ("divorced", lambda r: r["MaritalStatus"] == "Divorced"),
        ("worst work life balance", lambda r: r["WorkLifeBalance"] == "1"),
        (f"work life balance at the most common level {modal_balance}",
         lambda r: r["WorkLifeBalance"] == modal_balance),
        ("lowest job satisfaction", lambda r: r["JobSatisfaction"] == "1"),
        (f"job satisfaction at the highest level {top_satisfaction}",
         lambda r: r["JobSatisfaction"] == top_satisfaction),
        ("more than 10km from home",
         lambda r: int(r["DistanceFromHome"]) > 10),
        ("within 10km", lambda r: int(r["DistanceFromHome"]) <= 10),
    ]
    for label, predicate in factors:
        n, attrition = rate(rows, predicate)
        check(f"README states {attrition:.4f} attrition among the {n} who "
              f"{label}", row(readme, [attrition, n]),
              f"{absent(attrition)}, or not on the same line as {n}")
    for label, level in (("below college", "1"),
                         ("the highest level", str(max(r["Education"]
                                                       for r in rows)))):
        n, attrition = rate(rows, lambda r, lv=level: r["Education"] == lv)
        check(f"README states {attrition:.4f} attrition at education "
              f"{label}", quotes(readme, attrition), absent(attrition))
    incomes = {a: statistics.median(int(r["MonthlyIncome"]) for r in rows
                                    if r["Attrition"] == a)
               for a in ("Yes", "No")}
    check(f"README states median monthly income {incomes['Yes']:,.0f} for "
          f"leavers against {incomes['No']:,.0f} for stayers",
          all(quotes(readme, int(v)) for v in incomes.values()),
          absent(int(incomes["Yes"])))

    print("the split, recomputed from the notebook:")
    split = next(streams(c) for c in code
                 if "Training data set shape" in streams(c))
    train, test = (int(n) for n in re.findall(r"shape: \((\d+),", split))
    check(f"the {train}-row train and {test}-row test split accounts for "
          f"every one of the {n_rows} CSV rows", train + test == n_rows,
          f"{train} + {test}")
    distribution = streams(next(c for c in code
                                if "Distribution of testing" in streams(c)))
    train_side, test_side = distribution.split("Distribution of testing")
    train_stay, train_leave = (int(n) for n in
                               re.findall(r"^[01]\s+(\d+)$", train_side, re.M))
    stayers, leavers = (int(n) for n in
                        re.findall(r"^[01]\s+(\d+)$", test_side, re.M))
    check(f"the two class counts add up to the {train} and {test} rows",
          train_stay + train_leave == train and stayers + leavers == test,
          f"{train_stay}+{train_leave}, {stayers}+{leavers}")
    baseline = stayers / test
    check(f"README states the {train:,}/{test} split, {train_stay}/"
          f"{train_leave} and {stayers}/{leavers}",
          all(ratio(readme, a, b) for a, b in ((train, test),
                                               (train_stay, train_leave),
                                               (stayers, leavers))))
    check(f"README states the {test}-row test set and its {leavers} leavers",
          quotes(readme, test) and quotes(readme, leavers))

    print("feature sets, counted off the notebook's printed indexes:")
    vif_cell = next(c for c in code if "calculate_vif_" in "".join(c["source"])
                    and "Remaining variables" in streams(c))
    corr_cell = next(c for c in code if "get_remain_columns_using_corr"
                     in "".join(c["source"]))
    n_vif = index_size(streams(vif_cell), "Remaining variables")
    n_corr = index_size(streams(corr_cell), "Remaining columns")
    dropped = re.search(r"There are (\d+) columns to remove: \[(.*?)\]",
                        streams(corr_cell))
    dropped_names = re.findall(r"'([^']+)'", dropped.group(2))
    check(f"correlation drops {len(dropped_names)} columns, all named in the "
          f"README", int(dropped.group(1)) == len(dropped_names) and
          all(f"`{c}`" in readme for c in dropped_names),
          f"missing {[c for c in dropped_names if f'`{c}`' not in readme]}")
    thresholds = []
    for cell in (vif_cell, corr_cell):
        raw = re.search(r"thresh=([\d.]+)", "".join(cell["source"])).group(1)
        thresholds.append(f"{float(raw):g}")
    check(f"README states the VIF threshold {thresholds[0]} and the "
          f"correlation threshold {thresholds[1]}",
          all(phrase(readme, "threshold", re.escape(t) + AFTER)
              for t in thresholds))

    runs = evaluated_runs(cells)
    for name, size in (("vif", n_vif), ("corr", n_corr)):
        used = {r["features"] for k, r in runs.items() if f"|{name}|" in k}
        check(f"the {size} columns surviving {name} are the {size} the "
              f"models were given", used == {size}, f"models used {used}")
    print(f"the {len(runs)} model runs, recomputed from their confusion "
          f"matrices:")
    check("the results table covers exactly the runs in the notebook",
          sorted(k for k, _ in RUNS) == sorted(runs),
          f"notebook has {sorted(runs)}")
    check(f"README calls that {NUMBER_WORDS[len(runs)]} models",
          phrase(readme.lower(), rf"\b{NUMBER_WORDS[len(runs)]}",
                 r"(?:models|classifiers)\b"))
    for key, label in RUNS:
        run = runs.get(key)
        if run is None:
            continue
        printed = run["printed"]
        check(f"{label}: printed metrics match its confusion matrix "
              f"{[run['tn'], run['fp'], run['fn'], run['tp']]}",
              printed.get("Accuracy Score") == repr(round(run["accuracy"], 4))
              and printed.get("F1 Score") == repr(round(run["f1"], 4))
              and printed.get("Area Under Curve") == repr(round(run["auc"], 4)),
              f"printed {printed}, matrix gives {round(run['accuracy'], 4)} / "
              f"{round(run['f1'], 4)} / {round(run['auc'], 4)}")
        check(f"{label}: the ROC curve it drew agrees with the AUC it printed",
              round(run["roc_trace"], 4) == round(run["auc"], 4),
              f"trace {run['roc_trace']}, printed {run['auc']}")
        check(f"{label}: scored on all {test} test rows, {leavers} of them "
              f"leavers", run["rows"] == test and
              run["positives"] == leavers,
              f"{run['rows']} rows, {run['positives']} leavers")
        cells_wanted = [run["features"], run["accuracy"], run["precision"],
                        run["recall"], run["f1"], run["auc"],
                        run["probability_auc"]]
        check(f"README results row for {label}: {run['features']} features, "
              + ", ".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                          for v in cells_wanted[1:]),
              row(readme, cells_wanted),
              "no single line carries all seven")
    # The random forest row carries the same accuracy, F1 and label-AUC, so
    # match on what only the do-nothing row has: two undefined columns,
    # because it has neither a precision nor a probability to rank by.
    check(f"README shows the {baseline:.4f} do-nothing row: no leavers "
          f"predicted, F1 0.0000, label-AUC 0.5000, nothing to rank by",
          any(line.count("n/a") >= 2 and
              all(quotes(line, v) for v in (baseline, 0.0, 0.5))
              for line in readme.splitlines()))

    print("what the notebook's 'AUC (labels)' actually is:")
    # Compare the notebook's PRINTED "Area Under Curve" against balanced accuracy
    # recomputed from that same cell's confusion matrix. The earlier form compared
    # runs[...]["auc"] against the expression that defines it, which is constant-true:
    # it survived every mutation, including a rewritten confusion matrix.
    mismatched = [label for key, label in RUNS
                  if key in runs and "Area Under Curve" in runs[key]["printed"]
                  and abs(float(runs[key]["printed"]["Area Under Curve"])
                          - runs[key]["auc"]) > 5e-5]
    check("the printed 'Area Under Curve' is balanced accuracy on hard labels, "
          "for every run",
          not mismatched,
          f"printed value disagrees with the confusion matrix for: {mismatched}")
    blind = [(label, runs[key]) for key, label in RUNS
             if key in runs and runs[key]["tp"] == 0]
    for label, run in blind:
        check(f"{label} predicts no leavers yet ranks at "
              f"{run['probability_auc']:.4f}, so the README must say the "
              f"0.5000 column comes from predicted labels",
              "predicted labels" in readme and
              quotes(readme, run["probability_auc"]),
              "the phrase 'predicted labels' is not in the README"
              if "predicted labels" not in readme
              else absent(run["probability_auc"]))
    check("README names the run(s) that only match the do-nothing baseline: "
          + ", ".join(label for key, label in RUNS if key in runs and
                      runs[key]["accuracy"] <= baseline),
          all(label.split(",")[0].lower() in readme.lower()
              for key, label in RUNS
              if key in runs and runs[key]["accuracy"] <= baseline))

    headline_key = max(runs, key=lambda k: runs[k]["f1"])
    headline = runs[headline_key]
    print(f"the headline model, picked by recomputed F1: "
          f"{dict(RUNS)[headline_key]}")
    check("it is a logistic regression, and the README names one",
          headline_key.startswith("LogisticRegression") and
          "logistic regression" in readme.lower())
    check(f"README quotes its confusion matrix {headline['tn']}, "
          f"{headline['fp']}, {headline['fn']}, {headline['tp']}",
          sequence(readme, [headline["tn"], headline["fp"], headline["fn"],
                            headline["tp"]]))
    flagged = headline["fp"] + headline["tp"]
    check(f"README says it flags {flagged} of the {test} employees",
          quotes(readme, flagged))
    check(f"it beats the {baseline:.4f} baseline",
          headline["accuracy"] > baseline,
          f"{headline['accuracy']:.4f} vs {baseline:.4f}")
    tuned = runs["XGBClassifier|all|search"]
    gap = (headline["f1"] - tuned["f1"]) * 100
    claimed = re.search(r"(\d+) points of F1", readme)
    check(f"README puts the gap to the tuned XGBoost at {gap:.2f} points of "
          f"F1, written as a whole number",
          claimed is not None and abs(int(claimed.group(1)) - gap) < 1,
          f"README says {claimed.group(0)!r}" if claimed else
          "no '<n> points of F1' claim found")

    print("the randomized search, read off its own log:")
    search = next(streams(c) for c in code
                  if "Fitting 5 folds" in streams(c))
    folds, combos, fits = (int(n) for n in re.search(
        r"Fitting (\d+) folds for each of (\d+) candidates, totalling "
        r"(\d+) fits", search).groups())
    check(f"the log's {folds} folds by {combos} combinations is {fits} fits",
          folds * combos == fits)
    check(f"README states {combos} combinations, {folds} folds and "
          f"{fits:,} fits",
          all(quotes(readme, v) for v in (combos, folds, fits)))
    best = re.search(r"Best hyperparameters:\s*\n(\{.*?\})", search, re.S)
    winners = {n: float(v) for n, v in
               re.findall(r"'([a-z_]+)': ([\d.]+)", best.group(1))}
    # The README quotes XGBoost settings for two different models: the
    # plain classifier and the tuned one. So gather every value the
    # notebook actually committed for each knob, from the estimator repr
    # each XGBoost run printed, and require every `name=value` the README
    # writes to be one of them. At least five, so quoting nothing cannot
    # satisfy it.
    # sklearn truncates a long estimator repr with an ellipsis, so the
    # winning dict has to seed this: `subsample` never reaches the repr.
    committed = {name: {value} for name, value in winners.items()}
    for key, run in runs.items():
        if not key.startswith("XGBClassifier"):
            continue
        repr_ = re.search(r"XGBClassifier\((.*?)\)\n", streams(next(
            c for c in code if c["execution_count"] == run["cell"])), re.S)
        for name, value in re.findall(r"(\w+)=([\d.]+)", repr_.group(1)):
            if name in winners:
                committed.setdefault(name, set()).add(float(value))
    quoted = [(n, float(v)) for n, v in
              re.findall(r"([a-z_]+)=([\d.]+)", readme) if n in winners]
    unbacked = [f"{n}={v:g}" for n, v in quoted
                if v not in committed.get(n, set())]
    check(f"the {len(quoted)} XGBoost settings the README quotes are values "
          f"the notebook committed",
          len(quoted) >= 5 and not unbacked,
          f"{len(quoted)} quoted, unbacked: {unbacked or 'none'}, "
          f"committed: { {k: sorted(v) for k, v in committed.items()} }")
    tuned_repr = re.search(r"XGBClassifier\((.*?)\)\n", streams(next(
        c for c in code if c["execution_count"] ==
        runs["XGBClassifier|all|search"]["cell"])), re.S).group(1)
    shown = {n: float(v) for n, v in re.findall(r"(\w+)=([\d.]+)", tuned_repr)
             if n in winners}
    check("the tuned estimator the notebook went on to score is the one the "
          "search picked",
          shown and all(winners[n] == v for n, v in shown.items()),
          f"estimator shows {shown}, search picked "
          f"{ {k: winners[k] for k in shown} }")
    check(f"README states the search's winning "
          f"{winners['n_estimators']:g} estimators",
          phrase(readme, f"{winners['n_estimators']:g}", "estimators") or
          f"n_estimators={winners['n_estimators']:g}" in readme)
    grid_match = re.search(r"LogisticRegression\((.*?)\)",
                           streams(next(c for c in code if c["execution_count"] ==
                                        headline["cell"])))
    # Guard rather than crash. If the headline moves off the logistic regression the
    # earlier form raised AttributeError on None.group() midway through the run, which
    # aborts the remaining checks instead of reporting them.
    check("the headline cell still constructs a LogisticRegression",
          grid_match is not None,
          f"cell {headline['cell']} no longer prints one")
    if grid_match is None:
        # Skip the settings checks that depend on it, but keep going so the rest of
        # the run still reports rather than dying here.
        print(f"\n{len(problems)} README claim(s) drifted: {problems}")
        return 1
    grid = grid_match.group(1)
    settings = dict(re.findall(r"(\w+)=('?[\w.]+'?)", grid))
    # scikit-learn 1.8 deprecated `penalty` on LogisticRegression in favour of
    # `l1_ratio`, so the grid and this check read the mixing parameter instead:
    # 1.0 is the old l1, 0.0 is the old l2, and liblinear fits both.
    mixing = settings["l1_ratio"]
    regularisation = f"{float(settings['C']):g}"
    check(f"README states the grid's winning C={regularisation}, "
          f"l1_ratio={mixing}, {settings['solver']} solver",
          # Scoped to the sentence that names the best model. Plain containment over
          # the whole README passed even when this value was mutated, because the
          # "what moved" section legitimately quotes 2019's C=10 and C=0.1, and now
          # quotes both ends of l1_ratio as well.
          re.search(rf"The best model is[^.]*?C={regularisation}" + AFTER,
                    readme, re.S) and
          re.search(rf"The best model is[^.]*?l1_ratio={re.escape(mixing)}"
                    + AFTER, readme, re.S) and
          f"solver={settings['solver']}" in readme)

    deciles = decile_table(cells)
    print(f"the decile analysis over the same {test} test rows:")
    ones = [r[0] for r in deciles]
    population = [r[2] for r in deciles]
    check(f"the {len(deciles)} deciles hold all {leavers} leavers and all "
          f"{test} rows",
          sum(ones) == leavers and sum(population) == test,
          f"{sum(ones)} leavers over {sum(population)} rows")
    check("every printed hit rate is its own Ones over Population",
          all(round(r[3], 6) == round(r[0] / r[2], 6) for r in deciles))
    captured = []
    running = 0.0
    for one in ones:
        running += one
        captured.append(running / leavers)
    check(f"every printed capture share is the running Ones over the "
          f"{leavers} leavers",
          all(round(r[5], 6) == round(c, 6)
              for r, c in zip(deciles, captured)))
    for depth in (1, 2, 3):
        people = int(sum(population[:depth]))
        caught = int(sum(ones[:depth]))
        check(f"README decile row, top {depth}: {people} employees, {caught} "
              f"leavers, hit rate {caught / people:.4f}, "
              f"{captured[depth - 1]:.4f} of all leavers caught",
              row(readme, [people, caught, caught / people,
                           captured[depth - 1]]),
              "no single line carries all four")

    bands = score_bands(cells)
    print("the score bands over the same test set:")
    check(f"the {len(bands)} bands hold the same {stayers} stayers and "
          f"{leavers} leavers",
          sum(b[1] for b in bands) == stayers and
          sum(b[2] for b in bands) == leavers,
          f"{sum(b[1] for b in bands)} and {sum(b[2] for b in bands)}")
    high = [b for b in bands if b[0] >= 0.8]
    band_size = sum(b[1] + b[2] for b in high)
    check(f"the {band_size} employees scored above 0.80 are all leavers",
          sum(b[1] for b in high) == 0,
          f"{sum(b[1] for b in high)} stayers landed above 0.80")
    # Every place the README states this band's size has to agree. Bare containment
    # was too weak: the figure appears four times, so three could go stale while one
    # kept the check green.
    prose = _prose_only(readme)
    stated = {int(n) for n in
              re.findall(r"of the (\d+) test employees\s*\n?scored above 0\.80", prose)
              + re.findall(r"scored above 0\.80, all (\d+) left", prose)
              + re.findall(r"top band\s*\n?holds (\d+) people", prose)}
    check(f"every README mention of the top band says {band_size}",
          stated == {band_size},
          f"README states {sorted(stated) or 'nothing'}, notebook gives {band_size}")

    audit_checks(readme, test, leavers, headline,
                 {(len(cells), len(code))})

    print("\nimages:")
    for path in referenced_images(readme):
        check(f"README image {path} exists", (REPO / path).is_file())
    stale = sorted(STALE_IMAGES & set(referenced_images(readme)))
    check("the README links to no image whose numbers the notebook has since "
          "contradicted", not stale, f"still linked: {stale}")
    # And the files themselves are gone, not merely unlinked. A PNG with superseded
    # numbers baked into its pixels cannot be checked or corrected, only deleted.
    resurrected = sorted(s for s in STALE_IMAGES if (REPO / s).exists())
    check("no superseded figure has come back on disk",
          not resurrected, f"present again: {resurrected}")
    # The correlation grid's own ranking. The README's "no real winner on this feature
    # set" claim rests on these two mean CV F1 scores; the notebook now prints them, so
    # they are checkable rather than asserted.
    print("the correlation grid's cross-validated ranking:")
    ranking = re.search(
        r"Grid candidates ranked by mean CV F1 \(of (\d+)\):\s*\n"
        r"[^\n]*\n\s*1\s+([\d.]+)[^\n]*\n\s*2\s+([\d.]+)",
        "\n".join(streams(c) for c in code))
    check("the notebook prints the grid's candidate ranking", ranking is not None)
    if ranking:
        n_cand, first, second = ranking.group(1), ranking.group(2), ranking.group(3)
        check(f"README quotes the winning mean CV F1 ({first})", first in readme)
        check(f"README quotes the runner-up ({second})", second in readme)
        check(f"README quotes the candidate count ({n_cand})",
              re.search(rf"{n_cand} candidates", readme) is not None)
        gap = abs(float(first) - float(second))
        check(f"README states the gap between them ({gap:.4f})",
              f"{gap:.4f}" in readme, f"computed {gap:.4f}")

    markdown = "".join("".join(c["source"]) for c in cells
                       if c["cell_type"] == "markdown")
    for path in referenced_images(markdown):
        check(f"notebook image {path} exists", (REPO / path).is_file())

    if problems:
        print(f"\n{len(problems)} README claim(s) drifted: {problems}")
        return 1
    print("\nevery quoted README number matches its artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
