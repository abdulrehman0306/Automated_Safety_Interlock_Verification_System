"""
Builds notebooks/analysis.ipynb.

The notebook is generated rather than written by hand because a notebook that
duplicates run_study.py drifts away from it within a week. Everything here reads
artifacts out of results/ and never recomputes the study, so the notebook can
only ever show what the pipeline actually produced.
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# nbformat wants a stable id on every cell. Deriving it from a counter rather
# than a uuid so that rebuilding the notebook does not churn the whole file.
_COUNTER = iter(range(1, 10_000))


def md(text):
    return {"cell_type": "markdown", "id": f"md{next(_COUNTER):03d}",
            "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "id": f"cd{next(_COUNTER):03d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


CELLS = [
    md("""
# Automated safety interlock verification, walkthrough

This notebook reads what the pipeline produced. It does not rerun the study, so
nothing in here can disagree with `results/`. If a number looks wrong, the fix
is in `src/`, not here.

Run `python src/run_study.py --cached-only --attempts 1` and `python src/make_figures.py` first.
"""),
    code("""
import json, os, sys
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("../tests"))
from IPython.display import Image, display

RESULTS = os.path.abspath("../results")
FIGS = os.path.join(RESULTS, "figures")
"""),
    md("""
## 1. The machine

Every threshold in this project is a multiple of the rated current of a real
motor, and the two percentages are the ones I set in TIA Portal on the machines
I commissioned.
"""),
    code("""
from tag_list import (MOTOR, RATED_CURRENT_A, SAFE_MODE_CURRENT_A,
                      SHUTDOWN_CURRENT_A, INSULATION_LIMIT_C, tag_list_for_prompt)

print(f"{MOTOR.rated_power_kw} kW, {MOTOR.voltage_v} V, insulation class {MOTOR.insulation_class}")
print(f"rated current          {RATED_CURRENT_A:6.1f} A")
print(f"safe mode at 110 pct   {SAFE_MODE_CURRENT_A:6.1f} A")
print(f"shutdown at 125 pct    {SHUTDOWN_CURRENT_A:6.1f} A")
print(f"locked rotor           {RATED_CURRENT_A * MOTOR.locked_rotor_multiple:6.1f} A")
print(f"insulation limit       {INSULATION_LIMIT_C:6.1f} degC")
"""),
    md("""
## 2. The faults

Seven fault families. Two of them are healthy, included so the study can count
interlocks that trip when nothing is wrong. An interlock that nuisance trips
gets bypassed by an operator, and a bypassed interlock protects nobody.
"""),
    code("""display(Image(os.path.join(FIGS, "fault_traces.png")))"""),
    md("""
## 3. The question the rule checker cannot answer

This is the whole argument for having a second layer. Below is the trip delay
swept across three orders of magnitude with the threshold pinned at the 125
percent setting. Short delays nuisance trip on starting current. Long delays let
a stalled rotor cook. Both ends are made of numbers that look perfectly
reasonable on their own, and no static rule that looks at one parameter at a
time can tell them apart.
"""),
    code("""display(Image(os.path.join(FIGS, "delay_tradeoff.png")))"""),
    md("""
## 4. Stage 2, the deterministic checker

Every hand written case with a known fault. Most must be caught and two
must not be, and the two that must not be are the reason stage 3 exists.
"""),
    code("""
from known_bad_cases import CASES
from rule_checker import check_code, extract_parameters

for case in CASES:
    found = sorted({f.rule_id for f in check_code(case.code, case.requirement)})
    expected = sorted(set(case.expected_rules))
    mark = "ok " if found == expected else "BAD"
    print(f"{mark} {case.name:46s} {found}")
"""),
    md("""
### The two that get through

Both of these pass every deterministic rule. The first trips at 150 percent
after 30 seconds, and on a locked rotor the winding is past its insulation limit
in under eight. The second is textbook correct and completely blind to a blocked
fan cowl.
"""),
    code("""
from fault_sim import simulate_fault_family, evaluate_interlock
import numpy as np

rng = np.random.default_rng(5)
for name in ["physically_unsafe_but_structurally_perfect",
             "current_interlock_blind_to_cooling_failure"]:
    case = next(c for c in CASES if c.name == name)
    params = extract_parameters(case.code)
    print(f"{name}\\n  stage 2 findings: {[f.rule_id for f in check_code(case.code, case.requirement)]}")
    print(f"  extracted: {params}")
    for fault in ["locked_rotor", "cooling_failure"]:
        verdicts = [evaluate_interlock(s, params, 0.05)[1]
                    for s in simulate_fault_family(fault, 10, rng, 0)]
        bad = sum(1 for v in verdicts if v != "safe")
        print(f"  {fault:20s} fails {bad} of 10")
    print()
"""),
    md("""
## 5. Stage 3, the learned layer

Trained on five fault families, tested on two it has never seen. The bar chart
on the right is the one to read. Accuracy next to the majority class baseline,
because a model that always answers unsafe scores whatever the base rate is.
"""),
    code("""
display(Image(os.path.join(FIGS, "training.png")))

with open(os.path.join(RESULTS, "training_history.json")) as fh:
    metrics = json.load(fh)["metrics"]
for key in ["indist_accuracy", "heldout_accuracy", "heldout_majority_baseline",
            "heldout_recall_unsafe", "heldout_precision_unsafe",
            "heldout_predicted_positive_rate"]:
    print(f"{key:34s} {metrics[key]:.3f}")
print(f"per class: {metrics['per_fault_accuracy']}")
"""),
    md("""
## 6. The study

Two scopes. Scope A counts only failures the signal the interlock watches could
actually see. Scope B counts everything, including faults that signal is
physically blind to. The first version of this study reported only scope B, and
the headline unsafe rate was so high that both layers looked good for the wrong
reason.

Read every catch rate against the overall rejection rates printed first. A layer
that rejects almost everything catches almost everything, and that is not a
result.
"""),
    code("""
name = "study_results.json"
if not os.path.exists(os.path.join(RESULTS, name)):
    name = "study_results_mock.json"
with open(os.path.join(RESULTS, name)) as fh:
    summary = json.load(fh)["summary"]

if summary.get("is_mock"):
    print("!! hand written stand in corpus, NOT LLM output !!\\n")

def show(label, block):
    print(f"  {label:34s} {block['count']:4d} / {block['of']:<4d} "
          f"{block['rate']*100:6.1f} %   "
          f"[{block['ci_low']*100:5.1f}, {block['ci_high']*100:5.1f}]")

show("stage 2 rejected anything", summary["stage2_reject_rate_overall"])
show("stage 3 rejected anything", summary["stage3_reject_rate_overall"])
for scope in ["observable", "strict"]:
    print(f"\\n  scope: {scope}")
    for key in ["unsafe_ground_truth", "caught_by_stage2", "caught_by_stage3_only",
                "caught_by_both", "missed_by_both", "stage2_false_positives",
                "stage3_false_positives"]:
        show(key, summary[scope][key])
"""),
    code("""display(Image(os.path.join(FIGS, "study_results.png")))"""),
    md("""
## 7. What only the network caught, and what got through both

These two lists are the actual output of the project. Everything before this
point exists to produce them.
"""),
    code("""
for scope in ["observable", "strict"]:
    print(f"=== {scope} ===")
    for label in ["stage3_only_examples", "missed_examples"]:
        print(f"  {label}:")
        for item in summary[scope][label][:5]:
            p = item["params"]
            if p:
                print(f"    {item['req_id']:26s} {p['signal']} {p['direction']} "
                      f"{p['threshold']:.1f} delay {p['delay_s']:.1f}s -> {item['failing'][:3]}")
            else:
                print(f"    {item['req_id']:26s} {item['failing'][:3]}")
    print()
"""),
    md("""
## 8. Leave one fault out

One arbitrary holdout is not enough to say whether the learned layer
generalises. This holds out each fault family in turn. The margin column is
accuracy minus the majority baseline, and it is the only column worth reading.
"""),
    code("""
sweep = summary.get("leave_one_fault_out")
if not sweep:
    print("run: python src/run_study.py --mock --sweep")
else:
    print(f"{'held out':24s} {'acc':>6s} {'baseline':>9s} {'margin':>8s} {'recall':>8s}")
    for fault, row in sweep.items():
        margin = row["heldout_accuracy"] - row["majority_baseline"]
        print(f"{fault:24s} {row['heldout_accuracy']:6.3f} {row['majority_baseline']:9.3f} "
              f"{margin:+8.3f} {row['recall_unsafe']:8.3f}")
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    path = os.path.join(REPO_ROOT, "notebooks", "analysis.ipynb")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(NOTEBOOK, handle, indent=1)
    print(f"wrote {os.path.relpath(path, REPO_ROOT)} with {len(CELLS)} cells")
