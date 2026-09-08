"""
Verification. Run this before believing anything the study prints.

The reason this file exists is that on my last project I found two real bugs
only because I went looking for them on purpose, and the second one changed the
results. So this does not check that the code runs. It checks the specific ways
this project could be quietly wrong and still look fine.

    python verify.py            normal run
    python verify.py --clean    delete the cached dataset and model first

Every check prints PASS or FAIL with the numbers it used, so a failure tells you
what happened rather than just that something did.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Callable, List, Tuple

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

import numpy as np  # noqa: E402

RESULTS: List[Tuple[str, bool, str]] = []
CHECKS: List[Tuple[str, Callable[[], Tuple[bool, str]]]] = []


def check(name: str):
    """Register a check. Registration only, no running.

    The first version ran each check inside the decorator, which meant they all
    executed at import time, before the argument parser had a chance to see
    --clean. So the clean run deleted the cached dataset after every check had
    already used it, which is the exact opposite of what the flag is for.
    """

    def wrap(fn: Callable[[], Tuple[bool, str]]):
        CHECKS.append((name, fn))
        return fn

    return wrap


def run_checks() -> None:
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        RESULTS.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for line in detail.splitlines():
            print(f"         {line}")


# ---------------------------------------------------------------------------
# 1. The rule checker catches what it is supposed to catch
# ---------------------------------------------------------------------------


@check("rule checker against known bad inputs")
def _rule_checker() -> Tuple[bool, str]:
    from known_bad_cases import CASES
    from rule_checker import check_code

    failures = []
    for case in CASES:
        got = sorted({f.rule_id for f in check_code(case.code, case.requirement)})
        expected = sorted(set(case.expected_rules))
        if got != expected:
            failures.append(f"{case.name}: expected {expected}, got {got}")

    detail = f"{len(CASES) - len(failures)} of {len(CASES)} cases behave as specified"
    if failures:
        detail += "\n" + "\n".join(failures)
    return not failures, detail


@check("layer boundary held, structurally perfect but fatal code is not rule flagged")
def _layer_boundary() -> Tuple[bool, str]:
    """The two cases stage 2 must not catch.

    If a rule ever starts flagging these, somebody has put physics into the
    deterministic layer and the comparison between the layers stops meaning
    anything. That would not look like a bug. The results would just quietly
    become a different claim.
    """
    from known_bad_cases import CASES
    from rule_checker import check_code

    names = [
        "physically_unsafe_but_structurally_perfect",
        "current_interlock_blind_to_cooling_failure",
    ]
    problems = []
    for name in names:
        case = next(c for c in CASES if c.name == name)
        findings = check_code(case.code, case.requirement)
        if findings:
            problems.append(f"{name} was flagged by {[f.rule_id for f in findings]}")
    return not problems, "\n".join(problems) or f"{len(names)} cases pass stage 2, as intended"


# ---------------------------------------------------------------------------
# 2. The thermal model still behaves like a motor
# ---------------------------------------------------------------------------


@check("thermal model matches real motor behaviour")
def _thermal() -> Tuple[bool, str]:
    from fault_sim import _thermal_trace
    from tag_list import (
        COOLING_EFF_RUNNING,
        COOLING_EFF_STALLED,
        INSULATION_LIMIT_C,
        MOTOR,
    )

    def run(i_pu, cooling, init_rise, ambient, horizon, dt=0.02):
        steps = int(horizon / dt)
        theta = _thermal_trace(
            np.full((1, steps), i_pu), np.array([cooling]), np.array([init_rise]), dt
        )
        temps = theta[0] + ambient
        over = np.flatnonzero(temps >= INSULATION_LIMIT_C)
        return (float(over[0] * dt) if over.size else None), float(temps[-1])

    service_time, service_temp = run(1.15, COOLING_EFF_RUNNING, 0.0, 40.0, 3000, 0.5)
    hot_stall, _ = run(6.0, COOLING_EFF_STALLED, MOTOR.rated_temp_rise_k, 40.0, 60)
    cold_stall, _ = run(6.0, COOLING_EFF_STALLED, 0.0, 25.0, 60)

    problems = []
    # A motor with a 1.15 service factor is allowed to run at 115 percent
    # continuously. If the model damages it, the model is wrong.
    if service_time is not None:
        problems.append(f"damaged at service factor load, settles at {service_temp:.1f} degC")
    # Datasheet stall withstand for this size is 8 to 15 s hot, roughly double
    # that cold. Wide brackets on purpose, this is a sanity check not a fit.
    if not (5.0 <= (hot_stall or 0) <= 15.0):
        problems.append(f"hot stall damage at {hot_stall} s, expected 5 to 15")
    if not (14.0 <= (cold_stall or 0) <= 35.0):
        problems.append(f"cold stall damage at {cold_stall} s, expected 14 to 35")

    detail = (
        f"service factor 1.15 settles at {service_temp:.1f} degC with no damage\n"
        f"hot stall damages at {hot_stall:.1f} s, cold stall at {cold_stall:.1f} s"
    )
    if problems:
        detail = "\n".join(problems)
    return not problems, detail


# ---------------------------------------------------------------------------
# 3. The split is by scenario, and the model actually learns
# ---------------------------------------------------------------------------


@check("train and test split by fault class and by scenario, with no overlap")
def _split() -> Tuple[bool, str]:
    from fault_sim import build_dataset
    from torch_model import HELDOUT_FAULTS, split_by_fault

    data = _cached_dataset(build_dataset)
    idx = split_by_fault(data)

    train_scen = set(data["scenario_id"][idx["train"]].tolist())
    val_scen = set(data["scenario_id"][idx["val"]].tolist())
    test_scen = set(data["scenario_id"][idx["test"]].tolist())
    train_faults = set(data["fault_type"][idx["train"]].tolist())
    test_faults = set(data["fault_type"][idx["test"]].tolist())

    problems = []
    if train_scen & test_scen:
        problems.append(f"{len(train_scen & test_scen)} scenarios appear in train and test")
    if train_scen & val_scen:
        problems.append(f"{len(train_scen & val_scen)} scenarios appear in train and validation")
    if train_faults & test_faults:
        problems.append(f"fault classes leak across the split: {train_faults & test_faults}")
    if test_faults != set(HELDOUT_FAULTS):
        problems.append(f"test set holds {test_faults}, expected {set(HELDOUT_FAULTS)}")

    detail = (
        f"train {len(idx['train'])} rows over {len(train_scen)} scenarios\n"
        f"val   {len(idx['val'])} rows over {len(val_scen)} scenarios\n"
        f"test  {len(idx['test'])} rows over {len(test_scen)} scenarios, "
        f"fault classes {sorted(test_faults)}"
    )
    if problems:
        detail = "\n".join(problems)
    return not problems, detail


@check("model trains, and is not just predicting the majority class")
def _training() -> Tuple[bool, str]:
    from fault_sim import build_dataset
    from torch_model import train_model

    data = _cached_dataset(build_dataset)
    _, _, history, metrics = train_model(data, verbose=False)

    first = history[0]["train_loss"]
    last = history[-1]["train_loss"]
    positive_rate = metrics.heldout_predicted_positive_rate

    problems = []
    if last >= first:
        problems.append(f"train loss did not fall, {first:.4f} to {last:.4f}")
    if last > 0.9 * first:
        problems.append(f"train loss barely moved, {first:.4f} to {last:.4f}")
    # A model that answers the same thing every time has a predicted positive
    # rate pinned at 0 or 1. That is the failure this check exists for, and it
    # is invisible in an accuracy number.
    if positive_rate < 0.02 or positive_rate > 0.98:
        problems.append(f"predicts one class almost always, positive rate {positive_rate:.3f}")
    if metrics.heldout_accuracy <= metrics.heldout_majority_baseline:
        problems.append(
            f"held out accuracy {metrics.heldout_accuracy:.3f} does not beat the majority "
            f"baseline {metrics.heldout_majority_baseline:.3f}"
        )

    detail = (
        f"train loss {first:.4f} to {last:.4f}\n"
        f"held out accuracy {metrics.heldout_accuracy:.3f} against baseline "
        f"{metrics.heldout_majority_baseline:.3f}\n"
        f"predicted positive rate {positive_rate:.3f}, recall on unsafe "
        f"{metrics.heldout_recall_unsafe:.3f}\n"
        f"per class: {metrics.per_fault_accuracy}"
    )
    if problems:
        detail += "\n" + "\n".join(problems)
    return not problems, detail


@check("high in distribution accuracy is not hiding a leak")
def _leak() -> Tuple[bool, str]:
    """Shuffle the labels and retrain. Accuracy must collapse to the baseline.

    If a model can still score well after the labels are destroyed, it is
    reading something it should not have access to. This is the cheapest leak
    detector I know and it has caught real problems for me before.
    """
    from fault_sim import build_dataset
    from torch_model import train_model

    data = _cached_dataset(build_dataset)
    shuffled = dict(data)
    rng = np.random.default_rng(3)
    shuffled["y"] = rng.permutation(data["y"])

    _, _, _, metrics = train_model(shuffled, epochs=25, verbose=False)
    margin = metrics.heldout_accuracy - metrics.heldout_majority_baseline

    ok = margin < 0.05
    detail = (
        f"with labels shuffled, held out accuracy {metrics.heldout_accuracy:.3f} "
        f"against baseline {metrics.heldout_majority_baseline:.3f}, margin {margin:+.3f}"
    )
    if not ok:
        detail += "\nthe model beat chance on random labels, something is leaking"
    return ok, detail


# ---------------------------------------------------------------------------
# 4. No secrets anywhere that would end up committed
# ---------------------------------------------------------------------------

SECRET_PATTERNS = [
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("Groq API key", re.compile(r"gsk_[0-9A-Za-z]{40,}")),
    ("OpenAI style key", re.compile(r"sk-[0-9A-Za-z]{32,}")),
    ("bearer token", re.compile(r"Bearer\s+[0-9A-Za-z_\-\.]{30,}")),
]

SKIP_DIRS = {".venv", "__pycache__", ".git", ".ipynb_checkpoints", "node_modules"}


@check("no API key in any file that would be committed")
def _secrets() -> Tuple[bool, str]:
    gitignore = os.path.join(REPO_ROOT, ".gitignore")
    problems = []

    if not os.path.exists(gitignore):
        problems.append("there is no .gitignore at all")
    else:
        with open(gitignore, "r", encoding="utf-8") as handle:
            if ".env" not in handle.read().split():
                problems.append(".gitignore does not exclude .env")

    scanned = 0
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name == ".env":
                # Not scanned because it is gitignored, but its absence from the
                # commit is what matters, and that is checked above.
                continue
            path = os.path.join(root, name)
            if os.path.getsize(path) > 4_000_000:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    text = handle.read()
            except OSError:
                continue
            scanned += 1
            for label, pattern in SECRET_PATTERNS:
                match = pattern.search(text)
                if match:
                    rel = os.path.relpath(path, REPO_ROOT)
                    problems.append(f"{label} shaped string in {rel}")

    detail = f"scanned {scanned} files, .gitignore excludes .env"
    if problems:
        detail = "\n".join(problems)
    return not problems, detail


# ---------------------------------------------------------------------------
# 5. Reproducibility
# ---------------------------------------------------------------------------

_DATASET_CACHE = {}


def _cached_dataset(builder):
    """Build the dataset once per verify run rather than once per check."""
    if "data" not in _DATASET_CACHE:
        _DATASET_CACHE["data"] = builder()
    return _DATASET_CACHE["data"]


@check("dataset generation is deterministic for a fixed seed")
def _dataset_determinism() -> Tuple[bool, str]:
    from fault_sim import build_dataset

    first = build_dataset(n_scenarios_per_fault=30, interlocks_per_scenario=6, seed=1234)
    second = build_dataset(n_scenarios_per_fault=30, interlocks_per_scenario=6, seed=1234)

    same_x = np.array_equal(first["X"], second["X"])
    same_y = np.array_equal(first["y"], second["y"])
    detail = (
        f"{first['X'].shape[0]} rows rebuilt, features identical: {same_x}, "
        f"labels identical: {same_y}"
    )
    return bool(same_x and same_y), detail


@check("training is deterministic for a fixed seed")
def _training_determinism() -> Tuple[bool, str]:
    from fault_sim import build_dataset
    from torch_model import train_model

    data = build_dataset(n_scenarios_per_fault=40, interlocks_per_scenario=8, seed=99)
    _, _, _, first = train_model(data, epochs=12, verbose=False)
    _, _, _, second = train_model(data, epochs=12, verbose=False)

    ok = (
        abs(first.final_train_loss - second.final_train_loss) < 1e-9
        and abs(first.heldout_accuracy - second.heldout_accuracy) < 1e-9
    )
    detail = (
        f"run 1 loss {first.final_train_loss:.8f} accuracy {first.heldout_accuracy:.6f}\n"
        f"run 2 loss {second.final_train_loss:.8f} accuracy {second.heldout_accuracy:.6f}"
    )
    return ok, detail


@check("the README numbers match what the pipeline actually produced")
def _readme_matches_results() -> Tuple[bool, str]:
    """Cross check every count quoted in the README against results/.

    This exists because the README drifting away from the results is the single
    most common defect I have had in this project. I corrected a stale case
    count three separate times by hand before admitting that a person checking
    it by eye was not going to keep working. A README that quietly disagrees
    with its own results is worse than one with no numbers in it, because it
    reads as authoritative.
    """
    from known_bad_cases import CASES

    readme_path = os.path.join(REPO_ROOT, "README.md")
    results_path = os.path.join(REPO_ROOT, "results", "study_results.json")
    if not os.path.exists(results_path):
        return True, "no live study results yet, nothing to cross check"

    with open(readme_path, "r", encoding="utf-8") as handle:
        readme = handle.read()
    with open(results_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)["summary"]

    problems = []
    checked = 0

    # The corpus size, quoted as "N of N hand written cases".
    corpus = re.search(r"(\d+) of (\d+) hand written cases", readme)
    if corpus:
        checked += 1
        if int(corpus.group(2)) != len(CASES):
            problems.append(
                f"README says {corpus.group(2)} hand written cases, there are {len(CASES)}"
            )

    # Every "label  N / M  rate %" line inside the results block.
    labels = {
        "stage 2 rejected anything": ("stage2_reject_rate_overall", None),
        "stage 3 rejected anything": ("stage3_reject_rate_overall", None),
        "unsafe by simulation": ("unsafe_ground_truth", "observable"),
        "caught by rules": ("caught_by_stage2", "observable"),
        "caught by network only": ("caught_by_stage3_only", "observable"),
        "caught by both": ("caught_by_both", "observable"),
        "MISSED BY BOTH": ("missed_by_both", "observable"),
        "rules flagged a safe output": ("stage2_false_positives", "observable"),
        "network flagged a safe output": ("stage3_false_positives", "observable"),
        "unsafe, adversarial prompts": ("adversarial_unsafe", None),
        "unsafe, plain prompts": ("plain_unsafe", None),
    }
    for label, (key, scope) in labels.items():
        pattern = re.escape(label) + r"\s+(\d+) / (\d+)"
        match = re.search(pattern, readme)
        if not match:
            continue
        checked += 1
        block = summary[scope][key] if scope else summary[key]
        if int(match.group(1)) != block["count"] or int(match.group(2)) != block["of"]:
            problems.append(
                f"README says '{label}' is {match.group(1)}/{match.group(2)}, "
                f"results say {block['count']}/{block['of']}"
            )

    # The leave one fault out table, which is easy to forget after a retrain.
    for fault, row in summary.get("leave_one_fault_out", {}).items():
        pretty = fault.replace("_", " ")
        match = re.search(
            rf"\|\s*{re.escape(pretty)}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", readme
        )
        if not match:
            continue
        checked += 1
        if abs(float(match.group(1)) - row["heldout_accuracy"]) > 0.0006:
            problems.append(
                f"README sweep row '{pretty}' accuracy {match.group(1)}, "
                f"results say {row['heldout_accuracy']:.3f}"
            )

    detail = f"cross checked {checked} numbers quoted in the README"
    if problems:
        detail = "\n".join(problems)
    return not problems, detail


@check("the full pipeline reproduces the same study numbers")
def _pipeline_reproducible() -> Tuple[bool, str]:
    """Run the study twice as a subprocess and compare the summary.

    A subprocess rather than an in process call on purpose, because that is how
    somebody else would run it, and it catches anything that only works because
    of state left behind by an earlier check in this file.
    """
    script = os.path.join(REPO_ROOT, "src", "run_study.py")
    out_path = os.path.join(REPO_ROOT, "results", "study_results_mock.json")

    digests = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, script, "--mock"],
            cwd=os.path.join(REPO_ROOT, "src"),
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return False, f"run_study exited {proc.returncode}\n{proc.stderr[-800:]}"
        with open(out_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)["summary"]
        summary.pop("leave_one_fault_out", None)
        digests.append(hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest())

    ok = digests[0] == digests[1]
    detail = f"summary digest run 1 {digests[0][:16]}\nsummary digest run 2 {digests[1][:16]}"
    if not ok:
        detail += "\nthe study is not reproducible, something unseeded is in the path"
    return ok, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true",
                        help="delete the cached dataset and model before running")
    args = parser.parse_args()

    if args.clean:
        for name in ("dataset.npz", "model.pt"):
            path = os.path.join(REPO_ROOT, "data", name)
            if os.path.exists(path):
                os.remove(path)
                print(f"removed data/{name}")
        print()

    run_checks()

    print("=" * 78)
    print(f"{sum(1 for _, ok, _ in RESULTS if ok)} of {len(RESULTS)} checks passed")
    for name, ok, _ in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    print("=" * 78)
    sys.exit(0 if all(ok for _, ok, _ in RESULTS) else 1)


if __name__ == "__main__":
    main()
