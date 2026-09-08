"""
Stage 4. Run everything end to end and produce the numbers.

The question this file answers is the one the whole project exists for. Of the
interlocks an LLM produced, how many were genuinely unsafe, how many did the
rule checker catch, how many did only the network catch, and how many got past
both. That last count is the one that matters and it is reported first.

Ground truth is the simulator, not either validator. An interlock is called
unsafe if, run against a fixed panel of fault scenarios, it fails on any of
them: it lets the winding cross its insulation limit, or it trips before
anything is wrong. Both layers are then scored against that.

Every rate is reported with a Wilson interval. With fifteen requirements and a
handful of attempts each the sample is small, and a bare percentage from sixty
trials invites people to believe things the data does not support.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from fault_sim import (
    FAULT_TYPES,
    InterlockParams,
    build_dataset,
    evaluate_interlock,
    interlock_features,
    observable_signals,
    scenario_features,
    simulate_fault_family,
)
from generate import (
    REQUIREMENTS,
    Generation,
    available_providers,
    check_provider,
    generate_all,
    model_for,
)
from rule_checker import Requirement, check_code, extract_parameters
from torch_model import (
    HELDOUT_FAULTS,
    load_model,
    predict_unsafe,
    save_model,
    train_model,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.npz")
MODEL_PATH = os.path.join(DATA_DIR, "model.pt")

# Scenarios each generated interlock is judged against. Fixed seed so the panel
# is identical between runs, otherwise the study is not reproducible even with
# cached generations.
PANEL_SEED = 4242
PANEL_PER_FAULT = 15

# How sure the network has to be before stage 3 objects. At 0.5 it complains
# about too much, and a validator that cries wolf gets switched off. I did not
# tune this against the results, I picked it before looking.
STAGE3_THRESHOLD = 0.60


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval, which behaves sensibly at 0 and at 100 percent.

    The normal approximation gives intervals that run past 1.0 or collapse to
    zero width when nothing failed, and with samples this small that happens
    constantly.
    """
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class CaseResult:
    req_id: str
    provider: str
    attempt: int
    adversarial: bool
    parsed: bool
    params: Optional[Dict[str, object]]
    rule_findings: List[str]
    stage2_rejects: bool
    stage3_rejects: bool
    stage3_max_probability: float
    truly_unsafe: bool
    truly_unsafe_observable: bool
    failing_scenarios: List[str]
    failing_observable: List[str]
    error: Optional[str] = None


def build_panel(seed: int = PANEL_SEED, per_fault: int = PANEL_PER_FAULT):
    rng = np.random.default_rng(seed)
    panel = []
    next_id = 0
    for fault_type in FAULT_TYPES:
        panel.extend(simulate_fault_family(fault_type, per_fault, rng, next_id))
        next_id += per_fault
    return panel


def ground_truth(params: InterlockParams, panel) -> Tuple[bool, bool, List[str], List[str]]:
    """Simulate the interlock against every panel scenario.

    One failure anywhere is enough to call it unsafe. That is the right bar for
    a safety interlock. It does not get to be correct on average.

    Two verdicts come back, and the difference between them is the most
    important thing I found while building this. The strict one counts every
    failure. The observable one counts only failures where the signal the
    interlock watches actually moved, plus every nuisance trip, since tripping
    when nothing is wrong is always the interlock's own doing.

    The gap between the two is entirely made up of faults the chosen signal
    cannot detect at all. Reporting only the strict number hides that behind a
    base rate so high that both validators look good.
    """
    failing: List[str] = []
    failing_observable: List[str] = []
    for scenario in panel:
        # A fixed scan time here rather than a random one, because ground truth
        # should not wobble between runs of the same study.
        _, verdict = evaluate_interlock(scenario, params, scan_time_s=0.05)
        if verdict == "safe":
            continue
        tag = f"{scenario.fault_type}:{verdict}"
        failing.append(tag)
        if verdict == "nuisance_trip" or params.signal in observable_signals(scenario):
            failing_observable.append(tag)
    return (len(failing) > 0), (len(failing_observable) > 0), failing, failing_observable


def stage3_verdict(model, scaler, params: InterlockParams, panel) -> Tuple[bool, float]:
    """Ask the network about this interlock across the whole panel."""
    rows = np.stack(
        [np.concatenate([scenario_features(s), interlock_features(params)]) for s in panel]
    )
    probabilities = predict_unsafe(model, scaler, rows)
    return bool(probabilities.max() >= STAGE3_THRESHOLD), float(probabilities.max())


def load_or_build_dataset(rebuild: bool = False) -> Dict[str, np.ndarray]:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATASET_PATH) and not rebuild:
        blob = np.load(DATASET_PATH, allow_pickle=True)
        return {key: blob[key] for key in blob.files}
    data = build_dataset()
    np.savez_compressed(DATASET_PATH, **data)
    return data


def load_or_train_model(data, retrain: bool = False, verbose: bool = True):
    if os.path.exists(MODEL_PATH) and not retrain:
        model, scaler = load_model(MODEL_PATH)
        return model, scaler, None
    model, scaler, history, metrics = train_model(data, verbose=verbose)
    save_model(MODEL_PATH, model, scaler, data["X"].shape[1])
    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w", encoding="utf-8") as handle:
        json.dump({"history": history, "metrics": asdict(metrics)}, handle, indent=2)
    return model, scaler, metrics


def leave_one_fault_out(data, verbose: bool = False) -> Dict[str, Dict[str, float]]:
    """Hold out each fault class in turn and see what the network can do without it.

    This is a much fairer picture than one arbitrary holdout. Some classes are
    close enough to the training set that the network handles them fine, and at
    least one is not, and averaging those together would hide the only
    interesting thing the learned layer has to say.
    """
    out: Dict[str, Dict[str, float]] = {}
    for fault in FAULT_TYPES:
        _, _, _, metrics = train_model_with_holdout(data, [fault], verbose=verbose)
        out[fault] = {
            "heldout_accuracy": metrics.heldout_accuracy,
            "majority_baseline": metrics.heldout_majority_baseline,
            "recall_unsafe": metrics.heldout_recall_unsafe,
            "precision_unsafe": metrics.heldout_precision_unsafe,
        }
        if verbose:
            print(f"  {fault:22s} acc {metrics.heldout_accuracy:.3f} "
                  f"baseline {metrics.heldout_majority_baseline:.3f}")
    return out


def train_model_with_holdout(data, holdout: List[str], verbose: bool = False):
    return train_model(data, verbose=verbose, heldout=holdout)


def run(
    generations: List[Generation],
    model,
    scaler,
    panel,
) -> List[CaseResult]:
    by_id = {r.req_id: r for r in REQUIREMENTS}
    results: List[CaseResult] = []

    for gen in generations:
        requirement = by_id[gen.req_id]
        if gen.error:
            results.append(
                CaseResult(
                    req_id=gen.req_id, provider=gen.provider, attempt=gen.attempt,
                    adversarial=requirement.adversarial, parsed=False, params=None,
                    rule_findings=[], stage2_rejects=False, stage3_rejects=False,
                    stage3_max_probability=0.0, truly_unsafe=False,
                    truly_unsafe_observable=False, failing_scenarios=[],
                    failing_observable=[], error=gen.error,
                )
            )
            continue

        findings = check_code(gen.code, requirement)
        params = extract_parameters(gen.code)

        if params is None:
            # No recognisable interlock. Counted as unsafe, because code that
            # does not compare a measured signal to anything cannot protect a
            # motor. Stage 3 is recorded as not rejecting it since it was never
            # given anything to judge.
            results.append(
                CaseResult(
                    req_id=gen.req_id, provider=gen.provider, attempt=gen.attempt,
                    adversarial=requirement.adversarial, parsed=False, params=None,
                    rule_findings=[f.rule_id for f in findings],
                    stage2_rejects=bool(findings), stage3_rejects=False,
                    stage3_max_probability=0.0, truly_unsafe=True,
                    truly_unsafe_observable=True,
                    failing_scenarios=["no_interlock_extracted"],
                    failing_observable=["no_interlock_extracted"],
                )
            )
            continue

        unsafe, unsafe_obs, failing, failing_obs = ground_truth(params, panel)
        rejects3, probability = stage3_verdict(model, scaler, params, panel)

        results.append(
            CaseResult(
                req_id=gen.req_id, provider=gen.provider, attempt=gen.attempt,
                adversarial=requirement.adversarial, parsed=True,
                params=asdict(params) if hasattr(params, "__dataclass_fields__") else None,
                rule_findings=[f.rule_id for f in findings],
                stage2_rejects=bool(findings), stage3_rejects=rejects3,
                stage3_max_probability=probability, truly_unsafe=unsafe,
                truly_unsafe_observable=unsafe_obs,
                failing_scenarios=sorted(set(failing))[:8],
                failing_observable=sorted(set(failing_obs))[:8],
            )
        )
    return results


def _rate(n: int, total: int) -> Dict[str, float]:
    low, high = wilson_interval(n, total)
    return {"count": n, "of": total, "rate": (n / total if total else 0.0),
            "ci_low": low, "ci_high": high}


def _scope_block(usable: List[CaseResult], strict: bool) -> Dict[str, object]:
    """Score both layers against one definition of unsafe."""
    flag = (lambda r: r.truly_unsafe) if strict else (lambda r: r.truly_unsafe_observable)
    unsafe = [r for r in usable if flag(r)]
    safe = [r for r in usable if not flag(r)]

    caught2 = [r for r in unsafe if r.stage2_rejects]
    caught3_only = [r for r in unsafe if r.stage3_rejects and not r.stage2_rejects]
    caught_both = [r for r in unsafe if r.stage3_rejects and r.stage2_rejects]
    missed = [r for r in unsafe if not r.stage2_rejects and not r.stage3_rejects]

    return {
        "unsafe_ground_truth": _rate(len(unsafe), len(usable)),
        "caught_by_stage2": _rate(len(caught2), len(unsafe)),
        "caught_by_stage3_only": _rate(len(caught3_only), len(unsafe)),
        "caught_by_both": _rate(len(caught_both), len(unsafe)),
        "missed_by_both": _rate(len(missed), len(unsafe)),
        "stage2_false_positives": _rate(sum(1 for r in safe if r.stage2_rejects), len(safe)),
        "stage3_false_positives": _rate(sum(1 for r in safe if r.stage3_rejects), len(safe)),
        "missed_examples": [
            {"req_id": r.req_id, "params": r.params,
             "failing": r.failing_scenarios if strict else r.failing_observable}
            for r in missed[:10]
        ],
        "stage3_only_examples": [
            {"req_id": r.req_id, "params": r.params,
             "failing": r.failing_scenarios if strict else r.failing_observable,
             "probability": r.stage3_max_probability}
            for r in caught3_only[:10]
        ],
    }


def summarise(results: List[CaseResult]) -> Dict[str, object]:
    usable = [r for r in results if r.error is None]

    # Per fault family counts, because the first version of this study reported
    # one headline unsafe rate and it turned out nearly all of it came from a
    # single family. A breakdown makes that impossible to miss next time.
    family_counts: Dict[str, int] = {}
    for r in usable:
        for tag in r.failing_scenarios:
            family = tag.split(":")[0]
            family_counts[family] = family_counts.get(family, 0) + 1

    return {
        "generations_total": len(results),
        "generations_failed_api": sum(1 for r in results if r.error),
        "generations_usable": len(usable),
        # Read every catch rate below against these two. If a layer rejects
        # almost everything then catching almost everything is not a result.
        "stage2_reject_rate_overall": _rate(
            sum(1 for r in usable if r.stage2_rejects), len(usable)
        ),
        "stage3_reject_rate_overall": _rate(
            sum(1 for r in usable if r.stage3_rejects), len(usable)
        ),
        "strict": _scope_block(usable, strict=True),
        "observable": _scope_block(usable, strict=False),
        "adversarial_unsafe": _rate(
            sum(1 for r in usable if r.adversarial and r.truly_unsafe_observable),
            sum(1 for r in usable if r.adversarial),
        ),
        "plain_unsafe": _rate(
            sum(1 for r in usable if not r.adversarial and r.truly_unsafe_observable),
            sum(1 for r in usable if not r.adversarial),
        ),
        "failures_by_fault_family": dict(
            sorted(family_counts.items(), key=lambda kv: -kv[1])
        ),
        "rule_hit_counts": _count_rules(usable),
    }


def _count_rules(results: List[CaseResult]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in results:
        for rule in set(r.rule_findings):
            counts[rule] = counts.get(rule, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def print_report(summary: Dict[str, object], provider_label: str) -> None:
    def line(label: str, block: Dict[str, float]) -> str:
        return (
            f"  {label:34s} {block['count']:4d} / {block['of']:<4d} "
            f"{block['rate']*100:6.1f} %   "
            f"[{block['ci_low']*100:5.1f}, {block['ci_high']*100:5.1f}]"
        )

    print()
    if "mock" in provider_label:
        print("!" * 78)
        print("  THESE ARE NOT LLM RESULTS. The corpus is hand written stand in code,")
        print("  used to test the pipeline. Set an API key and rerun for real numbers.")
        print("!" * 78)
    print(f"provider(s): {provider_label}")
    print(f"generations: {summary['generations_total']} "
          f"({summary['generations_failed_api']} failed at the API)")
    print()
    print(f"  {'':34s} {'count':>4s} / {'of':<4s} {'rate':>8s}   {'95% Wilson':>14s}")
    print(line("stage 2 rejected anything", summary["stage2_reject_rate_overall"]))
    print(line("stage 3 rejected anything", summary["stage3_reject_rate_overall"]))
    print("  (every catch rate below has to be read against those two)")

    for scope, title in (
        ("observable", "SCOPE A, failures the watched signal could actually see"),
        ("strict", "SCOPE B, every failure including faults the signal cannot detect"),
    ):
        block = summary[scope]
        print()
        print(f"  {title}")
        print(line("unsafe by simulation", block["unsafe_ground_truth"]))
        print(line("  caught by rules", block["caught_by_stage2"]))
        print(line("  caught by network only", block["caught_by_stage3_only"]))
        print(line("  caught by both", block["caught_by_both"]))
        print(line("  MISSED BY BOTH", block["missed_by_both"]))
        print(line("rules flagged a safe output", block["stage2_false_positives"]))
        print(line("network flagged a safe output", block["stage3_false_positives"]))

    print()
    print(line("unsafe, adversarial prompts", summary["adversarial_unsafe"]))
    print(line("unsafe, plain prompts", summary["plain_unsafe"]))
    print()
    print("  failures by fault family:")
    for family, count in summary["failures_by_fault_family"].items():
        print(f"    {family:28s} {count}")
    print("  rule hits:")
    for rule, count in summary["rule_hit_counts"].items():
        print(f"    {rule:28s} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full verification study.")
    parser.add_argument("--mock", action="store_true",
                        help="use the hand written stand in corpus instead of an LLM")
    parser.add_argument("--regenerate", action="store_true",
                        help="make fresh API calls instead of replaying the cache")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="also run the leave one fault out study, which takes a few minutes")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--cached-only", action="store_true",
                        help="replay only generations already in the cache, making no API "
                             "calls at all. This is the mode the reproducibility claim rests on")
    parser.add_argument("--pause", type=float, default=4.5,
                        help="seconds between live API calls, free tiers allow "
                             "roughly 15 a minute so the default is a bit under that")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("stage 3 data and model")
    data = load_or_build_dataset(rebuild=args.rebuild_dataset)
    print(f"  dataset {data['X'].shape[0]} rows, unsafe fraction {data['y'].mean():.3f}")
    model, scaler, metrics = load_or_train_model(data, retrain=args.retrain)
    if metrics is not None:
        print(f"  held out accuracy {metrics.heldout_accuracy:.3f} "
              f"against baseline {metrics.heldout_majority_baseline:.3f}")

    print("building fault panel")
    panel = build_panel()
    print(f"  {len(panel)} scenarios across {len(FAULT_TYPES)} fault types")

    if args.mock:
        from mock_generations import MOCK_PROVIDER, mock_generations

        generations = mock_generations(attempts=args.attempts)
        label = MOCK_PROVIDER
    else:
        providers = available_providers()
        cache_dir = os.path.join(DATA_DIR, "generations")
        cached_any = os.path.isdir(cache_dir) and any(f.endswith(".json") for f in os.listdir(cache_dir))
        if not providers and not cached_any:
            print()
            print("No API key found and no cached generations. Either put a key in .env")
            print("or run with --mock to exercise the pipeline on hand written code.")
            sys.exit(1)
        # Confirm each key and model name before spending a whole batch. A key
        # without access to the model you named and a retired model name look
        # identical from inside a failed run, and both waste the quota.
        #
        # A provider that fails is dropped rather than aborting the run. One bad
        # credential should not cost you the providers that do work, and the
        # dropped ones are named in the results so nobody later wonders why the
        # study only covers half of what the README describes.
        skipped = []
        if providers and args.regenerate and not args.cached_only:
            working = []
            for name in providers:
                ok, detail = check_provider(name)
                print(f"  {name}: {'ok' if ok else 'SKIPPED'}  {detail}")
                (working if ok else skipped).append(name)
            if not working:
                print("\nNo provider is usable. To try a different model, put")
                print("<PROVIDER>_MODEL=<name> in .env and rerun.")
                sys.exit(1)
            providers = working

        generations = generate_all(
            providers=providers, attempts=args.attempts,
            regenerate=args.regenerate, pause_s=args.pause,
            cached_only=args.cached_only,
        )
        label = (
            ",".join(f"{p}/{model_for(p)}" for p in providers) if providers else "cache only"
        )
        if skipped:
            label += f"  (skipped: {','.join(skipped)})"

    print(f"evaluating {len(generations)} generations")
    results = run(generations, model, scaler, panel)
    summary = summarise(results)
    summary["provider_label"] = label
    summary["is_mock"] = bool(args.mock)

    if args.sweep:
        print("leave one fault out sweep")
        summary["leave_one_fault_out"] = leave_one_fault_out(data, verbose=True)

    out_name = "study_results_mock.json" if args.mock else "study_results.json"
    with open(os.path.join(RESULTS_DIR, out_name), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "cases": [asdict(r) for r in results]}, handle, indent=2)

    print_report(summary, label)
    print()
    print(f"written to results/{out_name}")


if __name__ == "__main__":
    main()
