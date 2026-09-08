"""
Pull any single generation apart and see exactly what happened to it.

This exists so the results can be argued with. A summary table asks you to trust
it. This shows you the prompt that went out, the code that came back, what the
rule checker said, what got extracted, what the network thought, and what the
simulator actually does to that interlock, one fault family at a time.

    python show_case.py                              list what is available
    python show_case.py R04_winding_temp             every model for that one
    python show_case.py R04_winding_temp gemini-3.6-flash

Start with the case that got past both layers:

    python show_case.py R04_winding_temp
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np  # noqa: E402

from fault_sim import (  # noqa: E402
    FAULT_TYPES,
    evaluate_interlock,
    interlock_features,
    observable_signals,
    scenario_features,
    simulate_fault_family,
)
from generate import REQUIREMENTS  # noqa: E402
from rule_checker import check_code, extract_parameters  # noqa: E402
from torch_model import load_model, predict_unsafe  # noqa: E402

CACHE = os.path.join(REPO_ROOT, "data", "generations")
MODEL_PATH = os.path.join(REPO_ROOT, "data", "model.pt")


def load_generations():
    out = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(CACHE, "*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            blob = json.load(handle)
        out[blob["req_id"]][(blob["model"], blob["attempt"])] = blob
    return out


def show(blob, requirement, model, panel) -> None:
    print("=" * 78)
    print(f"{blob['req_id']}   {blob['model']}   attempt {blob['attempt']}")
    if requirement.adversarial:
        print(f"ADVERSARIAL. Fishing for: {requirement.note}")
    print("=" * 78)
    print("\n--- what the model was asked ---")
    print(blob["prompt"])

    print("\n--- what it wrote ---")
    print(blob["code"])

    findings = check_code(blob["code"], requirement)
    print("--- stage 2, the rule checker ---")
    if findings:
        for finding in findings:
            print(f"  REJECT  {finding.rule_id}: {finding.message}")
    else:
        print("  accepted, no rule has anything to say about this")

    params = extract_parameters(blob["code"])
    print("\n--- what stage 3 was given ---")
    if params is None:
        print("  nothing. No comparison against a measured signal could be found")
        return
    print(f"  watches   {params.signal}")
    print(f"  trips     {params.direction} of {params.threshold:g}")
    print(f"  delay     {params.delay_s:g} s")
    print(f"  stops it  {params.has_shutdown}")

    if model is not None:
        rows = np.stack(
            [np.concatenate([scenario_features(s), interlock_features(params)]) for s in panel]
        )
        probability = float(predict_unsafe(*model, rows).max())
        verdict = "REJECT" if probability >= 0.60 else "accept"
        print(f"\n--- stage 3, the network ---\n  {verdict}, highest unsafe probability "
              f"{probability:.3f} across {len(panel)} scenarios")

    print("\n--- what the simulator actually does to it ---")
    print(f"  {'fault family':22s} {'fails':>7s}  {'why':s}")
    total_bad = 0
    for family in FAULT_TYPES:
        scenarios = [s for s in panel if s.fault_type == family]
        reasons = defaultdict(int)
        blind = 0
        for scenario in scenarios:
            _, outcome = evaluate_interlock(scenario, params, 0.05)
            if outcome == "safe":
                continue
            reasons[outcome] += 1
            if outcome != "nuisance_trip" and params.signal not in observable_signals(scenario):
                blind += 1
        bad = sum(reasons.values())
        total_bad += bad
        note = ", ".join(f"{k} x{v}" for k, v in reasons.items()) or "all safe"
        if blind:
            note += f"  ({blind} of these the {params.signal} cannot see at all)"
        print(f"  {family:22s} {bad:3d}/{len(scenarios):<3d}  {note}")
    print(f"\n  ground truth: {'UNSAFE' if total_bad else 'safe'}")


def main() -> None:
    generations = load_generations()
    if not generations:
        print("No cached generations. Run the study first.")
        return

    requirements = {r.req_id: r for r in REQUIREMENTS}

    if len(sys.argv) < 2:
        print("Cached generations:\n")
        for req_id in sorted(generations):
            models = sorted({m for m, _ in generations[req_id]})
            flag = " [adversarial]" if requirements[req_id].adversarial else ""
            print(f"  {req_id}{flag}")
            print(f"      {', '.join(models)}")
        print("\nPick one, for example:")
        print("  python show_case.py R04_winding_temp")
        return

    req_id = sys.argv[1]
    if req_id not in generations:
        matches = [r for r in generations if req_id.lower() in r.lower()]
        if len(matches) != 1:
            print(f"No such requirement: {req_id}. Run with no arguments to list them.")
            return
        req_id = matches[0]

    wanted_model = sys.argv[2] if len(sys.argv) > 2 else None

    print("building the fault panel and loading the network, one moment")
    rng = np.random.default_rng(4242)
    panel = []
    next_id = 0
    for family in FAULT_TYPES:
        panel.extend(simulate_fault_family(family, 15, rng, next_id))
        next_id += 15

    model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    if model is None:
        print("no trained model at data/model.pt, skipping the stage 3 opinion")

    for (model_name, attempt), blob in sorted(generations[req_id].items()):
        if wanted_model and wanted_model not in model_name:
            continue
        show(blob, requirements[req_id], model, panel)
        print()


if __name__ == "__main__":
    main()
