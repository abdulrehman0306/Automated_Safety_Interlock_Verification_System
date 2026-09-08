"""
Charts for the study.

Four figures. The second one is the only one I would put in front of somebody
who had five minutes, because it answers a question I actually had to answer at
work with a relay manual and a stopwatch: how long can the trip delay be before
the motor stops being protected, and how short before it trips on every start.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fault_sim import (
    FAULT_TYPES,
    InterlockParams,
    evaluate_interlock,
    simulate_fault_family,
)
from tag_list import INSULATION_LIMIT_C, RATED_CURRENT_A, SHUTDOWN_CURRENT_A

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def figure_fault_traces() -> str:
    """One panel per fault type, current on the left axis and winding temp on the right."""
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 14), constrained_layout=True)
    axes = axes.ravel()

    for position, (ax, fault) in enumerate(zip(axes, FAULT_TYPES)):
        scenario = simulate_fault_family(fault, 1, rng, 0)[0]
        ax.plot(scenario.t, scenario.current_a / RATED_CURRENT_A, color="tab:blue", lw=1.2)
        # Pinned rather than autoscaled. Left to itself, the locked rotor panel
        # scales to the sensor noise on a flat 6 pu trace and looks like a wild
        # oscillation between 6.44 and 6.55, which is the opposite of the point.
        ax.set_ylim(0, 7)
        ax.set_title(fault.replace("_", " "), fontsize=11)
        if position % 2 == 0:
            ax.set_ylabel("current, per unit", color="tab:blue")
        ax.set_xlabel("seconds")

        twin = ax.twinx()
        twin.plot(scenario.t, scenario.winding_temp_c, color="tab:red", lw=1.2)
        twin.axhline(INSULATION_LIMIT_C, color="tab:red", ls=":", lw=1.0)
        twin.set_ylim(0, 240)
        if position % 2 == 1:
            twin.set_ylabel("winding degC", color="tab:red")

        if scenario.damage_time_s is not None:
            ax.axvline(scenario.damage_time_s, color="black", ls="--", lw=1.0)
            ax.text(
                scenario.damage_time_s, 6.3,
                f" damage {scenario.damage_time_s:.0f} s", fontsize=8,
            )

    axes[-1].axis("off")
    fig.suptitle(
        "Simulated fault traces. Blue is current in per unit, red is winding "
        "temperature.\nDotted red is the class F limit, dashed black is when the "
        "winding crosses it."
    )
    path = os.path.join(FIG_DIR, "fault_traces.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_delay_tradeoff() -> str:
    """Sweep the trip delay and show where the interlock is actually safe.

    The threshold is pinned at the 125 percent setting I used on the real
    machines so the only thing moving is the delay. Too short and it trips on
    normal starting current. Too long and a stalled rotor cooks the winding
    before the timer expires. The window between the two curves is the answer,
    and it is narrower than most people expect.
    """
    rng = np.random.default_rng(99)
    panel = []
    for fault in FAULT_TYPES:
        panel.extend(simulate_fault_family(fault, 25, rng, len(panel)))

    delays = np.geomspace(0.05, 120.0, 40)
    nuisance = []
    too_slow = []
    for delay in delays:
        params = InterlockParams("Motor_Current", SHUTDOWN_CURRENT_A, "high", float(delay), True)
        n_nuisance = 0
        n_slow = 0
        for scenario in panel:
            _, verdict = evaluate_interlock(scenario, params, 0.05)
            if verdict == "nuisance_trip":
                n_nuisance += 1
            elif verdict in {"trip_too_late", "no_trip_damage"}:
                n_slow += 1
        nuisance.append(n_nuisance / len(panel))
        too_slow.append(n_slow / len(panel))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogx(delays, np.array(nuisance) * 100, label="trips when nothing is wrong", lw=2)
    ax.semilogx(delays, np.array(too_slow) * 100, label="motor damaged before it trips", lw=2)
    ax.axvline(3.0, color="black", ls="--", lw=1.2)
    ax.text(3.2, 60, "3 s, the setting I used\non the real machines", fontsize=9)
    ax.set_xlabel("trip delay, seconds")
    ax.set_ylabel("percent of simulated scenarios")
    ax.set_title(
        f"Trip delay tradeoff at a fixed {SHUTDOWN_CURRENT_A:.1f} A threshold "
        f"({len(panel)} scenarios)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "delay_tradeoff.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_training(history_path: str) -> str:
    with open(history_path, "r", encoding="utf-8") as handle:
        blob = json.load(handle)
    history: List[Dict[str, float]] = blob["history"]
    metrics = blob["metrics"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))

    left.plot([h["train_loss"] for h in history], label="train")
    left.plot([h["val_loss"] for h in history], label="validation")
    left.set_xlabel("epoch")
    left.set_ylabel("weighted BCE loss")
    left.set_title("Training loss")
    left.legend()
    left.grid(alpha=0.3)

    per_fault = metrics["per_fault_accuracy"]
    names = list(per_fault.keys())
    right.bar(names, [per_fault[n] * 100 for n in names], color="tab:blue", label="model")
    right.axhline(
        metrics["heldout_majority_baseline"] * 100, color="tab:red", ls="--",
        label="majority class baseline",
    )
    right.set_ylabel("accuracy, percent")
    right.set_title("Accuracy on fault classes never seen in training")
    right.set_ylim(0, 100)
    right.legend()
    right.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "training.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_study(results_path: str) -> str:
    with open(results_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)["summary"]

    # Two panels, and the right hand one is not optional. On its own the stacked
    # bar makes the network look like it is doing most of the work, when what it
    # is mostly doing is rejecting nearly everything. Putting the always reject
    # baseline next to it is the only honest way to show this.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    scopes = ["observable", "strict"]
    labels = ["failures the signal\ncould detect", "every failure\nincluding blind spots"]
    keys = [
        ("caught_by_stage2", "caught by rules only", "tab:green"),
        ("caught_by_both", "caught by both", "tab:olive"),
        ("caught_by_stage3_only", "caught by network only", "tab:blue"),
        ("missed_by_both", "missed by both", "tab:red"),
    ]

    bottom = np.zeros(len(scopes))
    for key, label, colour in keys:
        values = []
        for scope in scopes:
            block = summary[scope][key]
            count = block["count"]
            # caught_by_stage2 in the summary includes the overlap, so subtract
            # it here to make the bar segments add up to the total.
            if key == "caught_by_stage2":
                count -= summary[scope]["caught_by_both"]["count"]
            values.append(count)
        ax.bar(labels, values, bottom=bottom, label=label, color=colour)
        bottom += np.array(values, dtype=float)

    ax.set_ylabel("unsafe generations")
    ax.set_title("Where unsafe outputs were caught")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8)

    block = summary["observable"]
    caught_rules = block["caught_by_stage2"]["count"]
    caught_net = block["caught_by_stage3_only"]["count"] + block["caught_by_both"]["count"]
    n_unsafe = block["unsafe_ground_truth"]["count"]
    n_safe = block["unsafe_ground_truth"]["of"] - n_unsafe

    names = ["rule checker", "network", "reject\neverything"]
    catch = [
        100.0 * caught_rules / max(n_unsafe, 1),
        100.0 * caught_net / max(n_unsafe, 1),
        100.0,
    ]
    false_pos = [
        100.0 * block["stage2_false_positives"]["count"] / max(n_safe, 1),
        100.0 * block["stage3_false_positives"]["count"] / max(n_safe, 1),
        100.0,
    ]

    positions = np.arange(len(names))
    ax2.bar(positions - 0.2, catch, 0.4, label="caught, of unsafe", color="tab:green")
    ax2.bar(positions + 0.2, false_pos, 0.4, label="wrongly flagged, of safe", color="tab:red")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("percent")
    ax2.set_ylim(0, 105)
    ax2.set_title(
        f"Each layer against a validator that rejects everything\n"
        f"({n_unsafe} unsafe and {n_safe} safe generations)"
    )
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "study_results.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    made = [figure_fault_traces(), figure_delay_tradeoff()]

    history = os.path.join(RESULTS_DIR, "training_history.json")
    if os.path.exists(history):
        made.append(figure_training(history))
    else:
        print("no training_history.json yet, run run_study.py --retrain first")

    for name in ("study_results.json", "study_results_mock.json"):
        path = os.path.join(RESULTS_DIR, name)
        if os.path.exists(path):
            made.append(figure_study(path))
            break
    else:
        print("no study results yet, run run_study.py first")

    for path in made:
        print(f"wrote {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
