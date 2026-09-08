"""
Motor fault simulation, and the labelled dataset the PyTorch layer trains on.

This is the part of the project that decides what "unsafe" means, so it is worth
being clear about the modelling choice up front.

I did not build a dq axis machine model. That is a week of work and it would not
change a single label, because what actually decides whether a motor survives is
the winding temperature, and winding temperature comes from a thermal replica.
So there are two pieces here. A current trace per fault type, shaped to match
how that fault behaves on a real drive, and a single node thermal model of the
kind a thermal overload relay runs internally.

The thermal model is:

    dTheta/dt = (Ieff^2 * rated_rise * R - Theta) / (tau * R)

Theta is winding rise above ambient. R is the inverse of cooling effectiveness,
so a blocked fan raises both the settling temperature and the time constant. The
R cancels out of the initial heating rate, which is correct: when a motor stalls
it heats at a rate set by its losses and its thermal mass, and cooling has
nothing to do with it for the first few seconds.

Two sanity checks that made me trust it. At 115 percent of rated current, which
is the service factor, it settles at 92 K rise and never trips the insulation
limit, which is what a service factor means. Locked rotor from a hot start
crosses the limit at about 7.5 seconds against a 10 second stall withstand
rating. Neither of those numbers was tuned. They fell out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from tag_list import (
    COOLING_EFF_FAN_BLOCKED,
    COOLING_EFF_RUNNING,
    COOLING_EFF_STALLED,
    INSULATION_LIMIT_C,
    MOTOR,
    RATED_CURRENT_A,
)

# Rotor heating from negative sequence current. The usual figure quoted for
# induction machines is somewhere between 3 and 6 because the rotor sees slip
# frequency at nearly twice line frequency and the skin effect is brutal. I took
# 5. This is the number I am least confident about in the whole file.
NEG_SEQ_HEATING_FACTOR = 5.0

FAULT_TYPES = [
    "locked_rotor",
    "progressive_overload",
    "sudden_jam",
    "single_phasing",
    "cooling_failure",
    "healthy_start",
    "healthy_load_spike",
]

# Horizon and timestep per fault type. Slow faults get a coarser step so the
# arrays stay a sensible size. The floor of 0.2 s exists because the start
# inrush lasts a couple of seconds and I need enough samples inside it to tell a
# nuisance trip from a real one.
FAULT_TIMEBASE: Dict[str, Tuple[float, float]] = {
    "locked_rotor": (60.0, 0.05),
    "progressive_overload": (1200.0, 0.20),
    "sudden_jam": (150.0, 0.05),
    "single_phasing": (300.0, 0.10),
    "cooling_failure": (2400.0, 0.20),
    "healthy_start": (60.0, 0.05),
    "healthy_load_spike": (150.0, 0.05),
}

SIGNALS = ["Motor_Current", "Motor_Winding_Temp", "Motor_Speed"]

# Scales used to put each signal threshold onto a comparable footing before it
# goes into the network. Nothing clever, just the natural full scale of each.
SIGNAL_SCALE = {
    "Motor_Current": RATED_CURRENT_A,
    "Motor_Winding_Temp": INSULATION_LIMIT_C,
    "Motor_Speed": 1500.0,
}


@dataclass(frozen=True)
class InterlockParams:
    """The interlock reduced to the numbers that decide whether it works.

    This is what stage 3 actually judges. The rule checker reads Structured Text
    and produces one of these, and from that point on the code itself does not
    matter, only what it would do.
    """

    signal: str          # which measured tag the trip is keyed on
    threshold: float     # in the unit of that signal
    direction: str       # "high" trips on rising, "low" trips on falling
    delay_s: float       # on delay before the trip fires, 0 means immediate
    has_shutdown: bool   # does it actually drop the contactor, or only alarm


@dataclass
class Scenario:
    """One simulated fault, with the traces a PLC would see."""

    scenario_id: int
    fault_type: str
    dt: float
    t: np.ndarray
    current_a: np.ndarray
    winding_temp_c: np.ndarray
    speed_rpm: np.ndarray
    damage_time_s: Optional[float]   # None if the motor survives the horizon
    fault_onset_s: float             # trips before this are nuisance trips
    ambient_c: float
    load_pu: float
    cooling_eff: float
    neg_seq_pu: float


def _thermal_trace(
    i_eff_pu: np.ndarray,
    cooling_eff: np.ndarray,
    initial_rise_k: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Integrate the thermal replica for a whole batch of scenarios at once.

    Shapes are (n_scenarios, n_steps) for i_eff_pu and (n_scenarios,) for the
    rest. Doing it batched keeps the Python loop over timesteps only, which
    matters because the cooling failure case runs for 40 minutes of simulated
    time and a per scenario loop was taking longer than the training run.
    """
    n_scen, n_steps = i_eff_pu.shape
    r_thermal = 1.0 / cooling_eff
    tau_eff = MOTOR.winding_tau_running_s * r_thermal
    settle_gain = MOTOR.rated_temp_rise_k * r_thermal

    theta = np.empty((n_scen, n_steps), dtype=np.float32)
    current = initial_rise_k.astype(np.float64).copy()
    for k in range(n_steps):
        target = settle_gain * i_eff_pu[:, k] ** 2
        current = current + dt * (target - current) / tau_eff
        theta[:, k] = current
    return theta


def _current_profile(
    fault_type: str,
    rng: np.random.Generator,
    n: int,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per unit current, speed, negative sequence, cooling and onset.

    Returns arrays shaped (n, len(t)) for the traces and (n,) for the scalars.
    Every fault type gets its own shape because that is the whole point. A stall
    and a slowly seizing bearing produce completely different curves and an
    interlock that handles one can easily miss the other.
    """
    n_steps = t.size
    load_pu = rng.uniform(0.70, 1.00, size=n)
    inrush_s = rng.uniform(1.0, 2.5, size=n)
    lr_mult = rng.normal(MOTOR.locked_rotor_multiple, 0.4, size=n).clip(4.5, 7.5)

    # Healthy baseline that every scenario starts from: inrush, then settle to
    # load. The inrush matters more than it looks. It is the reason you cannot
    # simply set a low threshold with no delay, and half the interesting
    # tradeoff in this study lives inside these two seconds.
    tt = np.broadcast_to(t, (n, n_steps))
    in_inrush = tt < inrush_s[:, None]
    i_pu = np.where(in_inrush, lr_mult[:, None], load_pu[:, None]).astype(np.float64)
    speed = np.where(in_inrush, 0.0, 1470.0)
    speed = speed + rng.normal(0.0, 3.0, size=(n, n_steps))
    i_neg = np.zeros((n, n_steps))
    cooling = np.full(n, COOLING_EFF_RUNNING)
    onset = inrush_s.copy()

    if fault_type == "locked_rotor":
        # Stall on start. The rotor never turns, so current sits at locked rotor
        # value and there is no self ventilation at all. Onset is zero because
        # the fault is present from the moment the contactor closes, which also
        # means no trip in this scenario can be called a nuisance trip.
        i_pu[:] = lr_mult[:, None]
        speed = rng.normal(0.0, 2.0, size=(n, n_steps))
        cooling[:] = COOLING_EFF_STALLED
        onset[:] = 0.0

    elif fault_type == "progressive_overload":
        # Bearing going dry, or material building up on a conveyor. Current
        # creeps up over a minute or two. Slow enough that a human would not
        # notice on a gauge.
        start = rng.uniform(10.0, 60.0, size=n)
        ramp_s = rng.uniform(60.0, 400.0, size=n)
        final = rng.uniform(1.25, 1.70, size=n)
        frac = np.clip((tt - start[:, None]) / ramp_s[:, None], 0.0, 1.0)
        ramped = load_pu[:, None] + frac * (final[:, None] - load_pu[:, None])
        i_pu = np.where(in_inrush, lr_mult[:, None], ramped)
        onset = start

    elif fault_type == "sudden_jam":
        # Something goes into the machine that should not have. Step change.
        start = rng.uniform(8.0, 40.0, size=n)
        level = rng.uniform(2.2, 4.0, size=n)
        jammed = np.where(tt >= start[:, None], level[:, None], load_pu[:, None])
        i_pu = np.where(in_inrush, lr_mult[:, None], jammed)
        speed = np.where(tt >= start[:, None], rng.uniform(0.0, 200.0, size=(n, 1)), speed)
        onset = start

    elif fault_type == "single_phasing":
        # A fuse goes or a contactor pole fails to make. The two surviving
        # phases carry root three times the current, which on its own does not
        # look alarming. What kills the motor is the negative sequence component
        # cooking the rotor. This is the case a plain overcurrent trip is worst
        # at, and it is the fault class I hold out at test time.
        start = rng.uniform(10.0, 60.0, size=n)
        surviving = 1.732 * load_pu
        faulted = np.where(tt >= start[:, None], surviving[:, None], load_pu[:, None])
        i_pu = np.where(in_inrush, lr_mult[:, None], faulted)
        i_neg = np.where(tt >= start[:, None], load_pu[:, None], 0.0)
        onset = start

    elif fault_type == "cooling_failure":
        # Fan cowl packed with lint, which in a textile plant happens constantly.
        # Current never moves. Any interlock watching only current is blind to
        # this no matter what threshold or delay you give it, and that is the
        # cleanest demonstration in the study of why parameters alone do not
        # tell you whether an interlock is safe.
        start = rng.uniform(20.0, 120.0, size=n)
        cooling[:] = rng.uniform(0.25, 0.45, size=n)
        onset = start

    elif fault_type == "healthy_start":
        # Nothing wrong. Included so the study can count interlocks that trip on
        # a perfectly normal start. An interlock that does that gets bypassed by
        # an operator inside a week, and a bypassed interlock protects nobody.
        pass

    elif fault_type == "healthy_load_spike":
        # Brief legitimate load, a heavier batch going through. Must ride
        # through it.
        start = rng.uniform(15.0, 60.0, size=n)
        dur = rng.uniform(1.0, 6.0, size=n)
        level = rng.uniform(1.15, 1.45, size=n)
        spiking = (tt >= start[:, None]) & (tt < (start + dur)[:, None])
        i_pu = np.where(spiking, level[:, None], i_pu)

    else:
        raise ValueError(f"unknown fault type {fault_type}")

    return i_pu, speed, i_neg, cooling, load_pu, onset


def simulate_fault_family(
    fault_type: str,
    n_scenarios: int,
    rng: np.random.Generator,
    first_id: int,
) -> List[Scenario]:
    """Simulate n_scenarios instances of one fault type."""
    horizon, dt = FAULT_TIMEBASE[fault_type]
    t = np.arange(0.0, horizon, dt)

    i_pu, speed, i_neg, cooling, load_pu, onset = _current_profile(
        fault_type, rng, n_scenarios, t
    )

    ambient = rng.uniform(25.0, 45.0, size=n_scenarios)

    # Motors get restarted while still warm from the last run, and a hot start is
    # what turns a survivable stall into a burnt winding. Sampling the initial
    # rise between cold and full load equilibrium covers both.
    initial_rise = load_pu ** 2 * MOTOR.rated_temp_rise_k * rng.uniform(
        0.0, 1.0, size=n_scenarios
    )

    i_eff_sq = i_pu ** 2 + NEG_SEQ_HEATING_FACTOR * i_neg ** 2
    theta = _thermal_trace(np.sqrt(i_eff_sq), cooling, initial_rise, dt)
    winding_c = theta + ambient[:, None]

    # Measurement noise. The CT and the PT100 are not perfect and a threshold
    # sitting right on a noisy signal chatters, which is a real failure mode.
    current_a = i_pu * RATED_CURRENT_A
    current_a = current_a + rng.normal(0.0, 0.015 * RATED_CURRENT_A, current_a.shape)
    winding_c = winding_c + rng.normal(0.0, 1.0, winding_c.shape)

    scenarios = []
    for k in range(n_scenarios):
        over = np.flatnonzero(winding_c[k] >= INSULATION_LIMIT_C)
        damage_time = float(t[over[0]]) if over.size else None
        scenarios.append(
            Scenario(
                scenario_id=first_id + k,
                fault_type=fault_type,
                dt=dt,
                t=t,
                current_a=current_a[k].astype(np.float32),
                winding_temp_c=winding_c[k].astype(np.float32),
                speed_rpm=speed[k].astype(np.float32),
                damage_time_s=damage_time,
                fault_onset_s=float(onset[k]),
                ambient_c=float(ambient[k]),
                load_pu=float(load_pu[k]),
                cooling_eff=float(cooling[k]),
                neg_seq_pu=float(i_neg[k].max()),
            )
        )
    return scenarios


def evaluate_interlock(
    scenario: Scenario, params: InterlockParams, scan_time_s: float
) -> Tuple[Optional[float], str]:
    """Run one interlock against one fault trace and say what happens.

    Returns the trip time, or None if it never trips, plus a verdict string.

    The comparison against damage time is deliberately made on the uninterrupted
    trace. The question being asked is "would this interlock have saved the
    motor", so the right thing to compare against is what would have happened
    with nothing stopping it.
    """
    signal = {
        "Motor_Current": scenario.current_a,
        "Motor_Winding_Temp": scenario.winding_temp_c,
        "Motor_Speed": scenario.speed_rpm,
    }[params.signal]

    # The PLC only sees the process once per scan. A 3 second delay on a 100 ms
    # scan is 30 cycles, and rounding that badly changes trip times enough to
    # matter, so the sampling is modelled rather than assumed away.
    stride = max(1, int(round(scan_time_s / scenario.dt)))
    sampled = signal[::stride]
    sample_dt = stride * scenario.dt

    if params.direction == "high":
        condition = sampled > params.threshold
    else:
        condition = sampled < params.threshold

    trip_time: Optional[float] = None
    if params.has_shutdown:
        # Standard TON behaviour: the condition has to stay true continuously.
        # Any break resets the accumulator, which is why a noisy signal sitting
        # on the threshold can delay a trip forever.
        run = 0
        needed = int(np.ceil(params.delay_s / sample_dt))
        for idx, active in enumerate(condition):
            run = run + 1 if active else 0
            if active and run >= max(needed, 1):
                trip_time = idx * sample_dt
                break

    damage = scenario.damage_time_s

    if trip_time is not None and trip_time < scenario.fault_onset_s - 1e-9:
        # Tripped before anything was actually wrong. On a real machine this is
        # the interlock that gets jumpered out on the second shift.
        return trip_time, "nuisance_trip"

    if damage is None:
        return trip_time, "safe"

    if trip_time is None:
        return None, "no_trip_damage"

    if trip_time <= damage:
        return trip_time, "safe"

    return trip_time, "trip_too_late"


def observable_signals(scenario: Scenario) -> set:
    """Which measured signals actually move during this fault.

    Added after the first end to end run, and it changed the headline number.
    Judging every interlock against every fault made 92 percent of outputs
    unsafe, and nearly all of them for the same reason: a current interlock
    cannot see a blocked fan, so it fails the cooling failure scenarios no
    matter what threshold or delay it has. That is a true statement about
    current interlocks and a useless way to score parameter choices, because it
    swamps every other failure mode.

    So this works out, from the trace rather than from a hardcoded table,
    whether a given signal deviates at all between the fault starting and the
    motor being damaged. Failures on a signal that never moved are reported
    separately from failures the interlock had a genuine chance to catch.
    """
    end = scenario.damage_time_s if scenario.damage_time_s is not None else float(scenario.t[-1])
    onset_idx = int(np.searchsorted(scenario.t, scenario.fault_onset_s))
    end_idx = int(np.searchsorted(scenario.t, end))
    if end_idx <= onset_idx:
        end_idx = min(onset_idx + 1, scenario.t.size - 1)

    seen = set()
    window = slice(onset_idx, end_idx + 1)

    # Five percent of rated current, which is about three times the sensor noise
    # I put on the signal, so this is not picking up measurement wobble.
    base_i = float(scenario.current_a[onset_idx])
    if np.abs(scenario.current_a[window] - base_i).max() > 0.05 * RATED_CURRENT_A:
        seen.add("Motor_Current")

    base_t = float(scenario.winding_temp_c[onset_idx])
    if np.abs(scenario.winding_temp_c[window] - base_t).max() > 10.0:
        seen.add("Motor_Winding_Temp")

    base_n = float(scenario.speed_rpm[onset_idx])
    if np.abs(scenario.speed_rpm[window] - base_n).max() > 75.0:
        seen.add("Motor_Speed")

    return seen


def scenario_features(scenario: Scenario) -> np.ndarray:
    """Describe the fault by what is observable about it, never by its name.

    This is the most important design decision in the learned layer. If I fed
    the fault type in as a one hot code, then holding out a fault class at test
    time would give the network an all zero code it had never seen, and the
    generalisation test would be meaningless. Describing the fault by its
    physical shape instead means a held out class is genuinely just a new point
    in the same feature space, which is the situation you are in on a real plant
    where the next failure is never one you tabulated.
    """
    i_pu = scenario.current_a / RATED_CURRENT_A
    # Skip the start inrush when measuring the settled level, otherwise every
    # scenario looks like it peaked at 6 pu and settled high.
    after_start = scenario.t > (MOTOR.start_inrush_s + 1.0)
    settled = i_pu[after_start] if after_start.any() else i_pu

    over_1p5 = np.flatnonzero(settled > 1.5)
    t_after = scenario.t[after_start] if after_start.any() else scenario.t
    time_to_1p5 = float(t_after[over_1p5[0]]) if over_1p5.size else float(scenario.t[-1])

    # Rate of rise over the first 30 seconds after the start transient. Separates
    # a step change from a slow creep without needing to know which is which.
    window = t_after <= (t_after[0] + 30.0)
    early = settled[window]
    ramp = float((early[-1] - early[0]) / 30.0) if early.size > 1 else 0.0

    return np.array(
        [
            float(settled.max()),
            float(settled[-max(1, settled.size // 5):].mean()),
            np.log1p(time_to_1p5) / np.log1p(2400.0),
            ramp,
            scenario.neg_seq_pu,
            scenario.cooling_eff,
            scenario.ambient_c / 50.0,
            float(scenario.winding_temp_c[0] - scenario.ambient_c) / 100.0,
            scenario.load_pu,
        ],
        dtype=np.float32,
    )


def interlock_features(params: InterlockParams) -> np.ndarray:
    signal_onehot = [1.0 if params.signal == s else 0.0 for s in SIGNALS]
    threshold_norm = params.threshold / SIGNAL_SCALE[params.signal]
    return np.array(
        signal_onehot
        + [
            threshold_norm,
            1.0 if params.direction == "high" else -1.0,
            # Delays in this domain span three orders of magnitude, from a bare
            # scan cycle to two minutes, so a log scale is the only sensible way
            # to present it to a network.
            float(np.log1p(params.delay_s) / np.log1p(120.0)),
            1.0 if params.has_shutdown else 0.0,
        ],
        dtype=np.float32,
    )


FEATURE_NAMES = [
    "peak_current_pu",
    "settled_current_pu",
    "time_to_1p5_pu_log",
    "current_ramp_pu_per_s",
    "neg_seq_pu",
    "cooling_eff",
    "ambient_norm",
    "initial_rise_norm",
    "load_pu",
    "sig_is_current",
    "sig_is_temp",
    "sig_is_speed",
    "threshold_norm",
    "direction",
    "delay_log",
    "has_shutdown",
]


def sample_interlock(rng: np.random.Generator) -> InterlockParams:
    """Draw a plausible interlock, including the ways they go wrong.

    The proportions here are guesses shaped by what the LLM outputs actually
    looked like in the first generation batch. Inverted comparisons and alarm
    only logic both showed up often enough to be worth over representing so the
    network sees enough of them.
    """
    signal = rng.choice(SIGNALS, p=[0.60, 0.25, 0.15])

    if signal == "Motor_Current":
        threshold = float(rng.uniform(1.02, 3.20) * RATED_CURRENT_A)
        direction = "high" if rng.random() > 0.18 else "low"
    elif signal == "Motor_Winding_Temp":
        threshold = float(rng.uniform(80.0, 175.0))
        direction = "high" if rng.random() > 0.18 else "low"
    else:
        threshold = float(rng.uniform(50.0, 1400.0))
        direction = "low" if rng.random() > 0.18 else "high"

    delay_s = float(np.exp(rng.uniform(np.log(0.05), np.log(120.0))))
    has_shutdown = bool(rng.random() > 0.15)
    return InterlockParams(signal, threshold, direction, delay_s, has_shutdown)


def build_dataset(
    n_scenarios_per_fault: int = 220,
    interlocks_per_scenario: int = 22,
    seed: int = 20240921,
) -> Dict[str, np.ndarray]:
    """Produce the full labelled dataset.

    Traces are simulated once per scenario and then reused across many sampled
    interlocks, because the expensive part is integrating the thermal model and
    the cheap part is asking whether a given threshold would have caught it.
    """
    rng = np.random.default_rng(seed)
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    group_rows: List[str] = []
    scenario_rows: List[int] = []
    verdict_rows: List[str] = []

    next_id = 0
    for fault_type in FAULT_TYPES:
        scenarios = simulate_fault_family(fault_type, n_scenarios_per_fault, rng, next_id)
        next_id += n_scenarios_per_fault
        for scenario in scenarios:
            scen_feat = scenario_features(scenario)
            for _ in range(interlocks_per_scenario):
                params = sample_interlock(rng)
                scan = float(rng.uniform(0.01, 0.10))
                _, verdict = evaluate_interlock(scenario, params, scan)
                x_rows.append(np.concatenate([scen_feat, interlock_features(params)]))
                y_rows.append(0 if verdict == "safe" else 1)
                group_rows.append(fault_type)
                scenario_rows.append(scenario.scenario_id)
                verdict_rows.append(verdict)

    return {
        "X": np.stack(x_rows).astype(np.float32),
        "y": np.array(y_rows, dtype=np.int64),
        "fault_type": np.array(group_rows),
        "scenario_id": np.array(scenario_rows, dtype=np.int64),
        "verdict": np.array(verdict_rows),
    }


if __name__ == "__main__":
    # Quick physical sanity print. If these numbers stop matching the comments at
    # the top of the file, the model has drifted and the labels are suspect.
    rng = np.random.default_rng(1)
    for ft in FAULT_TYPES:
        scen = simulate_fault_family(ft, 40, rng, 0)
        dmg = [s.damage_time_s for s in scen if s.damage_time_s is not None]
        if dmg:
            tail = f"median damage time {np.median(dmg):8.1f} s"
        else:
            tail = "no damage within horizon"
        print(f"{ft:22s} damaged {len(dmg):3d}/40   {tail}")
