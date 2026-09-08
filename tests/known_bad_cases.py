"""
Hand written Structured Text with known faults, used to test the rule checker.

Every case here is something I either saw a model actually produce or would
expect it to produce. The expected_rules field is what the checker must find. A
case with an empty expected_rules must come back clean, and two of those are
deliberately unsafe code that stage 2 is not supposed to catch. Those two are
the whole argument for having stage 3, so if the checker ever starts flagging
them, the boundary between the layers has been crossed and the study results
stop meaning what they say.
"""

from dataclasses import dataclass
from typing import List

from rule_checker import Requirement


@dataclass
class Case:
    name: str
    code: str
    requirement: Requirement
    expected_rules: List[str]
    note: str = ""


DELAY_REQ = Requirement(
    req_id="test_delay",
    text="Stop the motor if current exceeds 125 percent of rated for 3 seconds.",
    requires_delay=True,
    expected_signal="Motor_Current",
)

NO_DELAY_REQ = Requirement(
    req_id="test_nodelay",
    text="Stop the motor immediately if current exceeds 125 percent of rated.",
    requires_delay=False,
    expected_signal="Motor_Current",
)


CASES: List[Case] = [
    Case(
        name="good_baseline",
        code="""
VAR
    Overload_Timer : TON;
END_VAR

Overload_Timer(IN := Motor_Current > 51.9, PT := T#3S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
    Overload_Alarm := TRUE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=[],
        note="what a correct answer looks like",
    ),
    Case(
        name="hallucinated_tag",
        code="""
IF Motor_Amps > 51.9 THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=NO_DELAY_REQ,
        expected_rules=["UNKNOWN_TAG"],
    ),
    Case(
        name="inverted_current_comparison",
        code="""
IF Motor_Current < 51.9 THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=NO_DELAY_REQ,
        expected_rules=["INVERTED_COMPARISON"],
        note="the one that scares me, it looks completely normal",
    ),
    Case(
        name="alarm_only_no_shutdown",
        code="""
IF Motor_Current > 51.9 THEN
    Overload_Alarm := TRUE;
END_IF;
""",
        requirement=NO_DELAY_REQ,
        expected_rules=["NO_SHUTDOWN_ACTION"],
    ),
    Case(
        name="missing_timer",
        code="""
IF Motor_Current > 51.9 THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=["MISSING_TIMER"],
    ),
    Case(
        name="timer_with_zero_preset",
        code="""
VAR
    Overload_Timer : TON;
END_VAR

Overload_Timer(IN := Motor_Current > 51.9, PT := T#0S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=["MISSING_TIMER"],
    ),
    Case(
        name="threshold_below_rated",
        code="""
IF Motor_Current > 25.0 THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=NO_DELAY_REQ,
        expected_rules=["THRESHOLD_OUT_OF_BAND"],
        note="trips on normal load, machine never runs",
    ),
    Case(
        name="threshold_impossibly_high",
        code="""
IF Motor_Current > 5000.0 THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=NO_DELAY_REQ,
        expected_rules=["THRESHOLD_OUT_OF_BAND"],
        note="can never be reached, so the interlock does nothing",
    ),
    Case(
        name="temperature_above_insulation_limit",
        code="""
IF Motor_Winding_Temp > 200.0 THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=NO_DELAY_REQ,
        expected_rules=["THRESHOLD_OUT_OF_BAND"],
        note="class F is gone at 155, tripping at 200 is tripping after the funeral",
    ),
    Case(
        name="inverted_speed_comparison",
        code="""
IF Motor_Speed > 100.0 THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=NO_DELAY_REQ,
        expected_rules=["INVERTED_COMPARISON"],
        note="stall detection has to trip on speed falling, not rising",
    ),
    Case(
        name="unbalanced_if_block",
        code="""
IF Motor_Current > 51.9 THEN
    Motor_Stop := TRUE;
""",
        requirement=NO_DELAY_REQ,
        expected_rules=["PARSE_ERROR"],
    ),
    Case(
        name="empty_output",
        code="",
        requirement=NO_DELAY_REQ,
        expected_rules=["PARSE_ERROR"],
    ),
    Case(
        name="multiple_faults_at_once",
        code="""
IF Motor_Amps < 25.0 THEN
    Overload_Alarm := TRUE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=["UNKNOWN_TAG", "NO_SHUTDOWN_ACTION", "MISSING_TIMER"],
        note="checker has to report all of them, not stop at the first",
    ),
    Case(
        name="named_constant_threshold",
        code="""
VAR CONSTANT
    TRIP_LEVEL : REAL := 51.9;
END_VAR
VAR
    Overload_Timer : TON;
END_VAR

Overload_Timer(IN := Motor_Current > TRIP_LEVEL, PT := T#3S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=[],
        note="threshold hidden behind a constant still has to be range checked",
    ),
    Case(
        name="comment_contains_bad_threshold",
        code="""
VAR
    Overload_Timer : TON;
END_VAR

(* trip level was 25.0 A before commissioning *)
Overload_Timer(IN := Motor_Current > 51.9, PT := T#3S); // was Motor_Amps > 9999.0
IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=[],
        note="an early version read commented out values as live code",
    ),
    # Everything below this line was added after the first live run against real
    # models. All of it is copied from what Gemini actually produced, and every
    # one of these shapes defeated the parser at first. I wrote the original
    # cases by hand and none of them used brackets around a condition, which is
    # the first thing all four models do. Hand written test inputs share the
    # blind spots of whoever wrote them.
    Case(
        name="live_bracketed_timer_condition",
        code="""
VAR
    Overload_Timer : TON;
END_VAR

Overload_Timer(IN := (Motor_Current > 51.9), PT := T#3S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=[],
        note="brackets truncated the argument list and the timer vanished entirely",
    ),
    Case(
        name="live_compound_timer_condition",
        code="""
VAR
  Overload_Timer : TON;
END_VAR

Overload_Timer(IN := Motor_Running AND (Motor_Current > 51.9), PT := T#3S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
    Overload_Alarm := TRUE;
END_IF;

IF Fault_Reset THEN
    Overload_Alarm := FALSE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=[],
        note="the interlock gated on Motor_Running as well, which is good practice",
    ),
    Case(
        name="live_two_stage_with_else",
        code="""
VAR
    Overload_Timer : TON;
END_VAR

IF Motor_Current > 45.65 THEN
    Safe_Mode_Active := TRUE;
    Overload_Alarm := TRUE;
ELSE
    Safe_Mode_Active := FALSE;
    Overload_Alarm := FALSE;
END_IF;

Overload_Timer(IN := (Motor_Current > 51.875), PT := T#3S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;

IF Fault_Reset THEN
    Motor_Stop := FALSE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=[],
        note="my own two stage thresholds, written back to me correctly",
    ),
    Case(
        name="live_timer_enabled_by_enclosing_if",
        code="""
VAR
    Stall_Timer : TON;
END_VAR

IF Start_Command AND Motor_Running AND (Motor_Speed < 50.0) THEN
    Stall_Timer(IN := TRUE, PT := T#5S);
ELSE
    Stall_Timer(IN := FALSE, PT := T#5S);
END_IF;

IF Stall_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;

IF Fault_Reset THEN
    Motor_Stop := FALSE;
END_IF;
""",
        requirement=Requirement(
            req_id="test_stall", text="Stall detection with a 5 second delay.",
            requires_delay=True, expected_signal="Motor_Speed",
        ),
        expected_rules=[],
        note="the trip condition is in the block around the timer, not in the call",
    ),
    Case(
        name="live_flag_mediated_shutdown",
        code="""
VAR
    Startup_Filter_Timer : TON;
END_VAR

Startup_Filter_Timer(IN := Motor_Running AND (Motor_Current > 41.5),
                     PT := T#3S);

IF Fault_Reset THEN
    Overload_Alarm := FALSE;
ELSIF Motor_Running AND NOT Safe_Mode_Active AND Startup_Filter_Timer.Q THEN
    Overload_Alarm := TRUE;
END_IF;

IF Overload_Alarm THEN
    Motor_Stop := TRUE;
ELSE
    Motor_Stop := NOT Start_Command;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=["THRESHOLD_OUT_OF_BAND"],
        note=(
            "two hops from comparison to contactor. 41.5 A is just under rated current, "
            "so it trips at normal full load"
        ),
    ),
    Case(
        name="live_reset_branch_is_not_an_inverted_trip",
        code="""
IF Motor_Winding_Temp > 140.0 THEN
    Motor_Stop := TRUE;
ELSIF Fault_Reset AND (Motor_Winding_Temp <= 140.0) THEN
    Motor_Stop := FALSE;
END_IF;
""",
        requirement=Requirement(
            req_id="test_reset", text="Shut down above 140 degC.",
            requires_delay=False, expected_signal="Motor_Winding_Temp",
        ),
        expected_rules=[],
        note=(
            "the <= is a reset interlock, not an inverted trip. Flagging it called good "
            "engineering dangerous, and every inverted comparison finding in the first "
            "live study was this same mistake"
        ),
    ),
    Case(
        name="live_hysteresis_deadband_is_not_inverted",
        code="""
VAR
    Overload_Timer : TON;
END_VAR

IF Motor_Current >= 45.65 THEN
    Safe_Mode_Active := TRUE;
ELSIF Fault_Reset AND Motor_Current < 45.65 THEN
    Safe_Mode_Active := FALSE;
END_IF;

Overload_Timer(IN := (Motor_Current >= 51.875) AND Motor_Running, PT := T#3S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=[],
        note="my own two stage logic with a de-latch, which must not read as inverted",
    ),
    Case(
        name="live_alias_then_timer",
        code="""
VAR
    Overload_Timer : TON;
END_VAR

Overload_Alarm := (Motor_Current > 51.9);

Overload_Timer(IN := Overload_Alarm AND Motor_Running, PT := T#30S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
ELSIF Fault_Reset THEN
    Motor_Stop := FALSE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=[],
        note=(
            "clean code, correct threshold, and a 30 second delay that lets a stalled "
            "rotor cook. Stage 2 has nothing to say about it"
        ),
    ),
    # The two below are the reason stage 3 exists. Both are accepted by every
    # rule in stage 2 and both destroy the motor.
    Case(
        name="physically_unsafe_but_structurally_perfect",
        code="""
VAR
    Overload_Timer : TON;
END_VAR

Overload_Timer(IN := Motor_Current > 62.3, PT := T#30S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=[],
        note="150 percent with a 30 second delay, both numbers reasonable, motor dead in 8 seconds on a stall",
    ),
    Case(
        name="current_interlock_blind_to_cooling_failure",
        code="""
VAR
    Overload_Timer : TON;
END_VAR

Overload_Timer(IN := Motor_Current > 51.9, PT := T#3S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
""",
        requirement=DELAY_REQ,
        expected_rules=[],
        note="textbook correct, and completely blind to a blocked fan cowl",
    ),
]
