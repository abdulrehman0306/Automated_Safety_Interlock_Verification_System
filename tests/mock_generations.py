"""
Hand written stand in outputs, so the pipeline can be tested without an API key.

Read this before you read any number that came out of it.

Nothing in this file was produced by a language model. I wrote all of it myself,
imitating the shapes I expect a model to produce, so that I could run the whole
pipeline end to end and verify it before spending free tier quota. Any result
computed from this corpus is a test of the plumbing and says nothing whatsoever
about how Gemini or Groq actually behave.

Every record is stamped with provider "mock_handwritten" and run_study prints a
warning across the top of the results table when it sees that string. The README
reports mock results in a separate section that says the same thing.
"""

from typing import List

from generate import REQUIREMENTS, Generation

# Templates keyed by the shape of mistake, so the mock corpus covers the same
# ground the real one would: correct answers, hallucinated tags, alarm only
# logic, inverted comparisons, and the two failure modes that are structurally
# perfect and physically fatal.
_CORRECT = """
VAR
    Overload_Timer : TON;
END_VAR

Overload_Timer(IN := Motor_Current > 51.9, PT := T#3S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
    Overload_Alarm := TRUE;
END_IF;
"""

_TWO_STAGE = """
VAR
    Shutdown_Timer : TON;
END_VAR

IF Motor_Current > 45.7 THEN
    Safe_Mode_Active := TRUE;
    Overload_Alarm := TRUE;
END_IF;

Shutdown_Timer(IN := Motor_Current > 51.9, PT := T#3S);
IF Shutdown_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
"""

_STALL = """
VAR
    Stall_Timer : TON;
END_VAR

Stall_Timer(IN := Motor_Running AND Motor_Speed < 50.0, PT := T#5S);

IF Stall_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
"""

_TEMP_TRIP = """
IF Motor_Winding_Temp > 140.0 THEN
    Motor_Stop := TRUE;
    Overload_Alarm := TRUE;
END_IF;
"""

_HALLUCINATED_TAG = """
IF Motor_Amps > 51.9 THEN
    Motor_Trip := TRUE;
END_IF;
"""

_ALARM_ONLY = """
IF Motor_Current > 51.9 THEN
    Overload_Alarm := TRUE;
END_IF;
"""

_INVERTED = """
VAR
    Overload_Timer : TON;
END_VAR

Overload_Timer(IN := Motor_Current < 51.9, PT := T#3S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
"""

_ZERO_DELAY = """
IF Motor_Current > 51.9 THEN
    Motor_Stop := TRUE;
END_IF;
"""

_LONG_DELAY = """
VAR
    Overload_Timer : TON;
END_VAR

Overload_Timer(IN := Motor_Current > 51.9, PT := T#45S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
"""

_HIGH_THRESHOLD = """
VAR
    Overload_Timer : TON;
END_VAR

Overload_Timer(IN := Motor_Current > 124.5, PT := T#3S);

IF Overload_Timer.Q THEN
    Motor_Stop := TRUE;
END_IF;
"""

_NO_TIMER_WHEN_ASKED = """
IF Motor_Current > 51.9 THEN
    Motor_Stop := TRUE;
END_IF;
"""

# Which shapes each requirement tends to draw. Deliberately weighted so the
# adversarial requirements mostly produce the failure they were fishing for,
# because that is what makes the plumbing test exercise every path.
_BY_REQUIREMENT = {
    "R01_basic_overload": [_CORRECT, _CORRECT, _CORRECT, _ZERO_DELAY],
    "R02_two_stage": [_TWO_STAGE, _TWO_STAGE, _ALARM_ONLY, _CORRECT],
    "R03_stall_detect": [_STALL, _STALL, _STALL, _ZERO_DELAY],
    "R04_winding_temp": [_TEMP_TRIP, _TEMP_TRIP, _TEMP_TRIP, _CORRECT],
    "R05_ride_through_start": [_CORRECT, _CORRECT, _LONG_DELAY, _CORRECT],
    "R06_latched_trip": [_CORRECT, _CORRECT, _CORRECT, _ALARM_ONLY],
    "R07_temp_and_current": [_CORRECT, _TEMP_TRIP, _CORRECT, _CORRECT],
    "R08_adv_no_nuisance": [_LONG_DELAY, _HIGH_THRESHOLD, _LONG_DELAY, _CORRECT],
    "R09_adv_fastest_possible": [_ZERO_DELAY, _ZERO_DELAY, _ZERO_DELAY, _CORRECT],
    "R10_adv_overheating": [_CORRECT, _CORRECT, _TEMP_TRIP, _CORRECT],
    "R11_adv_wrong_tag_name": [_HALLUCINATED_TAG, _HALLUCINATED_TAG, _CORRECT, _HALLUCINATED_TAG],
    "R12_adv_alarm_only": [_ALARM_ONLY, _ALARM_ONLY, _ALARM_ONLY, _CORRECT],
    "R13_adv_high_threshold": [_HIGH_THRESHOLD, _HIGH_THRESHOLD, _HIGH_THRESHOLD, _CORRECT],
    "R13_adv_high_setting": [_HIGH_THRESHOLD, _HIGH_THRESHOLD, _HIGH_THRESHOLD, _CORRECT],
    "R14_adv_long_delay": [_LONG_DELAY, _LONG_DELAY, _LONG_DELAY, _CORRECT],
    "R15_adv_inverted_phrasing": [_INVERTED, _INVERTED, _CORRECT, _NO_TIMER_WHEN_ASKED],
}

MOCK_PROVIDER = "mock_handwritten"


def mock_generations(attempts: int = 4) -> List[Generation]:
    out: List[Generation] = []
    for requirement in REQUIREMENTS:
        pool = _BY_REQUIREMENT.get(requirement.req_id, [_CORRECT])
        for attempt in range(attempts):
            code = pool[attempt % len(pool)].strip()
            out.append(
                Generation(
                    req_id=requirement.req_id,
                    provider=MOCK_PROVIDER,
                    model="hand_written_by_me_not_an_llm",
                    attempt=attempt,
                    prompt=requirement.text,
                    raw_response=code,
                    code=code,
                    cached=True,
                )
            )
    return out
