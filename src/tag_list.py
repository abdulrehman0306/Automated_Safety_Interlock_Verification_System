"""
The tag list and equipment data a real PLC program would have sitting behind it.

Everything in here is either a value I set myself on machines at the plant in
Karachi, or a nameplate figure for the class of motor those machines used. The
LLM is given a rendering of this file in its prompt, and the rule checker uses
the same data as ground truth. Keeping one source for both matters, because if
the prompt and the checker drift apart then the checker starts failing the model
for things the model was never told.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Tag:
    """One PLC variable.

    trip_direction is the field that earns its place here. You cannot tell
    statically that "IF Motor_Current < 52.0 THEN stop" is wrong unless you know
    this tag is supposed to trip on a rising value. Low water on a boiler trips
    low, motor overcurrent trips high, and an inverted comparison passes every
    syntax check while doing the exact opposite of its job. So the direction
    lives with the tag, not buried in the checker.
    """

    name: str
    datatype: str
    unit: str
    description: str
    trip_direction: Optional[str] = None  # "high", "low", or None for non analogue tags
    plausible_min: Optional[float] = None
    plausible_max: Optional[float] = None


@dataclass(frozen=True)
class MotorSpec:
    """Nameplate and thermal data for the drive motor.

    22 kW is the size of the largest drive I commissioned. Rated current is
    calculated rather than read off a plate, so treat it as approximate. Every
    threshold in the study is expressed as a multiple of it anyway, so a couple
    of percent either way changes nothing.
    """

    name: str
    rated_power_kw: float
    voltage_v: float
    efficiency: float
    power_factor: float
    insulation_class: str
    service_factor: float
    locked_rotor_multiple: float   # times rated current, straight off a DOL start
    stall_withstand_s: float       # how long it survives locked rotor starting hot
    start_inrush_s: float          # how long a healthy start draws inrush current
    winding_tau_running_s: float   # thermal time constant with the fan turning
    rated_temp_rise_k: float       # steady winding rise above ambient at rated load

    @property
    def rated_current_a(self) -> float:
        """Standard three phase current calculation, rounded to one decimal.

        The rounding is not cosmetic and it was not there originally. The prompt
        shows the model "rated current 41.5 A" because it is formatted to one
        decimal, while the rule checker was comparing against the full
        41.508843. So a model that used exactly the number I gave it got flagged
        for tripping below rated current, and the finding printed as "41.5,
        outside 41.5 to 249.1". Every threshold finding in the first live study
        was this artifact.

        One number everywhere fixes it. A nameplate would say 41.5 anyway, and
        the 0.02 percent difference changes no physics.
        """
        exact = (self.rated_power_kw * 1000.0) / (
            1.7320508 * self.voltage_v * self.power_factor * self.efficiency
        )
        return round(exact, 1)


# The motor the whole study is built around.
MOTOR = MotorSpec(
    name="M401_Conveyor_Drive",
    rated_power_kw=22.0,
    voltage_v=400.0,
    efficiency=0.90,
    power_factor=0.85,
    insulation_class="F",
    service_factor=1.15,
    locked_rotor_multiple=6.0,
    stall_withstand_s=10.0,
    start_inrush_s=1.5,
    winding_tau_running_s=420.0,
    rated_temp_rise_k=70.0,
)

RATED_CURRENT_A = MOTOR.rated_current_a

# Insulation limit. Class F is 155 degC at the hotspot, and the simulator calls
# the motor damaged when the winding node crosses it.
INSULATION_LIMIT_C = 155.0

# How well the motor gets rid of its heat, as a multiplier on thermal
# resistance. I started with a second thermal time constant for the stalled case
# and threw it away, because in a single node replica the initial heating rate
# already works out independent of cooling. Cooling only sets where the
# temperature settles and how fast it gets there in the long run, which is
# exactly the physical story. One number per condition is easier to defend than
# two time constants I would have had to invent.
COOLING_EFF_RUNNING = 1.00
COOLING_EFF_STALLED = 0.15   # shaft stopped, no self ventilation at all
COOLING_EFF_FAN_BLOCKED = 0.35  # turning, but the cowl is packed with lint

# The two thresholds I actually configured in TIA Portal. Safe mode at 110
# percent derates the machine and raises an alarm so somebody can go and look at
# it. Full shutdown at 125 percent. Both quoted as percentages of rated current
# because that is how they were entered.
SAFE_MODE_PERCENT = 110.0
SHUTDOWN_PERCENT = 125.0
SAFE_MODE_CURRENT_A = RATED_CURRENT_A * SAFE_MODE_PERCENT / 100.0
SHUTDOWN_CURRENT_A = RATED_CURRENT_A * SHUTDOWN_PERCENT / 100.0


TAGS = {
    t.name: t
    for t in [
        Tag(
            name="Motor_Current",
            datatype="REAL",
            unit="A",
            description="Measured stator current on the conveyor drive, from the CT on phase L1",
            trip_direction="high",
            plausible_min=0.0,
            plausible_max=RATED_CURRENT_A * MOTOR.locked_rotor_multiple,
        ),
        Tag(
            name="Motor_Winding_Temp",
            datatype="REAL",
            unit="degC",
            description="Stator winding temperature from the embedded PT100",
            trip_direction="high",
            plausible_min=0.0,
            plausible_max=200.0,
        ),
        Tag(
            name="Motor_Speed",
            datatype="REAL",
            unit="rpm",
            description="Shaft speed from the incremental encoder",
            trip_direction="low",
            plausible_min=0.0,
            plausible_max=1500.0,
        ),
        Tag(
            name="Motor_Running",
            datatype="BOOL",
            unit="",
            description="Contactor auxiliary contact, TRUE when the motor is energised",
        ),
        Tag(
            name="Start_Command",
            datatype="BOOL",
            unit="",
            description="Operator start request from the HMI",
        ),
        Tag(
            name="Motor_Stop",
            datatype="BOOL",
            unit="",
            description="Shutdown output. Setting this TRUE drops the main contactor",
        ),
        Tag(
            name="Safe_Mode_Active",
            datatype="BOOL",
            unit="",
            description="Reduced speed mode. Machine keeps running but derated",
        ),
        Tag(
            name="Overload_Alarm",
            datatype="BOOL",
            unit="",
            description="Alarm lamp and HMI banner. Does not stop anything on its own",
        ),
        Tag(
            name="Fault_Reset",
            datatype="BOOL",
            unit="",
            description="Operator acknowledgement, latched faults clear on rising edge",
        ),
        Tag(
            name="Overload_Timer",
            datatype="TON",
            unit="",
            description="On delay timer instance available for trip delays",
        ),
    ]
}

# Names the models reach for that do not exist here. Kept as a named list
# because a hallucinated tag is the easiest failure to catch and I want the
# README to be able to say how often it happened.
COMMON_HALLUCINATED_TAGS = [
    "MotorCurrent",
    "Motor_Amps",
    "Current_Actual",
    "Motor_Temperature",
    "EmergencyStop",
    "Motor_Trip",
    "Overcurrent_Flag",
]

# Any output that actually stops the machine. An interlock that only sets
# Overload_Alarm has protected nothing, and that is a mistake the models make
# often enough to deserve its own rule.
SHUTDOWN_OUTPUTS = {"Motor_Stop"}
NON_SHUTDOWN_OUTPUTS = {"Overload_Alarm", "Safe_Mode_Active"}

# Signals an interlock can sensibly watch. Used by the parameter extractor to
# decide which measured variable the generated logic is keyed on.
MEASURED_SIGNALS = {"Motor_Current", "Motor_Winding_Temp", "Motor_Speed"}


def tag_list_for_prompt() -> str:
    """Render the tag list the way it goes into the LLM prompt.

    Deliberately plain text. I tried a JSON version first and both models
    started replying with JSON instead of Structured Text, which cost me a whole
    batch of generations before I spotted why.
    """
    lines = [
        f"Equipment: {MOTOR.name}, {MOTOR.rated_power_kw} kW induction motor, "
        f"{MOTOR.voltage_v} V, rated current {RATED_CURRENT_A:.1f} A, "
        f"insulation class {MOTOR.insulation_class}, service factor "
        f"{MOTOR.service_factor}.",
        "",
        "Available tags, and these are the only names that exist:",
    ]
    for tag in TAGS.values():
        unit = f" [{tag.unit}]" if tag.unit else ""
        lines.append(f"  {tag.name} : {tag.datatype}{unit}  // {tag.description}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(tag_list_for_prompt())
    print()
    print(f"Rated current       : {RATED_CURRENT_A:.2f} A")
    print(f"Safe mode at {SAFE_MODE_PERCENT:.0f} pct : {SAFE_MODE_CURRENT_A:.2f} A")
    print(f"Shutdown at {SHUTDOWN_PERCENT:.0f} pct  : {SHUTDOWN_CURRENT_A:.2f} A")
    print(f"Locked rotor        : {RATED_CURRENT_A * MOTOR.locked_rotor_multiple:.2f} A")
