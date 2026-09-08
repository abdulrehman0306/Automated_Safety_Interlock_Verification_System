"""
Stage 2. Deterministic checking of generated Structured Text. No learning here.

Scope of this layer was a decision I had to make deliberately, and it changes
what the whole study measures. Every rule in this file looks at one thing at a
time: does this name exist, is this single number inside a sane band, is there a
shutdown at all, is there a timer, is this comparison pointing the right way.

What I explicitly did not do is write rules that reason about combinations, for
example "a threshold above 300 percent must have a delay under 5 seconds". Rules
like that encode the physics, and if I put them here then stage 3 has nothing
left to catch and the comparison between the two layers becomes a measure of how
hard I tried when writing rules. The boundary is the experiment. It only means
something if I hold it.

The parser is not a real IEC 61131-3 front end. It handles the subset the models
actually emit, which turns out to be plain IF blocks, TON instances and boolean
assignments. When it cannot make sense of something it says so through a
PARSE_ERROR finding rather than guessing, because a checker that quietly fails
open is worse than no checker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fault_sim import InterlockParams
from tag_list import (
    MOTOR,
    NON_SHUTDOWN_OUTPUTS,
    RATED_CURRENT_A,
    SHUTDOWN_OUTPUTS,
    TAGS,
)

# Reserved words and standard library names the checker should not mistake for
# undeclared variables.
ST_KEYWORDS = {
    "IF", "THEN", "ELSIF", "ELSE", "END_IF", "CASE", "OF", "END_CASE",
    "FOR", "TO", "BY", "DO", "END_FOR", "WHILE", "END_WHILE", "REPEAT",
    "UNTIL", "END_REPEAT", "VAR", "VAR_INPUT", "VAR_OUTPUT", "VAR_IN_OUT",
    "VAR_TEMP", "VAR_GLOBAL", "CONSTANT", "END_VAR", "FUNCTION_BLOCK",
    "END_FUNCTION_BLOCK", "PROGRAM", "END_PROGRAM", "FUNCTION", "END_FUNCTION",
    "TRUE", "FALSE", "AND", "OR", "NOT", "XOR", "MOD", "RETURN", "EXIT",
    "BOOL", "INT", "DINT", "REAL", "LREAL", "TIME", "WORD", "BYTE", "STRING",
    "TON", "TOF", "TP", "R_TRIG", "F_TRIG", "RS", "SR", "CTU", "CTD",
    "ABS", "SQRT", "MIN", "MAX", "LIMIT", "SEL", "MOVE", "AT", "END_STRUCT",
    "STRUCT", "TYPE", "END_TYPE", "IN", "PT", "Q", "ET", "PV", "CV",
}

# Legal band for a single threshold, judged on its own with no reference to the
# rest of the interlock. Anything outside these is wrong no matter what else the
# code says, which is what makes it a fair rule for this layer.
THRESHOLD_BANDS: Dict[str, Tuple[float, float, str]] = {
    "Motor_Current": (
        RATED_CURRENT_A,
        RATED_CURRENT_A * MOTOR.locked_rotor_multiple,
        "at or below rated current it trips on normal full load, above locked rotor "
        "current it can never trip",
    ),
    "Motor_Winding_Temp": (
        60.0,
        155.0,
        "below 60 degC it trips on a warm day, above the class F limit the insulation is already gone",
    ),
    "Motor_Speed": (
        0.0,
        1500.0,
        "outside the encoder range the comparison is meaningless",
    ),
}

COMPARISON_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_.]*)\s*(>=|<=|<>|>|<|=)\s*([A-Za-z_][A-Za-z0-9_.]*|-?\d+\.?\d*)"
)
ASSIGNMENT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*:=\s*([^;]+);")
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TIME_LITERAL_RE = re.compile(r"(?:T|TIME)#([0-9hmsHMS._]+)", re.IGNORECASE)
FB_CALL_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def find_calls(text: str) -> List[Tuple[str, str, int, int]]:
    """Find NAME(...) calls, counting brackets instead of trusting a regex.

    This replaced a regex that used [^)]* for the argument list, and the
    difference is not cosmetic. Every model in the study writes conditions
    wrapped in brackets:

        Overload_Timer(IN := (Motor_Current > 51.9), PT := T#3S);

    The old pattern stopped at the inner bracket, so the arguments came back
    truncated before PT, the timer was never registered at all, the .Q in the
    guard chain had nothing to resolve to, and the extractor concluded there was
    no interlock in a piece of code that was completely correct. It did that to
    28 of 60 real generations.

    My own test corpus never caught it because I wrote every case by hand
    without brackets. The lesson is that hand written test inputs share the
    blind spots of the person who wrote them.

    Returns name, argument text, and the span of the whole call.
    """
    calls: List[Tuple[str, str, int, int]] = []
    for match in FB_CALL_NAME_RE.finditer(text):
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
            index += 1
        if depth == 0:
            calls.append((match.group(1), text[match.end() : index - 1], match.start(), index))
    return calls


def split_top_level(args: str) -> List[str]:
    """Split an argument list on commas that are not inside brackets."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for char in args:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [p for p in parts if p.strip()]


@dataclass
class Requirement:
    """A plain language requirement, plus what a correct answer has to contain.

    requires_delay and expected_signal are the only pieces of intent the rule
    checker gets. Everything else it has to work out from the code, which is the
    honest situation: in practice nobody hands a verifier a machine readable
    spec, they hand it a sentence.
    """

    req_id: str
    text: str
    requires_delay: bool = False
    expected_signal: Optional[str] = None
    adversarial: bool = False
    note: str = ""


@dataclass
class Finding:
    rule_id: str
    message: str


@dataclass
class ParsedCode:
    """What the parser managed to understand about a block of Structured Text."""

    identifiers: List[str] = field(default_factory=list)
    declared_locals: Dict[str, str] = field(default_factory=dict)
    constants: Dict[str, float] = field(default_factory=dict)
    comparisons: List[Tuple[str, str, str]] = field(default_factory=list)
    timers: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # assignment target -> list of guard condition strings that reach it
    guarded_assignments: List[Tuple[str, str, List[str]]] = field(default_factory=list)
    # name -> the expression assigned to it, for Trip_Cond := Motor_Current > 51.9
    expr_aliases: Dict[str, str] = field(default_factory=dict)
    # name -> the guard chain under which it gets set TRUE
    true_conditions: Dict[str, str] = field(default_factory=dict)
    unbalanced_blocks: bool = False
    stripped: str = ""


def strip_comments(src: str) -> str:
    """Remove (* block *) and // line comments.

    Done before anything else because the models like to put example values in
    comments, and an early version of this checker was reading a commented out
    threshold as if it were live code.
    """
    src = re.sub(r"\(\*.*?\*\)", " ", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", " ", src)
    return src


def parse_time_literal(text: str) -> Optional[float]:
    """Turn T#3S or T#1M30S or T#500MS into seconds."""
    match = TIME_LITERAL_RE.search(text)
    if not match:
        return None
    body = match.group(1).lower()
    total = 0.0
    found = False
    # Milliseconds first, otherwise the "m" in "ms" gets eaten as minutes.
    for value, unit in re.findall(r"(\d+\.?\d*)\s*(ms|h|m|s|d)", body):
        found = True
        scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[unit]
        total += float(value) * scale
    if not found:
        return None
    return total


def parse_structured_text(src: str) -> ParsedCode:
    """Walk the code once and pull out everything the rules need.

    Line based rather than a proper grammar. I tried writing a real recursive
    descent parser first and abandoned it, because the models emit a narrow and
    fairly tidy subset and the extra machinery was buying nothing. If this ever
    has to handle CASE statements or nested function blocks, that decision needs
    revisiting.
    """
    parsed = ParsedCode()
    text = strip_comments(src)
    parsed.stripped = text

    # Local declarations. Anything declared in a VAR block is a legitimate name
    # even though it is not in my tag list.
    # The optional CONSTANT is not cosmetic. VAR CONSTANT is how the models
    # declare a named trip level, and without it here the declaration was
    # missed, the constant looked like an undeclared variable, and the threshold
    # never got range checked at all. That is a checker failing open, which is
    # the worst way for it to fail.
    for block in re.findall(
        r"\bVAR(?:_\w+)?\b\s*(?:CONSTANT\b)?(.*?)\bEND_VAR\b", text, re.DOTALL | re.IGNORECASE
    ):
        for line in block.split(";"):
            decl = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", line)
            if decl:
                parsed.declared_locals[decl.group(1)] = decl.group(2).upper()
                # Constants get their value recorded so a threshold written as a
                # named constant can still be range checked.
                value = re.search(r":=\s*(-?\d+\.?\d*)", line)
                if value:
                    parsed.constants[decl.group(1)] = float(value.group(1))

    body = re.sub(r"\bVAR(?:_\w+)?\b.*?\bEND_VAR\b", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Time literals have to come out before names are collected, otherwise T#3S
    # reads as an identifier T and an identifier S and every correct piece of
    # code gets flagged for two undeclared variables. Cost me a confused twenty
    # minutes the first time the known bad tests ran.
    ident_source = TIME_LITERAL_RE.sub(" ", body)
    parsed.identifiers = IDENTIFIER_RE.findall(ident_source)
    parsed.comparisons = [(a, op, b) for a, op, b in COMPARISON_RE.findall(body)]

    # Timer instances. Both call styles the models use are handled: the
    # condition passed straight into IN, and IN := TRUE inside an IF block.
    for name, args, _, _ in find_calls(body):
        if name.upper() in {"IF", "WHILE", "UNTIL"}:
            continue
        pt = parse_time_literal(args)
        if pt is None and "PT" not in args.upper():
            continue
        in_cond = ""
        for argument in split_top_level(args):
            key, sep, value = argument.partition(":=")
            if sep and key.strip().upper() == "IN":
                in_cond = value.strip()
                # Strip one layer of wrapping brackets so the condition reads
                # the same whether or not the model bracketed it.
                if in_cond.startswith("(") and in_cond.endswith(")"):
                    in_cond = in_cond[1:-1].strip()
        # A timer can be called more than once, typically once per arm of an
        # IF. Never let a later trivial condition overwrite a real one.
        previous = str(parsed.timers.get(name, {}).get("in_cond", "")).strip().upper()
        if previous not in {"", "TRUE", "1", "FALSE", "0"} and in_cond.strip().upper() in {
            "TRUE", "1", "FALSE", "0", ""
        }:
            in_cond = str(parsed.timers[name]["in_cond"])
        parsed.timers[name] = {
            "pt_s": pt if pt is not None else 0.0,
            "in_cond": in_cond,
        }

    # Guard chain tracking. Walk statement by statement, keeping a stack of the
    # conditions currently in force, so that when an assignment shows up we know
    # what had to be true to reach it.
    stack: List[str] = []
    depth = 0
    statements = re.split(r"(?i)(\bIF\b|\bELSIF\b|\bELSE\b|\bEND_IF\b)", body)
    pending_cond: Optional[str] = None
    for chunk in statements:
        token = chunk.strip().upper()
        if token == "IF":
            pending_cond = "IF"
            depth += 1
            continue
        if token == "ELSIF":
            if stack:
                stack.pop()
            pending_cond = "IF"
            continue
        if token == "ELSE":
            if stack:
                last = stack.pop()
                stack.append(f"NOT ({last})")
            continue
        if token == "END_IF":
            depth -= 1
            if stack:
                stack.pop()
            continue

        if pending_cond == "IF":
            cond, _, rest = chunk.partition("THEN") if "THEN" in chunk.upper() else (chunk, "", "")
            if not _:
                # No THEN found. Either a malformed block or a construct this
                # parser does not know about, and either way the guard chain
                # from here on is not trustworthy.
                parsed.unbalanced_blocks = True
                cond, rest = chunk, ""
            stack.append(cond.strip())
            pending_cond = None
            chunk = rest

        # A timer called as Timer(IN := TRUE, PT := T#5S) inside an IF takes its
        # real enable condition from the block around it, not from the call.
        # That has to be picked up here, during the guard walk, because this is
        # the only place the enclosing conditions are known. Three real
        # generations wrote their stall detection exactly this way.
        for name, args, _, _ in find_calls(chunk):
            if name not in parsed.timers or ":=" not in args:
                continue
            for argument in split_top_level(args):
                key, sep, value = argument.partition(":=")
                if not sep or key.strip().upper() != "IN":
                    continue
                if value.strip().upper() in {"TRUE", "1"} and stack:
                    existing = str(parsed.timers[name]["in_cond"]).strip().upper()
                    # FALSE counts as trivial as well as TRUE. The ELSE arm of
                    # the same block passes IN := FALSE, and since it is parsed
                    # second it had already overwritten the condition, which is
                    # why this fix did nothing the first time I tried it.
                    if existing in {"", "TRUE", "1", "FALSE", "0"}:
                        parsed.timers[name]["in_cond"] = " AND ".join(f"({g})" for g in stack)

        # Named arguments inside a function block call look exactly like
        # assignments, so Overload_Timer(IN := ..., PT := ...) was showing up as
        # an assignment to a variable called IN. Only calls whose brackets
        # contain := are stripped, so ordinary function calls on the right hand
        # side of a real assignment survive. Removed back to front so the spans
        # stay valid as the string shortens.
        for _, args, start, end in reversed(find_calls(chunk)):
            if ":=" in args:
                chunk = chunk[:start] + " " + chunk[end:]

        for target, value in ASSIGNMENT_RE.findall(chunk):
            parsed.guarded_assignments.append((target, value.strip(), list(stack)))

    if depth != 0:
        parsed.unbalanced_blocks = True

    # Two ways a model hides the comparison behind a name instead of putting it
    # in the guard, and both showed up in real generations:
    #
    #   Trip_Cond := (Motor_Current > 51.9);   IF Trip_Cond THEN ...
    #   IF Timer.Q THEN Overload_Alarm := TRUE; END_IF;  IF Overload_Alarm THEN ...
    #
    # The first is an expression alias. The second is a flag set TRUE under some
    # other guard chain. Without resolving both, the extractor sees a guard made
    # of one bare identifier, finds no comparison anywhere, and reports that
    # there is no interlock in code that is perfectly sound.
    literals = {"TRUE", "FALSE", "0", "1"}
    for target, value, guards in parsed.guarded_assignments:
        clean = value.strip().upper()
        if clean in literals:
            if clean == "TRUE" and guards and target not in parsed.true_conditions:
                parsed.true_conditions[target] = " AND ".join(f"({g})" for g in guards)
        elif target not in parsed.expr_aliases:
            parsed.expr_aliases[target] = value.strip()

    return parsed


def _expand_aliases(expr: str, parsed: ParsedCode, depth: int = 4) -> str:
    """Substitute named booleans back into an expression.

    Bounded rather than recursive because two flags can reference each other,
    and a fixed point is not guaranteed on code nobody has type checked.
    """
    for _ in range(depth):
        before = expr
        for source in (parsed.expr_aliases, parsed.true_conditions):
            for name, value in source.items():
                pattern = rf"\b{re.escape(name)}\b"
                # Skip self reference, otherwise Latch := Latch OR X grows
                # without limit until the depth cap saves us.
                if re.search(pattern, value):
                    continue
                if re.search(pattern, expr):
                    expr = re.sub(pattern, f"({value})", expr)
        if expr == before:
            break
    return expr


def _resolve_guards(
    guards: List[str], parsed: ParsedCode
) -> Tuple[List[str], float]:
    """Replace any timer .Q reference with the condition that feeds the timer.

    Without this the extractor sees a shutdown guarded by Overload_Timer.Q and
    concludes there is no threshold anywhere, which is the most common shape the
    models produce and would have made stage 3 useless.
    """
    resolved: List[str] = []
    delay = 0.0
    for guard in guards:
        # Expand names first so that a timer .Q hidden inside a flag becomes
        # visible, then resolve the timer, then expand again because the timer
        # condition can itself be written in terms of another flag.
        expanded = _expand_aliases(guard, parsed)
        for name, info in parsed.timers.items():
            # Test the expanded text, not the original guard. When the timer is
            # reached through a flag, as in IF Timer.Q THEN Alarm := TRUE
            # followed by IF Alarm THEN Motor_Stop := TRUE, the .Q only appears
            # after the alias has been substituted in.
            if re.search(rf"\b{re.escape(name)}\s*\.\s*Q\b", expanded, re.IGNORECASE):
                delay = max(delay, float(info["pt_s"]))
                in_cond = str(info["in_cond"])
                # IN := TRUE means the timer is enabled by the enclosing IF, so
                # the useful condition is elsewhere in the chain and there is
                # nothing to substitute.
                if in_cond and in_cond.strip().upper() not in {"TRUE", "1"}:
                    expanded = re.sub(
                        rf"\b{re.escape(name)}\s*\.\s*Q\b", f"({in_cond})", expanded,
                        flags=re.IGNORECASE,
                    )
        resolved.append(_expand_aliases(expanded, parsed))

    # A timer that is fed by an IF condition rather than by IN still contributes
    # its preset to the delay, so pick it up even when no .Q appears in a guard.
    if delay == 0.0 and parsed.timers:
        for info in parsed.timers.values():
            delay = max(delay, float(info["pt_s"]))
    return resolved, delay


def extract_parameters(src: str, parsed: Optional[ParsedCode] = None) -> Optional[InterlockParams]:
    """Reduce generated code to the parameters stage 3 can simulate.

    Returns None when there is no comparison against a measured signal anywhere,
    which means the code is not an interlock in any recognisable sense.
    """
    parsed = parsed or parse_structured_text(src)

    has_shutdown = any(
        target in SHUTDOWN_OUTPUTS and value.strip().upper() in {"TRUE", "1"}
        for target, value, _ in parsed.guarded_assignments
    )

    # Prefer the guard chain that leads to a shutdown. Fall back to alarm only
    # logic, because "it raised an alarm and left the motor running" is a real
    # answer that stage 3 should be allowed to judge rather than skip.
    candidates = [
        (target, guards)
        for target, value, guards in parsed.guarded_assignments
        if target in SHUTDOWN_OUTPUTS and value.strip().upper() in {"TRUE", "1"}
    ] or [
        (target, guards)
        for target, value, guards in parsed.guarded_assignments
        if target in NON_SHUTDOWN_OUTPUTS and value.strip().upper() in {"TRUE", "1"}
    ]

    guard_pool: List[str] = []
    delay = 0.0
    for _, guards in candidates:
        resolved, guard_delay = _resolve_guards(guards, parsed)
        guard_pool.extend(resolved)
        delay = max(delay, guard_delay)

    if not guard_pool:
        # Nothing guarded anything. Fall back to any comparison in the file so a
        # bare condition still yields parameters.
        guard_pool = [parsed.stripped]
        _, delay = _resolve_guards([""], parsed)

    # Search order matters. Current is the signal these requirements are mostly
    # about, so if the code compares several things, the current comparison is
    # the one that defines the interlock.
    for signal in ("Motor_Current", "Motor_Winding_Temp", "Motor_Speed"):
        for guard in guard_pool:
            for lhs, op, rhs in COMPARISON_RE.findall(guard):
                if lhs != signal:
                    continue
                value = _numeric_value(rhs, parsed)
                if value is None:
                    continue
                direction = "high" if op in {">", ">="} else "low"
                return InterlockParams(
                    signal=signal,
                    threshold=value,
                    direction=direction,
                    delay_s=delay,
                    has_shutdown=has_shutdown,
                )
    return None


def _numeric_value(token: str, parsed: ParsedCode) -> Optional[float]:
    try:
        return float(token)
    except ValueError:
        return parsed.constants.get(token)


def check_code(src: str, requirement: Requirement) -> List[Finding]:
    """Run every deterministic rule. An empty list means stage 2 accepted it."""
    findings: List[Finding] = []
    parsed = parse_structured_text(src)

    if not src.strip():
        return [Finding("PARSE_ERROR", "generation produced no code at all")]

    if parsed.unbalanced_blocks:
        findings.append(
            Finding("PARSE_ERROR", "IF and END_IF do not balance, or an IF has no THEN")
        )

    # Rule 1. Every referenced name has to exist.
    seen = set()
    for name in parsed.identifiers:
        if name in seen or name.upper() in ST_KEYWORDS:
            continue
        seen.add(name)
        if name in TAGS or name in parsed.declared_locals:
            continue
        if re.fullmatch(r"[0-9].*", name):
            continue
        findings.append(
            Finding("UNKNOWN_TAG", f"'{name}' is not in the tag list and is not declared locally")
        )

    # Rule 2. Each threshold judged on its own against a static band.
    for lhs, op, rhs in parsed.comparisons:
        if lhs not in THRESHOLD_BANDS:
            continue
        value = _numeric_value(rhs, parsed)
        if value is None:
            continue
        low, high, why = THRESHOLD_BANDS[lhs]
        # Strict at the bottom. A trip set exactly at rated current fires during
        # normal full load running, so it belongs on the wrong side of the line
        # rather than sitting on it. Motor_Speed is the exception, since a stall
        # trip at zero rpm is a legitimate thing to write.
        below = value <= low if lhs != "Motor_Speed" else value < low
        if below or value > high:
            findings.append(
                # Spelled out rather than printed as a range, because the low
                # bound is exclusive and "41.5, outside 41.5 to 249" reads like
                # a bug even when the finding is correct.
                Finding(
                    "THRESHOLD_OUT_OF_BAND",
                    f"{lhs} compared against {value:g}, which has to be above {low:g} "
                    f"and no more than {high:g}, {why}",
                )
            )

    # Rule 3. Something has to actually stop the machine.
    stops = [
        target
        for target, value, _ in parsed.guarded_assignments
        if target in SHUTDOWN_OUTPUTS and value.strip().upper() in {"TRUE", "1"}
    ]
    if not stops:
        alarms = [
            target
            for target, value, _ in parsed.guarded_assignments
            if target in NON_SHUTDOWN_OUTPUTS
        ]
        detail = (
            f"only sets {sorted(set(alarms))}, which does not drop the contactor"
            if alarms
            else "no output is written at all"
        )
        findings.append(Finding("NO_SHUTDOWN_ACTION", f"nothing sets Motor_Stop, {detail}"))

    # Rule 4. If the requirement asked for a delay there had better be a timer.
    if requirement.requires_delay:
        has_timer = bool(parsed.timers) or "TON" in src.upper()
        if not has_timer:
            findings.append(
                Finding("MISSING_TIMER", "requirement asked for a time delay and no timer is present")
            )
        elif parsed.timers and all(float(t["pt_s"]) <= 0.0 for t in parsed.timers.values()):
            findings.append(
                Finding("MISSING_TIMER", "a timer instance exists but its preset time is zero or missing")
            )

    # Rule 5. Comparison direction. Checked separately from everything else
    # because an inverted comparison passes every syntax check, produces code
    # that looks completely normal, and is the most dangerous single thing on
    # this list. A trip written as current below threshold does not protect the
    # motor, it shuts the machine down when it is idle and leaves it running
    # when it is burning.
    #
    # It only judges the comparison that actually drives the shutdown, and that
    # restriction is the whole rule. The first version checked every comparison
    # in the file and had a 100 percent false positive rate on real code,
    # because models write a reset branch:
    #
    #   IF Motor_Winding_Temp > 140.0 THEN Motor_Stop := TRUE;
    #   ELSIF Fault_Reset AND (Motor_Winding_Temp <= 140.0) THEN Motor_Stop := FALSE;
    #
    # The trip is correct and the <= is a perfectly sensible interlock on
    # resetting before the winding has cooled. Flagging it called good
    # engineering dangerous, and worse, it let the rule checker take credit for
    # catching genuinely unsafe outputs for a reason that was not true.
    params = extract_parameters(src, parsed)
    if params is not None:
        tag = TAGS.get(params.signal)
        if tag is not None and tag.trip_direction and params.direction != tag.trip_direction:
            findings.append(
                Finding(
                    "INVERTED_COMPARISON",
                    f"the trip on {params.signal} fires when it goes {params.direction}, "
                    f"but this tag is dangerous when it goes {tag.trip_direction}",
                )
            )

    return findings


def summarise(findings: List[Finding]) -> str:
    if not findings:
        return "accepted"
    return "; ".join(f"{f.rule_id}: {f.message}" for f in findings)
