"""
strudel.py -- pure Strudel-faithful pattern core, independent of any hardware.

Exports: Time, Span, Event, Pattern, ADSR, Note, Chord, pan buckets,
and the public pattern constructors (note, chord, seq, stack, cat, arrange …).
Nothing in this module depends on pyayay, numpy, or any sound chip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Callable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

Time = Fraction


def _frac(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(1_000_000)


@dataclass(frozen=True)
class Span:
    begin: Fraction
    end: Fraction

    def with_time(self, f: Callable[[Fraction], Fraction]) -> "Span":
        return Span(f(self.begin), f(self.end))

    def intersect(self, other: "Span") -> Optional["Span"]:
        b = max(self.begin, other.begin)
        e = min(self.end, other.end)
        if b >= e:
            return None
        return Span(b, e)


@dataclass(frozen=True)
class Event:
    """A value active over a span.

    ``whole`` is the event's *logical* span (e.g. the whole note), while
    ``part`` is the (possibly clipped) portion returned by a query.  The onset
    of an event is ``whole.begin``; we only trigger a note when its onset falls
    inside the queried frame, which keeps held/clipped fragments from
    re-triggering every frame.
    """

    whole: Optional[Span]
    part: Span
    value: object

    @property
    def has_onset(self) -> bool:
        return self.whole is not None and self.whole.begin == self.part.begin

    def with_value(self, f) -> "Event":
        return Event(self.whole, self.part, f(self.value))


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

QueryFn = Callable[[Span], List[Event]]


class Pattern:
    """A pattern is a function from a query span to a list of events."""

    def __init__(self, query: QueryFn):
        self.query = query

    # -- core query helper -------------------------------------------------

    def query_span(self, begin, end) -> List[Event]:
        return self.query(Span(_frac(begin), _frac(end)))

    # -- functor -----------------------------------------------------------

    def fmap(self, f) -> "Pattern":
        return Pattern(lambda span: [e.with_value(f) for e in self.query(span)])

    def with_value(self, f) -> "Pattern":
        return self.fmap(f)

    # -- time transforms ---------------------------------------------------

    def _with_query_time(self, f) -> "Pattern":
        return Pattern(lambda span: self.query(span.with_time(f)))

    def _with_event_time(self, f) -> "Pattern":
        def q(span):
            out = []
            for e in self.query(span):
                whole = e.whole.with_time(f) if e.whole else None
                out.append(Event(whole, e.part.with_time(f), e.value))
            return out
        return Pattern(q)

    def fast(self, factor) -> "Pattern":
        factor = _frac(factor)
        if factor == 0:
            return silence
        return (self._with_query_time(lambda t: t * factor)
                    ._with_event_time(lambda t: t / factor))

    def slow(self, factor) -> "Pattern":
        return self.fast(Fraction(1) / _frac(factor))

    def rev(self) -> "Pattern":
        def q(span: Span):
            out = []
            cyc = math.floor(span.begin)
            while cyc < span.end:
                cstart = Fraction(cyc)
                cend = cstart + 1
                qs = span.intersect(Span(cstart, cend))
                if qs is not None:
                    def reflect(t, cs=cstart, ce=cend):
                        return cs + (ce - t)
                    rb, re_ = reflect(qs.begin), reflect(qs.end)
                    inner = self.query(Span(min(rb, re_), max(rb, re_)))
                    for e in inner:
                        whole = e.whole.with_time(reflect) if e.whole else None
                        part = e.part.with_time(reflect)
                        if whole:
                            whole = Span(min(whole.begin, whole.end), max(whole.begin, whole.end))
                        part = Span(min(part.begin, part.end), max(part.begin, part.end))
                        out.append(Event(whole, part, e.value))
                cyc += 1
            return out
        return Pattern(q)

    def early(self, amount) -> "Pattern":
        amount = _frac(amount)
        return (self._with_query_time(lambda t: t + amount)
                    ._with_event_time(lambda t: t - amount))

    def late(self, amount) -> "Pattern":
        return self.early(-_frac(amount))

    def every(self, n: int, f: Callable[["Pattern"], "Pattern"]) -> "Pattern":
        """Apply ``f`` on every ``n``-th cycle (cycle 0, n, 2n ...)."""
        n = int(n)
        transformed = f(self)

        def q(span: Span):
            out = []
            cyc = math.floor(span.begin)
            while cyc < span.end:
                qs = span.intersect(Span(Fraction(cyc), Fraction(cyc + 1)))
                if qs is not None:
                    src = transformed if (cyc % n == 0) else self
                    out.extend(src.query(qs))
                cyc += 1
            return out
        return Pattern(q)

    def degrade_by(self, prob: float, seed: int = 0) -> "Pattern":
        """Randomly drop events with probability ``prob`` (deterministic)."""
        def q(span: Span):
            out = []
            for e in self.query(span):
                if not e.has_onset:
                    out.append(e)
                    continue
                h = hash((float(e.part.begin), seed)) & 0xFFFFFFFF
                if (h / 0xFFFFFFFF) >= prob:
                    out.append(e)
            return out
        return Pattern(q)

    # -- value transforms --------------------------------------------------

    def add(self, semitones) -> "Pattern":
        """Transpose notes (or add to numeric values) by ``semitones``."""
        def f(v):
            if isinstance(v, Note):
                return v.transpose(semitones)
            return v + semitones
        return self.fmap(f)

    def s(self, instrument) -> "Pattern":
        """Attach an instrument to every note in the pattern."""
        def f(v):
            n = _as_note(v)
            return replace(n, instrument=instrument)
        return self.fmap(f)

    def vol(self, v: int) -> "Pattern":
        return self.fmap(lambda x: replace(_as_note(x), volume=int(v)))

    def pan(self, p) -> "Pattern":
        """Place this pattern in a pan *bucket* -- ``"L"``/``"C"``/``"R"`` (left/
        centre/right), ``"A"`` (any free channel), or a number 0..1 snapped to
        the nearest bucket.

        ``p`` may also be a space-separated *mini-pattern* to automate the pan
        over a cycle (Strudel-style), e.g. ``.pan("L C R")`` sweeps the voice
        across the three positions; each event takes the pan active at its
        onset.
        """
        toks = str(p).split() if isinstance(p, str) else [p]
        if len(toks) <= 1:
            bucket = _pan_bucket(toks[0] if toks else p)
            return self.fmap(lambda x: replace(_as_note(x), pan=bucket))
        pan_pat = _fromList([_pan_bucket(t) for t in toks])

        def q(span: Span):
            out = []
            for e in self.query(span):
                onset = e.whole.begin if e.whole else e.part.begin
                vs = pan_pat.query(Span(onset, onset + Fraction(1, 1_000_000)))
                bucket = vs[0].value if vs else PAN_C
                out.append(e.with_value(lambda v, b=bucket: replace(_as_note(v), pan=b)))
            return out
        return Pattern(q)

    def priority(self, n: int) -> "Pattern":
        """Set voice-stealing priority (higher wins its pan bucket; ties broken
        by stack/document order)."""
        return self.fmap(lambda x: replace(_as_note(x), priority=int(n)))

    def sweep(self, semitones_per_cycle: float) -> "Pattern":
        return self.fmap(lambda x: replace(_as_note(x), sweep=float(semitones_per_cycle)))

    def vibrato(self, depth_semitones: float, rate_hz: float) -> "Pattern":
        return self.fmap(lambda x: replace(_as_note(x),
                                           vib_depth=float(depth_semitones),
                                           vib_rate=float(rate_hz)))

    # -- envelopes ---------------------------------------------------------

    def _patch_vol_env(self, **kw) -> "Pattern":
        def f(v):
            n = _as_note(v)
            base = n.vol_env or ADSR()
            return replace(n, vol_env=replace(base, **kw))
        return self.fmap(f)

    def attack(self, seconds: float) -> "Pattern":
        return self._patch_vol_env(a=float(seconds))

    def decay(self, seconds: float) -> "Pattern":
        return self._patch_vol_env(d=float(seconds))

    def sustain(self, level: float) -> "Pattern":
        """Sustain *level* in 0..1 (Strudel semantics -- not a duration)."""
        return self._patch_vol_env(s=float(level))

    def release(self, seconds: float) -> "Pattern":
        return self._patch_vol_env(r=float(seconds))

    def adsr(self, spec) -> "Pattern":
        """Set the amplitude ADSR from a Strudel ``"a:d:s:r"`` string or ADSR."""
        env = ADSR.parse(spec)
        return self.fmap(lambda x: replace(_as_note(x), vol_env=env))

    def penv(self, spec, peak: float = 12.0) -> "Pattern":
        """Set a *pitch* envelope (``peak`` semitones high).  ``spec`` is an
        ``"a:d:s:r"`` string or :class:`ADSR`; the sustain level scales the held
        pitch offset.  A short decay with ``s=0`` is the classic tracker
        pitch-blip on the note's attack."""
        env = ADSR.parse(spec, peak=peak)
        return self.fmap(lambda x: replace(_as_note(x), pitch_env=env))

    def ornament(self, *offsets) -> "Pattern":
        """Attach an AY-tracker *ornament*: a looping per-frame list of semitone
        offsets (``ornament(0, 4, 7)`` fakes a major chord on one channel -- the
        hardware-arpeggio trick materialised as a table rather than via
        ``.arp().fast()``)."""
        if len(offsets) == 1 and isinstance(offsets[0], (list, tuple)):
            offsets = tuple(offsets[0])
        orn = tuple(int(o) for o in offsets)
        return self.fmap(lambda x: replace(_as_note(x), ornament=orn))

    # -- structure ---------------------------------------------------------

    def struct(self, pattern_str: str) -> "Pattern":
        """Re-trigger this pattern's value on a boolean rhythm.

        ``"x ~ x x"`` -> hits on steps 0, 2, 3 of a 4-step cycle.
        """
        bools = _parse_struct(pattern_str)
        rhythm = _fromList(bools)
        return _app_left(rhythm, self, lambda b, v: (v if b else _REST))

    def arp(self, mode: str = "up") -> "Pattern":
        """Strudel-style arpeggiation.

        Expands a :class:`Chord` value held over an event's span into a
        sub-sequence of single notes *within that span*.  Sped up with
        ``.fast(n)`` this becomes the AY hardware-arpeggio chord trick.
        """
        def q(span: Span):
            out = []
            for e in self.query(span):
                notes = _chord_notes(e.value, mode)
                if not notes or e.whole is None:
                    out.append(e)
                    continue
                w = e.whole
                step = (w.end - w.begin) / len(notes)
                for i, nv in enumerate(notes):
                    nb = w.begin + step * i
                    ne = nb + step
                    sub_whole = Span(nb, ne)
                    part = sub_whole.intersect(e.part)
                    if part is not None:
                        out.append(Event(sub_whole, part, nv))
            return out
        return Pattern(q)

    def stutter(self, n: int) -> "Pattern":
        """Re-trigger each event ``n`` times within its own span (ratchet)."""
        n = int(n)

        def q(span: Span):
            out = []
            for e in self.query(span):
                if e.whole is None or n <= 1:
                    out.append(e)
                    continue
                w = e.whole
                step = (w.end - w.begin) / n
                for i in range(n):
                    nb = w.begin + step * i
                    sub = Span(nb, nb + step)
                    part = sub.intersect(e.part)
                    if part is not None:
                        out.append(Event(sub, part, e.value))
            return out
        return Pattern(q)

    def glitch(self, amount: float = 12.0, seed: int = 1) -> "Pattern":
        """Acid-glitch: jump each event's pitch by a deterministic pseudo-random
        amount in ``[-amount, +amount]`` semitones."""
        def f_event(e: Event) -> Event:
            onset = float(e.whole.begin) if e.whole else float(e.part.begin)
            h = hash((round(onset, 6), seed)) & 0xFFFFFFFF
            r = (h / 0xFFFFFFFF) * 2.0 - 1.0
            shift = round(r * amount)
            return e.with_value(lambda v: _as_note(v).transpose(shift))

        def q(span: Span):
            return [f_event(e) for e in self.query(span)]
        return Pattern(q)

    def slide(self, semitones_per_second: float) -> "Pattern":
        """Portamento glide: continuously bend each note's pitch."""
        return self.fmap(lambda x: replace(_as_note(x),
                                           sweep=float(semitones_per_second)))

    def gain(self, factor: float) -> "Pattern":
        """Scale note volume by ``factor`` (0..1+), clamped to 0..15.  Skips
        envelope-mode notes (volume == 16)."""
        def f(v):
            n = _as_note(v)
            if n.volume >= 16:
                return n
            return replace(n, volume=max(0, min(15, round(n.volume * factor))))
        return self.fmap(f)

    # -- combination -------------------------------------------------------

    def __or__(self, other: "Pattern") -> "Pattern":
        return stack(self, other)


# ---------------------------------------------------------------------------
# Primitive pattern constructors
# ---------------------------------------------------------------------------

silence = Pattern(lambda span: [])


class _Rest:
    """Sentinel for a rest produced by struct/mini-parsing."""
    __slots__ = ()


_REST = _Rest()


def pure(value) -> Pattern:
    """A pattern that repeats ``value`` once per cycle."""
    def q(span: Span):
        out = []
        cyc = math.floor(span.begin)
        while cyc < span.end:
            whole = Span(Fraction(cyc), Fraction(cyc + 1))
            part = whole.intersect(span)
            if part is not None:
                out.append(Event(whole, part, value))
            cyc += 1
        return out
    return Pattern(q)


def _fromList(values: Sequence) -> Pattern:
    """Lay ``values`` out evenly across one cycle (Strudel sequence)."""
    values = list(values)
    n = len(values)
    if n == 0:
        return silence

    def q(span: Span):
        out = []
        cyc = math.floor(span.begin)
        while cyc < span.end:
            base = Fraction(cyc)
            for i, v in enumerate(values):
                if v is _REST or v is None:
                    continue
                wb = base + Fraction(i, n)
                we = base + Fraction(i + 1, n)
                whole = Span(wb, we)
                part = whole.intersect(span)
                if part is not None:
                    out.append(Event(whole, part, v))
            cyc += 1
        return out
    return Pattern(q)


def seq(*values) -> Pattern:
    """Sequence of values across one cycle, e.g. ``seq("c4", "e4", "g4")``."""
    return _fromList([_parse_token(v) for v in values])


def stack(*patterns: Pattern) -> Pattern:
    """Play patterns simultaneously."""
    pats = list(patterns)

    def q(span: Span):
        out = []
        for p in pats:
            out.extend(p.query(span))
        return out
    return Pattern(q)


def cat(*patterns: Pattern) -> Pattern:
    """Concatenate patterns, one per cycle (Strudel ``cat``/``slowcat``)."""
    pats = list(patterns)
    n = len(pats)
    if n == 0:
        return silence

    def q(span: Span):
        out = []
        cyc = math.floor(span.begin)
        while cyc < span.end:
            idx = cyc % n
            p = pats[idx]
            qs = span.intersect(Span(Fraction(cyc), Fraction(cyc + 1)))
            if qs is not None:
                shift = Fraction(cyc) - Fraction(cyc // n)
                shifted = p._with_query_time(lambda t, s=shift: t - s)._with_event_time(lambda t, s=shift: t + s)
                out.extend(shifted.query(qs))
            cyc += 1
        return out
    return Pattern(q)


def _app_left(structure: Pattern, values: Pattern, combine) -> Pattern:
    """Combine two patterns, taking structure (timing) from ``structure`` and
    sampling ``values`` at each structural onset."""
    def q(span: Span):
        out = []
        for se in structure.query(span):
            sample_at = se.whole.begin if se.whole else se.part.begin
            vs = values.query(Span(sample_at, sample_at + Fraction(1, 1_000_000)))
            v = vs[0].value if vs else None
            combined = combine(se.value, v)
            if combined is _REST or combined is None:
                continue
            out.append(Event(se.whole, se.part, combined))
        return out
    return Pattern(q)


def window(pattern: Pattern, begin, end) -> Pattern:
    """Gate ``pattern`` so only events whose onset falls in ``[begin, end)``
    cycles survive.  The basic building block for song sections."""
    begin, end = _frac(begin), _frac(end)

    def q(span: Span):
        clip = span.intersect(Span(begin, end))
        if clip is None:
            return []
        out = []
        for e in pattern.query(clip):
            onset = e.whole.begin if e.whole else e.part.begin
            if begin <= onset < end:
                out.append(e)
        return out
    return Pattern(q)


def at(begin, pattern: Pattern) -> Tuple[Fraction, Pattern]:
    """Sugar for :func:`arrange` sections: ``at(8, drop)``."""
    return (_frac(begin), pattern)


def arrange(*sections, shift: bool = True) -> Pattern:
    """Place patterns at absolute cycle positions to build a song.

    Each section is ``(start_cycle, pattern)`` (use :func:`at`).  By default
    each pattern is *shifted* so its own cycle 0 lines up with ``start_cycle``;
    the section runs until the next section's start.

        arrange(at(0, intro), at(8, build), at(16, drop), at(32, outro))
    """
    secs = sorted(((_frac(s), p) for s, p in sections), key=lambda x: x[0])
    starts = [s for s, _ in secs]
    layers = []
    for i, (start, pat) in enumerate(secs):
        end = starts[i + 1] if i + 1 < len(starts) else None
        shifted = pat.late(start) if shift else pat
        layers.append(window(shifted, start, end if end is not None else Fraction(10 ** 9)))
    return stack(*layers)


def loop(pattern: Pattern, n: int) -> Pattern:
    """Identity for our cyclic patterns; kept for readability alongside
    :func:`arrange`/:func:`window`."""
    return pattern


# ---------------------------------------------------------------------------
# Notes, chords and pitch helpers
# ---------------------------------------------------------------------------

_NOTE_BASE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

A4_MIDI = 69
A4_FREQ = 440.0


def note_name_to_midi(name: str) -> int:
    name = name.strip().lower()
    if not name:
        raise ValueError("empty note name")
    letter = name[0]
    if letter not in _NOTE_BASE:
        return int(name)
    semis = _NOTE_BASE[letter]
    i = 1
    while i < len(name) and name[i] in "#sb":
        if name[i] in "#s":
            semis += 1
        else:
            semis -= 1
        i += 1
    octave = int(name[i:]) if i < len(name) else 4
    return (octave + 1) * 12 + semis


def midi_to_freq(midi: float) -> float:
    return A4_FREQ * (2.0 ** ((midi - A4_MIDI) / 12.0))


def note_to_midi_freq(freq: float) -> float:
    """Frequency back to a (fractional) MIDI number."""
    return A4_MIDI + 12.0 * math.log2(freq / A4_FREQ)


# ---------------------------------------------------------------------------
# ADSR envelopes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ADSR:
    """A Strudel-style ADSR envelope.

    Parameters
    ----------
    a, d, r : float       attack / decay / release **durations in seconds**.
    s : float             **sustain level** in 0..1 (held, not a duration).
    peak : float          the value the envelope reaches at the end of attack;
                          defaults to 1.0.  For a pitch envelope this is the
                          attack height in *semitones*; ``s`` scales the sustain.
    """

    a: float = 0.0
    d: float = 0.0
    s: float = 1.0
    r: float = 0.0
    peak: float = 1.0

    def at(self, t_in: float, dur: float) -> float:
        """Envelope value at ``t_in`` seconds into a note of length ``dur``."""
        sustain = self.s * self.peak
        if t_in < 0:
            return 0.0
        if t_in >= dur:
            if self.r <= 0:
                return 0.0
            rt = t_in - dur
            if rt >= self.r:
                return 0.0
            level_at_off = self._gated(dur, dur, sustain)
            return level_at_off * (1.0 - rt / self.r)
        return self._gated(t_in, dur, sustain)

    def _gated(self, t_in: float, dur: float, sustain: float) -> float:
        if self.a > 0 and t_in < self.a:
            return self.peak * (t_in / self.a)
        td = t_in - self.a
        if self.d > 0 and td < self.d:
            return self.peak + (sustain - self.peak) * (td / self.d)
        return sustain

    @classmethod
    def parse(cls, spec, peak: float = 1.0) -> "ADSR":
        """Parse a Strudel-style ``"a:d:s:r"`` string (or pass an ADSR through).

        Missing trailing fields default to 0 (a/d/r) or 1 (s).
        """
        if isinstance(spec, ADSR):
            return spec
        parts = [p.strip() for p in str(spec).split(":")]
        vals = [float(p) if p else None for p in parts]
        a = vals[0] if len(vals) > 0 and vals[0] is not None else 0.0
        d = vals[1] if len(vals) > 1 and vals[1] is not None else 0.0
        s = vals[2] if len(vals) > 2 and vals[2] is not None else 1.0
        r = vals[3] if len(vals) > 3 and vals[3] is not None else 0.0
        return cls(a=a, d=d, s=s, r=r, peak=peak)


# ---------------------------------------------------------------------------
# Chord intervals
# ---------------------------------------------------------------------------

_CHORD_INTERVALS = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "m": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}


# ---------------------------------------------------------------------------
# Pan buckets
#
# Three discrete logical routing slots -- LEFT / CENTRE / RIGHT -- plus ANY
# (wildcard).  These are pure routing concepts with no chip dependency; the
# physical channel mapping lives in ay_patterns.py.
# ---------------------------------------------------------------------------

PAN_L = "L"
PAN_C = "C"
PAN_R = "R"
PAN_ANY = "A"

_PAN_ALIASES = {
    "l": PAN_L, "left": PAN_L,
    "c": PAN_C, "centre": PAN_C, "center": PAN_C, "m": PAN_C, "mid": PAN_C,
    "r": PAN_R, "right": PAN_R,
    "a": PAN_ANY, "any": PAN_ANY, "*": PAN_ANY,
}


def _pan_bucket(value) -> str:
    """Coerce a pan spec to a bucket token (``L``/``C``/``R``/``A``).

    Accepts the tokens / aliases above, or a numeric 0..1 (snapped to the
    nearest of left/centre/right).
    """
    if value is None:
        return PAN_C
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _PAN_ALIASES:
            return _PAN_ALIASES[v]
        try:
            value = float(v)
        except ValueError:
            raise ValueError(f"unknown pan {value!r}")
    x = float(value)
    if x < 1.0 / 3.0:
        return PAN_L
    if x > 2.0 / 3.0:
        return PAN_R
    return PAN_C


# ---------------------------------------------------------------------------
# Note and Chord datatypes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Note:
    midi: float
    volume: int = 15
    pan: Optional[str] = None         # pan bucket: "L"/"C"/"R"/"A" (or None)
    instrument: Optional[object] = None
    sweep: float = 0.0
    vib_depth: float = 0.0
    vib_rate: float = 0.0
    vol_env: Optional["ADSR"] = None
    pitch_env: Optional["ADSR"] = None
    ornament: Optional[Tuple[int, ...]] = None
    priority: int = 0

    def transpose(self, semitones) -> "Note":
        return replace(self, midi=self.midi + semitones)

    @property
    def freq(self) -> float:
        return midi_to_freq(self.midi)


@dataclass(frozen=True)
class Chord:
    midis: Tuple[float, ...]
    template: Note = field(default_factory=lambda: Note(midi=0))

    def notes(self) -> List[Note]:
        return [replace(self.template, midi=m) for m in self.midis]


def _as_note(v) -> Note:
    if isinstance(v, Note):
        return v
    if isinstance(v, Chord):
        return replace(v.template, midi=v.midis[0])
    if isinstance(v, str):
        return _parse_token(v)
    if isinstance(v, (int, float)):
        return Note(midi=float(v))
    raise TypeError(f"cannot interpret {v!r} as a Note")


def _parse_token(tok):
    """Parse a single mini token into a Note / Chord / rest.

    Tokens: ``"c4"``, ``"c4:maj"`` (chord), ``"~"`` (rest), numbers (midi).
    """
    if isinstance(tok, (Note, Chord)):
        return tok
    if isinstance(tok, (int, float)):
        return Note(midi=float(tok))
    s = str(tok).strip()
    if s in ("~", "", "-"):
        return _REST
    if ":" in s:
        root, ctype = s.split(":", 1)
        intervals = _CHORD_INTERVALS.get(ctype.lower())
        if intervals is None:
            raise ValueError(f"unknown chord type {ctype!r}")
        base = note_name_to_midi(root)
        return Chord(tuple(base + i for i in intervals))
    return Note(midi=float(note_name_to_midi(s)))


def _chord_notes(value, mode: str) -> List[Note]:
    if isinstance(value, Chord):
        notes = value.notes()
    elif isinstance(value, Note):
        notes = [value]
    else:
        return []
    midis = [n.midi for n in notes]
    template = notes[0]
    order = _arp_order(midis, mode)
    return [replace(template, midi=m) for m in order]


def _arp_order(midis: Sequence[float], mode: str) -> List[float]:
    ms = list(midis)
    mode = mode.lower()
    if mode == "up":
        return sorted(ms)
    if mode == "down":
        return sorted(ms, reverse=True)
    if mode == "updown":
        up = sorted(ms)
        return up + up[-2:0:-1]
    if mode == "downup":
        down = sorted(ms, reverse=True)
        return down + down[-2:0:-1]
    if mode == "thumbup":
        up = sorted(ms)
        root = up[0]
        out = []
        for n in up[1:]:
            out += [root, n]
        return out or up
    return ms


def _parse_struct(s: str) -> List[bool]:
    return [tok == "x" or tok == "1" for tok in s.split()]


# ---------------------------------------------------------------------------
# Public note/chord pattern constructors
# ---------------------------------------------------------------------------

def note(spec) -> Pattern:
    """``note("c4 e4 g4 ~")`` or ``note(60)``.

    Space-separated tokens become a one-cycle sequence.
    """
    if isinstance(spec, (int, float, Note, Chord)):
        return pure(_parse_token(spec))
    toks = str(spec).split()
    if len(toks) == 1:
        return pure(_parse_token(toks[0]))
    return _fromList([_parse_token(t) for t in toks])


def chord(spec) -> Pattern:
    """``chord("c4:maj e4:min")`` -> a sequence of chords."""
    return note(spec)
