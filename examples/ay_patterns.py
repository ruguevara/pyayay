"""
ay_patterns.py -- a Strudel-flavoured, fluent pattern language for the AY/YM
sound chip, built on top of the ``pyayay`` C-extension.

This is an *exploration* module (a single self-contained file), not a packaged
library.  It shows how the AY-3-8910 / YM2149 can be driven like a tiny modular
synth from declarative, composable patterns -- including the chip's signature
tricks:

  * tone + envelope "buzzer" / PWM timbres (the envelope generator used as a
    second oscillator, à la a two-oscillator synth);
  * noise + tone percussion;
  * per-frame pitch / period sweeps and vibrato;
  * Strudel-style ``.arp()`` -- which, sped up, *is* the classic AY hardware
    arpeggio "chord" trick.

Design, in three layers
-----------------------

1. ``Pattern`` -- a Strudel-faithful pattern is a *function of time*:
   ``query(begin, end) -> [Event]`` over a cyclic timeline (1 cycle = 1 bar by
   convention).  Transforms (``fast``, ``slow``, ``rev``, ``every``, ``arp``,
   ``add`` ...) build new patterns from old ones.  Nothing is sampled to a grid
   until render time.

2. Instruments (``Buzzer``, ``Tone``, ``Percussion`` ...) -- turn an event's
   value (a ``Note`` carrying pitch + synth params) into per-frame AY register
   state, evaluating any sweeps / vibrato along the way.

3. ``render`` -- walks a fixed PSG frame grid (default 50 Hz, configurable),
   queries the pattern per frame, voice-allocates events onto the chip's three
   channels, and produces a ``[frames, 14]`` uint8 PSG array.  That array is fed
   straight to ``pyayay.Ayumi.render_psg`` to synthesise audio, and can also be
   written out as a ``.wav``.

Example
-------
    from ay_patterns import *

    bass  = note("c2 ~ c2 g1").s(Buzzer(shape="saw"))
    arp   = chord("c4:maj e4:min").arp("updown").fast(8).s(Tone())
    drums = stack(
        s(Percussion("kick")).struct("x ~ x ~"),
        s(Percussion("hat")).fast(8),
    )

    song = stack(bass, arp, drums)
    render(song, bpm=125, seconds=8, wav="out.wav")
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

import pyayay
from pyayay import EnvShape


# ---------------------------------------------------------------------------
# Time
#
# We follow Strudel: time is measured in *cycles* (one cycle == one bar by
# convention) and is rational, so that fast/slow/struct subdivisions stay exact.
# A "span" is a half-open interval [begin, end) of cycle-time.
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
            # allow zero-width only if both are the same instant (rare); else None
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
            # reflect each cycle the span touches around its centre
            out = []
            cyc = math.floor(span.begin)
            while cyc < span.end:
                cstart = Fraction(cyc)
                cend = cstart + 1
                qs = span.intersect(Span(cstart, cend))
                if qs is not None:
                    def reflect(t, cs=cstart, ce=cend):
                        return cs + (ce - t)
                    # reflecting swaps begin/end ordering; normalise the query
                    # span so the inner pattern still sees begin < end.
                    rb, re_ = reflect(qs.begin), reflect(qs.end)
                    inner = self.query(Span(min(rb, re_), max(rb, re_)))
                    for e in inner:
                        whole = e.whole.with_time(reflect) if e.whole else None
                        part = e.part.with_time(reflect)
                        # reflection flips begin/end ordering -> normalise
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

    def s(self, instrument: "Instrument") -> "Pattern":
        """Attach an instrument to every note in the pattern."""
        def f(v):
            note = _as_note(v)
            return replace(note, instrument=instrument)
        return self.fmap(f)

    def vol(self, v: int) -> "Pattern":
        return self.fmap(lambda x: replace(_as_note(x), volume=int(v)))

    def pan(self, p: float) -> "Pattern":
        return self.fmap(lambda x: replace(_as_note(x), pan=float(p)))

    def sweep(self, semitones_per_cycle: float) -> "Pattern":
        return self.fmap(lambda x: replace(_as_note(x), sweep=float(semitones_per_cycle)))

    def vibrato(self, depth_semitones: float, rate_hz: float) -> "Pattern":
        return self.fmap(lambda x: replace(_as_note(x),
                                           vib_depth=float(depth_semitones),
                                           vib_rate=float(rate_hz)))

    # -- structure ---------------------------------------------------------

    def struct(self, pattern_str: str) -> "Pattern":
        """Re-trigger this pattern's value on a boolean rhythm.

        ``"x ~ x x"`` -> hits on steps 0, 2, 3 of a 4-step cycle.
        """
        bools = _parse_struct(pattern_str)
        rhythm = _fromList(bools)
        # take the structure (timing) from `rhythm`, value from self
        return _app_left(rhythm, self, lambda b, v: (v if b else _REST))

    def arp(self, mode: str = "up") -> "Pattern":
        """Strudel-style arpeggiation.

        Expands a :class:`Chord` value held over an event's span into a
        sub-sequence of single notes *within that span*.  Sped up with
        ``.fast(n)`` this becomes the AY hardware-arpeggio chord trick (cycling
        chord notes once per frame to fake a chord on a single channel).
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
        amount in ``[-amount, +amount]`` semitones.  Pair with ``.stutter()`` /
        ``.fast()`` for the classic chip 'data corruption' fill."""
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
        """Portamento glide: continuously bend each note's pitch.  (Alias for a
        per-second ``sweep`` -- the engine's ``sweep`` is already per-second.)"""
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
                # shift so the chosen pattern's cycle `cyc` plays here
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
            # sample values at the structural event's onset
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
    cycles survive (the pattern keeps running on its own timeline -- this just
    masks it).  The basic building block for song sections."""
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
    each pattern is *shifted* so its own cycle 0 lines up with ``start_cycle``
    (so a 4-cycle loop written from cycle 0 plays correctly wherever you drop
    it); the section runs until the next section's start.

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
    """Identity for our cyclic patterns (they already repeat every cycle); kept
    for readability when expressing "loop this for n cycles" intent alongside
    :func:`arrange`/:func:`window`."""
    return pattern


# ---------------------------------------------------------------------------
# Notes, chords and parsing
# ---------------------------------------------------------------------------

_NOTE_BASE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

# MIDI note number for which we anchor pitch; A4 = 69 = 440 Hz.
A4_MIDI = 69
A4_FREQ = 440.0


def note_name_to_midi(name: str) -> int:
    name = name.strip().lower()
    if not name:
        raise ValueError("empty note name")
    letter = name[0]
    if letter not in _NOTE_BASE:
        # allow raw midi numbers
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
    """Inverse of :func:`midi_to_freq`: a frequency back to a (fractional) MIDI
    number, so a detune in semitones can be applied around it."""
    return A4_MIDI + 12.0 * math.log2(freq / A4_FREQ)


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


@dataclass(frozen=True)
class Note:
    midi: float
    volume: int = 15
    pan: Optional[float] = None
    instrument: Optional["Instrument"] = None
    sweep: float = 0.0          # semitones per cycle
    vib_depth: float = 0.0      # semitones
    vib_rate: float = 0.0       # Hz

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
        # collapse to root if used where a single note is expected
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
    if mode == "thumbup":  # 0,n: pedal root then each note
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


def s(instrument: "Instrument") -> Pattern:
    """A bare instrument trigger (value carries only the instrument).

    Useful for drums: ``s(Percussion("kick")).struct("x ~ x ~")``.
    Defaults to a mid note so tone/noise instruments have a pitch.
    """
    return pure(replace(Note(midi=note_name_to_midi("a3")), instrument=instrument))


# ---------------------------------------------------------------------------
# AY frequency math (verified empirically against the emulator)
#
#   tone_period = clock / (16 * freq)
#   envelope ramp frequency = clock / (256 * env_period)
#       -> a saw envelope (/|/|) sounds one cycle per ramp:  clock/(256*P)
#       -> a triangle envelope (/\/\) sounds one cycle per 2 ramps: clock/(512*P)
# ---------------------------------------------------------------------------

def tone_period_for_freq(freq: float, clock: float) -> int:
    if freq <= 0:
        return 0
    return max(1, min(0xFFF, round(clock / (16.0 * freq))))


def env_period_for_freq(freq: float, clock: float, ramps_per_cycle: int = 2) -> int:
    """``ramps_per_cycle``: 2 for triangle shapes, 1 for saw shapes."""
    if freq <= 0:
        return 0
    return max(1, min(0xFFFF, round(clock / (256.0 * ramps_per_cycle * freq))))


# ---------------------------------------------------------------------------
# AY register frame model
#
# A "voice state" is what one logical voice wants the chip to do this frame.
# The renderer allocates voices to the 3 hardware channels and assembles the
# 14-byte PSG register frame.
# ---------------------------------------------------------------------------

@dataclass
class VoiceState:
    tone_on: bool = False
    noise_on: bool = False
    env_on: bool = False
    tone_period: int = 0
    volume: int = 0                 # 0..15 (or env-mode flag handled separately)
    pan: float = 0.5
    # chip-global (envelope + noise are shared resources on the AY):
    env_period: Optional[int] = None
    env_shape: Optional[int] = None
    noise_period: Optional[int] = None


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

class Instrument:
    """Maps a triggered Note + time-into-note to a VoiceState for a frame.

    ``frame_state`` is called once per frame while a note is sounding.
        note     : the Note value
        t_in     : seconds since the note's onset
        dur      : the note's total duration in seconds
        clock    : chip clock in Hz
    """

    def frame_state(self, note: Note, t_in: float, dur: float, clock: float) -> VoiceState:
        raise NotImplementedError

    # shared helper: pitch with sweep + vibrato applied
    @staticmethod
    def _live_freq(note: Note, t_in: float, dur: float) -> float:
        midi = note.midi
        if note.sweep:
            # cycles elapsed unknown here; sweep is expressed per *second* of note
            midi += note.sweep * t_in
        if note.vib_depth and note.vib_rate:
            midi += note.vib_depth * math.sin(2 * math.pi * note.vib_rate * t_in)
        return midi_to_freq(midi)


class Tone(Instrument):
    """Plain square-wave tone channel."""

    def __init__(self, volume: int = 15):
        self.volume = volume

    def frame_state(self, note, t_in, dur, clock):
        f = self._live_freq(note, t_in, dur)
        vol = note.volume if note.volume is not None else self.volume
        return VoiceState(
            tone_on=True,
            tone_period=tone_period_for_freq(f, clock),
            volume=vol,
            pan=note.pan if note.pan is not None else 0.5,
        )


class Buzzer(Instrument):
    """The AY "buzzer": the envelope generator used as a pitched oscillator.

    The envelope runs as a repeating saw/triangle wave *at the note pitch*; that
    is the actual sound source (the chip's "two oscillator per channel" trick).

    Two clean modes -- use the :func:`buzz_bass`, :func:`pwm` and
    :func:`ringmod` factories rather than juggling flags by hand:

    * ``tone=False`` (default) -- *pure* buzzer: only the envelope sounds.  A
      clean, bright sawtooth/triangle bass.  This is what you want for a buzz
      bass.
    * ``tone=True`` with a non-zero ``detune`` / ``env_ratio`` -- the square
      tone and the envelope oscillator at *different* pitches, beating against
      each other for PWM / ring-mod colours.

    Avoid ``tone=True`` with ``detune=0`` and ``env_ratio=1``: the square and
    the envelope then sit at the *same* frequency with an undefined relative
    phase and partly cancel into a dirty, unpitched buzz.

    Parameters
    ----------
    shape : "saw" | "saw_down" | "tri"   envelope waveform
    detune : float    envelope pitch offset in semitones vs the tone
    env_ratio : float multiply the envelope frequency (0.5 = octave below)
    tone : bool       also sound the square tone (True) or pure buzzer (False)
    """

    _SHAPES = {
        "saw": (EnvShape.UP_UP_C, 1),           # /|/| repeating up-ramp
        "saw_down": (EnvShape.DOWN_DOWN_8, 1),  # \|\|
        "tri": (EnvShape.UP_DOWN_E, 2),         # /\/\
    }

    def __init__(self, shape: str = "saw", detune: float = 0.0,
                 env_ratio: float = 1.0, tone: bool = False):
        self.shape_enum, self.ramps = self._SHAPES[shape]
        self.detune = detune
        self.env_ratio = env_ratio
        self.tone = tone

    def frame_state(self, note, t_in, dur, clock):
        f = self._live_freq(note, t_in, dur)
        # the envelope oscillator carries the pitch (optionally detuned)
        env_f = midi_to_freq(note_to_midi_freq(f) + self.detune) * self.env_ratio
        env_p = env_period_for_freq(env_f, clock, ramps_per_cycle=self.ramps)
        # the square tone, if used, plays the note pitch
        tone_p = tone_period_for_freq(f, clock)
        return VoiceState(
            tone_on=self.tone,
            env_on=True,
            tone_period=tone_p,
            volume=16,  # 16 == "use envelope" (the R8 bit-4 flag)
            pan=note.pan if note.pan is not None else 0.5,
            env_period=env_p,
            env_shape=int(self.shape_enum),
        )


def buzz_bass(shape: str = "saw") -> Buzzer:
    """A clean pure-envelope buzzer bass (no square tone)."""
    return Buzzer(shape=shape, tone=False)


def pwm(shape: str = "saw", detune: float = 12.0) -> Buzzer:
    """Buzzer + square tone an octave apart by default -> a fat doubled timbre.

    (``detune`` is the envelope's offset from the tone in semitones; +12 puts
    the buzzer an octave above the square, which reads as a bright fat bass
    rather than the phase-cancelling mush of a near-zero detune.)
    """
    return Buzzer(shape=shape, detune=detune, tone=True)


def ringmod(shape: str = "tri", interval: float = 7.0) -> Buzzer:
    """Buzzer + square tone an interval apart -> ring-mod / metallic colour."""
    return Buzzer(shape=shape, detune=interval, tone=True)


class Percussion(Instrument):
    """Noise (+ optional tone) percussion with a fast period sweep.

    kind : "kick" | "snare" | "hat" | "tom"
    """

    _PRESETS = {
        # (use_tone, start_noise_period, end_noise_period, decay_s, base_freq)
        "kick":  (True,  4,  20, 0.12, 60.0),
        "snare": (True,  6,  14, 0.16, 180.0),
        "hat":   (False, 1,  3,  0.05, 0.0),
        "tom":   (True,  8,  16, 0.18, 120.0),
    }

    def __init__(self, kind: str = "kick"):
        (self.use_tone, self.np0, self.np1,
         self.decay, self.base_freq) = self._PRESETS[kind]

    def frame_state(self, note, t_in, dur, clock):
        frac = min(1.0, t_in / self.decay) if self.decay > 0 else 1.0
        vol = max(0, round(15 * (1.0 - frac)))
        noise_p = round(self.np0 + (self.np1 - self.np0) * frac)
        tone_p = 0
        if self.use_tone and self.base_freq > 0:
            # pitch drop for the body of the drum
            f = self.base_freq * (1.0 - 0.6 * frac)
            tone_p = tone_period_for_freq(f, clock)
        return VoiceState(
            tone_on=self.use_tone and self.base_freq > 0,
            noise_on=True,
            tone_period=tone_p,
            volume=vol,
            pan=note.pan if note.pan is not None else 0.5,
            noise_period=max(1, min(31, noise_p)),
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _assemble_frame(voices: List[VoiceState]) -> Tuple[np.ndarray, List[float]]:
    """Build a 14-byte PSG register frame from up to 3 voice states.

    Returns (regs[14] uint8, per-channel pan list).  Envelope and noise are
    chip-global; if multiple voices request them, the last one wins (a real
    limitation of the hardware -- worth being honest about in an explorer).
    """
    regs = np.zeros(14, dtype=np.uint8)
    pans = [0.5, 0.5, 0.5]

    mixer = 0  # bits: tone A/B/C (0-2), noise A/B/C (3-5); 0 = ON (active low)
    env_period = 0
    env_shape = 0
    noise_period = 0

    for ch, v in enumerate(voices[:3]):
        if v is None:
            mixer |= (1 << ch) | (1 << (ch + 3))  # both off
            continue
        # tone period (fine/coarse)
        regs[ch * 2] = v.tone_period & 0xFF
        regs[ch * 2 + 1] = (v.tone_period >> 8) & 0x0F
        # volume / envelope-mode
        if v.volume >= 16:
            regs[8 + ch] = 0x10  # envelope mode (bit 4 set, level bits ignored)
        else:
            regs[8 + ch] = v.volume & 0x0F
        # mixer (active low: clear bit to enable)
        if not v.tone_on:
            mixer |= (1 << ch)
        if not v.noise_on:
            mixer |= (1 << (ch + 3))
        pans[ch] = v.pan
        if v.env_period is not None:
            env_period = v.env_period
        if v.env_shape is not None:
            env_shape = v.env_shape
        if v.noise_period is not None:
            noise_period = v.noise_period

    regs[6] = noise_period & 0x1F
    regs[7] = mixer & 0x3F
    regs[11] = env_period & 0xFF
    regs[12] = (env_period >> 8) & 0xFF
    regs[13] = env_shape & 0x0F
    return regs, pans


def _render_grid(
    pattern: Pattern,
    n_frames: int,
    frames_per_cycle: float,
    fps: float,
    clock: float,
    poly_order: str = "low",
):
    """Core grid sampler.  Returns (psg[n_frames,14] uint8, pans list)."""
    psg = np.zeros((n_frames, 14), dtype=np.uint8)
    pan_track: List[List[float]] = []

    sec_per_frame = 1.0 / fps
    cyc_per_frame = 1.0 / frames_per_cycle

    # Each frame we re-query the pattern; a note's "time into note" (used for
    # sweeps / vibrato / drum decay) is derived from its onset (whole.begin).
    for fi in range(n_frames):
        cyc_begin = Fraction(fi) * _frac(cyc_per_frame)
        cyc_end = Fraction(fi + 1) * _frac(cyc_per_frame)
        events = pattern.query(Span(cyc_begin, cyc_end))

        # collect voices that have a note sounding in this frame
        frame_voices: List[Tuple[float, VoiceState]] = []  # (midi for sorting, state)

        for e in events:
            note = _as_note(e.value)
            if note.instrument is None:
                note = replace(note, instrument=_DEFAULT_INSTRUMENT)
            whole = e.whole if e.whole else e.part
            onset_sec = float(whole.begin) * frames_per_cycle * sec_per_frame
            now_sec = fi * sec_per_frame
            t_in = max(0.0, now_sec - onset_sec)
            dur = float(whole.end - whole.begin) * frames_per_cycle * sec_per_frame
            state = note.instrument.frame_state(note, t_in, dur, clock)
            frame_voices.append((note.midi, state))

        # voice allocation onto <=3 channels
        if poly_order == "high":
            frame_voices.sort(key=lambda x: -x[0])
        else:
            frame_voices.sort(key=lambda x: x[0])
        voices: List[Optional[VoiceState]] = [None, None, None]
        for i, (_, st) in enumerate(frame_voices[:3]):
            voices[i] = st

        regs, pans = _assemble_frame(voices)
        psg[fi] = regs
        pan_track.append(pans)

    return psg, pan_track


_DEFAULT_INSTRUMENT = Tone()


def render(
    pattern: Pattern,
    bpm: float = 120.0,
    beats_per_cycle: int = 4,
    seconds: Optional[float] = None,
    cycles: Optional[float] = None,
    cps: Optional[float] = None,
    fps: float = 50.0,
    sample_rate: int = 44100,
    clock: float = 1773400.0,
    chip_type=pyayay.ChipType.AY,
    master_volume: float = 0.4,
    wav: Optional[str] = None,
    poly_order: str = "low",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a pattern to audio (and optionally a WAV file).

    Tempo: set ``bpm`` + ``beats_per_cycle`` (musician-friendly), or pass
    ``cps`` directly (Strudel cycles-per-second).  Duration: give ``seconds``
    or ``cycles``.

    Returns ``(psg, left, right)`` -- the PSG register array and the two float32
    audio buffers.
    """
    if cps is None:
        cps = bpm / 60.0 / beats_per_cycle
    frames_per_cycle = fps / cps

    if seconds is None:
        if cycles is None:
            cycles = 4.0
        seconds = cycles / cps
    n_frames = int(round(seconds * fps))
    if n_frames <= 0:
        raise ValueError("nothing to render (duration is zero)")

    psg, pan_track = _render_grid(
        pattern, n_frames, frames_per_cycle, fps, clock, poly_order
    )

    # Pans can change per frame; pyayay applies pan via set_pan (not a register).
    # We render in pan-stable segments so per-channel panning is honoured.
    ay = pyayay.Ayumi(sample_rate=sample_rate, clock=clock, type=chip_type)
    ay.set_master_volume(master_volume)

    samples = int(math.ceil(n_frames / fps * sample_rate)) + sample_rate
    left = np.zeros(samples, dtype=np.float32)
    right = np.zeros(samples, dtype=np.float32)

    # render_psg consumes the whole frame array with an (inverted) mask
    # (True == "do not write this register this frame").  We write R0-R12 every
    # frame, but *suppress R13 (envelope shape) when it is unchanged*: writing
    # R13 retriggers the envelope generator, resetting the buzzer sawtooth phase
    # to 0.  Re-latching it every frame would chop any buzzer tone into a 50 Hz
    # buzz, so we only write R13 when the shape actually changes.
    mask = np.zeros_like(psg, dtype=bool)
    if n_frames > 1:
        r13 = psg[:, 13]
        unchanged = r13[1:] == r13[:-1]
        mask[1:, 13] = unchanged

    # apply a representative pan (first frame's) -- per-frame pan would require
    # segmenting; for the explorer we set pans from the first non-silent frame.
    pans = _pick_pans(pan_track)
    for ch in range(3):
        ay.set_pan(ch, pans[ch])

    ay.render_psg(psg, mask, left, right, fps)

    used = _frames_to_samples(n_frames, fps, sample_rate)
    left = left[:used]
    right = right[:used]

    if wav is not None:
        write_wav(wav, left, right, sample_rate)

    return psg, left, right


def _pick_pans(pan_track: List[List[float]]) -> List[float]:
    if not pan_track:
        return [0.25, 0.75, 0.5]
    return pan_track[0]


def _frames_to_samples(n_frames: int, fps: float, sample_rate: int) -> int:
    return int(round(n_frames / fps * sample_rate))


def write_wav(path: str, left: np.ndarray, right: np.ndarray, sample_rate: int):
    """Write stereo float buffers to a 16-bit PCM WAV file."""
    n = min(len(left), len(right))
    l = np.clip(left[:n], -1.0, 1.0)
    r = np.clip(right[:n], -1.0, 1.0)
    inter = np.empty(n * 2, dtype=np.int16)
    inter[0::2] = (l * 32767).astype(np.int16)
    inter[1::2] = (r * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(inter.tobytes())
