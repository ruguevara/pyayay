"""
ay_roll.py -- a tracker-style text view of a rendered PSG stream.

The renderer in ``ay_patterns`` produces a ``[frames, 14]`` uint8 PSG array --
exactly the register writes that drive the chip.  This module turns that array
(the ground truth) back into a vertical, tracker-like score so you can *read*
what the three channels are doing: note names, the tone/noise/envelope mixer
bits, volume, and for buzzer voices the envelope's note name + detune.

It's a debugging / inspection aid (a sibling to ``ay_analyze.py``): the analyser
turns a render into loudness/brightness/harmony numbers; this turns it into a
pattern you can eyeball for timing, voicing and arpeggio behaviour.

Layout (classic AY tracker: three channel columns A/B/C, rows down the page):

    row|  A: note oct TNE vol  |  B: ...            |  C: ...
    ---+------------------------+--- ...
    000|  C-3 ... T.. F         |  ...

Per channel each row shows, tracker-style (a field is printed only when it
*changes* -- held values blank out, like a tracker's empty rows):

    note    note name from the tone period (``C-3``, ``A#4`` ...), ``===`` on
            note-off (channel silent), blank when held.
    TNE     the mixer / envelope flags: ``T`` tone on, ``N`` noise on, ``E``
            envelope (buzzer) on -- a dot where off.
    vol     0..F hex amplitude, or ``Env`` when the channel is in envelope mode.

For an envelope (buzzer) channel an extra line names the *envelope* pitch and
its detune from the channel's tone note (the buzzer oscillator's own note),
plus the shape, since that -- not the volume -- is the audible pitch there.

Usage:
    from ay_roll import roll
    psg, L, R = render(track, bpm=138, beats_per_cycle=4, ...)
    print(roll(psg, bpm=138, beats_per_cycle=4, fps=50.0,
               rows_per_beat=4, bars=(0, 4)))      # first 4 bars
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from pyayay import EnvShape


_NAMES = ["C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-"]

# envelope shape glyphs, keyed by the R13 nibble (mirrors EnvShape labels)
_ENV_GLYPH = {
    int(EnvShape.UP_UP_C): "/|",      # 0x0C  /|/|  saw up
    int(EnvShape.DOWN_DOWN_8): "\\|",  # 0x08  \|\|  saw down
    int(EnvShape.UP_DOWN_E): "/\\",   # 0x0E  /\/\  triangle
}

DEFAULT_CLOCK = 1773400.0


# ---------------------------------------------------------------------------
# Register decoding (PSG frame -> per-channel musical reading)
# ---------------------------------------------------------------------------

def _midi_name(midi: float) -> str:
    """``60.0`` -> ``"C-4"``.  Rounds to the nearest semitone."""
    m = int(round(midi))
    return f"{_NAMES[m % 12]}{m // 12 - 1}"


def _period_to_midi(period: int, clock: float, divisor: float) -> Optional[float]:
    """Tone/envelope period back to a (fractional) MIDI note.

    ``divisor`` is 16 for the tone generator; for the envelope it is
    ``256 * ramps`` (512 for a triangle, 256 for a saw) -- the same constants
    the renderer uses forwards (see ``ay_patterns`` frequency math).
    """
    if period <= 0:
        return None
    freq = clock / (divisor * period)
    if freq <= 0:
        return None
    return 69.0 + 12.0 * math.log2(freq / 440.0)


@dataclass
class ChannelCell:
    """One channel's decoded state for a single frame."""
    note: Optional[str]        # tone note name, or None if tone period is 0
    tone_on: bool
    noise_on: bool
    env_on: bool
    volume: int                # 0..15, or 16 == envelope mode
    env_note: Optional[str]    # buzzer envelope note name (env_on only)
    env_detune: Optional[float]  # semitones: env note - tone note
    env_glyph: Optional[str]   # shape glyph (/\ etc.)


def decode_frame(regs: np.ndarray, clock: float = DEFAULT_CLOCK
                 ) -> List[ChannelCell]:
    """Decode one 14-byte PSG frame into three :class:`ChannelCell`."""
    mixer = int(regs[7])
    noise_p = int(regs[6]) & 0x1F
    env_p = (int(regs[12]) << 8) | int(regs[11])
    env_shape = int(regs[13]) & 0x0F
    glyph = _ENV_GLYPH.get(env_shape, f"x{env_shape:X}")
    # a triangle shape sounds one cycle per 2 ramps; saws one per ramp
    ramps = 2 if env_shape == int(EnvShape.UP_DOWN_E) else 1

    cells = []
    for ch in range(3):
        period = int(regs[ch * 2]) | ((int(regs[ch * 2 + 1]) & 0x0F) << 8)
        tone_on = not (mixer & (1 << ch))
        noise_on = not (mixer & (1 << (ch + 3)))
        vol_reg = int(regs[8 + ch])
        env_mode = bool(vol_reg & 0x10)
        volume = 16 if env_mode else (vol_reg & 0x0F)

        tone_midi = _period_to_midi(period, clock, 16.0)
        note = _midi_name(tone_midi) if tone_midi is not None else None

        env_note = env_detune = None
        if env_mode and env_p > 0:
            env_midi = _period_to_midi(env_p, clock, 256.0 * ramps)
            if env_midi is not None:
                env_note = _midi_name(env_midi)
                if tone_midi is not None:
                    env_detune = env_midi - tone_midi
        cells.append(ChannelCell(
            note=note, tone_on=tone_on, noise_on=noise_on, env_on=env_mode,
            volume=volume, env_note=env_note, env_detune=env_detune,
            env_glyph=glyph if env_mode else None,
            ))
    return cells


# ---------------------------------------------------------------------------
# Onset detection (tracker rows fire on triggers, not on every held frame)
# ---------------------------------------------------------------------------

def _is_sounding(c: ChannelCell) -> bool:
    """Whether the channel makes any sound this frame."""
    if not (c.tone_on or c.noise_on or c.env_on):
        return False
    # envelope mode is audible regardless of the volume nibble; otherwise a
    # zero volume is silence
    return c.env_on or c.volume > 0


def _retrigger(prev: Optional[ChannelCell], cur: ChannelCell) -> bool:
    """Heuristic note onset: the channel starts sounding, or its tone note /
    envelope note changes while sounding.  Used to decide when a tracker row
    prints a fresh note vs. blanks (a held note)."""
    if not _is_sounding(cur):
        return False
    if prev is None or not _is_sounding(prev):
        return True
    if cur.note != prev.note:
        return True
    if cur.env_on and cur.env_note != prev.env_note:
        return True
    return False


# ---------------------------------------------------------------------------
# Tracker row rendering
# ---------------------------------------------------------------------------

def _fmt_channel(label: str, prev: Optional[ChannelCell], cur: ChannelCell,
                 show_env: bool) -> str:
    """Format one channel column for one tracker row.

    Tracker convention: a fresh trigger prints the note; a held note blanks;
    silence prints ``===`` (note-off) once.
    """
    sounding = _is_sounding(cur)
    trig = _retrigger(prev, cur)
    was = prev is not None and _is_sounding(prev)

    if not sounding:
        note = "===" if was else "---"
    elif trig:
        note = cur.note or "???"
    else:
        note = "..."   # held

    tne = ("T" if cur.tone_on else ".") \
        + ("N" if cur.noise_on else ".") \
        + ("E" if cur.env_on else ".")

    if not sounding:
        vol = "  "
    elif cur.volume >= 16:
        vol = "EE"
    else:
        vol = f"{cur.volume:X} "

    col = f"{note:>3} {tne} {vol}"

    if show_env:
        if sounding and cur.env_on and cur.env_note:
            det = cur.env_detune if cur.env_detune is not None else 0.0
            col += f" {cur.env_glyph}{cur.env_note}{det:+04.1f}"
        else:
            col += " " * 9
    return col


def roll(
    psg: np.ndarray,
    bpm: float = 120.0,
    beats_per_cycle: int = 4,
    cps: Optional[float] = None,
    fps: float = 50.0,
    rows_per_beat: int = 4,
    bars: Optional[Tuple[float, float]] = None,
    clock: float = DEFAULT_CLOCK,
    show_env: bool = True,
) -> str:
    """Render a PSG array as a tracker-style text score.

    Parameters
    ----------
    psg : ndarray ``[frames, 14]``   the rendered register stream.
    bpm, beats_per_cycle / cps       tempo, matching the values passed to
                                     ``render`` (so rows line up with bars).
    fps                              PSG frame rate used for the render.
    rows_per_beat                    tracker row resolution (4 == 16th notes).
    bars                             ``(start_bar, end_bar)`` slice to show
                                     (default: the whole render).
    show_env                         add the envelope note/detune/shape column.

    Each tracker row aggregates the frames falling under it; the *last* frame in
    the row's window is decoded (so the row reads the state in force at the
    step), and onsets are detected against the previous row.
    """
    if cps is None:
        cps = bpm / 60.0 / beats_per_cycle
    frames_per_cycle = fps / cps                       # frames per bar
    frames_per_row = frames_per_cycle / (beats_per_cycle * rows_per_beat)
    n_frames = psg.shape[0]

    bar0, bar1 = (0.0, n_frames / frames_per_cycle) if bars is None else bars
    row0 = int(math.floor(bar0 * beats_per_cycle * rows_per_beat))
    row1 = int(math.ceil(bar1 * beats_per_cycle * rows_per_beat))

    rows_per_bar = beats_per_cycle * rows_per_beat
    width = 9 + (9 if show_env else 0)

    head = "row |bar:r| " + " | ".join(
        f"{lbl}: {'note T N E vol  env':<{width + 4}}"[:width + 4]
        for lbl in ("A", "B", "C"))
    sep = "-" * len(head)
    lines = [head, sep]

    prev_cells: List[Optional[ChannelCell]] = [None, None, None]
    for row in range(row0, row1):
        f_lo = int(round(row * frames_per_row))
        f_hi = int(round((row + 1) * frames_per_row))
        f_hi = min(max(f_hi, f_lo + 1), n_frames)
        if f_lo >= n_frames:
            break
        # decode the frame in force at this row (last frame of the window)
        cells = decode_frame(psg[f_hi - 1], clock)

        bar = row // rows_per_bar
        r_in_bar = row % rows_per_bar
        cols = [_fmt_channel(lbl, prev_cells[i], cells[i], show_env)
                for i, lbl in enumerate("ABC")]
        marker = ">" if r_in_bar == 0 else " "
        lines.append(f"{row:4d}|{bar:3d}:{r_in_bar:02d}|{marker}"
                     + " | ".join(cols))
        prev_cells = cells

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: dump a section of the megademo (or any render) as a roll
# ---------------------------------------------------------------------------

def _main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="tracker-style PSG roll")
    p.add_argument("--start", type=float, default=0.0, help="start bar")
    p.add_argument("--end", type=float, default=8.0, help="end bar")
    p.add_argument("--rpb", type=int, default=4, help="rows per beat")
    p.add_argument("--no-env", action="store_true", help="hide env column")
    args = p.parse_args(argv)

    from ay_megademo import track, BPM, BEATS
    from ay_patterns import render
    seconds = (args.end + 1) * BEATS * 60.0 / BPM
    psg, _, _ = render(track(), bpm=BPM, beats_per_cycle=BEATS,
                       seconds=seconds, fps=50.0)
    print(roll(psg, bpm=BPM, beats_per_cycle=BEATS, fps=50.0,
               rows_per_beat=args.rpb, bars=(args.start, args.end),
               show_env=not args.no_env))


if __name__ == "__main__":
    _main()
