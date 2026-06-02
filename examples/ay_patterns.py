"""
ay_patterns.py -- AY/YM chip instruments and renderer, built on strudel.py.

This module adds the chip-specific layer on top of the pure Strudel pattern
core: AY frequency math, Instrument subclasses (Tone, Buzzer, Percussion,
Sample), the pan-to-channel mapping, the PSG register assembler, and
``render()``.

It re-exports everything from ``strudel`` so existing ``from ay_patterns import
*`` call sites continue to work unchanged.

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
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np

import pyayay
from pyayay import EnvShape

from strudel import (  # noqa: F401 (re-exported)
    ADSR, Chord, Event, Note, Pattern, Span, Time,
    PAN_L, PAN_C, PAN_R, PAN_ANY,
    _as_note, _frac, _fromList, _pan_bucket, _parse_token, _REST,
    alt, arrange, at, cat, chord, euclid, fastcat, loop, midi_to_freq, note,
    note_name_to_midi, note_to_midi_freq, pure, seq, silence, slowcat,
    stack, timecat, window,
)


# ---------------------------------------------------------------------------
# AY frequency math
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
        fps      : frames per second (for per-frame ornament stepping)
    """

    def frame_state(self, note: Note, t_in: float, dur: float, clock: float,
                    fps: float = 50.0) -> VoiceState:
        raise NotImplementedError

    @staticmethod
    def _live_freq(note: Note, t_in: float, dur: float, fps: float = 50.0) -> float:
        midi = note.midi
        if note.sweep:
            midi += note.sweep * t_in
        if note.vib_depth and note.vib_rate:
            midi += note.vib_depth * math.sin(2 * math.pi * note.vib_rate * t_in)
        if note.pitch_env is not None:
            midi += note.pitch_env.at(t_in, dur)
        if note.ornament:
            frame = int(round(t_in * fps))
            midi += note.ornament[frame % len(note.ornament)]
        return midi_to_freq(midi)

    @staticmethod
    def _live_volume(note: Note, base_vol: int, t_in: float, dur: float) -> int:
        if note.vol_env is None:
            return base_vol
        return max(0, min(15, round(base_vol * note.vol_env.at(t_in, dur))))


class Tone(Instrument):
    """Plain square-wave tone channel."""

    def __init__(self, volume: int = 15):
        self.volume = volume

    def frame_state(self, note, t_in, dur, clock, fps=50.0):
        f = self._live_freq(note, t_in, dur, fps)
        vol = note.volume if note.volume is not None else self.volume
        vol = self._live_volume(note, vol, t_in, dur)
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

    * ``tone=False`` (default) -- *pure* buzzer: only the envelope sounds.
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

    def frame_state(self, note, t_in, dur, clock, fps=50.0):
        f = self._live_freq(note, t_in, dur, fps)
        env_f = midi_to_freq(note_to_midi_freq(f) + self.detune) * self.env_ratio
        env_p = env_period_for_freq(env_f, clock, ramps_per_cycle=self.ramps)
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
    """Buzzer + square tone an octave apart by default -> a fat doubled timbre."""
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
        "kick":  (True,  4,  20, 0.07, 60.0),
        "snare": (True,  6,  14, 0.09, 180.0),
        "hat":   (False, 1,  3,  0.03, 0.0),
        "tom":   (True,  8,  16, 0.11, 120.0),
    }

    def __init__(self, kind: str = "kick"):
        (self.use_tone, self.np0, self.np1,
         self.decay, self.base_freq) = self._PRESETS[kind]

    def frame_state(self, note, t_in, dur, clock, fps=50.0):
        frac = min(1.0, t_in / self.decay) if self.decay > 0 else 1.0
        vol = max(0, round(15 * (1.0 - frac)))
        noise_p = round(self.np0 + (self.np1 - self.np0) * frac)
        tone_p = 0
        if self.use_tone and self.base_freq > 0:
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


class Sample(Instrument):
    """A tracker-style *instrument* ("sample"): tone / noise / envelope flags
    plus a volume ADSR and an optional pitch ADSR + ornament, all in one struct.

    Tone + Noise
    ------------
    Set both ``tone=True`` and ``noise=True`` for the classic AY tone+noise
    timbre (a pitched square with a noise bite -- metallic leads, gritty basses,
    hand-claps).  ``noise_period`` is fixed by default; pass
    ``noise_sweep=(start, end)`` to sweep the noise colour across the volume
    envelope.

    Envelope (buzzer) layer
    -----------------------
    With ``env=True`` the chip's hardware envelope runs as the buzzer
    oscillator at the note pitch (see :class:`Buzzer`); the amplitude is then
    fixed to envelope-mode and the volume ADSR is ignored.

    Parameters
    ----------
    vol : ADSR | str       amplitude envelope ("a:d:s:r" string accepted).
    pitch : ADSR | str      optional pitch envelope (``pitch_peak`` semitones).
    pitch_peak : float      attack height of the pitch envelope, in semitones.
    ornament : seq[int]     optional per-frame semitone-offset table.
    tone, noise, env : bool which generators feed the channel.
    noise_period : int      fixed noise period (1..31) when ``noise=True``.
    noise_sweep : (int,int)  sweep noise period start->end across the vol env.
    env_shape : str         buzzer waveform when ``env=True`` ("saw"/"tri"/...).
    volume : int            base amplitude (0..15) the vol ADSR scales.
    """

    def __init__(self, vol="0:0:1:0", pitch=None, pitch_peak: float = 12.0,
                 ornament=None,
                 tone: bool = True, noise: bool = False, env: bool = False,
                 noise_period: int = 8,
                 noise_sweep=None,
                 env_shape: str = "saw", volume: int = 15):
        self.vol_env = ADSR.parse(vol)
        self.pitch_env = ADSR.parse(pitch, peak=pitch_peak) if pitch is not None else None
        self.ornament = tuple(int(o) for o in ornament) if ornament else None
        self.tone = tone
        self.noise = noise
        self.env = env
        self.noise_period = noise_period
        self.noise_sweep = noise_sweep
        self.env_shape = env_shape
        self.shape_enum, self.ramps = Buzzer._SHAPES[env_shape]
        self.volume = volume

    def frame_state(self, note, t_in, dur, clock, fps=50.0):
        n = note
        if n.vol_env is None and self.vol_env is not None:
            n = replace(n, vol_env=self.vol_env)
        if n.pitch_env is None and self.pitch_env is not None:
            n = replace(n, pitch_env=self.pitch_env)
        if n.ornament is None and self.ornament is not None:
            n = replace(n, ornament=self.ornament)

        f = self._live_freq(n, t_in, dur, fps)
        tone_p = tone_period_for_freq(f, clock)

        noise_p = self.noise_period
        if self.noise_sweep is not None:
            frac = self.vol_env.at(t_in, dur) / max(self.vol_env.peak, 1e-9)
            n0, n1 = self.noise_sweep
            noise_p = round(n0 + (n1 - n0) * (1.0 - frac))

        if self.env:
            env_f = midi_to_freq(note_to_midi_freq(f)) * 1.0
            env_p = env_period_for_freq(env_f, clock, ramps_per_cycle=self.ramps)
            return VoiceState(
                tone_on=self.tone, noise_on=self.noise, env_on=True,
                tone_period=tone_p, volume=16,
                pan=note.pan if note.pan is not None else 0.5,
                env_period=env_p, env_shape=int(self.shape_enum),
                noise_period=max(1, min(31, noise_p)) if self.noise else None,
            )

        base_vol = note.volume if note.volume is not None else self.volume
        if base_vol >= 16:
            base_vol = self.volume
        vol = self._live_volume(n, base_vol, t_in, dur)
        return VoiceState(
            tone_on=self.tone, noise_on=self.noise,
            tone_period=tone_p, volume=vol,
            pan=note.pan if note.pan is not None else 0.5,
            noise_period=max(1, min(31, noise_p)) if self.noise else None,
        )


def tone_noise(noise_period: int = 8, vol="0:0:1:0", **kw) -> Sample:
    """A tone+noise instrument: a pitched square with a noise bite mixed in."""
    return Sample(tone=True, noise=True, noise_period=noise_period, vol=vol, **kw)


# ---------------------------------------------------------------------------
# Bare instrument trigger (depends on Instrument, so lives here not strudel)
# ---------------------------------------------------------------------------

def s(instrument: Instrument) -> Pattern:
    """A bare instrument trigger (value carries only the instrument).

    Useful for drums: ``s(Percussion("kick")).struct("x ~ x ~")``.
    Defaults to a mid note so tone/noise instruments have a pitch.
    """
    return pure(replace(Note(midi=note_name_to_midi("a3")), instrument=instrument))


# ---------------------------------------------------------------------------
# Pan-to-channel mapping (AY-specific: exactly 3 channels)
#
# Fixed L/C/R buckets map to physical channels 0/1/2; ANY is seated after.
# The stereo pan value (0=left .. 1=right) each bucket/channel uses.
# ---------------------------------------------------------------------------

_BUCKET_CHANNEL = {PAN_L: 0, PAN_C: 1, PAN_R: 2}
_BUCKET_PAN_VALUE = {PAN_L: 0.0, PAN_C: 0.5, PAN_R: 1.0}
_CHANNEL_PAN_VALUE = {0: 0.0, 1: 0.5, 2: 1.0}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _assemble_frame(voices: List[VoiceState]) -> Tuple[np.ndarray, List[float]]:
    """Build a 14-byte PSG register frame from up to 3 voice states.

    Returns (regs[14] uint8, per-channel pan list).  A channel with no voice
    this frame reports pan ``None`` ("don't care") so it doesn't force a new
    pan segment in the render.  Envelope and noise are chip-global; if multiple
    voices request them, the last one wins.
    """
    regs = np.zeros(14, dtype=np.uint8)
    pans: List[Optional[float]] = [None, None, None]

    mixer = 0  # bits: tone A/B/C (0-2), noise A/B/C (3-5); 0 = ON (active low)
    env_period = 0
    env_shape = 0
    noise_period = 0

    for ch, v in enumerate(voices[:3]):
        if v is None:
            mixer |= (1 << ch) | (1 << (ch + 3))  # both off
            continue
        regs[ch * 2] = v.tone_period & 0xFF
        regs[ch * 2 + 1] = (v.tone_period >> 8) & 0x0F
        if v.volume >= 16:
            regs[8 + ch] = 0x10  # envelope mode (bit 4 set, level bits ignored)
        else:
            regs[8 + ch] = v.volume & 0x0F
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
):
    """Core grid sampler.  Returns (psg[n_frames,14] uint8, pans list)."""
    psg = np.zeros((n_frames, 14), dtype=np.uint8)
    pan_track: List[List[float]] = []

    sec_per_frame = 1.0 / fps
    cyc_per_frame = 1.0 / frames_per_cycle

    _DEFAULT = Tone()

    for fi in range(n_frames):
        cyc_begin = _frac(fi) * _frac(cyc_per_frame)
        cyc_end = _frac(fi + 1) * _frac(cyc_per_frame)
        events = pattern.query(Span(cyc_begin, cyc_end))

        frame_voices: List[_FrameVoice] = []

        for order, e in enumerate(events):
            note_val = _as_note(e.value)
            if note_val.instrument is None:
                note_val = replace(note_val, instrument=_DEFAULT)
            whole = e.whole if e.whole else e.part
            onset_sec = float(whole.begin) * frames_per_cycle * sec_per_frame
            now_sec = fi * sec_per_frame
            t_in = max(0.0, now_sec - onset_sec)
            dur = float(whole.end - whole.begin) * frames_per_cycle * sec_per_frame
            state = note_val.instrument.frame_state(note_val, t_in, dur, clock, fps)
            frame_voices.append(_FrameVoice(
                bucket=_pan_bucket(note_val.pan),
                priority=note_val.priority,
                order=order,
                state=state,
            ))

        voices = _allocate_channels(frame_voices)
        regs, pans = _assemble_frame(voices)
        psg[fi] = regs
        pan_track.append(pans)

    return psg, pan_track


@dataclass
class _FrameVoice:
    """One voice competing for a physical channel in a single frame."""
    bucket: str
    priority: int
    order: int
    state: VoiceState


def _allocate_channels(frame_voices: List["_FrameVoice"]
                       ) -> List[Optional["VoiceState"]]:
    """Resolve a frame's voices onto the 3 physical AY channels.

    The "pan *is* the channel" model:
      * each fixed pan bucket L/C/R maps to one physical channel (0/1/2) and is
        resolved *independently* -- the highest-priority voice in that bucket
        wins the channel; ties are broken by stack/document order;
      * "ANY" voices are seated afterwards into whatever channels are still free.
    """
    channels: List[Optional[VoiceState]] = [None, None, None]

    def best(cands: List[_FrameVoice]) -> _FrameVoice:
        return min(cands, key=lambda v: (-v.priority, v.order))

    for bucket, ch in _BUCKET_CHANNEL.items():
        cands = [v for v in frame_voices if v.bucket == bucket]
        if cands:
            v = best(cands)
            v.state.pan = _CHANNEL_PAN_VALUE[ch]
            channels[ch] = v.state

    any_voices = sorted((v for v in frame_voices if v.bucket == PAN_ANY),
                        key=lambda v: (-v.priority, v.order))
    for v in any_voices:
        free = next((ch for ch in (1, 0, 2) if channels[ch] is None), None)
        if free is None:
            break
        v.state.pan = _CHANNEL_PAN_VALUE[free]
        channels[free] = v.state

    return channels


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
    psg_path: Optional[str] = None,
    poly_order: str = "low",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a pattern to audio (and optionally a WAV / .psg file).

    Tempo: set ``bpm`` + ``beats_per_cycle`` (musician-friendly), or pass
    ``cps`` directly (Strudel cycles-per-second).  Duration: give ``seconds``
    or ``cycles``.

    Voices are placed on the 3 physical channels by their pan *bucket* (see the
    channel model).  ``poly_order`` is accepted for backwards compatibility but
    ignored.

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
        pattern, n_frames, frames_per_cycle, fps, clock
    )

    ay = pyayay.Ayumi(sample_rate=sample_rate, clock=clock, type=chip_type)
    ay.set_master_volume(master_volume)

    samples = int(math.ceil(n_frames / fps * sample_rate)) + sample_rate
    left = np.zeros(samples, dtype=np.float32)
    right = np.zeros(samples, dtype=np.float32)

    # Write R0-R12 every frame; suppress R13 (envelope shape) when unchanged
    # to avoid re-triggering the envelope generator (which resets the buzzer
    # sawtooth phase, chopping any buzzer tone into a 50 Hz buzz).
    mask = np.zeros_like(psg, dtype=bool)
    if n_frames > 1:
        r13 = psg[:, 13]
        unchanged = r13[1:] == r13[:-1]
        mask[1:, 13] = unchanged

    sample_cursor = 0
    for f0, f1, seg_pans in _pan_segments(pan_track):
        for ch in range(3):
            ay.set_pan(ch, seg_pans[ch])
        seg_samples = _frames_to_samples(f1 - f0, fps, sample_rate)
        ay.render_psg(psg[f0:f1], mask[f0:f1],
                      left[sample_cursor:], right[sample_cursor:], fps)
        sample_cursor += seg_samples

    left = left[:sample_cursor]
    right = right[:sample_cursor]

    if wav is not None:
        write_wav(wav, left, right, sample_rate)
    if psg_path is not None:
        write_psg(psg_path, psg, fps)

    return psg, left, right


def _pan_segments(pan_track: List[List[Optional[float]]]):
    """Split a per-frame pan track into maximal runs over which every channel's
    pan is stable, and yield ``(begin_frame, end_frame, pans)``.

    A per-frame entry of ``None`` for a channel means "no voice -- don't care",
    so it never forces a new segment: it inherits the run's current pan.  A new
    segment starts only when a channel reports a concrete pan that *differs*
    from the one in force.  Channels never assigned a pan default to centre.
    """
    n = len(pan_track)
    if n == 0:
        return
    start = 0
    cur: List[Optional[float]] = list(pan_track[0])
    for i in range(1, n):
        frame = pan_track[i]
        conflict = any(frame[ch] is not None and cur[ch] is not None
                       and frame[ch] != cur[ch] for ch in range(3))
        if conflict:
            yield start, i, [0.5 if p is None else p for p in cur]
            start = i
            cur = list(frame)
        else:
            for ch in range(3):
                if cur[ch] is None and frame[ch] is not None:
                    cur[ch] = frame[ch]
    yield start, n, [0.5 if p is None else p for p in cur]


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


def write_psg(path: str, psg: np.ndarray, fps: float = 50.0):
    """Write a ``[frames, 14]`` register array as a standard ``.psg`` file.

    The PSG format (Sergey Bulba's AY register-dump container) is:
      * a 16-byte header: ``"PSG"`` + ``0x1A``, then a version byte, an interrupt
        rate byte (frames/sec, ``0`` meaning the default 50), and 10 reserved
        zero bytes;
      * a body of frames, each opened by an end-of-interrupt marker ``0xFF`` and
        followed by ``reg, value`` byte pairs for the registers written that
        frame.  ``0xFD`` ends the stream.

    Only registers that *changed* since the previous frame are emitted, which
    shrinks the file and preserves the buzzer-friendly behaviour of not
    re-latching R13 needlessly.
    """
    psg = np.asarray(psg, dtype=np.uint8)
    if psg.ndim != 2 or psg.shape[1] < 14:
        raise ValueError("psg must be a [frames, >=14] array")

    rate = int(round(fps)) & 0xFF
    out = bytearray()
    out += b"PSG\x1a"
    out += bytes([0x00, rate]) + bytes(10)

    prev = None
    for frame in psg[:, :14]:
        out.append(0xFF)
        for reg in range(14):
            val = int(frame[reg])
            if prev is not None and val == int(prev[reg]):
                continue
            out += bytes([reg, val])
        prev = frame
    out.append(0xFD)

    with open(path, "wb") as f:
        f.write(out)
