"""
ay_analyze.py -- objective "listening" for AY pattern renders.

I can't hear the output, so this turns a render into numbers I can reason about:

  * a per-window loudness envelope (RMS in dBFS)  -> verify buildup / drop arc
  * spectral centroid per window (Hz)             -> verify brightness rises
  * dominant pitch classes per window             -> verify the harmony / key
  * note (onset) density from the PSG stream       -> verify activity changes
  * peak / clipping / silence flags                -> sanity

It also draws a tiny ASCII loudness + brightness "score" so the arc is visible
at a glance, and can dump per-section stats given section boundaries in seconds.

Usage:
    from ay_analyze import analyze, print_report
    psg, L, R = render(track, ...)
    print_report(L, R, sample_rate=44100, sections=[("intro",0,8),("drop",8,24)])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# Pitch-class names for harmony read-out.
_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class WindowStat:
    t0: float
    t1: float
    rms_db: float
    peak: float
    centroid_hz: float
    top_pitches: List[Tuple[str, float]]   # (pitch-class, strength 0..1)


def _rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-9 else -120.0


def _spectral_centroid(x: np.ndarray, sr: int) -> float:
    if len(x) < 64 or np.allclose(x, 0):
        return 0.0
    win = x * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(win), 1.0 / sr)
    s = spec.sum()
    if s <= 0:
        return 0.0
    return float((freqs * spec).sum() / s)


def _chroma(x: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 4000.0):
    """Fold the magnitude spectrum into 12 pitch classes (a chromagram)."""
    if len(x) < 256 or np.allclose(x, 0):
        return np.zeros(12)
    win = x * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(win), 1.0 / sr)
    chroma = np.zeros(12)
    for f, m in zip(freqs, spec):
        if f < fmin or f > fmax or m <= 0:
            continue
        midi = 69 + 12 * math.log2(f / 440.0)
        pc = int(round(midi)) % 12
        chroma[pc] += m
    if chroma.sum() > 0:
        chroma /= chroma.sum()
    return chroma


def analyze_window(mono: np.ndarray, sr: int, t0: float, t1: float) -> WindowStat:
    rms = _rms(mono)
    peak = float(np.abs(mono).max()) if len(mono) else 0.0
    centroid = _spectral_centroid(mono, sr)
    chroma = _chroma(mono, sr)
    order = np.argsort(chroma)[::-1]
    top = [(_PC_NAMES[i], float(chroma[i])) for i in order[:3] if chroma[i] > 0.04]
    return WindowStat(t0, t1, _db(rms), peak, centroid, top)


def analyze(left: np.ndarray, right: np.ndarray, sample_rate: int,
            window_s: float = 1.0) -> List[WindowStat]:
    mono = (left.astype(np.float64) + right.astype(np.float64)) * 0.5
    n = len(mono)
    win = int(window_s * sample_rate)
    stats = []
    for start in range(0, n, win):
        seg = mono[start:start + win]
        if len(seg) < win // 4:
            break
        t0 = start / sample_rate
        t1 = (start + len(seg)) / sample_rate
        stats.append(analyze_window(seg, sample_rate, t0, t1))
    return stats


# -- PSG-domain stats (onset density straight from the register stream) ------

def onset_density(psg: np.ndarray, fps: float, window_s: float = 1.0) -> List[float]:
    """Notes-per-second per window, inferred from tone-period register changes
    on the three channels (a crude but useful 'activity' meter)."""
    n_frames = psg.shape[0]
    per_win = max(1, int(window_s * fps))
    densities = []
    # period for each channel = R[2c] | (R[2c+1]<<8)
    periods = np.stack([
        psg[:, 0].astype(int) | (psg[:, 1].astype(int) << 8),
        psg[:, 2].astype(int) | (psg[:, 3].astype(int) << 8),
        psg[:, 4].astype(int) | (psg[:, 5].astype(int) << 8),
    ], axis=1)
    for start in range(0, n_frames, per_win):
        seg = periods[start:start + per_win]
        if len(seg) < 2:
            break
        changes = int((np.diff(seg, axis=0) != 0).sum())
        densities.append(changes / window_s)
    return densities


# -- ASCII visualisation -----------------------------------------------------

def _bar(value: float, lo: float, hi: float, width: int = 32, ch: str = "#") -> str:
    if hi <= lo:
        return ""
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    fill = int(round(frac * width))
    return ch * fill + "." * (width - fill)


def ascii_score(stats: Sequence[WindowStat]) -> str:
    """Two stacked ASCII meters: loudness (dBFS) and brightness (centroid)."""
    if not stats:
        return "(no data)"
    dbs = [s.rms_db for s in stats]
    cents = [s.centroid_hz for s in stats]
    db_lo, db_hi = -48.0, max(-6.0, max(dbs))
    c_lo, c_hi = 0.0, max(1.0, max(cents))
    lines = ["  t   loudness(dBFS)                    bright(Hz)"]
    for s in stats:
        t = f"{s.t0:4.0f}"
        lb = _bar(s.rms_db, db_lo, db_hi, 28, "#")
        cb = _bar(s.centroid_hz, c_lo, c_hi, 16, "=")
        lines.append(f"{t}  {lb} {s.rms_db:6.1f}  {cb} {s.centroid_hz:5.0f}")
    return "\n".join(lines)


# -- Section report ----------------------------------------------------------

def section_stats(left, right, sample_rate, name, t0, t1):
    s0 = int(t0 * sample_rate)
    s1 = int(t1 * sample_rate)
    mono = (left[s0:s1].astype(np.float64) + right[s0:s1].astype(np.float64)) * 0.5
    rms = _rms(mono)
    peak = float(np.abs(mono).max()) if len(mono) else 0.0
    centroid = _spectral_centroid(mono, sample_rate)
    chroma = _chroma(mono, sample_rate)
    order = np.argsort(chroma)[::-1]
    top = [(_PC_NAMES[i], round(float(chroma[i]), 2)) for i in order[:4]
           if chroma[i] > 0.04]
    # stereo balance
    lr = (_rms(left[s0:s1]), _rms(right[s0:s1]))
    bal = (lr[1] - lr[0]) / (lr[0] + lr[1] + 1e-9)
    return {
        "name": name, "t0": t0, "t1": t1,
        "rms_db": round(_db(rms), 1), "peak": round(peak, 3),
        "centroid_hz": round(centroid), "balance": round(bal, 2),
        "pitches": top,
    }


def print_report(left, right, sample_rate, psg=None, fps=50.0,
                 sections: Optional[List[Tuple[str, float, float]]] = None,
                 window_s: float = 2.0):
    dur = len(left) / sample_rate
    peak = float(np.abs(np.concatenate([left, right])).max())
    clipping = peak >= 0.999
    print(f"=== render report ===  duration {dur:.1f}s  peak {peak:.3f}"
          f"{'  !! CLIPPING' if clipping else ''}")
    stats = analyze(left, right, sample_rate, window_s)
    print(ascii_score(stats))

    if psg is not None:
        dens = onset_density(psg, fps, window_s)
        if dens:
            print("\nactivity (note-changes/s per "
                  f"{window_s:g}s window): "
                  + " ".join(f"{d:.0f}" for d in dens))

    if sections:
        print("\n--- sections ---")
        prev_db = None
        for name, t0, t1 in sections:
            st = section_stats(left, right, sample_rate, name, t0, t1)
            arrow = ""
            if prev_db is not None:
                d = st["rms_db"] - prev_db
                arrow = f"  ({'+' if d >= 0 else ''}{d:.1f} dB)"
            prev_db = st["rms_db"]
            pitches = " ".join(f"{p}:{w}" for p, w in st["pitches"])
            print(f"  {name:10s} {t0:5.1f}-{t1:4.1f}s  "
                  f"{st['rms_db']:6.1f} dBFS{arrow:14s}  "
                  f"bright {st['centroid_hz']:5d}Hz  "
                  f"bal {st['balance']:+.2f}  [{pitches}]")
    return stats
