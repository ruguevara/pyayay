"""
ay_demo_track.py -- a small showcase tune for the :mod:`ay_patterns` engine.

It exercises every AY trick the engine offers:

  * a tone+envelope "buzzer" bass (the two-oscillator timbre);
  * an arpeggiated lead with vibrato (chord cycled fast -> hardware-arp trick);
  * noise+tone percussion (kick / snare / hat).

Run it directly to render a WAV:

    python ay_demo_track.py            # -> ay_demo.wav
    python ay_demo_track.py out.wav    # custom path
"""

from __future__ import annotations

import sys

import numpy as np

from ay_patterns import (
    Buzzer, Percussion, Tone,
    chord, note, render, s, stack,
)


def demo() -> "Pattern":  # noqa: F821 (Pattern is the engine's return type)
    """A small showcase tune using every AY trick the engine offers."""
    bass = (
        note("c2 ~ c2 g1")
        .s(Buzzer(shape="saw", detune=0.0))
        .vol(16)
    )
    lead = (
        chord("c4:maj ~ a3:min g3:maj")
        .arp("updown")
        .fast(6)
        .s(Tone())
        .vibrato(0.3, 6.0)
        .pan(0.8)
    )
    drums = stack(
        s(Percussion("kick")).struct("x ~ ~ ~ x ~ ~ ~"),
        s(Percussion("snare")).struct("~ ~ ~ ~ x ~ ~ ~"),
        s(Percussion("hat")).fast(8).vol(8),
    ).pan(0.2)
    return stack(bass, lead, drums)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    out = argv[0] if argv else "ay_demo.wav"
    psg, left, right = render(demo(), bpm=125, seconds=8, wav=out)
    print(f"rendered {len(left)} samples, "
          f"peak L={np.abs(left).max():.3f} R={np.abs(right).max():.3f}, "
          f"-> {out}")


if __name__ == "__main__":
    main()
