"""
ay_demo_track.py -- a small showcase tune for the :mod:`ay_patterns` engine.

It exercises every AY trick the engine offers, with the focus on the
tracker-style **instruments / "samples"** -- the ADSR envelope system:

  * a volume-ADSR lead "sample" (attack/decay/sustain/release, Strudel-faithful:
    A/D/R in seconds, S a level), set fluently with ``.adsr("a:d:s:r")``;
  * a pitch-ADSR "blip" on the lead attack (``.penv``) -- the classic tracker
    pitch-snap at note-on;
  * a tone+noise "sample" (``tone_noise`` / ``Sample(tone=True, noise=True)``):
    the AY's two-generators-on-one-channel timbre, used here for a gritty
    metallic stab and a noise-swept snare;
  * an *ornament* (``.ornament(0,4,7)``) -- the hardware-arpeggio chord trick
    expressed as a per-frame semitone table instead of ``.arp().fast()``;
  * a tone+envelope "buzzer" bass (the two-oscillator timbre);
  * noise+tone percussion (kick / snare / hat).

Two ways to define an instrument, both shown below:

  * fluent / Strudel:  ``note(...).s(Tone()).adsr("0:.2:.4:.1").penv("0:.06:0:0")``
  * tracker struct:    ``note(...).s(Sample(vol="0:.2:.4:.1", tone=True, ...))``

A note on panning -- the AY has only **3 channels**, and panning is
**per-channel**, not per-note: ``render`` sets one pan per channel for the whole
track (``ay.set_pan(ch, ...)``).  The renderer voice-allocates the events that
sound in a frame onto the 3 channels **by pitch** (``poly_order="low"``: lowest
pitch -> channel 0, highest -> channel 2).  So a ``.pan()`` only lands where its
voice's channel does.  This demo therefore uses just **three stable pan
positions** -- left / centre / right -- and pans each voice to match the channel
its pitch will be allocated to (bass = low = left, lead/arp = high = right,
mid-range stab = centre), rather than pretending six voices can each have their
own independent pan.

Run it directly to render a WAV:

    python ay_demo_track.py            # -> ay_demo.wav
    python ay_demo_track.py out.wav    # custom path
"""

from __future__ import annotations

import sys

import numpy as np

from ay_patterns import (
    ADSR, Buzzer, Percussion, Sample, Tone,
    note, render, stack, tone_noise,
)


# The AY has 3 channels; panning is per-channel and fixed for the render.  The
# renderer allocates voices to channels by pitch (low -> ch0, high -> ch2), so
# we use exactly three stable pan positions and assign each voice the one its
# pitch register will land on.
PAN_LEFT = 0.15     # channel 0 -- the bass register
PAN_CENTRE = 0.5    # channel 1 -- mid-range stab / drums
PAN_RIGHT = 0.85    # channel 2 -- the lead / arp register


def demo() -> "Pattern":  # noqa: F821 (Pattern is the engine's return type)
    """A small showcase tune built around the new ADSR "sample" instruments.

    At most three voices sound at once, and each is pitched into a distinct
    register so the pitch-based voice allocation keeps it on a stable channel
    (and therefore a stable pan): bass left, mid centre, lead/arp right.
    """

    # --- LEFT (low) : buzzer bass -- hardware envelope as the oscillator -----
    bass = (
        note("c2 ~ c2 g1")
        .s(Buzzer(shape="saw", detune=0.1, tone=True))
        .vol(16)
        .pan(PAN_LEFT)
    )

    # --- RIGHT (high) : lead "sample" -- a volume ADSR + a pitch blip --------
    # Strudel-faithful ADSR: attack 0 s, decay 0.18 s down to a sustain *level*
    # of 0.35, release 0.1 s.  ``.penv`` snaps the pitch up +7 st and drops it
    # over 60 ms -- the tracker "blip" at note-on.
    lead = (
        note("c4 e4 g4 e4  f4 a4 c5 a4")
        .s(Tone())
        .adsr("0:0.18:0.35:0.1")
        .penv("0:0.06:0:0", peak=7)
        .pan(PAN_RIGHT)
    )

    # --- RIGHT (high) : ornament chord -- a fake major chord on ONE channel --
    # ``.ornament(0,4,7)`` cycles the chord tones per frame (the AY
    # hardware-arpeggio trick as a table); the same idea as
    # chord(...).arp().fast(), but stated as the instrument's own ornament.
    # It sits in the lead register and is sequenced *instead* of the lead, so it
    # too lands on the high channel.
    arp = (  # noqa: F841  -- swap-in alternative for high_slot (see below)
        note("c4 ~ a3 ~  f3 ~ g3 ~")
        .s(Tone())
        .ornament(0, 4, 7)
        .adsr("0:0.0:1:0.05")
        .vol(11)
        .pan(PAN_RIGHT)
    )

    # --- CENTRE (mid) : tone+noise "sample" -- a gritty metallic stab --------
    # tone + noise on one channel (the AY's two-generators timbre); a short
    # percussive volume ADSR, and the noise colour swept bright->dark so it
    # bites on the attack.
    stab = (  # noqa: F841  -- swap-in alternative for mid_slot (see below)
        note("c3 ~ ~ ~  c3 ~ eb3 ~")
        .s(tone_noise(noise_period=4, vol="0:0.1:0:0", noise_sweep=(2, 18)))
        .vol(14)
        .pan(PAN_CENTRE)
    )

    # --- CENTRE (mid) : drums -- kick + snare + hat, on ONE lane -------------
    # Percussion is pitched into the mid register (c3) so it allocates to the
    # centre channel, never outranking the lead.  The three drum hits are
    # interleaved on a single rhythm (``struct``) so at most one sounds per
    # step -- a single mid voice, not three stacked ones.
    drum_snare = Sample(tone=True, noise=True, noise_period=6,
                        vol=ADSR(a=0.0, d=0.16, s=0.0, r=0.0),
                        noise_sweep=(3, 20), volume=15)
    drums = stack(
        note("c3").s(Percussion("kick")).struct("x ~ ~ ~ x ~ ~ ~"),
        note("c3").s(drum_snare).struct("~ ~ ~ ~ x ~ ~ ~"),
        note("c3").s(Percussion("hat")).struct("~ x ~ x ~ x ~ x").vol(6),
    ).pan(PAN_CENTRE)

    # Three stable channel slots: bass = low = LEFT (ch0), stab/drums = mid =
    # CENTRE (ch1), lead/arp = high = RIGHT (ch2).  Each slot stays in its pitch
    # register so the voice allocator keeps it on its channel -- so the three
    # pans hold for the whole render.  Keep to one mid + one high voice at a
    # time (toggle to audition; the arp shares the lead's high slot, so swap).
    low_slot = bass
    mid_slot = drums          # or: mid_slot = stab  (the tone+noise stab)
    high_slot = lead          # or: high_slot = arp  (the ornament chord)

    return stack(
        low_slot,
        mid_slot,
        high_slot,
    )


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    out = argv[0] if argv else "ay_demo.wav"
    psg, left, right = render(demo(), bpm=125, seconds=8, wav=out)
    print(f"rendered {len(left)} samples, "
          f"peak L={np.abs(left).max():.3f} R={np.abs(right).max():.3f}, "
          f"-> {out}")


if __name__ == "__main__":
    main()
