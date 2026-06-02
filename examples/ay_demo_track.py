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
  * a tone+envelope "buzzer" bass (the two-oscillator timbre), with its
    envelope **shape alternated per bar** via ``alt`` (saw/triangle);
  * a priority-ducked ``.echo`` on the stab -- decaying trailing taps that the
    next dry hit cuts off by re-winning the channel;
  * noise+tone percussion (kick / snare / hat).

Two ways to define an instrument, both shown below:

  * fluent / Strudel:  ``note(...).s(Tone()).adsr("0:.2:.4:.1").penv("0:.06:0:0")``
  * tracker struct:    ``note(...).s(Sample(vol="0:.2:.4:.1", tone=True, ...))``

Channels & panning -- the AY has only **3 channels**, and pan is *per channel*.
The engine models this with three pan *buckets* -- ``"L"`` / ``"C"`` / ``"R"``
(left / centre / right) -- and treats each bucket as one physical channel.  Each
frame the renderer groups the sounding voices by bucket and keeps the highest
``.priority()`` voice in each (ties broken by stack order); the rest are dropped.
So you place voices by bucket and let several share one channel by precedence.

This demo squeezes **five voices onto the three channels**:

    LEFT   : the tone+noise stab
    CENTRE : the bass AND the whole drum kit, all sharing one channel --
             drums punch through (kick > snare > hat > bass), the bass fills the
             gaps.  The classic chip "bass ducks for the kick" sound.
    RIGHT  : the ornament-chord arp

Run it directly to render a WAV:

    python ay_demo_track.py            # -> ay_demo.wav
    python ay_demo_track.py out.wav    # custom path
"""

from __future__ import annotations

import sys

import numpy as np

from fractions import Fraction

from ay_patterns import (
    ADSR, Buzzer, Percussion, Sample, Tone,
    PAN_L, PAN_C, PAN_R,
    alt, note, render, s, stack, tone_noise,
)


# Precedence on the shared CENTRE channel: the drums punch through the bass.
PRI_KICK, PRI_SNARE, PRI_HAT, PRI_BASS = 4, 3, 2, 1


def demo() -> "Pattern":  # noqa: F821 (Pattern is the engine's return type)
    """A small showcase tune that squeezes five voices onto three channels.

    CENTRE carries the bass *and* the drum kit, sharing one channel by
    precedence; the stab sits LEFT and the ornament arp RIGHT.
    """

    # --- CENTRE : buzzer bass -- shares the channel with the drums -----------
    # The hardware envelope is the oscillator (the two-oscillator "buzzer").
    # Lowest priority on CENTRE, so any drum hit steals the channel from it and
    # the bass fills the gaps between hits.
    #
    # `alt(...)` alternates the envelope *shape* per bar -- saw on even bars,
    # triangle on odd -- the per-cycle ``<...>`` idea applied to a parameter
    # that lives on the instrument.  Shape sits on the Buzzer, so we alternate
    # whole instruments via ``.s``; the harmony is unchanged, only the timbre
    # flips each cycle.
    bass = (
        alt(
            note("c2 ~ c2 g1"), "s",
            Buzzer(shape="saw", detune=0.1, tone=True),
            Buzzer(shape="tri", detune=0.1, tone=True),
        )
        .vol(16)
        .pan(PAN_C)
        .priority(PRI_BASS)
    )

    # --- CENTRE : the drum kit -- kick + snare + hat, punching through -------
    # All three drums (and the bass above) share the one CENTRE channel;
    # ``.priority()`` (kick > snare > hat > bass) decides who wins each frame.
    # The snare is itself a tracker-style tone+noise Sample.
    drum_snare = Sample(tone=True, noise=True, noise_period=6,
                        vol=ADSR(a=0.0, d=0.09, s=0.0, r=0.0),
                        noise_sweep=(3, 20), volume=15)
    # Sparse hats (offbeats only) leave gaps on the downbeats for the bass to
    # sound through -- otherwise the kit would cover every frame and mute it.
    drums = stack(
        s(Percussion("kick")).struct("x ~ ~ ~ x ~ ~ ~").pan(PAN_C).priority(PRI_KICK),
        s(drum_snare).struct("~ ~ ~ ~ x ~ ~ ~").pan(PAN_C).priority(PRI_SNARE),
        s(Percussion("hat")).struct("~ ~ x ~ ~ ~ x ~").vol(6).pan(PAN_C).priority(PRI_HAT),
    )

    # --- LEFT : tone+noise "sample" -- a gritty metallic stab ----------------
    # tone + noise on one channel (the AY's two-generators timbre); a short
    # percussive volume ADSR, and the noise colour swept bright->dark so it
    # bites on the attack.
    #
    # ``.echo`` trails three decaying repeats one step apart into the rests.
    # Each tap carries a lower ``.priority`` than the dry hit, so when the next
    # stab lands it re-wins the LEFT channel and cuts its own lingering tail --
    # the transient-priority duck, falling straight out of the pan-bucket
    # allocator (no feedback bus needed).
    stab = (
        note("c3 ~ ~ ~  c3 ~ eb3 ~")
        .s(tone_noise(noise_period=4, vol="0:0.1:0:0", noise_sweep=(2, 18)))
        .vol(14)
        .pan(PAN_L)
        .priority(5)
        .echo(times=3, delay=Fraction(1, 8), feedback=0.5)
    )

    # --- RIGHT : ornament chord -- a fake major chord on ONE channel ---------
    # ``.ornament(0,4,7)`` cycles the chord tones per frame (the AY
    # hardware-arpeggio trick as a table); the same idea as
    # chord(...).arp().fast(), but stated as the instrument's own ornament.
    #
    # A second use of ``alt``: over a *value* transform here, lifting the whole
    # figure up a fourth on odd bars (call/response) -- the harmony moves with
    # the bar grid, no scheduler hack.
    arp = alt(
        note("c4 ~ a3 ~  f3 ~ g3 ~")
        .s(Tone())
        .ornament(0, 4, 7)
        .adsr("0:0.0:1:0.05")
        .vol(11)
        .pan(PAN_R),
        "add", 0, 5,
    )

    # Five voices, three channels: stab LEFT, bass+drums share CENTRE, arp RIGHT.
    return stack(
        stab,
        bass,
        drums,
        arp,
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
