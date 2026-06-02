"""
ay_megademo.py -- "Crystal Decline", a demoscene-style AY/YM track.

A longer arranged piece in the spirit of early-90s Future Crew / Purple Motion
module music (Second Reality / Unreal mood): C natural+harmonic minor, chromatic
mediant moves, maj7/min7/sus colours, a heroic vibrato lead, buzzer basses and
the odd acid-glitch fill.  Structure:

    intro   8 bars   atmospheric buzzer pad, sparse
    build   8 bars   arp wakes up, snare rolls, brightening
    drop   16 bars   full energy: buzzer bass + arp chords + drive
    fill    -        acid-glitch turnaround (inside the drop/solo seam)
    solo   16 bars   expressive harmonic-minor lead, vibrato + slides
    outro   8 bars   wind down, Picardy-ish bittersweet close

Three channels only (it's an AY!), so chords are faked with fast arpeggios and
the bass leans on the envelope "buzzer" for body.

The melodic/harmonic voices are tracker-style ``Sample`` instruments (see
``ay_patterns``): each bundles its timbre with a Strudel-faithful ADSR so notes
pluck, swell, ring and bite instead of holding a flat square -- a singing LEAD
(with an attack pitch-blip), a percussive PLUCK, a tone+noise STAB, a ringing
BELL and a tone+noise SNARE.  The basses stay on the pure-envelope buzzer, kept
in a low octave so the envelope period is large and the pitch stays in tune.

Run it:
    python ay_megademo.py                 # -> crystal_decline.wav + report
    python ay_megademo.py out.wav
    python ay_megademo.py --report-only   # analysis without writing a WAV
"""

from __future__ import annotations

import os
import sys

import numpy as np

from ay_patterns import (
    ADSR, Buzzer, Percussion, Sample,
    PAN_L, PAN_C, PAN_R,
    arrange, at, buzz_bass, chord, note, render, s, stack, tone_noise,
)


# ---------------------------------------------------------------------------
# Channel / pan plan (the AY has only 3 channels; pan *is* the channel).
#
#   LEFT   = the buzzer bass                 (foundation)
#   RIGHT  = the harmonic/melodic top        (pad / arp / lead / bell)
#   CENTRE = the drum kit, time-sharing one channel by precedence
#            (kick beats snare beats hat -- the most prominent hit wins the frame)
#
# Where there is no bass (intro / build) the pad borrows the free LEFT channel.
# ---------------------------------------------------------------------------

PRI_KICK = 3
PRI_SNARE = 2
PRI_HAT = 1


# ---------------------------------------------------------------------------
# Harmony.  C minor; chords named relative to the key.
#
#   i    = Cm        VI  = Abmaj7     III = Ebmaj
#   v/V  = G(7)      iv  = Fm         VII = Bbmaj
#   bII-ish chromatic colour and a B (leading tone) for harmonic-minor drama.
# ---------------------------------------------------------------------------

# Tempo: ~138 "BPM" feel.  beats_per_cycle = 4 => one cycle == one bar.
BPM = 138
BEATS = 4

# Instruments ("samples") ----------------------------------------------------
#
# The melodic/harmonic voices are tracker-style ``Sample`` instruments: each
# bundles its timbre with a Strudel-faithful ADSR (A/D/R in seconds, S a level)
# so notes pluck, swell and ring like a tracker patch rather than holding a flat
# square.  The basses stay on the AY "buzzer" (the hardware envelope as the
# pitched oscillator); they are *pure-envelope* (no square tone) and kept in
# their low octave -- there the envelope period is large, so the pitch is
# accurate.  (An octave-up envelope has tiny periods and goes audibly sharp; see
# CLAUDE.md.)

PAD = buzz_bass(shape="tri")          # smooth triangle pad buzzer
PAD_FIFTH = Buzzer(shape="tri")       # (pure tri; was a "ring" colour)
BASS = buzz_bass(shape="saw")         # clean, bright saw buzzer bass

# A singing lead: fast attack, a gentle decay to a *high* sustain level so the
# line carries (this is the melodic voice, not a pluck), and a short release so
# legato lines don't click.  A tiny up-pitch blip on the attack (penv) gives it
# the tracker "snap".
LEAD = Sample(tone=True, vol="0.01:0.08:1.0:0.05",
              pitch="0:0.04:0:0", pitch_peak=4, volume=15)

# The arp/chord pluck: very fast attack, quick decay to a low sustain so each
# arpeggio step reads as a distinct pluck rather than a smear.
PLUCK = Sample(tone=True, vol="0:0.08:0.25:0.04", volume=13)

# A gritty tone+noise stab for accents: pitched square with a noise bite whose
# colour sweeps bright->dark across the (short) volume decay.
STAB = tone_noise(noise_period=4, vol="0:0.12:0:0", noise_sweep=(2, 18))

# A ringing bell: struck (instant attack), long decay tail, no sustain -- the
# tracker "bell sample".
BELL = Sample(tone=True, vol="0:0.9:0:0", volume=12)

KICK = Percussion("kick")
HAT = Percussion("hat")

# A snare built as a tone+noise Sample: a noise burst with a tonal "body",
# decaying fast, the noise colour opening up as it hits.
SNARE = Sample(tone=True, noise=True, noise_period=6,
               vol=ADSR(a=0.0, d=0.14, s=0.0, r=0.0),
               noise_sweep=(3, 20), volume=15)


# ---------------------------------------------------------------------------
# The harmonic loop.
#
# Main 8-bar progression (two 4-bar phrases), C minor with the chromatic
# mediant surprise (Eb -> B-ish via the harmonic-minor V) Purple Motion loved.
# ---------------------------------------------------------------------------

N_BARS = 8  # length of the harmonic loop, in bars

PROG_A = "c3:min ab2:maj7 eb3:maj bb2:maj"      # i  VI  III  VII
PROG_B = "f3:min ab2:maj7 g3:maj g3:maj"        # iv VI  V    V   (harmonic-minor V)
PROG_FULL = PROG_A + " " + PROG_B

# Bass: one root per bar, tracking the 8-bar progression above (slowed x8 so
# each root lasts a whole bar instead of collapsing into bar 1 and repeating).
BASS_ROOTS = "c2 ab1 eb2 bb1 f2 ab1 g1 g1"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def pad_progression(prog: str, inst=PAD, vol=16, p=PAN_R, n_bars=N_BARS, shimmer=4):
    """A chord pad: one chord per bar (progression spread over ``n_bars``),
    gently arpeggiated so all notes are heard as a shimmer, not a hard chord.

    ``shimmer`` is the *fixed* number of arpeggio re-triggers per bar
    (``stutter`` before ``arp``).  This is what keeps the shimmer steady:
    arpeggiating the bare chord (``shimmer=1``) ties the onset rate to the chord
    *size* (3 onsets/bar for a triad, 4 for a maj7), so the pulse lurches -- it
    speeds up on the four-note chords and drags on the triads, which reads as
    the pad slowing down.  Stuttering onto a fixed ``shimmer`` grid first pins
    every bar to the same pulse regardless of how many notes the chord has.
    """
    return (
        chord(prog).slow(n_bars).stutter(shimmer).arp("up")
        .s(inst).vol(vol).pan(p)
    )


def arp_line(prog: str, speed=8, mode="updown", inst=PLUCK, vol=13, p=PAN_R,
             n_bars=N_BARS):
    """Fast arpeggio over the chord-per-bar progression (the chip 'chord').

    ``chord(prog).slow(n_bars)`` puts one chord per bar; ``.stutter(speed)``
    repeats each chord ``speed`` times within its bar; ``.arp(mode)`` then
    cycles the chord notes -- i.e. the AY hardware-arpeggio chord trick, kept in
    sync with the 8-bar progression.
    """
    return (
        chord(prog).slow(n_bars).stutter(speed).arp(mode)
        .s(inst).vol(vol).pan(p)
    )


def bass_roots(roots: str = BASS_ROOTS, n_bars: int = 8, inst=BASS, vol=16, p=PAN_L):
    """One sustained root per bar, tracking the progression (held buzzer bass)."""
    return note(roots).slow(n_bars).s(inst).vol(vol).pan(p)


def bass_drive(roots: str = BASS_ROOTS, n_bars: int = 8, inst=BASS, vol=16, p=PAN_L):
    """Pumping bass: the per-bar root retriggered on an 8th-note pulse.

    The root (held a whole bar) supplies the pitch; ``struct`` supplies a
    driving rhythm that re-triggers it.
    """
    held = note(roots).slow(n_bars)
    pulse = held.struct("x ~ x x ~ x x ~")  # syncopated demoscene pulse
    return pulse.s(inst).vol(vol).pan(p)


def drum_kit(kick_pat="x ~ ~ x ~ ~ x ~", snare_pat="~ ~ x ~ ~ ~ x ~",
             hat_pat="~ x ~ x ~ x ~ x", hat_gain=0.5, snare_vol=12):
    """The three drum voices, all on the CENTRE channel, time-sharing it by
    precedence: kick > snare > hat, so whichever lands on a frame wins it."""
    kick = s(KICK).struct(kick_pat).pan(PAN_C).priority(PRI_KICK)
    snare = s(SNARE).struct(snare_pat).vol(snare_vol).pan(PAN_C).priority(PRI_SNARE)
    hats = s(HAT).struct(hat_pat).gain(hat_gain).pan(PAN_C).priority(PRI_HAT)
    return stack(kick, snare, hats)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def intro():
    # 8-bar pad, gentle shimmer; a lonely bell motif tracing the chord tones.
    # No bass yet, so the pad takes the free LEFT channel; the bell sings RIGHT.
    pad = pad_progression(PROG_FULL, inst=PAD, vol=16, p=PAN_L, shimmer=4)
    bell = (
        note("c5 eb5 g4 bb4 ab4 c5 g4 g4")  # one bell tone per bar
        .slow(N_BARS)
        .s(BELL)                            # struck, long ringing decay
        .vol(11)
        .pan(PAN_R)
    )
    return stack(pad, bell)


def build():
    # First 6 bars: pad + arp + light drums, brightening.  Last 2 bars: a
    # tension riser -- everything thins to a fast snare roll + a held dominant
    # buzzer, the classic "breath before the drop".
    #
    # Still no bass, so the pad borrows LEFT; the arp leads on RIGHT; the drums
    # share CENTRE.  The arp outranks the pad would they ever collide (they are
    # on different channels here, so they don't).
    body_pad = pad_progression(PROG_FULL, inst=PAD_FIFTH, vol=16, p=PAN_L)
    body_arp = arp_line(PROG_FULL, speed=8, mode="up", inst=PLUCK, vol=11, p=PAN_R)
    body_drums = drum_kit(kick_pat="~ ~ ~ ~ ~ ~ ~ ~",
                          snare_pat="~ ~ ~ x ~ ~ ~ x", hat_pat="x x x x x x x x",
                          hat_gain=0.6, snare_vol=9)
    body = stack(body_pad, body_arp, body_drums)

    # tension bars: a snare roll accelerating, on a held G (the dominant)
    riser = stack(
        note("g3:maj").arp("up").fast(6).s(PLUCK).vol(10).pan(PAN_R),
        s(SNARE).fast(8).stutter(2).gain(0.7).pan(PAN_C).priority(PRI_SNARE),
    )
    # 0-6 = body (it loops every bar), 6-8 = riser
    return arrange(at(0, body), at(6, riser))


def drop():
    # driving, pumping buzzer bass that tracks the 8-bar progression.
    # Pure-envelope buzz bass (NOT the octave-up pwm one): at low octaves the
    # envelope period is large and the pitch is accurate; an octave-up envelope
    # has tiny periods with coarse, audibly out-of-tune steps.
    #
    # Now the three channels are full and stable: bass LEFT, arp RIGHT, drums
    # CENTRE (kick/snare/hat sharing the one channel by precedence).
    bass = bass_drive(inst=BASS, vol=16, p=PAN_L)
    arp = arp_line(PROG_FULL, speed=12, mode="updown", inst=PLUCK, vol=13, p=PAN_R)
    drums = drum_kit(
        kick_pat="x ~ ~ x ~ ~ x ~",
        snare_pat="~ ~ x ~ ~ ~ x ~",
        hat_pat="~ x ~ x ~ x ~ x",
    )
    return stack(bass, arp, drums)


def acid_fill():
    """A one-bar acid-glitch turnaround: a stuttered, pitch-glitched run on the
    gritty tone+noise STAB (CENTRE), punctuated by kick hits that outrank it."""
    glitch = (
        chord("c3:min").arp("up").fast(16)
        .glitch(amount=12, seed=5)
        .stutter(2)
        .s(STAB)                            # tone+noise: extra grit on the fill
        .vol(13)
        .pan(PAN_C)
    )
    kick = s(KICK).struct("x ~ x ~ x ~ x ~").pan(PAN_C).priority(PRI_KICK)
    return stack(glitch, kick)


def _phrase(tokens, depth=0.18, rate=5.0, slide=0.0):
    """A 4-bar melodic phrase: ``tokens`` is a string of 32 events (8th notes
    over 4 bars), played once across the phrase.  Gentle vibrato; optional
    slide into accents.  ``~`` holds/rests so the line can breathe.  The lead
    sings on RIGHT, high priority so it always owns its channel."""
    p = (note(tokens).slow(4).s(LEAD).vol(15).vibrato(depth, rate)
         .pan(PAN_R).priority(5))
    if slide:
        p = p.slide(slide)
    return p


def solo():
    # A singing lead in C (natural + harmonic) minor over the 8-bar loop.
    # Four distinct 4-bar phrases with a recurring 3-note motif (g-ab-g, the
    # i->VI sigh), clear rise-and-fall contour, breathing rests, and a held
    # note at each phrase end.  Vibrato is gentle (depth ~0.18 semitone) so the
    # pitch stays clearly defined rather than wobbling.
    #
    # Phrase A (bars 1-4, over Cm Ab Eb Bb): state the motif, climb, settle.
    phrase_a = _phrase(
        "g4  ~  ab4 g4  ~   ~   eb4 f4 "   # Cm   : sighing motif g-ab-g
        "g4  ~   ~  ~   c5  ~  bb4 ~  "     # Ab   : reach up, hold
        "g4  ab4 g4  ~  eb4 ~   d4  ~  "    # Eb   : motif again, descend
        "eb4 ~   ~  ~   ~   ~   ~   ~  "    # Bb   : land and hold
    )
    # Phrase B (bars 5-8, over Fm Ab G G): higher answer, leading-tone tension.
    phrase_b = _phrase(
        "c5  ~  d5  eb5 ~   ~  c5  ~  "     # Fm   : lift the register
        "f5  ~  eb5 ~   d5  ~  c5  ~  "     # Ab   : arch down
        "b4  ~  c5  d5  ~   ~  b4  ~  "     # G    : B natural -> leading tone
        "c5  ~   ~  ~   ~   ~  ~   ~  ",    # G    : resolve to tonic, hold
        slide=1.0,
    )
    # Phrase C (bars 9-12): the motif up an octave, more urgency.
    phrase_c = _phrase(
        "g5  ~  ab5 g5  ~   ~  eb5 f5 "     # Cm
        "g5  ~  f5  eb5 ~   ~  c5  ~  "     # Ab
        "d5  ~  eb5 g5  ~  f5  eb5 ~  "     # Eb
        "d5  ~  c5  ~   bb4 ~  c5  ~  "     # Bb : turn back down
    )
    # Phrase D (bars 13-16): climb to the climax note, big held resolution.
    phrase_d = _phrase(
        "c5  d5  eb5 f5  g5  ~  ab5 ~  "    # Fm  : scalar climb
        "g5  ~   f5  ~   eb5 ~  c5  ~  "    # Ab  : ease back
        "d5  ~   b4  ~   c5  d5 eb5 ~  "    # G   : leading-tone push
        "c5  ~   ~   ~   ~   ~  ~   ~  ",   # G   : final long tonic
        depth=0.22, rate=5.5,
    )
    lead = arrange(
        at(0, phrase_a), at(4, phrase_b),
        at(8, phrase_c), at(12, phrase_d),
    )
    # thin backing: held buzzer bass (LEFT) + a quiet slow arp shimmer and a
    # soft kick, both on CENTRE -- the kick outranks the shimmer on its hits.
    bass = bass_roots(inst=BASS, vol=14, p=PAN_L)
    arp = arp_line(PROG_FULL, speed=4, mode="up", inst=PLUCK, vol=7, p=PAN_C)
    kick = s(KICK).struct("x ~ ~ ~ x ~ ~ ~").vol(11).pan(PAN_C).priority(PRI_KICK)
    return stack(lead, bass, arp, kick)


def outro():
    # wind down to the pad; resolve with a Picardy-third C major shimmer.
    outro_prog = ("c3:min ab2:maj7 eb3:maj g3:maj "
                  "ab2:maj7 f3:min c3:maj c3:maj")
    # winding down: no bass, so the pad takes LEFT; the bell fades on RIGHT.
    pad = pad_progression(outro_prog, inst=PAD, vol=16, p=PAN_L, shimmer=4)
    bell = (
        note("g4 eb4 c4 g4 ab4 c4 c4 c4")  # descending to the tonic
        .slow(N_BARS)
        .s(BELL)                            # the ringing bell, fading out
        .vol(9)
        .pan(PAN_R)
    )
    return stack(pad, bell)


# ---------------------------------------------------------------------------
# Arrangement
# ---------------------------------------------------------------------------

# Section boundaries in *bars* (== cycles, since beats_per_cycle == 4).
SECTIONS = [
    ("intro", 0,  8),
    ("build", 8,  16),
    ("drop",  16, 31),
    ("fill",  31, 32),
    ("solo",  32, 48),
    ("outro", 48, 56),
]


def track():
    return arrange(
        at(0,  intro()),
        at(8,  build()),
        at(16, drop()),
        at(31, acid_fill()),
        at(32, solo()),
        at(48, outro()),
    )


def section_seconds(bpm=BPM, beats=BEATS):
    """Convert the bar-based SECTIONS into (name, t0, t1) seconds for the
    analysis report."""
    cps = bpm / 60.0 / beats           # cycles (bars) per second
    sec_per_bar = 1.0 / cps
    return [(name, b0 * sec_per_bar, b1 * sec_per_bar)
            for name, b0, b1 in SECTIONS]


def total_seconds(bpm=BPM, beats=BEATS):
    cps = bpm / 60.0 / beats
    end_bar = max(b1 for _, _, b1 in SECTIONS)
    return end_bar / cps


def render_track(wav=None, bpm=BPM, psg_path=None):
    seconds = total_seconds(bpm)
    return render(
        track(),
        bpm=bpm,
        beats_per_cycle=BEATS,
        seconds=seconds,
        fps=50.0,
        chip_type=__import__("pyayay").ChipType.YM,  # YM is a touch warmer
        master_volume=0.5,
        wav=wav,
        psg_path=psg_path,
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    report_only = "--report-only" in argv
    args = [a for a in argv if not a.startswith("--")]
    out = None if report_only else (args[0] if args else "crystal_decline.wav")

    # write a sibling .psg register dump next to the WAV (same stem)
    psg_out = os.path.splitext(out)[0] + ".psg" if out else None

    psg, L, R = render_track(wav=out, psg_path=psg_out)

    from ay_analyze import print_report
    print_report(L, R, 44100, psg=psg, fps=50.0,
                 sections=section_seconds(), window_s=2.0)
    if out:
        print(f"\n-> wrote {out} "
              f"({len(L)} samples, {len(L)/44100:.1f}s, "
              f"peak {np.abs(np.concatenate([L,R])).max():.3f})")
        print(f"-> wrote {psg_out} "
              f"({psg.shape[0]} frames, {os.path.getsize(psg_out)} bytes)")


if __name__ == "__main__":
    main()
