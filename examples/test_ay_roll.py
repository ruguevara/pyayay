"""Tests for the ay_roll tracker-style PSG viewer.

Run from the examples/ directory:

    pytest test_ay_roll.py

These check that the roll *decodes* a rendered PSG stream back to the right
musical reading (note names, mixer flags, envelope note), and that the megademo
intro shimmer is on a steady onset grid (the bug ay_roll was built to find:
arpeggiating a bare chord ties the pulse to the chord size, so triad bars and
maj7 bars pulse at different rates and the pad lurches).
"""

import numpy as np

import ay_roll as rl
from ay_patterns import Buzzer, Tone, note, render


# -- register decode --------------------------------------------------------

def test_period_to_midi_roundtrip():
    clock = rl.DEFAULT_CLOCK
    # a tone period for A4 should decode back to ~69
    from ay_patterns import tone_period_for_freq
    p = tone_period_for_freq(440.0, clock)
    midi = rl._period_to_midi(p, clock, 16.0)
    assert round(midi) == 69
    assert rl._midi_name(69) == "A-4"
    assert rl._midi_name(60) == "C-4"
    assert rl._midi_name(61) == "C#4"


def test_decode_tone_note_name():
    # render a single sustained C-4 tone and read the note back
    psg, _, _ = render(note("c4"), bpm=120, beats_per_cycle=4, cycles=1,
                       fps=50.0)
    cells = rl.decode_frame(psg[10])
    a = cells[1]               # default pan is centre -> channel B (ch1)
    assert a.tone_on and not a.env_on
    assert a.note == "C-4"


def test_decode_envelope_note_and_detune():
    # a pure buzzer bass: envelope mode on, env note tracks the tone note
    psg, _, _ = render(note("c2").s(Buzzer(shape="tri")),
                       bpm=120, beats_per_cycle=4, cycles=1, fps=50.0)
    cell = next(c for c in rl.decode_frame(psg[10]) if c.env_on)
    assert cell.volume >= 16            # envelope mode flag
    assert cell.env_note is not None
    # the buzzer plays the note pitch: env note within a semitone of the tone
    assert abs(cell.env_detune) < 1.0
    assert cell.env_glyph == "/\\"      # triangle


# -- roll rendering ---------------------------------------------------------

def test_roll_marks_bars_and_notes():
    psg, _, _ = render(note("c4 e4 g4 c5"), bpm=120, beats_per_cycle=4,
                       cycles=1, fps=50.0)
    text = rl.roll(psg, bpm=120, beats_per_cycle=4, fps=50.0,
                   rows_per_beat=4, bars=(0, 1), show_env=False)
    lines = text.splitlines()
    assert lines[0].startswith("row")
    # the four notes should all appear as fresh triggers somewhere in the roll
    body = "\n".join(lines[2:])
    for n in ("C-4", "E-4", "G-4", "C-5"):
        assert n in body


def _onset_frames(psg, ch):
    period = (psg[:, ch * 2].astype(int)
              | (psg[:, ch * 2 + 1].astype(int) << 8))
    return np.nonzero(np.diff(period) != 0)[0] + 1


def test_intro_shimmer_more_regular_than_bare_arp():
    """Regression for the intro lurch.

    Arpeggiating a *bare* chord (``shimmer=1``) ties the pad's onset spacing to
    the chord size: a triad spreads 3 notes over the bar, a maj7 spreads 4, so
    the gaps between onsets jump from bar to bar and the pad audibly speeds up
    on the maj7 bars and drags on the triads.  Stuttering onto a fixed
    ``shimmer`` grid first regularises the spacing.

    Compare the two constructions on the same progression and assert the
    fixed-grid pad has markedly *more even* inter-onset gaps (lower coefficient
    of variation) than the bare-arp one -- i.e. it stops lurching."""
    from ay_patterns import chord
    from ay_megademo import PROG_FULL, PAD, N_BARS, BPM, BEATS, PAN_L

    def cov_of_gaps(pat):
        psg, _, _ = render(pat, bpm=BPM, beats_per_cycle=BEATS, cycles=N_BARS,
                           fps=50.0)
        gaps = np.diff(_onset_frames(psg, 0)).astype(float)
        return gaps.std() / gaps.mean()

    base = chord(PROG_FULL).slow(N_BARS)
    bare = base.stutter(1).arp("up").s(PAD).vol(16).pan(PAN_L)
    fixed = base.stutter(4).arp("up").s(PAD).vol(16).pan(PAN_L)
    assert cov_of_gaps(fixed) < cov_of_gaps(bare) * 0.8


def test_intro_pad_pulse_lands_on_the_beat_grid():
    """The fixed-grid shimmer's *strong* onsets (the stutter pulse) land on the
    quarter-note grid every bar, regardless of chord size.  We check that each
    beat of the intro contains a pad onset very close to the beat boundary."""
    from ay_megademo import intro, BPM, BEATS
    psg, _, _ = render(intro(), bpm=BPM, beats_per_cycle=BEATS, cycles=8,
                       fps=50.0)
    cps = BPM / 60.0 / BEATS
    frames_per_beat = 50.0 / cps / BEATS
    onsets = _onset_frames(psg, 0)
    n_beats = int(psg.shape[0] / frames_per_beat)
    misses = 0
    for b in range(n_beats):
        target = b * frames_per_beat
        # is there a pad onset within a third of a beat of this beat boundary?
        if not any(abs(f - target) <= frames_per_beat / 3 for f in onsets):
            misses += 1
    # allow a couple of beats where a note legitimately holds across the grid
    assert misses <= 2, f"{misses}/{n_beats} beats have no on-grid pad pulse"
