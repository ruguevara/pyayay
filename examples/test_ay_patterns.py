"""Tests for the ay_patterns engine.

Run from the examples/ directory:

    pytest test_ay_patterns.py

These cover the pure pattern algebra (deterministic, fast) plus a couple of
light end-to-end render smoke tests that the AY tricks actually reach the
register stream.
"""

import math

import numpy as np
import pytest

import ay_patterns as ap
from ay_patterns import (
    Buzzer, Percussion, Tone,
    Span, arrange, at, chord, note, render, s, stack, cat, window,
)


# -- note / frequency math --------------------------------------------------

def test_note_name_to_midi():
    assert ap.note_name_to_midi("c4") == 60
    assert ap.note_name_to_midi("a4") == 69
    assert ap.note_name_to_midi("c#4") == 61
    assert ap.note_name_to_midi("db4") == 61  # enharmonic
    assert ap.note_name_to_midi("c5") == 72


def test_midi_to_freq():
    assert ap.midi_to_freq(69) == pytest.approx(440.0)
    assert ap.midi_to_freq(57) == pytest.approx(220.0)  # one octave down
    assert ap.midi_to_freq(81) == pytest.approx(880.0)  # one octave up


def test_tone_period_formula():
    clock = 1773400
    # period = clock / (16 * freq); verified empirically against the emulator
    assert ap.tone_period_for_freq(440.0, clock) == round(clock / (16 * 440))
    # clamped into the 12-bit range
    assert ap.tone_period_for_freq(0.0, clock) == 0
    assert ap.tone_period_for_freq(1e9, clock) >= 1
    assert ap.tone_period_for_freq(0.001, clock) <= 0xFFF


def test_env_period_formula():
    clock = 1773400
    # triangle: 2 ramps per cycle
    assert ap.env_period_for_freq(50.0, clock, ramps_per_cycle=2) == \
        round(clock / (256 * 2 * 50))
    # saw: 1 ramp per cycle -> twice the period of the triangle at same pitch
    # (period = clock / (256 * ramps * freq), so fewer ramps -> larger period)
    saw = ap.env_period_for_freq(50.0, clock, ramps_per_cycle=1)
    tri = ap.env_period_for_freq(50.0, clock, ramps_per_cycle=2)
    assert saw == pytest.approx(tri * 2, abs=1)


# -- pattern algebra --------------------------------------------------------

def _onsets(pat, b=0, e=1):
    return sorted(float(ev.whole.begin) for ev in pat.query(Span(b, e)))


def _values(pat, b=0, e=1):
    evs = sorted(pat.query(Span(b, e)), key=lambda ev: ev.whole.begin)
    return [ev.value for ev in evs]


def test_pure_one_per_cycle():
    p = ap.pure(ap.Note(midi=60))
    assert len(p.query(Span(0, 1))) == 1
    assert len(p.query(Span(0, 3))) == 3


def test_seq_divides_cycle():
    p = note("c4 e4 g4 ~")  # rest is dropped
    assert _onsets(p) == [0.0, 0.25, 0.5]
    midis = [v.midi for v in _values(p)]
    assert midis == [60, 64, 67]


def test_fast_slow():
    p = note("c4 e4")
    assert len(p.query(Span(0, 1))) == 2
    assert len(p.fast(2).query(Span(0, 1))) == 4
    assert len(p.slow(2).query(Span(0, 2))) == 2  # one full cycle stretched over 2


def test_rev():
    p = note("c4 e4 g4 a4")
    midis = [v.midi for v in _values(p.rev())]
    assert midis == [69, 67, 64, 60]


def test_add_transpose():
    p = note("c4").add(12)
    assert _values(p)[0].midi == 72


def test_every():
    # every 2nd cycle, transpose up an octave
    p = note("c4").every(2, lambda x: x.add(12))
    c0 = p.query(Span(0, 1))[0].value.midi
    c1 = p.query(Span(1, 2))[0].value.midi
    c2 = p.query(Span(2, 3))[0].value.midi
    assert c0 == 72  # cycle 0 transformed
    assert c1 == 60  # cycle 1 untouched
    assert c2 == 72  # cycle 2 transformed


def test_struct():
    d = s(Percussion("kick")).struct("x ~ x ~")
    assert _onsets(d) == [0.0, 0.5]


def test_cat_one_per_cycle():
    p = cat(note("c4"), note("e4"))
    assert p.query(Span(0, 1))[0].value.midi == 60
    assert p.query(Span(1, 2))[0].value.midi == 64
    assert p.query(Span(2, 3))[0].value.midi == 60  # wraps


def test_stack_simultaneous():
    p = stack(note("c4"), note("e4"))
    evs = p.query(Span(0, 1))
    assert len(evs) == 2
    assert {ev.value.midi for ev in evs} == {60, 64}


# -- chords and arpeggios ---------------------------------------------------

def test_chord_parse():
    c = ap._parse_token("c4:maj")
    assert isinstance(c, ap.Chord)
    assert c.midis == (60, 64, 67)


def test_arp_up_down():
    up = [ev.value.midi for ev in chord("c4:maj").arp("up").query(Span(0, 1))]
    assert up == [60, 64, 67]
    down = [ev.value.midi for ev in chord("c4:maj").arp("down").query(Span(0, 1))]
    assert down == [67, 64, 60]
    updown = [ev.value.midi for ev in chord("c4:maj").arp("updown").query(Span(0, 1))]
    assert updown == [60, 64, 67, 64]


def test_arp_fills_span():
    # three notes evenly fill the cycle
    evs = sorted(chord("c4:maj").arp("up").query(Span(0, 1)),
                 key=lambda e: e.whole.begin)
    assert [float(e.whole.begin) for e in evs] == \
        pytest.approx([0.0, 1 / 3, 2 / 3])


def test_slow_then_stutter_then_arp_tracks_progression():
    # The chip-chord idiom used by the demoscene track: an 8-bar progression
    # arpeggiated fast within each bar must keep one chord per bar.
    prog = "c4:min ab3:maj"  # bar 0 = Cm (C Eb G), bar 1 = Ab (Ab C Eb)
    pat = chord(prog).slow(2).stutter(4).arp("up")
    cm = {0, 3, 7}            # C, Eb, G
    ab = {8, 0, 3}           # Ab, C, Eb
    bar0 = {round(e.value.midi) % 12 for e in pat.query(Span(0, 1))}
    bar1 = {round(e.value.midi) % 12 for e in pat.query(Span(1, 2))}
    assert bar0 == cm
    assert bar1 == ab


# -- arrangement primitives -------------------------------------------------

def test_window_gates_onsets():
    p = note("c4")  # one onset per cycle at integer cycles
    w = window(p, 1, 3)
    assert _onsets(w, 0, 5) == [1.0, 2.0]  # only cycles 1 and 2 survive


def test_arrange_places_sections():
    a = note("c4").s(Tone())
    b = note("e4 g4").s(Tone())
    song = arrange(at(0, a), at(2, b))
    assert {round(e.value.midi) for e in song.query(Span(0, 1))} == {60}
    assert {round(e.value.midi) for e in song.query(Span(1, 2))} == {60}
    assert {round(e.value.midi) for e in song.query(Span(2, 3))} == {64, 67}


def test_arrange_shift_aligns_pattern_origin():
    # a pattern written from cycle 0 should play from its start at the section,
    # not from wherever its global cycle happens to be.
    melody = note("c4 d4 e4 f4")
    song = arrange(at(8, melody))
    first = sorted(song.query(Span(8, 9)), key=lambda e: e.whole.begin)
    assert [round(e.value.midi) for e in first] == [60, 62, 64, 65]


def test_stutter_ratchets():
    st = note("c4").stutter(4)
    assert _onsets(st) == [0.0, 0.25, 0.5, 0.75]
    # value is preserved
    assert all(round(e.value.midi) == 60 for e in st.query(Span(0, 1)))


def test_glitch_is_deterministic_and_bounded():
    g = note("c4 c4 c4 c4").glitch(amount=12, seed=3)
    m1 = [round(e.value.midi) for e in sorted(g.query(Span(0, 1)),
                                              key=lambda e: e.whole.begin)]
    m2 = [round(e.value.midi) for e in sorted(g.query(Span(0, 1)),
                                              key=lambda e: e.whole.begin)]
    assert m1 == m2                                   # deterministic
    assert all(abs(m - 60) <= 12 for m in m1)         # within +/- amount
    assert any(m != 60 for m in m1)                   # actually glitched


def test_gain_scales_volume_but_not_envelope():
    half = note("c4").vol(10).gain(0.5)
    assert _values(half)[0].volume == 5
    # envelope-mode notes (volume 16) are left untouched
    buzz = note("c4").s(Buzzer()).vol(16).gain(0.5)
    assert _values(buzz)[0].volume == 16


# -- end-to-end render smoke tests ------------------------------------------

def test_render_returns_shapes():
    p = note("c4 e4 g4").s(Tone())
    psg, left, right = render(p, bpm=120, seconds=1)
    assert psg.shape[1] == 14
    assert len(left) == len(right) > 0
    assert np.abs(left).max() <= 1.0  # no clipping


def test_render_tone_is_audible():
    p = note("a4").s(Tone())
    _, left, _ = render(p, seconds=1)
    assert np.abs(left).mean() > 0.01


def test_buzzer_sets_envelope_registers():
    p = note("c2").s(Buzzer(shape="saw")).vol(16)
    psg, _, _ = render(p, seconds=1)
    frame = psg[3]  # a settled frame
    assert frame[8] == 0x10           # R8: envelope-mode flag
    assert frame[11] or frame[12]     # R11/R12: envelope period set
    # R13 envelope shape is the saw enum (UP_UP_C == 12)
    assert frame[13] == int(Buzzer._SHAPES["saw"][0])


def test_buzzer_bass_is_clean_pitched_tone():
    """Regression: the pure-envelope buzzer must render as a clean pitched saw,
    not the broadband mush you get when (a) the envelope is retriggered every
    frame by re-latching R13, or (b) a same-frequency square tone fights it.

    We measure spectral clarity = (strongest FFT bin) / (total energy).  A
    clean buzzer concentrates energy in a fundamental + harmonics; noise
    spreads it out.  The dirty same-frequency tone+env case scored ~0.011; a
    clean pure buzzer scores several times higher.
    """
    from ay_patterns import buzz_bass

    p = note("c2").s(buzz_bass(shape="saw")).vol(16)
    _, left, right = render(p, seconds=2, chip_type=__import__("pyayay").ChipType.YM)
    mono = (left + right) * 0.5
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    clarity = float(spec.max() / (spec.sum() + 1e-9))
    assert clarity > 0.04, f"buzzer bass looks like noise (clarity={clarity:.4f})"


def test_render_suppresses_r13_retrigger():
    """The renderer must not re-write R13 (envelope shape) on frames where it is
    unchanged -- doing so retriggers the envelope and smears buzzer tones."""
    # build the same mask the renderer builds and confirm a steady buzzer note
    # only writes R13 on its first frame.
    p = note("c2").s(Buzzer(shape="saw")).vol(16)
    psg, _, _ = render(p, seconds=1)
    r13 = psg[:, 13]
    # the shape value is constant across the held note; ensure it's set once and
    # stays constant (the suppression is internal, but a stable R13 column with a
    # non-zero shape is the observable invariant).
    assert (r13 == r13[3]).all()
    assert r13[3] != 0


def test_sweep_changes_tone_period():
    p = note("a3").s(Tone()).sweep(12.0)  # +1 octave per second
    psg, _, _ = render(p, seconds=1)
    start = psg[1, 0] | (psg[1, 1] << 8)
    later = psg[40, 0] | (psg[40, 1] << 8)
    assert later < start  # rising pitch -> shorter period


def test_arp_fast_cycles_periods():
    # the AY hardware-arpeggio trick: chord notes cycled per frame
    p = chord("c4:maj").arp("up").fast(16).s(Tone())
    psg, _, _ = render(p, seconds=1)
    periods = {psg[i, 0] | (psg[i, 1] << 8) for i in range(5, 30)}
    assert len(periods) >= 2


def test_percussion_uses_noise():
    p = s(Percussion("hat")).fast(8)
    psg, _, _ = render(p, seconds=1)
    # mixer R7 bits 3-5 are noise enables (active low); at least one frame must
    # have a noise channel enabled (a cleared noise bit).
    noise_enabled = ((psg[:, 7] >> 3) & 0b111) != 0b111
    assert noise_enabled.any()


def test_render_zero_duration_errors():
    with pytest.raises(ValueError):
        render(note("c4"), seconds=0)


def test_bpm_vs_cps_consistent():
    # bpm=120, 4 beats/cycle -> cps = 0.5; 2 seconds = 1 cycle
    p = note("c4")
    psg_bpm, _, _ = render(p, bpm=120, beats_per_cycle=4, seconds=2, fps=50)
    psg_cps, _, _ = render(p, cps=0.5, seconds=2, fps=50)
    assert psg_bpm.shape == psg_cps.shape
    assert np.array_equal(psg_bpm, psg_cps)


# -- megademo integration ---------------------------------------------------

def test_megademo_structure_and_arc():
    """Lock in the demoscene track: it renders cleanly, the harmony tracks the
    C-minor progression in the bass, and the section dynamics rise into the
    solo then fall for the outro."""
    import ay_megademo as mega
    from ay_analyze import section_stats

    psg, L, R = mega.render_track()
    sr = 44100

    # renders long, no clipping, not silent
    assert len(L) / sr > 60
    assert np.abs(np.concatenate([L, R])).max() < 0.999
    assert np.abs(L).mean() > 0.01

    secs = {name: (t0, t1) for name, t0, t1 in mega.section_seconds()}
    db = {name: section_stats(L, R, sr, name, t0, t1)["rms_db"]
          for name, (t0, t1) in secs.items()}

    # the arc: intro is the quietest "musical" section; the drop lifts clearly
    # above the build; the solo is the loudest; the outro winds back down.
    assert db["drop"] > db["build"] + 1.0
    assert db["solo"] >= db["drop"] - 1.0
    assert db["outro"] < db["solo"] - 2.0
    assert db["intro"] < db["drop"]

    # bass tracks the progression: the first drop bar's lowest channel sits on
    # the tonic C; bar +2 (III = Eb) differs from it.
    fpc = 50.0 / (mega.BPM / 60.0 / mega.BEATS)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def root_pc(bar):
        from collections import Counter
        f0, f1 = int(round(bar * fpc)), int(round((bar + 1) * fpc))
        pcs = []
        for f in range(f0, f1):
            p = int(psg[f, 0]) | (int(psg[f, 1]) << 8)
            if p > 0:
                midi = round(69 + 12 * math.log2((1773400 / (16 * p)) / 440))
                pcs.append(midi % 12)
        return Counter(pcs).most_common(1)[0][0] if pcs else None

    assert names[root_pc(16)] == "C"        # bar 16: i (Cm)
    assert names[root_pc(18)] == "D#"       # bar 18: III (Eb)
