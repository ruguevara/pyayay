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
    ADSR, Buzzer, Percussion, Sample, Tone,
    Span, arrange, at, chord, note, render, s, stack, cat, window, tone_noise,
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
    # .pan("L") pins the lone voice to channel 0 so we can read its registers
    # (a centred voice now lands on channel 1; see the pan-bucket channel model).
    p = note("c2").s(Buzzer(shape="saw")).vol(16).pan("L")
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
    p = note("c2").s(Buzzer(shape="saw")).vol(16).pan("L")
    psg, _, _ = render(p, seconds=1)
    r13 = psg[:, 13]
    # the shape value is constant across the held note; ensure it's set once and
    # stays constant (the suppression is internal, but a stable R13 column with a
    # non-zero shape is the observable invariant).
    assert (r13 == r13[3]).all()
    assert r13[3] != 0


def test_sweep_changes_tone_period():
    p = note("a3").s(Tone()).sweep(12.0).pan("L")  # +1 octave per second
    psg, _, _ = render(p, seconds=1)
    start = int(psg[1, 0]) | (int(psg[1, 1]) << 8)
    later = int(psg[40, 0]) | (int(psg[40, 1]) << 8)
    assert later < start  # rising pitch -> shorter period


def test_arp_fast_cycles_periods():
    # the AY hardware-arpeggio trick: chord notes cycled per frame
    p = chord("c4:maj").arp("up").fast(16).s(Tone()).pan("L")
    psg, _, _ = render(p, seconds=1)
    periods = {int(psg[i, 0]) | (int(psg[i, 1]) << 8) for i in range(5, 30)}
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


# -- ADSR envelopes / tracker "samples" -------------------------------------

def _period(psg, frame, ch=0):
    """Reconstruct a 12-bit tone period (avoiding uint8 overflow on the shift)."""
    return int(psg[frame, ch * 2]) | (int(psg[frame, ch * 2 + 1]) << 8)


def test_adsr_stages():
    # A/D/R are seconds, S is a level (Strudel semantics).
    e = ADSR(a=0.1, d=0.2, s=0.5, r=0.1, peak=1.0)
    assert e.at(0.0, 1.0) == pytest.approx(0.0)     # attack start
    assert e.at(0.05, 1.0) == pytest.approx(0.5)    # half-way up the attack
    assert e.at(0.1, 1.0) == pytest.approx(1.0)     # peak
    assert e.at(0.2, 1.0) == pytest.approx(0.75)    # half-way through decay
    assert e.at(0.5, 1.0) == pytest.approx(0.5)     # sustain level
    # release runs *after* note-off (dur), gliding the sustain level to 0
    assert e.at(1.0, 1.0) == pytest.approx(0.5)     # at note-off
    assert e.at(1.05, 1.0) == pytest.approx(0.25)   # half through release
    assert e.at(1.2, 1.0) == pytest.approx(0.0)     # fully released


def test_adsr_parse():
    e = ADSR.parse("0:0.16:0.3:0.1")
    assert (e.a, e.d, e.s, e.r) == (0.0, 0.16, 0.3, 0.1)
    # peak carries through for pitch envelopes
    p = ADSR.parse("0:0.1:0:0", peak=12)
    assert p.peak == 12 and p.at(0.0, 1.0) == pytest.approx(12.0)
    # passing an ADSR through is a no-op
    assert ADSR.parse(e) is e


def test_volume_adsr_decays_amplitude():
    # decay to silence over 0.3 s: R8 amplitude must fall across the note
    # (.pan("L") pins the lone voice to channel 0 -- see the pan-bucket model).
    p = note("c4").s(Tone()).vol(15).adsr("0:0.3:0:0").pan("L")
    psg, _, _ = render(p, bpm=120, seconds=1, fps=50)
    amp = psg[:, 8] & 0x0F
    assert amp[0] > amp[8] > amp[15]
    assert amp[15] == 0


def test_fluent_adsr_matches_struct():
    # the fluent .attack/.decay/.sustain/.release build the same vol env a
    # Sample carries, so both APIs produce the same amplitude curve.
    fluent = note("c4").s(Tone()).vol(15).attack(0.0).decay(0.2).sustain(0.25).release(0.0).pan("L")
    struct = note("c4").s(Sample(vol="0:0.2:0.25:0", volume=15)).pan("L")
    pf, _, _ = render(fluent, seconds=1, fps=50)
    ps, _, _ = render(struct, seconds=1, fps=50)
    assert np.array_equal(pf[:, 8] & 0x0F, ps[:, 8] & 0x0F)


def test_pitch_env_blips_then_settles():
    # +7 st blip decaying over 60 ms: the pitch starts high (short period) and
    # settles to the note's own period.
    p = note("c4").s(Tone()).penv("0:0.06:0:0", peak=7).pan("L")
    psg, _, _ = render(p, seconds=1, fps=50)
    assert _period(psg, 0) < _period(psg, 10)        # starts higher pitched
    settled = _period(psg, 20)
    plain = _period(render(note("c4").s(Tone()).pan("L"), seconds=1, fps=50)[0], 20)
    assert settled == plain                          # returns to the base pitch


def test_ornament_fakes_a_chord_on_one_channel():
    # ornament(0,4,7) cycles three pitches per frame -> three distinct periods
    p = note("c4").s(Tone()).ornament(0, 4, 7).pan("L")
    psg, _, _ = render(p, seconds=1, fps=50)
    periods = {_period(psg, i) for i in range(3, 15)}
    assert len(periods) == 3


def test_tone_noise_enables_both_generators():
    p = note("a3").s(tone_noise(noise_period=5)).pan("L")
    psg, _, _ = render(p, seconds=1, fps=50)
    mix = int(psg[5, 7])
    assert not (mix & 0b001)      # tone A enabled (active low)
    assert not (mix & 0b1000)     # noise A enabled
    assert int(psg[5, 6]) == 5    # noise period honoured


def test_noise_sweep_changes_noise_period():
    # noise colour swept across the volume envelope's decay
    p = note("c3").s(tone_noise(noise_period=4, vol="0:0.2:0:0", noise_sweep=(2, 20)))
    psg, _, _ = render(p, seconds=1, fps=50)
    early = int(psg[1, 6])
    late = int(psg[9, 6])
    assert early != late          # the noise period moves


def test_sample_env_mode_runs_the_buzzer():
    # a Sample with env=True drives the hardware envelope (buzzer), not the
    # volume ADSR: R8 shows the env-mode flag and R13 the saw shape.
    p = note("c2").s(Sample(tone=False, env=True, env_shape="saw")).pan("L")
    psg, _, _ = render(p, seconds=1, fps=50)
    assert int(psg[5, 8]) == 0x10
    assert int(psg[5, 13]) == int(Buzzer._SHAPES["saw"][0])


def test_sample_instruments_reach_every_trick():
    """A stack of the new Sample instruments renders cleanly and reaches every
    AY trick at once: amplitude ADSRs, tone+noise, and the hardware buzzer.

    (Self-contained -- it does not depend on which layers the demo track happens
    to have enabled.)  Each voice gets its own pan bucket so all three survive
    the per-bucket voice allocation instead of fighting for the centre channel.
    """
    song = stack(
        note("c2 ~ c2 g1").s(Buzzer(shape="saw")).vol(16).pan("L"),     # buzzer
        note("c4 e4 g4 e4").s(Tone()).adsr("0:0.12:0.3:0.05").pan("C"),  # vol ADSR
        note("a3 ~ a3 ~").s(tone_noise(noise_period=5,
                                       vol="0:0.1:0:0",
                                       noise_sweep=(2, 18))).pan("R"),  # tone+noise
    )
    psg, L, R = render(song, bpm=125, seconds=4)
    assert np.abs(np.concatenate([L, R])).max() < 0.999      # no clipping
    assert np.abs(L).mean() > 0.01                           # not silent
    # amplitude ADSRs vary the R8-R10 levels frame to frame
    assert (np.diff(psg[:, 8:11].astype(int), axis=0) != 0).any()
    # noise generator used (tone+noise voice)
    assert (((psg[:, 7] >> 3) & 0b111) != 0b111).any()
    # buzzer env-mode reached (R8-R10 bit-4 flag)
    assert ((psg[:, 8:11] & 0x10) != 0).any()


# -- pan buckets / virtual-channel model ------------------------------------

def _alloc(*specs):
    """Build _FrameVoice list from (bucket, priority, order) tuples and allocate.

    Returns the list of which order-index (or None) sits on each of ch0/1/2.
    """
    from ay_patterns import _FrameVoice, _allocate_channels, VoiceState
    fvs = []
    for order, (bucket, prio) in enumerate(specs):
        st = VoiceState(tone_on=True, tone_period=100 + order, volume=10)
        st._order = order  # tag so we can identify the survivor
        fvs.append(_FrameVoice(bucket=bucket, priority=prio, order=order, state=st))
    chans = _allocate_channels(fvs)
    return [None if c is None else c._order for c in chans]


def test_buckets_map_to_fixed_channels():
    from ay_patterns import PAN_L, PAN_C, PAN_R
    # L->ch0, C->ch1, R->ch2
    assert _alloc((PAN_L, 0), (PAN_C, 0), (PAN_R, 0)) == [0, 1, 2]


def test_per_bucket_steal_is_independent_and_leaves_channels_idle():
    from ay_patterns import PAN_C, PAN_R
    # two centre voices fight for ch1 (higher priority wins); ch0/ch2 stay idle
    # even though they are free -- "pan is the channel".
    res = _alloc((PAN_C, 5), (PAN_C, 9), (PAN_R, 1))
    assert res[0] is None            # left idle
    assert res[1] == 1               # the priority-9 centre voice won ch1
    assert res[2] == 2               # the right voice


def test_priority_then_order_breaks_ties():
    from ay_patterns import PAN_C
    # equal priority -> lower document/stack order wins the bucket
    assert _alloc((PAN_C, 3), (PAN_C, 3))[1] == 0
    # higher priority wins regardless of order
    assert _alloc((PAN_C, 1), (PAN_C, 8))[1] == 1


def test_any_bucket_spills_into_free_channels():
    from ay_patterns import PAN_L, PAN_ANY
    # ANY seats after L/C/R, into a free channel (centre preferred)
    res = _alloc((PAN_L, 0), (PAN_ANY, 0))
    assert res[0] == 0               # the explicit left voice
    assert res[1] == 1               # ANY spilled into the free centre channel
    assert res[2] is None
    # ANY is dropped when no channel is free
    from ay_patterns import PAN_C, PAN_R
    res2 = _alloc((PAN_L, 0), (PAN_C, 0), (PAN_R, 0), (PAN_ANY, 0))
    assert res2 == [0, 1, 2]         # the 4th (ANY) voice found no free channel


def test_numeric_pan_snaps_to_bucket():
    # back-compat: a numeric .pan(x) snaps to the nearest L/C/R channel
    from ay_patterns import render
    left = render(note("c4").s(Tone()).pan(0.0), seconds=1)[0]
    centre = render(note("c4").s(Tone()).pan(0.5), seconds=1)[0]
    right = render(note("c4").s(Tone()).pan(1.0), seconds=1)[0]
    # voice lands on ch0 / ch1 / ch2 respectively (its tone period is non-zero)
    assert (left[:, 0] != 0).any() and (left[:, 2] == 0).all()
    assert (centre[:, 2] != 0).any()      # ch1 fine byte at index 2
    assert (right[:, 4] != 0).any()       # ch2 fine byte at index 4


def test_pan_automation_moves_the_voice_between_channels():
    # .pan("L R") moves the voice from ch0 to ch2 across the cycle: the first
    # half of the bar plays on ch0, the second half on ch2.
    p = note("c4 c4").pan("L R").s(Tone())
    psg, _, _ = render(p, bpm=120, beats_per_cycle=4, seconds=2, fps=50)
    # first note (cycle [0,0.5)) -> ch0; second (cycle [0.5,1)) -> ch2
    first = psg[2]
    half = psg.shape[0] // 2
    second = psg[half + 2]
    assert int(first[0]) != 0 and int(first[4]) == 0     # ch0 used, ch2 silent
    assert int(second[4]) != 0 and int(second[0]) == 0   # ch2 used, ch0 silent


def test_pan_automation_is_audible_in_stereo():
    # a voice panned hard-left then hard-right shifts the stereo balance.
    p = note("c4 c4").pan("L R").s(Tone())
    _, L, R = render(p, bpm=120, beats_per_cycle=4, seconds=2, fps=50)
    half = len(L) // 2
    # first half: louder on the left; second half: louder on the right
    assert np.abs(L[:half]).mean() > np.abs(R[:half]).mean()
    assert np.abs(R[half:]).mean() > np.abs(L[half:]).mean()


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


# -- PSG file output --------------------------------------------------------

def _decode_psg(buf):
    """Minimal PSG-stream decoder: rebuild the [frames, 14] register array."""
    assert buf[:4] == b"PSG\x1a"
    i = 16
    regs = [0] * 14
    frames = []
    started = False
    while i < len(buf):
        b = buf[i]
        if b == 0xFD:
            break
        if b == 0xFF:                      # new frame
            if started:
                frames.append(regs.copy())
            started = True
            i += 1
            continue
        if b == 0xFE:                      # skip N*4 frames
            for _ in range(buf[i + 1] * 4):
                frames.append(regs.copy())
            i += 2
            continue
        regs[b] = buf[i + 1]               # reg, value pair
        i += 2
    frames.append(regs.copy())             # flush the final frame
    return np.array(frames, dtype=np.uint8)


def test_write_psg_roundtrip(tmp_path):
    # a small render exercising tone, buzzer (env) and a rest
    pat = stack(
        note("c4 e4 g4 ~").s(Tone()),
        note("c2").s(Buzzer(shape="tri")),
    )
    psg, _, _ = render(pat, bpm=120, beats_per_cycle=4, cycles=2, fps=50.0)

    path = tmp_path / "x.psg"
    ap.write_psg(str(path), psg, fps=50.0)
    buf = path.read_bytes()

    assert buf[:4] == b"PSG\x1a"
    assert buf[5] == 50                     # interrupt rate byte
    assert buf[-1] == 0xFD                  # end-of-data marker

    decoded = _decode_psg(buf)
    assert decoded.shape == (psg.shape[0], 14)
    # the delta stream must reconstruct the original registers exactly
    assert np.array_equal(decoded, psg[:, :14])


def test_render_writes_psg_file(tmp_path):
    path = tmp_path / "out.psg"
    psg, _, _ = render(note("c4 e4").s(Tone()), bpm=120, cycles=1, fps=50.0,
                       psg_path=str(path))
    assert path.exists()
    decoded = _decode_psg(path.read_bytes())
    assert np.array_equal(decoded, psg[:, :14])
