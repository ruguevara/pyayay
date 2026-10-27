from array import array

import numpy as np
import pytest

from pyayay import Ayumi

SR = 22050
CLOCK = 1772700
SAMPLES = 441
T_FRAME = 70908


def make_ay():
    ay = Ayumi(SR, CLOCK)
    ay.set_master_volume(0.5)
    return ay


def to_int16(x):
    return np.clip(np.rint(x * 32767.0), -32768, 32767).astype(np.int16)


def sample_index(t, samples=SAMPLES, t_frame=T_FRAME):
    return (t * samples + t_frame - 1) // t_frame


def reference_split(ay, kwrites, samples=SAMPLES, remove_dc=True):
    """Render via R[...] = v and process_block, applying (k, reg, value) before sample k."""
    left = np.zeros(samples, dtype=np.float32)
    right = np.zeros(samples, dtype=np.float32)
    pos = 0
    for k, reg, value in kwrites:
        if k > pos:
            ay.process_block(left[pos:k], right[pos:k], k - pos, remove_dc=remove_dc)
            pos = k
        ay.R[reg] = value
    if pos < samples:
        ay.process_block(left[pos:], right[pos:], samples - pos, remove_dc=remove_dc)
    return left.astype(np.float64), right.astype(np.float64)


def reference_frame(ay, writes, samples=SAMPLES, t_frame=T_FRAME, remove_dc=True):
    """Same frame rendered via R[...] = v and process_block segments split at the computed k."""
    kwrites = [(sample_index(t, samples, t_frame), reg, value) for t, reg, value in writes]
    return reference_split(ay, kwrites, samples, remove_dc)


def flat(writes):
    return array('I', [x for w in writes for x in w])


def as_stereo(data):
    return np.frombuffer(data, dtype='<i2').reshape(-1, 2)


def as_mono(data):
    return np.frombuffer(data, dtype='<i2')


SETUP = [(0, 7, 0b00111000), (0, 0, 100), (0, 1, 0), (0, 2, 150), (0, 4, 37), (0, 5, 1),
         (0, 8, 15), (0, 9, 12), (0, 10, 16), (0, 11, 0), (0, 12, 2), (0, 13, 14)]

WRITES = [(3000, 0, 50), (3000, 8, 10), (10001, 6, 7), (10001, 7, 0b00110000),
          (40000, 13, 14), (40160, 13, 14), (40160, 1, 2), (55555, 9, 0), (70907, 8, 3)]


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("remove_dc", [False, True])
def test_equivalence_with_process_block(stereo, remove_dc):
    ay, ref = make_ay(), make_ay()
    frames = [SETUP, WRITES, [], WRITES[::2]]
    for writes in frames:
        data = ay.render_frame(flat(writes), SAMPLES, T_FRAME, stereo=stereo, remove_dc=remove_dc)
        left, right = reference_frame(ref, writes, remove_dc=remove_dc)
        if stereo:
            got = as_stereo(data)
            assert got.shape == (SAMPLES, 2)
            np.testing.assert_array_equal(got[:, 0], to_int16(left))
            np.testing.assert_array_equal(got[:, 1], to_int16(right))
        else:
            got = as_mono(data)
            assert got.shape == (SAMPLES,)
            np.testing.assert_array_equal(got, to_int16((left + right) * 0.5))


def test_input_shapes():
    writes = SETUP + WRITES
    a, b, c = make_ay(), make_ay(), make_ay()
    flat_out = a.render_frame(flat(writes), SAMPLES, T_FRAME, stereo=True)
    np_flat_out = b.render_frame(np.array(writes, dtype=np.uint32).ravel(), SAMPLES, T_FRAME, stereo=True)
    np_2d_out = c.render_frame(np.array(writes, dtype=np.uint32), SAMPLES, T_FRAME, stereo=True)
    assert flat_out == np_flat_out == np_2d_out
    assert len(flat_out) == SAMPLES * 4


def test_empty_writes():
    ay, ref = make_ay(), make_ay()
    ay.render_frame(flat(SETUP), SAMPLES, T_FRAME)
    reference_frame(ref, SETUP)
    for empty in (array('I'), np.zeros((0, 3), dtype=np.uint32)):
        data = ay.render_frame(empty, SAMPLES, T_FRAME)
        left, right = reference_frame(ref, [])
        np.testing.assert_array_equal(as_mono(data), to_int16((left + right) * 0.5))


def tone_on_ay():
    # Constant level on channel A (tone and noise off), DC filter off
    ay = Ayumi(SR, CLOCK)
    ay.set_pan(0, 0.5)
    ay.render_frame(flat([(0, 7, 0b00111111), (0, 8, 15)]), SAMPLES, T_FRAME, remove_dc=False)
    return ay


def mono_split(ay, kwrites, samples=SAMPLES):
    left, right = reference_split(ay.copy(), kwrites, samples, remove_dc=False)
    return to_int16((left + right) * 0.5)


def check_placement(t, k, samples=SAMPLES, t_frame=T_FRAME):
    """A volume drop at t lands exactly before sample k: not one sample earlier or later."""
    ay = tone_on_ay()
    out = as_mono(ay.copy().render_frame(flat([(t, 8, 0)]), samples, t_frame, remove_dc=False))
    np.testing.assert_array_equal(out, mono_split(ay, [(k, 8, 0)], samples))
    if k > 0:
        assert not np.array_equal(out, mono_split(ay, [(k - 1, 8, 0)], samples))
    if k < samples:
        assert not np.array_equal(out, mono_split(ay, [(k + 1, 8, 0)], samples))


def test_placement_t0_affects_sample0():
    check_placement(0, 0)


def test_placement_after_boundary_goes_to_next_sample():
    # Sample 10 starts at t = 10 * T_FRAME / SAMPLES = 1607.89...
    t_after = (10 * T_FRAME + SAMPLES - 1) // SAMPLES  # first integer t after boundary 10
    assert t_after == 1608
    check_placement(1, 1)
    check_placement(t_after - 1, 10)
    check_placement(t_after, 11)


def test_placement_exact_boundary():
    # t_frame divisible by samples: t = 5 * 160 is exactly on boundary 5
    samples, t_frame = 441, 441 * 160
    check_placement(5 * 160 - 1, 5, samples, t_frame)
    check_placement(5 * 160, 5, samples, t_frame)
    check_placement(5 * 160 + 1, 6, samples, t_frame)


def test_same_k_last_write_wins():
    t = 2000
    assert sample_index(t) == sample_index(t + 2)
    a, b = tone_on_ay(), tone_on_ay()
    out_a = a.render_frame(flat([(t, 8, 3), (t + 1, 8, 9), (t + 2, 8, 0)]), SAMPLES, T_FRAME)
    out_b = b.render_frame(flat([(t, 8, 0)]), SAMPLES, T_FRAME)
    assert out_a == out_b


def test_write_at_end_of_frame_holds_for_next_frame():
    t = T_FRAME - 1
    assert sample_index(t) == SAMPLES
    ay = tone_on_ay()
    base_ay = ay.copy()
    out = ay.render_frame(flat([(t, 8, 0)]), SAMPLES, T_FRAME, remove_dc=False)
    assert out == base_ay.render_frame(array('I'), SAMPLES, T_FRAME, remove_dc=False)
    # The write is in effect from sample 0 of the next frame
    nxt = as_mono(ay.render_frame(array('I'), SAMPLES, T_FRAME, remove_dc=False))
    np.testing.assert_array_equal(nxt, mono_split(base_ay, [(0, 8, 0)]))


def test_copy_registers_are_independent():
    a = Ayumi()
    b = a.copy()
    b.R[0] = 55
    assert a.get_tone_period(0) == 1
    assert b.get_tone_period(0) == 55


def test_r13_rewrite_restarts_envelope():
    # Writing the same shape again restarts the envelope, as on a real AY
    setup = [(0, 7, 0b00111111), (0, 8, 16), (0, 11, 0), (0, 12, 1), (0, 13, 12)]
    ay = make_ay()
    ay.render_frame(flat(setup), SAMPLES, T_FRAME, remove_dc=False)
    a, b = ay.copy(), ay.copy()
    out_a = as_mono(a.render_frame(flat([(30000, 13, 12)]), SAMPLES, T_FRAME, remove_dc=False))
    out_b = as_mono(b.render_frame(array('I'), SAMPLES, T_FRAME, remove_dc=False))
    k = sample_index(30000)
    np.testing.assert_array_equal(out_a[:k], out_b[:k])
    assert not np.array_equal(out_a, out_b)
    left, right = reference_split(ay.copy(), [(k, 13, 12)], remove_dc=False)
    np.testing.assert_array_equal(out_a, to_int16((left + right) * 0.5))


def test_mono_is_average_of_stereo():
    a, b = make_ay(), make_ay()
    for i, p in enumerate((0.1, 0.9, 0.3)):
        a.set_pan(i, p)
        b.set_pan(i, p)
    for writes in (SETUP, WRITES):
        mono = as_mono(a.render_frame(flat(writes), SAMPLES, T_FRAME)).astype(np.int32)
        st = as_stereo(b.render_frame(flat(writes), SAMPLES, T_FRAME, stereo=True)).astype(np.int32)
        avg = (st[:, 0] + st[:, 1]) / 2
        # Mono is rounded from the float average, stereo values are rounded per channel,
        # so the two can differ by one LSB
        assert np.max(np.abs(mono - np.rint(avg))) <= 1
        assert np.mean(mono == np.rint(avg)) > 0.4


def test_clamping():
    ay = Ayumi(SR, CLOCK)
    for i in range(3):
        ay.set_pan(i, 0.0)
    ay.set_master_volume(4.0)
    data = as_stereo(ay.render_frame(flat([(0, 7, 0b00111111), (0, 8, 15), (0, 9, 15), (0, 10, 15)]),
                                     SAMPLES, T_FRAME, stereo=True, remove_dc=False))
    assert data[:, 0].max() == 32767


@pytest.mark.parametrize("writes", [
    [(100, 0, 1), (99, 0, 1)],          # unsorted t
    [(T_FRAME, 0, 1)],                  # t == t_frame
    [(T_FRAME + 5, 0, 1)],              # t > t_frame
    [(0, 14, 1)],                       # reg 14
    [(0, 0, 256)],                      # value 256
])
def test_errors_values(writes):
    ay = make_ay()
    with pytest.raises(ValueError):
        ay.render_frame(flat(writes), SAMPLES, T_FRAME)


@pytest.mark.parametrize("buf", [
    array('H', [0, 0, 1]),
    array('i', [0, 0, 1]),
    array('Q', [0, 0, 1]),
    np.array([0, 0, 1], dtype=np.float32),
    np.array([0, 0, 1, 0], dtype=np.uint32),            # not a multiple of 3
    np.zeros((2, 4), dtype=np.uint32),                  # wrong shape
    np.zeros((2, 3, 1), dtype=np.uint32),               # wrong ndim
    np.zeros((2, 6), dtype=np.uint32)[:, ::2],          # non-contiguous
    b"\x00" * 12,
])
def test_errors_buffer(buf):
    ay = make_ay()
    with pytest.raises(ValueError):
        ay.render_frame(buf, SAMPLES, T_FRAME)


def test_errors_sizes():
    ay = make_ay()
    with pytest.raises(ValueError):
        ay.render_frame(array('I'), 0, T_FRAME)
    with pytest.raises(ValueError):
        ay.render_frame(array('I'), SAMPLES, 0)


def test_error_leaves_state_untouched():
    a, b = make_ay(), make_ay()
    with pytest.raises(ValueError):
        a.render_frame(flat([(0, 8, 15), (0, 14, 0)]), SAMPLES, T_FRAME)
    assert a.render_frame(array('I'), SAMPLES, T_FRAME) == b.render_frame(array('I'), SAMPLES, T_FRAME)
