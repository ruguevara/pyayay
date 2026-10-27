from array import array
import math

import numpy as np
import pytest

from pyayay import Ayumi

SR = 44100
CLOCK = 1773447   # real 128K AY clock: a 882-sample block is 35468.94 cycles, not an integer
SPF = 882


def make_ay(clock=CLOCK, sr=SR):
    ay = Ayumi(sr, clock)
    ay.set_master_volume(0.5)
    return ay


def u64(writes):
    return np.array(writes, dtype=np.uint64).reshape(-1, 3)


def sample_at(t, sr=SR, clock=CLOCK):
    return -(-t * sr // clock)


def to_int16(x):
    return np.clip(np.rint(np.asarray(x, dtype=np.float64) * 32767.0), -32768, 32767).astype(np.int16)


def reference(ay, writes, n, remove_dc=True):
    """Render n samples via R[...] = v and process_block, writes placed at absolute sample_at(t)."""
    start = ay.samples_rendered
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    pos = 0
    for t, reg, value in writes:
        k = sample_at(int(t)) - start
        if k > pos:
            ay.process_block(left[pos:k], right[pos:k], k - pos, remove_dc=remove_dc)
            pos = k
        ay.R[int(reg)] = int(value)
    if pos < n:
        ay.process_block(left[pos:], right[pos:], n - pos, remove_dc=remove_dc)
    return left, right


SETUP = [(0, 7, 0b00111000), (0, 0, 100), (0, 2, 150), (0, 4, 37), (0, 5, 1),
         (0, 8, 15), (0, 9, 12), (0, 10, 16), (0, 12, 2), (0, 13, 14)]


def timer_writes(start_cycle, end_cycle, period, reg=8, values=(15, 9)):
    """A timer effect: alternate a register every `period` AY cycles on the absolute timeline."""
    first = -(-start_cycle // period) * period
    return [(t, reg, values[(t // period) % 2]) for t in range(first, end_cycle + 1, period)]


def blocks_of(ay, writes, sizes, dtype=np.float32, stereo=True, remove_dc=True):
    """Render consecutive blocks, giving each block exactly the writes that fall into it."""
    out = []
    i = 0
    for n in sizes:
        end_cycle = ay.cycle_at(ay.samples_rendered + n)
        j = i
        while j < len(writes) and writes[j][0] <= end_cycle:
            j += 1
        buf = np.zeros((n, 2) if stereo else n, dtype=dtype)
        ay.render(buf, u64(writes[i:j]) if j > i else None, remove_dc=remove_dc)
        out.append(buf)
        i = j
    assert i == len(writes)
    return np.concatenate(out)


def test_timeline_counter():
    ay = make_ay()
    assert ay.samples_rendered == 0
    ay.render(np.zeros(100, np.float32))
    L = np.zeros(50, np.float32)
    ay.process_block(L, L.copy(), 50)
    assert ay.samples_rendered == 150
    ay.samples_rendered = 10**12
    assert ay.samples_rendered == 10**12
    ay.reset(SR, CLOCK)
    assert ay.samples_rendered == 0
    assert ay.copy().samples_rendered == 0


def test_sample_at_cycle_at():
    ay = make_ay()
    rng = np.random.default_rng(1)
    for t in [0, 1, 40, 35468, 35469] + [int(x) for x in rng.integers(0, 2**40, 200)]:
        k = ay.sample_at(t)
        assert k == sample_at(t)
        # cycle_at(k) is the last cycle that lands at or before sample k
        assert ay.cycle_at(k) >= t
        assert ay.sample_at(ay.cycle_at(k) + 1) == k + 1
    assert ay.cycle_at(SPF) == SPF * CLOCK // SR == 35468


def test_large_values_exact():
    # Exact integer math far from 0: 1e12 samples is ~260 days at 44.1 kHz
    ay = make_ay()
    s = 10**12
    assert ay.cycle_at(s) == s * CLOCK // SR
    t = s * CLOCK // SR
    assert ay.sample_at(t) == -(-t * SR // CLOCK)
    assert ay.sample_at(2**64 - 1) == -(-(2**64 - 1) * SR // CLOCK)
    with pytest.raises(OverflowError):
        ay.cycle_at(2**64 - 1)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("remove_dc", [False, True])
def test_equivalence_with_process_block(stereo, remove_dc):
    ay, ref = make_ay(), make_ay()
    writes = SETUP + timer_writes(1, 4 * 35469, 5917) + [(70000, 13, 14), (70000, 1, 2)]
    writes.sort(key=lambda w: w[0])
    got = blocks_of(ay, writes, [SPF] * 4, stereo=stereo, remove_dc=remove_dc)
    left, right = reference(ref, writes, 4 * SPF, remove_dc=remove_dc)
    if stereo:
        np.testing.assert_array_equal(got[:, 0], left)
        np.testing.assert_array_equal(got[:, 1], right)
    else:
        np.testing.assert_array_equal(got, ((left.astype(np.float64) + right) * 0.5).astype(np.float32))
    assert ay.samples_rendered == 4 * SPF


def test_block_cuts_do_not_matter():
    # The point of the absolute timeline: the same writes give the same audio
    # however the render is split into blocks (no per-frame phase glitch).
    writes = SETUP + timer_writes(1, 20 * 35469, 2953) + timer_writes(1, 20 * 35469, 1009, reg=13, values=(12, 12))
    writes.sort(key=lambda w: w[0])   # stable: same-t writes keep their order
    total = 20 * SPF
    one = blocks_of(make_ay(), writes, [total])
    frames = blocks_of(make_ay(), writes, [SPF] * 20)
    rng = np.random.default_rng(7)
    sizes = []
    while sum(sizes) < total:
        sizes.append(int(min(rng.integers(1, 700), total - sum(sizes))))
    ragged = blocks_of(make_ay(), writes, sizes)
    np.testing.assert_array_equal(one, frames)
    np.testing.assert_array_equal(one, ragged)


def test_int16_matches_float():
    a, b = make_ay(), make_ay()
    writes = SETUP + timer_writes(1, 2 * 35469, 3001)
    f = blocks_of(a, writes, [SPF, SPF], dtype=np.float32)
    i = blocks_of(b, writes, [SPF, SPF], dtype=np.int16)
    np.testing.assert_array_equal(i, to_int16(f))
    a, b = make_ay(), make_ay()
    fm = blocks_of(a, writes, [SPF, SPF], dtype=np.float32, stereo=False)
    im = blocks_of(b, writes, [SPF, SPF], dtype=np.int16, stereo=False)
    np.testing.assert_array_equal(im, to_int16(fm))


def test_array_module_buffers():
    # No numpy needed: array('Q') writes, array('h') mono out, memoryview for stereo
    a, b = make_ay(), make_ay()
    flat = array('Q', [x for w in SETUP for x in w])
    mono = array('h', bytes(2 * SPF))
    a.render(mono, flat)
    ref = np.zeros(SPF, np.int16)
    b.render(ref, u64(SETUP))
    assert list(mono) == ref.tolist()
    stereo = array('h', bytes(4 * SPF))
    a.render(memoryview(stereo).cast('B').cast('h', [SPF, 2]))
    ref2 = np.zeros((SPF, 2), np.int16)
    b.render(ref2)
    assert list(stereo) == ref2.ravel().tolist()


def tone_ay():
    ay = Ayumi(SR, CLOCK)
    ay.set_pan(0, 0.5)
    ay.render(np.zeros(SPF, np.float32), u64([(0, 7, 0b00111111), (0, 8, 15)]), remove_dc=False)
    return ay


def test_placement_on_absolute_timeline():
    # A write lands on the first sample boundary at or after its cycle, not one sample off
    base = tone_ay()
    s0 = base.samples_rendered
    for t in [base.cycle_at(s0) + 1, base.cycle_at(s0 + 10), base.cycle_at(s0 + 10) + 1, base.cycle_at(s0 + 500)]:
        k = base.sample_at(t) - s0
        a = base.copy()
        got = np.zeros(SPF, np.float32)
        a.render(got, u64([(t, 8, 0)]), remove_dc=False)
        for kk, equal in ((k, True), (k - 1, False), (k + 1, False)):
            r = base.copy()
            L = np.zeros(SPF, np.float32)
            R = np.zeros(SPF, np.float32)
            if kk > 0:
                r.process_block(L[:kk], R[:kk], kk, remove_dc=False)
            r.R[8] = 0
            r.process_block(L[kk:], R[kk:], SPF - kk, remove_dc=False)
            assert np.array_equal(got, ((L.astype(np.float64) + R) * 0.5).astype(np.float32)) == equal


def test_write_at_block_end_holds_for_next_block():
    a = tone_ay()
    b = a.copy()
    end = a.samples_rendered + SPF
    t = a.cycle_at(end)            # lands exactly on the end of this block
    assert a.sample_at(t) == end
    x, y = np.zeros(SPF, np.float32), np.zeros(SPF, np.float32)
    a.render(x, u64([(t, 8, 0)]), remove_dc=False)
    b.render(y, remove_dc=False)
    np.testing.assert_array_equal(x, y)
    a.render(x, remove_dc=False)
    b.render(y, u64([(t, 8, 0)]), remove_dc=False)   # same write at the start of the next block
    np.testing.assert_array_equal(x, y)


def test_same_sample_last_write_wins():
    a, b = tone_ay(), None
    b = a.copy()
    t = a.cycle_at(a.samples_rendered + 100) - 3
    assert a.sample_at(t) == a.sample_at(t + 3)
    x, y = np.zeros(SPF, np.float32), np.zeros(SPF, np.float32)
    a.render(x, u64([(t, 8, 3), (t + 1, 8, 9), (t + 3, 8, 0)]))
    b.render(y, u64([(t, 8, 0)]))
    np.testing.assert_array_equal(x, y)


def out(n=SPF):
    return np.zeros(n, np.float32)


@pytest.mark.parametrize("writes", [
    [(200, 0, 1), (100, 0, 1)],          # unsorted t
    [(35470, 0, 1)],                     # after the end of the first block
    [(0, 14, 1)],                        # reg 14
    [(0, 0, 256)],                       # value 256
])
def test_errors_writes(writes):
    ay = make_ay()
    with pytest.raises(ValueError):
        ay.render(out(), u64(writes))
    assert ay.samples_rendered == 0


def test_error_write_in_the_past():
    ay = make_ay()
    ay.render(out())
    with pytest.raises(ValueError, match="before the current position"):
        ay.render(out(), u64([(35000, 8, 1)]))
    assert ay.samples_rendered == SPF


@pytest.mark.parametrize("buf", [
    array('I', [0, 0, 1]),                               # uint32: needs uint64
    np.array([0, 0, 1], dtype=np.int64),
    np.array([0, 0, 1, 0], dtype=np.uint64),            # not a multiple of 3
    np.zeros((2, 4), dtype=np.uint64),
    np.zeros((2, 6), dtype=np.uint64)[:, ::2],          # non-contiguous
])
def test_errors_writes_buffer(buf):
    with pytest.raises(ValueError):
        make_ay().render(out(), buf)


@pytest.mark.parametrize("buf", [
    np.zeros(10, np.float64),
    np.zeros(10, np.int32),
    np.zeros((10, 3), np.float32),
    np.zeros((10, 2, 1), np.float32),
    np.zeros(20, np.float32)[::2],
    np.zeros((10, 4), np.float32)[:, :2],
    np.zeros(0, np.float32),
])
def test_errors_out_buffer(buf):
    with pytest.raises(ValueError):
        make_ay().render(buf)


def test_error_out_readonly():
    with pytest.raises((ValueError, BufferError)):
        make_ay().render(bytes(100))


def test_error_non_integer_clock():
    ay = Ayumi(SR, 1773400.5)
    ay.render(out())          # no writes: fine
    with pytest.raises(ValueError, match="integer clock"):
        ay.render(out(), u64([(SPF * 40, 8, 1)]))
    with pytest.raises(ValueError):
        ay.sample_at(0)
