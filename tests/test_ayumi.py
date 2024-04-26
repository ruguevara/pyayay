from array import array
import math

import pytest
import numpy as np

from pyayay import Ayumi, EnvShape, ChipType

def bypass_initial_click(ay, duration_s=0.03):
    sample_rate = ay.get_sample_rate()
    samples = int(math.ceil(duration_s * sample_rate))
    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)
    ay.process_block(outLeft, outRight, samples)

def test_assignment():
    ay = Ayumi()
    for i in range(14):
        ay.R[i] = 42

    with pytest.raises(TypeError):
        ay.R[0] = 10101010101.0101

    with pytest.raises(TypeError):
        ay.R[0] = 10101010101

def test_bounds():
    ay = Ayumi()
    with pytest.raises(IndexError):
        ay.R[14] = 42

def test_process_array():
    ay = Ayumi()
    outLeft  = array('f', [0.0] * 50)
    outRight = array('f', [0.0] * 50)
    ay.process_block(outLeft, outRight, 50)

def test_process_tone():
    ay = Ayumi()
    ay.set_pan(0, 0.5)
    samples = 44100 * 1

    ay.set_tone_period(0, 100)
    assert ay.get_tone_period(0) == 100
    ay.set_volume(0, 15)
    ay.set_mixer(0, True, False, False)
    bypass_initial_click(ay)

    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)

    ay.process_block(outLeft, outRight, samples)

    assert np.abs(outLeft).mean() > 0.2
    assert np.abs(outRight).mean() > 0.2

def test_process_tone_R():
    ay = Ayumi()
    ay.set_pan(0, 0.5)
    bypass_initial_click(ay)
    samples = 44100 * 1

    ay.R[1] = 100
    ay.R[0] = 0
    ay.R[7] = 0b00111110
    ay.R[8] = 15

    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)

    ay.process_block(outLeft, outRight, samples)

    assert np.abs(outLeft).mean() > 0.2
    assert np.abs(outRight).mean() > 0.2

def test_process_silince():
    ay = Ayumi()
    bypass_initial_click(ay)
    samples = 44100 * 1

    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)

    ay.process_block(outLeft, outRight, samples)

    assert np.abs(outLeft).mean() < 0.001
    assert np.abs(outRight).mean() < 0.001

def test_reset():
    ay = Ayumi()
    ay.set_pan(0, 0.5)
    ay.set_tone_period(0, 100)
    ay.set_volume(0, 15)
    ay.set_mixer(0, True, False, False)

    samples = 44100 * 1

    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)

    ay.process_block(outLeft, outRight, samples)
    ay.reset()
    bypass_initial_click(ay)
    ay.process_block(outLeft, outRight, samples)

    assert np.abs(outLeft).mean() < 0.001
    assert np.abs(outRight).mean() < 0.001

def test_clock():
    ay = Ayumi(clock=1773400)
    assert ay.can_change_clock() == True
    assert ay.can_change_clock_continously() == True
    assert ay.get_clock_values() == []
    assert ay.get_clock() == 1773400
    ay.set_clock(2000000)
    assert ay.get_clock() == 2000000
    ay.reset(clock=1773400)
    assert ay.get_clock() == 1773400

def test_sample_rate():
    ay = Ayumi(44100)
    assert ay.get_sample_rate() == 44100
    ay.reset(48000)
    assert ay.get_sample_rate() == 48000
    ay.set_sample_rate(96000)
    assert ay.get_sample_rate() == 96000

def test_type():
    ay = Ayumi()
    assert ay.get_type() == Ayumi.AY
    ay = Ayumi(type=Ayumi.YM)
    assert ay.get_type() == Ayumi.YM
    ay.set_type(Ayumi.AY)
    assert ay.get_type() == Ayumi.AY
    ay.reset(type=Ayumi.YM)
    assert ay.get_type() == Ayumi.YM

def test_pan():
    ay = Ayumi()

    ay.set_pan(0, 0)
    assert ay.get_pan(0) == 0
    ay.set_pan(0, 1)
    assert ay.get_pan(0) == 1

    ay.set_tone_period(0, 100)
    ay.set_volume(0, 15)
    ay.set_mixer(0, True, False, False)
    bypass_initial_click(ay)

    samples = 44100 * 1

    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)

    ay.set_pan(0, 1)
    ay.process_block(outLeft, outRight, samples)

    assert np.abs(outLeft).mean() < 0.1
    assert np.abs(outRight).mean() > 0.25

    ay.set_pan(0, 0)
    ay.process_block(outLeft, outRight, samples)

    assert np.abs(outLeft).mean() > 0.25
    assert np.abs(outRight).mean() < 0.1

def test_mixer():
    ay = Ayumi()
    ay.set_tone_period(0, 100)
    ay.set_volume(0, 15)
    ay.set_mixer(0, False, False, False)
    ay.set_pan(0, 0)
    bypass_initial_click(ay)

    samples = 44100 * 1

    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)
    ay.process_block(outLeft, outRight, samples)
    assert np.abs(outLeft).mean() < 0.001

    ay.set_mixer(0, False, True, False)
    ay.process_block(outLeft, outRight, samples)
    assert 0.15 < np.abs(outLeft).mean() < 0.25

    ay.set_mixer(0, False, False, True)
    ay.set_envelope_shape(EnvShape.UP_DOWN_E)
    ay.set_envelope_period(1)
    ay.process_block(outLeft, outRight, samples)
    assert np.abs(outLeft).mean() > 0.25

    ay.set_mixer(0, True, False, True)
    ay.set_envelope_shape(EnvShape.UP_DOWN_E)
    ay.set_envelope_period(1)
    ay.process_block(outLeft, outRight, samples)
    assert 0.15 < np.abs(outLeft).mean() < 0.25

def test_master_volume():
    ay = Ayumi()
    ay.set_pan(0, 0)
    ay.set_master_volume(1)
    ay.set_mixer(0, True, False, False)
    ay.set_tone_period(0, 100)
    ay.set_volume(0, 15)
    bypass_initial_click(ay)

    samples = 44100 * 1
    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)
    ay.process_block(outLeft, outRight, samples)
    assert 0.45 < np.abs(outLeft).mean()

    ay.set_master_volume(0.5)
    bypass_initial_click(ay)
    ay.process_block(outLeft, outRight, samples)
    assert 0.2 < np.abs(outLeft).mean() < 0.3

def test_copy():
    ay = Ayumi()
    ay.set_tone_period(0, 100)
    ay.set_volume(0, 15)
    ay.set_mixer(0, True, False, False)
    ay.set_pan(0, 0.5)
    bypass_initial_click(ay)

    from copy import copy
    ay2 = copy(ay)
    ay2 = ay.copy()
    assert ay2.get_tone_period(0) == 100
    assert ay2.get_volume(0) == 15
    assert ay2.get_pan(0) == 0.5

    samples = 44100 * 1
    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)
    ay2.process_block(outLeft, outRight, samples)

    assert np.abs(outLeft).mean() > 0.2
    assert np.abs(outRight).mean() > 0.2

def test_envelope():
    ay = Ayumi()
    ay.set_pan(0, 0)
    ay.set_mixer(0, False, False, True)
    ay.set_volume(0, 0)  # volume must not affect envelope
    ay.set_tone_period(0, 100)
    ay.set_envelope_shape(EnvShape.UP_DOWN_E)
    ay.set_envelope_period(1)

    assert ay.get_envelope_shape() == EnvShape.UP_DOWN_E
    assert ay.get_envelope_period() == 1
    bypass_initial_click(ay)

    samples = 44100 * 1
    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)
    ay.process_block(outLeft, outRight, samples)
    assert np.abs(outLeft).mean() > 0.2

    ay.set_envelope_shape(EnvShape.DOWN_HOLD_BOTTOM_0)
    ay.process_block(outLeft, outRight, samples)
    assert np.abs(outLeft).mean() < 0.01

    # modulated envelope
    ay.set_mixer(0, True, False, True)
    ay.set_envelope_shape(EnvShape.UP_DOWN_E)
    ay.set_envelope_period(4095)
    ay.process_block(outLeft, outRight, samples)
    assert 0.1 < np.abs(outLeft).mean() < 0.3

def test_noise():
    ay = Ayumi()
    ay.set_pan(0, 0)
    ay.set_mixer(0, False, True, False)
    ay.set_volume(0, 15)  # volume affects noise
    ay.set_tone_period(0, 100)
    ay.set_noise_period(16)
    assert ay.get_noise_period() == 16
    bypass_initial_click(ay)

    samples = 44100 * 1
    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)
    ay.process_block(outLeft, outRight, samples)
    assert 0.4 < np.abs(outLeft).mean() < 0.6

    ay.set_volume(0, 10)  # volume is not linear
    ay.process_block(outLeft, outRight, samples)
    assert 0.1 < np.abs(outLeft).mean() < 0.2

def test_registers_array():
    ay = Ayumi()
    ay.set_pan(0, 0.5)
    ay.set_registers(
        [1, 0, 7, 8],
        [100, 0, 0b00111110, 15]
    )

    bypass_initial_click(ay)
    samples = 44100 * 1

    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)

    ay.process_block(outLeft, outRight, samples)

    assert np.abs(outLeft).mean() > 0.2
    assert np.abs(outRight).mean() > 0.2

def test_registers_array_bounds():
    ay = Ayumi()
    with pytest.raises(IndexError):
        ay.set_registers([15], [42])

def test_registers_masked_error():
    ay = Ayumi()
    # array size must be 14
    with pytest.raises(ValueError):
        ay.set_registers_masked(
            np.array([100, 0, 0b00111110, 15], dtype=np.uint8),
            np.array([True, False, True, True], dtype=bool),
        )

def test_process_block_errors():
    ay = Ayumi()
    outLeft  = array('f', [0.0] * 50)
    outRight = array('f', [0.0] * 50)
    with pytest.raises(ValueError):
        ay.process_block(outLeft, outRight, 51)

    with pytest.raises(ValueError):
        ay.process_block(outLeft, outRight[:20], 50)

    # only float arrays are accepted
    outRightD = array('d', [0.0] * 50)
    with pytest.raises(ValueError):
        ay.process_block(outLeft, outRightD, 50)

    # samples must be positive
    with pytest.raises(ValueError):
        ay.process_block(outLeft, outRight, 0)

def test_registers_masked():
    ay = Ayumi()
    ay.set_pan(0, 0)
    # R           0  1    2  3  4  5  6  7   8   9  10 11 12 13
    R = np.array([0, 100, 0, 0, 0, 0, 0, 62, 15, 0, 0, 0, 0, 0], dtype=np.uint8)
    M = np.array([0, 0,   1, 1, 1, 1, 1, 0,  0,  1, 1, 1, 1, 1], dtype=bool)
    # NOTE mask is inverted, so 1 means "do not change"

    ay.set_registers_masked(R, M)

    bypass_initial_click(ay)
    samples = 44100 * 1
    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)
    ay.process_block(outLeft, outRight, samples)

    assert np.abs(outLeft).mean() > 0.45

def test_psg_play():
    data = np.array([
        [249,   0, 148,   0,  40,   1,   3,  40,  13,  29,  13,  74,   0, 12],
        [249,   0, 158,   4,  40,   1,   3,  56,  13,  15,  13,  74,   0, 12]], dtype=np.uint8)
    mask = np.array([
        [False,  True, False,  True, False, False, False, False, False, False, False, False,  True, False],
        [ True,  True, False, False,  True,  True,  True, False,  True, False,  True,  True,  True,  True]])

    ay = Ayumi()
    samples = 44100 // 50 * 2
    outLeft  = np.zeros(samples, dtype=np.float32)
    outRight = np.zeros(samples, dtype=np.float32)
    ay.render_psg(data, mask, outLeft, outRight, 50)

    with pytest.raises(ValueError):
        ay.render_psg(data, mask, outLeft, outRight, 10)

    with pytest.raises(ValueError):
        ay.render_psg(data, mask[:-1], outLeft, outRight, 10)
