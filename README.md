
| |
|-|
| ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ruguevara/pyayay/python-package.yml?style=flat&logo=github&label=tests) ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ruguevara/pyayay/cibuildwheel.yml?style=flat&logo=github) ![PyPI - License](https://img.shields.io/pypi/l/pyayay)|
| ![PyPI](https://img.shields.io/pypi/v/pyayay?logo=python&logoColor=white) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fruguevara%2Fpyayay%2Fmain%2Fpyproject.toml&logo=python&logoColor=white) ![PyPI - Wheel](https://img.shields.io/pypi/wheel/pyayay?logo=python&logoColor=white) |


# PyAYay

PyAYay is a Python wrapper for the AY/YM sound chip emulator. Currently it supports only the [Ayumi](https://github.com/true-grue/ayumi) emulator by Peter Sovietov.

## Installation

### From PyPI

You can install the package from PyPI:

```bash
pip install pyayay
```

Yep, that's it!

### From source

```bash
git clone https://github.com/ruguevara/pyayay.git
cd pyayay
pip install .
```

Or install in development mode:

```bash
git clone https://github.com/ruguevara/pyayay.git
cd pyayay
pip install -e .
```

## Usage

```python
from pyayay import Ayumi, ChipType, EnvShape

ay = Ayumi(sample_rate=44100, clock=1773400, type=ChipType.AY)
# or just Ayumi() for default values
```

Set panning for channels, for example in ACB order, and the master volume:
```python
ay.set_pan(0, 0.25)  # A left
ay.set_pan(1, 0.75)  # B right
ay.set_pan(2, 0.5)   # C center

ay.set_master_volume(0.75)
```

Use setters to set the channel parameters.
Do not forget to call `set_mixer` to enable the channel.

```python
ay.set_mixer(0, True, False, False)  # Tone, Noise and Envelope for A channel
ay.set_tone_period(0, 100)
ay.set_volume(0, 15)
```

You can set the envelope shape and period, for example:

```python
ay.set_mixer(0, True, False, True)  # Turn on envelope modulation for A channel
ay.set_envelope_shape(EnvShape.UP_DOWN_E)
ay.set_envelope_period(1024)
```

To generate sound use `process_block` method:

```python
samples = 44100 * 2  # 2 seconds
outLeft  = np.zeros(samples, dtype=np.float32)
outRight = np.zeros(samples, dtype=np.float32)

ay.process_block(outLeft, outRight, samples)
```

## Examples of using the R0-R13 registers and PSG rendering

You can use AY/YM registers R0-R13 directly:

```python
ay.R[1] = 100
ay.R[0] = 0
ay.R[7] = 0b00111110
ay.R[8] = 15
```

or, to set number of register at once:

```python
ay.set_registers(
    [1, 0, 7, 8],
    [100, 0, 0b00111110, 15]
)
```

or, to set some of the registers with the 'mask' array.
NOTE that mask is inverted, so 1 means "do not change".
This is useful for PSG frame playing.

```python
# R           0  1    2  3  4  5  6  7   8   9  10 11 12 13
R = np.array([0, 100, 0, 0, 0, 0, 0, 62, 15, 0, 0, 0, 0, 0], dtype=np.uint8)
M = np.array([0, 0,   1, 1, 1, 1, 1, 0,  0,  1, 1, 1, 1, 1], dtype=bool)

ay.set_registers_masked(R, M)
```

And render PSG data from Numpy arrays in one call:
```python
data = np.array([
    [249,   0, 148,   0,  40,   1,   3,  40,  13,  29,  13,  74,   0, 12],
    [249,   0, 158,   4,  40,   1,   3,  56,  13,  15,  13,  74,   0, 12]], dtype=np.uint8)
mask = np.array([
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]], dtype=bool)
fps = 50
ay.render_psg(data, mask, outLeft, outRight, fps)
```

## Rendering with timed register writes on an absolute timeline

`render(out, writes=None, remove_dc=True)` renders `len(out)` samples into a buffer you pass in
and applies register writes at exact positions on a timeline that runs from the creation of the
chip (or the last `reset`, `set_clock`, `set_sample_rate` or `set_type`).

- `out` is a writable float32 or int16 buffer. Shape `(n,)` gives mono `(left + right) / 2`,
  shape `(n, 2)` gives interleaved stereo. float32 holds Ayumi's raw values (as `process_block`),
  int16 holds them scaled as described in [Output scaling](#output-scaling).
- `writes` is a uint64 buffer of `(t, reg, value)` triples (`array('Q')` or `numpy.uint64`,
  flat or `(n, 3)`). `t` is the **absolute** time in AY clock cycles, non-decreasing.
  A write at `t` is applied right before absolute sample `ceil(t * sample_rate / clock)`.
  The math is exact 64-bit integer math, so the clock must be an integer number of Hz.
- `ay.samples_rendered` is the timeline position in samples. `render`, `process_block` and
  `render_psg` all advance it. It can be set, e.g. to seek.
- `ay.sample_at(t)` is the absolute sample a write at cycle `t` lands on; `ay.cycle_at(s)` is the
  last cycle that lands at or before sample `s`. A block of `n` samples takes exactly the writes
  with `t <= ay.cycle_at(ay.samples_rendered + n)`. Writes for an earlier position or past the end
  of the block raise `ValueError`.

Because positions are absolute, how you cut the render into blocks does not change the output:
one long block, 50 Hz frames, or blocks of random size give the same samples. Timer effects keep
their phase across frame boundaries even when a frame is not a whole number of AY cycles
(1773447 Hz at 44100 Hz is 35468.94 cycles per 882-sample frame).

```python
import numpy as np
from pyayay import Ayumi

ay = Ayumi(sample_rate=44100, clock=1773447)
ay.set_master_volume(0.6)
setup = [(0, 7, 0b00111110), (0, 0, 100), (0, 8, 15)]
period = 2953                     # timer effect: toggle the volume every 2953 AY cycles
timer = [(t, 8, 15 if (t // period) % 2 else 9) for t in range(period, 10 * 35469, period)]
writes = setup + timer

out = np.zeros(882, dtype=np.int16)
i = 0
for frame in range(10):
    end = ay.cycle_at(ay.samples_rendered + len(out))
    j = i
    while j < len(writes) and writes[j][0] <= end:
        j += 1
    ay.render(out, np.array(writes[i:j], dtype=np.uint64))
    i = j
    # ... use out
```

For an emulator that records writes in CPU T-states inside a frame (the AY runs at half the CPU
clock), the absolute AY cycle of a write at T-state `t` of frame `f` is
`f * (t_frame // 2) + (t + 1) // 2`. Rounding up keeps a write from landing before its time.

## Rendering a frame with timed register writes

`render_frame(writes, samples, t_frame, stereo=False, remove_dc=True) -> bytes` renders one
emulated frame and applies register writes at their position inside it. This is meant for
emulators that record `out` writes to the AY during a frame.

- `writes` is a buffer of unsigned 32-bit ints with `(t, reg, value)` triples: a flat
  `array('I')` / `numpy.uint32` array of length 3n, or a contiguous `(n, 3)` array. It can be empty.
  `t` is the T-state offset inside the frame (`0 <= t < t_frame`, non-decreasing), `reg` is 0..13,
  `value` is 0..255. Anything else raises `ValueError`, and the chip state is left unchanged.
- A write at `t` is applied right before sample `k = ceil(t * samples / t_frame)` is rendered
  (integer math), so it lands on the first sample boundary at or after its time. Writes with the
  same `k` are applied in order. A write with `k == samples` takes effect at the start of the next frame.
- Writes go through the same path as `ay.R[reg] = value`. A write to R13 always restarts the
  envelope (counter and segment reset), even if the shape value is the same, as on a real AY.
- The result is little-endian int16 PCM: `samples` values for mono (`(left + right) / 2`),
  or `2 * samples` interleaved `L, R` values for stereo. `remove_dc` has the same meaning as in
  `process_block`.

Example: a 128K frame (70908 T-states) at 50 Hz and 22050 Hz, with the AY clock scaled to
`70908 * 25 = 1772700` Hz:

```python
from array import array
from pyayay import Ayumi

ay = Ayumi(sample_rate=22050, clock=1772700)
ay.set_master_volume(0.6)

writes = array('I', [
    #  t      reg  value
    0,      7,   0b00111110,  # tone A on
    0,      8,   15,          # volume A
    0,      0,   100,         # tone A period, low byte
    35000,  0,   80,          # pitch change in the middle of the frame
])
pcm = ay.render_frame(writes, 441, 70908)  # 441 int16 samples = 882 bytes
```

### Output scaling

Each float sample (after `set_master_volume`) is multiplied by 32767, rounded to nearest and
clamped to int16. Mono is computed from the float average, so it can differ by one LSB from the
average of the rounded stereo values.

Ayumi's output is not limited to -1..1. Each channel contributes `dac_level * pan_left` to the left
output and `dac_level * pan_right` to the right one (max DAC level is 1.0), so the peak is the sum of
the pan gains plus about 9% overshoot from the decimation filter on sharp steps. Measured with all
three channels at full volume (AY and YM):

| panning                          | remove_dc=False   | remove_dc=True    |
|----------------------------------|-------------------|-------------------|
| default ACB (0.25, 0.75, 0.5)    | -0.18 .. +1.64    | -1.52 .. +1.62    |
| all channels hard left (0, 0, 0) | -0.36 .. +3.28    | -3.05 .. +3.24    |

With the default panning, `set_master_volume(0.6)` keeps the output inside int16 without clipping.
Louder settings clip at -32768 / 32767.

For more usage examples see [tests](tests/test_ayumi.py).

## License
We use MIT license, see [LICENSE](LICENSE) file.
