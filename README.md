
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

For more usage examples see [tests](tests/test_ayumi.py).

## License
We use MIT license, see [LICENSE](LICENSE) file.
