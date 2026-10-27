"""render_frame smoke test, run both natively and under Pyodide.

Prints JSON; the two outputs must be identical.
"""
import hashlib
import json
import struct
from array import array

from pyayay import Ayumi

SAMPLES, T_FRAME = 441, 70908
WRITES = array('I', [
    0, 7, 0b00110000, 0, 0, 100, 0, 2, 150, 0, 3, 1, 0, 6, 7,
    0, 8, 15, 0, 9, 12, 0, 10, 16, 0, 12, 2, 0, 13, 14,
    20000, 0, 60, 40000, 13, 14, 70907, 8, 3,
])


def render(stereo):
    ay = Ayumi(22050, 1772700)
    ay.set_master_volume(0.6)
    first = ay.render_frame(WRITES, SAMPLES, T_FRAME, stereo=stereo)
    second = ay.render_frame(array('I'), SAMPLES, T_FRAME, stereo=stereo)
    data = first + second
    values = struct.unpack('<%dh' % (len(data) // 2), data)
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest()[:16],
        "at_100": list(values[100:104]),
        "at_200": list(values[200:204]),
        "min": min(values),
        "max": max(values),
    }


try:
    Ayumi().render_frame(array('I', [0, 14, 0]), SAMPLES, T_FRAME)
    error = None
except ValueError as e:
    error = str(e)

print(json.dumps({"mono": render(False), "stereo": render(True), "error": error}, indent=1))
