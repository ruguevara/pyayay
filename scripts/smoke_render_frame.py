"""render_frame and render smoke test, run both natively and under Pyodide.

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


def render_timeline(stereo):
    # Absolute uint64 timeline, far from 0, timer writes crossing block ends
    ay = Ayumi(44100, 1773447)
    ay.set_master_volume(0.6)
    ay.samples_rendered = 10**11
    t0 = ay.cycle_at(ay.samples_rendered)
    period = 2953
    writes = [(t0 + 1, 7, 0b00111110), (t0 + 1, 0, 100), (t0 + 1, 8, 15)]
    writes += [(t0 + t, 8, 15 if (t // period) % 2 else 9) for t in range(period, 3 * 35468, period)]
    out = array('h', bytes(2 * (882 * 2 if stereo else 882)))
    view = memoryview(out).cast('B').cast('h', [882, 2]) if stereo else out
    data = b''
    i = 0
    for _ in range(3):
        end = ay.cycle_at(ay.samples_rendered + 882)
        j = i
        while j < len(writes) and writes[j][0] <= end:
            j += 1
        ay.render(view, array('Q', [x for w in writes[i:j] for x in w]))
        data += out.tobytes()
        i = j
    values = struct.unpack('<%dh' % (len(data) // 2), data)
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest()[:16],
        "at_100": list(values[100:104]),
        "min": min(values),
        "max": max(values),
        "position": ay.samples_rendered,
        "cycle_at": ay.cycle_at(ay.samples_rendered),
    }


try:
    Ayumi().render_frame(array('I', [0, 14, 0]), SAMPLES, T_FRAME)
    error = None
except ValueError as e:
    error = str(e)

print(json.dumps({"mono": render(False), "stereo": render(True),
                  "timeline_mono": render_timeline(False), "timeline_stereo": render_timeline(True),
                  "error": error}, indent=1))
