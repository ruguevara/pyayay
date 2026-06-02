---
name: psg-file-output
description: render() and the megademo can write a standard .psg register dump
metadata:
  type: reference
---

`ay_patterns.write_psg(path, psg, fps=50.0)` writes a `[frames,14]` register
array as a standard `.psg` file (Bulba AY register-dump container: 16-byte
`"PSG\x1a"` header + rate byte, body of `0xFF`-delimited frames with `reg,value`
pairs, `0xFD` terminator). It emits only changed registers (delta stream), which
keeps R13 from re-latching needlessly (buzzer-safe).

`render(..., psg_path="x.psg")` writes it alongside the WAV. `ay_megademo.py`'s
`main` auto-writes a sibling `.psg` next to the WAV (same stem), e.g.
`crystal_decline-01.wav` -> `crystal_decline-01.psg`.

Round-trip decode is exact (test in `examples/test_ay_patterns.py`:
`test_write_psg_roundtrip`). Note: `0xFF` also occurs as a legitimate register
*value* inside frames, not only as the frame marker — decode by position, not by
counting `0xFF`. Related viewer: [[ay-roll-tracker-viewer]].
