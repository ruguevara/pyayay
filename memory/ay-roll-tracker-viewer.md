---
name: ay-roll-tracker-viewer
description: examples/ay_roll.py — tracker-style text view of a rendered PSG stream
metadata:
  type: reference
---

`examples/ay_roll.py` is a debugging sibling to `ay_analyze.py`: instead of
loudness/harmony numbers, it decodes a rendered `[frames,14]` PSG array back into
a vertical, tracker-style score (3 channel columns A/B/C, rows down the page).
Per channel per row it shows the note name (from the tone period), the T/N/E
mixer/envelope bits, volume (or `EE` for envelope mode), and for buzzer voices
the envelope's note name + detune from the tone note + shape glyph (`/\`, `/|`).
Tracker convention: fresh trigger prints the note, held blanks (`...`), silence
is `===`.

Generate it:

    python ay_roll.py --start 0 --end 4 --rpb 4        # bars 0-4 of the megademo
    python ay_roll.py --start 16 --end 20 --no-env     # drop section, no env col

Or programmatically: `from ay_roll import roll; print(roll(psg, bpm=..., beats_per_cycle=..., fps=50.0, rows_per_beat=4, bars=(0,4)))`.

`--rpb` = rows per beat (4 = 16th notes). Decodes from the PSG array, so it shows
exactly what the chip plays. This is how the intro lurch was diagnosed — see
[[intro-shimmer-lurch]].
