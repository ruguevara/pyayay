---
name: intro-shimmer-lurch
description: Why the megademo intro "slowed down" and how the pad shimmer fix works
metadata:
  type: project
---

The megademo (`examples/ay_megademo.py`) intro audibly "slowed down". Cause: the
pad was `chord(prog).slow(8).stutter(1).arp("up")` — with `stutter(1)` (a no-op)
the only motion is the arp, whose onset rate equals the **chord size**. The
progression mixes triads (3 notes) and maj7 chords (4 notes), so the pad pulsed
at 3 onsets/bar on triad bars and 4/bar on maj7 bars — speeding up and dragging
bar to bar.

Fix: `pad_progression` now defaults `shimmer=4` and the intro/outro callers pass
`shimmer=4` instead of `1`, so the chord is stuttered onto a fixed per-bar grid
*before* the arp. The quarter-note pulse is then steady regardless of chord size
(chord-size difference becomes fast in-beat arp fills). Regression tests in
`examples/test_ay_roll.py`.

Built `examples/ay_roll.py` to find this — see [[ay-roll-tracker-viewer]].
