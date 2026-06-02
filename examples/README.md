# `examples/` — a Strudel-style pattern language for the AY chip

A self-contained composition layer built **on top of** the `pyayay` extension.
Not part of the shipped package and not on the import path of the package tests —
run it from inside `examples/`:

```bash
pytest test_ay_patterns.py          # the engine's test suite
python ay_demo_track.py             # -> ay_demo.wav (short showcase)
python ay_megademo.py               # -> the ~97 s "Crystal Decline" track
```

### Files

| File | What it is |
|---|---|
| `strudel.py` | **Level 1–2 + 4 (ADSR).** The pure pattern algebra: `Pattern`, `Event`, `Note`/`Chord`, the builders (`note`, `chord`, `seq`, `stack`, `cat`, `euclid`, …) and transforms (`.fast`, `.arp`, `.echo`, `alt`, …), plus the `ADSR` envelope. Chip-agnostic. |
| `ay_patterns.py` | **Level 3–5.** The AY-specific layer: instruments (`Tone`, `Buzzer`, `Percussion`, `Sample`), `VoiceState`, the grid sampler, the voice-stealing allocator, and `render()`. Re-exports the `strudel` names so a tune imports only `ay_patterns`. |
| `ay_demo_track.py` | A short showcase tune exercising every trick (ADSR, buzzer, ornament, `alt`, priority-ducked `echo`, the shared-channel drum kit). |
| `ay_megademo.py` | "Crystal Decline" — an arranged ~97 s demoscene track (intro / build / drop / solo / outro). |
| `ay_analyze.py` | Offline "listening": turns a render into loudness/brightness/harmony numbers + an ASCII arc (there is no listening in CI). |
| `ay_roll.py` | A tracker-style text view of a rendered PSG stream, for eyeballing timing/voicing. |
| `test_ay_patterns.py` | The engine tests (pure pattern algebra + a few end-to-end render smoke tests). |

---

## Architecture: the five levels

The engine is a pipeline from a **lazy pure function over time** down to **raw AY
register bytes**, then audio. The crossing from "pure values" to "chip state"
happens at exactly one place (`_render_grid`).

```
╔══════════════════════════════════════════════════════════════════════════╗
║ LEVEL 1 — PATTERN ALGEBRA              strudel.py        PURE, LAZY        ║
║   A Pattern IS a function:  query(Span) -> [Event]                         ║
║   Builders + transforms. Nothing computes until queried; transforms        ║
║   return NEW patterns. (note, chord, seq, stack, cat, .fast, .arp, .echo…) ║
╠══════════════════════════════════════════════════════════════════════════╣
║ LEVEL 2 — EVENTS                       strudel.py        PURE VALUES       ║
║   Event(whole: Span, part: Span, value: Note|Chord)                        ║
║   Note is an immutable dataclass: pitch + automation params. No registers. ║
╠══════════════════════════════════════════════════════════════════════════╣
║ LEVEL 3 — GRID SAMPLER                 _render_grid      THE BOUNDARY      ║
║   Walks frames 0..N. Per frame: query the span, derive t_in/dur,           ║
║   call instrument.frame_state per event -> VoiceState, then ALLOCATE.      ║
╠══════════════════════════════════════════════════════════════════════════╣
║ LEVEL 4 — INSTRUMENTS                   frame_state      PURE (per frame)  ║
║   Note + t_in -> VoiceState. _live_freq / _live_volume / ADSR / ornament.  ║
║   VoiceState = a chip-channel *intent* (tone_period, volume, env, noise).  ║
╠══════════════════════════════════════════════════════════════════════════╣
║ LEVEL 5 — PSG / EMULATOR     _assemble_frame + pyayay.Ayumi.render_psg     ║
║   VoiceStates -> 14 raw AY register bytes -> Ayumi DSP -> float samples.   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### Level 1 — queryable patterns *(pure, lazy)*

A `Pattern` (`strudel.py`, `class Pattern`) holds no data and no time. It is a
function `query(Span) -> [Event]`: you hand it a window of rational cycle-time
(`Span` of `Fraction`; 1 cycle == 1 bar by convention) and it **computes** the
events landing in that window.

- **Builders** construct patterns: `note`, `chord`, `seq`, `stack`, `cat`,
  `euclid`, `pure`, `silence`.
- **Transforms return new patterns** that wrap the old one's `query` — they never
  mutate and never render: `.fast`/`.slow`/`.rev`/`.every`/`.early`/`.late`,
  `.add`, `.arp`, `.echo`, `.stutter`, `.degrade_by`, `.struct`, `.gain`,
  `.s` (attach instrument), `.pan`, `.priority`, `.ornament`, `.adsr`, and the
  builder `alt` (per-cycle parameter alternation).

Two transforms that *expand* the event stream (important for the chip tricks):

- **`.arp(mode, rate)`** turns a chord-value event into `rate` single-note events
  (default `rate = len(notes)`). An explicit `rate` makes the pulse
  chord-size-independent. → produces **many events**.
- **`.echo(times, delay, feedback)`** emits the dry event plus decaying trailing
  copies with **descending priority**. → produces **many events**.

Transforms that only *stamp a field* on the Note value (no new events):
`.ornament(...)`, `.adsr(...)`, `.pan(...)`, `.priority(...)`, `.add(...)`.

### Level 2 — events *(pure values)*

`Event(whole, part, value)` (`strudel.py`, `class Event`):

- `whole` — the note's logical span; its **onset** is `whole.begin`.
- `part` — the (possibly clipped) portion this query returned.
- `value` — a `Note` (`frozen=True` dataclass) or a `Chord`.

A `Note` carries pitch plus **automation parameters** — `volume`, `pan` (a
*bucket* `"L"/"C"/"R"/"A"`, not a number yet), `instrument`, `sweep`, `vib_*`,
`vol_env`/`pitch_env` (ADSR objects), `ornament` (a semitone tuple), and
`priority`. It holds **no register values and no per-frame state**.

### Level 3 — the grid sampler *(the boundary; imperative driver)*

`_render_grid` (`ay_patterns.py`) is where pure values become chip state. Per
frame `fi`:

1. **Query** the frame's span → `[Event]`.
2. Derive **`t_in`** = seconds since *this note's* onset, and **`dur`** = note
   length in seconds. This single clock drives all Level-4 automation.
3. Call `note.instrument.frame_state(note, t_in, dur, clock, fps)` → a
   **`VoiceState`** (Level 4).
4. Wrap as `_FrameVoice(bucket, priority, order, state)`.
5. **`_allocate_channels`** → up to 3 surviving VoiceStates *(voice stealing)*.
6. **`_assemble_frame`** → the **14 register bytes** for this frame (Level 5).

So events stay pure values until step 3, become a chip-channel *intent*
(VoiceState) in step 3, and become actual **register bytes** in step 6.

### Level 4 — instruments *(pure, per frame)*

`Instrument.frame_state(note, t_in, dur, …) -> VoiceState` maps one Note at one
instant onto a chip-channel intent. The two automation helpers both read the
**same `t_in`** but write **different VoiceState fields**, so they compose:

- `_live_freq` (pitch) sums: `sweep·t_in`, vibrato, `pitch_env.at(t_in)` (pitch
  ADSR), and the **ornament** `table[round(t_in·fps) % len]` (stepped *inside*
  the instrument — one held note, no re-trigger).
- `_live_volume` (loudness) applies `vol_env.at(t_in)` (volume ADSR).
- `ADSR.at(t_in, dur)` (`strudel.py`) is a pure piecewise A→D→S→R function.

`VoiceState` (`ay_patterns.py`) is a *request* to the chip: `tone_on`,
`noise_on`, `env_on`, `tone_period`, `volume`, `pan`, and the **chip-global**
`env_period`/`env_shape`/`noise_period` (shared AY resources — last writer wins
in `_assemble_frame`).

### Level 5 — PSG bytes and the emulator

`_assemble_frame(voices) -> regs[14]` (`ay_patterns.py`) packs up to 3
VoiceStates into raw AY registers (R0–R13: tone periods, noise period, mixer,
volumes, envelope). `render()` then feeds the whole `psg[N,14]` array to
`pyayay.Ayumi.render_psg` to produce the float audio.

---

## Where voice stealing happens

**One place, once per frame:** `_allocate_channels` (`ay_patterns.py`). It
resolves a frame's `_FrameVoice` list onto the 3 physical channels under the
**"pan *is* the channel"** model:

- Each fixed bucket maps to one channel — **L→ch0, C→ch1, R→ch2** — and is
  resolved **independently**. The highest-`priority` voice in a bucket wins;
  ties break by document/stack `order` (`best = min(key=(-priority, order))`).
  **The losers are dropped even if another channel is idle.**
- A wildcard bucket **`"A"`/`PAN_ANY`** is seated *afterwards*, by priority, into
  whatever channel is still free (centre-preferred). It can spill into an idle
  channel; the fixed buckets cannot.

This is why a drum kit stacked on one bucket plays by precedence
(kick > snare > hat) while another channel sits empty, and why a priority-ducked
`.echo` tail gets cut by a fresh dry hit in the same bucket — the duck isn't in
`echo`, it's realized here.

---

## What is pure, and the one exception

| Thing | Pure? |
|---|---|
| Pattern `query`, transforms | **Pure** (deterministic; `echo`/`degrade`/`glitch` hash the onset). |
| `Event` / `Note` / `Chord` | **Pure**, immutable (`frozen=True`). |
| `frame_state`, `_live_freq`, `_live_volume`, `ADSR.at` | **Pure**: `Note + t_in → VoiceState`/scalar. |
| `_assemble_frame` | **Pure**: VoiceStates → bytes. |
| `_render_grid` | Imperative driver (fills `psg[fi]`). The boundary. |
| `_allocate_channels` | **Mutates** the chosen VoiceState's `.pan` in place. |
| `pyayay.Ayumi` | Stateful C DSP. |

`VoiceState` is the only mutable value: `_allocate_channels` writes the channel's
numeric pan back onto `v.state.pan`. This is safe because every frame builds
**fresh** VoiceStates from immutable Notes — nothing is shared across frames, and
the Notes themselves are never touched.

---

## Two render-level concerns that don't fit the per-frame model

- **R13 mask.** Writing the envelope-shape register (R13) *retriggers* the AY
  envelope, resetting a buzzer's sawtooth phase — re-latching it every frame
  chops any buzzer into a 50 Hz buzz. `render()` writes R0–R12 every frame but
  **masks R13 on frames where the shape is unchanged**. A cross-frame concern,
  handled at Level 5, not in any instrument.
- **Pan segments.** `set_pan` is a chip-global call (not a register), so
  `render()` splits the timeline into maximal runs of stable per-channel pan
  (`_pan_segments`) and calls `render_psg` once per run. An idle channel reports
  pan `None` ("don't care") and never forces a new segment.

---

## `arp` vs `ornament` — the same goal, opposite mechanisms

Both fake a chord on one channel, but at different levels:

| | `.arp(mode, rate)` | `.ornament(0,4,7)` |
|---|---|---|
| Level | 1 (emits events) | 2 stamp + 4 eval (one held note) |
| Per tick | a **new note event** → its own `frame_state` | the pitch number is stepped *inside* `frame_state` |
| ADSR | **re-attacks every tick** (plucked) | runs **once** across the whole note |
| Tick ceiling | the frame grid — `rate > frames_per_cycle` aliases | always one step per frame |

Use **`arp`** for a plucked arpeggio (you *want* each note re-attacked) and
**`ornament`** for a sustained chip-chord on a buzzer or held timbre (one trigger,
no R13 re-latch).
