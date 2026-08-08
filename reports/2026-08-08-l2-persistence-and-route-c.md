# Status — 2026-08-08 · L2 persistence adjudicated, route C settled

## Headline

* **Eviction hypothesis REFUTED** on a clean six-arm experiment. Lever (b) is dead.
* **The weights are not the DRAM traffic** — pinning them buys 0.56 % where the demand attribution
  implied ~18 %. Lever (c) moves **down**, not up.
* **Route C is dead by probe**, not by assertion. `terms ∝ E_c` stands as law.
* **1.86 TB per launch — 539× compulsory — remains unattributed.** Open anomaly, named as such.

## L2 persistence ladder (`conv1_90`, si_medium fp32, `E_c`=1, NUMA-pinned, all bit-equal)

| arm | reserved | window | ms | vs baseline |
|---|---|---|---|---|
| `co0MiB_win0` | 0 | off | 582.449 | 1.000× |
| `co2MiB_win1` | 3.75 MiB | on | 579.176 | 1.0057× |
| `co4MiB_win1` | 7.50 MiB | on | 579.164 | 1.0057× |
| `co8MiB_win1` | 11.25 MiB | on | 579.161 | 1.0057× |
| **`co32MiB_win0`** | 33.75 MiB | **off** | **1240.141** | **0.470×** |
| `co32MiB_win1` | 33.75 MiB | on | 1236.475 | 0.471× |

**Reusable fact:** reserving 56 % of a 60 MiB L2 costs **2.13×**; 19 % is free.

**My earlier single-arm result (2.12× slower) measured my own carve-out, not persistence.** The
control — carve-out with the window *off* — reproduces it to 0.24 %. Running that control was the
difference between retiring a hypothesis on a false negative and knowing what happened.

## Route C probe

    range_constexpr + named regs : COMPILED
     range(dynamic) + scalar acc : COMPILED
    range(dynamic) + indexed acc : FAILED -- ArithValue cannot be interpreted as an integer

Small IR *or* per-iteration registers, never both. Full write-up:
`findings/the-tracer-is-the-expander.md`.

## Banked measurements (unaffected by the open anomaly)

| result | value |
|---|---|
| layout requirement (transpose) | **1.228×** on `conv1_90`; **1.441×** on the whole forward |
| edge batching `E_c`=4 | **1.325×** |
| bytes law on DRAM bytes | predicts time to **0.5–1.2 %** |
| composition, post-layout | 1401.9 → **972.8 ms**; hill 4.50× → **3.12×** |
| composition correctness | **S1C PASS**, energy rel 1.117e-15 |

## In flight

* Track 1 — DRAM-bytes confirmation under `ncu` (window on vs off), then source-attributed ncu.
* Track 2 — two-conv generalisation, then composition at **N=5** per D47.

## Open on the reviewer's side

* The **threaded-backend artefact** (82.9 % parallel / 6.75×@8T / ~9 min). No such measurement
  exists in this repo; D84 measured a **1.6× kernel-level Amdahl ceiling** instead.
