# Keyed by identity: one bug, four times

**Status: stands.** My bug, four separate instances, each caught by a different mechanism.

## The shape

An object appears in several *roles* within one operation. Code stores or looks it up **by its
identity** rather than by its role, so the roles collapse and one of them silently wins.

It has produced a wrong answer, an unsound type check, a missing kernel argument, and a
misplaced barrier, in four different layers of this compiler, over about a week.

| # | site | the multi-role object | what collapsed | detected by | error |
|---|---|---|---|---|---|
| 1 | `zippel/vjp.py` — VJP transform (D21) | an operand read twice by one path: `x*x` is `operands=(0,0)` | `operands.index(k)` returned the first position, so the product rule emitted one contribution instead of two | FP64 force validation against autograd | **2.6e-01** on F, forward exact |
| 2 | `zippel/ir.py` — type checker | the same, with *different slices*: `x[0:3]*x[3:8]` | `p.operands.index(j)` resolved both groups to the first slice, so extents were checked against the wrong operand | grepping the idiom repo-wide after fixing #1 | latent — accepted an ill-typed program |
| 3 | `codegen/emit_reduce.py` — T3 gather | a buffer gathered through *two* index maps: `evec_0` has inputs `(pos, pos, shifts)` with maps `(dst, src, None)` | a `{buffer → index map}` dict kept the last, so `pos[dst] − pos[src]` became `pos[src] − pos[src]` | the per-kernel ordering bound | **1.51e+01** vs a 6.18e-14 bound |
| 4 | `codegen/emit_tile.py` — T2 barrier placement | two `TileAssign`s agreeing on every field; the dataclass compares by value | `sched.assigns.index(a)` returned the first match, selecting where a shared-memory barrier is emitted | **the standing `.index(` audit instituted by this very finding**, minutes after writing it | latent |

Instance 3 had a second head: the same dict also built the kernel's parameter list, so `dst` was
omitted from `TENSOR_ORDER` while the kernel body still referenced `m_dst`. One lossy dict, two
distinct failures — a wrong answer and a malformed kernel.

## Why the fix keeps not generalising

Each instance was fixed correctly and locally, and each fix taught the *specific* lesson rather
than the general one:

* #1 → "iterate all occurrences in the VJP".
* #2 → "iterate by position in the validator".
* #3 → "carry the gather per factor".
* #4 → "carry the position instead of searching for it".

Four narrow lessons where one general rule was available after the first.

The general rule, which should have been extracted at #1: **when an operation names its inputs
positionally, every derived structure must be keyed positionally too.** `Op.inputs` is a tuple
with meaningful positions — `(pos, pos, shifts)` — and `index_maps` is a parallel tuple. Any map
keyed on the *value* at a position rather than the position itself is lossy the moment two
positions hold the same value. That is not an edge case in this IR; it is how `x*x`,
`pos[dst]-pos[src]`, and every squared norm are expressed.

## What each detection mechanism says about itself

The three detections are more interesting than the three bugs:

* #1 was caught by a **numerical oracle** — the force check — and only because the forward was
  simultaneously exact, which localised it to the derivative.
* #2 was caught by **reading**, prompted by #1. It was latent and no test would have found it,
  because no program in the repo exercised it. Grepping an idiom after fixing an instance of it
  is cheap and found a real hole.
* #3 was caught by a **standing check that runs unbidden**. Nothing in the suite targeted it; the
  ordering bound simply refuses any kernel that disagrees with its schedule by more than
  reordering explains.

* #4 was caught by the **standing audit this document instituted**, on its first run.

Only #3's mechanism scales without anyone anticipating the failure — that is the argument for the
bound (D25) from a different direction, and why the planted-fault battery exists. #4's mechanism
scales only as far as someone keeps running the audit, which is exactly as far as I keep
remembering to.

## Standing action

`grep -rn "\.index(" --include=*.py` is now part of reviewing any change that walks `Op.inputs`,
`ContractionPath.operands`, `index_maps`, or a schedule's assignment list.

Its first run found instance #4 — which is the argument for the audit, and also an argument
against trusting that four is the last one. Surviving occurrences:

* `blocks/eso2_ir.py:189` — `inputs.index(buf)` in a **deduplicating** builder, where the list is
  distinct by construction. Correct, and left alone.

Everything else is positional.
