# A greedy fusion partition that could not be scheduled

**Status:** self-finding, fixed. Affects a number reported at Gate 1.

## What I reported

The Gate 1 complexity table carried a "fusion groups" column — 42 / 46 / 107 for fwd / force /
dbwd — described as "kernel *launches* left after a cheap greedy producer–consumer fusion pass".
I framed 107 as the tractability argument for Phase 2: 903 ops collapsing to 107 launches.

## What was wrong

The pass merged an op into any group containing a fusable producer, checking only local
properties (same segment axis, matching trailing extents, no gather/scatter on the consumer). It
never checked that the resulting **group** graph stayed acyclic. It does not:

| program | groups | in a dependence cycle |
|---|---|---|
| fwd | 42 | 36 |
| force | 46 | 40 |
| dbwd | 107 | **101** |

LayerNorm is the archetype. For `y = (x - mean(x)) * invstd`:

* `x` (the linear output) starts group A;
* `mean(x)` reduces `x` to a scalar — different trailing extents, so not fusable — and starts
  group B;
* `x - mean(x)` *is* fusable with `x`, so it joins group A.

Group A now needs `mean(x)` from B, and B needs `x` from A. Two kernels, each waiting on the
other's output. There is no launch order that satisfies both, so the partition is not a
pessimistic launch count — it is an unachievable one.

## The fix

Acyclicity becomes a precondition of every merge, not a post-hoc check. Before putting op `n`
into group `G`, for each other producer group `H` of `n` we ask whether `G` can already reach
`H`; if so the merge would close a cycle and is refused, and `n` starts its own group. The
reachability walk is a plain DFS over ~100–320 groups, so the pass stays cheap.

| program | ops | reported | corrected | ops per launch |
|---|---|---|---|---|
| fwd | 101 | 42 | **48** | 2.1 |
| force | 290 | 46 | **115** | 2.5 |
| dbwd | 903 | 107 | **320** | 2.8 |

## Consequence, stated plainly

The launch-count claim is three times worse for dbwd than reported. 903 ops become 320 launches,
not 107. Nothing measured changes — no timing, no correctness result, no memory figure — and the
**archetype** counts (35 / 78 / 149), which are what Phase 2's *emitter* effort actually scales
with, are untouched. But the "107 launches" line in the Gate 1 summary was wrong and is corrected
in REPORT.md §8.2.

## How it was found

Building the Phase 2 emitter. Enumerating group 0's live-ins, I noticed `mean_9` was a live-in of
the same group that produced `rl0_8`, and `mean_9` is a reduction *of* `rl0_8`. A group whose
input is computed from its own output is not schedulable, which prompted the Kahn sort across all
three programs.

The general lesson is the one D21 taught in a different place: a pass that only checks *local*
legality can produce a globally invalid result, and the check that would have caught it (does
this partition admit a topological order?) is cheap and now runs in `tests/test_ir_core.py`.
Structural claims about a compiler IR need structural tests, not just numerical ones — every
correctness test in the suite passed throughout, because the partition never affected a computed
value.
