# Kernel templates and the selection rule

Phase 2 lowers segmented-polynomial IR to CuTe DSL. It does not do so with one general strategy:
the IR's ops span three orders of magnitude in trailing extent, from a `[9,9]` Wigner block to a
`[9,256]` message, and no single thread mapping serves both. This document defines the four
templates, and the rule that picks one for a given fusion group.

The rule is a **compiler input**, not documentation of what we happened to write. `emit_source`
refuses a group whose template preconditions do not hold, rather than emitting something that
silently spills.

---

## 1. The templates

| | name | parallel axis | trailing axes | structure exploited by |
|---|---|---|---|---|
| **T1** | register-resident scalar | segment | all unrolled into registers | eliding zero terms |
| **T2** | cooperative tile | segment × channel | channel on threads, rest unrolled | eliding zero terms |
| **T3** | reduction / segment-rooted | output segment | unrolled | atomics or a sorted spine |
| **T4** | persistent spine | CTA-resident over many segments | as T1/T2 | reuse across the chain |

### T1 — register-resident scalar

One thread owns one segment element and holds every trailing value of every live buffer in
registers. The whole fusion group becomes straight-line scalar code; internal buffers never reach
memory. Implemented in `codegen/schedule.py` + `codegen/emit.py`.

**Precondition:** `Schedule.peak_live_values() <= REGISTER_BUDGET` (168).

**Status:** done. The Wigner chain — five chained 9×9 matmuls, six internal `[E,9,9]` buffers —
emits at 356 terms and 108 live scalars, bit-exact against the FP64 interpreter.

### T2 — cooperative tile

For groups carrying a **dense channel axis**. Channels are distributed across the warp, each
thread holding a tile slice; the small structured axes (coefficient `m`, degree `l`) stay
unrolled per thread. Cross-channel contractions (the SO(2) linear layers) go through shared
memory or an MMA atom; per-channel work stays in registers.

The channel axis is *dense* — there is no structural sparsity in it, and 128 or 256 channels do
not fit one thread's registers. Forcing T1 on such a group is the error the register budget
exists to prevent: under T1 a per-edge 128→128 Linear puts the whole `[128,128]` weight in one
thread's registers, and the SO(2) conv group needs **492 929** live scalars per thread. T1
refuses both. Across the forward, **21 of 44** index-map-free groups fit the 168-register budget;
the rest are T2.

**Precondition:** a trailing axis of extent ≥ 32 that the group never contracts *and* never
slices unevenly, so it can be split across threads without cross-lane traffic in the pointwise
ops.

**Status:** in progress (Phase 2 S1).

### T3 — reduction / segment-rooted

For ops that change segment axis: a gather (`index_maps`), a scatter-add (`out_index_map`), or a
reduction to a coarser segment. The parallel axis is the *output* segment, and the kernel walks
the contributing input elements. T1 and T2 both exclude these by construction — the fusion pass
starts a new group at any index map, because an index map re-maps the segment axis and breaks the
shared loop nest.

Two sub-strategies, chosen by contention: atomics into the output when the average fan-in is
small, or a sorted/segmented spine when it is large. The edge→node scatter in this block has mean
degree 45 (Si) to 78 (Cu), which is high enough that this needs measuring, not assuming.

**Status:** Phase 2 S2.

### T4 — persistent spine

One CTA takes a strip of segment elements and carries the *whole* fwd/bwd/dbwd chain for them
without returning to the launcher, so values shared across passes are recomputed from registers
rather than reloaded. This is the template that would express the three-pass joint schedule
directly, and it is the most speculative of the four.

**Status:** Phase 2 S3, explicitly optional-if-time.

---

## 2. The selection rule

For a fusion group `G`, with

* `R` = peak live scalars per thread if emitted as T1 (`Schedule.peak_live_values()`, which
  counts loaded live-in elements as well as computed ones — an earlier version counted only the
  latter and reported 128 for a group whose thread holds 16 384 weight elements),
* `C` = the largest trailing axis never contracted within `G`,
* `d` = structural density, emitted terms / dense terms (`n_terms / dense_term_count`),

```
if G contains any index map (gather, scatter-add, or segment change):
    T3                              # the segment axis is re-mapped; no shared loop nest exists
                                    # KNOWN DEFECT (D36): this verdict is correct for the gather
                                    # itself and wrong to extend to ops fused alongside it --
                                    # it drags a [E,9,256] op out of T2's channel-parallel form
                                    # into T3's fully-unrolled one, 23,040 terms instead of
                                    # 5,123. Mitigated in S1 by a width cap; fixed in S2 by a
                                    # channel-parallel T3.
elif R <= REGISTER_BUDGET:
    T1                              # everything fits; strictly the cheapest
elif C >= 32:
    T2 with the channel axis on threads
else:
    split G and re-run              # no template applies; the group is too wide and too dense
```

T4 is never selected by this rule. It is a whole-program scheduling decision applied *after* a
template is chosen, not an alternative to one.

### Where the dense-MMA tile wins — measured, not assumed

Within T2 the contraction can be a dense MMA tile or a sparsity-eliding one. D22 predicted this
from structural density: at lmax=2 a `[9,9]` block occupies 3.4–13.7 % of an MMA tile, so a dense
tile should "pay quadratically for the zeros". **That prediction is wrong at these extents**, and
`bench/template_crossover.py` measures why (GH200, E=65536, C=128, bf16):

| lmax | density | dense-padded | dense-exact | per-degree blocks | structure wins by |
|---|---|---|---|---|---|
| 1 | 3.9 % | 0.257 ms | 0.631 ms | 0.561 ms | 0.46× |
| **2** | **13.7 %** | **0.263 ms** | 0.543 ms | 1.002 ms | **0.26×** |
| 3 | 32.8 % | 0.263 ms | 0.262 ms | 1.518 ms | 0.17× |
| 4 | 16.1 % | 0.362 ms | 0.851 ms | 2.218 ms | 0.16× |
| 6 | 11.1 % | 0.767 ms | 1.289 ms | 3.853 ms | 0.20× |
| 8 | 10.5 % | 1.236 ms | 2.298 ms | 5.809 ms | 0.21× |

Two facts, both against the density argument:

1. **Decomposing the block-diagonal into per-degree GEMMs loses everywhere** (0.16–0.87×). There
   is no crossover in this direction and none is expected: small-GEMM inefficiency and extra
   launches cost more than the skipped zeros save.
2. **Padding is cheaper than not padding.** At lmax=2 the padded tile performs (16/9)² = 3.2× the
   multiply-adds of the exact one and runs **2.1× faster**, because 16 is an MMA-friendly extent
   and 9 is not. If structural zeros were the binding cost, this is impossible.

So the rule is **not** "use a dense tile above some density". It is:

> **Structure is worth exploiting only *inside* a single kernel, by never *visiting* the zero
> terms. It is never worth exploiting by decomposing into more launches or smaller tiles.**

Which is exactly what T1 does and what T2 must do. The FLOPs saved are not the mechanism; the
*loads and stores avoided* are — consistent with FlashSO2's postmortem, which found L1TEX issue
throughput binding and compute idle, and with D23.

**"Visit", not "emit" — the rule binds every layer that walks the index space.** The first
statement of this rule said *emit*, and that turned out to be a loophole rather than a synonym.
The kernel does not emit zero terms; the **compiler still walks them**, enumerating the dense
index space once in `nonzero_masks` and again in `build_schedule`, filtering afterwards both
times.

**[profile]** Attributed by `cProfile`, not inferred: on forward g13, `_path_assignments` and
`_offset` account for 1.13 s of a 2.06 s schedule build, and `nonzero_masks` — the first of the
two walks — is 0.84 s, **41 %**. Fusing the passes so masks apply *during* enumeration is worth
up to that fraction of schedule construction, ≈ 28 s of dbwd's 68 s whole-program time (D33).

*An earlier version of this paragraph cited a 390-term group costing 9× a 321-term one. That
citation is withdrawn: those are T2 term counts and the work timed was the T1 schedule built
first for the register-budget check, whose counts are 99 840 and 41 088. There was no anomaly
(D33).*

So structural sparsity is a property the **whole pipeline** must honour, not an output property
of the emitter: schedule enumeration, mask propagation, and kernel emission each walk the same
index space and each must skip what is provably zero rather than discard it afterwards. The
masks needed to do this are already computed — they are simply applied one layer too late
(D30, scheduled as the first commit of S2).

**What is still unmeasured.** The third arm — a single fused kernel with channels on the warp
that skips zero terms — is T2 itself, so this table cannot yet say where a dense MMA tile beats
it. That crossover gets measured against the real T2 kernel once it exists (Phase 2 S1), not
extrapolated from these two arms. Until then the rule above is stated as a *constraint* (do not
decompose) rather than a *threshold*.

**Not comparable to FlashSO2's curve.** These numbers time a bmm on operands already
materialized per edge. FlashSO2's 0.55×/0.46×/0.33× include the `x_node` gather and the radial
multiply. Different measured boundaries; the two tables must not be read against each other.

---

## 3. Consequences for the fusion partition

The partition (`zippel.simplify.fusion_groups`) and the template rule interact:

* A group is only as good as its template. A partition that produces groups needing 3329 live
  scalars has not made a scheduling decision, it has deferred one.
* The partition must stay **acyclic**, or it is not a launch count at all. That is a hard
  constraint in the pass and a permanent test (`tests/test_ir_core.py`), after an unconstrained
  version put 101 of 107 dbwd groups into dependence cycles —
  `findings/fusion-partition-cycles.md`.
* Corrected group counts under the acyclic pass: **48 / 115 / 320** for fwd / force / dbwd,
  against 101 / 290 / 903 ops.

Any future number describing the partition ships with the structural test that validates it, in
the same commit. A planning artifact without a guard is how 107 got reported.
