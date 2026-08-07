# Phase 2 Work Order — CuTe DSL fused kernels

**Status: APPROVED with amendments, 2026-08-07.** Drafted from the template taxonomy
(`docs/templates.md`), D22–D27, and the corrected fusion-group table. Amendments folded in:
traffic model (§2 S1, §6), per-kernel error bounds (§5), S1/S2 comparators and acceptance
(§2, §3), T3 scope (§2 S2).

---

## 0. What Phase 2 is for

M1 tests whether three-pass joint compilation on a differentiation-closed IR beats per-operator
stacks on conservative training, in wall-clock and peak memory. Phase 1 established the IR half:
the vocabulary is closed under two VJPs, and the derived programs are exact against three
independent legs. Phase 2 has to make that produce *kernels*, or the bet is untested.

**The deliverable is the fusion partition, not a faster rotation (D23).** FlashSO2 spent nine
measured experiments establishing that the rotation itself has no headroom left inside its
contract, and concluded that the only remaining lever is *what the stage emits* — fusing
consumers so the dense store is never made. A Phase 2 that produces a 1.05× rotation and
materializes the same intermediates has missed the bet even if every kernel is individually fast.

**Non-goals carried over unchanged from the M1 work order:** no frontend or torch.compile-backend
integration, no MACE/DeePMD, no AMD/HIP/Triton portability layer, no LLM/agent loop, no autotuner
beyond a small manual sweep, no benchmark website, no multi-GPU, no general dynamic shapes.

---

## 1. Design commitments already made

| # | decision | basis |
|---|---|---|
| D22 | SIMT small-tile emission; **no WGMMA design attempted** | FlashSO2's end-to-end 0.55×/0.46×/0.33× at lmax 4/6/8, diagnosed structural |
| D23 | target the materialization contract, not kernel micro-efficiency | FlashSO2's own postmortem names it the only >10 % lever left |
| D24 | structure pays *inside* one kernel, never by decomposing launches | `bench/template_crossover.py`: per-degree GEMMs lose 0.16–0.87× at every lmax; padding is *cheaper* than not padding |
| D25 | two-tier exactness; every kernel ships its own ordering bound as metadata | T2 cannot be bit-exact against `einsum`; the bound is certified by a same-order reference |
| D26 | estimators that gate routing are upper bounds by construction, or carry a falsification test | `peak_live_values` under-reported 128 for a group needing 16 384 registers, and the refusal test failed by *not raising* |
| D27 | the per-group **byte** model is the optimisation objective, calibrated before it decides anything | D24's operational consequence: bytes are the mechanism, not FLOPs |

D24 corrects D22's stated mechanism while leaving its decision intact. The zeros are not the
binding cost — loads and stores are. That is why the templates elide zero *terms* inside a kernel
and never split a tile to chase sparsity.

---

## 2. Stages

### S1 — fused forward (E), templates T1 + T2

Emit the forward program as fused kernels under the selection rule. 101 ops, 48 acyclic fusion
groups, 35 archetypes.

* **S1a — T1, register-resident.** *Done.* Wigner chain: 7 ops, 6 internal `[E,9,9]` buffers
  never stored, 356 terms at 54 % structural elision, 108 live scalars, **bit-exact** against the
  FP64 interpreter over 9 576 real edges.
* **S1b — T2, cooperative tile.** *First kernels correct.* The channel-carrying groups: radial
  MLP (`[E,128]`, Linear+LayerNorm+SiLU) and SO(2) conv (`[E,9,256]`). Channels on threads,
  cross-channel contractions through gmem for live-ins and **shared memory for values produced
  in-group** — the latter live in other threads' registers, which is the only case that forces a
  barrier. 21 of 44 index-map-free forward groups fit T1; the rest are T2.
  Remaining: the `[E,9,256]` conv groups, and the multi-edge-per-CTA performance variant.
* **S1t — traffic model (early, before grouping decisions).** A per-group byte estimator:
  live-in bytes + live-out bytes + shared-memory spill. **Calibrated against ncu-measured DRAM
  traffic on the two kernels that already work** (Wigner T1, radial T2), with the error logged.
  **±20 %, or it is recalibrated before it drives any fusion or template decision** (D27). This
  is the objective function for S2 grouping and for the later rematerialisation choice, so it is
  not permitted to select its own inputs. Per D26 it is an upper bound by construction where it
  can be, and carries a falsification test where it cannot.
* **S1c — compose.** Drive the whole forward through emitted kernels, matching the interpreter
  end-to-end, then the reference block. **Includes minimal T3** — the edge gather, the edge→node
  scatter-add, the readout reduction, and LayerNorm mean/var — i.e. exactly the reduction-rooted
  kernels the forward needs and no more. *Full reduction-class generality remains S2.*

  **Every forward group is assigned or refused, before measurement:**

  | group | ops | verdict |
  |---|---|---|
  | 21 groups | — | **T1** — register-resident, built and validated |
  | 18 groups | — | **T2** — cooperative tile, built and validated |
  | g0 `evec_0` | edge[x:3], gathers `pos[src]`, `pos[dst]` | **T3 gather** |
  | g35 `cat_83`, `rotin_84` | edge[m:9,c:256], gathers `x_node[src]`, `x_node[dst]` | **T3 gather** |
  | g43 `scatter_100` | node[m:9,c:128], scatter-add by `dst` | **T3 scatter-add** |
  | g47 `E_105` | graph[], scatter through an all-zeros index | **T3 segment reduction** |
  | g5 `mean_9`, g9 `mean_18` | `c,c->` over 128 channels → edge scalar | **T3 intra-feature reduction** |
  | g7, g11 `var/vareps/invstd` | same, plus `rsqrt` | **T3 intra-feature reduction** |
  | g46 `rol1_104` | `i,oi->o` with output extent **1** — a reduction wearing a linear's clothes | **T3 intra-feature reduction** |

  Nothing is refused. The five groups the census called "unassigned" are one class, not five
  problems: `channel_axis` returned `None` for each because their *output* channel axis is rank-0
  (or extent 1), which is the signature of a reduction rather than a missing template.

  **Plan inconsistency, logged:** §2's S2 entry scopes T3 as an S2 deliverable, and S1's exit
  requires the forward to run end-to-end. The forward cannot run end-to-end without four
  index-mapped groups and five reduction groups. The two statements were written at different
  times and cannot both hold; the resolution is the split above — S1 builds the instances the
  forward needs, S2 builds the class. Recorded rather than silently reconciled, because it was a
  planning error and the next one is likelier to be caught if this one is visible.

**S1 exit:** forward correct end-to-end, plus a performance table.

**S1 comparators, parity-or-better:**

1. **Eager forward** at the Phase 3 measured boundary, same fixtures, same inputs.
2. **FlashSO2's measured curve at the lmax=4 anchor**, run **in its own environment** rather than
   quoted from its `RESULTS.md`. The anchor exists for exactly this comparison (D12) and is
   forward-only. Boundaries must be made comparable before the numbers are: FlashSO2's timings
   include the `x_node` gather and the radial multiply, so the comparison is constructed
   deliberately and its boundary stated, never assumed (D24).

### S2 — jointly scheduled (E, F), adds T3

Force is 290 ops, 115 acyclic groups, 78 archetypes. What S2 adds beyond S1 is **T3, the
reduction-rooted template class**, which covers two things that look different and are not:

* **segment reductions** — the edge→node scatter-add, the readout, and every transposed gather
  the VJP produces (the closure lemma turns each forward gather into a scatter-add);
* **LayerNorm-class intra-feature reductions** — mean and variance over the channel axis.

They belong to one class because both reduce along an axis the surrounding template treats as
parallel, and both therefore force a group boundary today. That second kind is why the acyclic
constraint fragments the radial MLP into five groups: `mean(x)` cannot fuse with the `x` it
reduces. T3 is what merges them, so it is not only about scatter-add.

Grouping decisions in S2 are made against the **calibrated traffic model** (S1t), not by op
count. Keep-vs-recompute for every intermediate shared between the E and F passes is logged in
`DECISIONS.md` as it is decided — that ledger is a Phase 2 deliverable, not bookkeeping.

**S2 acceptance:**

1. **(E, F) correct at template-tier tolerances** — T1 groups bit-exact, T2 groups within their
   emitted ordering bound (D25), asserted automatically per kernel.
2. **No dense inter-stage store of the conv output.** This is the D23 lever stated as a
   falsifiable condition rather than an aspiration: the `[E,9,256]` message tensor is 2.28 GiB at
   si_medium and is exactly the "dense store that is never made". Verified two ways — the traffic
   estimator must show it absent from the live-out set, **and** one ncu spot check must confirm
   the measured DRAM traffic is consistent with its absence. Estimator agreement alone is not
   evidence, since the estimator is our own artifact.
3. Equivariance, permutation and translation property tests re-run **on the generated kernels**,
   not only on the IR; finite-difference spot checks on F.

### S3 — full training step, T4 optional-if-time

dbwd is 903 ops, 320 acyclic groups, 149 archetypes, 57.32 GiB unscheduled peak-live at
si_medium. S3 emits the whole fwd/bwd/dbwd chain and schedules it jointly.

**T4 (persistent spine) evaluation is explicitly optional-if-time.** If S1 and S2 land and time
remains, measure whether a CTA-resident spine beats a sequence of fused launches. If it does not
land, S3 ships as jointly-scheduled fused launches and the report says so plainly. T4 is the most
speculative template and must not be allowed to consume the stage that carries the actual
measurement.

**S3 exit:** the full conservative training step correct against double-autograd and the FP64
interpreter, with `gradgradcheck` spot checks and one finite-difference leg through dbwd.

---

## 3. Gate 2 acceptance

**Gate 2 passes on: S2 correct, plus the S1 performance table.**

Explicitly:

* S1 forward correct end-to-end *and* timed against both comparators (eager; FlashSO2 at the
  lmax=4 anchor in its own env), parity-or-better;
* S2 (E, F) correct at template-tier tolerances, **and** no dense inter-stage store of the conv
  output — estimator plus one ncu spot check;
* S3 reported as reached, partial, or blocked, **stated plainly either way**; it is not a gate
  condition;
* green `pytest`, pasted into `REPORT.md`;
* every deviation logged in `DECISIONS.md`, dated, one line each.

**The Gate 2 table reports, for S1 and S2:** wall-clock against each comparator, **kernel launch
counts**, and **peak memory versus eager**. Launches and peak memory are not secondary columns —
they are the two axes on which joint compilation is supposed to differ from a per-operator stack,
and a wall-clock-only table would hide a result on either.

A clean negative result at Gate 2 — "the emitted kernels are correct and slower" — is a valid
outcome and must be reported as such. The only failure mode is a fabricated or gamed positive.

---

## 4. Measurement discipline (binding)

Inherited from the M1 work order and Gate 1's variance work, restated because Phase 2 produces
the first numbers anyone will want to quote:

* All reported numbers come from an **exclusive `sbatch` node** with `numactl --cpunodebind=0
  --membind=0`. Login-node GPUs are for development and `pytest` only.
* The verdict protocol is **N = 5 independent allocations, median-of-medians with the full
  range**. The error bar reported is the cross-allocation **spread**, never the flattering
  in-allocation IQR.
* **A speedup claim must exceed the measured baseline jitter for its configuration**: ~2 % at the
  medium fixtures, ~4–5 % at the small ones (`bench/results/verdict_summary.json`).
* Baselines run at their recommended fast settings, never crippled. Identical measured boundary
  and identical inputs for every party. Weights are runtime tensors — shape specialization only,
  no constant-folding. No tolerance loosening. Everything reproducible via `bench/run_all.sh`.
* Peak memory via `max_memory_allocated` reset per config, with an NVML cross-check.

---

## 5. Correctness ladder

Unchanged in structure from Phase 1, extended to the generated kernels:

```
generated CuTe DSL kernel
  ↓ T1: bit-exact in FP64.  T2: within the reduction-order bound (see below)
FP64 segmented-polynomial interpreter
  ↓ validated in Phase 1 at 2.6e-16 / 4.97e-15 / 3.7e-16
blocks/eso2_ref.py
  ↓ validated against fairchem 2.11 UMA SO2_Convolution at K4L2
fairchem
```

**The bar depends on whether the reduction order matches**, and both forms are stronger than a
loose tolerance:

* **T1 — bit-exactness.** Its sums are short and the emitter adds the same values in the same
  order as the interpreter, so any nonzero difference is a codegen bug. The Wigner chain meets
  this at 0.000e+00.
* **T2 — the reduction-order bound**, `eps * sqrt(n) * scale` for an n-term FP64 sum. A 128-wide
  channel contraction cannot be bit-exact against `einsum`: the interpreter reduces in a blocked
  order and the emitted kernel sequentially with FMA contraction, and neither is more correct.
  Measured 1.55e-15 against a bound of 3.62e-15 — and confirmed to be *purely* ordering, because
  a naive same-order FP64 reference differs from the interpreter by the identical 1.55e-15. The
  bound stays tight enough that a real error cannot hide under it: a wrong term or a missing
  barrier moves the result by O(1), not by an ulp.

**Every kernel ships its own bound, as a mechanism rather than a convention (D25).** The emitter
computes the ordering-error bound from the reduction tree it just emitted and attaches it to the
generated module as metadata; the harness asserts `measured <= bound` for every emitted kernel
automatically. A kernel cannot enter the suite without a bound, and a bound cannot be loosened
without changing the schedule that produced it — which is the property that stops "the tolerance
was widened" from ever being a quiet step.

Precision-appropriate tolerances apply only to the fp32/bf16 performance variants.

Property tests (rotation equivariance, permutation invariance, translation invariance) re-run on
generated kernels at every stage. The planted-sign-flip falsification test extends to the emitted
path, so the oracle is shown to have discriminating power over kernels and not only over IR.

---

## 6. Risks, and what each one costs

| risk | signal | fallback |
|---|---|---|
| T2 cross-channel contraction is slow in CuTe DSL | S1b misses FlashSO2's curve by >2× | report the gap; the memory axis is independently measurable and does not depend on it |
| the 320-launch dbwd partition is launch-bound | S3 wall-clock dominated by launch overhead | this is what T4 exists for; if T4 does not land, report launch-bound as the finding |
| register pressure forces T2 groups to split | more launches than the table predicts | re-measure and report the real count; the table is a planning artifact, and per §7 it ships with its guard |
| the win is memory-only, not wall-clock | speedup inside jitter, peak memory clearly lower | that is a legitimate verdict — the go criterion is ≥1.5× on *either* axis |
| the traffic model is wrong and picks bad groupings | calibration error >20 % against ncu on the two working kernels | recalibrate before it decides anything; it is gated on this by construction (D27), and until it passes, grouping stays op-count-based and says so |
| T3 does not land, so LayerNorm stays fragmented | S2 group count near S1's, mean/var still splitting groups | report the fragmentation as the measured cost of the acyclic constraint; it bounds the fusion win rather than invalidating it |

Blocked >~2 h on one issue: write the blocker into `REPORT.md`, move to the next independent
task.

---

## 7. Standing rule adopted after the 107 correction

**Never report a planning-artifact number without its structural guard existing first.** The
Gate 1 fusion-group count (42/46/107) was reported as a launch count while 101 of 107 dbwd groups
sat in dependence cycles, making it unachievable rather than optimistic. Every correctness test
passed throughout, because the partition never affected a computed value.

So: any number describing a *structure* (launch counts, group counts, live-range estimates,
schedule depth) ships in the same commit as the test that would falsify it. Numerical tests do
not cover structural claims.
