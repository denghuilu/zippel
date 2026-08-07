# Phase 2 Work Order — CuTe DSL fused kernels

**Status: DRAFT, awaiting review.** Drafted by me from the template taxonomy (`docs/templates.md`),
D22/D23/D24, and the corrected fusion-group table. Not started beyond S1a.

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
* **S1b — T2, cooperative tile.** The channel-carrying groups: radial MLP (`[E,128]`, two
  Linear+LayerNorm+SiLU stages) and SO(2) conv (`[E,9,256]`). Channels distributed across the
  warp, per-thread tile slice, cross-channel contractions through smem. **Bit-exact FP64 first,
  performance variants after** — the S1a discipline.
* **S1c — compose.** Drive the whole forward through emitted kernels, matching the interpreter
  end-to-end, then the reference block.

**S1 exit:** forward correct end-to-end, plus a performance table against B1/B2/B3 at the Phase 3
measured boundary. Parity target is FlashSO2's measured curve *for the rotation stage only*, and
only where boundaries are made comparable — their numbers include the `x_node` gather and radial
multiply, so a like-for-like comparison has to be constructed deliberately, not assumed (D24).

### S2 — jointly scheduled (E, F), adds T3

Force is 290 ops, 115 acyclic groups, 78 archetypes. What S2 adds beyond S1 is the segment-rooted
template: the edge→node scatter-add and the readout reduction, plus every transposed gather the
VJP produces (the closure lemma turns each forward gather into a scatter-add and vice versa).

Keep-vs-recompute for every intermediate shared between the E and F passes is logged in
`DECISIONS.md` as it is decided — that ledger is a Phase 2 deliverable, not bookkeeping.

**S2 exit:** F matches autograd and the FP64 interpreter; equivariance, permutation and
translation property tests re-run **on the generated kernels**, not only on the IR; finite-
difference spot checks on F.

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

* S1 forward correct end-to-end *and* timed against baselines at the measured boundary;
* S2 (E, F) correct — autograd, interpreter, property tests on generated kernels, FD spot check;
* S3 reported as reached, partial, or blocked, **stated plainly either way**; it is not a gate
  condition;
* green `pytest`, pasted into `REPORT.md`;
* every deviation logged in `DECISIONS.md`, dated, one line each.

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
  ↓ bit-exact in FP64 (the emitter reorders nothing, so any difference is a codegen bug)
FP64 segmented-polynomial interpreter
  ↓ validated in Phase 1 at 2.6e-16 / 4.97e-15 / 3.7e-16
blocks/eso2_ref.py
  ↓ validated against fairchem 2.11 UMA SO2_Convolution at K4L2
fairchem
```

**Bit-exactness is the bar for a generated kernel in FP64**, not a tolerance: the schedule
reorders no arithmetic relative to the interpreter, so any nonzero difference is a bug rather
than rounding. Precision-appropriate tolerances apply only to the fp32/bf16 performance variants.

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
