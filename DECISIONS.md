# DECISIONS

One dated line per deviation from the M1 work order: what changed, why.

## 2026-08-06 — Phase 0 setup

- **D1. Repo root moved `/capstor/scratch/cscs/dlu/iclr/spir-m1` → `/iopsstor/scratch/cscs/dlu/iclr/spir-m1`.**
  (The directory was later renamed to `zippel`; see D14.)
  `/capstor/scratch/cscs/dlu` is at 306.9 % of its 1,000,000-inode quota with grace **expired**;
  file creation fails with EDQUOT (reproduced: `touch: cannot touch '.../.spir_probe/t': Disk quota
  exceeded`). `/iopsstor/scratch/cscs/dlu` is writable with 727 T free and no file limit. Reads from
  `/capstor` are unaffected, so the conda base, FlashSO2 and fairchem stay readable in place.

- **D2. Python env is a conda env, not a pip venv; prefix is `/iopsstor/scratch/cscs/dlu/envs/spir`.**
  User instruction ("create a new conda environment named spir"). Built as
  `conda create -p ... --clone iclr` rather than from scratch: the existing `iclr` env already has a
  known-good aarch64 py3.13.14 / torch 2.13.0+cu130 / nvidia-cutlass-dsl 4.5.2 / fairchem-core 2.11.0
  stack that is hard to reproduce from public indices. Prefix is on `/iopsstor` because a clone into
  `miniforge3/envs/` would hit the D1 quota. Python is still 3.13 and packages are still pip wheels,
  so the work order's Section-1 intent is preserved.

- **D3. Ground-truth fairchem module is UMA `SO2_Convolution` / `Edgewise` (fairchem-core 2.11.0),
  not the `esen/` package.** The standalone `esen/` model package exists only in
  `fairchem_core==2.0.0`, which declares `Requires-Python >=3.9,<3.13` and cannot be installed on
  py3.13. UMA's `escn_md_block.py` + `nn/so2_layers.py` is its maintained descendant with the same
  math. Known algebraic deltas to state in REPORT.md: `to_m` hoisted out of `SO2_Convolution` into
  the Wigner matrix; `SO2_m_Conv` returns `(real, imag)` instead of `cat`; envelope always applied;
  `GateActivation(m_prime=True)`. Block hyperparameters still come from the smallest **published
  eSEN** config (K4L2), per the work order.

- **D4. Wigner-D is expressed in rational form; the declared transcendental set is NOT widened.**
  User instruction: algebraically simplify `sin/cos ∘ acos/atan2` where trivial. With
  β = acos(ŷ): `cos kβ = T_k(ŷ)` (polynomial) and `sin kβ = √(1−ŷ²)·U_{k−1}(ŷ)` where
  `√(1−ŷ²) = (1−ŷ²)·rsqrt(1−ŷ²)`; with α = atan2(x̂, ẑ): `cos α = ẑ·rsqrt(x̂²+ẑ²)`,
  `sin α = x̂·rsqrt(x̂²+ẑ²)` and higher harmonics by De Moivre. Every Wigner entry is therefore
  rational in the normalized edge vector, needing only `rsqrt` — already in the declared set
  {exp, sigmoid/SiLU, rsqrt, poly-envelope}. Closure holds without adding sin/cos/acos/atan2.
  **Future work (out of M1 scope):** a fully angle-free Wigner recursion built directly from the
  edge unit vector, avoiding the Euler parameterisation altogether.

- **D5. A double-differentiable `acos` replaces fairchem's `Safeacos` in *all* implementations.**
  User decision. fairchem's `Safeacos.forward` calls `ctx.save_for_backward(x.clamp(...))` with the
  clamp evaluated under no-grad, so the saved tensor carries no graph and the first derivative's
  dependence on `x` is invisible to a second differentiation. Measured on GH200 in FP64 against
  `uᵀW(pos)v`: energy exact, force agrees to 8.9e-16, but **‖∂²‖ = 3103.208 vs 3091.557, a 5.5 %
  relative error** — silent, no exception. Consequences: (a) every implementation computes identical
  math so the benchmark is apples-to-apples; (b) **fairchem autograd is disqualified as the dbwd
  oracle** — the FP64 SP-IR interpreter plus finite differences take that role; (c) the shipped
  defect is reported in REPORT.md as a verification finding.
  Note `Safeatan2` is double-differentiable and correct, but its `backward` is
  `@torch.compiler.disable`d — recorded as evidence for the B2 torch.compile fallback inventory.

- **D6. Neighbor lists use the full 6.0 Å ragged cutoff list; the published `max_neighbors=30` cap
  is NOT applied.** The cap is a data-loader detail with a nondeterministic tie-break among
  equidistant neighbors, and it is not part of the block's math. Applying it would change edge count
  by ~1.5–2.5× and inject nondeterminism into a benchmark that must be reproducible. Average degree
  is recorded per fixture instead.

- **D7. γ (the Wigner roll angle) is a seeded input, not `torch.rand_like` inside forward.**
  fairchem draws γ randomly per call (`init_edge_rot_euler_angles`); the SO(2) convolution is exact
  in γ so this is legitimate for training, but it makes forward non-reproducible and would poison
  both correctness tests and benchmark variance. γ is generated once per fixture from a fixed seed
  and passed in as a per-edge input to every implementation. Verified: seeding makes it reproducible
  and it does differ run-to-run without a seed.

- **D8. SLURM account is `lp16`** (work order Section 1), not the user's habitual `c33`. Confirmed
  valid via `sacctmgr` association and `id` (`groups=...,33070(lp16)`), and far less used
  (RawUsage 11.6 M vs 690.8 M). Partition `normal`, `--gres=gpu:4`, 24 h cap.

- **D9. The energy readout uses per-`l` rotation invariants, not eSEN's linear `l=0` head.**
  eSEN reads the `l = 0` row linearly, but does so after a *stack* of blocks. On a single block
  that head is degenerate: the `l = 0` output row is fed only by the m = 0 branch, so
  `c1_m.{0,1}.weight` and `c2_m.{0,1}.weight` — all four m > 0 convolution weight tensors, i.e.
  the SO(2) machinery this milestone exists to measure — receive **exactly zero gradient**
  (measured, not inferred). The readout instead takes `l = 0` linearly plus the per-`l` squared
  norms `Σ_m x[l,m,c]²`, which are invariant because each Wigner block is orthogonal. Every output
  row is live, every parameter gets a gradient, and E stays exactly rotation-invariant. Guarded by
  `tests/test_ref_block.py::test_all_parameters_receive_gradient`.

## 2026-08-06 — Gate 0 review follow-ups

- **D11. Fixtures regenerated under a canonical schema; the Gate 0 baseline table was
  re-measured on them.** Gate 0 review required either proving the canonical writer reproduces
  the existing `.npz` bit-exactly, or regenerating and re-running. Bit-exact reproduction was
  **not** available: the writer had diverged in three independent ways (per-fixture seeds
  `1000+idx` → a single `seed=0`; an `atoms.wrap()` that was dropped; and the neighbour-list
  backend moving from ASE to vesin once vesin was installed), which changes both positions and
  edge order. So option (b): regenerate, then re-run B1 and B2 on the new fixtures.
  Schema v1 is now: `schema_version`, `pos`, `cell`, `atomic_numbers`, `edge_index`, `shifts`,
  `meta` (JSON). `make_fixtures.py` is the only writer, `load.py` the only reader (its duplicate
  `load()` was removed), and the loader rejects a mismatched `schema_version` loudly.
  **Edges are sorted into a canonical order** (src, dst, then shift) so the content hash does not
  depend on which neighbour-list backend produced them. Two hashes are recorded per fixture:
  `sha256_content` over array bytes in fixed key order (reproducible anywhere — this is what the
  round-trip test asserts) and `sha256_file` over the `.npz` bytes (not reproducible across runs,
  because `np.savez_compressed` embeds zip timestamps; recorded for artifact integrity only).
  Edge counts shifted slightly, as expected: si_small 9 620 → 9 576, si_medium 259 542 → 259 474.

- **D12. A secondary `lmax = 4` shape bucket exists solely as the Phase-2 S1 forward anchor.**
  FlashSO2 cannot run the M1 config (D10), which would leave stage S1 with no fused-kernel
  forward to be "within noise of". The anchor bucket is therefore scoped as narrowly as possible:
  **forward only — no backward, no double backward, no radial/gate/envelope training path** —
  used for (i) a correctness check of our generated forward against the FP64 interpreter at a
  second shape, and (ii) a wall-clock comparison against FlashSO2's forward, run in FlashSO2's
  own environment. It is **excluded from the Gate 3 verdict table** and from every speedup or
  peak-memory claim, which stay at the published eSEN config (lmax = 2). Its only purpose is to
  stop "our forward is plausible" from being an unchecked assertion.

- **D13. Max-batch metric is stated in GiB, primary budget 80 GiB.** The work order says "80 GB"
  against a 95.6 GiB card, which is ambiguous about base-10 vs base-2 and about whether the budget
  is the card or a fixed target. Resolved at Gate 0 review: **primary = largest replication factor
  of the medium fixture whose measured `torch.cuda.max_memory_allocated` stays ≤ 80 GiB**, found
  by binary search; **secondary = the same at the full-card 95.6 GiB**. Units are written as GiB
  everywhere, and the number reported is measured peak allocation, not a capacity estimate.

- **D10. B4 (FlashSO2 forward reference) is not applicable at the M1 block config.** FlashSO2
  `main` enforces `SUPPORTED_LMAXES = (4, 6, 8)` with `mmax == lmax` and raises
  `ValueError("FlashSO2 main supports only lmax in (4, 6, 8) ...")`
  (`flash_so2/_triton/common/layout.py:30`). The smallest published eSEN config is **lmax = 2**, so
  FlashSO2 cannot run it. Consequences: B4 is reported as not-applicable rather than as a number,
  and the Phase-2 S1 sanity anchor ("within noise of FlashSO2 forward on the medium fixture") is
  unavailable at this config. If a fused-kernel forward anchor turns out to be needed in Phase 2,
  the option is a *secondary* lmax = 4 configuration used only as an anchor, never mixed into the
  headline lmax = 2 table. Not pursued unless Phase 2 requires it.

## 2026-08-06 — repository rename

- **D14. Working tree renamed `spir-m1` → `zippel`, and the git remote is
  `git@github.com:denghuilu/zippel.git`.** Instructed at Gate 0 review after the remote was
  created. Absolute paths updated in `env.sh`, `tests/conftest.py`, `tests/test_environment.py`,
  `bench/count_launches.py` and `slurm/bench.sbatch`, and the environment-variable prefix renamed
  `SPIR_M1_* → ZIPPEL_*` (`ZIPPEL_ROOT`, `ZIPPEL_ENV`, `ZIPPEL_CACHE_ROOT`) so a repo called
  `zippel` does not carry `SPIR_M1_CACHE_ROOT` in its cache configuration.

  Three references to `spir-m1` are kept deliberately, as historical record rather than oversight:
  D1 above and REPORT.md §1 both quote the work order's original `/capstor/.../spir-m1` path, and
  PLAN.md's deviations table records the path as approved at planning time. D1's *target* was
  restored to `.../spir-m1` so it describes only the Gate-0 filesystem move; this entry records
  the later rename, keeping the two decisions separate.

  **JIT caches were deleted, not moved.** Cached compiler artifacts can key on absolute source
  paths, so carrying them across a path change risks a stale hit — precisely the silent-re-JIT
  failure mode the cache pinning exists to prevent. They rebuild on first use.

  `slurm/bench.sbatch` now `cd`s to `${ZIPPEL_ROOT:-/iopsstor/scratch/cscs/dlu/iclr/zippel}`
  rather than relying on `SLURM_SUBMIT_DIR`, so a job submitted from anywhere lands in the repo.

  The conda environment keeps its name `spir` at `/iopsstor/scratch/cscs/dlu/envs/spir`: it was
  named by explicit instruction, renaming it is slow and risks breaking the working torch 2.13 /
  CuTe DSL stack, and nothing depends on its name matching the repo's. `ZIPPEL_ENV` points at it.

  Note this is a deviation from the work order's scope fence ("No new project names or
  branding"); it is a direct instruction from the reviewer, recorded here rather than silently.

## 2026-08-06 — Phase 1 kickoff

- **D15. The Python package is `zippel`; the IR is called the "segmented-polynomial IR" in prose,
  never "SPIR".** Speech collision with SPIR-V. `spir/` was renamed to `zippel/`; no module name
  contains `spir`. (The Phase 1 work order cites this as "D11", but D11 here is already the
  canonical-fixture regeneration — recorded under this number instead so the cross-reference
  resolves.)

- **D16. `poly_envelope` is a vocabulary *family*, carrying a derivative-order attribute.**
  Vocabulary v1.1 fixes eight `scalar_map` functions. Seven have derivatives expressible as
  products of the other seven, so they close trivially. `poly_envelope` does not: its derivative
  is a *different* piecewise polynomial, and the dynamic indicator `[d < 1]` cannot be written as
  a `segmented_contraction` over static coefficients. Two options: add `poly_envelope_d1`/`_d2` as
  separate vocabulary entries (widening the set, which the closure test exists to prevent), or
  give the single entry an integer `order` that differentiation increments. The second is chosen:
  closure is then exact — `d/dx poly_envelope(k) = poly_envelope(k+1)` — and the vocabulary stays
  at eight named functions. Flagged rather than buried, since it is the one place where "the
  derivative stays inside the set" needs a definition rather than a proof.

- **D17. A `ContractionPath` names which operands it reads.** Implementing the VJP exposed that
  one subscript group per op-input cannot express **addition**: `einsum("ic,ic->ic", a, b)` is a
  product. Cotangent accumulation is addition, so the model could not have been made to work.
  Paths now carry `operands: tuple[int, ...]`, so a sum is one op with two single-operand paths
  and a product is one op with one two-operand path. `docs/ir.md` updated in the same commit.

- **D18. The transpose of a broadcast is a scatter-add through an all-zeros index map.**
  A `none`-segment operand (a parameter) is broadcast over the segment axis in the forward
  direction; its VJP must therefore *sum* over that axis. This is not a gather/scatter swap and
  initially failed the type check. It needs no new op — scattering into a length-1 buffer with an
  all-zeros index is a segment sum, already in the vocabulary — so closure is unaffected. The
  transform takes a `zero_index` map from segment name to such a buffer.

- **D19. Axis selection and scalar placement are contractions against a static unit operand,
  not slice-and-reduce.** Extracting component `i` of an edge vector as a rank-0 scalar is
  naturally written `"x->"` with the operand sliced to `[i:i+1]` — but that sums an index
  appearing in only one operand, which the vocabulary rejects because its transpose needs a
  broadcast (docs/ir.md 2.1). The same problem appears in reverse when placing a scalar into a
  1x1 slot of a matrix: `"->ij"` produces indices no operand supplies.

  Both are solved the same way, without touching the vocabulary: contract against a
  `none`-segment buffer of ones (`unit`, shape [1,1]; `unit_mat`, shape [1,1,1]). Selection
  becomes `"x,x->"` and placement becomes `",ij->ij"`, so every index is supplied by some
  operand and every transpose is an ordinary contraction. This is philosophically the right
  shape for this IR anyway — selection *is* a contraction against a static coefficient table,
  which is what "segmented polynomial" means.

  Worth recording because it was found by the type checker rejecting three successive
  formulations, not by design: the restriction in 2.1 is doing real work.

## 2026-08-07 — Phase 1 assembly

- **D20. The element-embedding table lookup is outside the IR program.** `emb_src`/`emb_dst`
  arrive as edge-segment inputs. A table lookup is a gather along a *non-segment* axis, which
  the v1.1 index-map model does not carry, and it is position-independent — so it lies outside
  the position→E path that forces and the double backward flow along, which is what Phase 1
  tests. Its VJP is the same scatter-add lemma already exercised. Adding it would mean either a
  second kind of gather or a vocabulary extension; neither is warranted for what it buys.

- **D21. A path may read the same operand more than once, and each occurrence gets its own
  VJP contribution.** `x*x` is a single path with `operands = (0, 0)`. Taking
  `operands.index(k)` finds only the first occurrence and silently halves the derivative —
  which is exactly what happened: the forward was exact at 2e-16 while F was wrong by 2.6e-01.
  The transform now iterates over every position where operand `k` appears.

  Worth separating from the diamond test: that one covers **buffer**-level accumulation (one
  buffer, several consuming ops). This is **path**-level accumulation (one path, one operand,
  several occurrences). They are different sites and the first does not imply the second, so
  there is now a dedicated regression test for each.

- **D22. `1 - u` is never materialised; the sigmoid and silu derivatives are expanded instead.**
  Writing `sigmoid' = y(1-y)` needs a broadcast constant `1` matching each operand's type, and
  the block applies sigmoid and silu to buffers of four different shapes — so it would have
  needed a ones buffer per shape. Expanding to `y - y²` and `s + xs - xs·s` keeps every
  derivative a plain sum of contraction paths over buffers that already exist, and removes an
  input from the transform's signature.


## 2026-08-07 — D22: Phase 2 emits SIMT small-tile kernels, not WGMMA. My plan's premise was wrong.

PLAN.md justified a WGMMA design by claiming lmax=2 is "the most favourable point" on FlashSO2's
measured curve, because the block-diagonal is densest there: "35 nonzeros of 81 = 43%, vs
969/9216 = 10.5% at lmax 8". That comparison is unpadded for lmax 2 (81 = 9^2) and padded for
lmax 8 (9216 = 96^2). A dense WGMMA tile costs nnz/nc_pad^2, and under consistent accounting:

  lmax   nc  nnz   nnz/nc^2   nc_pad(32)  nnz/pad^2   nc_pad(16)  nnz/pad^2
     2    9   35      43.2%           32       3.4%           16      13.7%
     4   25  165      26.4%           32      16.1%           32      16.1%
     8   81  969      14.8%           96      10.5%           96      10.5%

lmax=2 is the *worst* point, not the best: nc=9 cannot fill any WGMMA N/K granularity. The
configuration FlashSO2 measured at 0.55x of production Triton is lmax=4 at 16.1%; we would be
building at 3.4-13.7%. Their diagnosis -- "the dense N x K WGMMA tile pays quadratically for the
block-diagonal zeros" -- applies to us harder than to them.

Decision: Phase 2 emits **SIMT small-tile kernels** -- per-degree register tiles over the segment
axis, the design their production Triton uses and their nine-experiment campaign could not beat by
changing the engine. WGMMA is not attempted. This is a design decision taken on someone else's
measured evidence rather than by re-deriving it at a cost of days, and it is recorded as such.

## 2026-08-07 — D23: Phase 2's target is the materialization contract, not kernel micro-efficiency.

FlashSO2's postmortem closes with the one lever their campaign left open:

  "The only lever with >10% of headroom left is the contract: the dense [E, nc, 2C] bf16 store is
   838 MB (58% of all DRAM traffic at lmax 4). ... Chasing the remaining in-kernel percents is not
   worth further effort; changing what Stage 1 emits is."

and, from the INT8 follow-up:

  "Any real win must reduce instructions or stores (e.g. fusing consumers of pre_conv so the dense
   store is never made), not input bytes."

That is this program's bet, reached independently by a different project on the same hardware and
the same block. It is corroboration of the hypothesis, not evidence for the conclusion -- they
identified the lever; whether joint three-pass compilation actually pulls it is exactly what M1
measures.

Consequence for Phase 2: the deliverable is the **fusion partition**, not a faster rotation. Gate
1 measured 42 / 46 / 107 fusion groups against 101 / 290 / 903 ops for fwd / force / dbwd, and
peak-live of 6.03 / 20.73 / 57.32 GiB under a naive unscheduled order. Those stores are the
target. A Phase 2 that produces a 1.05x rotation and materializes the same intermediates has
missed the bet even if every kernel is fast.

## 2026-08-07 — D24: D22's decision stands, its stated mechanism does not. Measured.

D22 rejected a dense-MMA rotation and gave the reason as structural density: a dense tile "pays
quadratically for the block-diagonal zeros", with lmax=2 at 3.4-13.7 % occupancy of an MMA tile.
The template selection rule needs that as a *curve*, so `bench/template_crossover.py` measures it
on GH200 -- batched `[E,nc,nc] @ [E,nc,C]`, E=65536, C=128, three strategies.

    bf16          density   dense-pad   dense-exact   per-block   block wins by
    lmax 2         13.7 %     0.263 ms      0.543 ms    1.002 ms          0.26x
    lmax 4         16.1 %     0.362 ms      0.851 ms    2.218 ms          0.16x
    lmax 8         10.5 %     1.236 ms      2.298 ms    5.809 ms          0.21x

Two results, both against the density argument:

1. **Decomposing the block-diagonal into per-degree GEMMs loses at every lmax** (0.16-0.87x).
   Exploiting structure by *splitting launches* costs far more in small-GEMM efficiency than the
   skipped zeros save.
2. **Padding is cheaper than not padding.** At lmax=2 bf16 the padded tile does (16/9)^2 = 3.2x
   the multiply-adds of the exact one and is 2.1x *faster*, because 16 is an MMA-friendly extent
   and 9 is not. If the zeros were the binding cost this could not happen.

So density does not predict dense-MMA cost at these extents, and D22's mechanism is wrong. The
*decision* is unaffected -- it rests on FlashSO2's end-to-end measurement (0.55x/0.46x/0.33x),
and their postmortem's actual diagnosis is not FLOPs but "the gather cannot become a bulk copy",
"WGMMA fragment layouts fight coalescing", and L1TEX issue throughput as the top limiter. Their
own words: "Idle compute today, but it caps any future compute-bound version" -- i.e. the zeros
were explicitly *not* the binding cost, which I should have read more carefully before restating
it as one.

Not comparable, and deliberately not compared: these numbers measure a bmm on operands already
materialized per edge, while FlashSO2's include the x_node gather and the radial multiply.
Different measured boundaries.

What this means for the selection rule (docs/templates.md): structure is worth exploiting only
*inside* one kernel, by never emitting the zero terms (T1/T2), never by decomposing into more
launches. The third arm -- a single fused kernel skipping zeros with channels on the warp -- is
what T2 is, and the crossover will be re-measured against it once T2 exists rather than
extrapolated now.

## 2026-08-07 — D25: two-tier exactness policy, and every kernel ships its own bound.

A generated kernel is checked against the FP64 interpreter at one of two tiers, decided by the
emitter from the schedule, never by the person writing the test:

* **T1 — bit-exact.** The schedule's sums are short and the emitter adds the same values in the
  same order as the interpreter, so any nonzero difference is a codegen bug. Wigner chain:
  0.000e+00 over 9 576 edges.
* **T2 — bounded by reduction order.** A 128-wide channel contraction cannot match `einsum`
  bitwise: the interpreter reduces blocked, the emitted kernel sequentially with FMA
  contraction, and neither is more correct. The bound is `eps * sqrt(n) * scale` over the
  kernel's deepest reduction. Certified rather than assumed: a naive *same-order* FP64
  reference differs from the interpreter by the identical 1.554e-15, which is what proves the
  discrepancy is ordering and not arithmetic.

Mechanism, not convention: `emit_source` computes the bound from the reduction tree it just
emitted and attaches it as kernel metadata, and the harness asserts `measured <= bound` for
every emitted kernel automatically. A kernel cannot be added without a bound, and a bound
cannot be loosened without changing the schedule that produced it.

This is the precision contract arriving earlier than planned. It was to be a Phase 3 concern
(mixed-precision policy per table row); T2 forced it at S1b because the honest bar stopped
being "equal". Noted in REPORT.md.

## 2026-08-07 — D26: resource estimators that gate routing must be upper bounds, or carry a falsification test.

`Schedule.peak_live_values()` decides whether a group is emitted as T1 or refused. It counted
only computed values and ignored the live-in elements each thread holds, reporting **128** for a
group whose thread reads all 16 384 weight elements of a `[128,128]` Linear. The T1-refusal test
did not catch it -- it failed by *not raising*, which is the signature of an estimator that is
wrong in the permissive direction. Corrected, the SO(2) conv group needs 492 929 live scalars
per thread, not the 3 329 first reported.

Policy, generalising it: **any estimator whose value causes the compiler to route or refuse must
be an upper bound by construction, or must ship a test that fails when it under-reports.** A
precondition check that undercounts is worse than no check, because it converts "this will spill
catastrophically" into "this is T1-eligible" and carries the authority of a measurement.

Applies to `peak_live_values` (registers), the smem estimate, and the traffic model of D27 --
which is why that one is calibrated against ncu rather than trusted.

## 2026-08-07 — D27: the traffic estimator is the optimisation objective, and it is calibrated before it is used.

D24 established that the FLOPs saved are not the mechanism -- the loads and stores avoided are.
Its operational consequence is that Phase 2 needs a per-group **byte** model, not a term count:
live-in bytes + live-out bytes + smem spill, per fusion group.

That model becomes the objective for S2 grouping and for the later rematerialisation choice, so
it is calibrated before it decides anything: measured against ncu DRAM traffic on the two
kernels that already work (Wigner T1, radial T2), with the error logged. **±20 % or it is
recalibrated before it drives any fusion or template decision.** An uncalibrated cost model that
selects its own inputs is how a compiler talks itself into a bad schedule.

## 2026-08-07 — D28: the traffic model has two roles, and the T1 calibration disqualifies it from only one.

Measured: the model predicts T2 groups to 2.8 % and T1 groups to +26.9 %, always in the same
direction -- it **under-predicts** T1 traffic, i.e. the kernel moves more than the model charges.
That single number licenses one role and forbids the other:

* **As a budget / refusal bound (D26): valid.** D26 asks that an estimator gating a decision be
  an upper bound. Once the sign is known to be one-directional, the model *inverted* -- charge
  T1 groups their measured 1.27x factor -- is a conservative ceiling, and a ceiling is all a
  refusal test needs. "Will this group's traffic fit the budget" tolerates a 27 % margin.
* **As a ranking objective (D27): invalid.** Choosing between two fusions asks for the *sign of a
  difference*, and a 27 % one-sided error swamps any gap smaller than that. A model that ranks A
  above B when the true order is the reverse is worse than no model, because it carries the
  authority of a number.

**Operating rule for S2 grouping.** The model decides only when the predicted gap between
alternatives exceeds the uncertainty band for the templates involved (2.8 % where both are T2,
26.9 % where any T1 group participates). Alternatives *inside* the band are not modelled --
they are emitted and timed, and the measurement decides. That keeps the model in the role it has
earned and puts the burden of proof on the hardware everywhere else.

This also sets the priority for improving it: the L2-reuse term is worth modelling exactly to the
extent that S2 turns out to have close calls involving T1 groups. If it does not, the band is
adequate and the effort belongs elsewhere.

## 2026-08-07 — D29: traffic measurements use real fixture connectivity; only values are synthetic.

Stage 2 of `bench/traffic_calibrate.py` synthesizes inputs from IR types because the full FP64
forward at si_medium materialises ~40 GiB and gets OOM-killed, and because DRAM traffic depends
on shapes and access patterns rather than values.

"Access patterns" is doing real work in that sentence, and it is where the shortcut can go wrong.
A gather or scatter through a **random** index buffer has entirely different L2 line reuse from
one through a real neighbour list, where edges are largely sorted by source atom and consecutive
edges touch overlapping cache lines. That is precisely the term the T1 residual identified as
unmodelled, so measuring it against synthetic connectivity would be measuring the wrong graph.

Rule, asserted in the harness rather than remembered: **index buffers come from the real fixture;
only floating-point values are synthetic.** A traffic measurement that cannot obtain real
connectivity fails rather than silently substituting zeros -- an all-zeros index map is the
best-case gather (perfect reuse, one line) and would flatter every scatter-add kernel S2 emits.

## 2026-08-07 — D30: "near-linear schedule construction" is an S3 entry criterion. Measured 1.27; not yet met.

The dbwd emission preflight stalled — 320 acyclic groups, fifteen minutes, not one row — so the
question it was asked (does the AST-chunking fix hold at S3 sizes?) was answered by a different
one: **schedule construction, not emission, is the scaling limit.** `bench/schedule_scaling.py`
measures it.

Cost is driven by the **dense index-space volume** the constructor walks, not by the sparse terms
it emits. Fitting the forward's 8 measured groups log-log:

    t ~ volume^1.27   R^2 0.976      <- mechanistic
    t ~ terms^1.00    R^2 0.348      <- no relationship worth the name

That is the finding in one line, and it is slightly embarrassing in the right way: the sparsity
pass exists so the *kernel* does not pay for structural zeros, and the *compiler* pays for them
in full. A group with 390 emitted terms takes 7.1 s while one with 321 takes 0.77 s, because the
first walks a 99 840-element index space and the second 41 088.

Absolute cost matters as much as the exponent. The largest forward group (658 048 volume, 5 132
terms) takes **94 s** to schedule.

*Corrected after measuring dbwd:* I first extrapolated to 5 M- and 20 M-volume dbwd groups at ~21
minutes and ~2 hours apiece, reasoning from "dbwd is 9x the forward". That was the wrong axis.
dbwd's largest index space is **666 112**, essentially the forward's 658 048 — the growth is in
group *count* (251 schedulable vs 44), not group size. Fitted on dbwd's own measurements,
`t = 9.25e-7 * volume^1.40` (R^2 0.951) over a total volume of 7 747 454 gives a whole-program
schedule time of **~19.6 min**, with 8.7 of those minutes in five groups. Wrong by about an order
of magnitude, and wrong in the direction that would have justified more alarm than the situation
warrants.

**Entry criterion for S3, recorded now so it is not negotiated later:** schedule construction
must be near-linear in index-space volume (k <= 1.2) *and* the whole dbwd program must schedule
in bounded wall-clock. Currently k = 1.27 — marginally over — with the constant the larger
problem.

**Fix timing, ruled: the first commit of S2, before any grouping search.** My original plan was
to defer it as gold-plating. That is wrong for one specific reason: D28's operating rule says
alternatives whose predicted traffic gap falls inside the uncertainty band get *emitted and
timed* rather than modelled — which places schedule construction in the inner loop of a grouping
search, where a 52-115 s build multiplies by the number of candidates considered. Deferring is
cheap only while the constructor runs once per group.

The fix: skip provably-zero regions *during* enumeration instead of filtering after. The masks
are already computed and merely applied one layer too late. Acceptance is a re-run of
`bench/schedule_scaling.py` meeting k <= 1.2 with the constant-factor improvement reported
against today's table. S1c proceeds first, untouched.
