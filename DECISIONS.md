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

  *(Reversed by D34 on 2026-08-07 — the env is now `envs/zippel`. The reasoning below stood
  until the rename was explicitly requested, and the "risks breaking the stack" concern was
  answered by verification rather than by assertion.)*

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

**[extrapolation → superseded by measurement, D37]** Absolute cost matters as much as the
exponent. The largest forward group (658 048 volume, 5 132 terms) takes **94 s** to schedule.

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

**Resolved, and the diagnosis it was based on was wrong.** See D31 — the profile shows
enumeration is 3 % of the cost and my liveness analysis is 97 %. Criterion now met: k = 0.96–1.01
across all three programs, dbwd whole-program 68 s. The text below records the reasoning as it
stood.

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


## 2026-08-07 — D31: the schedule-construction cost was my liveness analysis, not the enumeration. D30 met.

D30 blamed schedule construction on walking the dense index space, and the fix was ruled to be
"apply the zero-masks during enumeration", scheduled as S2's first commit. **`cProfile` refutes
that diagnosis.** On dbwd g210 (265 s total):

    peak_live_values      256.8 s      97 %
    build_schedule          7.7 s       3 %   <- the enumeration the ruling targeted
    dict.get       454,954,368 calls, 121 s

`Schedule.peak_live_values` rescanned the whole live set at every assignment to find values whose
last use had passed -- O(assignments x live), and it is called on every group because it is the
D26 register-budget precondition that decides T1 vs T2. Bucketing values by the step they die at
expresses the same work once per value instead of once per value per step. Five lines.

    group                     before      after   speedup
    fwd  g13   99,840 vol      7.15 s    0.78 s      9.2x
    fwd  g36  492,160 vol     52.31 s    4.80 s     10.9x
    fwd  g40  658,048 vol     94.22 s    7.16 s     13.2x
    dbwd g210 331,136 vol    115.10 s    2.85 s     40.4x

    exponent vs volume       fwd 1.27 -> 0.97 (R^2 0.99)
                             force    -> 0.96 (R^2 0.97)
                             dbwd 1.40 -> 1.01 (R^2 0.99)

    whole-program dbwd        19.6 min -> 68 s      17x

**D30's dual criterion is met**: k <= 1.2 on all three programs, and the whole dbwd program
schedules in about a minute.

Two things worth stating plainly:

1. **I fixed this now rather than at S2's first commit as ruled.** The ruling targeted
   enumeration; the measurement says enumeration is 3 %. The fix that was ruled on would have
   bought at most a few percent, and this one is five lines with no design risk and speeds up
   S1c's own repeated builds. The *timing* intent -- fast construction before it enters a search
   inner loop -- is satisfied earlier rather than later. The masks-during-enumeration change is
   now a ~3 % optimisation and I would not spend S2's first commit on it; that is for review.
2. **The profile contradicted a hypothesis I had already written into three documents.** D30,
   REPORT 8.5d and docs/templates.md all attribute the cost to walking the dense index space, and
   two of those went out before anyone profiled anything. The volume *correlation* was real --
   R^2 0.976 -- because bigger index spaces produce more assignments and the quadratic scan is in
   the assignment count. A tight fit to a plausible mechanism is not evidence for that mechanism,
   which is the same lesson as the DCGM constant, one layer up.

## 2026-08-07 — D32: every causal attribution ships with its evidence class.

Three times now a plausible mechanism has been inferred from a good fit and been wrong: the DCGM
constant (a 0.4 % residual "confirming" a bandwidth that exceeded the device's peak), the
schedule-cost attribution (R² 0.976 against index-space volume, 97 % of it actually a quadratic
liveness scan), and the `math_dtype` claim (a residual plus a warning joined into a mechanism
without varying anything). Each was caught, but only by someone going back to measure.

**Law, applying to REPORT, `findings/` and `docs/`:** every causal claim carries its evidence
class inline.

| class | means | may be stated as |
|---|---|---|
| **[correlation]** | X and Y move together; nothing was varied or attributed | a correlation, and only that |
| **[profile]** | cost or behaviour attributed to a named component by direct measurement | a mechanism, within what the profiler resolves |
| **[intervention]** | the suspected cause was changed and the effect moved | a cause |

**A correlation may ship only as a correlation, never as a mechanism.** "X is driven by Y" needs
[profile] or [intervention]; with [correlation] the sentence must read "X tracks Y", and the
mechanism must be named as unverified or omitted.

This binds review as well as writing: a mechanism claim gets asked its evidence class before
ratification. The three shipped documents that carry unlabelled mechanism sentences —
REPORT §8.5c/§8.5d, `findings/traffic-model-calibration.md`, `docs/templates.md` — are relabelled
retroactively.

## 2026-08-07 — D33: the "never visit zero terms" evidence is retired; the principle is re-costed.

D24's generalisation cited a pair of forward groups: one emitting 390 terms took 7.15 s to
schedule while one emitting 321 took 0.77 s, offered as proof that cost follows the dense index
space rather than the emitted terms.

**That citation was an artifact and is withdrawn.** Those are *T2* term counts, and the work
being timed is the **T1** schedule the constructor builds first for the register-budget check.
Its term counts are 99 840 and 41 088 — a 2.4× ratio against a post-fix 3.2× time ratio (0.800 s
vs 0.250 s). There was never a 9× anomaly to explain; I compared a count from one schedule
against a build of another. [profile]

**The principle survives, re-costed.** Post-liveness-fix, profiling g13 shows enumeration is now
the dominant term: `_offset` 0.62 s and `_path_assignments` 0.51 s of 2.06 s, and `nonzero_masks`
— which walks the whole index space once, before `build_schedule` walks it again — is 0.84 s,
**41 %**. Fusing the two passes so masks are applied during enumeration would save up to that
fraction of schedule construction. [profile]

**Correcting the figure I reported.** I told review the change was worth "~3 %". That was 7.7 s
of a 264 s total whose other 97 % has since been removed; against the correct denominator it is
**~41 % of schedule time, ≈ 28 s of dbwd's 68 s**. The backlog ruling was made on the wrong
number and should be re-taken on this one — though at 28 seconds absolute, "don't gold-plate"
still looks like the right answer, which is why this is a correction and not an appeal.


## 2026-08-07 — D34: the conda env is renamed `spir` -> `zippel`, reversing D11.

D11 declined this rename on three grounds: the name came from an explicit instruction, renaming
is slow, and it "risks breaking the working torch 2.13 / CuTe DSL stack". The first is superseded
by a later instruction from the same reviewer. The third was a real risk and was answered by
measurement rather than by assertion.

**Clone, not rename.** The env was created with `conda create -p <prefix>`, and a prefix env bakes
its absolute path into shebangs and activation scripts under `bin/`. `mv` leaves `pip`, `f2py` and
every console-script entry point pointing at a path that no longer exists — broken in a way that
surfaces later and confusingly. `tools/switch_env.sh` does clone → verify → switch → cleanup, and
the old env is not removed until the new one has run the full suite.

Verification, all four checks green:

| check | result |
|---|---|
| version-for-version | identical: py3.13.14, torch 2.13.0, cutlass-dsl 4.5.2, fairchem 2.11.0, e3nn 0.6.0, ase 3.29.0, vesin 0.6.1, cuequivariance 0.11.0 |
| shebang audit | **0** scripts still pointing at the old prefix — the clone rewrote them, so it is a real env and not a cosmetic copy |
| CuTe DSL compile-and-run | max abs err **0.000e+00** on GH200 — precisely D11's stated risk |
| full pytest suite in the new env | green |

**Numbers predating the rename are not invalidated and are not re-run.** The environments are
byte-identical by construction and by the version check; only the prefix differs. REPORT's
environment row names the current env and says so.

`PLAN.md` is *not* rewritten: it records the env as it was created, with the rename noted inline.
Editing a plan to match what later happened destroys the record of what was planned.

Two bugs in the verification script itself, both worth noting because both were failures of the
checker rather than the thing checked:

1. `grep -rl ... | wc -l` under `set -euo pipefail` — grep exits 1 when it finds nothing, so the
   script died **precisely when the audit passed**. A verification that fails on success.
2. The CuTe DSL smoke test was a heredoc piped to `python -`, which fails with "DSL does not
   support REPL mode, save the function to a file instead" — the same constraint that already
   forced `codegen/emit.py` to write generated kernels to disk. Learned twice; it is now
   `tools/_env_smoke.py`, a real file.

## 2026-08-07 — D35: compile time is quadratic in group width, so fusion width needs a cap.

The cost ledger (D-directive, now populated) gives the first real compile-time data, and it is
the opposite shape from schedule construction. **[profile/measurement]** — 12 forward groups,
FP64, `cute.compile` wall-clock:

    compile_s = 1.63e-5 * terms^1.97      R^2 0.903

**[correlation + extrapolation 4.5x beyond the largest measured point]** — refit to
`terms^1.87`, R^2 0.942 once the point was measured; the prediction it carried was 2.1x high
(D37). Against schedule construction's `terms^0.97-1.01` (D31). Whole-forward compile is **477.5 s for
47 groups**; extrapolated, the one skipped group at 23 040 terms is **~109 minutes on its own —
13.7x the other 47 combined.**

**This is a direct constraint on the fusion pass, and it opposes D23.** D23 says fuse harder so
the dense store is never made; a wider group has more terms; terms cost quadratically to compile.
The `cat_83 + rotin_84` group is the concrete case: fusing them elides one `[E,9,256]`
intermediate — 2.28 GiB at si_medium, exactly the store D23 exists to remove — and costs ~109
minutes of compile to do it. Splitting them pays the store and compiles in ~2 minutes.

So **fusion width is not free**, and the S2 grouping search has to weigh a byte saving against a
compile cost rather than maximising fusion. A width cap belongs in the selection rule; where it
sits is a measurement, not a guess, and it is the first thing S2's search should establish.

**Stated uncertainty.** The fit mixes templates and one point deviates badly: `scatter_100` (T3,
1 152 terms) took 72.1 s against 17.8 s predicted, 4x. Since `rotin` is *also* T3, the 109-minute
figure could be a substantial under-estimate. It is left running to convert the extrapolation
into a measurement, because that is cheap and the number matters for S3 — where dbwd has 320
groups and the same quadratic applies.

Consequence for S3 (D30's whole-program criterion): D30 measured *schedule* time and set an entry
criterion on it, and I made the constructor linear on that basis. The ledger now says compile
dominates schedule by **24x** (477.5 s vs 19.5 s) at forward scale. D30's criterion was aimed at
the smaller of the two costs. It stands as written, but the S3 entry criterion that actually
binds is compile time, and D30 should be read alongside this.

## 2026-08-07 — D36: fusing a gather into a channel-heavy op inverts its template. The cap fixes a symptom of that.

D35 framed wide groups as a compile-cost problem. Measuring the cap showed something sharper.

    cap        groups   total emitted terms   largest group        predicted compile
    none           48                36,832   23,040  (T3)                114.4 min
    10,000         55                16,177    5,123  (T2)                  8.8 min

Splitting **reduces total emitted terms by 56 %**. A pure compile-cost tradeoff would have left
the term count alone and only changed how it was distributed. It does not, and the reason is the
selection rule: *any* index map routes a group to T3, and T3 unrolls every trailing axis. So
fusing the gather `cat_83` into `rotin_84` drags a `[E,9,256]` op out of T2 -- where 256 channels
are parallel across threads -- into T3, where they are unrolled into 23 040 straight-line terms.

**The fusion does not merely cost compile time; it forces a worse template.** That is a real
defect in the selection rule as written (docs/templates.md 2): routing on "contains an index map"
is correct about *why* T1/T2 cannot apply to the gather itself, and wrong to extend that verdict
to every op fused alongside it.

**For S1c: use `max_volume=10_000`.** It is the right decision for the wrong-ish reason -- it
splits the group because the group is wide, not because the template inverted -- but it produces
the correct partition here, and 8.8 minutes of whole-forward compile against 114 makes S1c
measurable today.

**For S2: the principled fix is a channel-parallel T3**, so a gather or scatter can keep the
channel axis on threads instead of unrolling it. That is exactly the "full reduction-class
generality" S2 already owns, and this is the concrete argument for why it is needed rather than
merely tidy. Recorded now so S2 inherits the reason, not just the task.

Caveat on the cap's metric: it caps *index-space volume*, which tracks emitted terms for T1 and
T3 but overestimates for T2 by the channel extent, since T2's channel axis is symbolic. So the
cap is conservative for T2 groups and cannot shrink a single op that is already wide --
`conv1_90` alone is 655 744 volume and no cap splits it. Volume is a proxy, and where it is a bad
one is written down rather than discovered later.

## 2026-08-07 — D37: the rotin measurement lands; D35's extrapolation overpredicted 2.1x.

D35 left the widest forward group running specifically to convert its extrapolation into a
measurement. It has.

| | terms | compile | correctness |
|---|---|---|---|
| predicted (D35, `terms^1.97`) | 23 040 | 6 563 s = 109 min | — |
| **measured** | 23 040 | **3 085 s = 51.4 min** | **0.00e+00** — bit-exact |

**Overpredicted by 2.1x.** Refitting with the measured point included:

    compile_s = 3.00e-5 * terms^1.87      R^2 0.942   (was 1.63e-5 * terms^1.97, R^2 0.903)

The exponent drops from 1.97 to 1.87 and the fit improves, which is the expected shape: an
extrapolation 4.5x beyond the largest measured point was leaning on the tail of a fit whose
R^2 was 0.903. **[measurement, superseding extrapolation]**

**What does not change.** Compile is still strongly superlinear and still dominates: 51 minutes
for one group against 8 minutes for the other 47 combined under the D36 cap. D35's decision — cap
fusion width — and D36's finding — the fusion inverts the template — both stand on the measured
number as firmly as on the extrapolated one. The cap buys roughly 6x rather than 13x, and 6x is
still the difference between a measurable S1c and an unmeasurable one.

**What this says about my own numbers.** I reported 109 minutes to review as a prediction and
labelled it as one, and it was wrong by a factor of two. The lesson is not that the extrapolation
was illegitimate — it was the only number available and it drove a decision that survives
measurement — but that a fit extrapolated 4.5x beyond its data earns a wide error bar, and I gave
it none. Where a prediction drives a decision that can wait for a measurement, the measurement
should be taken; that is why this one was left running, and it is the practice to keep.

Also worth recording: `rotin` at 23 040 terms is **bit-exact**, 0.00e+00 against the FP64
interpreter, which is the largest single kernel this compiler has produced.

## 2026-08-07 — D38: the si_medium OOM was host RAM, not device memory. Oracle runs move to compute nodes.

`bench/traffic_calibrate.py` and REPORT §8.5c both recorded that "the full FP64 forward at
si_medium materialises ~40 GiB and was OOM-killed", and I let that stand as though it were a GPU
limit. It is not: `zippel/interp.py` sets **`DEVICE = "cpu"`**, so the FP64 oracle is
CPU-resident and the kill was the host OOM killer. Surfaced when the S1c composition compared a
CUDA tensor against a CPU one.

Two consequences, and the second matters more.

1. The attribution is corrected wherever it appears. A ~40 GiB host allocation on a login node
   shared with other users is a different fact from a 40 GiB GPU allocation on a 96 GiB card —
   the first is antisocial and the second would have been fine.
2. **Big-fixture oracle comparisons run on compute nodes from now on**, not on the login node.
   The oracle is the one part of this pipeline that is deliberately unoptimised, FP64 and
   host-resident, and si_medium is 27x si_small. Running it under `sbatch` costs nothing we care
   about and stops a verification step from being the most disruptive thing this project does to
   a shared machine.

Evidence class of the original claim: **[correlation]** — a job died, a large number was
available, and I joined them. Nothing was measured about *which* memory ran out. Exactly the
pattern D32 exists to catch, one week and three instances later.

## 2026-08-07 — D39: the S1c wall-clock deficit is occupancy starvation at si_small, and it makes a falsifiable prediction.

The first S1c number is 0.110x — the fused forward is 9.1x slower than eager at si_small f32,
while using 4x fewer launches and 1.39x less peak memory. Three axes disagreeing that sharply
is a diagnosis waiting to be made, and it is available without a profiler. **[analysis, not
measurement — the prediction below is what converts it]**

A GH200 has 132 SMs x 2048 threads = 270 336 thread slots. The emitted kernels' thread counts:

    template  kernels   threads (si_small)   occupancy
    T1             22                9,576        3.5 %
    T3              9                9,576        3.5 %   (two at 216 threads = 0.08 %)
    T2             24            1,225,728       453 %    (saturated)

**31 of 55 kernels leave 96.5 % of the GPU idle**, and two of them occupy less than a single SM.
T1 maps one thread per segment element, which is the premise that lets it hold every trailing
value in registers -- correct, and at 9 576 edges it asks a 270 000-slot machine to run 9 576
threads. si_small is **28x too small** to fill this GPU under that mapping.

**Falsifiable prediction, before the sweep reports it.** si_medium has 259 474 edges = **96 % of
the machine's thread slots** under the same mapping. If occupancy starvation is the dominant
term, the si_medium speedup must improve by close to an order of magnitude relative to si_small.
If it does not, this diagnosis is wrong and the cost is elsewhere -- inlined loads, launch
latency across 55 sequential kernels, or the unvectorised scalar bodies.

Recorded before the answer arrives, because a diagnosis offered after the fact explains anything.

Consequences either way:

* If confirmed, T1's thread mapping is the S1 performance variant to write first -- multiple
  segment elements per thread, or the trailing axes distributed where they are parallel -- and
  it costs nothing in correctness because the schedule is unchanged.
* The two 216-thread kernels are node-rooted reductions and are simply too small to launch
  separately at any fixture size; they belong fused into a neighbour or run on a single block.
* None of this touches the memory axis, which is already 1.39x favourable on the pass with the
  least to gain.

## 2026-08-07 — D40: the Gate 0 contamination repeated, in a script that never inherited the fix.

Gate 0 lost a measurement to two runners writing the same result files: an exclusive-node sbatch
job and a login-node run I had started in parallel (REPORT §8.7). The fix was provenance fields
plus an exclusive `flock` in `bench/run_all.sh`, and it was recorded as fixed "at the cause".

It was not fixed at the cause. It was fixed in **one script**.

`bench/s1c_local.sh`, written today, has no lock. A sweep launched while the session was on
`daint-ln001` survived a teardown; the session later moved into a compute-node allocation
(`nid005562`); I relaunched there; and both wrote `bench/results/s1c_local_rep*_*.json` on the
same shared filesystem. Three files had already landed carrying `host=daint-ln001`, written
minutes earlier, while the "current" sweep was on a different machine entirely.

Caught by the provenance field — the *other* half of the Gate 0 fix, which did generalise
because it lives in the result rather than in a script. Without `host` in the JSON the numbers
would have looked ordinary.

Actions taken:

* the three login-node results are quarantined under `bench/results/quarantine/`, not deleted —
  they are real measurements of a contended host and might be worth something later, but they
  are not what the sweep was measuring;
* the login-node processes were killed over `ssh` (they were unreachable from the allocation, so
  "kill the stray job" needed the other host);
* `s1c_local.sh` gains the `flock` that `run_all.sh` has, and result filenames now carry the
  hostname. A lock prevents concurrency; a host-qualified path prevents *confusion*, and the two
  failures here were one of each.

**The lesson is about the shape of the original fix, not about locking.** "Fixed at the cause"
meant adding a lock to the script that failed. Every later script started without one, and the
guard did not exist anywhere it could be inherited from. A fix that lives in one call site is a
patch; the provenance field survived because it was attached to the artifact instead.

## 2026-08-07 — D41: S1c measured under a deliberate protocol deviation; S1 wall-clock NOT met.

**Result.** Fused forward vs eager at an identical boundary: wall-clock **0.036–0.107×**, peak
memory **1.39–1.42×**, launches 55 vs 224–263. The wall-clock criterion for S1 is **not met** and
S1 stays open.

**Deviation, and why it is defensible.** Measured on a single-node `salloc` (`nid005562`), n = 2–4
per configuration, not the pinned N = 5 independent allocations. The pinned protocol exists to
capture between-node and placement variance, which a single allocation cannot see. It is justified
here by proportion: the measured within-node spread is **0.04–0.65 %** against an effect of
**10–28×**. No plausible placement variance closes a 28x gap, and reps 3–5 would have cost ~1.5 h
of compile to refine a number already stable to half a percent. They were cancelled as
zero-information.

**The deviation does not generalise.** Any verdict-class table — anything that decides the bet —
returns to N = 5 independent allocations. The rule is that a single-node measurement is adequate
only when the effect exceeds the unmeasured variance by orders of magnitude, which is exactly the
condition that will *not* hold near a 1.0x crossover.

**The hill, so the number is not read as a ratio in a vacuum.** Eager's full conservative training
step at si_medium fp32 is 311.63 ms. Our fused *forward alone* is 1401.9 ms. Before three-pass
fusion competes at all, the forward must drop below eager's entire fwd+bwd+dbwd: **≈ 4.5×**.

**What remains true.** The memory ratio holds at 1.39–1.42x across a 27x size range and both
dtypes. That is the axis the bet rides on, measured on the pass with the least to gain, and it is
the one piece of the S1c result that is already favourable.

## 2026-08-07 — D42: the S1c deficit is uncoalesced weight access in T2. Three hypotheses refuted, one quantitatively consistent.

Per-kernel breakdown, si_medium f32, 55 kernels, 1392 ms total. **[measurement]**

    rank  grp  tmpl        ms   share    cum   instr/thr   issue-bound     meas/bound  ops
       1   43    T2   714.803   51.3%  51.3%      10,246   10.2-40.7 ms         70.3x  conv1_90
       2   46    T2   344.688   24.8%  76.1%       5,126    5.1-20.4            67.7x  conv2_95
       3   39    T2   190.453   13.7%  89.8%       1,537    7.6-30.5            25.0x  conv1_m0_86

    template  kernels        ms   share
          T1       22      3.20    0.2%
          T2       24   1376.62   98.9%
          T3        9     12.51    0.9%

**Three kernels are 89.8 % of the forward, all of them the SO(2) convolution, all T2.**

**D39 is refuted a second time and more sharply.** I diagnosed occupancy starvation, predicted
si_medium would improve ~10x, and it got worse. Now the breakdown shows the T1 kernels — the ones
at 3.5 % occupancy that I identified as the problem — are **0.2 % of runtime**. Making them
infinitely fast changes nothing. The cost is entirely in T2, the template that is *saturated* at
453 % occupancy. I was not merely wrong about the magnitude; I was pointing at the wrong kernels.

**Four hypotheses, three dead:**

| hypothesis | test | verdict |
|---|---|---|
| occupancy starvation | T1 (3.5 % occupied) is 0.2 % of runtime | **refuted** |
| issue-bound | measured/issue-floor is 18–70x | **refuted** |
| register spilling | ~113 live scalars per thread against 255 registers; T2 has no budget check, but does not need one here | **refuted** |
| uncoalesced access | see below | **consistent** |

**The mechanism. [static analysis, quantitatively consistent with measurement]** `c1_w1a` has
type `none[j:2, o:128, k:2, c:256]`, and T2 puts the thread index on `o`. Row-major, consecutive
threads therefore read addresses **512 elements = 2 048 B apart**, so all 32 lanes of a warp touch
32 distinct 128-byte lines on every load. Arithmetic:

    33,212,672 threads x 10,246 instr, half of them loads
      -> 5.32e9 warp-level load instructions
      -> 20 ms at one coalesced load per cycle per SM
      -> x32 transactions when uncoalesced = 651 ms
    measured: 714.8 ms

**651 against 715 — 9 % apart, with no fitted parameter.** None of the other three hypotheses
lands within an order of magnitude. This is not proof; it is the only surviving explanation that
predicts the number, and it is stated as such.

**Intervention (c), re-aimed by review and now targeted by data:** block-cooperative restructure
of `conv1_90` — stage the shared weight tile in smem with a *coalesced* cooperative load, then
have each thread read its slice from smem. `c1_w2a`/`c1_w2b` are 128 KiB and fit the ~228 KiB/SM
budget outright; `c1_w1a`/`c1_w1b` are 512 KiB and need tiling over `k`. Predicted effect is
bounded below by removing the 32x transaction amplification on the weight reads; no point
estimate is offered, and the measurement decides.

**A gap found while testing the spilling hypothesis, recorded though it is not the cause here:**
`emit_tile_source` (T2) has **no register-budget precondition at all**. D26 requires estimators
that gate routing to be upper bounds or carry a falsification test, and T1's budget check is
exactly that — but T2 was written without one, so a T2 group that *did* spill would do so
silently. It does not spill today; that is luck, not a guard.
