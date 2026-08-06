# DECISIONS

One dated line per deviation from the M1 work order: what changed, why.

## 2026-08-06 — Phase 0 setup

- **D1. Repo root moved `/capstor/scratch/cscs/dlu/iclr/spir-m1` → `/iopsstor/scratch/cscs/dlu/iclr/spir-m1`.**
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
