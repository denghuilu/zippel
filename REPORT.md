# M1 Vertical Slice — Report

**The bet under test.** The energy–force–double-backward computation of an equivariant MLIP can
be expressed in a differentiation-closed, verification-friendly segmented-polynomial IR (SP-IR),
and three-pass (fwd/bwd/dbwd) joint compilation on that IR stably outperforms existing
per-operator stacks (eager / torch.compile / cuEquivariance + autograd) on conservative training —
wall-clock and peak memory.

**Status: Gate 0 and Gate 1 both PASSED.** Phase 2 (CuTe DSL fused kernels) is next. Every number
below is measured; nothing is projected, and no baseline is reported at a setting that cripples
it.

The IR is called the **segmented-polynomial IR** throughout; the package is `zippel`. "SPIR"
appears only where this document quotes the original work order.

Benchmark numbers come from **exclusive `sbatch` allocations**, never the shared login node;
every measurement records `host`, `slurm_job` and `exclusive`. Units are **GiB** (2³⁰) throughout.
The small fixtures are host-dispatch-bound and are reported as a range rather than a number
(§5); si_medium is the primary fixture.

Two of the three per-operator baselines resolved as *structural findings* rather than numbers, and
both were traced to a specific cause rather than left as "it didn't work":
* **torch.compile cannot run the measured unit at all** — AOTAutograd rejects the double backward;
  a backend ladder locates the failure precisely (§5, B2).
* **cuEquivariance has no backend that is both correct and scalable** for eSEN's shared weights —
  and the correctness half was triaged to an **upstream bug**, with a hand oracle showing
  `fused_tp` and `naive` agree while `indexed_linear` does not (§5, B3). Draft issues for both
  fairchem and cuEquivariance are written and deliberately **not filed**, pending review.

---

## 1. Environment and versions

| item | value |
|---|---|
| Node | `daint-ln001` (CSCS Alps login node), 4× **NVIDIA GH200 120GB**, all idle |
| GPU | sm_90a, compute capability 9.0, 97 871 MiB (95.6 GiB) each |
| Driver / CUDA | 590.48.01 / CUDA 13.1 (driver); torch built against CUDA 13.0 |
| OS / arch | SLES 15 SP6, `aarch64` (Neoverse-V2), 288 cores, 856 GB RAM |
| `nvcc` | **absent** and not required — CuTe DSL compiles through NVRTC/NVVM |
| Env | conda `zippel` at `/iopsstor/scratch/cscs/dlu/envs/zippel`, Python **3.13.14** (cloned from `spir` and renamed, D34; results predating the rename were produced in the byte-identical `spir` env) |
| torch | **2.13.0+cu130** | 
| nvidia-cutlass-dsl | **4.5.2** (libs-base 4.5.2) |
| fairchem-core | **2.11.0** (e3nn 0.6.0, ase 3.29.0) |
| cuequivariance / -torch / -ops-torch-cu13 | 0.11.0 / 0.11.0 / 0.11.0 |
| vesin | 0.6.1 |
| pytest | 9.1.0 |
| SLURM | account `lp16`, partition `normal`, `--gres=gpu:4`, 24 h cap |

Full pinning in `requirements.lock` (170 packages).

**CuTe DSL smoke test — PASS.** A minimal `axpy` written with `@cute.kernel` / `@cute.jit` and
`from_dlpack` compiled and ran on GH200: `N = 1048576`, max abs err **9.537e-07** vs the torch
reference. This was run before any other GPU work, per the work order.

### Operational constraints discovered

- **Cold imports from `/iopsstor` are very slow**: `import torch` took **186 s** and
  `import cutlass` **38 s** on first access (Lustre, cold page cache). Warm imports are normal.
  This affects interactive iteration only — benchmark numbers are CUDA-event timed after warmup,
  so they are unaffected. Long-running commands must be backgrounded.
- `requirements.lock` records `triton` as a local-file wheel reference inherited from the cloned
  env; a clean-clone reinstall needs that wheel or an equivalent `triton==3.7.1` aarch64 build.

### Repo and cache locations

The work order specifies `/capstor/scratch/cscs/dlu/iclr/spir-m1`. The repo lives at
**`/iopsstor/scratch/cscs/dlu/iclr/zippel`**, which is two separate changes, both approved at Gate
0 review and recorded separately:

1. **Filesystem** `/capstor` → `/iopsstor` (DECISIONS.md D1), because `/capstor/scratch/cscs/dlu`
   is at **306.9 % of its 1 000 000-inode quota with the grace period expired**, so file
   *creation* there fails with `EDQUOT` (reproduced with `touch`). Reads are unaffected, so the
   conda base, FlashSO2 and fairchem remain readable in place.
2. **Directory name** `spir-m1` → `zippel` (DECISIONS.md D14), to match the git remote
   `git@github.com:denghuilu/zippel.git`. The environment-variable prefix moved with it
   (`ZIPPEL_ROOT`, `ZIPPEL_ENV`, `ZIPPEL_CACHE_ROOT`). The conda environment keeps its original
   name, `spir`.

**Every compiler/JIT cache is project-local under that repo, on `/iopsstor`** — verified by
`tests/test_environment.py`, which fails if any resolves off `/iopsstor`, onto `/capstor`, or onto
node-local storage:

| variable | path |
|---|---|
| `CUTE_DSL_CACHE_DIR` | `$REPO/.jit-cache/cute_dsl` |
| `TRITON_CACHE_DIR` | `$REPO/.jit-cache/triton` |
| `TORCHINDUCTOR_CACHE_DIR` | `$REPO/.jit-cache/inductor` |
| `QUACK_CACHE_DIR` | `$REPO/.jit-cache/quack` |
| `PYTHONPYCACHEPREFIX`, `TMPDIR` | `$REPO/.jit-cache/{pycache,tmp}` |

Two corrections were needed to reach that state: `TORCHINDUCTOR_CACHE_DIR` was **unset** (so
inductor was caching to node-local `/tmp`, which Alps purges), and `TRITON_CACHE_DIR` was
inheriting `~/.bashrc`'s *shared* `/iopsstor/scratch/cscs/dlu/.cache/triton` — on the right
filesystem, but not self-contained, so "reproducible from a clean clone" would have depended on
another project's cache. `tests/conftest.py` now sets these **unconditionally** rather than via
`setdefault`, with `ZIPPEL_CACHE_ROOT` as the deliberate override.

**One cache is deliberately not redirected:** `CUDA_CACHE_PATH`, which Alps points at
`/dev/shm/dlu/cuda_cache` (node-local RAM). It holds driver-level PTX→SASS JIT results, which our
kernels do not trigger — CuTe DSL emits `sm_90a` cubin through NVRTC directly. Moving it onto
currently-degraded Lustre would cost real time for no reproducibility gain. Flagged rather than
changed.

### `blocks/Jd.pt` provenance

| field | value |
|---|---|
| Content | list of 12 FP64 tensors, shapes (1,1), (3,3), (5,5), … (23,23) — the `J` change-of-basis matrices for `l = 0..11` used by `wigner_D` |
| Source | `fairchem-core` **2.11.0**, `src/fairchem/core/models/uma/Jd.pt` |
| Copied from | `.../envs/iclr/lib/python3.13/site-packages/fairchem/core/models/uma/Jd.pt` |
| SHA256 | `b4059c45be246dcb6c49c545670b65c56550eb0c2e7a9c92b4b50a92d370dbe2` (verified byte-identical to upstream) |
| License | MIT (fairchem-core, Meta Platforms) |
| Upstream lineage | fairchem's `wigner_D` is the **e3nn 0.4.0** recipe (`Xa @ J @ Xb @ J @ Xc`), taken deliberately because e3nn 0.5.0 switched to `torch.matrix_exp`, which is much slower |

It is committed rather than regenerated so the reference has a fixed, hashable basis; the same
tensors are what fairchem itself uses, which is why §3's cross-check is meaningful.

---

## 2. Block config and fixtures

**Config — smallest published eSEN (K4L2 / eSEN-sm).** Source:
`fairchem_core-2.0.0` → `configs/puma/training_release/backbone/K4L2.yaml`; cutoff and
`max_neighbors` from the OMol25 eSEN-sm top-level config; cross-checked against
[arXiv:2502.12147](https://arxiv.org/abs/2502.12147) App. A.1 ("Lmax=2, Mmax=2", 6 Å).

| parameter | value | parameter | value |
|---|---|---|---|
| `lmax` / `mmax` | 2 / 2 | `cutoff` | 6.0 Å |
| `sphere_channels` | 128 | `num_distance_basis` | 64 (gaussian) |
| `hidden_channels` | 128 | `edge_channels` | 128 |
| `edge_channels_list` | [320, 128, 128] | `act_type` | gate |
| num_coeffs | 9 | `num_layers` (full model) | 4 |

Derived and pinned in `tests/test_ref_vs_fairchem.py::test_k4l2_derived_shapes_are_as_published`:
`num_channels_rad = 1536`, `edge_split = [768, 512, 256]`, `m_split = (3, 4, 2)`,
`extra_m0_output_channels = 256`. Block parameter count 1 417 217.

**The `esen/` package could not be installed**: it exists only in `fairchem-core 2.0.0`, which
declares `Requires-Python >=3.9,<3.13`. Its maintained descendant — fairchem 2.11's UMA
`SO2_Convolution` / `Edgewise` — is installed and is used as the cross-check target, instantiated
at exactly the K4L2 hyperparameters above (DECISIONS.md D3). Known algebraic deltas UMA vs eSEN:
`to_m` hoisted out of `SO2_Convolution` into the Wigner matrix; `SO2_m_Conv` returns
`(real, imag)` rather than a `cat`; envelope always applied; `GateActivation(m_prime=True)`.

**Fixtures** (`fixtures/*.npz`, fixed seeds, perturbed by 0.05 Å gaussian, PBC, full ragged
neighbour list at 6.0 Å — the config's `max_neighbors=30` cap is deliberately not applied,
DECISIONS.md D6):

| fixture | atoms | edges | avg degree |
|---|---|---|---|
| `si_small` | 216 | 9 576 | 44.33 |
| **`si_medium`** (primary) | **5 832** | **259 474** | **44.49** |
| `si_large` | 46 656 | 2 075 158 | 44.48 |
| `cu_small` | 256 | 19 966 | 77.99 |
| `cu_medium` | 5 324 | 415 272 | 78.00 |
| `cu_large` | 48 668 | 3 796 130 | 78.00 |

No self-loops and nothing beyond cutoff, asserted at generation: a zero-length edge would make the
edge frame undefined (both 1/|v| and sin β collapse) and silently break equivariance and the
finite-difference checks.

**Schema and integrity.** `make_fixtures.py` is the only writer, `fixtures/load.py` the only
reader; each `.npz` carries a `schema_version` and the loader rejects a mismatch loudly rather
than misreading it. Edges are sorted into a canonical order (src, dst, shift) so the content hash
does not depend on whether vesin or ASE produced the list. `fixtures/manifest.json` pins two
hashes per fixture: `sha256_content` over the array bytes in fixed key order (reproducible
anywhere — this is what `bench/run_all.sh` verifies and what the round-trip test asserts) and
`sha256_file` over the `.npz` bytes (not reproducible across runs, since `np.savez_compressed`
embeds zip timestamps; kept for artifact integrity only).

`x_node` and the Wigner roll `gamma` are regenerated from the stored seed at load time rather than
stored, so every implementation provably receives bit-identical inputs while the large fixture
stays a few MB instead of ~430 MB.

These fixtures were **regenerated** at Gate 0 review under the canonical schema, and the Gate 0
baseline table was re-measured on them; bit-exact reproduction of the previous files was not
available (DECISIONS.md D11).

---

## 3. Reference validation

`blocks/eso2_ref.py` is a standalone, torch-only, FP64-capable implementation of the block:
gather → Wigner rotation fused with the l→m′ reordering → conv1 (radial-modulated, emits gate
scalars) → gate → conv2 → polynomial envelope → rotate back → scatter-add → per-atom readout.

### The measured unit, exactly

Everything timed in this report is one **conservative training step**. Written out, with
`h ∈ ℝ^{N×9×128}` the scattered per-node output in l-major order:

```
node invariants   z_i = [ h_i[0, :] ‖ Σ_{m} h_i[1..3, :]²  ‖ Σ_{m} h_i[4..8, :]² ]   ∈ ℝ^{384}
                        └ l=0 (already invariant)  └ ‖l=1‖²      └ ‖l=2‖²
readout           E   = Σ_i  W₂ · SiLU(W₁ z_i + b₁) + b₂          W₁: 384→128, W₂: 128→1
forces            F   = −∂E/∂pos                                   (create_graph=True)
loss              L   = w_E · mean((E − E_ref)²) + w_F · mean((F − F_ref)²),  w_E = w_F = 1
backward          L.backward()   →  grads for all 23 parameter tensors
```

The `Σ_m h[l]²` terms are what make E rotation-invariant: each Wigner block is orthogonal, so a
per-`l` squared norm is preserved exactly. **This is a deliberate departure from eSEN**, which
reads `h[0, :]` linearly — correct after a *stack* of blocks, but degenerate on one block, where
the `l = 0` row is fed only by the m = 0 branch and all four m > 0 convolution weight tensors get
exactly zero gradient (§4.3, DECISIONS.md D9). Since the readout defines what the benchmark
measures, it is stated here in full and guarded by
`tests/test_ref_block.py::test_every_parameter_has_nonzero_grad_norm_on_real_fixtures`, which
asserts a strictly positive gradient norm for each of the 23 parameter tensors on real fixtures.

`E_ref` and `F_ref` are zeros: M1 measures the *cost* of the step, not model quality, and zero
targets keep the loss well-defined without inventing physics.

**Against fairchem 2.11 UMA, max relative error** (work order tolerances: FP32 ≤ 1e-5,
FP64 ≤ 1e-10):

| component | FP64 | FP32 |
|---|---|---|
| `SO2_Convolution` conv1 (output) | **0.00e+00** | **0.00e+00** |
| `SO2_Convolution` conv1 (gate scalars) | **0.00e+00** | **0.00e+00** |
| `SO2_Convolution` conv2 | **0.00e+00** | **0.00e+00** |
| `GateActivation` | **0.00e+00** | **0.00e+00** |
| Wigner-D (rational form vs fairchem) | 8.26e-15 | 1.51e-06 |

The convolution and gate agree **bit-exactly** because the reference performs the same operations
in the same order. The Wigner-D row is not bit-exact by construction: it is computed by an
algebraically different (rational) route — see §4.

The `l→m′` permutation and per-m sizes are asserted equal to fairchem's `CoefficientMapping.to_m`
and `GateActivation.expand_index` exactly.

---

## 4. Three findings that change how the block must be written

### 4.1 fairchem's shipped rotation has a silently wrong double backward

`Safeacos.forward` calls `ctx.save_for_backward(x.clamp(...))`. The clamp is evaluated under
no-grad, so the saved tensor carries no graph and the first derivative's dependence on `x` is
invisible to a second differentiation. Measured on GH200 in FP64 against `uᵀW(pos)v` — a genuinely
direction-dependent scalar (note `‖Wv‖²` and `Σ W²` are both rotation-invariant and give a vacuous
~1e-15 result, which is easy to mistake for agreement):

| quantity | fairchem `Safeacos` | double-differentiable `acos` | agreement |
|---|---|---|---|
| E | 23.47743315 | 23.47743315 | exact |
| ‖F‖ | 20.71385 | 20.71385 | 8.9e-16 |
| ‖∂²‖ | **3103.208** | **3091.557** | **5.5 % relative — wrong** |

**Localization.** The defect is a *graph-structure* bug, not a numerical edge case, so it is not
confined to the clamp band: `x_clamped` is computed inside `forward`, where no graph is recorded,
so the first derivative is a constant with respect to `x` **everywhere**, whether or not the clamp
is active. `bench/safeacos_localization.py` measures this against the analytic
`d²/dx² acos(x) = −x(1−x²)^(−3/2)` across four bands (deep interior |x| ≤ 0.5, mid, near-edge, and
the clamp band |x| > 1−1e-7); results in `bench/results/safeacos_localization.json`.
`docs/upstream_fairchem_issue_DRAFT.md` holds the draft issue text — **not filed**, pending review,
with an explicit TODO list of what must be settled first (re-check against fairchem `main`,
duplicate search, whether to attach a `gradgradcheck` regression test as a PR).

No exception is raised. Energies and forces are correct; only the grad-of-grad is wrong — i.e.
exactly the quantity conservative *training* backpropagates through. A second configuration gave
7.1 % relative error, so the magnitude is input-dependent, not a fixed offset.

Consequences, all adopted: (a) fairchem autograd is **disqualified as the double-backward oracle**
— the FP64 reference plus finite differences take that role; (b) a double-differentiable `acos`
is used in every implementation so all parties compute identical math and the comparison stays
apples-to-apples (DECISIONS.md D5); (c) this is direct evidence for the *verification-friendly*
half of the bet — a hand-written backward was wrong for an unknown period, in shipped code, in a
way that only shows up at second order.

Separately, `Safeatan2.backward` is decorated `@torch.compiler.disable` — a guaranteed graph break,
recorded as evidence for the B2 torch.compile fallback inventory.

### 4.2 The Euler-angle path can be made rational, so the declared vocabulary stays closed

The work order's SP-IR transcendental set is {exp, sigmoid/SiLU, rsqrt, polynomial envelope}, which
does not contain the `acos`/`atan2`/`sin`/`cos` that Wigner-D construction appears to need. Per the
user's instruction to simplify `sin/cos ∘ acos/atan2` where trivial, write
`s2 = x̂² + ẑ²`; because the vector is normalised, `s2 == 1 − ŷ²` **exactly**, so `r = √s2` is both
`sin β` and the `atan2` radius and a single guard covers both degeneracies:

```
cos β    = ŷ                       cos kβ = T_k(ŷ)                    (polynomial)
sin β    = r = s2 · rsqrt(s2)      sin kβ = r · U_{k−1}(ŷ)            (polynomial × r)
cos α    = ẑ · rsqrt(s2)           cos kα, sin kα by de Moivre        (polynomial)
sin α    = x̂ · rsqrt(s2)
```

γ is a *random* roll and carries no position dependence, so `cos kγ` / `sin kγ` are per-edge
constants hoisted to inputs. Every Wigner-D entry is therefore a polynomial in
(x̂, ŷ, ẑ, rsqrt(s2)); the only non-polynomial primitive is `rsqrt`, **already in the declared
set**. The vocabulary is not widened and differentiation closure is preserved (DECISIONS.md D4).

Verified: matches fairchem's Wigner to **3.3e-15** (FP64, E = 4096) and is orthogonal to 1.3e-15.
And it repairs §4.1 by construction:

| implementation | rel. err. of F vs oracle | rel. err. of dbwd vs oracle |
|---|---|---|
| rational (ours) | 1.4e-15 | **5.1e-16** |
| fairchem `Safeacos` | 9.0e-17 | **7.1e-02** |

An independent central-difference check of ∂F/∂pos agrees to **9.7e-11**.

A fully angle-free Wigner recursion (direct from the unit vector, no Euler parameterisation) is
out of M1 scope and logged as future work.

### 4.3 A single-block energy head must not read only `l = 0`

eSEN's readout is linear on the `l = 0` row, applied after a *stack* of blocks. On one block that
is degenerate: the `l = 0` output row is fed only by the m = 0 branch, so **all four m > 0
convolution weight tensors receive exactly zero gradient** (`c1_m.{0,1}.weight`,
`c2_m.{0,1}.weight`) and the SO(2) machinery under test goes dead. The readout therefore takes
`l = 0` linearly plus the per-`l` squared norms `Σ_m x[l,m,c]²`, which are invariant because the
Wigner blocks are orthogonal. Every output row is live and E remains exactly rotation-invariant
(DECISIONS.md D9). `tests/test_ref_block.py::test_all_parameters_receive_gradient` guards it.

---

## 5. Gate 0 baseline table

### Measurement provenance

Every `Measurement` records `host`, `slurm_job` and `exclusive`. The numbers below come from
**job 4376140 on `nid005332`, exclusive, 35 min** — the definitive run that closes standing
thread (a). Units are **GiB** (2³⁰).

> **The small fixtures are not reproducible and are reported as a range, not a number.**
> Two *exclusive-node* runs of the identical configuration disagree by 80 %:
>
> | si_small fp32 | median | IQR | SM clock |
> |---|---|---|---|
> | job 4376123 | 29.79 ms | 0.23 | 1980 MHz |
> | job 4376140 | 53.67 ms | 15.96 | 1980 MHz |
>
> Clocks were pinned at maximum in both, so this is neither throttling nor another tenant.
> These steps are **host-dispatch-bound** — roughly 3 500 kernel launches in 30–55 ms, i.e.
> ~10 µs per launch — so host-side scheduling dominates and the GPU is mostly idle. The medium
> fixtures, which are device-bound, are stable to 0.3 % across the same two runs
> (si_medium fp32: 312.10 vs 312.31 ms). **si_medium is the primary fixture and the one any
> claim should rest on.** No attempt was made to tune the small-fixture numbers into looking
> better; the instability is itself evidence for the launch-bound diagnosis.
>
> **B3's timings come from a different interpreter** (torch 2.11.0, since cuEquivariance's ops
> extension will not load against 2.13), so B3 rows are not comparable to B1/B2 rows. B3 carries
> its own eager control in that interpreter; only the ratio is meaningful. See §5, B3.

**B1 — eager PyTorch**, conservative training step, median of 100 iters (30 at the 50 k
fixtures), job 4376140:

| fixture | atoms | edges | fp32 ms | tf32 ms | bf16 ms | fp32 peak GiB | bf16 peak GiB |
|---|---|---|---|---|---|---|---|
| si_small *(unstable)* | 216 | 9 576 | 53.67 | 44.92 | 57.44 | 1.48 | 0.85 |
| cu_small *(unstable)* | 256 | 19 966 | 56.68 | 66.89 | 62.50 | 3.00 | 1.68 |
| **si_medium** | **5 832** | **259 474** | **312.31** | **217.66** | **191.96** | **38.13** | **20.91** |
| cu_medium | 5 324 | 415 272 | 499.07 | 347.17 | 305.04 | 60.90 | 33.35 |
| si_large | 46 656 | 2 075 158 | **OOM** | OOM | OOM | — | — |
| cu_large | 48 668 | 3 796 130 | **OOM** | OOM | OOM | — | — |

IQR ≤ 2.28 ms on every medium row (≤ 1.0 %); 9.9–18.5 ms on the small rows (16–39 %).

### Variance diagnosis: why si_small is unreproducible

Two exclusive allocations of the identical configuration differed by 80 % on si_small fp32
(29.79 ms IQR 0.23 on `nid006330`, job 4376123; 53.67 ms IQR 15.96 on `nid005332`, job 4376140).
Provenance from both runs rules out the obvious explanations and points at one:

| | job 4376123 | job 4376140 |
|---|---|---|
| node | nid006330 | nid005332 |
| GPU / driver | GH200 120GB, 590.48.01 | identical |
| max SM clock / power cap | 1980 MHz / 900 W | identical |
| allocated CPUs, exclusivity | 288, exclusive | identical |
| CPU/NUMA pinning | **none** | **none** |

**The clock–variance correlation is inverted, and that is the diagnostic.** Per-measurement NVML
clocks in job 4376140:

| fixture | SM clock at measurement | IQR |
|---|---|---|
| si_small / cu_small | **1980 MHz — maximum boost** | 16–39 % |
| si_medium / cu_medium | 1785–1965 MHz | 0.2–1.0 % |

The *noisy* configurations run at full boost; the *stable* ones downclock. A GPU that is
genuinely working draws power and drops off max boost — so sitting pinned at 1980 MHz while
producing 30 % variance says the GPU is idle, waiting. The medium fixtures, which actually load
the device, are 100× tighter.

**Dominant source: unpinned CPU/NUMA placement relative to the driving GPU — confirmed by
intervention.** A GH200 node presents 4 GPUs across 4 NUMA domains. Neither original run pinned
the process, so it could land on a socket that is not local to GPU 0; for a workload issuing
~3 500 launches in 30–55 ms (~10 µs each) every launch then crosses the interconnect.

The N = 5 protocol run (array 4377448, five *different* nodes, `numactl --cpunodebind=0
--membind=0`) tests that directly:

| si_small fp32 | median | in-allocation IQR |
|---|---|---|
| unpinned, job 4376123 | 29.79 ms | 0.8 % |
| unpinned, job 4376140 | 53.67 ms | 29.7 % |
| **pinned, 5 allocations** | **27.67 ms** (range 27.49–28.58) | **≤ 2.4 %** |

Pinning removes the 80 % between-run swing, collapses the within-run IQR from 16–39 % to
0.8–2.4 %, and lands *faster than either* unpinned run — which is what local-socket placement
predicts, since the unpinned runs were paying interconnect crossings on some launches. The
hypothesis is therefore confirmed by intervention, not merely consistent with the provenance.

Residual spread after pinning is 3.9–5.3 % for the small fixtures against 0.5–2.1 % for the
medium ones, so a dispatch-bound configuration remains roughly 2–5× noisier than a device-bound
one even under ideal placement. That residual is the honest floor for host jitter on this
machine, and it is the number Phase 3 comparisons must clear.

### Verdict-table protocol (binding from here)

Any number that enters a verdict table is produced by `slurm/verdict.sbatch` +
`bench/aggregate_verdict.py`, which enforce:

* **N = 5 independent allocations**, as a SLURM array — not 5 repeats inside one allocation.
  Repeats within an allocation cannot observe between-node and between-placement variance, which
  the above identifies as the dominant term.
* **Median-of-medians** as the central value, and the **full range** across allocations as the
  error bar. The within-allocation IQR is reported alongside but is never the headline: it is the
  smaller, more flattering number and it hides precisely the variance that matters.
* **Host pinning** via `numactl --cpunodebind=0 --membind=0` (available at `/usr/bin/numactl`),
  with the placement echoed into the job log so it is auditable rather than assumed.

**Host jitter is a measurement target, not noise to be averaged away.** Baseline jitter,
characterised on 5 pinned allocations across 5 nodes (`bench/results/verdict_summary.json`):

| configuration | med-of-med | full range | spread | max in-allocation IQR |
|---|---|---|---|---|
| si_small fp32 | 27.67 ms | 27.49–28.58 | 3.9 % | 2.4 % |
| si_small bf16 | 29.86 ms | 28.80–30.39 | 5.3 % | 0.8 % |
| si_medium fp32 | 311.63 ms | 308.62–315.02 | 2.1 % | 0.3 % |
| si_medium bf16 | 191.44 ms | 190.95–191.85 | 0.5 % | 0.2 % |
| cu_medium fp32 | 497.45 ms | 493.82–504.35 | 2.1 % | 0.3 % |
| cu_medium bf16 | 304.93 ms | 303.94–306.30 | 0.8 % | 0.3 % |

**A Phase 3 speedup claim must exceed the spread for its configuration**, not the IQR: ~2 % at
the medium fixtures, ~4–5 % at the small ones. Where spread greatly exceeds the in-allocation
IQR, the configuration is dispatch-bound — diagnostic information about what the workload *is*,
and the same axis Phase 2's fusion attacks, so it is reported rather than smoothed.

### Secondary metric: max batch (GiB budgets)

Binary search over the replication factor of si_medium against a *measured* peak allocation:

| precision | budget | k | atoms | measured peak GiB |
|---|---|---|---|---|
| fp32 | primary 80 GiB | 2 | 11 664 | 76.22 |
| fp32 | full card 95.6 GiB | 2 | 11 664 | 76.22 |
| bf16 | primary 80 GiB | 3 | 17 496 | 62.59 |
| bf16 | full card 95.6 GiB | 4 | 23 328 | 83.43 |

fp32 cannot reach k = 3 under either budget: one medium cell already costs 38.13 GiB, so the
third copy would exceed the card. That is the same memory wall the 50 k fixtures hit, restated
as a batch limit.

Re-measuring on the regenerated fixtures moved nothing materially at the medium sizes, which is
the expected outcome given the fixtures changed only in seed, wrapping and neighbour-list backend,
not in size or physics.

Two results here matter for the bet:

**Eager cannot reach the 50 k fixture at all.** It fails trying to allocate a single
34 986 786 816 B (35.0 GB) tensor with 28.0 GB free of 102.0 GB. A single interaction block on a
46 656-atom cell exceeds a 95.6 GiB GH200 — so the memory axis is not a secondary metric here, it
is the difference between running and not running.

**The small fixtures are launch-bound, not compute-bound.** si_small takes 29.4 ms for 9 620
edges while si_medium takes 312 ms for 27× more edges — only ~10.6× the time. Profiling one
si_small fp32 step (total device time 29.76 ms ≈ the 29.44 ms wall median):

| op | calls | self device time |
|---|---|---|
| `aten::mul` | 441 | 2.23 ms |
| `aten::add_` | 172 | 0.93 ms |
| `aten::mm` | 70 | 5.58 ms |
| `aten::bmm` | 45 | 1.89 ms |
| `elementwise_kernel` | 134 | 1.70 ms |
| `vectorized_elementwise_kernel` | 250 | 1.01 ms |

Hundreds of individually-dispatched elementwise ops per training step — 441 `mul` and 172 `add_`
alone — against only 115 matmuls is the per-operator stack's structural cost, and it is precisely
what three-pass joint compilation is meant to collapse. Recorded as the Phase-3
bottleneck-analysis starting point.

*(An exact leaf-kernel launch count is pending: the obvious way to compute it — summing `count`
over `key_averages()` rows with non-zero device time — double counts, because an `aten::mm` row
and the `sm90_xmma_gemm...` row it launched both carry device time. `bench/count_launches.py`
counts CUPTI leaf kernels instead and will be run on the exclusive node with the final numbers.)*

*Fairness check performed:* the reference originally rebuilt the constant `to_m` permutation
on every forward with a Python loop of single-element assignments. That was removed (cached as a buffer)
and the step re-measured: si_small 30.46 vs 29.44 ms, si_medium 311.58 vs 312.24 ms — no
significant effect, so the table above stands and no baseline was crippled by that artifact.

### B2 — torch.compile cannot run the conservative training step at all

This is the strongest single piece of Gate-0 evidence, and it is a hard failure rather than a
partial fallback. Every variant that reaches AOTAutograd dies with:

```
RuntimeError: torch.compile with aot_autograd does not currently support double backward
```

A backend ladder locates the blocking layer precisely (fp32, medians of 100 iters):

| variant | what it exercises | si_small | si_medium |
|---|---|---|---|
| eager | control | 33.76 ms | 311.63 ms |
| `torch.compile(backend="eager")` | dynamo capture only, no AOTAutograd | 36.50 ms | 311.86 ms |
| `torch.compile(backend="aot_eager")` | AOTAutograd, no codegen | **RuntimeError** | **RuntimeError** |
| `torch.compile()` (inductor) | full stack | **RuntimeError** | **RuntimeError** |
| `mode="reduce-overhead"`, `fullgraph=True`, `mode="max-autotune-no-cudagraphs"` | — | **RuntimeError** | **RuntimeError** |

So the failure is **not** in graph capture and **not** in our code:

- **The block captures cleanly.** `torch._dynamo.explain` on the reference forward reports
  `graph_count = 1`, `graph_break_count = 0`, `op_count = 235` — one whole-block graph, zero
  breaks.
- **AOTAutograd is the blocker.** `backend="eager"` (dynamo capture, eager execution, never
  enters AOTAutograd) is the *only* compiled variant that runs — and it delivers no speedup
  (36.50 vs 33.76 ms at si_small; 311.86 vs 311.63 ms at si_medium, i.e. within noise), because
  nothing is fused or codegen'd.

**Therefore the best achievable torch.compile hybrid for this workload is "compile nothing that
matters".** There is no partial-fallback story to tune: the conservative training step's defining
feature — a second derivative through the compiled region — is exactly what the compiled backward
does not support. For the Phase 3 table, B2's honest entry at every size is *cannot run*, with
`backend="eager"` reported as the no-op upper bound.

Separately, a real fairchem stack additionally hits graph breaks in the rotation itself:
`torch._dynamo.explain` on `eulers_to_wigner ∘ init_edge_rot_euler_angles` reports
`graph_count = 4`, `graph_break_count = 3`, the first being
*"Skip calling `torch.compiler.disable()`d function `Safeatan2.backward`"* — the decorator noted
in §4.1. Our rational rotation has no such break, which is why the block above captures in one
graph.

*Operational note:* inductor's parallel compile-worker pool deadlocked repeatedly on this shared
login node (parent process blocked in `do_wait`, zero output, GPU idle).
`TORCHINDUCTOR_COMPILE_THREADS=1` makes compilation deterministic; compile time is outside the
measured region either way. `baselines/b2_probe.py` is the fast ladder used above;
`baselines/b2_compile.py` keeps the fuller inventory including the fairchem probe.

### B3 — cuEquivariance: expressible, fused, and double-differentiable (with a platform caveat)

The survey question the work order asks — *can the current segmented-polynomial API express this
SO(2) block at all?* — is answered **yes**, and B3 is implemented and numerically validated
rather than written off.

`cuequivariance.group_theory.experimental.escn.escn_tp_compact` builds a `SegmentedPolynomial`
whose path structure is exactly eSEN's per-m complex contraction. Read off the descriptor
(lmax = mmax = 2), input/output segments run m = −2,−1,0,+1,+2 and the weight blocks are
[m=0], [|m|=1 W1], [|m|=1 W2], [|m|=2 W1], [|m|=2 W2], with paths

```
(W1, −m, −m) +c    (W1, +m, +m) +c        c = 1/sqrt(u)   for m = 0
(W2, +m, −m) +c    (W2, −m, +m) −c        c = 1/sqrt(2u)  for m > 0
```

i.e. `out_r = W1·x_r − W2·x_i`, `out_i = W1·x_i + W2·x_r`, with cuEq's `+m` segment holding the
real part. The `1/sqrt(...)` factors come from `normalize_paths_for_operand(2)` and are known
exactly (measured: 0.036084 = 1/√768, 0.031250 = 1/√1024, 0.044194 = 1/√512), so our weights map
onto cuEq's by an exact rescale — which makes B3 *validatable*, not merely benchmarkable:

| check | FP64 | FP32 |
|---|---|---|
| cuEq `fused_tp` vs plain-torch contraction, identical weights | **1.63e-15** | **4.05e-07** |

Three platform facts, all measured:

1. **`uniform_1d` cannot run this descriptor.** It rejects the 2-D `uv` weight operand:
   *"For method 'uniform_1d', only 0D (scalar) or 1D operands are supported."* `fused_tp` is the
   only fused method available for the SO(2) path.
2. **Double backward through `fused_tp` works** under plain `torch.autograd` with
   `create_graph=True`. This resolves an ambiguity in NVIDIA's tracker: cuEq issue #264 reports
   second-derivative failures for `fused_tp`, but under `torch.func`; the plain-autograd path
   used by conservative training is fine.
3. **`cuequivariance-ops-torch-cu13==0.11.0` requires torch ≤ 2.11.0 on this platform.** Against
   torch 2.13.0 its extension fails to load with
   `undefined symbol: torch::Library::_def(c10::FunctionSchema&&, c10::OperatorName*, const std::vector<at::Tag>&, torch::_RegisterOrVerify)`
   — a libtorch C++ ABI break. Measured across three torch versions:

   | torch | cuEq ops extension |
   |---|---|
   | 2.13.0+cu130 (main stack) | fails to load |
   | 2.12.1+cu130 | fails to load |
   | **2.11.0+cu130** | **loads** |

   0.11.0 is the newest published build, so there is no forward fix available. Without the
   extension every method silently degrades to `method="naive"` (pure PyTorch) with only a
   warning — reporting that as "cuEquivariance" would violate the anti-gaming rule that baselines
   run at their recommended fast settings, so B3 runs in its own torch-2.11.0 interpreter with an
   **eager control measured in the same interpreter**, and the B3/control ratio is what is
   compared. Both numbers are reported. The control confirms the two interpreters are comparable:
   eager measures 37.43 ms / 312.28 ms (si_small / si_medium, fp32) under torch 2.11.0 against
   29.44 ms / 312.24 ms under torch 2.13.0 — the medium fixture, which is the primary one, agrees
   to 0.01 %.

#### B3 timings and an unresolved structural mismatch

| | si_small (9 620 edges) | si_medium (259 542 edges) |
|---|---|---|
| B3 `cueq[fused_tp]` | 701.73 ms, **85.38 GB** | **OOM** |
| B3 eager control (same interpreter) | 35.46 ms, 1.61 GB | 312.79 ms, 40.97 GB |

**These are not yet a fair characterisation of cuEquivariance and are not reported as one.**
85 GB for 9 620 edges is an artefact of how eSEN's weight sharing maps onto this descriptor, and
the cause is understood:

`escn_tp_compact` descends from the **eSCN** formulation, where the radial network emits a
*per-edge weight matrix* — so operand 0 (the weights, 622 592 values here) is a batched input with
one row per edge. **eSEN instead shares W across edges** and varies only a per-edge diagonal gain.
Expressing the shared-weight form through this descriptor makes cuEq materialise a
`[E, 622 592]` operand: 24 GB at si_small, 646 GB at si_medium.

Passing a one-row weight table plus `input_indices={0: zeros(E)}` — the pattern cuEq's docs use
for shared weights — removed the explicit `expand` but *not* the materialisation. Sweeping the
backends against the conv1-shaped descriptor at si_small (peak allocation, one forward) shows why,
and exposes a sharper problem than a missing option:

| method | weights as | peak | max rel err vs plain-torch contraction |
|---|---|---|---|
| `fused_tp` | one-row + `input_indices` | 24.18 GB | **1.63e-15** (FP64) |
| `fused_tp` | expanded | 24.18 GB | 1.63e-15 |
| **`indexed_linear`** | one-row + `input_indices` | **0.25 GB** | **0.665** (FP64) — *wrong* |
| `indexed_linear` | expanded | — | `KeyError` (index map required, by design) |
| `uniform_1d` | — | — | cannot run: rejects the 2-D `uv` weight operand |

So the two properties we need do not coexist on this descriptor:

- **`fused_tp` is numerically exact but densifies** the weight operand to `[E, 622 592]`
  regardless of how it is called — 24 GB at 9 620 edges, 646 GB at si_medium's 259 542.
- **`indexed_linear` scales but does not compute the same function.** It is 97× smaller on the
  weight operand and runs si_medium without OOM, but returns 0.665 relative error in FP64 (0.712
  in FP32) against the plain-torch contraction that `fused_tp` reproduces to 1.63e-15 from the
  *same* weights.

The obvious explanation — a weight-block memory-order convention — was tested and **ruled out**.
Sweeping both packings of the `(u, v)` blocks against the plain-torch reference (FP32):

| method | packing | rel err |
|---|---|---|
| `fused_tp` | **(u,v), transposed** | **6.13e-07** ✅ |
| `fused_tp` | (v,u), untransposed | 1.577 |
| `indexed_linear` | (u,v), transposed | 0.712 |
| `indexed_linear` | (v,u), untransposed | 1.210 |

`fused_tp` confirms our packing is the correct one; `indexed_linear` is wrong under **both**, so
the discrepancy is not memory order. It also warns that it ignores `math_dtype`, so its FP64 path
may not be FP64.

*Scope of that claim:* this establishes that `indexed_linear` does not reproduce
`escn_tp_compact`'s semantics under either natural weight packing. It does **not** prove the
backend can never serve this workload — it may require a differently-constructed descriptor rather
than a differently-packed operand. Determining that needs cuEquivariance internals and is
time-boxed out of M1; what is reported is what was measured.

For completeness, `indexed_linear`'s timings (numerically invalid, listed only to show it is not a
hidden performance win being left on the table): si_small 79.58 ms / 1.63 GB, si_medium
1092.38 ms / 42.04 GB — 2.2× and 3.5× *slower* than the eager control at essentially identical
memory.

The root cause of the mismatch is architectural, not a bug: `escn_tp_compact` descends from
**eSCN**, where the radial network emits a *per-edge weight matrix*, so operand 0 is a batched
input. **eSEN shares W across edges** and varies only a per-edge diagonal gain. cuEquivariance
expresses eSEN's *math* exactly; what is in question is whether it can execute eSEN's
*weight-sharing structure* at scale.

The 701.73 ms / 85.38 GB numbers above are `fused_tp` and are therefore dominated by that
densification. They stay in the table with this caveat rather than being quietly dropped, and they
are **not** presented as "cuEquivariance is 20× slower than eager".

#### Triage of the 0.665 error: **a cuEquivariance bug, not our mapping**

Gate 0 review required this classified before any conclusion was written.
`bench/b3_triage.py` builds a **hand oracle transcribed directly from the descriptor's own
operand/path list** — a literal reading of `uv,u,v` semantics, with no eSEN assumptions and no
layout guessing:

```
for each path (w_seg, in_seg, out_seg) with coefficient c:
    out[out_seg][v] += c * sum_u  weight[w_seg][u, v] * x[in_seg][u]
```

and runs a size ladder from a 1×1 single-segment scalar up to the eSEN conv1 shape, sweeping the
within-segment ordering (`mul_ir` vs `ir_mul`, i.e. u-major vs v-major weight blocks). FP64:

| case | paths | `fused_tp` | `indexed_linear` | `naive` |
|---|---|---|---|---|
| `1x0 -> 1x0`, m=0 | 1 | 0.0 | **0.0** ✅ | 0.0 |
| `2x0 -> 3x0`, m=0 | 1 | 0.0 | **1.7e-08** ✅ | 7.0e-17 |
| `2x1 -> 2x1`, m=1 | 5 | 0.0 | **8.3e-01** ❌ | 1.2e-16 |
| `2x0+2x1`, m=1 | 5 | 0.0 | **4.3e-01** ❌ | 0.0 |
| `2x0+2x1+2x2`, m=2 | 9 | 1.4e-16 | **6.2e-01** ❌ | 1.4e-16 |
| eSEN conv1 shape | 9 | 7.7e-16 | **6.0e-01** ❌ | 0.0 |

**Classification: cuEq bug.** Two independent backends — `fused_tp` and `naive` — agree with the
hand oracle to machine precision on **6/6** cases, and with each other. `indexed_linear` disagrees
with all three, on the same descriptor, the same weights and the same inputs. Our weight mapping
is therefore correct; if it were wrong, `fused_tp` and `naive` would fail too.

The v-major sweep is wrong everywhere (0.59–1.57), which independently confirms u-major is the
right block ordering, so the discrepancy is not a `mul_ir`/`ir_mul` convention.

**Localization: `indexed_linear` fails on multi-path descriptors.** It is correct on both
single-path cases and wrong on every multi-path one. The 1.7e-08 on `2x0 -> 3x0` is FP32-level,
consistent with its own warning that it ignores `math_dtype` — i.e. correct, just computed in
single precision, which is itself worth reporting.

*What this ladder does not separate:* at m ≥ 1 an eSCN descriptor introduces three things at once
— multiple paths, **weight-segment reuse** (W1 serves both `(−m,−m)` and `(+m,+m)`; W2 serves
`(+m,−m)` and `(−m,+m)`), and a **negative coefficient** on the last of those. Any of the three
could be the trigger. The minimal failing case is small enough to hand to upstream as-is:
`2x1 -> 2x1`, m_max=1 — 12 weights, 6 inputs, 5 paths, 0.83 relative error.

Draft issue in `docs/upstream_cuequivariance_issue_DRAFT.md` — **not filed**, pending review.

---

**B3 verdict — a coverage gap, stated precisely.** cuEquivariance 0.11.0 *can* express eSEN's
SO(2) convolution: `escn_tp_compact` reproduces it to 1.63e-15 under `fused_tp`, and double
backward works through it under plain `torch.autograd` (which also resolves the ambiguity in
cuEq #264 — that report is about `torch.func`, not the plain-autograd path conservative training
uses). What no backend currently provides is *both* correctness and eSEN's shared-weight structure
at scale:

| backend | correct? | scales? |
|---|---|---|
| `fused_tp` | ✅ 1.63e-15 | ❌ densifies weights to `[E, 622 592]`; OOM at si_medium |
| `indexed_linear` | ❌ **upstream bug** (0.665) | ✅ 39.2 GiB at si_medium, but 3.5× slower than eager |
| `uniform_1d` | — | ❌ rejects the 2-D `uv` weight operand outright |

so **B3 contributes no valid wall-clock number to the Phase 3 table**, and is reported as this
coverage gap instead — which is the outcome the work order anticipates ("if no, write the coverage
gap into REPORT.md; do not force it"). The underlying reason is architectural: `escn_tp_compact`
descends from eSCN's *per-edge* weight matrices, while eSEN shares weights and varies a per-edge
diagonal gain.

*Earlier harness bug, recorded so the first run is not misread.* The very first B3 attempt OOM'd
at si_small because the shared weights were `expand`ed to `[E, 622 592]` explicitly. That was our
error, not cuEquivariance's; reporting it as "cuEquivariance runs out of memory" would have been a
false negative against a baseline.

### B4 — not applicable at this block config

FlashSO2 `main` enforces `SUPPORTED_LMAXES = (4, 6, 8)` with `mmax == lmax` and raises for
anything else (`flash_so2/_triton/common/layout.py:30`). The smallest published eSEN config is
**lmax = 2**, so FlashSO2 cannot run it. B4 is therefore reported as not-applicable rather than as
a number, and the Phase-2 S1 sanity anchor ("within noise of FlashSO2 forward") is unavailable at
this config (DECISIONS.md D10).

## 6. Gate 0 status

Green (`pytest`, FP64 unless noted):

```
61 passed in 6.33s
```

- reference ≡ fairchem (table §3)
- E rotation-invariant, F equivariant; E translation- and permutation-invariant
- **E exactly independent of the random roll γ** (spread 0.000e+00 over 4 seeds, and between
  γ = 0 and γ = 1.234). Checked for vacuity, because a bit-identical result is otherwise
  suspicious: γ changes the Wigner matrix by `max|ΔW| = 1.84` and the rotated edge message by
  `max|Δmsg| = 8.15` — O(1) changes to every intermediate — while E stays bit-for-bit equal at
  −2.2105725124966868. This passes only if the per-m complex contraction, the gate's
  real/imaginary pairing, and the l→m′ permutation all commute with a roll about the edge axis,
  so it is a sharp check on the whole assembly; the test asserts spread < 1e-12 rather than
  bit-identity
- F vs central differences; double backward vs central differences
- every parameter receives a non-zero gradient

**Gate 0 verdict: PASS**, with two baselines resolved as structural findings rather than numbers.

| Gate 0 requirement | status |
|---|---|
| fixtures exist | ✅ 6 fixtures, Si + Cu at 3 sizes, avg degree recorded |
| reference validated against fairchem | ✅ bit-exact on conv1/conv2/gate; Wigner 8.26e-15 FP64, 1.51e-06 FP32 |
| B1 baseline table | ✅ all sizes, 3 precisions (large = OOM, itself a result) |
| B2 baseline table | ✅ resolved: **cannot run** — AOTAutograd rejects double backward at every size |
| B3 baseline | ✅ resolved: expressible and double-differentiable (1.63e-15); the memory-viable backend is broken **upstream** (triaged, draft issue written), so no valid wall-clock number — reported as a coverage gap |
| B4 baseline | ⛔ not applicable — FlashSO2 requires lmax ∈ {4,6,8} |

Outstanding, carried into Phase 3 rather than blocking Gate 1: re-running every number on an
exclusive `sbatch` node.

### What Gate 0 already says about the bet

Nothing here is a measurement of *our* compiler — Phase 1 has not started. But three of the
per-operator stacks the bet must beat are now characterised, and two of them do not clear the
bar of *running at all*:

- **torch.compile: cannot express the measured unit.** Not a fallback, a hard `RuntimeError`. The
  strongest per-operator competitor on wall-clock is unavailable for conservative training.
- **Eager: cannot reach 50 k atoms.** One interaction block OOMs a 95.6 GiB GH200. Peak memory is
  not a secondary axis for this workload.
- **cuEquivariance: expresses the math but cannot execute eSEN's weight sharing.** `fused_tp` is
  exact but densifies the shared weight operand per edge (OOM at si_medium); `indexed_linear`
  scales but computes a different function — triaged to an upstream bug on multi-path descriptors,
  not to our mapping. It also needs an older torch than the rest of the stack, and covers only the
  two per-m contractions — the rotation, gate, envelope, scatter and readout stay eager, bounding
  the launch count it could remove even if it worked.

That makes the memory axis and the launch-count axis the two places where a fused three-pass
compiler has room to win, and it makes the wall-clock comparison against torch.compile moot on
this workload. Both are recorded here *before* any of our own numbers exist, so the Phase 3
comparison cannot be framed after the fact.

---

### Secondary metric: max batch, in GiB

Units are **GiB** (2³⁰ bytes) throughout. The work order's "80 GB" against a 95.6 GiB card is
ambiguous between base-10 and base-2 and between "the card" and "a fixed target"; resolved at Gate
0 review as DECISIONS.md D13:

| line | budget | definition |
|---|---|---|
| **primary** | **80 GiB** | largest replication factor of the medium fixture whose *measured* `torch.cuda.max_memory_allocated` stays ≤ 80 GiB |
| secondary | 95.6 GiB | the same against the full card |

Found by binary search over the replication factor (`bench/max_batch.py`); a configuration counts
as fitting only if a complete conservative training step runs *and* its peak stays under budget —
OOM counts as not fitting. Replication builds a block-diagonal graph of k disjoint copies with
node indices offset per copy, so degree and edge statistics stay identical to the fixture while
the work scales. To be measured on the exclusive node alongside the rest of the Phase 3 table.

---

## 7. Honest limitations

To be completed at Gate 3. M1 does **not** test: the full model (only one interaction block),
dynamic shapes (fixed buckets only), other architectures (no MACE/DeePMD), portability (SM90a
only), or multi-GPU.

---

## 8. Gate 1 — segmented-polynomial IR and source-to-source VJP

**Status: complete, pending review.** 102 tests green in 20.6 s (budget: < 5 min).

### 8.1 The complexity table (regenerate with `python bench/ir_stats.py`)

eSEN-sm (K4L2), FP32 sizing for peak-live bytes, no rematerialization and no fusion:

| program | ops pre-CSE | post-CSE | paths | signatures | families | **archetypes** | **fusion groups** | peak GiB (si_small) | peak GiB (si_medium) |
|---|---|---|---|---|---|---|---|---|---|
| fwd | 106 | 101 | 193 | 48 | 45 | **35** | **48** | 0.23 | 6.03 |
| force | 407 | 290 | 486 | 145 | 136 | **78** | **115** | 0.77 | 20.73 |
| dbwd | 1221 | 903 | 1467 | 379 | 352 | **149** | **320** | 2.12 | 57.32 |

Four progressively coarser groupings of the same ops, because they answer different questions:

* **signatures** — ops differing only in which buffers they point at count once;
* **families** — additionally abstract slice *offsets*, keeping extents;
* **archetypes** — what an **emitter** must know how to write. A generated kernel takes its
  coefficient and slice tables as runtime data and its extents as parameters; what it *cannot*
  take at runtime is its shape of computation — which operands are gathered, whether the result
  is scattered, and the contraction pattern up to index renaming. Path *count* is dropped too,
  since a kernel loops over a path table;
* **fusion groups** — kernel *launches* left after a cheap greedy producer–consumer fusion pass
  (fusable = consumer reads producer, same segment axis, matching trailing extents, no gather or
  scatter on the consumer) **subject to the group graph staying acyclic** — see the correction
  below, which is why this column is larger than the one first reported at Gate 1.

Neither is a timing; the interpreter is an oracle, and no number here belongs in a performance
table.

### 8.2 Reading it

**Growth is linear in program size, which is the good news.** Each VJP multiplies the op count
by roughly three: 101 → 290 → 903 post-CSE, i.e. 2.9× then 3.1×. That is what reverse-mode
should cost, and it means dbwd does not blow up combinatorially — the double backward is about
9× the forward, not exponentially larger. CSE earns its place at the dbwd scale, removing 26 %
of ops (1221 → 903) where it removed only 5 % on the forward, because the two VJP passes
regenerate the same intermediates. The vocabulary holds throughout: every derived program passes
`assert_closed`, and the derivatives are exact against three independent legs (§8.3).

**The right unit is the emitter, not the kernel.** An earlier draft of this section read the 352
as "far too many kernels to hand-write", which was the wrong frame: the compiler emits kernels,
humans write emitters. Counting what an emitter must actually handle gives **149 archetypes** for
dbwd. The collapse from 352 → 149 is exactly the VJP's doing: the transposes are structurally
identical to one another and differ only in extents and slice tables, which is what the closure
lemma predicts.

**Correction — the fusion-group column reported at Gate 1 was not a valid launch count.** It read
42 / 46 / 107; it is 48 / 115 / 320. The greedy pass merged an op into any group containing a
fusable producer, without checking that the resulting *group* graph stayed acyclic. It does not:
36 of 42 forward groups, 40 of 46 force groups and **101 of 107 dbwd groups** sat in dependence
cycles, meaning pairs of kernels each waiting on the other's output. LayerNorm is the archetype —
`x - mean(x)` fuses with `x`, while `mean(x)` reduces `x` into a group of its own, so the two
groups point at each other. Such a partition cannot be scheduled at all, so 107 was not a
pessimistic estimate of the launch count but an unachievable one. Constraining every merge to
preserve acyclicity gives the corrected column, verified by a Kahn topological sort in
`tests/test_ir_core.py`.

This materially weakens the launch-count claim: dbwd is **320** launches, not 107 — three times
worse, and only 2.8× below its 903 ops rather than 8.4× below. It does not change any measured
number, any correctness result, or the archetype counts, which are what Phase 2's emitter effort
scales with. Found while building the Phase 2 emitter, by noticing that a group's live-in was
computed from that same group's output.

For scale, eager issues roughly 3 500 CUDA kernels for one si_small training step (§5). The
320 figure is not directly comparable — profiler-counted kernels include ones a single `aten` op
expands into, and 320 is an unscheduled estimate rather than a measurement — but the order of
magnitude is the point, and it is the launch-bound axis that the small-fixture variance
independently implicates. On memory, the 57.32 GiB unscheduled peak at si_medium is **not** directly
comparable to eager's measured 38.13 GiB: our liveness model sums every live buffer with no
allocator reuse, while `max_memory_allocated` benefits from reuse of freed blocks. The honest
reading is that it is a pessimistic upper bound on the unscheduled program, and that Phase 2's
memory win has to come from fusion and rematerialization rather than from scheduling order
alone — order alone cannot beat an allocator that already reuses.

### 8.3 Validation ladder

All FP64, CPU, deterministic. Tiny synthetic programs plus one real fixture (si_small).

| check | result |
|---|---|
| 4.1 interpreter fwd ≡ reference (si_small) | **2.6e-16** (tolerance 1e-12) |
| 4.2 IR force ≡ autograd | **4.97e-15** |
| 4.2 IR parameter grads ≡ autograd | 1.9e-15 … 4.7e-16 across `c1_w0`, `c2_w0`, `ro_w1`, `rad_w0` |
| 4.3 IR dbwd ≡ double-autograd | L 3.7e-16, dL/dpos **4.90e-15** |
| 4.3 third leg: central differences of L | agrees to < 1e-6 |
| 4.4 translation / rotation / permutation invariance of E | < 1e-11 |
| 4.4 ΣF = 0 | < 1e-10 relative |
| 4.5 `poly_envelope` C² across the cutoff | residual vanishes at the correct rate |
| 4.6 PIT on a complex-product rewrite | accepted; planted sign flip rejected |
| closure after every transform | green, plus a falsification test |

Net torque is deliberately **not** tested: it is not a valid invariant on a periodic cell, so a
torque check here would test the fixture rather than the block.

**Oracle independence.** Bit-exact forward agreement between the IR interpreter and
`blocks/eso2_ref.py` implies shared torch op ordering, not independent derivation — both call the
same primitives in the same sequence. Forward independence therefore rests on the fairchem
cross-check (§3), and derivative independence on the autograd and finite-difference legs, not on
interpreter-versus-reference agreement.

### 8.4 The bug worth naming

The forward was exact at 2e-16 while forces were wrong by 2.6e-01. Cause: a path may read the
same operand more than once — `x*x` is one path with `operands = (0, 0)` — and the transform took
`operands.index(k)`, finding only the first occurrence and silently halving the derivative.

This is a *different* site from the diamond test, which covers one buffer consumed by several
ops. This is one path containing several occurrences of one operand. The first does not imply the
second; both now have dedicated regression tests (DECISIONS.md D21). It is also a good argument
for the three-leg dbwd check: a two-leg comparison between two implementations that share an
op ordering can agree while both are wrong.

### 8.5 Vocabulary accounting: `sin` and `cos` are never used

v1.1 declares eight scalar functions; the assembled programs use **six**. Neither `sin` nor `cos`
appears in fwd, force or dbwd — the rational Wigner rewrite means the forward never forms an
angle, and no derivative rule can introduce them (`rsqrt'` adds `reciprocal`, nothing adds
trigonometry).

So **position → E is a rational function of the atomic positions with `rsqrt` as the sole
non-polynomial primitive.** That is a representability result, not an optimisation, and it
improves the exact-verification outlook: `rsqrt` yields a square root of a rational, the same
kind of algebraic number as the Clebsch-Gordan coefficients, so the whole path lives in one real
algebraic extension rather than a transcendental one. Full write-up in
`findings/vocabulary-shrink.md`; the limits of the PIT claim in `findings/pit-exactness.md`.

`sin`/`cos` are left in v1.1 rather than removed, so an architecture that genuinely needs them
(an S² grid activation, or spherical harmonics of an angle) stays expressible.

### 8.5b The precision contract, arriving early

Phase 3 was to decide a precision policy per table row. Phase 2's S1b forced part of it at S1b
instead, because the honest correctness bar stopped being "equal".

A T1 kernel and the FP64 interpreter sum the same short terms in the same order, so any
difference is a codegen bug and bit-equality is the bar; the Wigner chain meets it at 0.000e+00.
A T2 kernel contracts over 128-320 channels, where the interpreter reduces blocked and the kernel
sequentially with FMA contraction. Neither is more correct, and demanding equality would be
demanding that two valid summation orders agree.

So each kernel now **ships its own bound** (D25). The emitter computes the reduction depth of the
schedule it just emitted and attaches it as module metadata; `codegen/bounds.py` turns that into
a number against the real inputs — `2·(depth−1)·eps·max Σ|terms|`, the standard reordering bound,
taken over Σ|terms| rather than over the result so that cancellation cannot make it too tight —
and the harness asserts `measured ≤ bound` for every emitted kernel automatically.

| kernel | template | depth | bound | measured |
|---|---|---|---|---|
| wigner_chain | T1 | 5 | 1.776e-15 | **0.000e+00** (exact) |
| radial_lin0 | T2 | 321 | 9.321e-13 | 1.554e-15 |
| radial_stage2 | T2 | 129 | 1.658e-13 | 1.554e-15 |

That the T2 discrepancy is *purely* ordering was established rather than assumed: a naive
same-order FP64 reference differs from the interpreter by the identical 1.554e-15.

The mechanism matters more than the numbers. A bound cannot be loosened without changing the
schedule that produced it, so "the tolerance was widened" cannot be a quiet step.

**The bound's design point, stated so it is not mistaken for something it is not.** The failure
modes it exists to catch — a dropped term, a missing barrier, a transposed index, a wrong slice —
do not perturb a result by ulps. They produce O(1) errors, roughly **ten orders of magnitude**
above the 1e-15 ordering noise the bound is drawn at. Against that separation a loose worst-case
bound over Σ|terms| is not a weakness: it is a razor sitting in a vast empty gap, and its
looseness (measured errors run 0.2–0.9 % of it) costs nothing because nothing real lands in
between. What it is *not* is a precision certificate. It makes no claim that a kernel is accurate
to its bound, only that no *structural* error is present; a bug that shifted the answer by a few
ulps would pass, and no test here asserts otherwise. T1's separate bit-equality assertion is the
tight check, and it applies wherever the summation orders genuinely coincide.

### 8.5b-bis Two classes of contract

The checks in this pipeline fall into two kinds, and keeping them distinct is what stops either
from being asked to do the other's job.

**Structural contracts** are exact and run *before* anything executes: the IR type checker,
`assert_closed`, the Kahn acyclicity guard on the fusion partition, the register-budget
precondition, and `MetadataMismatch` on a generated module's declared `SEGMENT` /
`REDUCTION_DEPTH`. They answer "is this a well-formed thing to run at all", they admit no
tolerance, and their failures are absolute. The unschedulable 107-launch partition was a
structural failure, and it went unnoticed precisely because no structural check existed for it —
every *numerical* test passed throughout, since the partition never affected a computed value.

**Numerical contracts** are bounded and run *after* execution: the per-kernel ordering bound,
checked as `measured ≤ bound` on every emitted kernel. They answer "did this compute what it
claimed", they are inherently approximate, and their bound is derived from the schedule rather
than chosen. `invar_101` was a numerical failure — well-formed, well-typed, correctly
partitioned, and wrong by 4.47e+00.

The planted-fault battery certifies the numerical layer only: emitter faithfulness *given a
correct schedule*. A fault introduced during schedule construction would be inherited by both the
kernel and its bound, which would then agree; that layer belongs to the structural contracts.

This is the L1/L3 verification architecture in miniature — cheap exact checks on structure,
bounded checks on values — arrived at here by having been bitten once at each layer rather than
by design.

### 8.5c A calibrated traffic model, and an instrument blocker

D24 established that bytes, not FLOPs, are the mechanism, so Phase 2's objective function is a
per-group DRAM **byte** model (D27). It is calibrated before it decides anything.

**[intervention]** throughout this subsection: the instrument was calibrated by changing the
known traffic and observing the response, and the model corrected by changing the model and
re-measuring.

**Blocker: `ncu` and CUPTI are not installed on this system.** `nvidia-cuda-cupti-cu13` resolves
only to a 0.0.1 stub and `libcupti.so` does not load. `dcgmi` is present, so the substitute is
DCGM's `dram_active` counter. Being coarser, the *instrument* is calibrated first against
`copy_` at known traffic: the raw counter reads a consistent −16 % against the device's 4.0 TB/s
nominal peak, and the fitted constant is **K = 4.777e12**, with a residual of **0.4 %**.

**K is not a bandwidth, and I first recorded it as one.** The device is a GH200 120GB SKU with
96 GB HBM3 — 102.0 GB frame buffer, 60 MiB L2, 2 619 MHz memory clock — whose nominal peak is
4.0 TB/s, and whose *measured* achieved copy bandwidth (CUDA events, no DCGM involved) is
3.387–3.596 TB/s. K exceeds both, and nothing can move data faster than the peak. So K is a
**calibrated effective instrument constant**, ≈ achieved bandwidth ÷ instrument response, folding
DCGM's ~27 % under-reporting of the busy fraction together with the real bandwidth into one
number that the fit cannot separate.

The 0.4 % residual distinguishes nothing here, and treating it as evidence was the error: a
constant fitted to reproduce known traffic reproduces known traffic well under either story. Only
the device's capacity and an achieved-bandwidth measurement taken *without* the instrument under
test discriminate, and both say HBM3. Everything derived from K is unchanged and remains valid —
it is fitted to this instrument on this machine — but it is not portable to another GPU or
another DCGM version without re-fitting, which the bandwidth framing would have wrongly implied.
`findings/dcgm-bandwidth-constant.md`.

Against that instrument, after four modelling corrections (full buffers → element fraction →
32-byte sectors → 128-byte L2 lines, each forced by measurement — `findings/traffic-model-calibration.md`):

| template | worst error | D27 gate |
|---|---|---|
| T2 (dense channel access) | **2.8 %** | **OPEN** |
| T1 (sparse strided reads) | **26.9 %** | **CLOSED** |

The gate is per-template, because a single verdict would either forbid the well-supported T2 use
or permit the unsupported T1 one. Sparse strided reads are still under-predicted; the residual is
partial L2 line reuse across a 648-byte per-edge stride, which compulsory-traffic accounting
cannot express. I stopped rather than tune two more constants into agreement — that would be
fitting the objective to the kernel it is supposed to judge.

### 8.5d Compiler cost: schedule construction is the scaling limit

The Phase 2 emitter was preflighted at S3 sizes to check that the 48-term AST chunking holds.
It does not get that far: the preflight ran fifteen minutes over dbwd's 320 groups without
emitting a row, because **building** a schedule — not rendering it — is the limit.

`bench/schedule_scaling.py` measures what drives it, over the forward's groups:

| predicted volume | emitted terms | build |
|---|---|---|
| 41 088 | 321 | 0.77 s |
| 99 840 | 390 | 7.15 s |
| 492 160 | 769 | 52.3 s |
| 658 048 | 5 132 | 94.2 s |

Fitted log-log: **t ~ volume^1.27 (R² 0.976)** against **t ~ terms^1.00 (R² 0.348)**.
**[correlation]** — cost *tracks* index-space volume. The mechanism inferred from this fit at the
time (that the constructor's cost *is* the index walk) was wrong, and the correction is below;
the fit itself stands.

**Measured at dbwd scale, and it corrects an extrapolation published here an hour earlier.** That
draft reasoned from "dbwd is 9× the forward" to 5 M- and 20 M-volume groups costing 21 minutes
and 2 hours apiece. dbwd's groups are not bigger — its largest index space is **666 112** against
the forward's 658 048 — there are simply 5.7× more of them (251 schedulable vs 44). The
extrapolation assumed the wrong axis of growth and was wrong by roughly an order of magnitude.

| | forward | force | dbwd |
|---|---|---|---|
| schedulable groups | 44 | 107 | 251 |
| max index-space volume | 658 048 | 663 808 | 666 112 |
| largest group build | 94.2 s | 92.0 s | 115.1 s (at 331 k volume) |
| fitted exponent vs volume | 1.27 | — | **1.40** (R² 0.951) |
| total volume, all groups | — | — | 7 747 454 |
| **whole-program schedule time** | — | — | **≈ 19.6 min** |

So dbwd schedules in about twenty minutes, not hours: tolerable, but with 8.7 of those minutes
spent on five groups, and squarely in the way of D28's emit-and-time arm, which puts schedule
construction inside a grouping search's inner loop.

**Profiled, and the cost was not where this section said it was.** `cProfile` on dbwd g210:
`peak_live_values` 256.8 s of 265 s (97 %), `build_schedule` 7.7 s (3 %), 4.5×10⁸ `dict.get`
calls. The liveness scan — the D26 register-budget precondition, run on every group — rescanned
the whole live set at every assignment, making it quadratic in assignment count. Bucketing values
by the step they die at makes it linear, in five lines.

| | before | after |
|---|---|---|
| fwd g40 (658 048 volume) | 94.2 s | **7.2 s** |
| dbwd g210 (331 136 volume) | 115.1 s | **2.8 s** |
| exponent vs volume, fwd / force / dbwd | 1.27 / — / 1.40 | **0.97 / 0.96 / 1.01** (R² 0.99 / 0.97 / 0.99) |
| whole-program dbwd | 19.6 min | **68 s** |

**D30's dual criterion is met** — k ≤ 1.2 on all three programs, whole-program dbwd about a
minute (D31).

The index-space *correlation* reported above was real (R² 0.976) and the *mechanism* inferred
from it was wrong: larger index spaces produce more assignments, and the quadratic scan was in
the assignment count. A tight fit to a plausible mechanism is not evidence for that mechanism —
the same lesson as the DCGM constant (§8.5c), one layer up. Applying the zero-masks during
enumeration, which this section previously proposed and which was ruled into S2's first commit,
is now measured to be worth ~3 %.

Per-kernel schedule / emit / compile wall-clock is recorded from here on (`codegen/costs.py`),
with no analysis attached, so the compile-time column exists when it is asked for.

### 8.6 Standing threads

| thread | status |
|---|---|
| (a) exclusive-node sbatch replacing the PROVISIONAL Gate 0 table | **closed.** Job 4376140 on `nid005332`, exclusive, 35 min, 7/7 stages. §5 now carries its numbers, with the small fixtures reported as unstable rather than as a number |
| (b) cuEq trigger isolation + updated draft issue | **closed.** `bench/cueq_isolate.py`; the trigger is **two or more paths accumulating into one output segment** — my three named candidates (weight reuse, negative coefficients, path count) are each individually fine at ~1e-16. Draft rewritten against hand-built descriptors → `findings/cueq-indexed-linear-trigger.md` |
| (c) `math_dtype` finding split into its own repro + draft | **closed, and it refuted my own Gate 0 claim.** `bench/cueq_math_dtype.py`: `indexed_linear` does *not* silently compute in FP32 — it computes at operand precision exactly (0.00e+00 in FP64) and ignores requests to *reduce* precision. The real finding is a three-way contract disagreement → `findings/cueq-math-dtype.md` |
| (d) `findings/` ledger | 5 entries: `vocabulary-shrink.md`, `pit-exactness.md`, `self-halved-derivative.md`, `cueq-indexed-linear-trigger.md`, `cueq-math-dtype.md` |
| (e) rename + push, remote `main` and `gate-0` verified | done — both refs confirmed on the remote |

### 8.7 A measurement-integrity failure, and the fix

Job 4376123 completed cleanly, but its `bench/results/*.json` were **partially overwritten** by a
login-node run I had started in parallel: both wrote the same directory, and the local run
finished B1 last. The exclusive-node log held `cu_medium fp32 = 498.09 ms, IQR 0.89`; the JSON on
disk held the contended `947.59 ms, IQR 191.31`, and I committed it. A partial overwrite is the
worst kind, because the file still looks plausible.

Fixed at the cause, not the symptom: every `Measurement` now records `host`, `slurm_job` and
`exclusive`, so provenance is checkable after the fact; and `bench/run_all.sh` takes an exclusive
`flock` on the results directory so a second run refuses rather than overwrites. The contaminated
JSONs were deleted and job 4376140 regenerates all of them coherently.

The same run also exposed a real bug in the max-batch metric: it reported bf16 `k=4` at 20.90 GiB
against `k=3` at 62.59 GiB — a larger batch using less memory. The exponential probe advanced its
lower bound without carrying the corresponding peak, so every result reported the `k=1` peak.
Fixed.

### 8.9 Parked

* **`ncu` cross-validation of the DCGM methodology.** `ncu` is absent from the default
  environment but very likely available on Alps through a `uenv` image. A single `ncu` run on the
  `copy_` calibration kernel would independently confirm the byte counts DCGM's `dram_active`
  implies, and would separate the two factors currently folded into K
  (`findings/dcgm-bandwidth-constant.md`). Low priority and explicitly **not blocking**: the
  instrument is already validated against known traffic to 0.4 %, which is what the traffic model
  needs. Worth one hour if one ever frees up.
