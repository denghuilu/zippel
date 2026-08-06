# M1 Vertical Slice — Execution Plan

## Context

M1 is the go/no-go milestone for a compiler program. It exists to **verify or falsify one
proposition** with real numbers on real hardware — not to build a framework:

> The energy–force–double-backward computation of an equivariant MLIP can be expressed in a
> differentiation-closed, verification-friendly **segmented-polynomial IR (SP-IR)**, and
> **three-pass (fwd/bwd/dbwd) joint compilation** on that IR stably outperforms existing
> **per-operator stacks** (eager / torch.compile / cuEquivariance + autograd) on **conservative
> training** — wall-clock and peak memory.

Scope is **one eSEN-style SO(2) interaction block**, not a full model. A clean negative result is
valid and valuable; a fabricated or gamed positive result is the only real failure mode.

---

## Findings from Phase-0 reconnaissance (already verified, not assumed)

These four change the plan and are load-bearing.

**1. `/capstor` cannot accept new files — the work order's repo path is unusable.**
```
/capstor/scratch/cscs/dlu   3,068,662 files / 1,000,000 limit = 306.9%, grace EXPIRED
touch: cannot touch '/capstor/scratch/cscs/dlu/iclr/.spir_probe/t': Disk quota exceeded
```
`/iopsstor/scratch/cscs/dlu` writes fine (727 T free, no file limit). `/capstor` reads are
unaffected, so the conda base, FlashSO2, and fairchem stay readable in place.
→ **Repo, conda env, and JIT caches all go on `/iopsstor`.** Asked the user; no reply within the
window, so proceeding with the recommended non-destructive option. Reversible at any time.

**2. fairchem's shipped rotation has a silently wrong double backward.** Measured on GH200, FP64,
`uᵀW(pos)v` (a genuinely non-invariant scalar — note `‖Wv‖²` and `Σ W²` are both invariant and
give a vacuous ~1e-15 result):

| quantity | fairchem `Safeacos` | double-differentiable acos | agreement |
|---|---|---|---|
| E | 23.47743315 | 23.47743315 | exact |
| ‖F‖ | 20.71385 | 20.71385 | 8.9e-16 |
| ‖∂²‖ | **3103.208** | **3091.557** | **5.5 % rel — WRONG** |

Cause: `Safeacos.forward` does `ctx.save_for_backward(x.clamp(...))`, and that clamp runs under
no-grad, so the saved tensor carries no graph; the first derivative's dependence on `x` is
invisible to the second differentiation. `Safeatan2` is fine (but is `@torch.compiler.disable`d —
free evidence for B2's fallback documentation).
→ fairchem autograd **cannot** be the dbwd oracle. Per the user's decision: use a
double-differentiable acos in **all** implementations so every party computes identical math, and
report the shipped defect in `REPORT.md` as a verification finding.

**3. The Euler-angle path can be made rational, so the declared vocabulary stays closed.**
Per the user's instruction to algebraically simplify `sin/cos ∘ acos/atan2` where trivial:
- β = acos(ŷ) ⇒ cos kβ = `T_k(ŷ)` (**polynomial**), sin kβ = `√(1−ŷ²)·U_{k−1}(ŷ)`, and
  `√(1−ŷ²) = (1−ŷ²)·rsqrt(1−ŷ²)` — **rsqrt is already in the set**.
- α = atan2(x̂, ẑ) ⇒ cos α = `ẑ·rsqrt(x̂²+ẑ²)`, sin α = `x̂·rsqrt(x̂²+ẑ²)`; cos kα / sin kα follow by
  De Moivre — **polynomial** in those two.
- γ is `torch.rand_like` (a *random* roll, and equivariance is exact in it), so sin kγ / cos kγ are
  **position-independent per-edge constants** — seed once, hoist to inputs.

⇒ Every Wigner-D entry is rational in the normalized edge vector. **No sin/cos/acos/atan2 needed;
closure holds under the declared set** {exp, sigmoid/SiLU, rsqrt, poly-envelope} plus a structural
`select` (which `PolynomialEnvelope`'s `where(d<1,·,0)` already requires). This also removes the
`Safeacos` bug by construction. A full angle-free Wigner recursion is **out of M1 scope** →
`DECISIONS.md` as future work.

**4. Naive CuTe DSL loses on this rotation — a documented FlashSO2 result.** Its
`agent/s1-rotation-cutedsl` prototype reached only 0.55×/0.46×/0.33× of production Triton at
lmax 4/6/8, because a dense WGMMA N×K tile pays quadratically for block-diagonal zeros; it only
won (by 4–6 %) after both-endpoint programs plus a one-gather Wigner tile. **At our lmax=2 the
block-diagonal is densest** (1+9+25 = 35 nonzeros of 81 = 43 %, vs 969/9216 = 10.5 % at lmax 8), so
this is the most favourable point in that curve — but Phase 2 must adopt the winning idioms from
the start, not rediscover them.

### Verified environment

`daint-ln001`, **4× GH200 120GB idle on the login node** (95.6 GiB each, sm_90a, driver 590.48.01 /
CUDA 13.1). No system `nvcc` and none needed — CuTe DSL goes through NVRTC/NVVM.
**CuTe DSL smoke test already passes**: an `axpy` via `@cute.kernel`/`@cute.jit` + `from_dlpack`
compiled and ran, max abs err 9.5e-07.
SLURM: account **`lp16` is valid** (and far less used than the user's habitual `c33`); partitions
`normal`/`low` 24 h, `debug` 30 min, `--gres=gpu:4`. No uenv, no containers — self-contained conda
env on scratch is the local convention.
Base env `iclr` is near-perfect: py 3.13.14 · torch 2.13.0+cu130 · **nvidia-cutlass-dsl 4.5.2** ·
fairchem-core 2.11.0 · e3nn 0.6.0 · ase 3.29.0 · pytest 9.1.0. Missing only `cuequivariance*`
and `vesin`, both of which have **aarch64 cp313 wheels** (0.11.0 / 0.6.1).

### Block config — smallest published eSEN (K4L2 / eSEN-sm)

Source: `fairchem_core-2.0.0` → `configs/puma/training_release/backbone/K4L2.yaml`; cutoff/neighbors
from the OMol25 eSEN-sm top-level config; cross-checked against arXiv:2502.12147 App. A.1
("Lmax=2, Mmax=2, 3.2 M params", 6 Å).

| | value | | value |
|---|---|---|---|
| `lmax` / `mmax` | **2 / 2** | `cutoff` | **6.0 Å** |
| `sphere_channels` | **128** | `num_distance_basis` | **64** (gaussian) |
| `hidden_channels` | **128** | `edge_channels` | **128** |
| radial MLP | **[320, 128, 128] → 1536** | `act_type` | **gate** |

`so2_conv_1` takes `2·128 = 256` (source⊕target), `internal_weights=False` (radial ON),
`extra_m0_output_channels = lmax·hidden = 256` (the gate scalars);
`so2_conv_2` is `internal_weights=True` (**no radial**). Derived: `fc_m0.in_features = 3·256 = 768`,
m=1 → 512, m=2 → 256, `num_channels_rad = 1536`. num_coeffs = 9.

**eSEN 2.0.0's `esen/` package cannot be installed** (it declares `Requires-Python >=3.9,<3.13`).
The maintained equivalent — fairchem 2.11's UMA `SO2_Convolution` / `Edgewise` — **is** installed and
will be instantiated at exactly these hyperparameters as the fairchem cross-check. Known algebraic
deltas UMA-vs-eSEN (to state in `REPORT.md`): `to_m` hoisted into the Wigner matrix, `SO2_m_Conv`
returns a tuple instead of `cat`, envelope always applied, `GateActivation(m_prime=True)`.

---

## Deliverable layout

Repo root **`/iopsstor/scratch/cscs/dlu/iclr/zippel`**, `git init` immediately.

```
zippel/
  PLAN.md  DECISIONS.md  REPORT.md  requirements.lock
  spir/       ir.py  interp.py  vjp.py  pit.py
  blocks/     eso2_ref.py  eso2_spir.py
  codegen/    emit.py  buckets.py
  baselines/  b1_eager.py  b2_compile.py  b3_cueq.py  b4_flashso2.py
  fixtures/  bench/  slurm/  tests/
  .jit-cache/ {triton,cute_dsl,quack}    # gitignored
```

---

## Phase 0 — Environment, reference, baselines

1. **Env.** `conda create -p /iopsstor/scratch/cscs/dlu/envs/spir --clone iclr` (clone, not
   build-from-scratch — torch 2.13.0+cu130 aarch64 is known-good and hard to reproduce). Then
   `pip install cuequivariance-torch==0.11.0 cuequivariance-ops-torch-cu13==0.11.0 vesin==0.6.1`.
   Freeze `requirements.lock`. **Prefix must be on `/iopsstor`** — a clone into
   `miniforge3/envs/` would hit EDQUOT.
   Env pinning, mirrored into `tests/conftest.py` at import time (FlashSO2's idiom, so a bare
   `pytest` is also persistent):
   `CUTE_DSL_CACHE_DIR`, `TRITON_CACHE_DIR`, `QUACK_CACHE_DIR` → `$REPO/.jit-cache/*`;
   `pytest addopts = -p no:cacheprovider`; `PYTHONPYCACHEPREFIX`; `PYTHONNOUSERSITE=1`.
2. **`blocks/eso2_ref.py`** — standalone, dependency-light, FP64-capable PyTorch reference for the
   whole block: gather → Wigner rotate (rational form, §3 above) → conv1 → gate → conv2 → envelope →
   rotate back → `index_add_` → per-atom energy head. **γ is seeded and passed in as an input**, never
   `rand_like` inside forward. Validate against fairchem 2.11 UMA `SO2_Convolution`/`Edgewise` at
   K4L2 (FP32 ≤ 1e-5, FP64 ≤ 1e-10), feeding fairchem the same γ and the corrected acos.
3. **Fixtures** (`fixtures/*.npz`, fixed seeds): perturbed bulk **Si diamond** (3³=216 / 9³=5832 /
   18³=46656 atoms) and **Cu fcc** (4³=256 / 11³=5324 / 23³=48668), PBC, `vesin` ragged neighbor
   lists at 6.0 Å. Expect ~45 nbrs/atom (Si) and ~77 (Cu). **Medium is primary.** The published
   `max_neighbors=30` cap is a data-loader detail with a nondeterministic tie-break → **not applied**;
   record avg degree and log the choice in `DECISIONS.md`.
4. **Baselines**, good-faith tuned, identical measured boundary:
   - **B1** eager fp32 + bf16-AMP.
   - **B2** best `torch.compile` hybrid — the *documentation of where it falls back is itself the
     deliverable*. Already-known fallback seeds: `Safeatan2.backward` is `@torch.compiler.disable`d,
     and inductor + double-backward is a reported breakage class.
   - **B3** cuEquivariance. **Survey first, and expect this to be the risky one.**
     `cuequivariance.group_theory.experimental.escn.escn_tp_compact` is a genuine 1:1 structural
     match to `SO2_m_Conv` (same complex mixing, same `c=-1.0` sine path), but: it is unexported,
     its tests only *construct* descriptors and never execute them, it carries `# TODO` stubs, the
     `uniform_1d` backend **cannot** run it (2-D `uv` weight operand rejected) so it needs
     `fused_tp`, and `fused_tp` has an **open second-derivative bug** (cuEq #264). → **First action
     is a `gradgradcheck` smoke test on `fused_tp`.** If it fails, write the coverage gap into
     `REPORT.md` and stop — do not force it. Also assert `method == "fused_tp"`; without
     `cuequivariance-ops-torch` it silently degrades to `naive`.
   - **B4** FlashSO2 `devel` forward as a fused-kernel forward reference point only.
5. **Gate 0** — fixtures exist; reference validated; baseline table (median + IQR, peak memory) at
   all three sizes in `REPORT.md`.

## Phase 1 — SP-IR core + symbolic differentiation (CPU/interpreter)

1. **`spir/ir.py`** — buffers typed by segment axis (node/edge/graph), block structure (per-l or
   per-m blocks × channels), dtype. Closed vocabulary: `segmented_contraction` (multilinear, static
   sparse coefficient table, operand index maps — the gather and the scatter-add are *index maps*,
   not separate ops) and `scalar_map` over {exp, sigmoid/SiLU, rsqrt, poly envelope} + structural
   `select`. Derive every layout table from `lmax`; memoize in a frozen dataclass; pin the lmax=2
   literals in a test (FlashSO2 `layout.py` idiom).
2. **`spir/vjp.py`** — source-to-source VJP. `bwd = VJP(fwd)`; `dbwd = VJP` of the `(E,F)` program
   under force-loss cotangents. **Closure test is load-bearing**: assert derived programs use only
   the same vocabulary. If closure fails, that is a finding — write it up, do not widen silently.
3. **`blocks/eso2_spir.py`** — the full block in SP-IR. Validation ladder, all FP64:
   interpreter fwd ≡ reference · IR bwd ≡ autograd (params **and** positions) · IR dbwd ≡
   double-autograd incl. `gradgradcheck` spot checks · property tests: rotation equivariance
   (E invariant, F rotates as a vector), permutation invariance, translation invariance of E.
4. **`spir/pit.py`** — one Schwartz–Zippel randomized polynomial-identity test on a nontrivial
   multilinear rewrite, to establish the mechanism.
5. Borrow FlashSO2's test architecture: `CaseSpec` as the single parametrize axis; tolerances as
   data (`cos`/`rel`/`nr`, not `atol`/`rtol`); `assert_close` **refuses a vacuous assertion**; and a
   **falsification test** that plants a sign flip to prove the FP64 arbiter has discriminating power.
6. **Gate 1** — all green in `pytest`. Proves the representability + differentiation-closure half.

## Phase 2 — CuTe DSL fused kernels

Lower SP-IR to **fused CuTe DSL kernels**; hand-guided emission per shape bucket, manual sweep
≤ ~24 tile/pipeline configs. Adopt from the start (do not rediscover): **packed block-diagonal
Wigner** `[E, Σ(2l+1)²]` with closed-form offset `l(4l²−1)/3` unpacked into a register tile in-kernel;
**permutation as pointer arithmetic**, never a fold-bmm; **both-endpoint programs** (grid `(E,)`);
one-gather Wigner tile.

- **S1** fused forward (E) — within noise of FlashSO2 forward on medium.
- **S2** jointly scheduled (E, F).
- **S3** full training step, fwd/bwd/dbwd jointly scheduled; every keep-vs-recompute decision for
  shared intermediates (radial weights, Wigner blocks, edge messages) logged in `DECISIONS.md`.

Per stage: match the FP64 interpreter at precision-appropriate tolerances; re-run equivariance
property tests **on the generated kernels**; finite-difference spot checks for F and one dbwd
direction.
**Gate 2** — at least S2 correct; state plainly whether S3 is reached, partial, or blocked.

## Phase 3 — The number

CUDA-event timing around the defined boundary (neighbor-list construction excluded for everyone);
≥ 20 warmup, ≥ 100 iters (≥ 30 at 50 k); median + IQR; NVML clocks recorded alongside.
Peak memory via `max_memory_allocated` (reset per config) + NVML cross-check. Same precision policy
per table row. Secondary: max batch of replicated medium cells at **80 GB** (work-order figure kept
as a hardware-independent budget; GH200's actual 95.6 GiB reported as context — user did not answer
the question, revisit if they prefer).

**All reported numbers come from an exclusive `sbatch` node** (`--account=lp16 --partition=normal
--gres=gpu:4 --exclusive --hint=nomultithread`), never the shared login node — login GPUs are for
dev and `pytest` only.

Anti-gaming, binding: baselines at their recommended fast settings; identical boundary and inputs;
weights are runtime tensors (shape specialization only, no constant-folding); no tolerance
loosening; everything reproducible via `bench/run_all.sh`.

**Expect the 50 k fixture to OOM for some implementations** — at E ≈ 2.1 M a single `[E,9,256]` fp32
message tensor is 19.3 GB, before double-backward saves. That is a legitimate result on the memory
axis; report max feasible size per implementation rather than hiding it.

**Gate 3 / verdict** — table + one paragraph answering the bet directly. strong go ≥ 1.5×
wall-clock **or** ≥ 1.5× peak-memory (at ≤ 1.05× time); weak zone 1.2–1.5× with bottleneck profile;
no-go < 1.2× on both, said plainly. Thresholds are review defaults — tune the kernels, never the
experiment.

---

## Working agreements

- `PLAN.md` written into the repo before code; every deviation gets a dated line in `DECISIONS.md`.
- **Stop at every phase gate**, print a gate summary, wait for review.
- Blocked > ~2 h on one issue → write the blocker into `REPORT.md`, move to the next independent task.
- No gate marked passed without a green `pytest` run pasted into `REPORT.md`.
- Scope fence honoured: no frontend/compile-backend integration, no MACE/DeePMD, no AMD/Triton
  portability layer, no agent loop, no real autotuner, no website, no multi-GPU, no dynamic shapes.

### Deviations from the work order (to be logged in `DECISIONS.md` on day one)

| # | Work order says | Actual | Why |
|---|---|---|---|
| 1 | repo at `/capstor/.../iclr/spir-m1` | `/iopsstor/.../iclr/spir-m1` | `/capstor` inode quota exhausted, grace expired — EDQUOT reproduced |
| 2 | Python 3.13 **pip venv** | conda env `spir` (py 3.13.14), pip inside | user instruction; clone of known-good `iclr` |
| 3 | validate against "the actual fairchem module" | fairchem 2.11 UMA `SO2_Convolution` at K4L2 | eSEN 2.0.0 requires py<3.13; UMA is the maintained descendant |
| 4 | transcendental set as declared | unchanged, via rational Wigner rewrite | user instruction; preserves closure |
| 5 | — | corrected double-differentiable acos everywhere | user decision; fairchem's is wrong by 5.5 % in dbwd |
| 6 | `max_neighbors=30` in eSEN config | full 6 Å ragged list | cap has a nondeterministic tie-break; not block math |

### Verification

- `pytest` (login-node GPUs) is the gate for Phases 1–2; `bench/run_all.sh` under `sbatch` for Phase 3.
- Ground truth ladder: FP64 SP-IR interpreter ← validated against `eso2_ref.py` ← validated against
  fairchem UMA. Finite differences and `gradgradcheck` backstop the dbwd, since **fairchem autograd
  is disqualified as a dbwd oracle**.
- Equivariance/permutation/translation property tests run at every level, including on generated kernels.
- Falsification test proves the oracle detects a planted bug.
