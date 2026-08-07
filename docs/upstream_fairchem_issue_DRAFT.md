# DRAFT — upstream issue for fairchem. **Not filed.** For review before submission.

Status: localization measured (below). Remaining blockers before this is sendable are listed in
the TODO block at the end.

---

**Title:** `Safeacos` silently produces an incorrect second derivative (affects conservative
force training)

**Labels:** bug, autograd

### Summary

> **Mechanism wording is kept in sync with `findings/compiled-ran-clean-wrong.md`**, which is the single source of truth for this defect's one-line description. Edit there first.

`Safeacos.forward` saves a `clamp`ed tensor under no-grad, so the second derivative loses its
dependence on `x`. `Safeacos` in `fairchem/core/models/uma/common/rotation.py` therefore returns
correct values and correct first derivatives, but an **incorrect second derivative**, with no
error or warning. Models
trained with a force term in the loss — i.e. conservative (gradient) force training, which
backpropagates through `F = -dE/dpos` — take a wrong gradient signal through the `beta` Euler
angle of the edge frame.

### Cause

```python
class Safeacos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_clamped = x.clamp(-1 + EPS, 1 - EPS)
        ctx.save_for_backward(x_clamped)     # <-- computed inside forward, i.e. under no_grad
        return torch.acos(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x_clamped,) = ctx.saved_tensors
        denom = torch.sqrt(1 - x_clamped.pow(2)).clamp(min=EPS)
        return -grad_output / denom
```

The body of `Function.forward` runs with grad disabled, so `x_clamped` is a plain tensor with no
`grad_fn` and `requires_grad=False`. The returned gradient therefore depends on `grad_output`
*differentiably*, but on `x` **not at all** — as far as autograd is concerned, `d(acos)/dx` is a
constant with respect to `x`. Differentiating a second time drops the
`d²/dx² acos(x) = -x * (1-x²)^(-3/2)` contribution entirely.

`Safeatan2` in the same file does not have this problem: it saves the raw inputs `y, x` (which do
carry graph) rather than a value computed inside `forward`.

### Reproducer

```python
import torch
from fairchem.core.models.uma.common.rotation import Safeacos

x = torch.randn(64, dtype=torch.float64).clamp(-0.9, 0.9).requires_grad_(True)

# Safeacos: the first derivative does not depend on x, so this raises
out = Safeacos.apply(x).sum()
(g,) = torch.autograd.grad(out, x, create_graph=True)
torch.autograd.grad(g.square().sum(), x)
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

# torch.acos on a clamped input: works, and is the correct value
xc = x.detach().clamp(-1 + 1e-7, 1 - 1e-7).requires_grad_(True)
out = torch.acos(xc).sum()
(g,) = torch.autograd.grad(out, xc, create_graph=True)
(gg,) = torch.autograd.grad(g.square().sum(), xc)   # fine
```

In the full rotation path the error does not surface as an exception, because `x` still reaches
the output through `Safeatan2` and the normalisation — the second derivative is silently
*incomplete* rather than absent. Measured on one GH200 in FP64, contracting the Wigner matrices
with fixed vectors (`u^T W(pos) v`, chosen to be direction-dependent — note `||W v||^2` and
`sum(W^2)` are rotation-invariant and give a vacuous ~1e-15 agreement):

| quantity | `Safeacos` | double-differentiable `acos` | agreement |
|---|---|---|---|
| E | 23.47743315 | 23.47743315 | exact |
| ‖F‖ | 20.71385 | 20.71385 | 8.9e-16 |
| ‖∂²‖ | **3103.208** | **3091.557** | **5.5 % relative** |

A second random configuration gave 7.1 %, so the magnitude is input-dependent rather than a fixed
offset.

**This is not a clamp-band edge case.** Measured against the analytic
`d2/dx2 acos(x) = -x (1-x^2)^(-3/2)` in FP64, the second derivative is *structurally absent* in
every band, including the deep interior where the clamp never fires:

| band | `Safeacos` d2 | analytic d2 | result |
|---|---|---|---|
| deep interior, \|x\| <= 0.5 | 0 | 2.40e+01 | autograd refuses: no graph |
| mid, \|x\| in [0.5, 0.9] | 0 | 2.35e+02 | autograd refuses: no graph |
| near-edge, \|x\| in [0.9, 1-1e-3] | 0 | 5.14e+04 | autograd refuses: no graph |
| clamp band, \|x\| > 1-1e-7 | 0 | 3.54e+17 | autograd refuses: no graph |

So the fix is not a matter of widening or tightening `EPS`: the derivative's dependence on `x` is
missing from the graph regardless of where `x` lies. (In the full rotation the failure is quieter
— `x` still reaches the output through `Safeatan2` and the normalisation, so the second derivative
comes out nonzero but *incomplete* rather than raising.)

### Suggested fix

Save the *input*, not a value computed inside `forward`, and clamp in `backward`:

```python
class Safeacos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.acos(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        denom = torch.sqrt(1 - x.clamp(-1 + EPS, 1 - EPS).pow(2)).clamp(min=EPS)
        return -grad_output / denom
```

Because the clamp now happens inside `backward`, it is part of the differentiable graph and the
second derivative is recovered. (A `torch.autograd.gradgradcheck` on `Safeacos.apply` would have
caught this and would guard against regressions.)

### Impact

- Direct-force models are unaffected (no second derivative is taken).
- **Conservative / gradient-force training is affected**: the loss backpropagates through `F`, so
  the parameter updates receive a wrong contribution from the edge-frame `beta` angle.
- Energies and forces at inference are unaffected.

### Environment

fairchem-core 2.11.0, torch 2.13.0+cu130, CUDA 13.0, NVIDIA GH200 (sm_90a), aarch64, Python 3.13.

---

## TODO before this is sendable

1. Re-check against fairchem `main`, not just 2.11.0 — the file may have changed.
2. Search existing fairchem issues for a duplicate before filing.
3. Decide whether to include the `gradgradcheck` regression test as a PR alongside the report.
4. Confirm the suggested fix actually restores the correct second derivative end-to-end (patch
   `Safeacos`, re-run `bench/safeacos_localization.py`, and check the full-rotation `‖∂²‖` lands
   on the double-differentiable reference) before proposing it upstream.

*(Interior-vs-clamp-band localization: done — see the band table above. The defect is present in
every band, so it is a graph-structure bug rather than an `EPS` tuning question.)*
