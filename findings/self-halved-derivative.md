# Self-finding: the harness caught a halved derivative in its own VJP

Filed with the same rigour as the upstream findings, because a bug we found in our own code is
evidence about the harness in exactly the way an upstream bug is evidence about a dependency.
The difference is only in who has to fix it.

## Symptom

The IR forward matched the reference block to **2.03e-16** while forces were wrong by
**2.6e-01** — a 26 % error in `F = -∂E/∂pos` with an exact energy. Parameter gradients were
wrong by 1.7e-01 at the same time.

That combination is the diagnostic: an exact forward with a badly wrong first derivative rules
out the whole forward path and points squarely at the transform.

## Minimal repro

```python
from zippel.ir import BufferType, ContractionPath, IndexType, Program
from zippel.vjp import vjp
from zippel.interp import run
import torch

t = BufferType("edge", (("c", 3),)); sl = (slice(None),)
p = Program(); p.add_input("x", t); p.add_input("ze", IndexType("edge"))

# x*x as ONE path reading operand 0 twice
sq = p.contract(["x"], t, [ContractionPath(1.0, "c,c->c", (0, 0), (sl, sl), sl)])
tot = p.contract([sq, sq], BufferType("graph", ()),
                 [ContractionPath(1.0, "c,c->", (0, 1), (sl, sl), ())],
                 out_index_map="ze")
p.outputs = (tot,)
seed = p.add_input("seed", BufferType("graph", ()))
grads = vjp(p, tot, ["x"], seed=seed, zero_index={"edge": "ze"})
# before the fix: exactly half of autograd's d/dx of sum((x*x)**2)
```

## Mechanism

A `ContractionPath` names which operands it reads, by index. Nothing stops a path naming the
same operand twice — `x*x` is one path with `operands = (0, 0)`, which is how squaring, the
per-`l` squared-norm readout, and `s2 = x̂² + ẑ²` are all written.

The transpose rule located the operand with:

```python
j = p.operands.index(k)          # WRONG: finds only the first occurrence
```

`list.index` returns the first match. For `x*x` the product rule needs **two** contributions —
one holding each factor fixed — and the rule emitted one. The derivative came out exactly halved
wherever a path read an operand twice, and correct everywhere else, which is why the error was
26 % rather than 100 %: only part of the graph was affected.

Fix: iterate over every position at which operand `k` appears.

```python
occurrences = [(p, j) for p in op.paths
               for j, kk in enumerate(p.operands) if kk == k]
```

## Why the existing tests missed it

There was already a dedicated cotangent-accumulation test — the diamond, `x → {a, b} → merge`.
It passed throughout. The two are different sites:

| site | shape | test |
|---|---|---|
| **buffer**-level | one buffer consumed by several ops | `test_cotangent_accumulation_on_a_diamond` |
| **path**-level | one path reading one operand several times | `test_repeated_operand_in_one_path_gets_the_product_rule` |

Having the first does not imply the second, and the vocabulary makes the second easy to reach
without noticing: `x*x` looks like a single multiplication, not like an accumulation site.

## What caught it

Not a unit test — the **block-level comparison against an independent oracle**. The unit tests
covered every `scalar_map` derivative individually and the diamond, and all passed. The bug only
surfaced when the assembled block's forces were compared to torch autograd on the real block.

Two process consequences, both now in place:

1. **Regression test**: `tests/test_ir_block.py::test_repeated_operand_in_one_path_gets_the_product_rule`
   builds the minimal repro above and asserts against autograd.
2. **The three-leg dbwd check earns its keep.** The work order asked for IR ≡ double-autograd
   *plus* finite differences rather than two legs. Two legs would still have caught this one
   (autograd is genuinely independent of our transform), but the general hazard it guards is
   real: our interpreter and the reference share torch's op ordering, so agreement between them
   is weaker evidence than it looks. That is now stated explicitly in REPORT.md §8.3.

## Status

Fixed, tested, recorded as DECISIONS.md D21. No other site in the codebase used
`operands.index`; the transform is the only consumer of that field.
