# The segmented-polynomial IR — specification v1.1

Prose name: **segmented-polynomial IR**. Never "SPIR" (speech collision with SPIR-V); the Python
package is `zippel` (DECISIONS.md D15).

A program is an ordered, SSA-style list of ops. Each op names its inputs, produces exactly one
typed buffer, and is pure. Buffers are immutable; there is no aliasing and no mutation.

---

## 1. Buffer type model

```
BufferType = (segment, axes, dtype)
```

| field | meaning |
|---|---|
| `segment` | `node` \| `edge` \| `graph` \| `none` — the leading, *dynamic* axis |
| `axes` | ordered tuple of `(name, size)` — the block-structured trailing axes, all static |
| `dtype` | `f64` in the interpreter; `f32`/`bf16` are a Phase 2 concern |

Concrete shape is `(len(segment), *sizes)`, where `len(node)=N`, `len(edge)=E`, `len(graph)=G`,
and `len(none)=1` (parameters and constants). The trailing axes carry the block structure: for
this block, an `(l, m)`-flattened coefficient axis (`coeff`, size `(lmax+1)²`) and a `chan` axis.

**Shape checking happens at construction time.** `Program.add` validates every op's operand types
against its rule and computes the result type; an ill-typed program cannot be built. Phase 1 does
**not** carry irrep selection-rule typing — types exist here only to make contractions
well-defined. That is Phase 2 scope.

Index buffers (`edge_index_src`, `edge_index_dst`, gate expansion indices, scatter targets) are
integer buffers with `axes = ()`; they are inputs, never produced by ops, and never
differentiated.

---

## 2. Op vocabulary v1.1

Exactly two ops. Closure is checked mechanically after every transform
(`zippel.vjp.assert_closed`).

### 2.1 `segmented_contraction`

```
inputs      : (b_0, …, b_{n-1})
index_maps  : (m_0, …, m_{n-1})   each an index-buffer name or None
out_index_map : m_out or None
paths       : tuple[ContractionPath, ...]
out_type    : BufferType
```

with

```
ContractionPath = (coeff, subscripts, operands: tuple[int,...], in_slices, out_slice)
```

`operands` names *which of the op's inputs this path reads*, by index, and `subscripts` has one
spec per entry in it. That is what lets one op express both a product (one path over two
operands, `"ic,ic->ic"`) and a **sum** (two paths, each over one operand, `"ic->ic"`). With one
spec per op-input instead, addition would be inexpressible — `einsum("ic,ic->ic")` is a product
(DECISIONS.md D17).

Semantics — accumulate over paths, starting from zero:

```
for p in paths:
    operands = [ (b_k[m_k] if m_k else b_k)[:, *p.in_slices[j]]  for j, k in enumerate(p.operands) ]
    contrib  = p.coeff * einsum(p.subscripts, *operands)          # over trailing axes only
    if m_out:  out[:, *p.out_slice].index_add_(0, m_out, contrib)
    else:      out[:, *p.out_slice] += contrib
```

The leading segment axis is *batched*, never contracted — `subscripts` range over trailing axes
only. `index_maps[k]` performs a **gather** along the segment axis; `out_index_map` performs a
**scatter-add**. Both are what make the op "segmented".

**Restriction (asserted).** Every index that is summed out by a path must appear in at least two
operands. A single-operand reduction such as `"mc->c"` has no expressible transpose within the
vocabulary (its VJP would need a broadcast, which `einsum` cannot express as a contraction), so it
is rejected at construction. In practice this costs nothing: the one reduction in this block is
the per-`l` squared norm `Σ_m h[l,m,c]²`, which is bilinear (`"mc,mc->c"`) and so already has `m`
in both operands.

**What this one op realizes** — the index-map pattern for each use:

| use | pattern |
|---|---|
| neighbour gather (node→edge) | `index_maps = (edge_index_src,)`, `out_index_map = None` |
| scatter-add (edge→node) | `index_maps = (None,)`, `out_index_map = edge_index_dst` |
| Wigner-D application | `subscripts = "ij,jc->ic"`, operands (rot `[E,9,9]`, msg `[E,9,C]`) |
| rotate back | `subscripts = "ji,jc->ic"` — the same buffer, transposed subscript |
| per-m conv1/conv2 | `in_slices` select the m-block rows; `subscripts = "ic,oc->io"` against a `none`-segment weight |
| radial modulation (Hadamard) | `subscripts = "ic,ic->ic"` — a diagonal index pattern, not a separate op |
| elementwise add | two paths, `coeff = ±1`, `subscripts = "ic->ic"`, distinct operands |
| scalar broadcast | `subscripts = "x,->x"` |
| energy reduction (node→graph) | `out_index_map = zeros(N)` into a `graph` buffer of length 1 |

### 2.2 `scalar_map`

```
inputs : (b,)          # one operand
fn     : exp | sigmoid | silu | rsqrt | reciprocal | sin | cos | poly_envelope
order  : int           # poly_envelope only; derivative order, default 0
out_type = in_type
```

Elementwise over the whole buffer. Domain assumptions:

* `rsqrt`, `reciprocal`: argument **> 0**. Guaranteed here — every fixture asserts a minimum edge
  length at generation, and the two guarded quantities (`|r|²` and `s2 = sin²β`) are clamped away
  from zero in the reference construction.
* `poly_envelope`: the degree-5 polynomial cutoff, `p(d) = 1 + d⁵(a + d(b + c·d))` for `d < 1` and
  `0` otherwise, with `a = −(p+1)(p+2)/2`, `b = p(p+2)`, `c = −p(p+1)/2`, `p = 5`. **C² at
  `d = 1` by construction** (`p(1) = p'(1) = p''(1) = 0`), which is what makes the double backward
  well-defined at the cutoff.

**`poly_envelope` is the one vocabulary entry that is a *family*.** Its derivative is a *different*
piecewise polynomial, and a piecewise polynomial is not a product of the other seven entries — the
dynamic indicator `[d < 1]` cannot be written as a `segmented_contraction` over static
coefficients. Rather than widen the vocabulary with `poly_envelope_d1`/`_d2` as separate entries,
the op carries an integer `order` and differentiation increments it. Closure is therefore exact:
`d/dx poly_envelope(order=k) = poly_envelope(order=k+1)`. Recorded as DECISIONS.md D16.

---

## 3. VJP rules

Source-to-source: each rule *emits IR ops*. Nothing is evaluated during the transform.

### 3.1 `segmented_contraction`

**Core lemma.** The VJP of a `segmented_contraction` with respect to each operand is another
`segmented_contraction`, obtained by transposing the index maps and the einsum subscripts.

For operand `k`, per path:

| forward | VJP w.r.t. operand `k` |
|---|---|
| paths not reading operand `k` | dropped — they contribute nothing |
| `subscripts = in_0,…,in_k,…,in_{n-1} -> out` | `in_0,…,out,…,in_{n-1} -> in_k` (swap slot `k` with the output spec) |
| `index_maps[k] = m_k` (gather) | `out_index_map = m_k` (scatter-add) |
| `out_index_map = m_out` (scatter) | `index_maps[cot] = m_out` (gather) |
| `in_slices[k]`, `out_slice` | swapped: read `out_slice` from the cotangent, write `in_slices[k]` |
| `coeff` | unchanged |

Gather and scatter-add are exact transposes of one another, which is why the swap is the whole
rule. One case is not a plain swap: a `none`-segment operand is **broadcast** over the segment
axis in the forward direction, and the transpose of a broadcast is a **sum over that axis**. No
new op is needed — it is a scatter-add through an all-zeros index map into a length-1 buffer,
which the vocabulary already expresses (DECISIONS.md D18). The transform therefore takes a
`zero_index` map from segment name to an all-zeros index buffer. The other operands are carried through unchanged, so the VJP of a degree-`n` multilinear
path is `n` paths of the same degree — the op is closed under differentiation, and degree does not
grow.

### 3.2 `scalar_map`

The VJP is `cot * f'(x)`, emitted as a `scalar_map` for `f'` plus a Hadamard
`segmented_contraction` (`"ic,ic->ic"`). Every `f'` stays inside the vocabulary:

| `f` | `f'` | emitted as |
|---|---|---|
| `exp` | `exp(x)` | reuses the forward **output** `y` — no new op |
| `sigmoid` | `σ(1−σ)` | reuses `y`; one contraction `y·(1−y)` |
| `silu` | `σ(x) + x·σ(x)·(1−σ(x))` | one `sigmoid`, three contractions |
| `rsqrt` | `−½·x^{−3/2}` | `−½ · rsqrt(x) · reciprocal(x)` — reuses `y`, one `reciprocal` |
| `reciprocal` | `−x^{−2}` | `−y²` — reuses `y`, one contraction |
| `sin` | `cos(x)` | one `cos` |
| `cos` | `−sin(x)` | one `sin`, coefficient `−1` |
| `poly_envelope(k)` | `poly_envelope(k+1)` | order increment (§2.2) |

Reusing the forward output where the derivative is a function of `y` rather than `x` is what keeps
the derived programs small; it is also the first thing CSE would find anyway.

### 3.3 Cotangent accumulation

A buffer consumed by `k` ops receives the **sum** of `k` contributions. The transform walks the
program in reverse, maintaining a `dict[buffer → list[contribution]]`, and materialises the sum
only when the buffer is popped. This is the classic AD bug site, so it has a dedicated
diamond-shaped test (`x → {a, b} → merge`) checked against torch autograd rather than being
covered incidentally.

### 3.4 Program constructions

| program | definition |
|---|---|
| `fwd` | inputs → `E` (a `graph` buffer of size 1) |
| `force` | `fwd` + VJP slice w.r.t. `pos`, negated: `F = −∂E/∂pos` |
| `dbwd` | VJP of the `(E, F)` + loss program w.r.t. all parameters *and* positions, with `L = w_E·MSE(E) + w_F·MSE(F)` exactly as the measured unit defines it (REPORT.md §3) |

### 3.5 Closure assertion

After **every** transform, `assert_closed(program)` checks that every op is `segmented_contraction`
or `scalar_map`, and that every `scalar_map.fn` is in the v1.1 set. If anything escapes, the
transform raises rather than widening the vocabulary — an escape is a finding about the bet and is
written into REPORT.md.

---

## 4. Simplification

`zippel/simplify.py` implements CSE by structural hashing over `(op kind, attributes, input ids)`
in topological order, and DCE by reachability from the program's declared outputs. Neither
reassociates arithmetic, so both are exact in floating point — no tolerance is involved and no
result changes.

---

## 5. Deliberately out of scope in Phase 1

No codegen, no fusion or scheduling beyond CSE/DCE, no autotuning, no irrep selection-rule typing,
no rematerialization. The interpreter is an **oracle, not a contender** — its timings never appear
in a benchmark table.
