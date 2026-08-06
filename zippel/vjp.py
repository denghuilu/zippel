"""Source-to-source VJP on the segmented-polynomial IR.

Nothing is evaluated here: each rule *emits IR ops*. The claim this file exists to support is
that the vocabulary is closed under differentiation -- so `assert_closed` runs after every
transform, and raises rather than widening the vocabulary if anything escapes.

Core lemma (docs/ir.md 3.1): the VJP of a `segmented_contraction` w.r.t. each operand is
another `segmented_contraction`, obtained by swapping the einsum output spec with that
operand's spec and swapping gather with scatter-add. Degree does not grow.
"""

from __future__ import annotations

from zippel.ir import (
    SCALAR_FNS,
    BufferType,
    ContractionPath,
    IndexType,
    Program,
    default_spec,
    full_slice,
)


class ClosureError(RuntimeError):
    """Raised when a derived program leaves vocabulary v1.1 -- a finding, not a nuisance."""


def assert_closed(prog: Program) -> None:
    escapes = []
    for name, op in prog.ops.items():
        if op.kind not in ("segmented_contraction", "scalar_map"):
            escapes.append(f"{name}: op kind {op.kind!r}")
        elif op.kind == "scalar_map" and op.fn not in SCALAR_FNS:
            escapes.append(f"{name}: scalar_map fn {op.fn!r}")
    if escapes:
        raise ClosureError(
            "derived program left vocabulary v1.1 -- this is a finding about the bet, not "
            "something to fix by widening the vocabulary:\n  " + "\n  ".join(escapes)
        )


# ----------------------------------------------------------------------------------------
# cotangent accumulation
# ----------------------------------------------------------------------------------------


class Cotangents:
    """Accumulates contributions per buffer.

    A buffer consumed by k ops receives the sum of k contributions. Contributions are held in
    a list and summed only when the buffer is popped. This is the classic AD bug site, so it
    is explicit here and covered by a dedicated diamond test rather than incidentally.
    """

    def __init__(self, prog: Program):
        self.prog = prog
        self._parts: dict[str, list[str]] = {}

    def add(self, buf: str, contribution: str) -> None:
        self._parts.setdefault(buf, []).append(contribution)

    def has(self, buf: str) -> bool:
        return bool(self._parts.get(buf))

    def pop(self, buf: str) -> str | None:
        """Materialise the sum of contributions, or None if there are none."""
        parts = self._parts.pop(buf, [])
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]

        # One contraction, one single-operand path per contribution: k consumers cost one op,
        # not k-1 chained adds.
        t = self.prog.type_of(parts[0])
        spec, sl = default_spec(t), full_slice(t)
        paths = [ContractionPath(1.0, f"{spec}->{spec}", (i,), (sl,), sl)
                 for i in range(len(parts))]
        return self.prog.contract(inputs=parts, out_type=t, paths=paths, hint="acc")


# ----------------------------------------------------------------------------------------
# transpose rules
# ----------------------------------------------------------------------------------------


def _vjp_contraction(prog: Program, op, cot: str, k: int,
                     zero_index: dict[str, str]) -> str | None:
    """VJP of a segmented_contraction w.r.t. input k. Emits one contraction, or None.

    Per path (docs/ir.md 3.1): swap the einsum output spec with operand k's spec, swap that
    operand's slice with the output slice, turn the forward gather on k into a scatter-add
    and the forward scatter into a gather of the cotangent. `coeff` is unchanged.

    Paths that do not read operand k contribute nothing.

    One case is not a plain index-map swap: a `none`-segment operand is *broadcast* over the
    segment axis in the forward direction, and the transpose of a broadcast is a **sum over
    that axis**. It is expressed with the vocabulary already has -- a scatter-add through an
    all-zeros index map into a length-1 buffer -- so no new op is needed (DECISIONS.md D18).
    """
    if not any(k in p.operands for p in op.paths):
        return None

    t_out = prog.type_of(op.inputs[k])
    inputs = list(op.inputs) + [cot]
    index_maps = list(op.index_maps) + [op.out_index_map]
    cot_idx = len(op.inputs)

    # A path may read operand k more than once -- `x*x` is one path with operands (0, 0).
    # The product rule then gives one contribution *per occurrence*, so this iterates over
    # every position j rather than taking `operands.index(k)`, which would find only the
    # first and silently halve the derivative (DECISIONS.md D21).
    occurrences = [(p, j) for p in op.paths
                   for j, kk in enumerate(p.operands) if kk == k]

    paths = []
    for p, j in occurrences:
        specs, out_spec = p.parse()
        new_specs = list(specs)
        new_specs[j] = out_spec                      # the cotangent takes k's slot
        new_out_spec = specs[j]

        new_operands = list(p.operands)
        new_operands[j] = cot_idx

        in_sl = [p.slices_for(i) for i in range(len(p.operands))] if p.in_slices else \
                [() for _ in p.operands]
        new_in_slices = list(in_sl)
        new_in_slices[j] = p.out_slice               # read the cotangent at the output slice
        new_out_slice = in_sl[j]                     # write at operand k's slice

        paths.append(ContractionPath(
            coeff=p.coeff,
            subscripts=",".join(new_specs) + "->" + new_out_spec,
            operands=tuple(new_operands),
            in_slices=tuple(new_in_slices),
            out_slice=tuple(new_out_slice),
        ))

    out_index_map = op.index_maps[k]                 # forward gather -> scatter-add
    if out_index_map is None and t_out.segment == "none":
        # transpose of a broadcast: sum over the contribution's segment axis
        contrib_seg = (prog.type_of(op.out_index_map).segment
                       if op.out_index_map is not None else op.out_type.segment)
        if contrib_seg != "none":
            out_index_map = zero_index[contrib_seg]

    return prog.contract(
        inputs=inputs, out_type=t_out, paths=paths,
        index_maps=index_maps, out_index_map=out_index_map, hint="d",
    )


def _vjp_scalar(prog: Program, op, cot: str, y: str, ones: str | None = None) -> str:
    """VJP of a scalar_map: cot * f'(x). Every f' stays inside the vocabulary (docs/ir.md 3.2).

    No constant-one buffer is needed anywhere. Writing sigmoid' as `y(1-y)` would require a
    broadcast `1` matching each operand's type, and the block applies sigmoid and silu to
    buffers of four different shapes -- so it would need a ones buffer per shape. Expanding
    the products instead (`y - y^2`, `s + xs - xs*s`) keeps every derivative a plain sum of
    contraction paths over buffers that already exist.
    """
    x, fn = op.inputs[0], op.fn
    t = prog.type_of(x)
    spec, sl = default_spec(t), full_slice(t)

    if fn == "exp":                       # f' = y
        return prog.mul(cot, y, hint="dexp")
    if fn == "sigmoid":                   # f' = y - y^2
        d = prog.contract(
            [y, y], t,
            [ContractionPath(1.0, f"{spec}->{spec}", (0,), (sl,), sl),
             ContractionPath(-1.0, f"{spec},{spec}->{spec}", (0, 1), (sl, sl), sl)],
            hint="dsig")
        return prog.mul(cot, d, hint="dsg")
    if fn == "silu":                      # f' = s + x*s - x*s*s,   s = sigmoid(x)
        s_ = prog.scalar(x, "sigmoid", hint="s")
        xs = prog.mul(x, s_, hint="xs")
        d = prog.contract(
            [s_, xs, s_], t,
            [ContractionPath(1.0, f"{spec}->{spec}", (0,), (sl,), sl),
             ContractionPath(1.0, f"{spec}->{spec}", (1,), (sl,), sl),
             ContractionPath(-1.0, f"{spec},{spec}->{spec}", (1, 2), (sl, sl), sl)],
            hint="dsilu")
        return prog.mul(cot, d, hint="dsl")
    if fn == "rsqrt":                     # f' = -1/2 * y * reciprocal(x)
        r = prog.scalar(x, "reciprocal", hint="rx")
        return prog.mul(cot, prog.mul(y, r, hint="yr"), coeff=-0.5, hint="drsq")
    if fn == "reciprocal":                # f' = -y^2
        return prog.mul(cot, prog.mul(y, y, hint="y2"), coeff=-1.0, hint="drec")
    if fn == "sin":                       # f' = cos(x)
        return prog.mul(cot, prog.scalar(x, "cos", hint="cx"), hint="dsin")
    if fn == "cos":                       # f' = -sin(x)
        return prog.mul(cot, prog.scalar(x, "sin", hint="sx"), coeff=-1.0, hint="dcos")
    if fn == "poly_envelope":             # f' = poly_envelope(order + 1)   (D16)
        return prog.mul(cot, prog.scalar(x, "poly_envelope", order=op.order + 1, hint="denv"),
                        hint="dpe")
    raise ClosureError(f"no VJP rule for scalar_map {fn!r}")


# ----------------------------------------------------------------------------------------
# the transform
# ----------------------------------------------------------------------------------------


def vjp(prog: Program, output: str, wrt: list[str], seed: str, ones: str | None = None,
        zero_index: dict[str, str] | None = None) -> dict[str, str]:
    """Extend `prog` in place with the VJP of `output` w.r.t. `wrt`, seeded by `seed`.

    `zero_index` maps a segment name to an all-zeros index buffer, used to transpose
    broadcasts of `none`-segment operands into segment sums.

    Returns {buffer -> cotangent buffer}; buffers the output does not depend on are absent.
    """
    zero_index = zero_index or {}
    cots = Cotangents(prog)
    cots.add(output, seed)

    # Walk the ops that existed before this call, in reverse. Ops appended by the transform
    # are part of the derived program, not something to differentiate again here.
    original = list(prog.topo())
    for name in reversed(original):
        if not cots.has(name):
            continue
        op = prog.ops[name]
        cot = cots.pop(name)
        if op.kind == "scalar_map":
            cots.add(op.inputs[0], _vjp_scalar(prog, op, cot, y=name))
        else:
            for k, src in enumerate(op.inputs):
                if isinstance(prog.type_of(src), IndexType):
                    continue
                g = _vjp_contraction(prog, op, cot, k, zero_index)
                if g is not None:
                    cots.add(src, g)

    grads = {w: cots.pop(w) for w in wrt}
    grads = {w: g for w, g in grads.items() if g is not None}
    assert_closed(prog)
    return grads


__all__ = ["vjp", "assert_closed", "ClosureError", "Cotangents"]
