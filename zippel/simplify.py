"""CSE and dead-code elimination on the segmented-polynomial IR.

Neither pass reassociates arithmetic, so both are **exact in floating point** — no tolerance
is involved and no result changes. That matters here: the derived programs are validated
against FP64 oracles, and a simplifier that perturbed values would make those comparisons
meaningless.

CSE is structural hashing over `(op kind, attributes, input ids)` in topological order
(docs/ir.md section 4). DCE is reachability from the program's declared outputs.
"""

from __future__ import annotations

from dataclasses import replace

from zippel.ir import IndexType, Op, Program


def _rewrite_inputs(op: Op, sub: dict[str, str]) -> Op:
    return replace(
        op,
        inputs=tuple(sub.get(i, i) for i in op.inputs),
        index_maps=tuple(sub.get(m, m) if m else m for m in op.index_maps),
        out_index_map=sub.get(op.out_index_map, op.out_index_map) if op.out_index_map else None,
    )


def cse(prog: Program) -> Program:
    """Common-subexpression elimination by structural hashing.

    Runs in definition order, which is already topological, so an op's inputs are canonical
    by the time it is hashed.
    """
    out = Program(inputs=dict(prog.inputs), outputs=prog.outputs, _counter=prog._counter)
    seen: dict[tuple, str] = {}
    sub: dict[str, str] = {}

    for name in prog.topo():
        op = _rewrite_inputs(prog.ops[name], sub)
        key = op.key()
        if key in seen:
            sub[name] = seen[key]
            continue
        seen[key] = name
        out.ops[name] = replace(op, name=name)

    out.outputs = tuple(sub.get(o, o) for o in prog.outputs)
    return out


def dce(prog: Program, keep: tuple[str, ...] | None = None) -> Program:
    """Drop ops no declared output depends on."""
    roots = set(keep or ()) | set(prog.outputs)
    live: set[str] = set()
    stack = [r for r in roots if r in prog.ops]
    live.update(stack)

    while stack:
        op = prog.ops[stack.pop()]
        for src in (*op.inputs, *(m for m in op.index_maps if m),
                    *( (op.out_index_map,) if op.out_index_map else () )):
            if src in prog.ops and src not in live:
                live.add(src)
                stack.append(src)

    out = Program(inputs=dict(prog.inputs), outputs=prog.outputs, _counter=prog._counter)
    for name in prog.topo():
        if name in live:
            out.ops[name] = prog.ops[name]
    return out


def simplify(prog: Program, keep: tuple[str, ...] | None = None) -> Program:
    """DCE then CSE then DCE: the second DCE collects ops orphaned by CSE's rewrites."""
    return dce(cse(dce(prog, keep)), keep)


# ----------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------


def op_counts(prog: Program) -> dict[str, int]:
    counts = {"total": len(prog.ops), "segmented_contraction": 0, "scalar_map": 0, "paths": 0}
    for op in prog.ops.values():
        counts[op.kind] += 1
        counts["paths"] += len(op.paths)
    return counts


def signatures(prog: Program) -> dict[tuple, int]:
    """Distinct op signatures and their multiplicity.

    The Phase 2 kernel-count proxy: ops sharing a signature can be served by one generated
    kernel with different pointers, so this is an upper bound on how many kernels hand
    scheduling has to write.
    """
    out: dict[tuple, int] = {}
    for op in prog.ops.values():
        out[op.signature()] = out.get(op.signature(), 0) + 1
    return out


def contraction_signatures(prog: Program) -> dict[tuple, int]:
    return {s: n for s, n in signatures(prog).items() if s[0] == "segmented_contraction"}


def kernel_families(prog: Program) -> dict[tuple, int]:
    """Coarser than `signatures`: slice *offsets* are abstracted away, extents kept.

    A generated kernel can take slice offsets as runtime arguments, so two ops that differ
    only in where they read and write are the same kernel. `signatures` counts them
    separately and is therefore an upper bound on how many kernels Phase 2 must write; this
    is the corresponding lower bound. The truth is in between and depends on how much the
    emitter parameterises, which is a Phase 2 decision.
    """
    def extent(sl):
        if sl.start is None and sl.stop is None:
            return None
        return (sl.stop or 0) - (sl.start or 0)

    out: dict[tuple, int] = {}
    for op in prog.ops.values():
        if op.kind != "segmented_contraction":
            continue
        key = (str(op.out_type),
               tuple(m is not None for m in op.index_maps),
               op.out_index_map is not None,
               tuple(sorted(
                   (p.subscripts, p.operands,
                    tuple(tuple(extent(x) for x in g) for g in p.in_slices),
                    tuple(extent(x) for x in p.out_slice))
                   for p in op.paths)))
        out[key] = out.get(key, 0) + 1
    return out


__all__ = ["cse", "dce", "simplify", "op_counts", "signatures", "contraction_signatures",
           "kernel_families"]
