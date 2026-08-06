"""The segmented-polynomial IR: types, ops, and program construction.

Spec: docs/ir.md. A program is an ordered, SSA-style list of pure ops, each producing exactly
one typed buffer. Shape checking happens at construction time -- an ill-typed program cannot
be built.

Vocabulary v1.1 is exactly two ops, `segmented_contraction` and `scalar_map`. That is the
point of the exercise: if the derivative of every op is expressible in the same two ops, the
energy-force-double-backward computation is differentiation-closed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Literal

Segment = Literal["node", "edge", "graph", "none"]
SEGMENTS: tuple[Segment, ...] = ("node", "edge", "graph", "none")

#: `scalar_map` functions. Widening this set is exactly what the closure test exists to catch.
SCALAR_FNS: tuple[str, ...] = (
    "exp", "sigmoid", "silu", "rsqrt", "reciprocal", "sin", "cos", "poly_envelope",
)


# ----------------------------------------------------------------------------------------
# types
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferType:
    """(segment axis, block-structured trailing axes, dtype)."""

    segment: Segment
    axes: tuple[tuple[str, int], ...] = ()
    dtype: str = "f64"

    def __post_init__(self):
        if self.segment not in SEGMENTS:
            raise ValueError(f"unknown segment {self.segment!r}; expected one of {SEGMENTS}")
        names = [n for n, _ in self.axes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate axis names in {self.axes}")
        for name, size in self.axes:
            if size <= 0:
                raise ValueError(f"axis {name!r} has non-positive size {size}")

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(s for _, s in self.axes)

    @property
    def rank(self) -> int:
        return len(self.axes)

    def __str__(self) -> str:
        return f"{self.segment}[{','.join(f'{n}:{s}' for n, s in self.axes)}]:{self.dtype}"


@dataclass(frozen=True)
class IndexType:
    """An integer buffer used as a gather source or scatter target. Never differentiated."""

    segment: Segment

    def __str__(self) -> str:
        return f"{self.segment}[]:idx"


# ----------------------------------------------------------------------------------------
# ops
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractionPath:
    """One multilinear term of a `segmented_contraction`.

    `operands` names which of the op's inputs this path reads, by index. That is what lets a
    single op express both a product (one path over two operands, `"ic,ic->ic"`) and a sum
    (two paths, each over one operand, `"ic->ic"`) -- see DECISIONS.md D17.

    `subscripts` ranges over *trailing* axes only; the segment axis is batched, never
    contracted. `in_slices[j]` slices the trailing axes of `operands[j]`.
    """

    coeff: float
    subscripts: str
    operands: tuple[int, ...]
    in_slices: tuple[tuple[slice, ...], ...] = ()
    out_slice: tuple[slice, ...] = ()

    def parse(self) -> tuple[list[str], str]:
        lhs, arrow, rhs = self.subscripts.partition("->")
        if not arrow:
            raise ValueError(f"subscripts must be explicit (contain '->'): {self.subscripts!r}")
        return [s.strip() for s in lhs.split(",")], rhs.strip()

    def slices_for(self, j: int) -> tuple[slice, ...]:
        return self.in_slices[j] if self.in_slices else ()

    def key(self) -> tuple:
        def sl(s: slice) -> tuple:
            return (s.start, s.stop, s.step)
        return (self.coeff, self.subscripts, self.operands,
                tuple(tuple(sl(s) for s in grp) for grp in self.in_slices),
                tuple(sl(s) for s in self.out_slice))


@dataclass(frozen=True)
class Op:
    kind: str
    inputs: tuple[str, ...]
    out_type: BufferType
    index_maps: tuple[str | None, ...] = ()
    out_index_map: str | None = None
    paths: tuple[ContractionPath, ...] = ()
    fn: str | None = None
    order: int = 0
    name: str = ""

    def key(self) -> tuple:
        """Structural hash key: op kind + attributes + input ids (docs/ir.md section 4)."""
        return (self.kind, self.inputs, str(self.out_type), self.index_maps,
                self.out_index_map, tuple(p.key() for p in self.paths), self.fn, self.order)

    def signature(self) -> tuple:
        """Shape-and-structure identity, ignoring *which* buffers feed it.

        The Phase 2 kernel-count proxy: two ops with the same signature can be served by one
        generated kernel with different pointers. Path coefficients are dropped because they
        are kernel constants, not shape.
        """
        if self.kind == "scalar_map":
            return ("scalar_map", self.fn, self.order, str(self.out_type))
        return ("segmented_contraction", str(self.out_type),
                tuple(m is not None for m in self.index_maps),
                self.out_index_map is not None,
                tuple(p.key()[1:] for p in self.paths))


# ----------------------------------------------------------------------------------------
# program
# ----------------------------------------------------------------------------------------


@dataclass
class Program:
    inputs: dict[str, BufferType | IndexType] = field(default_factory=dict)
    ops: dict[str, Op] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    _counter: itertools.count = field(default_factory=lambda: itertools.count())

    def add_input(self, name: str, type_: BufferType | IndexType) -> str:
        if name in self.inputs or name in self.ops:
            raise ValueError(f"buffer {name!r} already defined")
        self.inputs[name] = type_
        return name

    def type_of(self, name: str) -> BufferType | IndexType:
        if name in self.inputs:
            return self.inputs[name]
        if name in self.ops:
            return self.ops[name].out_type
        raise KeyError(f"undefined buffer {name!r}")

    def _fresh(self, hint: str) -> str:
        return f"{hint}_{next(self._counter)}"

    def contract(self, inputs, out_type, paths, index_maps=None, out_index_map=None,
                 hint="c") -> str:
        inputs = list(inputs)
        index_maps = list(index_maps or [None] * len(inputs))
        if len(index_maps) != len(inputs):
            raise ValueError("index_maps must have one entry per input")
        _check_contraction(self, inputs, index_maps, out_index_map, out_type, paths)
        name = self._fresh(hint)
        self.ops[name] = Op(
            kind="segmented_contraction", inputs=tuple(inputs), out_type=out_type,
            index_maps=tuple(index_maps), out_index_map=out_index_map,
            paths=tuple(paths), name=name,
        )
        return name

    def scalar(self, x: str, fn: str, order: int = 0, hint: str | None = None) -> str:
        if fn not in SCALAR_FNS:
            raise ValueError(f"{fn!r} is outside vocabulary v1.1 {SCALAR_FNS}")
        if order and fn != "poly_envelope":
            raise ValueError("only poly_envelope carries a derivative order (D16)")
        t = self.type_of(x)
        if not isinstance(t, BufferType):
            raise TypeError(f"scalar_map on a non-value buffer {x!r}")
        name = self._fresh(hint or fn)
        self.ops[name] = Op(kind="scalar_map", inputs=(x,), out_type=t, fn=fn,
                            order=order, name=name)
        return name

    # -- convenience builders -------------------------------------------------------------

    def add(self, a: str, b: str, ca: float = 1.0, cb: float = 1.0, hint="add") -> str:
        """ca*a + cb*b as ONE contraction with two single-operand paths."""
        t = self.type_of(a)
        spec = default_spec(t)
        sl = full_slice(t)
        return self.contract(
            inputs=[a, b], out_type=t,
            paths=[ContractionPath(ca, f"{spec}->{spec}", (0,), (sl,), sl),
                   ContractionPath(cb, f"{spec}->{spec}", (1,), (sl,), sl)],
            hint=hint,
        )

    def mul(self, a: str, b: str, coeff: float = 1.0, hint="mul") -> str:
        """Elementwise product: one path over both operands."""
        t = self.type_of(a)
        spec = default_spec(t)
        sl = full_slice(t)
        return self.contract(
            inputs=[a, b], out_type=t,
            paths=[ContractionPath(coeff, f"{spec},{spec}->{spec}", (0, 1), (sl, sl), sl)],
            hint=hint,
        )

    def scale(self, a: str, coeff: float, hint="scale") -> str:
        t = self.type_of(a)
        spec = default_spec(t)
        sl = full_slice(t)
        return self.contract(
            inputs=[a], out_type=t,
            paths=[ContractionPath(coeff, f"{spec}->{spec}", (0,), (sl,), sl)],
            hint=hint,
        )

    def consumers(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, op in self.ops.items():
            for src in op.inputs:
                out.setdefault(src, []).append(name)
        return out

    def topo(self) -> list[str]:
        return list(self.ops)

    def __str__(self) -> str:
        lines = [f"  in  {n}: {t}" for n, t in self.inputs.items()]
        for name, op in self.ops.items():
            if op.kind == "scalar_map":
                suffix = f"[order={op.order}]" if op.order else ""
                lines.append(f"  {name} = {op.fn}{suffix}({op.inputs[0]})  : {op.out_type}")
            else:
                gm = ",".join(m or "-" for m in op.index_maps)
                lines.append(
                    f"  {name} = contract({','.join(op.inputs)}) gather[{gm}] "
                    f"scatter[{op.out_index_map or '-'}] paths={len(op.paths)}  : {op.out_type}"
                )
        lines.append(f"  out {', '.join(self.outputs)}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# helpers + validation
# ----------------------------------------------------------------------------------------


def default_spec(t: BufferType) -> str:
    return "".join(chr(ord("a") + i) for i in range(t.rank))


def full_slice(t: BufferType) -> tuple[slice, ...]:
    return tuple(slice(None) for _ in t.axes)


def sliced_sizes(t: BufferType, slices: tuple[slice, ...]) -> tuple[int, ...]:
    sizes = t.sizes
    if not slices:
        return sizes
    return tuple(len(range(*slices[i].indices(s))) if i < len(slices) else s
                 for i, s in enumerate(sizes))


def _check_contraction(prog, inputs, index_maps, out_index_map, out_type, paths) -> None:
    if not paths:
        raise ValueError("a segmented_contraction needs at least one path")

    for k, (buf, imap) in enumerate(zip(inputs, index_maps)):
        t = prog.type_of(buf)
        if not isinstance(t, BufferType):
            raise TypeError(f"operand {k} ({buf!r}) is an index buffer, not a value")
        if imap is not None:
            if not isinstance(prog.type_of(imap), IndexType):
                raise TypeError(f"index_maps[{k}] = {imap!r} is not an index buffer")
            if t.segment == "none":
                raise TypeError(f"cannot gather from a 'none'-segment buffer {buf!r}")
        elif t.segment not in (out_type.segment, "none"):
            if out_index_map is None:
                raise TypeError(
                    f"operand {k} ({buf!r}) has segment {t.segment!r} but the result is "
                    f"{out_type.segment!r}, with no index map and no scatter"
                )

    if out_index_map is not None and not isinstance(prog.type_of(out_index_map), IndexType):
        raise TypeError(f"out_index_map {out_index_map!r} is not an index buffer")

    for p in paths:
        specs, out_spec = p.parse()
        if len(specs) != len(p.operands):
            raise ValueError(
                f"path has {len(specs)} subscript groups but names {len(p.operands)} operands: "
                f"{p.subscripts!r}"
            )
        if p.in_slices and len(p.in_slices) != len(p.operands):
            raise ValueError("in_slices must have one entry per operand named by the path")
        for j in p.operands:
            if not 0 <= j < len(inputs):
                raise ValueError(f"path names operand {j}, but the op has {len(inputs)} inputs")

        # docs/ir.md 2.1: a summed index must appear in >= 2 operands, else its transpose
        # would need a broadcast, which einsum cannot express as a contraction.
        counts: dict[str, int] = {}
        for spec in specs:
            for ch in set(spec):
                counts[ch] = counts.get(ch, 0) + 1
        for ch, n in counts.items():
            if ch not in out_spec and n < 2:
                raise ValueError(
                    f"path {p.subscripts!r} sums index {ch!r} appearing in only one operand; "
                    "its transpose is not expressible in the vocabulary (docs/ir.md 2.1)"
                )
        for ch in out_spec:
            if ch not in counts:
                raise ValueError(
                    f"path {p.subscripts!r} produces index {ch!r} that no operand supplies"
                )

        # trailing-axis extents must agree between the path and the declared output type
        want = sliced_sizes(out_type, p.out_slice)
        if len(out_spec) != len(want):
            raise ValueError(
                f"path output spec {out_spec!r} has rank {len(out_spec)} but the sliced result "
                f"has rank {len(want)} ({out_type})"
            )
        extent: dict[str, int] = {}
        for j, spec in zip(p.operands, specs):
            t = prog.type_of(inputs[j])
            sizes = sliced_sizes(t, p.slices_for(p.operands.index(j)))
            if len(spec) != len(sizes):
                raise ValueError(
                    f"operand {j} spec {spec!r} has rank {len(spec)} but its sliced shape is "
                    f"{sizes} ({t})"
                )
            for ch, size in zip(spec, sizes):
                if extent.setdefault(ch, size) != size:
                    raise ValueError(
                        f"index {ch!r} has inconsistent extents {extent[ch]} and {size} "
                        f"in path {p.subscripts!r}"
                    )
        for ch, size in zip(out_spec, want):
            if extent.get(ch, size) != size:
                raise ValueError(
                    f"output index {ch!r} extent {extent[ch]} does not match the result's {size}"
                )


__all__ = [
    "BufferType", "IndexType", "ContractionPath", "Op", "Program",
    "SCALAR_FNS", "SEGMENTS", "default_spec", "full_slice", "sliced_sizes",
]
