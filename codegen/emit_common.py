"""The emission substrate every template shares. Extracted per D70a; gated by D73.

Three emitters (`emit.py` T1, `emit_tile.py` T2, `emit_reduce.py` T3) grew by copying. The copies
then drifted in the ways copies do, and the **keyed-by-identity bug family** — four instances, in
`vjp.py`'s `operands.index(k)`, `ir.py`'s validator, `emit_reduce.py`'s buffer→index dict and
`emit_tile.py`'s `assigns.index(a)` — appeared in exactly the places that had been duplicated
rather than shared. A bug class that recurs across copies is a statement about the copies.

**What was actually duplicated, measured before extracting:**

* `_chunked_sum` and `CHUNK = 48` — **byte-identical in all three files**.
* `_sym` — identical in T1 and T3; T2 differs only in rendering a channel component as `_c`.
* `_ref` — all three emit `m_<buf>[<lead>, <trailing…>]` and differ *only* in how `<lead>` is
  computed (plain / gathered / plain) and how trailing components are rendered.

So the substrate is the shape, and the hooks are the two genuinely per-template decisions:
**how a leading segment coordinate is formed**, and **how an index component is rendered**. Any
future template supplies those two and inherits the rest.

**Bit-exactness is the whole contract.** Every function here reproduces the string its callers
previously built, character for character. The refactor gate (D73) is what proves it.
"""

from __future__ import annotations

from zippel.ir import Program

#: CuTe DSL type names, keyed by the project's dtype tags.
DTYPE = {"f64": "Float64", "f32": "Float32"}

#: Registers a thread may hold before the guards refuse a group. See `codegen/bounds.py` and D26.
REGISTER_BUDGET = 168

#: Terms per emitted statement. One `a + b + … + z` of thousands of terms overflows CPython's AST
#: recursion limit while the DSL parses the generated module (the SO(2) conv group has 5 123).
#: Chunking is strictly left-to-right, so the summation order — and therefore the ordering bound —
#: is unchanged by it.
CHUNK = 48


def render_plain(i) -> str:
    """Default index rendering: the literal coordinate. T1 and T3."""
    return str(i)


def sym(buf: str, idx: tuple, render=render_plain) -> str:
    """Register name for a value, named by its index.

    T2 passes a renderer that maps the channel component to `_c`, because two factors differing
    only in channel offset are different *memory reads* and not different registers — the offset
    belongs in the reference, never in the name.
    """
    return f"v_{buf}" + "".join(f"_{render(i)}" for i in idx)


def lead_plain(prog: Program, buf: str) -> str:
    """Default leading coordinate: the segment index, or `0` for a `none`-segment buffer.

    Every buffer carries a leading segment coordinate, matching the interpreter's convention that
    a `none`-segment buffer is stored with a length-1 segment axis
    (`zippel/interp.py:segment_length`). So a static `[9,9]` operand is `m_jd[0, i, j]`, and the
    two conventions cannot drift apart silently.
    """
    return "e" if prog.type_of(buf).segment != "none" else "0"


def ref(prog: Program, buf: str, idx: tuple, lead: str | None = None,
        render=render_plain, perm: tuple | None = None) -> str:
    """A gmem reference `m_<buf>[<lead>, <trailing…>]`.

    `lead` overrides the default segment coordinate — T3 passes `m_<gather>[e]` for a gathered
    read, which is one indirection and nothing more. `perm` permutes the trailing axes; it is a
    *layout* change applied identically to the emitted index order and to the tensor handed in at
    launch, and the generated module publishes it as `TRANSPOSE` so the launch side reads it back
    instead of recomputing it (D52, which cost an illegal memory access).
    """
    order = idx if perm is None else tuple(idx[k] for k in perm)
    coords = [lead if lead is not None else lead_plain(prog, buf)] + [render(i) for i in order]
    return f"m_{buf}[{', '.join(coords)}]"


def chunked_sum(target: str, parts: list[str], uid: int) -> list[str]:
    """Left-to-right accumulation in blocks of `CHUNK`. Order-preserving by construction."""
    if len(parts) <= CHUNK:
        return [f"{target} = " + " + ".join(parts)]
    lines, acc = [], None
    for k in range(0, len(parts), CHUNK):
        piece = parts[k:k + CHUNK]
        name = f"_s{uid}_{k // CHUNK}"
        lines.append(f"{name} = " + " + ".join(([acc] if acc else []) + piece))
        acc = name
    lines.append(f"{target} = {acc}")
    return lines


def metadata_block(segment: str, template: str, esha: str, depth, exact: bool,
                   notes: str = "", after_segment: str = "", after_sha: str = "",
                   extra: str = "") -> str:
    """The contract a generated kernel ships about itself. **Assembly is shared; prose is not.**

    Not decoration: `build_kernel` refuses to load a module missing any required field, and
    `EMITTER_SHA` makes a stale artefact from a previous emitter version a load-time error rather
    than a silently wrong result. `SEGMENT` is here because a node-rooted group launched with the
    edge count segfaults, which is how the field came to be declared.

    Three slots carry template-specific text without duplicating the assembly: `notes` above the
    fields, `after_segment` between `SEGMENT` and `TEMPLATE` (T3's `DRIVING_SEGMENT`, which differs
    from `SEGMENT` exactly when the kernel scatters), and `after_sha` between `EMITTER_SHA` and
    `REDUCTION_DEPTH` (T2's `TRANSPOSE`/`STAGED`). They exist so the **shared thing is the field
    set, order and formatting**, while each template keeps its own explanation of its own
    contract. Unifying
    the prose too would change the emitted text for no gain and would forfeit the byte-identity
    that makes the D73 gate an equality check rather than a judgement call.
    """
    lines = ([notes] if notes else []) + [f'SEGMENT = "{segment}"']
    if after_segment:
        lines.append(after_segment)
    lines += [f'TEMPLATE = "{template}"', f'EMITTER_SHA = "{esha}"']
    if after_sha:
        lines.append(after_sha)
    lines += [f'REDUCTION_DEPTH = {depth}', f'EXACT = {exact}']
    if extra:
        lines.append(extra)
    return "\n".join(lines)


__all__ = ["DTYPE", "REGISTER_BUDGET", "CHUNK", "render_plain", "sym", "lead_plain",
           "ref", "chunked_sum", "metadata_block"]
