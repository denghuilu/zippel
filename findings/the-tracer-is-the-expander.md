# The tracer is the expander — route C is not expressible in the CuTe DSL

> **Status: CONFIRMED by probe.** Route C was pre-registered in D77 as the only route that could
> dissolve the `terms ∝ E_c` compile ceiling. It cannot, and the reason is an API reality rather
> than a performance result. Filed *after* running the probe, not before — the conclusion was
> directed twice while unsupported, and refusing to file it unmeasured is what made the measurement
> happen.

## What route C was

Edge batching needs `E_c` accumulator sets live across a loop over edges — that interleaving is
what captures the weight reuse (D76). Three places can expand that loop:

| route | source text | traced IR | who expands | pays |
|---|---|---|---|---|
| A — emitter-text unroll | `N × E_c` terms | large | the emitter | parse + trace + backend |
| B — `range_constexpr` | `N` terms | large | the DSL tracer | trace + backend |
| **C — `range` + `LoopUnroll`** | `N` terms | **small** | **the backend** | backend only |

Only C keeps the IR small, so only C could relieve R9 (compile cost as the scaling limit), which
D84 promoted from a note to a hard constraint at **56 minutes for one kernel at `E_c`=4**.

## The probe

Three minimal kernels, `E = 4`, one accumulator per iteration
(`scratchpad/route_c_probe.py`):

```
  range_constexpr + named regs : COMPILED   y[:5]=[1.0, 1.0, 1.0, 1.0, 1.0]
   range(dynamic) + scalar acc : COMPILED   y[:5]=[4.0, 0.0, 0.0, 0.0, 4.0]
  range(dynamic) + indexed acc : FAILED
      DSLRuntimeError: '<class ...arith.ArithValue>' object cannot be interpreted as an integer
```

## What it means

* **`range_constexpr` works and expands at trace time.** The loop variable is a Python `int`
  during tracing, so per-iteration named registers are fine — and the emitted IR is therefore the
  *same size* as route A's. B ≡ A in IR, exactly as D77's table predicted.
* **`range` (dynamic) works for a single loop-carried scalar.** A real loop exists in the IR.
* **`range` (dynamic) cannot hold per-iteration state.** The loop variable is an
  `arith.ArithValue` — an MLIR SSA value, not an integer — so it cannot index a Python container.
  There is no way to say "accumulator number `e`" when `e` is only known at run time.

**So the structure edge batching requires — a dynamic loop carrying `E_c` distinct accumulators —
is not expressible.** You may have a small IR *or* per-iteration registers, never both. The
expansion that creates the registers is the same expansion that creates the IR, and the tracer
performs it. **The tracer is the expander.**

## Consequences

1. **Route C is dead, and the compile ceiling does not dissolve.** `terms ∝ E_c` stands as law.
2. **A and B differ only in Python-side parse cost**, which D81 measured at **0.05 s of a 286 s
   compile — 0.02 %**. So B is not worth implementing either: it saves a rounding error and the
   collapse D80 pre-registered is confirmed.
3. **The compile lever moves to what remains**: splitting the dominant group via a lower
   `max_volume` (a trade, since it changes fusion), and arm/job-level parallelism (bounded by
   independent jobs, not cores — D84's 1.6× kernel-level ceiling).
4. **Route D** — if it is ever specified — must not assume a dynamic edge loop with per-edge state.

## The methodological note

This finding was directed as settled twice before any probe existed, and declining to file it is
what produced the evidence. The probe cost minutes. **A conclusion that is probably true is still
not a measurement**, and `findings/` is where this project keeps the difference — the same
distinction that killed the `CUTE_DSL_CACHE_DIR` claim, where a plausible conclusion from a search
that could not have found the thing reached a report twice.

One incidental repeat worth recording: the probe first failed under a heredoc with *"DSL does not
support REPL mode, save the function to a file instead."* That is the **third** time this session.
The DSL parses its decorated functions from source, so anything piped to `python -` cannot work.
It is now written down rather than re-learned.
