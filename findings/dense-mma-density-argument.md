# Dense MMA tiles and the block diagonal: right call, wrong reason

**Status: MECHANISM CORRECTED.** The decision stands; the reason given for it does not.

## The decision (D22), which is upheld

Phase 2 emits SIMT small-tile kernels and does not attempt a dense-WGMMA design. That is still
the right call, and nothing below reopens it.

## The reason I gave, which was wrong

D22 justified it on **structural density**. A Wigner rotation at lmax=2 is block-diagonal with 35
nonzeros of 81, and a dense MMA tile must pad `nc = 9` to an MMA-friendly extent, so the tile is
3.4 % occupied at pad-32 or 13.7 % at pad-16. The argument: a dense tile "pays quadratically for
the block-diagonal zeros", and lmax=2 is the worst point on that curve.

The arithmetic was right. The mechanism was not.

## What the measurement shows

`bench/template_crossover.py`, GH200, batched `[E,nc,nc] @ [E,nc,C]`, E=65536, C=128, bf16:

| lmax | density | dense-padded | dense-exact | per-degree blocks | structure wins by |
|---|---|---|---|---|---|
| 1 | 3.9 % | 0.257 ms | 0.631 ms | 0.561 ms | 0.46× |
| **2** | **13.7 %** | **0.263 ms** | 0.543 ms | 1.002 ms | **0.26×** |
| 3 | 32.8 % | 0.263 ms | 0.262 ms | 1.518 ms | 0.17× |
| 4 | 16.1 % | 0.362 ms | 0.851 ms | 2.218 ms | 0.16× |
| 6 | 11.1 % | 0.767 ms | 1.289 ms | 3.853 ms | 0.20× |
| 8 | 10.5 % | 1.236 ms | 2.298 ms | 5.809 ms | 0.21× |

Two facts, both against the density argument:

1. **Decomposing the block-diagonal into per-degree GEMMs loses at every lmax** (0.16–0.87×).
   Exploiting the structure by splitting launches costs far more in small-GEMM inefficiency than
   the skipped zeros save.
2. **Padding is cheaper than not padding.** At lmax=2 the padded tile performs `(16/9)² = 3.2×`
   the multiply-adds of the exact one and runs **2.1× faster**, because 16 is an MMA-friendly
   extent and 9 is not.

If structural zeros were the binding cost, (2) is impossible. Density does not predict dense-MMA
cost at these extents.

## What was actually true

FlashSO2's postmortem — the source D22 cited — does not say the zeros cost. It says the gather
cannot become a bulk copy, that WGMMA fragment layouts fight coalescing, and that L1TEX issue
throughput is the top limiter at 75–84 % of peak. Its own words about the arithmetic waste:

> Idle compute today, but it caps any future compute-bound version.

That is a statement that the zeros are **not** binding. I restated a cost as binding that its own
source called non-binding, and the density arithmetic — which is correct as arithmetic — made the
wrong mechanism look quantified.

## Why the distinction is load-bearing

Acting rightly for a wrong reason generalises wrongly. The density argument, taken forward, would
have said: *exploit the block structure wherever density is low*. The measurement says the
opposite — exploiting it by decomposition loses everywhere, and the only profitable form is
eliding zero terms **inside a single kernel**, where the saving is loads and stores rather than
FLOPs. That is now the selection rule in `docs/templates.md`, and it is a constraint ("never
decompose") rather than a density threshold.

Had the two been kept together, this measurement would have read as confirming D22 — the dense
tile is indeed a bad idea — and the false mechanism would have survived into every later
template decision. Which is the reason this status category exists.

## Consequences recorded

* D24 in `DECISIONS.md` — the measurement and the correction.
* `docs/templates.md` §2 — the rule stated as a constraint, with the unmeasured third arm (a
  fused sparse tile with channels on the warp) named as unmeasured rather than extrapolated.
* D27 — the operational consequence: if bytes are the mechanism, Phase 2 needs a calibrated
  **byte** model as its objective, not a term count.
