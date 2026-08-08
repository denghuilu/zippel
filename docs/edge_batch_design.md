# Lever (a): edge-batched CTAs with k-tiled smem weight staging — design and pre-registration

Written **before implementation and before any number**, per the standing rule. The arithmetic
below is all derived from measured quantities (D59, D64, D68) and commits the sweep to a frame.

## Why this is not the arm that was struck

D46 established, and D60 struck staging on, the fact that `conv1_90`'s weight reuse is **entirely
cross-CTA** — thread `c` of every CTA reads the same weight slice, and shared memory cannot reach
across CTAs. D64 then measured that the *ideal* traffic **is** that re-read: 1.366 MB per CTA
measured against 1.3194 MB predicted as "every weight once, plus this edge's row", 3.6 % apart,
weights 99.3 % of it.

Edge batching changes the mechanism rather than re-litigating the verdict. With `E_c` edges per
CTA, the weight read is amortised over `E_c` edges **inside one CTA**, which is precisely the
intra-CTA reuse smem exists to capture. The staging machinery is revived legally because its
precondition now holds.

## The shape

Block stays **128 threads, one channel each** — not `128 × E_c`, which would exceed the 1024-thread
block limit for `E_c > 8` and is the obvious wrong turn. Grid becomes `ceil(n_seg / E_c)`. Each
thread walks `E_c` edges.

    for k_tile in tiles(contraction_axis):
        stage weights[k_tile] into smem     # cooperative, coalesced, padded (D48)
        barrier
        for e_local in 0 .. E_c-1:
            acc[e_local] += contribution of k_tile to edge e_local
        barrier
    for e_local in 0 .. E_c-1:
        store acc[e_local]

**The unavoidable tension, stated now.** Weight reuse across edges requires the *edge* loop inside
the *k-tile* loop, which requires **`E_c` accumulator sets live simultaneously**. Putting edges
outside instead would keep registers flat but re-stage the weights per edge and capture no reuse
at all. There is no third arrangement; the sweep is therefore a **byte-saving versus
register/occupancy trade**, and the interesting question is where it turns over.

**Tile order preserves summation order.** The emitted terms already run in increasing contraction
index (`_s7_0 = … w[c*257+0] + … w[c*257+1] + …`), and `_chunked_sum` accumulates left to right.
Tiling on contiguous ranges of that axis therefore keeps the arithmetic order, so **bit-equality
against the E_c=1 kernel is the correctness bar**, exactly as in the factorial — not a loosened
tolerance.

## Pre-registered quantitative model

**Weight demand ∝ 1/E_c; edge-stream term constant.** Per-edge ideal traffic:

    bytes(E_c) = 1.3107 MB / E_c  +  0.0087 MB

| `E_c` | predicted bytes/edge | vs `E_c`=1 | regs/thread ≈ 32 + 9(E_c−1) | blocks/SM | occupancy |
|---|---|---|---|---|---|
| 1 | 1.3194 MB | 1.00× | 32 | 16 | 100 % |
| 4 | 0.3364 MB | **3.92×** | 59 | 8 | 50 % |
| 8 | 0.1725 MB | **7.65×** | 95 | 5 | 31 % |
| 16 | 0.0906 MB | **14.6×** | 167 | 3 | 19 % |
| 32 | 0.0497 MB | 26.6× | 311 | — | **exceeds the 168-register budget** |

The output is 9 f32 per thread per edge (1 140.27 MiB / 259 474 edges / 128 channels), so
accumulators scale as `9·E_c`. **`E_c`=32 is predicted to be refused by the register guard before
it is ever launched** — recorded as a prediction, so the guard firing is a confirmation and not a
surprise.

**Occupancy is the counter-pressure and it is not optional.** Achieved occupancy falls 100 → 50 →
31 → 19 %. `B_smem` demonstrated what starvation does: at 6.17 % it sustained 0.68 TB/s against
the baseline's 3.13 (17 % of peak vs 78 %). Interpolating that crudely, bandwidth is expected to
hold near peak at 50 % occupancy and to degrade below ~30 %.

**Smem budget: ≤ 14 KiB, not ≤ 48 KiB.** The 48 KiB figure avoids the dynamic-smem opt-in but
**does not preserve occupancy**: at 48 KiB the smem limit is 4 blocks/SM, below the register limit
of 16, and smem becomes the binding constraint — the same mechanism that killed `B_smem`. Keeping
the register limit binding needs `228 / 16 ≈ 14 KiB`. The k-tile is sized to that, and the
resulting tile width is reported.

**Predicted wall-clock, as intervals with their assumptions.** Assuming (i) the kernel stays
bandwidth-bound, (ii) achieved bandwidth degrades with occupancy as sketched above, (iii) no
spill, (iv) staging overhead small relative to the byte saving:

| `E_c` | predicted | assumption that would break it |
|---|---|---|
| 4 | **150–250 ms** | bandwidth already falls off at 50 % occupancy |
| 8 | **100–200 ms** | ditto, more so |
| 16 | **70–180 ms** | 19 % occupancy starves MLP as `B_smem` did |
| 32 | refused | — |

against `A_transpose`'s 582.023 ms. Intervals, not point estimates, because assumption (ii) is an
interpolation between two measurements and nothing more.

## Interpretation cells, fixed in advance

* **byte-proportional** — times track the `1/E_c` byte curve within the stated intervals.
  → The bytes law governs this transformation too. Adopt the largest feasible `E_c` and re-measure
  the composition. **D47 fires**: any of these crosses the 3× margin, so that re-measure is N=5.
* **plateau** — gains saturate at some `E_c*` well above the byte curve.
  → A second bottleneck binds. Identify it *before* extending the sweep. If the plateau sits above
  eager's per-kernel µs/edge, the D67 MMA door opens; if below, it does not.
* **antagonistic** — larger `E_c` is *slower* than smaller.
  → Occupancy loss outruns byte saving. Record the crossover `E_c`, cap the rule there, and the
  result is a measured statement about the trade rather than a failure. **This cell is written
  because the last factorial had no home for its antagonistic outcome and I had to admit that
  mid-result.**
* **all-null** — no `E_c` beats `E_c`=1 despite bytes provably falling.
  → The bandwidth-bound reading is wrong, or the byte model does not apply to this transformation.
  That would put D59's regime finding itself in question and sends the next step back to counters,
  not forward to another arm.

## Guards active throughout

* register bound (`inlined_live_upper_bound` × `E_c` accumulator sets) — refusal is data
* occupancy accounting against the 16-block register limit, reported per `E_c`
* ordering bound + **bit-equality against `E_c`=1**, since tile order preserves summation order
* planted-fault check, per standing practice, to prove the bar discriminates
