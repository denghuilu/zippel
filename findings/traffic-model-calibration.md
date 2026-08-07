# Calibrating the traffic model: what it predicts, and what it does not

**Status: stands.** The D27 gate is open for T2 and closed for T1, and that split is the result.

## Why the instrument is not ncu

The work order asks for ncu-measured DRAM traffic. **Neither `ncu` nor CUPTI is installed on this
system** (`nvidia-cuda-cupti-cu13` resolves only to a 0.0.1 stub; `libcupti.so` does not load).
`dcgmi` is present, so the substitute is DCGM's `dram_active` counter (field 1005) — the fraction
of cycles the memory interface is transferring — giving `bytes ≈ dram_active × bandwidth × time`.

That is coarser than exact byte counters, so **the instrument is calibrated before it is trusted
about anything else**, against `copy_` at three sizes where the traffic is known exactly (2N).

| MiB copied | known B/iter | raw B/iter | raw error |
|---|---|---|---|
| 256 | 536,870,912 | 451,206,199 | −16.0 % |
| 512 | 1,073,741,824 | 899,123,357 | −16.3 % |
| 1024 | 2,147,483,648 | 1,786,097,811 | −16.8 % |

The bias is systematic, and the fitted constant is **K = 4.777e12** against the 4.0 TB/s nominal
peak. Residual after the fit: **0.4–3.1 %** across runs, so the instrument is linear across the
range probed and the constant is trustworthy in service.

**K is not a bandwidth** — see `dcgm-bandwidth-constant.md`, which corrects an earlier claim here
that it was. This device is a GH200 96GB HBM3 part with a 4.0 TB/s nominal peak and a measured
3.6 TB/s achieved copy rate; K exceeds both, so it must be an effective constant (achieved
bandwidth ÷ instrument response) rather than a physical rate. **The residual distinguishes
nothing between those readings**: a constant fitted to reproduce known traffic will fit it well
whether the peak was underestimated or the counter under-reports. Only device identity and an
independent bandwidth measurement discriminate.

## What the model got wrong, three times

| model | wigner_chain (T1) | radial_lin0 (T2) | radial_stage2 (T2) |
|---|---|---|---|
| charge every live-in in full | **−28.1 %** | +6.5 % | +5.7 % |
| charge the element fraction actually read | **+90.2 %** | +2.3 % | +1.3 % |
| charge 32-byte sector occupancy | **+53.6 %** | +1.1 % | −0.1 % |
| charge 128-byte line occupancy | **+26.9 %** | +2.0 % | −2.8 % |

Each row is a real correction, and each was forced by measurement rather than reasoning:

1. Charging full buffers ignores that the emitter loads **only the elements its terms reference**
   — the sparsity that makes T1 worth doing shows up in the bytes.
2. Charging the element fraction is wrong in the other direction: DRAM does not move elements. A
   `[9,9]` FP64 block whose l=1 sub-block is read touches 9 of 81 elements but far more than 9/81
   of the bytes.
3. Sector occupancy (32 B, the L1↔L2 granularity) is the wrong granularity for DRAM traffic.
4. Line occupancy (128 B, the L2↔DRAM granularity) is the right one, and it is what the model
   uses.

## Where it stands

**T2 — OPEN at 2.8 %.** Dense channel access is predicted essentially exactly. The model may
drive fusion and template decisions for these groups.

**T1 — CLOSED at 26.9 %.** Sparse strided reads are still *under*-predicted. The per-`l` Wigner
blocks are charged 17 %/33 %/67 % of their buffers, and the hardware moves more than that. The
residual is almost certainly partial L2 line reuse across the 648-byte per-edge stride: whether a
line fetched for edge `e` survives until its other elements are wanted depends on how much of the
168 MB buffer passes through a 60 MB L2 in between, which compulsory-traffic accounting cannot
express by construction.

I stopped iterating here deliberately. Two more granularity constants would have brought T1
inside 20 %, but that would have been **fitting the constant to the kernel**, not modeling the
hardware — and the whole point of D27 is that the objective function must not be allowed to
select its own inputs. A model that is honestly closed for one template is more useful than one
tuned until it agrees.

## Consequence

`codegen.traffic.calibrated(template)` is a per-template gate rather than a global verdict,
because a single answer would either forbid the well-supported T2 use or permit the unsupported
T1 one. S2 grouping decisions may use the model for T2-class groups and must not use it for
sparse-read groups until the L2-reuse term is modeled and re-measured.
