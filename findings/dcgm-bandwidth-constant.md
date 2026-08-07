# The 4.777 TB/s constant is not a bandwidth

**Status: MECHANISM CORRECTED.** The calibration fit stays in service unchanged. The story I told
about what it recovered was wrong.

## What I claimed

Calibrating DCGM `dram_active` against `copy_` at known traffic, the raw counter read a
consistent −16 % against my assumed 4.0 TB/s. I fitted the constant that makes known traffic come
out right, got **4.777 TB/s**, and wrote:

> not a fudge factor absorbing error — it is the recovery of a physically real number, since
> GH200's HBM3e is ~4.9 TB/s and my nameplate constant was simply wrong.

## What the device actually is

| probe | value |
|---|---|
| `torch.cuda.get_device_properties(0).name` | `NVIDIA GH200 120GB` |
| total memory | 102 005 473 280 B = 95.00 GiB = 102.0 GB |
| `nvidia-smi` FB total | 97 871 MiB |
| L2 | 60 MiB · SMs 132 · sm_90 |
| max memory clock | 2 619 MHz |

~96 GB of frame buffer is the **HBM3** part at a nominal 4.0 TB/s, not the 141 GB HBM3e part at
4.8 TB/s. And the decisive measurement, independent of DCGM entirely — CUDA-event timing of a
copy, where achieved bandwidth can never exceed peak:

| workload | achieved |
|---|---|
| `copy_` 512 MiB | 3.387 TB/s |
| `copy_` 1024 MiB | 3.515 TB/s |
| `copy_` 2048 MiB | 3.596 TB/s |
| `sum` 1024 MiB (read-only) | 3.421 TB/s |

3.596 TB/s achieved is 90 % of a 4.0 TB/s peak — a normal streaming-copy efficiency. It would be
75 % of a 4.8 TB/s peak, which is implausibly poor for a large contiguous copy. **The fitted
4.777 exceeds both the measured achieved rate and the nominal peak**, and no bandwidth can exceed
the peak. So it is not a bandwidth.

## What it actually is

A **calibrated effective constant**, absorbing two factors that the fit cannot separate:

    K  =  achieved bandwidth  /  instrument response

With achieved ≈ 3.5 TB/s during the copy and K = 4.777, DCGM's `dram_active` under-reports the
true busy fraction by roughly 0.73×. The fit folds the real bandwidth and that instrument bias
into one number, which is exactly what a calibration constant is for — and exactly why it must
not be labelled a bandwidth.

## Why the good fit proved nothing

**The 0.4 % residual does not distinguish the two stories, and I treated it as if it did.** A
constant fitted to make known traffic come out right will fit known traffic well under *either*
explanation: "the peak was higher than I assumed" and "the counter under-reports by 27 %" produce
the identical K and the identical residual. Goodness of fit measures the linearity of the
instrument across the range probed — which is genuinely good news, and is why the constant is
trustworthy in service — but it carries no information about *which* factor the constant absorbs.

Only two things discriminated: the device's memory capacity, and an achieved-bandwidth
measurement taken without the instrument under test. Neither was expensive. I should have run
both before writing a physical interpretation into a commit message.

## What changes, and what does not

* **Unchanged:** the calibration procedure, the fitted value, the 0.4 % residual, every traffic
  number derived from it, and the per-template gate (T2 open at 2.8 %, T1 closed at 26.9 %). The
  constant works because it is fitted to this instrument on this machine.
* **Changed:** the constant is named and documented as an effective instrument constant, not a
  bandwidth. It is not portable to another machine, another GPU, or another DCGM version without
  re-fitting — which the previous framing would have wrongly implied it was, since a hardware
  bandwidth *would* transfer.

The general lesson is the one this status category exists for: a model that predicts well is not
thereby explained correctly, and the cheapest way to find out is to measure the thing the
explanation is *about* rather than the thing the model predicts.
