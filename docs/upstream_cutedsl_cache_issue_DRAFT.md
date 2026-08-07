# DRAFT — upstream note for NVIDIA/cutlass (CuTe DSL). **Not filed.** For review.

**Weaker than the other two drafts in this directory, deliberately.** Nothing below is incorrect
behaviour; it is behaviour that is easy to misconfigure and gives no diagnostic when misconfigured.
Filing it is a judgement call, not an obligation.

**Title:** CuTe DSL file cache has no dedicated path variable and its dump root is the CWD

**Version:** nvidia-cutlass-dsl 4.5.2

### Summary

The Python DSL's persistent compile cache (`base_dsl/cache_helpers.py`,
`load_cache_from_path` / `dump_cache_to_path`) defaults its location to

```python
tmp_dir = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
# -> $TMPDIR/<user>/cutlass_python_cache
```

There is no `CUTE_DSL`-prefixed variable for the cache *path*. The only cache-related knob is
`CUTE_DSL_DISABLE_FILE_CACHING` (on/off). Separately, `get_default_file_dump_root()` returns
`Path.cwd()`, so IR dumps land wherever the process was started.

### Why this is awkward in practice

On a shared HPC system (CSCS Alps, GH200), `TMPDIR` is set per-job and purged between jobs, and
is shared with every other tool that respects it. A user who wants a project-local, reproducible
compile cache — the same thing `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` provide — has to
know that CuTe DSL's cache rides on `TMPDIR`, because the plausible-looking
`CUTE_DSL_CACHE_DIR` does not exist and setting it fails silently.

We set `CUTE_DSL_CACHE_DIR` for several weeks, alongside the Triton and Inductor equivalents,
and concluded twice that the cache was broken: every run recompiled ~55 kernels from scratch
(~9 min). The cache was in fact working whenever our environment script — which also redirects
`TMPDIR` — was sourced, and silently absent whenever a command bypassed it.

### Suggestions, in decreasing order of usefulness

1. **A dedicated variable** for the cache directory, e.g. `CUTE_DSL_CACHE_DIR`, falling back to
   the current `TMPDIR` behaviour. This is what users will try first, and it currently no-ops.
2. **Log the resolved cache path** at info level on first compile, and log cache hit/miss. The
   machinery for this exists (`jit_time_profiling` already logs a hit rate); it is just not on
   by default, so a misconfigured cache is indistinguishable from a cold one.
3. **A dedicated variable for the dump root**, rather than `Path.cwd()` — a library writing
   artifacts into the caller's working directory is surprising, and in a repo it means generated
   IR appears in `git status`.

### Environment

GH200 (sm_90a), aarch64, Python 3.13.14, torch 2.13.0+cu130, nvidia-cutlass-dsl 4.5.2, CSCS Alps.

---

## TODO before this is sendable

1. Confirm against the current release — 4.5.2 is what we have pinned, and this may already be
   addressed.
2. Check whether a documented mechanism exists that we simply did not find; the finding rests on
   a `grep` of the installed package, not on documentation.
3. Decide whether it is worth filing at all. It costs a reviewer's attention and reports no
   incorrect behaviour.

**Status: awaiting Denghui's review. Not filed.**
