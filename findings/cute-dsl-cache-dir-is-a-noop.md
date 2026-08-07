# `CUTE_DSL_CACHE_DIR` does nothing, and the real cache follows `TMPDIR`

**Status: stands.** Diagnosed inside a one-hour timebox (R8's first materialization). Two
separate problems, one mine and one upstream's.

## The symptom

Compiling the composed forward — 55 kernels — took ~9 minutes, every time, across three
consecutive runs. `.jit-cache/cute_dsl` held one 4 KB entry throughout, while
`CUTE_DSL_CACHE_DIR` was set, exported, and confirmed visible inside the child process.

## Problem 1, mine: the variable is not read

**`CUTE_DSL_CACHE_DIR` appears zero times in `nvidia-cutlass-dsl` 4.5.2.** Not in the env
manager, not in the cache helpers, nowhere in the package. It has been a no-op since the day it
was written into `env.sh` and `PLAN.md`, where it sits alongside `TRITON_CACHE_DIR` and
`QUACK_CACHE_DIR` — which *are* real — and inherited their credibility by association.

The variables the DSL actually reads are `CUTE_DSL`-prefixed and enumerable
(`base_dsl/env_manager.py`); the one governing the file cache is
**`CUTE_DSL_DISABLE_FILE_CACHING`**, a boolean defaulting to `False`, i.e. caching is on.

Nobody checked. The pinning block in `env.sh` carries a long comment about why caches must be
project-local and reproducible from a clean clone, and one of its four exports was decorative.

## Problem 2, upstream's: the cache path is `TMPDIR`, and the dump root is the CWD

`load_cache_from_path` / `dump_cache_to_path` default to `get_default_generated_ir_path()`:

```python
tmp_dir = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
```

resolving to `$TMPDIR/<user>/cutlass_python_cache`. There is **no dedicated variable** for the
compile cache — it rides on `TMPDIR`, which every other tool on the system also uses and which
Alps purges between jobs. A separate helper, `get_default_file_dump_root()`, returns
`Path.cwd()`, so IR dumps land wherever the process happens to be.

Measured on this machine:

| `TMPDIR` | resolved cache path | populated |
|---|---|---|
| `env.sh`'s `.jit-cache/tmp` | `.jit-cache/tmp/dlu/cutlass_python_cache` | **yes** — 9 files, 300 KB |
| default | `/tmp/dlu/cutlass_python_cache` | no |

So the cache **was working** whenever `env.sh` was sourced. My direct invocations —
`env CUTE_DSL_CACHE_DIR=... python ...` — bypassed `env.sh`, inherited the system `TMPDIR`, and
wrote into a location Alps clears. Every "the cache is broken" observation came from a run that
had carefully set the variable that does nothing while omitting the one that matters.

## What was actually wrong, in order

1. I set a variable that does not exist, for weeks, and it looked authoritative because it sat
   in a block of variables that do.
2. I then observed a cold cache and concluded the cache was broken — **[correlation]**: the
   variable was set, the cache was empty, and I joined them. What I had not done was ask where
   the cache *would* be if the variable were ignored.
3. The upstream design makes that easy to get wrong: the cache location is a side effect of a
   general-purpose variable, and the obvious-looking specific variable does not exist.

## Consequences

* `env.sh` keeps `TMPDIR` pointed into the repo — that line was doing all the work and its
  comment did not know it. `CUTE_DSL_CACHE_DIR` is retained **only** as a documented no-op with
  a note, because silently deleting it would lose the record of the mistake.
* Anything invoking the tools directly must source `env.sh` or set `TMPDIR` itself. The N=5
  sbatch sources `env.sh`, so it caches; its time limit was nonetheless budgeted for cold
  compiles, which is the safe direction to have guessed wrong.
* Worth reporting upstream as a usability issue, not a bug: a compile cache that rides on
  `TMPDIR` with no dedicated override, plus a dump root fixed to the CWD, is easy to misconfigure
  and gives no diagnostic when it happens. Draft in
  `docs/upstream_cutedsl_cache_issue_DRAFT.md`; **not filed**, pending review, and much weaker
  than the other two upstream drafts since nothing here is incorrect — only surprising.

## The general lesson

An environment variable that is set and ignored produces exactly the same observable as one that
is honoured and unhelpful. Distinguishing them costs one `grep` of the package, and I ran it only
after building a whole narrative — "the cache is broken", reported twice to review — on top of
the assumption that setting it meant something.
