# REFUTED — `CUTE_DSL_CACHE_DIR` is real; `cute.compile()` disables the cache by design

> **Status: REFUTED.** Both of this entry's central claims were wrong, and the upstream note it
> produced has been withdrawn unfiled. Kept because it reached a report — twice — and because the
> way it was wrong is more instructive than the conclusion.

## What I claimed

1. "`CUTE_DSL_CACHE_DIR` appears **zero times** in nvidia-cutlass-dsl 4.5.2. It has been a no-op
   since Phase 0."
2. "The real cache follows `TMPDIR`, and `env.sh` already redirects it, so the cache was working
   all along for anything that sourced `env.sh`."

Both false. I reported the first to review as a finding and the second as R8's closure.

## What is actually true

**`CUTE_DSL_CACHE_DIR` is read.** `base_dsl/env_manager.py` builds its option names by f-string
interpolation — `f"{prefix}_CACHE_DIR"` with `prefix = "CUTE_DSL"` — so the literal string
`CUTE_DSL_CACHE_DIR` never appears in the source. **I grepped for the literal.** The variable is
honoured; my search method could not have found it however hard it looked.

**The cache never populates because we ask it not to.** `cute.compile()` is
`CompileCallable.__call__`, and it sets, unconditionally:

```python
kwargs["compile_only"] = True
kwargs["no_cache"] = True
```

`dsl.py` then does `if not no_cache and compile_only: no_cache = True` with the warning *"Cache
is disabled as user wants to compile only."* Every kernel this project builds goes through
`cute.compile()`, so the file cache is bypassed **by design, at our request**. The library is
behaving exactly as its own code says it will.

## The evidence that settled it

A canonical kernel, compiled in two separate processes with `CUTE_DSL_LOG_LEVEL=20`:

```
run 1: JIT cache miss  module_hash=[2d2c2769…3d33a274]   compile_seconds=0.199
run 2: JIT cache miss  module_hash=[2d2c2769…3d33a274]   compile_seconds=0.203
files written to CUTE_DSL_CACHE_DIR: 0
```

Three independent facts fall out of that one run, and it is worth listing them separately
because they retire different hypotheses:

1. **`JIT cache miss` on both runs, zero files written.** The cache is not being populated.
2. **The module hash is byte-identical across processes** — `2d2c2769…3d33a274` both times. This
   retires the *fingerprint hypothesis* outright: if our emitted source carried a run-varying
   element (a temp path, a generated module name, a nonce, a timestamp), the hash would differ
   and the cache would be missing for a reason that was our fault in a second, separate way. It
   does not. Whatever else is wrong, our source is stable across runs — which is also a
   non-obvious property of a code generator that writes files into a directory keyed by name, and
   worth having confirmed rather than assumed.
3. **`dump_cache_to_path`'s `JIT cache : dumping` line never appears**, so the write is not
   failing — it is never attempted, because the call is gated off upstream.

Fact 2 is what made fact 3 worth chasing. Had the hashes differed, the investigation would have
stopped at "our fingerprint varies" and never reached the real gate.

## What was actually wrong, in order

1. **A literal grep against a dynamically-constructed name.** The single check I called
   "definitive" — one `grep` of the package — was structurally incapable of finding what it
   looked for. I described it as costing one command and being conclusive; it was one command and
   conclusive of nothing.
2. **A second wrong mechanism built on the first.** Having "established" the variable was dead, I
   found the `TMPDIR` path, saw the cache dir populated, and closed R8. The 9 files I counted
   there were **pytest temporary fixtures**, not cache entries.
3. **An upstream note drafted on both.** It has been deleted rather than filed. It would have
   asked NVIDIA to add a variable that exists and to explain behaviour their code documents.

## The correction that matters

Absence of evidence from a search is only evidence of absence if the search could have found the
thing. Names built by interpolation, dispatch tables, `getattr`, and generated code are all
invisible to a literal grep — and "it appears zero times in the package" is exactly the kind of
statement that sounds like a measurement while being a property of my query.

The right move, available at the time and taken only after two wrong reports, was to turn on the
library's own logging and let it say what it was doing.

## Consequences

* No upstream filing. `docs/upstream_cutedsl_cache_issue_DRAFT.md` deleted.
* `env.sh`'s comment claiming `CUTE_DSL_CACHE_DIR` is a no-op is corrected — the variable works;
  we simply never let the cache run.
* **The compile cost is real and is our design choice**: ~9 minutes per rep, cold, every time,
  because `cute.compile()` trades the cache for an explicitly compiled callable we can launch
  repeatedly. Whether that trade is right — the alternative is the `@cute.jit` call path, which
  caches but re-enters the dispatch layer per launch — is an open question for S2, and now an
  informed one.
