# CUDA Torch Repair Lock Design

## Scope

Prevent concurrent CUDA Torch repair attempts against one Python environment. The change applies only to `EnsureCUDATorch` in `internal/transcription/adapters/base_adapter.go`.

## Behavior

1. Keep the existing fast path: return immediately when CUDA Torch is already available.
2. If a repair is needed, acquire a process-local mutex keyed by `envPath`.
3. After acquiring that mutex, check CUDA Torch again.
4. If the caller that held the lock first repaired the environment, waiting callers return without invoking `uv pip install`.
5. If CUDA Torch is still unavailable, retain the existing GPU detection and CUDA-wheel reinstall behavior.
6. Repairs for distinct `envPath` values may proceed concurrently.

## Error Handling

Existing behavior is preserved: `EnsureCUDATorch` returns an error when reinstalling Torch fails, and adapters retain their current warning-only handling of that error.

## Test

Add a Linux unit test using temporary fake `python` and `uv` executables. Three concurrent calls for the same environment must all succeed while exactly one invokes the fake `uv pip install`. This proves the post-lock recheck suppresses redundant reinstalls.

## Out of Scope

- Cross-process locking across multiple Scriberr containers.
- Changes to adapter startup concurrency.
- Changes to CUDA package versions, installer commands, or error policy.
