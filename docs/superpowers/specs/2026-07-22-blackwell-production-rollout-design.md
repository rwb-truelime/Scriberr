# Blackwell Production Rollout Design

## Scope

Commit and push the CUDA Torch repair lock change on `blackwell-aarch64-support`, build a new ARM64 Blackwell image, deploy it with Docker Compose project `src`, and retain additive archive tags for all pre-existing Scriberr images.

## Commit Scope

Commit only the CUDA repair implementation, its regression test, and the `docs/superpowers/` design and plan audit trail. Do not stage the pre-existing runtime artifacts:

- `internal/transcription/adapters/whisperx_adapter.go.bak`
- `model-cache/`
- `scriberr_data/`

## Image Tags

The newly built image retains Compose's `scriberr:local-blackwell` tag and gains:

- `scriberr:production-blackwell`
- `scriberr:production-blackwell-$SHORT_SHA`, where `SHORT_SHA` is the seven-character SHA of the pushed commit.

Existing images keep their current tags and receive these archive aliases:

- `375cc06b491d` -> `scriberr:archive-20260722-production-pre-cuda-torch-lock-375cc06b`
- `a1731e27bb9a` -> `scriberr:archive-20260722-pre-hf-token-redaction-a1731e2`
- `df55ebcd8f47` -> `scriberr:archive-20260424-pre-rebase-origin-main-df55ebc`

No image is deleted or pruned.

## Deployment

1. Capture `HF_TOKEN` from the currently running `scriberr` container into a shell variable without printing or writing it to disk.
2. Exit before stopping the service if that value is empty.
3. Build with `docker-compose.build.blackwell.yml` from the current checkout.
4. Stop the current Compose project with `docker compose -p src ... down`, without `-v` or image-removal flags.
5. Start the new container with `docker compose -p src ... up -d --force-recreate` and the preserved token.

The external `caddy-proxy` network and named Scriberr data/model/cache volumes remain intact.

## Verification

After startup, verify all of the following:

- `scriberr` is running and uses the freshly built image ID.
- The service returns a successful HTTP response.
- `nvidia-smi` inside the container sees the NVIDIA GB10.
- The immutable and moving production tags point at the running image.
- The archive tags point at their intended older image IDs.

## Rollback

The prior production image is preserved as `scriberr:archive-20260722-production-pre-cuda-torch-lock-375cc06b`. A rollback consists of retagging that image as `scriberr:local-blackwell`, then rerunning the same Compose `down` and `up` sequence with a non-empty `HF_TOKEN`.
