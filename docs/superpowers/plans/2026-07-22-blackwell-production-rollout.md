# Blackwell Production Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit and push the CUDA Torch repair lock, then build, tag, deploy, and verify a fresh Scriberr Blackwell production image while preserving all rollback images.

**Architecture:** The Git commit is created before building so the immutable production tag identifies the exact deployed source revision. Existing image IDs receive additive archive tags before Compose moves `scriberr:local-blackwell` to the new build. The deployment captures `HF_TOKEN` only from the running container into one Fish-shell variable, then uses the same Compose project to replace the container without deleting volumes or images.

**Tech Stack:** Git, Docker Engine, Docker Compose, NVIDIA Container Toolkit, Fish shell, existing `docker-compose.build.blackwell.yml`.

## Global Constraints

- Work in `/home/truelime/docker-projects/scriberr/src` on `blackwell-aarch64-support`; do not create a worktree or merge branches.
- Commit `internal/transcription/adapters/base_adapter.go`, `internal/transcription/adapters/base_adapter_test.go`, and `docs/superpowers/` only.
- Do not stage `internal/transcription/adapters/whisperx_adapter.go.bak`, `model-cache/`, or `scriberr_data/`.
- Do not delete or prune Docker images, volumes, or the external `caddy-proxy` network.
- Deploy only with `docker-compose.build.blackwell.yml` and Compose project `src`.
- Never print, write, commit, or log the value of `HF_TOKEN`.

---

### Task 1: Commit and Push the Audited CUDA Repair Change

**Files:**
- Modify: `internal/transcription/adapters/base_adapter.go`
- Modify: `internal/transcription/adapters/base_adapter_test.go`
- Create: `docs/superpowers/specs/2026-07-22-cuda-torch-repair-lock-design.md`
- Create: `docs/superpowers/plans/2026-07-22-cuda-torch-repair-lock.md`
- Create: `docs/superpowers/specs/2026-07-22-blackwell-production-rollout-design.md`
- Create: `docs/superpowers/plans/2026-07-22-blackwell-production-rollout.md`

**Interfaces:**
- Consumes: the tested keyed CUDA-Torch repair lock and its regression test in the current checkout.
- Produces: a pushed `fork/blackwell-aarch64-support` commit whose seven-character SHA is used by the immutable production image tag.

- [ ] **Step 1: Verify the intended staging scope**

Run from the current checkout:

```fish
set repo /home/truelime/docker-projects/scriberr/src
git -C $repo status --short
git -C $repo diff --check
```

Expected: the two adapter files are modified; `docs/superpowers/` is untracked; the three pre-existing runtime artifacts remain untracked and are not changed.

- [ ] **Step 2: Stage only the code, test, and audit documents**

```fish
git -C $repo add internal/transcription/adapters/base_adapter.go
git -C $repo add internal/transcription/adapters/base_adapter_test.go
git -C $repo add docs/superpowers
git -C $repo diff --cached --check
git -C $repo diff --cached --stat
git -C $repo status --short
```

Expected: staged paths are exactly the two adapter files and `docs/superpowers/`; no `.bak`, cache, or runtime-data path is staged.

- [ ] **Step 3: Commit the reviewed implementation and audit trail**

```fish
git -C $repo commit -m "fix: serialize CUDA Torch repairs per environment"
git -C $repo rev-parse --short HEAD
```

Expected: Git creates one commit and prints its seven-character SHA.

- [ ] **Step 4: Push the branch and verify parity**

```fish
git -C $repo push fork blackwell-aarch64-support
git -C $repo rev-list --left-right --count fork/blackwell-aarch64-support...HEAD
```

Expected: the parity count is `0\t0`.

### Task 2: Archive Images, Build, Deploy, and Verify Production

**Files:**
- Modify at runtime only: Docker image tags and the `scriberr` container.
- Uses: `docker-compose.build.blackwell.yml`

**Interfaces:**
- Consumes: pushed commit SHA from Task 1, current container `scriberr`, image tags `scriberr:local-blackwell`, `scriberr:pre-hf-token-redaction`, and `scriberr:pre-rebase-origin-main`.
- Produces: `scriberr:production-blackwell`, `scriberr:production-blackwell-$short_sha`, three archive tags, and a running `scriberr` container using the fresh build.

- [ ] **Step 1: Capture runtime prerequisites without exposing the token**

Run this Fish script in one remote shell. Do not print `$hf_token`.

```fish
set repo /home/truelime/docker-projects/scriberr/src
set compose $repo/docker-compose.build.blackwell.yml
set hf_token (docker inspect scriberr --format '{{range .Config.Env}}{{println .}}{{end}}' | string match -r '^HF_TOKEN=.*' | string replace -r '^HF_TOKEN=' '')

if test -z "$hf_token"
    printf '%s\n' 'HF_TOKEN is absent from the running container; aborting before build or downtime.' >&2
    exit 1
end

set previous_production (docker inspect scriberr --format '{{.Image}}')
set previous_hf (docker image inspect scriberr:pre-hf-token-redaction --format '{{.Id}}')
set previous_rebase (docker image inspect scriberr:pre-rebase-origin-main --format '{{.Id}}')
set short_sha (git -C $repo rev-parse --short HEAD)

if not string match -q 'sha256:375cc06b491dc7fd3d8d3f52c883d7074ad67a9cb5cbf43bea96a1c160f28c12' $previous_production
    printf '%s\n' 'Unexpected current production image; aborting without changes.' >&2
    exit 1
end
if not string match -q 'sha256:a1731e27bb9a9d28f8989dc68839db90bfb1f8f7f619f65def6d88d1cf2b3b3b' $previous_hf
    printf '%s\n' 'Unexpected pre-HF-redaction image; aborting without changes.' >&2
    exit 1
end
if not string match -q 'sha256:df55ebcd8f4718ffea03e13bc36ad166eb93388b841cfb0406703aaf7228c217' $previous_rebase
    printf '%s\n' 'Unexpected pre-rebase image; aborting without changes.' >&2
    exit 1
end
```

Expected: the script is silent on success and leaves `hf_token`, three prior image IDs, and `short_sha` available in its shell session.

- [ ] **Step 2: Add archive tags before building**

Continue in the same Fish shell:

```fish
docker tag $previous_production scriberr:archive-20260722-production-pre-cuda-torch-lock-375cc06b
docker tag $previous_hf scriberr:archive-20260722-pre-hf-token-redaction-a1731e2
docker tag $previous_rebase scriberr:archive-20260424-pre-rebase-origin-main-df55ebc

test (docker image inspect scriberr:archive-20260722-production-pre-cuda-torch-lock-375cc06b --format '{{.Id}}') = $previous_production
test (docker image inspect scriberr:archive-20260722-pre-hf-token-redaction-a1731e2 --format '{{.Id}}') = $previous_hf
test (docker image inspect scriberr:archive-20260424-pre-rebase-origin-main-df55ebc --format '{{.Id}}') = $previous_rebase
```

Expected: all three `test` commands exit `0`; no existing image tag is deleted.

- [ ] **Step 3: Build and tag the fresh Blackwell image**

Continue in the same Fish shell:

```fish
env HF_TOKEN="$hf_token" docker compose -p src -f $compose build
set new_image (docker image inspect scriberr:local-blackwell --format '{{.Id}}')

if test "$new_image" = "$previous_production"
    printf '%s\n' 'Build did not create a new Scriberr image; aborting before deployment.' >&2
    exit 1
end

docker tag $new_image scriberr:production-blackwell
docker tag $new_image scriberr:production-blackwell-$short_sha

test (docker image inspect scriberr:production-blackwell --format '{{.Id}}') = $new_image
test (docker image inspect scriberr:production-blackwell-$short_sha --format '{{.Id}}') = $new_image
```

Expected: `scriberr:local-blackwell`, `scriberr:production-blackwell`, and the immutable commit-specific production tag all resolve to the new image ID.

- [ ] **Step 4: Replace the current Compose container without deleting data or images**

Continue in the same Fish shell:

```fish
docker compose -p src -f $compose down
env HF_TOKEN="$hf_token" docker compose -p src -f $compose up -d --force-recreate
```

Expected: Docker removes only the prior container and recreates `scriberr`; no `-v`, `--rmi`, image-delete, or prune command is used.

- [ ] **Step 5: Verify service readiness, GPU access, running image, and tags**

Continue in the same Fish shell:

```fish
set healthy 0
for attempt in (seq 1 60)
    if docker exec scriberr curl --fail --silent --show-error --output /dev/null http://127.0.0.1:8080/
        set healthy 1
        break
    end
    sleep 2
end

if test $healthy -ne 1
    printf '%s\n' 'Scriberr did not return HTTP success within 120 seconds.' >&2
    exit 1
end

set running_image (docker inspect scriberr --format '{{.Image}}')
test "$running_image" = "$new_image"
docker exec scriberr nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
test (docker image inspect scriberr:production-blackwell --format '{{.Id}}') = $running_image
test (docker image inspect scriberr:production-blackwell-$short_sha --format '{{.Id}}') = $running_image
test (docker image inspect scriberr:archive-20260722-production-pre-cuda-torch-lock-375cc06b --format '{{.Id}}') = $previous_production
test (docker image inspect scriberr:archive-20260722-pre-hf-token-redaction-a1731e2 --format '{{.Id}}') = $previous_hf
test (docker image inspect scriberr:archive-20260424-pre-rebase-origin-main-df55ebc --format '{{.Id}}') = $previous_rebase
docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}}' | string match 'scriberr:*'
```

Expected: HTTP readiness succeeds, `nvidia-smi` reports NVIDIA GB10, the running image equals both production tags, and every archive tag resolves to the intended prior image.

- [ ] **Step 6: Roll back only if deployment verification fails**

If Step 5 fails after `down` and `up`, retain the generated diagnostics and keep the original remote Fish shell open so its already-validated `$hf_token` remains available. Run:

```fish
set rollback_image (docker image inspect scriberr:archive-20260722-production-pre-cuda-torch-lock-375cc06b --format '{{.Id}}')

if test -z "$hf_token"
	printf '%s\n' 'HF_TOKEN is absent from the original deployment shell; do not perform rollback.' >&2
    exit 1
end

docker tag $rollback_image scriberr:local-blackwell
docker compose -p src -f $compose down
env HF_TOKEN="$hf_token" docker compose -p src -f $compose up -d --force-recreate
```

Expected: this rollback sequence is not run during a successful release. It preserves all archive tags and volumes.
