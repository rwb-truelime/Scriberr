# CUDA Torch Repair Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent concurrent CUDA Torch reinstalls against one Python environment while preserving parallel repairs for distinct environments.

**Architecture:** `EnsureCUDATorch` retains its inexpensive unlocked readiness check. When it detects unavailable CUDA Torch, it obtains a process-local mutex stored by `envPath`, then repeats that readiness check before retaining the existing GPU detection and `uv pip install --reinstall` path. The second check makes waiting callers observe the completed repair and return without reinstalling.

**Tech Stack:** Go standard library (`sync.Map`, `sync.Mutex`), existing `os/exec` command execution, Go testing package.

## Global Constraints

- Work in the existing `blackwell-aarch64-support` checkout; do not create a worktree.
- Modify only CUDA repair synchronization and its direct regression test.
- Preserve the existing wheel URL, `uv pip install --reinstall` command, GPU detection, and adapter warning-only error handling.
- Locks are process-local and keyed exactly by `envPath`.
- Do not commit unless the user explicitly requests it.

---

### Task 1: Serialize CUDA Torch Repair Per Environment

**Files:**
- Modify: `internal/transcription/adapters/base_adapter.go:22-123`
- Modify: `internal/transcription/adapters/base_adapter_test.go:3-25`

**Interfaces:**
- Consumes: `EnsureCUDATorch(envPath string) error`, invoked by Parakeet, Canary, and Sortformer using the shared `/app/whisperx-env/parakeet` environment.
- Produces: `EnsureCUDATorch(envPath string) error` with unchanged callers and error contract; concurrent calls for one `envPath` invoke the installer at most once after a successful repair.

- [ ] **Step 1: Write the failing concurrency regression test**

Replace the imports in `internal/transcription/adapters/base_adapter_test.go` with:

```go
import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)
```

Append this helper and test after `TestRedactedCommandArgsMasksHuggingFaceTokens`:

```go
func writeTestExecutable(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(contents), 0755); err != nil {
		t.Fatalf("write executable %s: %v", path, err)
	}
}

func TestEnsureCUDATorchSerializesRepairForSameEnvironment(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test uses POSIX shell scripts")
	}

	tempDir := t.TempDir()
	envPath := filepath.Join(tempDir, "env")
	venvBin := filepath.Join(envPath, ".venv", "bin")
	if err := os.MkdirAll(venvBin, 0755); err != nil {
		t.Fatalf("create virtual environment directory: %v", err)
	}

	statePath := filepath.Join(tempDir, "cuda-ready")
	installLogPath := filepath.Join(tempDir, "installs")
	writeTestExecutable(t, filepath.Join(venvBin, "python"), `#!/bin/sh
if [ -f "$CUDA_TEST_STATE" ]; then
  printf 'True\n'
else
  printf 'False\n'
fi
`)

	fakeBin := filepath.Join(tempDir, "bin")
	if err := os.MkdirAll(fakeBin, 0755); err != nil {
		t.Fatalf("create fake bin directory: %v", err)
	}
	writeTestExecutable(t, filepath.Join(fakeBin, "uv"), `#!/bin/sh
printf 'install\n' >> "$CUDA_TEST_INSTALL_LOG"
sleep 1
touch "$CUDA_TEST_STATE"
`)
	writeTestExecutable(t, filepath.Join(fakeBin, "nvidia-smi"), `#!/bin/sh
exit 0
`)

	t.Setenv("CUDA_TEST_STATE", statePath)
	t.Setenv("CUDA_TEST_INSTALL_LOG", installLogPath)
	t.Setenv("PATH", fakeBin+string(os.PathListSeparator)+os.Getenv("PATH"))

	start := make(chan struct{})
	results := make(chan error, 3)
	for range 3 {
		go func() {
			<-start
			results <- EnsureCUDATorch(envPath)
		}()
	}
	close(start)

	for range 3 {
		if err := <-results; err != nil {
			t.Fatalf("EnsureCUDATorch returned an error: %v", err)
		}
	}

	installs, err := os.ReadFile(installLogPath)
	if err != nil {
		t.Fatalf("read install log: %v", err)
	}
	if got := strings.Count(string(installs), "install\n"); got != 1 {
		t.Fatalf("expected one CUDA Torch reinstall, got %d", got)
	}

	output, err := exec.Command(filepath.Join(venvBin, "python"), "-c", "import torch").Output()
	if err != nil {
		t.Fatalf("check repaired CUDA Torch: %v", err)
	}
	if got := strings.TrimSpace(string(output)); got != "True" {
		t.Fatalf("expected CUDA Torch to be available after repair, got %q", got)
	}
}
```

The fake installer records its invocation before sleeping and setting CUDA-ready state. Without synchronization, all three callers observe `False` before the first installer changes the state, so the assertion receives three install records.

- [ ] **Step 2: Run the regression test and verify the expected failure**

Run:

```fish
go test ./internal/transcription/adapters -run '^TestEnsureCUDATorchSerializesRepairForSameEnvironment$' -count=1
```

Expected: FAIL with `expected one CUDA Torch reinstall, got 3`.

- [ ] **Step 3: Add the keyed lock and post-lock CUDA check**

Extend the existing package-level variable block in `internal/transcription/adapters/base_adapter.go`:

```go
var (
	envCacheMutex        sync.RWMutex
	envCache             = make(map[string]bool)
	requestGroup         singleflight.Group
	cudaTorchRepairLocks sync.Map // map[string]*sync.Mutex
)
```

Replace `EnsureCUDATorch` with the following implementation and add `cudaTorchAvailable` immediately after it:

```go
func EnsureCUDATorch(envPath string) error {
	venvPython := filepath.Join(envPath, ".venv", "bin", "python")
	if _, err := os.Stat(venvPython); err != nil {
		return nil // No venv yet, nothing to fix
	}

	if cudaTorchAvailable(venvPython, envPath) {
		return nil
	}

	lock, _ := cudaTorchRepairLocks.LoadOrStore(envPath, &sync.Mutex{})
	repairLock := lock.(*sync.Mutex)
	repairLock.Lock()
	defer repairLock.Unlock()

	// Another caller may have repaired this environment while this call waited.
	if cudaTorchAvailable(venvPython, envPath) {
		return nil
	}

	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		logger.Info("No GPU detected, keeping CPU torch", "env", envPath)
		return nil
	}

	wheelURL := GetPyTorchWheelURL()
	logger.Info("Forcing CUDA torch install (UV resolved CPU-only wheel on this platform)",
		"env", envPath, "wheel_url", wheelURL)

	cmd := exec.Command("uv", "pip", "install",
		"torch", "torchaudio",
		"--index-url", wheelURL,
		"--reinstall",
		"--python", venvPython,
	)
	installOut, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to force-install CUDA torch: %w: %s", err, strings.TrimSpace(string(installOut)))
	}

	logger.Info("CUDA torch installed successfully", "env", envPath)
	return nil
}

func cudaTorchAvailable(venvPython, envPath string) bool {
	checkCmd := exec.Command(venvPython, "-c", "import torch; print(torch.cuda.is_available())")
	out, err := checkCmd.CombinedOutput()
	if err != nil {
		logger.Warn("Torch import failed (likely corrupted install), will reinstall",
			"error", err, "output", string(out), "env", envPath)
		return false
	}
	if strings.TrimSpace(string(out)) != "True" {
		return false
	}

	logger.Info("Torch CUDA already available, no override needed", "env", envPath)
	return true
}
```

- [ ] **Step 4: Run the targeted test and verify it passes**

Run:

```fish
go test ./internal/transcription/adapters -run '^TestEnsureCUDATorchSerializesRepairForSameEnvironment$' -count=1
```

Expected: PASS. The test completes after one fake installer sleep and records exactly one reinstall.

- [ ] **Step 5: Run adapter tests and the full Go suite**

Run:

```fish
go test ./internal/transcription/adapters -count=1
go test ./... -count=1
```

Expected: both commands exit `0` with no test failures.

- [ ] **Step 6: Inspect the exact diff; do not commit**

Run:

```fish
git diff --check
git diff -- internal/transcription/adapters/base_adapter.go internal/transcription/adapters/base_adapter_test.go
git status --short
```

Expected: only the synchronization implementation, its test, and the already-approved planning documents are new or modified; leave the existing untracked runtime artifacts untouched. Do not create a commit unless the user explicitly requests one.
