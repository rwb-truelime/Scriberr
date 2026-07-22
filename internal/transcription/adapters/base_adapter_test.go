package adapters

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRedactedCommandArgsMasksHuggingFaceTokens(t *testing.T) {
	token := "hf_secret_value"
	args := []string{
		"python", "whisperx", "audio.wav",
		"--hf_token", token,
		"--hf-token", token,
		"--model", "large-v3",
	}

	got := RedactedCommandArgs(args)

	if strings.Contains(got, token) {
		t.Fatalf("redacted command exposed token: %s", got)
	}
	if !strings.Contains(got, "--hf_token ******") || !strings.Contains(got, "--hf-token ******") {
		t.Fatalf("redacted command did not mask both token flags: %s", got)
	}
	if !strings.Contains(got, "--model large-v3") {
		t.Fatalf("redacted command removed non-secret arguments: %s", got)
	}
}

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
