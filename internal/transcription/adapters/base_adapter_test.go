package adapters

import (
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
