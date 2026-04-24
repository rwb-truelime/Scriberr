#!/usr/bin/env python3
"""
Voxtral-mini transcription script for Scriberr
Transcribes audio using Mistral's Voxtral-mini model
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from transformers import VoxtralForConditionalGeneration, AutoProcessor


def transcribe_audio(
    audio_path: str,
    output_path: str,
    language: str = "en",
    model_id: str = "mistralai/Voxtral-mini",
    device: str = "auto",
    max_new_tokens: int = 8192,
    temperature: float = 0.0,
    do_sample: bool = False,
    repetition_penalty: float = 1.0,
    num_beams: int = 1,
    top_p: float = 1.0,
    top_k: int = 50,
    no_repeat_ngram_size: int = 0,
) -> dict:
    """
    Transcribe audio using Voxtral-mini model.

    Args:
        audio_path: Path to input audio file
        output_path: Path to output JSON file
        language: Language code (e.g., 'en', 'es', 'fr')
        model_id: HuggingFace model ID
        device: Device to use ('cpu', 'cuda', or 'auto')
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Dictionary containing transcription results
    """
    # Determine device
    # if device == "auto":
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Voxtral model on {device}...", file=sys.stderr)

    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_id)

    # Use appropriate dtype based on device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )

    print(f"Model loaded successfully", file=sys.stderr)
    print(f"Processing audio: {audio_path}", file=sys.stderr)

    # Prepare transcription request using the proper method
    inputs = processor.apply_transcription_request(
        language=language, audio=audio_path, model_id=model_id
    )

    # Move inputs to device with correct dtype
    inputs = inputs.to(device, dtype=dtype)

    print(f"Generating transcription...", file=sys.stderr)

    # Build generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    if do_sample:
        gen_kwargs["do_sample"] = True
        if top_p < 1.0:
            gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if num_beams > 1:
        gen_kwargs["num_beams"] = num_beams
    if no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    print(f"Generation params: {gen_kwargs}", file=sys.stderr)

    # Generate transcription
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )

    # Decode only the newly generated tokens (skip the input prompt)
    decoded_outputs = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )

    transcription_text = decoded_outputs[0]

    print(f"Transcription completed ({len(transcription_text)} chars)", file=sys.stderr)

    # Prepare output in Scriberr format
    # Note: Voxtral doesn't provide word-level timestamps, so we create a single segment
    result = {
        "text": transcription_text,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 0.0,  # Duration unknown without audio analysis
                "text": transcription_text,
                "words": [],  # Voxtral doesn't provide word-level timestamps
            }
        ],
        "language": language,
        "model": model_id,
        "has_word_timestamps": False,  # Important: Voxtral doesn't support timestamps
    }

    # Write output
    output_file = Path(output_path)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Results written to {output_path}", file=sys.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Voxtral-mini model"
    )
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument("output_path", type=str, help="Path to output JSON file")
    parser.add_argument(
        "--language", type=str, default="en", help="Language code (default: en)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mistralai/Voxtral-mini",
        help="HuggingFace model ID (default: mistralai/Voxtral-mini)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate (default: 8192)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (1.0=none)")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam search width")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling cutoff")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="Block repeating n-grams of this size")

    args = parser.parse_args()

    try:
        transcribe_audio(
            audio_path=args.audio_path,
            output_path=args.output_path,
            language=args.language,
            model_id=args.model_id,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            top_p=args.top_p,
            top_k=args.top_k,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
