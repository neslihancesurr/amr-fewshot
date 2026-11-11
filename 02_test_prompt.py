
import sys
import json
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from ollama_client import chat

from pathlib import Path

# Configuration
MODEL = "gemma3:latest"
NUM_SHOTS = 10
NUM_CTX = 8192
NUM_PREDICT = 1024

PROJECT_ROOT = Path(__file__).resolve().parent

PROMPT_DIR = PROJECT_ROOT / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SYSTEM_FILE = PROMPT_DIR / "system1.txt"
DEMO_FILE = PROMPT_DIR / "fewshot1.json"

def load_data():

    # Load system prompt
    system_prompt = SYSTEM_FILE.read_text(encoding="utf-8")

    # Remove placeholder if present
    system_prompt = system_prompt.replace(
        "[DEMONSTRATIONS WILL BE INSERTED HERE]", ""
    ).strip()

    # Load few-shot examples
    with open(DEMO_FILE, encoding="utf-8") as f:
        demonstrations = json.load(f)

    return system_prompt, demonstrations


def format_input(example):
    """Format a sentence for E1."""

    return example["sentence"]


def build_messages(system_prompt, demonstrations, sentence):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Add few-shot examples
    for example in demonstrations[:NUM_SHOTS]:

        messages.append({
            "role": "user",
            "content": format_input(example)
        })

        messages.append({
            "role": "assistant",
            "content": example["amr"]
        })

    # Add actual test sentence
    messages.append({
        "role": "user",
        "content": sentence
    })

    return messages


def main():

    print(f"Model: {MODEL}")
    print(f"Few-shot examples: {NUM_SHOTS}")

    system_prompt, demonstrations = load_data()

    if len(demonstrations) < NUM_SHOTS:
        raise ValueError("Not enough few-shot examples.")

    sentence = input("\nEnter a Turkish sentence: ").strip()

    if not sentence:
        raise ValueError("Please provide a sentence.")

    messages = build_messages(
        system_prompt,
        demonstrations,
        sentence
    )

    print("\nGenerating AMR graph...\n")

    output, prompt_tokens = chat(
        config.OLLAMA_URL,
        MODEL,
        messages,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
        think=getattr(config, "THINK", False)
    )

    print("=" * 60)
    print("INPUT:")
    print(sentence)

    print("\nGENERATED AMR:")
    print(output.strip())
    print("=" * 60)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "model": MODEL,
        "experiment": "E1",
        "num_shots": NUM_SHOTS,
        "demonstration_ids": [
            ex["id"] for ex in demonstrations[:NUM_SHOTS]
        ],
        "sentence": sentence,
        "raw_output": output,
        "prompt_tokens": prompt_tokens,
        "num_ctx": NUM_CTX,
        "num_predict": NUM_PREDICT
    }

    output_file = OUTPUT_DIR / "gemma3_e1_smoke_test.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
