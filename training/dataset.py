"""
training/dataset.py
Prepares the LIAR dataset for instruction fine-tuning.

Maps the original 6 labels to 4 cleaner verdict classes:
  pants-fire, false        → FALSE
  barely-true, half-true   → MOSTLY_FALSE
  mostly-true              → MOSTLY_TRUE
  true                     → VERIFIED

Run: python dataset.py
Output: data/train.jsonl, data/validation.jsonl, data/test.jsonl
"""

from datasets import load_dataset
import json
import os

LABEL_MAP = {
    "pants-fire":  "FALSE",
    "false":       "FALSE",
    "barely-true": "MOSTLY_FALSE",
    "half-true":   "MOSTLY_FALSE",
    "mostly-true": "MOSTLY_TRUE",
    "true":        "VERIFIED",
}

# Confidence we assign to gold labels (real labels don't have scores)
LABEL_CONFIDENCE = {
    "FALSE":       90,
    "MOSTLY_FALSE": 72,
    "MOSTLY_TRUE": 72,
    "VERIFIED":    90,
}

SYSTEM_PROMPT = (
    "You are SachBol, an expert fact-checking AI. "
    "Analyze the given news claim and classify its accuracy. "
    "Respond ONLY with valid JSON — no prose outside the JSON block."
)


def build_user_message(sample: dict) -> str:
    statement = sample.get("statement", "").strip()
    speaker   = sample.get("speaker", "").strip()
    subject   = sample.get("subject", "").strip()

    context_parts = []
    if speaker:
        context_parts.append(f"Speaker: {speaker}")
    if subject:
        context_parts.append(f"Topic: {subject}")
    context_line = (" | ".join(context_parts) + "\n") if context_parts else ""

    return (
        f"Fact-check this claim:\n\n"
        f"CLAIM: {statement}\n"
        f"{context_line}"
        f"\nRespond with JSON only:\n"
        f'{{"verdict": "VERIFIED|MOSTLY_TRUE|MOSTLY_FALSE|FALSE", '
        f'"confidence": <0-100>, "reasoning": "<brief explanation>"}}'
    )


def build_assistant_message(label: str) -> str:
    return json.dumps({
        "verdict":    label,
        "confidence": LABEL_CONFIDENCE.get(label, 70),
        "reasoning":  f"Based on available evidence this claim is classified as {label}.",
    })


def format_sample(sample: dict) -> dict | None:
    raw_label = sample.get("label", "")
    label = LABEL_MAP.get(raw_label)
    if label is None:
        return None  # Skip unknown labels

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": build_user_message(sample)},
            {"role": "assistant", "content": build_assistant_message(label)},
        ]
    }


def prepare_dataset(output_dir: str = "data") -> str:
    os.makedirs(output_dir, exist_ok=True)

    print("Loading LIAR dataset from HuggingFace...")
    dataset = load_dataset("liar")

    split_counts = {}
    for split in ["train", "validation", "test"]:
        raw   = dataset[split]
        items = [fmt for s in raw if (fmt := format_sample(s)) is not None]

        out_path = os.path.join(output_dir, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        split_counts[split] = len(items)
        print(f"  {split:12s}: {len(items):5d} samples → {out_path}")

    print(f"\nTotal samples: {sum(split_counts.values())}")

    # Label distribution check
    print("\nLabel distribution (train):")
    train_raw = dataset["train"]
    dist: dict[str, int] = {}
    for s in train_raw:
        mapped = LABEL_MAP.get(s["label"], "SKIP")
        dist[mapped] = dist.get(mapped, 0) + 1
    for k, v in sorted(dist.items()):
        print(f"  {k:<14s}: {v}")

    return output_dir


if __name__ == "__main__":
    prepare_dataset()
