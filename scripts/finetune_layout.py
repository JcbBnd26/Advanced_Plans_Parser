#!/usr/bin/env python
"""Fine-tune LayoutLMv3 on project-specific layout data.

Usage
-----
    python scripts/finetune_layout.py --data-dir data/layout_training \\
        --output-dir data/layout_model --epochs 10

The training data directory should contain subdirectories per page::

    data/layout_training/
        page_001/
            image.png        # Full-page render
            tokens.json      # [{text, x0, y0, x1, y1, label}, ...]
        page_002/
            image.png
            tokens.json

Each token entry in ``tokens.json`` must have a ``label`` field matching
one of the layout label names (notes, title_block, legend, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _load_training_examples(data_dir: Path) -> list[dict]:
    """Load page directories into training examples."""
    examples = []
    for page_dir in sorted(data_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        img_path = page_dir / "image.png"
        tokens_path = page_dir / "tokens.json"
        if not img_path.exists() or not tokens_path.exists():
            log.warning("Skipping %s (missing image.png or tokens.json)", page_dir.name)
            continue
        with open(tokens_path) as f:
            token_data = json.load(f)
        examples.append(
            {
                "image_path": str(img_path),
                "tokens": token_data,
                "page_dir": str(page_dir),
            }
        )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune LayoutLMv3 on project layout data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing page subdirectories with image.png + tokens.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/layout_model"),
        help="Output directory for fine-tuned model (default: data/layout_model)",
    )
    parser.add_argument(
        "--base-model",
        default="microsoft/layoutlmv3-base",
        help="Base model to fine-tune from (default: microsoft/layoutlmv3-base)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Training batch size (default: 2)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (default: auto-detect cuda/cpu)",
    )
    args = parser.parse_args()

    # Check dependencies
    try:
        import torch
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        print(
            "ERROR: transformers and torch are required. "
            "Install with: pip install 'plancheck[layout]'",
            file=sys.stderr,
        )
        return 1

    from PIL import Image

    from plancheck.analysis.layout_model import LAYOUT_LABELS, LayoutModel
    from plancheck.models import GlyphBox

    # Load data
    if not args.data_dir.exists():
        print(f"ERROR: Data directory not found: {args.data_dir}", file=sys.stderr)
        return 1

    examples = _load_training_examples(args.data_dir)
    if not examples:
        print("ERROR: No valid training examples found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(examples)} training examples from {args.data_dir}")

    # Initialize model
    layout_model = LayoutModel(
        model_name_or_path=args.base_model,
        device=args.device,
        num_labels=len(LAYOUT_LABELS),
    )

    # Prepare training dataset
    label_to_id = {l: i for i, l in enumerate(LAYOUT_LABELS)}

    class LayoutDataset(torch.utils.data.Dataset):
        def __init__(self, examples_list, processor):
            self.examples = examples_list
            self.processor = processor

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            ex = self.examples[idx]
            image = Image.open(ex["image_path"]).convert("RGB")
            page_width, page_height = image.size

            words = [t["text"] for t in ex["tokens"]]
            boxes = []
            labels = []
            for t in ex["tokens"]:
                x0 = max(0, int(t["x0"] / page_width * 1000))
                y0 = max(0, int(t["y0"] / page_height * 1000))
                x1 = min(1000, int(t["x1"] / page_width * 1000))
                y1 = min(1000, int(t["y1"] / page_height * 1000))
                boxes.append([x0, y0, max(x0, x1), max(y0, y1)])
                labels.append(
                    label_to_id.get(t.get("label", "unknown"), label_to_id["unknown"])
                )

            # Truncate
            max_len = 512
            words = words[:max_len]
            boxes = boxes[:max_len]
            labels = labels[:max_len]

            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding="max_length",
            )

            # Squeeze batch dim
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}

            # Pad labels (CLS=-100, then labels, then PAD=-100)
            padded = [-100] + labels
            while len(padded) < max_len:
                padded.append(-100)
            encoding["labels"] = torch.tensor(padded[:max_len])

            return encoding

    # Load processor
    processor = LayoutLMv3Processor.from_pretrained(args.base_model, apply_ocr=False)

    dataset = LayoutDataset(examples, processor)

    # Split into train/val (90/10)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    print(f"Train: {n_train}, Val: {n_val}")

    # Load model for training
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LAYOUT_LABELS),
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Training args
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    # Save best model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"\nFine-tuned model saved to {args.output_dir}")
    print("Set ml_layout_model_path in config to use it.")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
