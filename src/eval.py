import argparse
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts import get_variants
from metrics import is_correct_boolq, extract_yes_no

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_out_path(user_value: Optional[str], default_rel: str) -> Path:
    """
    If user provides a path:
      - absolute paths stay absolute
      - relative paths are resolved relative to PROJECT_ROOT
    If user doesn't provide a path:
      - use PROJECT_ROOT / default_rel
    """
    if not user_value:
        return PROJECT_ROOT / default_rel
    p = Path(user_value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def load_boolq_subset(n: int, seed: int = 42) -> List[Dict[str, str]]:
    """
    BoolQ validation subset of size n (shuffled deterministically).
    Map answer to gold in {'yes','no'}.
    """
    ds = load_dataset("boolq", split="validation")
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    return [{"question": ex["question"], "gold": ("yes" if ex["answer"] else "no")} for ex in ds]


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    do_sample = temperature > 0.0
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt prefix if present
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]

    return decoded.strip()


def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else None,
    ).to(device)
    model.eval()

    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct,google/gemma-2-2b-it",
        help="Comma-separated model names from HF Hub.",
    )
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=6)  # tight for yes/no
    ap.add_argument(
        "--temperatures",
        type=str,
        default="0.0,0.3,0.7",
        help="Comma-separated temperatures (e.g. 0.0,0.3,0.7).",
    )
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_summary", type=str, default=None)
    ap.add_argument("--plots_dir", type=str, default=None)
    args = ap.parse_args()

    model_names = parse_csv_list(args.models)
    temperatures = parse_float_list(args.temperatures)

    out_csv = resolve_out_path(args.out_csv, "results/results.csv")
    out_summary = resolve_out_path(args.out_summary, "results/summary.csv")
    plots_dir = resolve_out_path(args.plots_dir, "plots")

    device = pick_device()
    print(f"Using device: {device}")
    print(f"Models: {model_names}")
    print(f"Temperatures: {temperatures}")
    print(f"Samples: {args.n}")

    items = load_boolq_subset(args.n, seed=args.seed)
    variants = get_variants()

    rows = []

    for model_name in model_names:
        print(f"\nLoading model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name, device)

        for temp in temperatures:
            desc = f"{model_name} @ T={temp}"
            for item in tqdm(items, desc=desc):
                q = item["question"]
                gold = item["gold"]

                for vname, variant in variants.items():
                    prompt = variant.format(q)

                    pred_raw = generate_answer(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=temp,
                    )

                    pred_label = extract_yes_no(pred_raw)
                    correct = is_correct_boolq(pred_raw, gold)

                    rows.append(
                        {
                            "model": model_name,
                            "temperature": temp,
                            "variant": vname,
                            "question": q,
                            "gold": gold,
                            "prediction_raw": pred_raw,
                            "prediction_label": pred_label,
                            "is_correct": correct,
                        }
                    )

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved full results: {out_csv}")

    # Summary
    summary = (
        df.groupby(["model", "temperature", "variant"])["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "accuracy"})
        .sort_values(["model", "temperature", "accuracy"], ascending=[True, True, False])
    )

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    print(f"Saved summary: {out_summary}")

    plots_dir.mkdir(parents=True, exist_ok=True)
    from plot import plot_bars_at_temperature, plot_accuracy_vs_temperature_per_model

    # Compare variants at T=0 
    t0 = 0.0 if 0.0 in temperatures else temperatures[0]
    plot_bars_at_temperature(
        summary_csv=str(out_summary),
        temperature=t0,
        out_path=str(plots_dir / "accuracy_bars_t0.png"),
    )
    print(f"Saved plot: {plots_dir / 'accuracy_bars_t0.png'}")

    plot_accuracy_vs_temperature_per_model(
        summary_csv=str(out_summary),
        out_dir=str(plots_dir),
    )
    print(f"Saved per-model plots into: {plots_dir}")
    print("\nTop lines of summary:")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()