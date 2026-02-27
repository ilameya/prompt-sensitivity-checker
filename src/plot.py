from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def _safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s[:180]


def plot_bars_at_temperature(summary_csv: str, temperature: float, out_path: str) -> None:
    df = pd.read_csv(summary_csv)
    df_t = df[df["temperature"] == temperature].copy()

    if df_t.empty:
        raise ValueError(f"No rows found for temperature={temperature} in {summary_csv}")

    models = list(df_t["model"].unique())
    n_models = len(models)

    plt.figure(figsize=(max(8, 4 * n_models), 5))

    if n_models == 1:
        m = models[0]
        d = df_t[df_t["model"] == m].sort_values("accuracy", ascending=False)
        plt.bar(d["variant"], d["accuracy"])
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy by Prompt Variant @ T={temperature} ({m})")
        plt.xticks(rotation=20)
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
        return

    pivot = df_t.pivot_table(index="variant", columns="model", values="accuracy", aggfunc="mean")
    pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)

    pivot.plot(kind="bar", ylim=(0, 1))
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy by Prompt Variant @ T={temperature}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_accuracy_vs_temperature_per_model(summary_csv: str, out_dir: str) -> None:
    df = pd.read_csv(summary_csv)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    for model_name in df["model"].unique():
        d = df[df["model"] == model_name].copy()
        d = d.sort_values(["variant", "temperature"])

        plt.figure(figsize=(9, 5))

        for variant in d["variant"].unique():
            dv = d[d["variant"] == variant]
            plt.plot(dv["temperature"], dv["accuracy"], marker="o", label=variant)

        plt.ylim(0, 1)
        plt.xlabel("Temperature")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs Temperature ({model_name})")
        plt.legend()
        plt.tight_layout()

        out_path = out_dir_path / f"accuracy_vs_temperature_{_safe_filename(model_name)}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()