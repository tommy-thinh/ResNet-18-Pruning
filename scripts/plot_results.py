"""Regenerate the README's measured-data figures without loading models or GPUs."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "scripts" / "figures"
UNSTRUCTURED = ("75", "50", "25", "12_5", "6_25")
COLORS = {
    "dense": "#555555", "unstructured": "#0072B2", "semi24": "#E69F00",
    "channels": "#009E73", "original": "#D55E00", "random": "#CC79A7",
}


def load(name: str) -> dict:
    path = ROOT / "results" / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "completed":
        raise ValueError(f"Cannot plot an incomplete result: {path}")
    return data


def matched_runtime(reference: dict, candidate: dict) -> None:
    for key in ("framework", "device", "hardware", "precision", "batch_size",
                "cuda_graph", "warmup", "repeats"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Incompatible latency conditions: {key}")


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated scripts/figures/{name}.{{png,svg,pdf}}")


def pruning_accuracy(baseline, unstructured, semi24, channels, deployment):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    sparsity = [100 * r["after"]["sparsity"] for r in unstructured]
    for stage, label, marker in (("before", "Before fine-tuning", "s"),
                                  ("after", "After fine-tuning", "o")):
        axes[0].plot(sparsity, [100 * r[stage]["exact_match_accuracy"] for r in unstructured],
                     marker=marker, linestyle="--" if stage == "before" else "-", label=label)
    axes[0].axhline(100 * baseline["exact_match_accuracy"], color=COLORS["dense"],
                   linestyle=":", label="Dense reference")
    axes[0].set(xlabel="Eligible-weight sparsity (%)", ylabel="Test exact match (%)",
                title="(a) Unstructured accuracy recovery")
    axes[0].legend(fontsize=8)

    x = np.arange(3)
    for offset, stage, label, color in ((-.19, "before", "Before fine-tuning", "#56B4E9"),
                                        (.19, "after", "After fine-tuning", COLORS["channels"])):
        values = [100 * r[stage]["exact_match_accuracy"] for r in [semi24, *channels]]
        bars = axes[1].bar(x + offset, values, .36, color=color, label=label)
        axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axes[1].axhline(100 * baseline["exact_match_accuracy"], color=COLORS["dense"], linestyle=":")
    axes[1].set(xticks=x, xticklabels=["2:4", "Channels\n25% removed", "Channels\n50% removed"],
                ylim=(0, 100), ylabel="Test exact match (%)", title="(b) Structural damage and recovery")
    axes[1].legend(fontsize=8, loc="upper center")

    dense_gpu = deployment[0][1]["gpu_latency"]["median_ms"]
    gpu = [deployment[i + 1][1]["gpu_latency"]["median_ms"] for i in range(len(unstructured))]
    axes[2].plot([0, *sparsity], [dense_gpu, *gpu], marker="o", color=COLORS["unstructured"])
    axes[2].axhline(dense_gpu, color=COLORS["dense"], linestyle=":")
    axes[2].set(xlabel="Eligible-weight sparsity (%)", ylabel="Median GPU latency (ms / batch)",
                ylim=(0, dense_gpu * 1.18), title="(c) Zeros do not shrink dense execution")
    fig.suptitle("Accuracy: TensorFlow FP32 | Latency: TensorRT FP16-enabled, A30, batch 64", fontsize=11)
    save(fig, "pruning_accuracy")


def deployment_latency(deployment):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), layout="constrained")
    y = np.arange(len(deployment))
    for ax, field, title in zip(axes, ("gpu_latency", "host_latency"),
                                ("(a) Resident GPU execution (CUDA graph)",
                                 "(b) Host input/output (no CUDA graph)")):
        median = np.array([metrics[field]["median_ms"] for _, metrics, _ in deployment])
        p95 = np.array([metrics[field]["p95_ms"] for _, metrics, _ in deployment])
        ax.barh(y, median, color=[COLORS[group] for _, _, group in deployment], height=.6)
        ax.errorbar(median, y, xerr=np.array([np.zeros(len(y)), p95 - median]),
                    fmt="none", color="#222222", capsize=3, linewidth=1)
        for index, value in enumerate(median):
            ax.text(p95[index] + max(p95) * .025, index,
                    f"{value:.4f}  ({median[0] / value:.2f}x)", va="center", fontsize=8)
        ax.set(yticks=y, yticklabels=[label for label, _, _ in deployment],
               xlabel="Latency (ms / batch; lower is better)", title=title,
               xlim=(0, max(p95) * 1.48))
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    fig.suptitle("A30 / TensorRT 10.13.3.9 / FP16-enabled / batch 64\n"
                 "Bars: median; whisker ends: p95 (not confidence intervals); labels: median and dense-relative speedup",
                 fontsize=11)
    save(fig, "deployment_latency")


def efficiency_tradeoffs(baseline, unstructured, semi24, channels, lth, deployment):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    groups = [
        ("Dense", [baseline], "dense", "X"),
        ("Unstructured (all densities)", [r["after"] for r in unstructured], "unstructured", "o"),
        ("2:4", [semi24["after"]], "semi24", "s"),
        ("Channel pruning", [r["after"] for r in channels], "channels", "D"),
        ("LTH original", [lth["original"]], "original", "^"),
        ("LTH random", [lth["random"]], "random", "v"),
    ]
    for ax, field, scale, title in zip(axes.flat, ("parameters", "macs", "model_mb"),
                                      (1e6, 1e6, 1),
                                      ("Parameters (millions)", "Conv/Dense MACs (millions / image)",
                                       "Inference .keras checkpoint (MiB)")):
        for label, records, group, marker in groups:
            ax.scatter([r[field] / scale for r in records],
                       [100 * r["exact_match_accuracy"] for r in records],
                       label=label, color=COLORS[group], marker=marker, s=55, alpha=.85)
        ax.set(xlabel=title, ylabel="TF FP32 test exact match (%)", ylim=(76, 84), xlim=(0, None))
        for record, label in zip(channels, ("25%", "50%")):
            m = record["after"]
            ax.annotate(label, (m[field] / scale, 100 * m["exact_match_accuracy"]),
                        xytext=(5, 7), textcoords="offset points", fontsize=8)
    ax = axes[1, 1]
    markers = {"dense": "X", "unstructured": "o", "semi24": "s", "channels": "D"}
    for label, metrics, group in deployment:
        point = (metrics["gpu_latency"]["median_ms"], 100 * metrics["exact_match_accuracy"])
        ax.scatter(*point, color=COLORS[group], marker=markers[group], s=55, alpha=.85)
        if group in ("dense", "channels"):
            ax.annotate(label, point, xytext=(5, 7), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Median GPU latency (ms / batch 64)", ylabel="TRT FP16-enabled test exact match (%)",
           ylim=(76, 84), xlim=(.21, .54))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Accuracy-efficiency trade-offs: left is cheaper, up is more accurate\n"
                 "TensorFlow accuracy for model resources; TensorRT accuracy for deployment latency; one training seed",
                 fontsize=11)
    save(fig, "efficiency_tradeoffs")


def sparse_tactics():
    batch64_names = ("latency-dense", "latency-semi24-dense", "latency-semi24-sparse")
    batch64_results = [load(name) for name in batch64_names]
    for result in batch64_results[1:]:
        matched_runtime(batch64_results[0]["runtime"], result["runtime"])
    batch256 = load("latency-semi24-batch256")
    sets = ([r["metrics"] for r in batch64_results],
            [batch256[key] for key in ("dense", "sparse_dense_tactics", "sparse")])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, batch, records in zip(axes, (64, 256), sets):
        gpu = [r["gpu_latency"]["median_ms"] for r in records]
        bars = ax.bar(range(3), gpu, color=[COLORS["dense"], "#56B4E9", COLORS["semi24"]])
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ratio = gpu[1] / gpu[2]
        ax.set(xticks=range(3), xticklabels=["Original\ndense", "Pruned\nsparsity OFF", "Pruned\nsparsity ON"],
               ylim=(0, max(gpu) * 1.35), ylabel="Median GPU latency (ms / batch)",
               title=f"Batch {batch}: {ratio:.3f}x same-weight speedup")
        ax.text(.5, .94, "Same pruned weights: compare OFF against ON", transform=ax.transAxes,
                ha="center", fontsize=8)
    fig.suptitle("2:4 on A30 / TensorRT FP16-enabled / resident GPU / CUDA graph\n"
                 "Separate engine profiles; absolute batch latency is not per-image latency", fontsize=11)
    save(fig, "sparse_tactics")


def lottery_ticket(lth):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout="constrained")
    round_colors = ("#0072B2", "#009E73", "#E69F00", "#D55E00")
    for r, color, marker in zip(lth["rounds"], round_colors, ("o", "s", "^", "D")):
        h = r["training"]["history"]
        epochs = np.arange(1, len(h["val_exact_match_accuracy"]) + 1)
        axes[0].plot(epochs, np.array(h["val_exact_match_accuracy"]) * 100, color=color,
                     marker=marker, markevery=20, markersize=3,
                     label=f"R{r['round']}: {100 * r['density']:.2f}% retained")
    axes[0].set(xlabel="Epoch within each reset/retrain round", ylabel="Validation exact match (%)",
                title="(a) Iterative discovery (not test scores)", ylim=(0, 100))
    axes[0].legend(fontsize=8, loc="lower right")

    final_training = lth["rounds"][-1]["training"]
    for name, t, color in (("Original", final_training, COLORS["original"]),
                            ("Random", lth["random_training"], COLORS["random"])):
        h = t["history"]
        epochs = np.arange(1, len(h["exact_match_accuracy"]) + 1)
        axes[1].plot(epochs, np.array(h["exact_match_accuracy"]) * 100,
                     color=color, linestyle=":", label=f"{name}: train")
        axes[1].plot(epochs, np.array(h["val_exact_match_accuracy"]) * 100,
                     color=color, label=f"{name}: validation")
        best = t["best_epoch"]
        axes[1].scatter(best, h["val_exact_match_accuracy"][best - 1] * 100,
                        color=color, marker="X", s=45, zorder=4)
    axes[1].set(xlabel="Epoch (same maximum budget; adaptive LR)", ylabel="Exact match (%)",
                title="(b) Final mask: initialization comparison", ylim=(0, 100))
    axes[1].legend(fontsize=8, loc="lower right")

    values = [100 * lth[key]["exact_match_accuracy"] for key in ("dense", "original", "random")]
    bars = axes[2].bar(range(3), values,
                       color=[COLORS[key] for key in ("dense", "original", "random")])
    axes[2].bar_label(bars, fmt="%.2f%%", padding=4)
    axes[2].set(xticks=range(3), xticklabels=["Dense", "Original\ninitialization", "Random\ninitialization"],
                ylabel="Test exact match (%)", ylim=(0, 100), title="(c) Final test comparison: one seed")
    density = lth["rounds"][-1]["density"] * 100
    fig.suptitle(f"LTH adaptation: {density:.2f}% eligible Conv weights retained; projection/Dense kernels excluded\n"
                 "TensorFlow FP32 / A30 / batch 64; each discovery round starts again from initialization",
                 fontsize=11)
    save(fig, "lottery_ticket")


def main():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.alpha": .15,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.prop_cycle": matplotlib.cycler(color=["#0072B2", "#D55E00", "#009E73"]),
    })
    baseline = load("baseline")["metrics"]
    unstructured = [load(f"unstructured-{suffix}") for suffix in UNSTRUCTURED]
    semi24 = load("semi24")
    channels = [load(f"channels-{removal}") for removal in (25, 50)]
    lth = load("lth-round4")
    sources = [("Dense", "latency-dense", "dense")]
    sources += [(f"Unstructured: {100 * r['after']['sparsity']:g}% sparse",
                 f"latency-unstructured-{suffix}", "unstructured")
                for suffix, r in zip(UNSTRUCTURED, unstructured)]
    sources += [("2:4", "latency-semi24-sparse", "semi24"),
                ("Channels: 25% removed", "latency-channels-25", "channels"),
                ("Channels: 50% removed", "latency-channels-50", "channels")]
    reference_runtime = load("latency-dense")["runtime"]
    deployment = []
    for label, source, group in sources:
        result = load(source)
        matched_runtime(reference_runtime, result["runtime"])
        for field in ("gpu_latency", "host_latency"):
            timing = result["metrics"][field]
            if not 0 < timing["median_ms"] <= timing["p95_ms"] < float("inf"):
                raise ValueError(f"Invalid timing summary: {source}.{field}")
        deployment.append((label, result["metrics"], group))
    for r in lth["rounds"]:
        t = r["training"]
        if len(t["history"]["val_exact_match_accuracy"]) != t["epochs_run"]:
            raise ValueError("LTH history length differs from recorded epochs")
        if not np.isclose(r["density"], (1 - lth["prune_rate"]) ** r["round"], atol=1e-6):
            raise ValueError("LTH density does not match iterative pruning schedule")
    pruning_accuracy(baseline, unstructured, semi24, channels, deployment)
    deployment_latency(deployment)
    efficiency_tradeoffs(baseline, unstructured, semi24, channels, lth, deployment)
    sparse_tactics()
    lottery_ticket(lth)


if __name__ == "__main__":
    main()
