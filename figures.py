"""Figures for the single run shown in the notebook."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from ecg_data import load_record, phys_stream
from train import configured_model
import study


def save_figure(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(study.RESULTS / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def quality_and_cost():
    keys = study.PAPER + study.EXTENSION
    names = [study.NAMES[k] for k in keys]
    tests = [study.result(k)["test"] for k in keys]
    y = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5), layout="constrained")
    fig.suptitle("How accurately did each model classify the ECG, and at what cost?")
    for ax, values, title, fmt in (
            (axes[0], [100*t["accuracy"] for t in tests], "Correct predictions (%)", ".1f"),
            (axes[1], [t["efficiency"]["mflops"] for t in tests],
             "Estimated work per window (MFLOPs)", ".2f")):
        bars = ax.barh(y, values, color="#4477AA" if ax is axes[0] else "#228877")
        ax.bar_label(bars, labels=[format(v, fmt) for v in values], padding=3, fontsize=8)
        ax.set_xlim(0, 110 if ax is axes[0] else max(values)*1.22)
        ax.set_xlabel(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    save_figure(fig, "test_accuracy_and_cost")


def training_history():
    families = {
        "GRU": ("gru", "skip_gru", "phys_skip_gru", "mlp_skip_gru"),
        "LSTM": ("lstm", "skip_lstm", "phys_skip_lstm", "mlp_skip_lstm"),
    }
    styles = (("#4477AA", "-"), ("#CC8800", "--"), ("#228877", "-"), ("#AA3377", "-."))
    curves = {}
    for keys in families.values():
        for key in keys:
            run = study.result(key)
            history = run["history"]
            if not history:
                raise ValueError(f"No training history for {study.NAMES[key]}.")
            values = np.array([[row[field] for field in
                               ("epoch", "train_loss", "val_score", "val_update_fraction")]
                              for row in history], dtype=float)
            epochs, loss, score, fraction = values.T
            if (not np.isfinite(values).all() or np.any(epochs < 1)
                    or np.any(epochs != np.floor(epochs)) or np.any(np.diff(epochs) <= 0)
                    or np.any(loss < 0) or np.any((score < 0) | (score > 1))
                    or np.any((fraction < 0) | (fraction > 1))):
                raise ValueError(f"Invalid training metrics for {study.NAMES[key]}.")
            best = np.flatnonzero(epochs == run["best_epoch"])
            if len(best) != 1:
                raise ValueError(f"Selected checkpoint is absent from the history for {study.NAMES[key]}.")
            curves[key] = (epochs, loss, score, 100 * fraction, best[0])

    with plt.rc_context({"font.size": 11, "axes.titlesize": 13, "legend.fontsize": 10}):
        for family, keys in families.items():
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
            fig.subplots_adjust(left=.13, right=.97, top=.88, bottom=.22, hspace=.28)
            fig.suptitle(f"{family}: training loss and validation performance", fontsize=16)
            for key, (color, ls) in zip(keys, styles):
                epochs, loss, score, _, best = curves[key]
                for ax, values in zip(axes, (loss, score)):
                    ax.plot(epochs, values, color=color, ls=ls, lw=2, label=study.NAMES[key])
                    ax.plot(epochs[best], values[best], "D", color=color, ms=6)
            axes[0].set(ylabel="Training loss\n(includes skip penalty)")
            axes[1].set(xlabel="Epoch", ylabel="Validation within-record AUROC", ylim=(0, 1))
            axes[1].axhline(.5, color=".6", ls=":", lw=1)
            for ax in axes:
                ax.grid(axis="y", alpha=.25)
                ax.spines[["top", "right"]].set_visible(False)
            fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
                       bbox_to_anchor=(.5, .06), ncol=2, frameon=False)
            fig.text(.5, .025, "Diamonds: validation-selected checkpoint. Lines end at the last recorded epoch.",
                     ha="center", fontsize=10)
            save_figure(fig, f"{family.lower()}_training_history")

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 9))
        fig.subplots_adjust(left=.12, right=.97, top=.88, bottom=.22, hspace=.85)
        fig.suptitle("How did the update rate change during training?", fontsize=16)
        for ax, (family, keys) in zip(axes, families.items()):
            for key, (color, ls) in zip(keys, styles):
                epochs, _, _, percent, best = curves[key]
                ax.plot(epochs, percent, color=color, ls=ls, lw=2, label=study.NAMES[key])
                ax.plot(epochs[best], percent[best], "D", color=color, ms=6)
            ax.set(title=f"{family}: updates on validation windows",
                   ylabel="Samples updated (%)", ylim=(-3, 105))
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", alpha=.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2, frameon=False)
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Epoch")
        fig.text(.5, .02, "Diamonds: validation-selected checkpoint. 100% means no skipping.",
                 ha="center", fontsize=10)
        save_figure(fig, "training_update_rates")


def skip_strips():
    families = {cell: (cell, "skip_"+cell, "phys_skip_"+cell) for cell in ("gru", "lstm")}
    runs, predictions = {}, {}
    ref = None
    for trio in families.values():
        for key in trio:
            runs[key] = study.result(key)
            with np.load(study.RESULTS / key / "test_predictions.npz", allow_pickle=False) as z:
                p = predictions[key] = dict(z)
            if ref is None:
                ref = p
                eligible = ref["y"] == 1
            elif any(not np.array_equal(ref[k], p[k]) for k in ("rec", "start", "y")):
                raise ValueError(f"{study.NAMES[key]} was tested on different ECG windows.")
            eligible &= p["prob"] >= runs[key]["val"]["threshold"]
    hits = np.flatnonzero(eligible)
    if not len(hits):
        print("No positive test window was correctly classified by all six models. No strips drawn.")
        return
    i = hits[0]
    rid, start = int(ref["rec"][i]), int(ref["start"][i])
    meta = runs["gru"]["meta"]
    fs, win = meta["fs"], meta["win"]
    record = load_record(study.DATA, rid, "auto", meta["decim"], study.CACHE)
    # RR intervals and the beat template carry over from earlier windows in this recording.
    feats = phys_stream(record, meta["decim"], meta["delay_steps"])[start:start+win]
    signal = record["sig"][start:start+win, 0]
    time = np.arange(win)/fs
    beats = record["beat_idx"]
    ectopic = (beats >= start) & (beats < start+win) & np.isin(record["beat_cls"], [1, 2, 3])
    if not ectopic.any():
        raise ValueError("Selected positive window has no reference ectopic beat.")
    for cell, trio in families.items():
        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(8, 6))
        fig.subplots_adjust(left=.12, right=.97, top=.77, bottom=.12, hspace=.5)
        fig.suptitle(f"When did the {cell.upper()} models update or skip?", fontsize=15)
        fig.text(.5, .90, f"Record {rid}, {start/fs:g}-{(start+win)/fs:g} s | "
                 "All three correctly detected a window with an ectopic beat", ha="center", fontsize=9)
        colors = ["#0072B2", "#D55E00"]
        handles = [Line2D([], [], marker="o", linestyle="", color=c) for c in colors]
        fig.legend(handles, ["Skipped sample", "State update"], loc="center",
                   bbox_to_anchor=(.5, .85), ncol=2, frameon=False)
        for ax, key in zip(axes, trio):
            run, pred = runs[key], predictions[key]
            inputs = record["sig"][start:start+run["meta"]["seq_len"]]
            if run["meta"]["n_feats"]:
                inputs = np.column_stack((inputs, feats))
            model = configured_model(SimpleNamespace(**run["config"]), run["meta"], 1)
            model.load_state_dict(torch.load(study.RESULTS / key / "best.pt",
                                             map_location="cpu", weights_only=True))
            with torch.inference_mode():
                logits, aux = model.eval()(torch.from_numpy(inputs)[None])
            score = logits.sigmoid().item()
            mask = (np.ones(len(inputs), dtype=bool) if key == cell else aux["mask"][0].numpy() > .5)
            if (not np.isclose(score, pred["prob"][i], atol=1e-5, rtol=0)
                    or score < run["val"]["threshold"] or mask.sum() != pred["updates"][i]):
                raise ValueError(f"Cannot verify the saved prediction and skip decisions for {key}.")
            mask = mask[:win]  # Omit the decision-delay tail from every row.
            ax.plot(time, signal, color=".7", lw=.65)
            for value, color in enumerate(colors):
                sel = mask == value
                ax.scatter(time[sel], signal[sel], s=3, color=color, lw=0, rasterized=True)
            ax.set_title(study.NAMES[key], loc="left", fontsize=10)
            ax.set_title(f"{100*(~mask).mean():.1f}% skipped", loc="right", fontsize=9)
            ax.set_xlim(0, win/fs)
            ax.spines[["top", "right"]].set_visible(False)
        axes[1].set_ylabel("ECG amplitude (normalized MLII)")
        axes[-1].set_xlabel("Seconds from window start")
        save_figure(fig, f"{cell}_updates_and_skips")
