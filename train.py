import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

from ecg_data import build_datasets, label_names, FEAT_NAMES, FEAT_GROUPS, DET_MACS, FEAT_MACS
from models import build_model, count_params
from harness import evaluate, measure_latency, print_report

LOOP_MODELS = ("skip_gru", "skip_lstm", "phys_skip_gru", "phys_skip_lstm", "mlp_skip_gru", "mlp_skip_lstm",
               "rand_gru", "rand_lstm")


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--data_dir", required=True, help="folder containing SampleData/ (or the record folders)")
    a.add_argument("--model", default="gru", help="gru | lstm | skip_gru | skip_lstm | phys_skip_gru | phys_skip_lstm | "
                                                  "mlp_skip_gru | mlp_skip_lstm | rand_gru | rand_lstm | cnn | tcn")
    a.add_argument("--hidden", type=int, default=32, help="32: best GRU width on val within-record AUROC (16/32/64), 1/4 the FLOPs of 64")
    a.add_argument("--layers", type=int, default=1)
    a.add_argument("--dropout", type=float, default=0.0)
    a.add_argument("--no_chrono", action="store_true", help="disable chrono gate-bias init")
    a.add_argument("--readout", default="last", choices=["last", "attn"],
                   help="RNN head input: last state, or last state + attention pool over the computed states")
    # skip rnn
    a.add_argument("--lam", type=float, default=1e-5, help="penalty per state update")
    a.add_argument("--p_skip", type=float, default=0.5, help="rand_gru skip probability (match a trained skip_gru)")
    a.add_argument("--max_skip_sec", type=float, default=1.5, help="phys / mlp gates force an update after this long")
    a.add_argument("--phys_feats", default="all", choices=["all", "rr", "morph"], help="gate feature ablation")
    a.add_argument("--gate_hidden", type=int, default=8)
    a.add_argument("--no_cuda_graph", action="store_true", help="train loop models eagerly on GPU (much slower)")
    # data
    a.add_argument("--win_sec", type=float, default=10.0, help="window length in seconds")
    a.add_argument("--decim", type=int, default=2, help="downsample factor from 360 Hz (2 -> 180 Hz, T=1800 for 10 s)")
    a.add_argument("--channels", default="auto", help="auto (MLII) | both | 0 | 1")
    a.add_argument("--delay_sec", type=float, default=0.5, help="decision delay after each window: phys gate look-ahead, extra ECG input for the other models")
    a.add_argument("--keep_paced", action="store_true", help="keep paced records 107, 217")
    a.add_argument("--label_mode", default="binary", choices=["binary", "aami4"])
    a.add_argument("--split", default="fixed", choices=["fixed", "kfold"])
    a.add_argument("--fold", type=int, default=0)
    a.add_argument("--n_folds", type=int, default=5)
    a.add_argument("--train_overlap", type=float, default=0.0,
                   help="overlap fraction of training label intervals; 0 gives 10 s stride and 0.5 s observation overlap")
    a.add_argument("--cache_dir", default="cache")
    # optimisation
    a.add_argument("--epochs", type=int, default=60)
    a.add_argument("--batch_size", type=int, default=128)
    a.add_argument("--lr", type=float, default=2e-3)
    a.add_argument("--clip", type=float, default=1.0)
    a.add_argument("--patience", type=int, default=15, help="early stopping on val score; skip gates first collapse "
                                                            "updates, so short patience stops before the classifier learns")
    a.add_argument("--warmup", type=int, default=5, help="skip models: no checkpoint selection / early stop in the "
                                                         "first N epochs, the gate is still settling")
    a.add_argument("--max_minutes", type=float, default=None, help="stop training after this wall-clock budget")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out_dir", default=None)
    a.add_argument("--max_train_windows", type=int, default=None, help="subsample train set (smoke tests)")
    return a.parse_args()


class GraphStep(nn.Module):
    # make_graphed_callables needs tensor outputs. Replaying one CUDA graph per batch removes the kernel
    # launch overhead of the per-step python loop (~13x faster here), gradients are identical.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, aux = self.model(x)
        return logits, aux.get("extra_loss", logits.new_zeros(()))


@torch.no_grad()
def save_skip_examples(model, d, probs, thr, meta, path, device, k=2):
    # a few TP / TN / FP / FN test windows with their update masks for the red/blue plots
    model.eval()
    y, pred = d["y"], (probs >= thr).astype(int)
    pick, kinds = [], []
    for name, sel, order in [("TP", (y == 1) & (pred == 1), -1), ("TN", (y == 0) & (pred == 0), 1),
                             ("FP", (y == 0) & (pred == 1), -1), ("FN", (y == 1) & (pred == 0), 1)]:
        idx = np.where(sel)[0]
        idx = idx[np.argsort(order * probs[idx])][:k]
        pick += idx.tolist(); kinds += [name] * len(idx)
    if not pick:
        return
    _, aux = model(torch.as_tensor(d["X"][pick]).to(device))
    T, n_sig = d["X"].shape[1], meta["n_sig"]
    beats = np.full((len(pick), 64), -1); bcls = np.full((len(pick), 64), -1)
    for j, i in enumerate(pick):
        bi, bc = meta["beats"][int(d["rec"][i])]
        s = (bi >= d["start"][i]) & (bi < d["start"][i] + T)
        n = min(s.sum(), 64)
        beats[j, :n], bcls[j, :n] = (bi[s] - d["start"][i])[:n], bc[s][:n]
    ev = d["X"][pick][:, :, -1] if meta["n_feats"] else np.zeros((len(pick), T))
    np.savez(path, x=d["X"][pick][:, :, :n_sig], mask=aux["mask"].cpu().numpy(), event=ev, kind=np.array(kinds),
             prob=probs[pick], rec=d["rec"][pick], start=d["start"][pick], beat_idx=beats, beat_cls=bcls, fs=meta["fs"])


def val_score(m, task):
    if task == "binary":     # prefer within-record AUROC (cannot be inflated by recognising the patient)
        s = m["auroc_within_record"] if m["n_records_with_both_classes"] > 0 else m["auroc"]
    else:
        s = m["macro_f1"]
    return -1.0 if s != s else s        # NaN -> worst


def configured_model(args, meta, n_outputs):
    gated = args.model.startswith(("phys", "mlp"))
    feat_keep = None if args.phys_feats == "all" else FEAT_GROUPS[args.phys_feats] + [len(FEAT_NAMES) - 1]
    return build_model(args.model, meta["n_sig"], n_outputs, args.hidden, args.layers, args.dropout,
                       chrono_T=None if args.no_chrono else meta["seq_len"], lam=args.lam, p_skip=args.p_skip,
                       max_skip=int(round(args.max_skip_sec * meta["fs"])) if gated else None,
                       n_feats=len(FEAT_NAMES), feat_keep=feat_keep, gate_hidden=args.gate_hidden,
                       det_macs=args.decim * DET_MACS, feat_macs=FEAT_MACS, readout=args.readout)


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "binary" if args.label_mode == "binary" else "multiclass"
    names = label_names(args.label_mode)
    out_dir = args.out_dir or os.path.join("runs", args.model)
    os.makedirs(out_dir, exist_ok=True)
    phys = args.model.startswith("phys")

    data, meta = build_datasets(args.data_dir, args.win_sec, args.decim, args.channels, args.label_mode,
                                args.split, args.fold, args.n_folds, args.train_overlap, args.cache_dir,
                                phys=phys, delay_sec=args.delay_sec, keep_paced=args.keep_paced)
    tr, va, te = data["train"], data["val"], data["test"]
    if len(tr["y"]) == 0 or len(va["y"]) == 0 or len(te["y"]) == 0:
        raise SystemExit("train/val/test must each contain at least one record")
    if args.max_train_windows and len(tr["y"]) > args.max_train_windows:
        keep = np.random.RandomState(args.seed).choice(len(tr["y"]), args.max_train_windows, replace=False)
        tr = {k: v[keep] for k, v in tr.items()}

    K = 1 if task == "binary" else len(names)
    model = configured_model(args, meta, K).to(device)
    print(f"[model] {args.model} params={count_params(model):,} device={device}")

    # class imbalance handling
    cnt = np.bincount(tr["y"], minlength=2 if task == "binary" else K).astype(float)
    if task == "binary":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cnt[0] / max(cnt[1], 1)))
        to_target = lambda y: y.float()
    else:
        w = cnt.sum() / (K * np.maximum(cnt, 1))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
        to_target = lambda y: y
    loss_fn = loss_fn.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    Xtr = torch.as_tensor(tr["X"]); ytr = torch.as_tensor(tr["y"])
    bs = min(args.batch_size, len(Xtr))
    step = GraphStep(model)
    use_graph = device == "cuda" and args.model in LOOP_MODELS and not args.no_cuda_graph
    if use_graph:
        model.train()
        t0 = time.time()
        step = torch.cuda.make_graphed_callables(step, (torch.zeros(bs, *Xtr.shape[1:], device=device),))
        print(f"[train] CUDA graph captured for batch {bs} ({time.time() - t0:.0f}s)")
    best, best_ep, bad, hist = -2.0, -1, 0, []
    warm = args.warmup if args.model in LOOP_MODELS else 0
    t_start = time.time()
    ckpt = os.path.join(out_dir, "best.pt")
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot, nb, t0 = 0.0, 0, time.time()
        # graph needs a fixed batch shape, so the last partial batch is dropped (reshuffled every epoch)
        last = len(perm) - bs + 1 if use_graph else len(perm)
        for i in range(0, last, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            logits, extra = step(xb)
            loss = loss_fn(logits, to_target(yb)) + extra       # extra = Skip RNN budget loss (0 otherwise)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += loss.item(); nb += 1
        t_ep = time.time() - t0
        vm, _, _ = evaluate(model, va, task, names, device=device)
        s = val_score(vm, task)
        e = vm["efficiency"]
        hist.append(dict(epoch=ep, train_loss=tot / nb, val_score=s, val_accuracy=vm["accuracy"],
                         val_balanced_accuracy=vm["balanced_accuracy"], val_auroc=vm.get("auroc"),
                         val_auprc=vm.get("auprc"), val_updates=e["mean_updates"],
                         val_update_fraction=e["update_fraction"], val_mflops=e["mflops"], train_sec=t_ep))
        print(f"[ep {ep:02d}] loss={tot / nb:.4f}  val_{'auroc_in_rec' if task == 'binary' else 'macroF1'}={s:.4f}  "
              f"bal_acc={vm['balanced_accuracy']:.3f}  upd={100 * e['update_fraction']:.1f}%  "
              f"({t_ep:.0f}s + {time.time() - t0 - t_ep:.0f}s val)")
        if ep <= warm:
            best_ep = ep
            torch.save(model.state_dict(), ckpt)
        elif s > best:
            best, best_ep, bad = s, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("[train] early stop"); break
        if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
            print("[train] time budget reached"); break
    train_time = time.time() - t_start

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    vm, _, _ = evaluate(model, va, task, names, device=device)                    # tune threshold on val
    thr = vm.get("threshold") if task == "binary" else None
    tm, probs, eff = evaluate(model, te, task, names, threshold=thr, device=device)   # frozen for test
    lat = measure_latency(model, te["X"])
    delay = meta["delay_sec"]
    print_report(args.model, tm, task)
    print(f"  latency (batch 1, 1 CPU thread): {lat['ms_per_sequence']:.1f} ms/window median, p95 {lat['ms_p95']:.1f} | "
          f"decision delay {delay:.2f}s | train time {train_time / 60:.1f} min, best epoch {best_ep}")
    print("  per-record (test): " + ", ".join(
        f"{r}:{v['accuracy']:.2f}" for r, v in tm["per_record"].items()))

    cfg = {**vars(args), "task": task}
    json.dump(dict(config=cfg, run_name=os.path.basename(os.path.normpath(out_dir)),
                   meta={k: v for k, v in meta.items() if k != "beats"}, n_params=count_params(model),
                   best_epoch=best_ep, train_time_sec=train_time, cuda_graph=use_graph, decision_delay_sec=delay,
                   history=hist, val=vm, test=tm, latency=lat),
              open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    np.savez(os.path.join(out_dir, "test_predictions.npz"), prob=probs, y=te["y"], rec=te["rec"], start=te["start"],
             updates=eff["updates"], macs=eff["macs"], threshold=thr if thr is not None else np.nan)
    if args.model in LOOP_MODELS and task == "binary":
        save_skip_examples(model, te, probs, thr, meta, os.path.join(out_dir, "skip_examples.npz"), device)
    print(f"[done] results -> {out_dir}/results.json")


if __name__ == "__main__":
    main()
