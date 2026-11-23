"""
main testing & evaluation harness.  works for any model that follows models.py form.

evals:
  * quality  : accuracy, balanced accuracy, sensitivity, specificity, precision, F1, AUROC, AUPRC,
               confusion matrix (binary); accuracy, macro-F1, per-class P/R/F1 (multiclass)
  * per-record breakdown (patients differ a lot -- always look at it)
  * efficiency: mean/std/min/max RNN state updates per sequence, % of the T steps used, MACs per part and
               FLOPs (= 2 x MACs) per window, batch-1 CPU latency on real windows
  * record_bootstrap: CIs by resampling test records (one seed only, so this is our error bar)
Decision threshold (binary) is tuned on the VALIDATION set (Youden's J) and then frozen for the test set.
"""
import copy
import json
import sys
import time
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score)


@torch.no_grad()
def predict(model, X, batch_size=256, device="cpu"):
    # Returns scores (n,) sigmoid prob [binary] or (n,K) softmax prob, updates (n,) (nan for conv nets), MACs per part
    model.eval()
    probs, upd, macs = [], [], {}
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size]).to(device)
        logits, aux = model(xb)
        probs.append((torch.sigmoid(logits) if logits.dim() == 1 else torch.softmax(logits, -1)).cpu())
        u = aux.get("updates") if isinstance(aux, dict) else None
        upd.append((u.float().cpu() if u is not None else torch.full((len(xb),), float("nan"))))
        for k, v in aux.get("macs", {}).items():
            macs.setdefault(k, []).append(v.float().cpu())
    if not probs:
        return np.zeros(0), np.zeros(0), {}
    return torch.cat(probs).numpy(), torch.cat(upd).numpy(), {k: torch.cat(v).numpy() for k, v in macs.items()}


def tune_threshold(y, p):
    # Threshold maximising Youden's J = sens + spec - 1 (binary). F1 was used before, but with ~37% positive
    # windows "everything positive" already gives F1 ~0.55, so weak models collapsed to specificity ~0
    if len(np.unique(y)) < 2:
        return 0.5
    best_t, best_j = 0.5, -1
    for t in np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197))):
        pred = p >= t
        j = pred[y == 1].mean() - pred[y == 0].mean()
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t


def _safe(fn, *a):
    try:
        return float(fn(*a))
    except ValueError:      # examople only one class present
        return float("nan")


def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return dict(threshold=float(thr), accuracy=float((tp + tn) / max(len(y), 1)),
                balanced_accuracy=float((sens + spec) / 2), sensitivity=float(sens), specificity=float(spec),
                precision=float(prec), f1=float(2 * prec * sens / max(prec + sens, 1e-12)),
                auroc=_safe(roc_auc_score, y, p), auprc=_safe(average_precision_score, y, p),
                confusion=[[int(tn), int(fp)], [int(fn), int(tp)]], n=int(len(y)), n_pos=int(y.sum()))


def multiclass_metrics(y, p, class_names):
    pred = p.argmax(1)
    labels = list(range(len(class_names)))
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=_safe(balanced_accuracy_score, y, pred),
                macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
                per_class={c: dict(precision=float(a), recall=float(b), f1=float(c_), support=int(d))
                           for c, a, b, c_, d in zip(class_names, pr, rc, f1, sup)},
                confusion=confusion_matrix(y, pred, labels=labels).tolist(), n=int(len(y)))


def within_record_auroc(y, p, rec):
    # Mean AUROC computed inside each record that has both classes. Pooled AUROC can be inflated by recognising WHICH patient a window comes from (one record all-normal, another all-abnormal); this one only rewards separating abnormal from normal windows of the same patient
    vals = [roc_auc_score(y[rec == r], p[rec == r]) for r in np.unique(rec)
            if 0 < y[rec == r].sum() < (rec == r).sum()]
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def evaluate(model, d, task, class_names, threshold=None, batch_size=256, device="cpu"):
    # d: dict(X, y, rec)  returns metrics dict (+ per-record + efficiency), probs, and per-window updates / MACs
    p, upd, macs = predict(model, d["X"], batch_size, device)
    y, rec, T = d["y"], d["rec"], d["X"].shape[1]
    if task == "binary":
        thr = tune_threshold(y, p) if threshold is None else threshold
        m = binary_metrics(y, p, thr)
        m["auroc_within_record"], m["n_records_with_both_classes"] = within_record_auroc(y, p, rec)
        per_rec = {}
        for r in np.unique(rec):
            s = rec == r
            pr_ = (p[s] >= thr).astype(int)
            per_rec[int(r)] = dict(n=int(s.sum()), n_pos=int(y[s].sum()), accuracy=float((pr_ == y[s]).mean()),
                                   sensitivity=float(pr_[y[s] == 1].mean()) if (y[s] == 1).any() else None,
                                   specificity=float(1 - pr_[y[s] == 0].mean()) if (y[s] == 0).any() else None,
                                   mean_updates=float(upd[s].mean()),
                                   auroc=_safe(roc_auc_score, y[s], p[s]) if 0 < y[s].sum() < s.sum() else None)
    else:
        m = multiclass_metrics(y, p, class_names)
        per_rec = {int(r): dict(n=int((rec == r).sum()), accuracy=float((p[rec == r].argmax(1) == y[rec == r]).mean()),
                                mean_updates=float(upd[rec == r].mean())) for r in np.unique(rec)}
    m["per_record"] = per_rec
    tot = sum(macs.values()) if macs else np.full(len(p), np.nan)
    m["efficiency"] = dict(seq_len=int(T), mean_updates=float(upd.mean()), std_updates=float(upd.std()),
                           min_updates=float(upd.min()), max_updates=float(upd.max()),
                           update_fraction=float(upd.mean() / T),
                           macs={k: float(v.mean()) for k, v in macs.items()}, mflops=float(2 * tot.mean() / 1e6))
    return m, p, dict(updates=upd, macs=tot)


@torch.no_grad()
def measure_latency(model, X, n=30):
    # Batch-1, single-thread CPU, real test windows. Skip models use model.stream so skipped steps are really skipped.
    # Detector / feature extraction of the phys gate is counted in MACs, not timed here.
    m = copy.deepcopy(model).cpu().eval()
    run = m.stream if hasattr(m, "stream") else (lambda x: m(x[None])[0])
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    idx = np.linspace(0, len(X) - 1, min(n, len(X))).astype(int)
    run(torch.as_tensor(X[idx[0]]))
    ts = []
    for i in idx:
        x = torch.as_tensor(X[i])
        t0 = time.perf_counter()
        run(x)
        ts.append(time.perf_counter() - t0)
    torch.set_num_threads(nt)
    ts = np.array(ts) * 1e3
    return dict(n=len(ts), ms_per_sequence=float(np.median(ts)), ms_p95=float(np.percentile(ts, 95)),
                ms_per_step=float(np.median(ts)) / X.shape[1])


def record_bootstrap(y, p, rec, thr, n_boot=1000, seed=0, p_ref=None, thr_ref=None):
    """95% CI by resampling test RECORDS with replacement (windows of one patient are correlated).
    With p_ref (another model on the same windows) it returns the CI of the paired difference p - p_ref
    and the fraction of resamples where the difference is <= 0."""
    keys = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auroc", "auprc"]
    rng = np.random.RandomState(seed)
    rids = np.unique(rec)
    where = {r: np.where(rec == r)[0] for r in rids}
    out = []
    for _ in range(n_boot):
        sel = np.concatenate([where[r] for r in rng.choice(rids, len(rids))])
        a = binary_metrics(y[sel], p[sel], thr)
        s = [a[k] for k in keys]
        if p_ref is not None:
            b = binary_metrics(y[sel], p_ref[sel], thr_ref)
            s = [s_ - b[k] for s_, k in zip(s, keys)]
        out.append(s)
    out = np.array(out)
    lo, hi = np.nanpercentile(out, [2.5, 97.5], axis=0)
    res = {k: (float(l), float(h)) for k, l, h in zip(keys, lo, hi)}
    if p_ref is not None:
        res["p_le_0"] = {k: float(np.mean(out[:, i] <= 0)) for i, k in enumerate(keys)}
    return res


def print_report(name, test, task):
    print(f"\n=== {name}: TEST ===")
    if task == "binary":
        k = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auroc",
             "auroc_within_record", "auprc"]
        print("  " + "  ".join(f"{a}={test[a]:.3f}" for a in k))
        print(f"  threshold={test['threshold']:.3f}  confusion [[TN,FP],[FN,TP]]={test['confusion']}")
    else:
        print(f"  accuracy={test['accuracy']:.3f}  macro_f1={test['macro_f1']:.3f}")
        for c, v in test["per_class"].items():
            print(f"  {c}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (n={v['support']})")
    e = test["efficiency"]
    print(f"  RNN updates/seq: mean={e['mean_updates']:.1f} +- {e['std_updates']:.1f} of T={e['seq_len']} "
          f"({100 * e['update_fraction']:.1f}%) | {e['mflops']:.2f} MFLOPs/window "
          f"{ {k: round(v) for k, v in e['macs'].items()} }")


def compare(paths):
    #  table of result files 
    rows = []
    for p in paths:
        r = json.load(open(p))
        t, lat = r["test"], r.get("latency", {})
        rows.append((r.get("run_name", r["config"].get("model", p)), r["n_params"], t, lat))
    task = json.load(open(paths[0]))["config"]["task"]
    cols = (["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auroc", "auroc_within_record", "auprc"]
            if task == "binary" else ["accuracy", "macro_f1", "balanced_accuracy"])
    head = (f"{'model':<22}{'params':>8}" + "".join(f"{c.replace('_record', '_rec')[:12]:>13}" for c in cols)
            + f"{'updates':>10}{'upd %':>8}{'MFLOPs':>9}{'ms/seq':>9}")
    print(head)
    print("-" * len(head))
    for name, n, t, lat in rows:
        e = t["efficiency"]
        print(f"{name:<22}{n:>8}" + "".join(f"{t[c]:>13.3f}" for c in cols)
              + f"{e['mean_updates']:>10.1f}{100 * e['update_fraction']:>8.1f}{e.get('mflops', float('nan')):>9.2f}"
              + f"{lat.get('ms_per_sequence', float('nan')):>9.2f}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "compare":
        compare(sys.argv[2:])
    else:
        print(__doc__)
