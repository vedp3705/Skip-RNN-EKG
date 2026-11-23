"""
Task: each sample is one fixed-length ECG window (a long 1-D sequence); the label says
whether the window contains an abnormal (ectopic) heartbeat.  Long sequences with lots
of redundancy (flat baseline between QRS complexes) are exactly the regime Skip RNN targets.

Split: RECORD-level (inter-patient).  Windows from the same recording are highly correlated,
so a window-level random split would leak and massively inflate scores.  The split is defined
on record IDs (not on whatever happens to be on disk), so it stays identical whether you run
on the 17-record sample or on all 48 records.
"""
import json
import os
import numpy as np
from scipy.signal import butter, lfilter, sosfilt

PREP_VERSION = "v4"     # bump when preprocessing or the default window protocol changes
NORM_SEC = 10.0         # causal running amplitude scale; windows start after this much signal

# Records  201/202 = same subject).
ALL_RECORDS = (
    list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125))
    + [200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217,
       219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
)
assert len(ALL_RECORDS) == 48

#   * 202 moved from DS2 to the train side: 201 and 202 are the SAME subject (PhysioNet note),
#     the original DS1/DS2 split puts them on opposite sides, which leaks.
#   * Paced records 102, 104, 107, 217 are in the split but dropped by default in build_datasets
#     (AAMI / de Chazal); 102 and 104 also have no MLII. --keep_paced puts 107 (train) and 217 (test) back.
FIXED_SPLIT = {
    "val":  [108, 114, 118, 205, 209, 223],
    "test": [100, 103, 104, 105, 111, 113, 117, 121, 123, 200, 210, 212, 213, 214,
             217, 219, 221, 222, 228, 231, 232, 233, 234],
}
FIXED_SPLIT["train"] = [r for r in ALL_RECORDS
                        if r not in FIXED_SPLIT["val"] and r not in FIXED_SPLIT["test"]]
assert sorted(sum(FIXED_SPLIT.values(), [])) == sorted(ALL_RECORDS)

SAME_SUBJECT_GROUPS = [(201, 202)]

# 102/104 have no MLII (V5/V2 only) so they cannot be used with a single MLII model.
# Paced records are dropped by default as in AAMI / de Chazal (--keep_paced keeps 107, 217).
NO_MLII = [102, 104]
PACED = [102, 104, 107, 217]


CLASS_NAMES = ["N", "S", "V", "F", "Q"]
BEAT_TO_AAMI = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,      # normal, LBBB, RBBB, atrial/nodal escape
    "A": 1, "a": 1, "J": 1, "S": 1,              # supraventricular ectopic
    "V": 2, "E": 2, "!": 2,                       # ventricular ectopic, ventricular flutter wave (207)
    "F": 3,                                       # fusion of ventricular + normal
    "/": 4, "f": 4, "Q": 4,                       # paced, fusion of paced+normal, unclassifiable
}

# physiological gate inputs: first 8 are per-beat, dt and event are per step
FEAT_NAMES = ["rr", "rr_rel", "rr_prev", "corr", "amp", "e_hi", "e_mid", "e_lo", "dt", "event"]
FEAT_GROUPS = {"rr": [0, 1, 2, 8], "morph": [3, 4, 5, 6, 7]}
DET_MACS = 22       # detector cost per raw 360 Hz sample (bandpass, derivative, square, MWI, highpass)
FEAT_MACS = 400     # per detected beat (template corr, amplitude, haar energies, template update)

def window_labels(counts, mode):
    """counts: (n, 5) beats per AAMI class inside each window."""
    if mode == "binary":      # 1 = window contains >=1 S, V or F beat. Q (paced/unknown) is ignored.
        return ((counts[:, 1] + counts[:, 2] + counts[:, 3]) > 0).astype(np.int64)
    if mode == "aami4":       # 0=N, 1=S, 2=V, 3=F ; priority V > F > S > N
        y = np.zeros(len(counts), dtype=np.int64)
        y[counts[:, 1] > 0] = 1
        y[counts[:, 3] > 0] = 3
        y[counts[:, 2] > 0] = 2
        return y
    raise ValueError(mode)


def label_names(mode):
    return ["normal", "abnormal"] if mode == "binary" else ["N", "S", "V", "F"]


def resolve_data_dir(data_dir):
    """Accept .../Data, .../Data/SampleData, or a flat folder of record folders."""
    for cand in (os.path.join(data_dir, "SampleData"), os.path.join(data_dir, "Data", "SampleData"), data_dir):
        if os.path.isdir(cand) and any(
                os.path.exists(os.path.join(cand, str(r), f"{r}.hea")) or os.path.exists(os.path.join(cand, f"{r}.hea"))
                for r in ALL_RECORDS):
            return cand
    raise FileNotFoundError(f"No MIT-BIH records found under {data_dir}")


def record_path(root, rid):
    for p in (os.path.join(root, str(rid), str(rid)), os.path.join(root, str(rid))):
        if os.path.exists(p + ".hea"):
            return p
    return None


def discover_records(data_dir):
    root = resolve_data_dir(data_dir)
    present = [r for r in ALL_RECORDS if record_path(root, r) is not None]
    return root, present, [r for r in ALL_RECORDS if r not in present]


def _pick_channels(sig_names, channels):
    """auto: MLII (by name, 114 has it on channel 1). both: [MLII, the other]. Records without MLII -> None."""
    if channels in ("auto", "both"):
        if "MLII" not in sig_names:
            return None
        first = sig_names.index("MLII")
        if channels == "auto":
            return [first]
        return [first] + [i for i in range(len(sig_names)) if i != first]
    return [int(channels)]


def detect_r_causal(x, fs=360):
    """Pan-Tompkins style R-peak detector that only uses past samples.
    Returns R locations and the sample at which each one is detected (both at fs)."""
    b = sosfilt(butter(2, [5, 15], btype="bandpass", fs=fs, output="sos"), x)
    d = np.zeros_like(b)
    d[4:] = (2 * b[4:] + b[3:-1] - b[1:-3] - 2 * b[:-4]) / 8
    w = int(0.15 * fs)
    cs = np.cumsum(np.concatenate([np.zeros(w), d ** 2]))
    mwi = ((cs[w:] - cs[:-w]) / w).tolist()
    hp = np.abs(sosfilt(butter(2, 0.5, btype="highpass", fs=fs, output="sos"), x))
    ad = np.abs(d)

    # a MWI peak is confirmed once the signal drops to half of it
    cand = []
    pk, pki, tr, rising = 0.0, 0, 0.0, True
    for n, v in enumerate(mwi):
        if rising:
            if v > pk:
                pk, pki = v, n
            elif v < 0.5 * pk:
                cand.append((pki, pk, n))
                rising, tr = False, v
        elif v <= tr:
            tr = v
        else:
            rising, pk, pki = True, v, n

    learn = 2 * fs
    spk, npk = 0.25 * max(mwi[:learn]), 0.5 * float(np.mean(mwi[:learn]))
    thr = npk + 0.25 * (spk - npk)
    ref, twave, sw, lb = int(0.2 * fs), int(0.36 * fs), int(0.075 * fs), int(0.25 * fs)
    rr, out = [], []
    last_r, last_slope, best = -10 * fs, None, None

    def accept(i, v, t, searchback):
        nonlocal spk, last_r, last_slope, best
        r = i - lb + int(np.argmax(hp[max(i - lb, 0):i + 1]))
        if out and r - last_r < ref:
            return
        if out:
            rr.append(r - last_r)
            del rr[:-8]
        spk = 0.25 * v + 0.75 * spk if searchback else 0.125 * v + 0.875 * spk
        last_r, last_slope, best = r, ad[max(i - sw, 0):i + 1].max(), None
        out.append((r, t))

    for i, v, t in cand:
        if t < learn:
            continue
        trig = last_r + int(1.66 * (np.mean(rr) if rr else fs))
        if best is not None and trig < t:
            # search-back runs when the next candidate is confirmed, so the beat is only known at t
            accept(best[0], best[1], t, True)
            thr = npk + 0.25 * (spk - npk)
        if i - last_r < ref:
            continue
        if v > thr:
            if i - last_r < twave and ad[max(i - sw, 0):i + 1].max() < 0.5 * last_slope:
                npk = 0.125 * v + 0.875 * npk
            else:
                accept(i, v, t, False)
        else:
            npk = 0.125 * v + 0.875 * npk
            if v > 0.5 * thr and (best is None or v > best[1]):
                best = (i, v, t)
        thr = npk + 0.25 * (spk - npk)
    out = np.array(out, dtype=np.int64).reshape(-1, 2)
    return out[:, 0], out[:, 1]


def _haar_bands(seg):
    # relative energy in ~22-90 Hz, 11-22 Hz, <11 Hz for a 64 sample beat at 360 Hz
    a, e = seg, []
    for _ in range(4):
        e.append(np.sum(((a[0::2] - a[1::2]) / np.sqrt(2)) ** 2))
        a = (a[0::2] + a[1::2]) / np.sqrt(2)
    e = np.array([e[1] + e[2], e[3], np.sum(a ** 2)])
    return e / (e.sum() + 1e-12)


def beat_features(hp, r, t_det, fs=360, half=32, alpha=0.05):
    """Per detected beat: RR timing + morphology relative to a running template (causal).
    Returns the sample where the features become available and an (n, 8) feature table."""
    ev = np.maximum.accumulate(np.maximum(t_det, r + half))
    f = np.zeros((len(r), 8), np.float32)
    tmpl, rrs = None, []
    for k in range(len(r)):
        if k > 0:
            rr = (r[k] - r[k - 1]) / fs
            f[k, 0] = rr
            if rrs:
                f[k, 1] = np.log(rr / np.mean(rrs))
                f[k, 2] = np.log(rr / rrs[-1])
            rrs = (rrs + [rr])[-8:]
        if r[k] - half < 0 or r[k] + half > len(hp):
            continue
        seg = hp[r[k] - half:r[k] + half]
        if tmpl is None:
            tmpl = seg.copy()
        c = np.corrcoef(seg, tmpl)[0, 1]
        f[k, 3] = 0.0 if np.isnan(c) else c
        f[k, 4] = np.log((np.ptp(seg) + 1e-6) / (np.ptp(tmpl) + 1e-6))
        f[k, 5:8] = _haar_bands(seg) - _haar_bands(tmpl)
        tmpl = (1 - alpha) * tmpl + alpha * seg
    return ev, f


def unreadable_mask(ann, ch, n):
    # '~' subtype: bits 0-3 noisy, bits 4-7 unreadable (per signal), -1 = all unreadable. State holds until next '~'.
    q = [(int(s), int(st)) for s, sym, st in zip(ann.sample, ann.symbol, ann.subtype) if sym == "~"]
    bad = np.zeros(n, bool)
    for (s, st), (s_next, _) in zip(q, q[1:] + [(n, 0)]):
        if st == -1 or (st > 0 and (st >> (4 + ch)) & 1):
            bad[s:s_next] = True
    return bad


def phys_stream(rec, decim, delay, fs=360):
    """Gate features per (decimated) step. The RNN reads sample n while real time is n + delay,
    so the features are whatever the detector has produced by then."""
    L = len(rec["sig"])
    pos = np.arange(L) * decim
    k = np.searchsorted(rec["ev"], pos + delay * decim, "right") - 1
    ok = k >= 0
    F = np.zeros((L, len(FEAT_NAMES)), np.float32)
    F[ok, :8] = rec["bfeat"][k[ok]]
    F[:, 8] = -2.0
    F[ok, 8] = np.clip((rec["r_idx"][k[ok]] - pos[ok]) / fs, -2, 2)
    F[1:, 9] = (k[1:] != k[:-1]) & ok[1:]
    return F


def preprocess(raw, fs=360, decim=2):
    """raw (L, C) at fs -> model signal at fs/decim + detector output / beat features at fs.
    Every step only uses past samples, so a prefix of raw gives a prefix of the output."""
    # High-pass only (causal) for baseline wander; the recordings are already 0.1-100 Hz analog.
    x = sosfilt(butter(2, 0.5, btype="highpass", fs=fs, output="sos"), raw, axis=0)
    hp = x[:, 0].copy()
    if decim > 1:
        # Anti-alias at 80% of the new Nyquist (72 Hz for decim=2).
        x = sosfilt(butter(8, 0.8 * fs / decim / 2, fs=fs, output="sos"), x, axis=0)[::decim]
    # Lead gain differs across records: divide by a running RMS over the past ~NORM_SEC (bias-corrected EMA),
    # the high-pass already removed the mean. Clipped against artefacts.
    a = 1.0 / (NORM_SEC * fs / decim)
    ema = lambda v: lfilter([a], [1.0, a - 1.0], v, axis=0)
    x = np.clip(x / (np.sqrt(ema(x ** 2) / ema(np.ones_like(x))) + 1e-8), -10, 10).astype(np.float32)
    r, t_det = detect_r_causal(raw[:, 0], fs)
    ev, bf = beat_features(hp, r, t_det, fs)
    return {"sig": x, "r_idx": r, "r_det": t_det, "ev": ev, "bfeat": bf}


def load_record(root, rid, channels="auto", decim=2, cache_dir=None):
    """Returns dict(sig (L,C) float32, beat_idx (B,), beat_cls (B,), bad (L,)) at the decimated rate,
    plus detector output r_idx / r_det / ev / bfeat at 360 Hz. None if the record has no MLII."""
    key = f"{rid}_{channels}_d{decim}_{PREP_VERSION}.npz"
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cp = os.path.join(cache_dir, key)
        if os.path.exists(cp):
            with np.load(cp, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
    import wfdb  # imported lazily so cached runs don't need it
    p = record_path(root, rid)
    rec = wfdb.rdrecord(p)
    ann = wfdb.rdann(p, "atr")
    fs = rec.fs
    ch = _pick_channels(rec.sig_name, channels)
    if ch is None:
        return None
    raw = rec.p_signal[:, ch].astype(np.float64)
    out = preprocess(raw, fs, decim)
    idx, cls = [], []
    for s, sym in zip(ann.sample, ann.symbol):
        if sym in BEAT_TO_AAMI:
            idx.append(s // decim)
            cls.append(BEAT_TO_AAMI[sym])
    out.update(beat_idx=np.array(idx, dtype=np.int64), beat_cls=np.array(cls, dtype=np.int64),
               bad=unreadable_mask(ann, ch[0], len(raw))[::decim][:len(out["sig"])])
    if cache_dir:
        # A disconnected Colab runtime must not leave a half-written record in the cache.
        part = cp + ".partial.npz"
        np.savez_compressed(part, **out)
        os.replace(part, cp)
    return out


def make_windows(rec, win, stride, reserve=0, first=0, extra=0):
    """Slice a record into windows. Returns X (n, win + extra, C) float32, counts (n, 5), starts (n,).
    Labels count the beats in [start, start + win). `reserve` samples after each window must also exist
    (decision delay, the last `extra` of them are model input) and windows touching an unreadable segment
    are dropped, so every model sees the same windows. Windows start at `first` (normaliser warm-up)."""
    sig = rec["sig"]
    L = len(sig)
    starts = np.arange(first, L - win - reserve + 1, stride)
    if "bad" in rec and rec["bad"].any():
        cb = np.concatenate([[0], np.cumsum(rec["bad"])])
        starts = starts[cb[starts + win + reserve] - cb[starts] == 0]
    X = np.lib.stride_tricks.sliding_window_view(sig, win + extra, axis=0)[starts]      # (n, C, win + extra)
    X = np.ascontiguousarray(X.transpose(0, 2, 1))                                       # (n, win + extra, C)
    counts = np.zeros((len(starts), len(CLASS_NAMES)), dtype=np.int64)
    bi, bc = rec["beat_idx"], rec["beat_cls"]
    for c in range(len(CLASS_NAMES)):
        pos = np.sort(bi[bc == c])
        counts[:, c] = np.searchsorted(pos, starts + win, "left") - np.searchsorted(pos, starts, "left")
    return X, counts, starts


def fixed_split(present):
    return {k: [r for r in v if r in present] for k, v in FIXED_SPLIT.items()}


def grouped_kfold_split(present, abnormal_frac, fold, n_folds):
    """Record-level stratified K-fold. Records are sorted by abnormal-window fraction and dealt
    round-robin into folds (so every fold has a similar mix); 201+202 are kept together.
    test = fold k, val = fold (k+1) % n, train = the rest."""
    groups, seen = [], set()
    for r in present:
        if r in seen:
            continue
        g = [r]
        for grp in SAME_SUBJECT_GROUPS:
            if r in grp:
                g = [x for x in grp if x in present]
        seen.update(g)
        groups.append(g)
    groups.sort(key=lambda g: (np.mean([abnormal_frac[r] for r in g]), g[0]))
    folds = [[] for _ in range(n_folds)]
    for i, g in enumerate(groups):
        folds[i % n_folds].extend(g)
    test, val = folds[fold], folds[(fold + 1) % n_folds]
    train = [r for f in range(n_folds) if f not in (fold, (fold + 1) % n_folds) for r in folds[f]]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def build_datasets(data_dir, win_sec=10.0, decim=2, channels="auto", label_mode="binary",
                   split="fixed", fold=0, n_folds=5, train_overlap=0.0, cache_dir=None, verbose=True,
                   phys=False, delay_sec=0.5, keep_paced=False):
    """
    Returns dict: split -> dict(X (n,T,C), y (n,), rec (n,), start (n,)), plus meta.
    Validation / test label windows do not overlap; training labels overlap by `train_overlap`.
    Adjacent observation horizons can overlap by the decision delay, within the same record.
    phys=True appends the gate features (FEAT_NAMES) as extra channels after the signal channels.
    Every model decides delay_sec after the window ends: the phys gate uses beats confirmed by then,
    the other models get those delay_sec of ECG appended to their input instead.
    """
    root, present, missing = discover_records(data_dir)
    excluded = sorted(set(PACED if not keep_paced else []) | set(NO_MLII if channels in ("auto", "both") else []))
    present = [r for r in present if r not in excluded]
    fs = 360 // decim if 360 % decim == 0 else 360 / decim
    win = int(round(win_sec * 360 / decim))
    delay = int(round(delay_sec * 360 / decim))
    first = int(round(NORM_SEC * 360 / decim))
    extra = 0 if phys else delay
    if verbose:
        print(f"[data] {len(present)}/48 records used from {root}, excluded {excluded}")
        if missing:
            print(f"[data] not on disk (skipped, will be used automatically once added): {missing}")
        print(f"[data] window = {win_sec}s -> T={win}+{extra} steps @ {fs:g} Hz, channels={channels}, "
              f"labels={label_mode}, delay={delay} steps, phys={phys}, first start {NORM_SEC:g}s")

    recs = {r: load_record(root, r, channels, decim, cache_dir) for r in present}
    n_sig = recs[present[0]]["sig"].shape[1]
    if phys:
        for r in present:
            recs[r] = dict(recs[r], sig=np.concatenate([recs[r]["sig"], phys_stream(recs[r], decim, delay)], 1))

    if split == "fixed":
        parts = fixed_split(present)
    else:
        frac = {}
        for r in present:
            _, cnt, _ = make_windows(recs[r], win, win, delay, first)
            frac[r] = float(window_labels(cnt, "binary").mean())
        parts = grouped_kfold_split(present, frac, fold, n_folds)

    stride_train = max(1, int(round(win * (1 - train_overlap))))
    data = {}
    for name, rids in parts.items():
        Xs, ys, rs, ss = [], [], [], []
        for r in rids:
            X, cnt, st = make_windows(recs[r], win, stride_train if name == "train" else win, delay, first, extra)
            Xs.append(X); ys.append(window_labels(cnt, label_mode))
            rs.append(np.full(len(X), r)); ss.append(st)
        if Xs:
            data[name] = dict(X=np.concatenate(Xs), y=np.concatenate(ys),
                              rec=np.concatenate(rs), start=np.concatenate(ss))
        else:
            data[name] = dict(X=np.zeros((0, win + extra, recs[present[0]]["sig"].shape[1]), np.float32),
                              y=np.zeros(0, np.int64), rec=np.zeros(0, int), start=np.zeros(0, int))
    meta = dict(win=win, seq_len=win + extra, fs=fs, decim=decim, channels=data["train"]["X"].shape[-1], n_sig=n_sig,
                n_feats=len(FEAT_NAMES) if phys else 0, delay_steps=delay, delay_sec=delay / fs,
                first_start=first, norm_sec=NORM_SEC,
                label_mode=label_mode, records={k: v for k, v in parts.items()}, missing=missing,
                excluded=excluded, split=split, prep_version=PREP_VERSION,
                beats={r: (recs[r]["beat_idx"], recs[r]["beat_cls"]) for r in present})
    if verbose:
        for name in ("train", "val", "test"):
            d = data[name]
            if len(d["y"]) == 0:
                print(f"[data] {name:5s}: EMPTY (no records of this split on disk)")
                continue
            dist = np.bincount(d["y"], minlength=2 if label_mode == "binary" else 4)
            print(f"[data] {name:5s}: {len(d['y']):6d} windows from {len(parts[name]):2d} records {parts[name]} | "
                  f"class counts {dist.tolist()}")
    return data, meta


if __name__ == "__main__":     
    import sys
    d, m = build_datasets(sys.argv[1], cache_dir="cache")
    print(json.dumps({k: v for k, v in m.items() if k not in ("records", "beats")}, indent=1))
