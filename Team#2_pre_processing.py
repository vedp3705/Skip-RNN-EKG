import pathlib
import pandas as pd
import wfdb
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})

r_list=wfdb.get_record_list('mitdb')

QUALITY_SUBTYPE_DECODE = {
    0x00: "cc", # both clean
    0x01: "nc", # ch0 noisy, ch1 clean
    0x02: "cn", # ch0 clean, ch1 noisy
    0x03: "nn", # both noisy
    0x11: "uc", # ch0 unreadable, ch1 clean
    0x12: "un", # ch0 unreadable, ch1 noisy
    0x20: "cu", # ch0 clean, ch1 unreadable
    0x21: "nu", # ch0 noisy, ch1 unreadable
    0x33: "uu", # both unreadable
}

def _is_bad(letter: str) -> bool:
    # keep merely-noisy beats (still expert-labeled); drop only unreadable 'u'
    return letter in ("u",)

def build_quality_masks(ann, n_samples: int, n_channels: int = 2, u_pad_s: float = 0.5, fs_hint: float = 360):
    # Build one keep/drop mask per channel from the wfdb quality annotations.
    # True = keep, False = drop. '~' marks quality-change intervals, 'U' marks unreadable spots.
    masks = [np.ones(n_samples, dtype=bool) for _ in range(n_channels)]

    # Signal-quality change intervals ('~')
    tilde_idx = [i for i, s in enumerate(ann.symbol) if s == '~']
    events = []
    for i in tilde_idx:
        pos = ann.sample[i]
        st = int(ann.subtype[i]) if hasattr(ann, "subtype") else 0x00
        pair = QUALITY_SUBTYPE_DECODE.get(st, "cc")
        events.append((pos, pair))
    # Pair consecutive '~' into [start,end) intervals
    events.sort(key=lambda t: t[0])
    if len(events) % 2 == 1:
        events.append((n_samples, events[-1][1])) # odd count -> last interval runs to end
    for (a, pair_a), (b, pair_b) in zip(events[0::2], events[1::2]):
        state = pair_a
        for ch in range(min(n_channels, 2)): # MIT-BIH has 2 channels
            if _is_bad(state[ch]):
                masks[ch][a:b] = False

    # 'U' (unreadable) - drop a short window around each event
    fs = getattr(ann, "fs", fs_hint)
    u_pad = int(round(u_pad_s * fs))
    for pos, sym in zip(ann.sample, ann.symbol):
        if sym == 'U':
            a = max(0, pos - u_pad)
            b = min(n_samples, pos + u_pad)
            for ch in range(n_channels):
                masks[ch][a:b] = False

    return masks

def combine_keep_mask(masks, mode="all", channel=None):
    # 'all' = AND across channels, 'any' = OR, or pick one channel with channel=int
    masks = [np.asarray(m, dtype=bool) for m in masks]
    if channel is not None:
        return masks[channel]
    if mode == "all":
        return np.logical_and.reduce(masks)
    elif mode == "any":
        return np.logical_or.reduce(masks)
    else:
        raise ValueError("mode must be 'all', 'any', or provide channel=int")

def compact_and_reindex(signal, ann, keep_mask):
    # Drop the masked-out samples and re-map the annotation indices onto the shortened signal
    keep_mask = np.asarray(keep_mask, dtype=bool)
    N = keep_mask.size
    keep_idx = np.flatnonzero(keep_mask)

    x_compact = signal[keep_idx] if signal.ndim == 1 else signal[keep_idx, :]

    inv = np.full(N, -1, dtype=np.int64)
    inv[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)

    idx = np.asarray(ann.sample)
    sym = np.asarray(ann.symbol)
    if idx.size and idx.max() >= N:
        raise ValueError("Annotation index exceeds signal length.")
    survive = inv[idx] != -1
    ann_idx_new = inv[idx[survive]]
    ann_sym_new = sym[survive]
    return x_compact, ann_idx_new, ann_sym_new, keep_idx

# not used in this run, kept in case we want to write the cleaned annotations back out
def write_reindexed_ann(record_path, samples_new, symbols_new, ext="qc"):
    wfdb.wrann(record_path, ext,
               sample=samples_new.tolist(),
               symbol=symbols_new.tolist())


fs= 360
bandpass_filter = scipy.signal.firwin(1001, [0.5, 40], pass_zero=False, fs=fs, window=('kaiser',14))
w, h = scipy.signal.freqz(bandpass_filter,fs)

pre_proc_dir = "Preprocessed Dataset"
#Save next to this script, not the current working directory
dir = pathlib.Path(__file__).parent / pre_proc_dir
dir.mkdir(parents=True, exist_ok=True)

#Download the raw database once so we read from disk instead of re-streaming every run
raw_dir = pathlib.Path(__file__).parent / "mitdb_raw"
#Re-download only if any record's .dat is missing (handles interrupted downloads)
have_all = all((raw_dir / f"{rec}.dat").exists() for rec in r_list)
if not have_all:
    print("Downloading mitdb (one-time)...")
    wfdb.dl_database('mitdb', str(raw_dir))

#Record to draw the sanity-check plot for (visual spot-check)
plot_rec = "100"
plot_data = None

for i, rec in enumerate(r_list, start=1):
    #Skip records we've already preprocessed
    if (dir / f"{rec}_signal.npy").exists():
        print(f"[{i}/{len(r_list)}] {rec}: already done, skipping")
        continue

    r = wfdb.rdrecord(str(raw_dir / rec))
    ann = wfdb.rdann (str(raw_dir / rec), 'atr')

    x = r.p_signal # shape (N, C)
    N, C = x.shape[0], x.shape[1]
    signal_names = r.sig_name

    #Filter before masking so seams from dropped samples don't cause filter ringing
    filt_x = scipy.signal.filtfilt(bandpass_filter, 1.0, x, axis=0)

    masks = build_quality_masks(ann, N, n_channels=C, u_pad_s=0.5, fs_hint=r.fs)
    keep_mask = combine_keep_mask(masks, mode="all") # a sample must be clean in all channels

    x_compact, ann_idx_new, ann_sym_new, keep_idx = compact_and_reindex(filt_x, ann, keep_mask)
    tx = np.arange(x_compact.shape[0]) / fs

    assert not np.isnan(x_compact).any(), f"{rec}: NaNs in cleaned signal"
    assert ann_idx_new.size == 0 or ann_idx_new.max() < x_compact.shape[0], f"{rec}: label out of range"

    #Per-record z-score per channel so patient gain differences don't reach the model
    x_compact = (x_compact - x_compact.mean(axis=0)) / (x_compact.std(axis=0) + 1e-8)

    kept_pct = 100 * x_compact.shape[0] / N
    print(f"[{i}/{len(r_list)}] {rec}: samples {N} -> {x_compact.shape[0]} ({kept_pct:.1f}% kept), "
          f"labels {len(ann.sample)} -> {len(ann_idx_new)}")

    data_tag = dir / f"{rec}"
    np.save(f"{data_tag}_signal.npy", x_compact)

    pd.DataFrame({"sample":ann_idx_new,"symbol":ann_sym_new}).to_csv(f"{data_tag}_ann.csv", index=False)

    if rec == plot_rec:
        plot_data = (tx, x_compact, ann_idx_new, ann_sym_new, signal_names, r.record_name)

print("Done. Preprocessed all records.")

#If the plot record was skipped this run, load it back from the saved files
if plot_data is None:
    sig = np.load(dir / f"{plot_rec}_signal.npy")
    tbl = pd.read_csv(dir / f"{plot_rec}_ann.csv")
    names = wfdb.rdheader(str(raw_dir / plot_rec)).sig_name
    plot_data = (np.arange(sig.shape[0]) / fs, sig,
                 tbl["sample"].to_numpy(), tbl["symbol"].to_numpy(), names, plot_rec)

#Quick sanity-check plot for one chosen record
tx, x_compact, ann_idx_new, ann_sym_new, signal_names, record_name = plot_data

#Zoom to a few seconds so individual beats are visible (a full 30-min record looks like a solid band)
plot_secs = 10
win = tx <= plot_secs
plt.plot(tx[win], x_compact[win, 0])

plt.title(f'Record {record_name} (first {plot_secs} s)')

show = set(['N','V','A','L','R','B'])
sel = [i for i, s in enumerate(ann_sym_new) if s in show and ann_idx_new[i] < plot_secs * fs]
if sel:
    si = ann_idx_new[sel]
    plt.scatter(si / fs, x_compact[si, 0], s=12, color='r', label="Beats")
    for ti, s in zip(si, np.array(ann_sym_new)[sel]):
            plt.text(ti / fs, x_compact[ti, 0] + 0.05, s, fontsize=8, color='r', ha='center')

plt.xlabel('Time (sec)')
plt.ylabel(f'{signal_names[0]} Signal')
plt.legend()
plt.show()