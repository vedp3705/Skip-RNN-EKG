"""Run the two fixed ECG comparisons, one model at a time."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen

import pandas as pd

from models import cell_macs

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
DATA = ROOT / "Data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
DATABASE = "https://physionet.org/files/mitdb/1.0.0/"
SEED = 2
PAPER = ("gru", "skip_gru", "random_gru_updates", "lstm", "skip_lstm", "random_lstm_updates")
EXTENSION = ("phys_skip_gru", "random_gru_compute", "mlp_skip_gru", "phys_skip_lstm", "mlp_skip_lstm", "tcn")
NAMES = {
    "gru": "GRU (no skipping)", "skip_gru": "Skip-GRU",
    "random_gru_updates": "Random GRU (same update rate as Skip-GRU)",
    "lstm": "LSTM (no skipping)", "skip_lstm": "Skip-LSTM",
    "random_lstm_updates": "Random LSTM (same update rate as Skip-LSTM)",
    "phys_skip_gru": "Physiology-guided GRU",
    "random_gru_compute": "Random GRU (same compute as physiology-guided GRU)",
    "mlp_skip_gru": "GRU with MLP gate (no physiology)",
    "phys_skip_lstm": "Physiology-guided LSTM",
    "mlp_skip_lstm": "LSTM with MLP gate (no physiology)",
    "tcn": "Temporal convolutional network (TCN)",
}
RANDOM_TARGETS = {
    "random_gru_updates": ("rand_gru", "skip_gru"),
    "random_lstm_updates": ("rand_lstm", "skip_lstm"),
    "random_gru_compute": ("rand_gru", "phys_skip_gru"),
}


def prepare_data(download_missing=False, folder=DATA):
    """Verify local records. The network is used only when explicitly requested."""
    manifest = (SCRIPTS / "record_checksums.txt").read_text(encoding="ascii").splitlines()
    repair = []
    for line in manifest:
        expected, name = line.split()
        path = folder / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            repair.append((path, expected))
    if repair and not download_missing:
        raise FileNotFoundError(f"{len(repair)} missing or damaged recording files, including "
                                f"{repair[0][0].name}. Set DOWNLOAD_MISSING = True and rerun setup.")
    if repair:
        folder.mkdir(parents=True, exist_ok=True)
    for path, expected in repair:
        # An interrupted download must not replace a complete recording.
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            part = Path(temp) / path.name
            with urlopen(DATABASE + path.name, timeout=60) as response, part.open("wb") as target:
                shutil.copyfileobj(response, target)
            if hashlib.sha256(part.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Download did not match the database checksum: {path.name}")
            part.replace(path)
    print(f"{len(manifest)//3} ECG records ready. Downloaded {len(repair)} files.")


def result(key):
    return json.loads((RESULTS / key / "results.json").read_text(encoding="utf-8"))


def random_probability(key):
    _, parent = RANDOM_TARGETS[key]
    target = result(parent)["val"]["efficiency"]
    steps = target["seq_len"]
    updates = target["mean_updates"]
    if key == "random_gru_compute":
        baseline = result("gru")
        steps = baseline["meta"]["seq_len"]
        macs = sum(target["macs"].values())
        head = baseline["val"]["efficiency"]["macs"]["head"]
        updates = (macs - head) / cell_macs("gru", baseline["meta"]["n_sig"], baseline["config"]["hidden"])
    # Random skipping always updates the first sample; only the remaining steps are random.
    p = 1 - (updates - 1) / (steps - 1)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError(f"Validation budget from {NAMES[parent]} cannot be matched: p_skip={p}")
    return p


def run_section(keys):
    for key in keys:
        out = RESULTS / key
        model = RANDOM_TARGETS[key][0] if key in RANDOM_TARGETS else key
        flags = ["--model", model, "--seed", str(SEED), "--lam", "1e-5", "--hidden", "32",
                 "--layers", "1", "--epochs", "60", "--patience", "15", "--warmup", "5",
                 "--channels", "auto", "--win_sec", "10", "--delay_sec", "0.5",
                 "--decim", "2", "--train_overlap", "0", "--split", "fixed"]
        if key in RANDOM_TARGETS:
            flags.extend(["--p_skip", str(random_probability(key))])
        # Resume only a finished run from this protocol, even if the notebook was interrupted.
        digest = hashlib.sha256()
        for name in ("ecg_data.py", "train.py", "models.py", "harness.py", "record_checksums.txt"):
            digest.update((SCRIPTS / name).read_bytes())
        signature = {"args": flags, "code_and_data": digest.hexdigest()}
        done = out / "finished.json"
        if done.exists() and json.loads(done.read_text()) == signature and all(
                (out / n).is_file() for n in ("results.json", "best.pt", "test_predictions.npz")):
            print(f"Using completed run: {NAMES[key]}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        done.unlink(missing_ok=True)
        print(f"Training {NAMES[key]} ...", flush=True)
        cmd = [sys.executable, "-u", str(SCRIPTS / "train.py"), *flags, "--data_dir", str(DATA),
               "--cache_dir", str(CACHE), "--out_dir", str(out)]
        log = out / "training.log"
        with log.open("w", encoding="utf-8") as f:
            status = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if status.returncode:
            raise RuntimeError(f"Training failed; see {log}\n{log.read_text(errors='replace')[-1800:]}")
        temp = out / "finished.tmp"
        temp.write_text(json.dumps(signature), encoding="utf-8")
        temp.replace(done)
        print(f"Finished {NAMES[key]} (best epoch: {result(key)['best_epoch']}).", flush=True)


def results_table(keys, extended=False):
    rows = []
    for key in keys:
        t = result(key)["test"]
        e = t["efficiency"]
        row = {"Model": NAMES[key], "Correct predictions (%)": 100*t["accuracy"]}
        if extended:
            row.update({"Positive windows detected (%)": 100*t["sensitivity"],
                        "Negative windows correctly rejected (%)": 100*t["specificity"],
                        "Within-record AUROC": t["auroc_within_record"],
                        "Estimated MFLOPs per window": e["mflops"]})
        else:
            row["Mean updates per window"] = e["mean_updates"]
            row["Estimated MFLOPs per window"] = e["mflops"]
        row["Samples skipped (%)"] = None if key == "tcn" else 100*(1-e["update_fraction"])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / ("extended_results.csv" if extended else "paper_style_results.csv"), index=False)
    return table.round({"Within-record AUROC": 3, **{c: 2 for c in table if c != "Within-record AUROC" and c != "Model"}})
