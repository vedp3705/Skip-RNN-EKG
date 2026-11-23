import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import figures
from models import cell_macs
import study


class StudyTests(unittest.TestCase):
    def test_missing_data_requires_opt_in_and_verified_download(self):
        with tempfile.TemporaryDirectory() as temp:
            scripts = Path(temp)
            data = scripts / "Data"
            sample = b"example header"
            (scripts / "record_checksums.txt").write_text(
                f"{hashlib.sha256(sample).hexdigest()}  100.hea\n", encoding="ascii")
            with patch.object(study, "SCRIPTS", scripts), patch.object(study, "urlopen") as fetch:
                with self.assertRaises(FileNotFoundError):
                    study.prepare_data(False, data)
                fetch.assert_not_called()
                fetch.return_value = io.BytesIO(b"incomplete")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    study.prepare_data(True, data)
                self.assertFalse((data / "100.hea").exists())
                fetch.return_value = io.BytesIO(sample)
                study.prepare_data(True, data)
                fetch.reset_mock()
                study.prepare_data(True, data)
                fetch.assert_not_called()
                self.assertEqual((data / "100.hea").read_bytes(), sample)

    def test_random_controls_match_the_intended_validation_budget(self):
        cell_cost = cell_macs("gru", 1, 32)
        runs = {
            "skip_gru": {"val": {"efficiency": {"seq_len": 11, "mean_updates": 4}}},
            "skip_lstm": {"val": {"efficiency": {"seq_len": 11, "mean_updates": 7}}},
            "phys_skip_gru": {"val": {"efficiency": {"seq_len": 10, "mean_updates": 2,
                "macs": {"cell": 2 * cell_cost, "features": 3 * cell_cost, "head": 32}}}},
            "gru": {"meta": {"seq_len": 11, "n_sig": 1}, "config": {"hidden": 32},
                    "val": {"efficiency": {"macs": {"head": 32}}}},
        }
        with patch.object(study, "result", side_effect=runs.__getitem__):
            for name, updates in (("random_gru_updates", 4), ("random_lstm_updates", 7),
                                  ("random_gru_compute", 5)):
                self.assertAlmostEqual(1 + 10 * (1 - study.random_probability(name)), updates)

    def test_training_plots_use_recorded_metrics_and_selected_epoch(self):
        def result(key):
            steps = 1800 if key.startswith("phys") else 1890
            count = 2 if key.startswith("phys") else 3
            return {"best_epoch": 2, "history": [
                {"epoch": i + 1, "train_loss": .9 - .1 * i, "val_score": .9 - .1 * i,
                 "val_auroc": .1, "val_updates": steps * fraction,
                 "val_update_fraction": fraction}
                for i, fraction in enumerate([1., .5, .25][:count])]}

        with patch.object(study, "result", side_effect=result), patch.object(figures, "save_figure") as save:
            try:
                figures.training_history()
                self.assertEqual(save.call_count, 3)
                gru, lstm, updates = [call.args[0] for call in save.call_args_list]
                for learning in (gru, lstm):
                    for ax in learning.axes:
                        np.testing.assert_allclose(ax.lines[0].get_ydata(), [.9, .8, .7])
                        np.testing.assert_array_equal(ax.lines[1].get_xdata(), [2])
                        np.testing.assert_allclose(ax.lines[1].get_ydata(), [.8])
                        np.testing.assert_array_equal(ax.lines[4].get_xdata(), [1, 2])
                for ax in updates.axes:
                    np.testing.assert_allclose(ax.lines[0].get_ydata(), [100, 50, 25])
                    np.testing.assert_allclose(ax.lines[4].get_ydata(), [100, 50])
                    np.testing.assert_array_equal(ax.lines[5].get_xdata(), [2])
                    np.testing.assert_allclose(ax.lines[5].get_ydata(), [50])
            finally:
                for call in save.call_args_list:
                    plt.close(call.args[0])

    def test_training_plots_reject_missing_or_invalid_history(self):
        row = {"epoch": 1, "train_loss": .5, "val_score": .7, "val_update_fraction": .25}
        for history, best in (([], 1), ([row], 2),
                              ([{**row, "val_update_fraction": float("nan")}], 1)):
            with self.subTest(history=history, best=best):
                with patch.object(study, "result", return_value={"history": history, "best_epoch": best}):
                    with self.assertRaises(ValueError):
                        figures.training_history()


if __name__ == "__main__":
    unittest.main()
