ECG Skip Study

This study tests whether a GRU or LSTM can skip state updates while still detecting ectopic or fusion beats in 10-second ECG windows from the MIT-BIH Arrhythmia Database. The tables report one fixed run per model (seed 2), not averages across runs.

Review the saved results

Open ECG_skip_comparison.ipynb. The tables and figures are saved in the notebook. They can be read without connecting to a runtime or mounting Google Drive.

1. Section 2: Compare no skipping, learned skipping, and random skipping within each model family. Read accuracy alongside the number of updates and estimated work per window.
2. Section 3: Compare each physiology-guided GRU or LSTM with the same model type using the same kind of skip gate without heartbeat information. The random GRU in this comparison provides a separate control for estimated computation.
3. Section 4: Compare the update locations on one correctly classified ECG strip. Both figures show the same interval; the tables summarize the full test set.
4. Section 5: Inspect training loss, validation within-record AUROC, and the percentage of samples updated over training. Diamonds mark the selected checkpoints. These plots focus on GRU and LSTM with no skipping, learned skipping, and heartbeat-guided skipping, plus the GRU and LSTM with an MLP gate and no heartbeat features.

The results folder also contains the tables (paper_style_results.csv and extended_results.csv) and figures (PNG and PDF).

Section 5 reads existing results; it does not train any models. With completed results available, run setup and then Section 5 to add these plots without rerunning Sections 2-4. The figures are saved as gru_training_history, lstm_training_history, and training_update_rates (PNG and PDF). Training loss includes the skip penalty where applicable. Update percentages use each model's own input length, and are measured on validation windows, not training or test windows.

Reproduce the run in Google Colab

1. Use a writable copy of ECG_Skip_Study in My Drive. If starting from a download, extract the folder if needed and upload it with the notebook, Scripts, and Data together. For a fresh training run, leave results out of the copy.
2. Open ECG_skip_comparison.ipynb in Colab. In the first code cell, PROJECT_FOLDER is the path relative to My Drive. The default is ECG_Skip_Study; change it only if the folder name or location differs.
3. Leave DOWNLOAD_MISSING = False to use the 44 included recordings. Set it to True only to download missing or damaged files. Preprocessing runs automatically. Colab may need internet access to install packages.
4. Select Runtime > Change runtime type, choose a GPU, then select Runtime > Run all and allow Drive access.

Packages

The notebook checks for these Python packages and installs any that are missing:

- PyTorch (torch >= 2.1)
- NumPy
- SciPy
- scikit-learn
- WFDB
- pandas
- Matplotlib (>= 3.6)

No separate package installation is required when running in Colab. Package installation requires internet access. For a local Python environment, install the same packages before running the notebook.

Training a fresh copy can take several hours. If the run stops, select Run all again in the same copy. The notebook reuses completed models with matching settings and files; it restarts an interrupted model. Checkpoints and training logs are in each model's folder under results.

Sources

Skip RNN, Campos et al.: https://arxiv.org/abs/1708.06834

MIT-BIH Arrhythmia Database v1.0.0, Moody and Mark: https://doi.org/10.13026/C2F305

Data attribution and license: Data/NOTICE.txt
