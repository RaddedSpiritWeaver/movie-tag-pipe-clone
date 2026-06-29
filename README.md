run `uv sync` to install dependancies.
run `uv run download.py` to download clips from `data/annotations.txt`.
run `uv run extract.py` to extract data.
run `uv run run_experiment.py --skip-done --exp experimentN` to run experiments from `run_experiment.py` and save to `results/experimentN`
run `uv run train_<MODALITY>.py ...` to run a specific modality.