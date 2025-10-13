# 📊 Market Data Regime Detection 

<div align="center">
<img src="src/data/market master logo.jpg" alt="Market Master Logo" width="300"/>
</div>

Project for detecting market regimes using machine learning models (HMM, Random Forest, XGBoost, and others). This repository includes scripts for preparing features, training HMMs, running experiment grids, evaluating models, and visualizing states on market data (e.g., EURUSD 15M).

## ✨ What it does

Analyzes market data to classify the current state into different regimes (e.g., bullish, bearish, ranging). The main workflow in this repository allows you to:

- Build feature sets from a CSV of prices.
- Train HMMs with splits by size (sliding train/val/test).
- Run parallelizable grids on combinations of features / number of states / covariance type.
- Evaluate models and export states and probabilities in CSV/JSON.
- Visualize states over the price series with a Streamlit app (`app.py`).

## Main structure

- `app.py` — Streamlit viewer for HMM evaluation results. Loads `data/dataset_raw/...` and `src/data/results/eval_grid_best_*_states.csv` files.
- `src/` — Main source code:
- `train_hmm.py` — Trains HMMs using splits by size. Saves models in `src/data/models/` and results in `src/data/results/`.
- `experiment_hmm.py` — Runs experimental grids (parallelizable with processes or threads) on sets of features, states, and covariances.
- `evaluate_hmm.py` — Recalculates states/probabilities for saved models and generates summaries (transitions, durations, conditional statistics).
- `preprocessing.py`, `build_feature_sets.py` — (support scripts) to generate and validate CSVs of features used by experiments.
- `data/` — Data and artifacts:
- `dataset_raw/DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01.csv` — Price dataset used by default.
- `features/` — Feature CSVs and JSONs with subsets (e.g., `fs_best_score_subsets.json`).
- `models/`, `results/` — Training and evaluation outputs (.pkl models, .csv states, .json summaries, images).

## Authors

- **SilverDelgado** — [GitHub](https://github.com/SilverDelgado)
- **Sauvageduck24** — [GitHub](https://github.com/Sauvageduck24)

---

<div align="center">

**⭐ If you like this project, leave us a star! ⭐**

*Developed with ❤️ for traders and quant developers*

</div>
